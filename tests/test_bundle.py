import json
import shutil
from dataclasses import replace

import numpy as np
import pytest

mlflow = pytest.importorskip("mlflow")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from src.api.bundle import (  # noqa: E402
    MANIFEST,
    MODEL_DIR,
    MODEL_FILE,
    archive_bundle,
    export_bundle,
    file_hashes,
    load_bundle,
    sha256_file,
    verify_bundle,
)
from src.api.main import create_app, feature_frame  # noqa: E402
from src.api.model import (  # noqa: E402
    TRUSTED_ARTIFACT_SHA256,
    ModelNotReady,
    Settings,
)
from src.api.schemas import PredictionRequest  # noqa: E402
from src.ml import config  # noqa: E402
from tests.test_api import (  # noqa: E402
    GOOD_MODEL,
    WRONG_THRESHOLD_MODEL,
    register_synthetic,
    valid_request,
)


# ---------------------------------------------------------
# A synthetic registered model and its exported bundle
# ---------------------------------------------------------

def registered_file_sha256(uri, name, root):
    """SHA-256 of the registered model.skops, read straight from the store."""
    path = mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{name}/1", dst_path=str(root / f"{name}-check"),
        tracking_uri=uri,
    )
    return sha256_file(f"{path}/{MODEL_FILE}")


@pytest.fixture(scope="module")
def registry(tmp_path_factory):
    root = tmp_path_factory.mktemp("bundle")
    uri = f"sqlite:///{(root / 'mlflow.db').as_posix()}"
    good_sha = register_synthetic(root, uri, GOOD_MODEL)
    wrong_sha = register_synthetic(root, uri, WRONG_THRESHOLD_MODEL, threshold=0.3)
    settings = Settings(tracking_uri=uri, model_name=GOOD_MODEL, trusted_sha256=good_sha,
                        trusted_artifact_sha256=registered_file_sha256(uri, GOOD_MODEL, root))
    yield {"root": root, "settings": settings, "wrong_sha": wrong_sha}
    mlflow.set_tracking_uri(None)


@pytest.fixture(scope="module")
def bundle(registry):
    out = registry["root"] / "model_bundle"
    manifest = export_bundle(registry["settings"], out)
    return {"dir": out, "manifest": manifest,
            "settings": replace(registry["settings"], bundle_dir=str(out))}


@pytest.fixture
def bundle_copy(bundle, tmp_path):
    """A writable copy of the bundle for tampering tests."""
    copy = tmp_path / "bundle"
    shutil.copytree(bundle["dir"], copy)
    return copy, replace(bundle["settings"], bundle_dir=str(copy))


# ---------------------------------------------------------
# Export
# ---------------------------------------------------------

def test_export_writes_manifest_and_unchanged_model_files(bundle, registry):
    out, manifest = bundle["dir"], bundle["manifest"]
    assert sorted(path.name for path in out.iterdir()) == [MANIFEST, MODEL_DIR]
    assert manifest["model_name"] == GOOD_MODEL and manifest["version"] == "1"
    assert manifest["files"] == file_hashes(out / MODEL_DIR)
    assert manifest["files"][MODEL_FILE] == registry["settings"].trusted_artifact_sha256
    assert set(manifest["version_tags"]) == {"selected_threshold", "model_sha256",
                                             "record_type", "retrained"}


def test_bundle_contains_no_data_or_local_paths(bundle):
    manifest_text = (bundle["dir"] / MANIFEST).read_text(encoding="utf-8")
    assert str(bundle["dir"].parent) not in manifest_text
    names = set(bundle["manifest"]["files"])
    assert MODEL_FILE in names and "MLmodel" in names
    assert not any(name.endswith((".csv", ".json", ".parquet")) for name in names)


def test_export_refuses_a_non_empty_folder(bundle, registry):
    with pytest.raises(FileExistsError):
        export_bundle(registry["settings"], bundle["dir"])


def test_export_refuses_an_untrusted_model_file(registry, tmp_path):
    settings = replace(registry["settings"], trusted_artifact_sha256="0" * 64)
    with pytest.raises(ModelNotReady, match="trusted registered artifact"):
        export_bundle(settings, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_export_refuses_a_model_that_fails_verification(registry, tmp_path):
    settings = replace(registry["settings"], model_name=WRONG_THRESHOLD_MODEL,
                       trusted_sha256=registry["wrong_sha"])
    with pytest.raises(ModelNotReady, match="decision threshold"):
        export_bundle(settings, tmp_path / "out")
    assert not (tmp_path / "out").exists()


# ---------------------------------------------------------
# Serving from the bundle
# ---------------------------------------------------------

def test_api_serves_the_bundle_like_the_registry(bundle, registry):
    with TestClient(create_app(registry["settings"])) as from_registry, \
            TestClient(create_app(bundle["settings"])) as from_bundle:
        assert from_bundle.get("/health").json() == {
            "status": "ok", "model_loaded": True, "model_version": "1", "detail": None}
        assert from_bundle.get("/model-info").json() == from_registry.get("/model-info").json()
        for changes in ({}, {"number_inpatient": 9}, {"medical_specialty": "Surgery-General"}):
            a = from_bundle.post("/predict", json=valid_request(**changes)).json()
            b = from_registry.post("/predict", json=valid_request(**changes)).json()
            assert np.isclose(a["readmission_probability"], b["readmission_probability"])
            assert a["high_risk"] == b["high_risk"]


def test_bundle_setting_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("READMISSION_MODEL_BUNDLE", "/model-bundle")
    assert Settings.from_env().bundle_dir == "/model-bundle"
    monkeypatch.setenv("READMISSION_MODEL_BUNDLE", "")
    assert Settings.from_env().bundle_dir is None


def test_trusted_values_cannot_be_changed_from_the_environment(monkeypatch):
    monkeypatch.setenv("READMISSION_MODEL_BUNDLE", "/model-bundle")
    settings = Settings.from_env()
    assert settings.trusted_artifact_sha256 == TRUSTED_ARTIFACT_SHA256
    assert settings.trusted_sha256 == Settings().trusted_sha256
    assert settings.expected_threshold == Settings().expected_threshold


def test_feature_frame_matches_the_model_signature(bundle):
    loaded = load_bundle(bundle["settings"])
    frame = feature_frame(PredictionRequest.model_validate(valid_request()))
    assert list(frame.columns) == list(loaded.feature_columns) == config.FEATURE_COLUMNS
    assert loaded.model.predict_proba(frame).shape == (1, 2)


# ---------------------------------------------------------
# Tampered or missing bundles fail closed (HTTP 503)
# ---------------------------------------------------------

def assert_bundle_rejected(settings, reason):
    with pytest.raises(ModelNotReady, match=reason):
        load_bundle(settings)
    with TestClient(create_app(settings)) as test_client:
        health = test_client.get("/health")
        assert health.status_code == 503 and reason in health.json()["detail"]
        assert test_client.post("/predict", json=valid_request()).status_code == 503


def edit_manifest(copy, **changes):
    path = copy / MANIFEST
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(changes)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_missing_bundle(registry, tmp_path):
    assert_bundle_rejected(replace(registry["settings"], bundle_dir=str(tmp_path / "none")),
                           "bundle is unavailable")


def test_changed_model_file(bundle_copy):
    copy, settings = bundle_copy
    with open(copy / MODEL_DIR / MODEL_FILE, "ab") as handle:
        handle.write(b"\0")
    assert_bundle_rejected(settings, "do not match its manifest")


def test_added_file(bundle_copy):
    copy, settings = bundle_copy
    (copy / MODEL_DIR / "extra.pkl").write_bytes(b"x")
    assert_bundle_rejected(settings, "do not match its manifest")


def test_removed_file(bundle_copy):
    copy, settings = bundle_copy
    (copy / MODEL_DIR / "conda.yaml").unlink()
    assert_bundle_rejected(settings, "do not match its manifest")


def test_manifest_rewritten_for_a_replaced_model_file(bundle_copy):
    """Updating the manifest's hashes is not enough: model.skops is pinned."""
    copy, settings = bundle_copy
    with open(copy / MODEL_DIR / MODEL_FILE, "ab") as handle:
        handle.write(b"\0")
    edit_manifest(copy, files=file_hashes(copy / MODEL_DIR))
    assert_bundle_rejected(settings, "trusted registered artifact")


def test_manifest_with_wrong_threshold(bundle_copy, bundle):
    copy, settings = bundle_copy
    tags = {**bundle["manifest"]["version_tags"], "selected_threshold": "0.3"}
    edit_manifest(copy, version_tags=tags)
    assert_bundle_rejected(settings, "decision threshold")


def test_manifest_with_wrong_run_hash(bundle_copy):
    copy, settings = bundle_copy
    edit_manifest(copy, run_tags={"model_sha256": "0" * 64})
    assert_bundle_rejected(settings, "provenance hash")


def test_manifest_for_another_model(bundle_copy):
    copy, settings = bundle_copy
    edit_manifest(copy, model_name="another-model")
    assert_bundle_rejected(settings, "different registered model")


def test_manifest_not_ready(bundle_copy):
    copy, settings = bundle_copy
    edit_manifest(copy, status="PENDING_REGISTRATION")
    assert_bundle_rejected(settings, "not READY")


# ---------------------------------------------------------
# Archive for another machine (fresh clone)
# ---------------------------------------------------------

def test_archive_is_reproducible_and_holds_only_the_bundle(bundle, registry, tmp_path):
    import zipfile
    first = archive_bundle(registry["settings"], bundle["dir"], tmp_path / "a.zip")
    second = archive_bundle(registry["settings"], bundle["dir"], tmp_path / "b.zip")
    assert first == second == sha256_file(tmp_path / "a.zip")
    assert (tmp_path / "a.zip.sha256").read_text(encoding="utf-8").startswith(first)
    names = zipfile.ZipFile(tmp_path / "a.zip").namelist()
    assert names[0] == f"model_bundle/{MANIFEST}"
    assert sorted(names[1:]) == sorted(f"model_bundle/{MODEL_DIR}/{name}"
                                       for name in bundle["manifest"]["files"])


def test_unzipped_archive_passes_every_check(bundle, registry, tmp_path):
    import zipfile
    archive_bundle(registry["settings"], bundle["dir"], tmp_path / "bundle.zip")
    zipfile.ZipFile(tmp_path / "bundle.zip").extractall(tmp_path / "unzipped")
    loaded = verify_bundle(registry["settings"], tmp_path / "unzipped" / "model_bundle")
    assert loaded.version == "1" and loaded.threshold == pytest.approx(
        float(bundle["manifest"]["version_tags"]["selected_threshold"]))


def test_archive_refuses_a_tampered_bundle(bundle_copy, registry, tmp_path):
    copy, _ = bundle_copy
    (copy / MODEL_DIR / "extra.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ModelNotReady, match="manifest"):
        archive_bundle(registry["settings"], copy, tmp_path / "out.zip")
    assert not (tmp_path / "out.zip").exists()


def test_archive_does_not_overwrite(bundle, registry, tmp_path):
    target = tmp_path / "exists.zip"
    target.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        archive_bundle(registry["settings"], bundle["dir"], target)
    assert target.read_bytes() == b"keep"
