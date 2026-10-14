import json
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.ml import config
from src.ml.tracking import setup_experiment, sha256_file

mlflow = pytest.importorskip("mlflow")

from mlflow import MlflowClient  # noqa: E402

from src.ml import register_model  # noqa: E402
from src.ml.log_phase3_to_mlflow import RECORD_TYPE as PHASE3_RECORD_TYPE  # noqa: E402
from src.ml.register_model import (  # noqa: E402
    MODEL_ALIAS,
    PHASE3_PARENT_RUN_NAME,
    RECORD_TYPE,
    load_selected,
    load_verified_model,
    register,
)


THRESHOLD = 0.1617830043681171
MODEL_NAME = "test-readmission-classifier"


def toy_rows(n=200, seed=0):
    """Rows with numeric and string columns, like the real feature frame."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "time_in_hospital": rng.integers(1, 14, n),
        "num_medications": rng.integers(1, 40, n),
        "gender": rng.choice(["Female", "Male"], n),
        "insulin": rng.choice(["No", "Up", "Down", "Steady"], n),
    })


def toy_model(rows):
    """A small sigmoid-calibrated Random Forest pipeline (test fixture only)."""
    y = (rows["time_in_hospital"] + rng_noise(len(rows)) > 8).astype(int)
    pipeline = Pipeline([
        ("prep", ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["gender", "insulin"]),
        ], remainder="passthrough")),
        ("model", RandomForestClassifier(n_estimators=10, max_depth=3, random_state=0)),
    ])
    return CalibratedClassifierCV(pipeline, method="sigmoid", cv=3).fit(rows, y)


def rng_noise(n):
    return np.random.default_rng(1).normal(0, 2, n)


def write_phase3_dir(root, model, threshold=THRESHOLD):
    (root / "models").mkdir(parents=True)
    joblib.dump(model, root / "models" / "random_forest.joblib")
    metrics = {
        "split_fingerprint_sha256": "ab" * 32,
        "positive_class": "readmitted <30 days",
        "selected_model": "random_forest",
        "models": {"random_forest": {"selected_threshold": threshold}},
        "test_selected_model": {"roc_auc": 0.67},
    }
    (root / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return root


def log_phase3_parent(uri, experiment, artifact_root, phase3, model_sha256="file"):
    """
    A stand-in for the Phase 3 parent run that log_phase3_to_mlflow writes:
    same name and tags, with `selected_model_sha256` set to the file's hash
    ("file"), another value, or left out (None).
    """
    if model_sha256 == "file":
        model_sha256 = sha256_file(phase3 / "models" / "random_forest.joblib")
    tags = {
        "record_type": PHASE3_RECORD_TYPE,
        "metrics_json_sha256": sha256_file(phase3 / "metrics.json"),
    }
    if model_sha256 is not None:
        tags["selected_model_sha256"] = model_sha256
    experiment_id = setup_experiment(experiment, uri, artifact_root)
    with mlflow.start_run(experiment_id=experiment_id,
                          run_name=PHASE3_PARENT_RUN_NAME, tags=tags):
        pass
    return experiment_id


@pytest.fixture(scope="module")
def registered(tmp_path_factory):
    """Register a toy model once in a temporary SQLite store (slow to create)."""
    root = tmp_path_factory.mktemp("registry")
    uri = f"sqlite:///{(root / 'mlflow.db').as_posix()}"
    rows = toy_rows()
    phase3 = write_phase3_dir(root / "phase3", toy_model(rows))
    log_phase3_parent(uri, "test-experiment", root / "mlartifacts", phase3)
    summary = register(phase3, uri, "test-experiment", root / "mlartifacts",
                       MODEL_NAME, rows=rows)
    client = MlflowClient(tracking_uri=uri)
    yield {"summary": summary, "client": client, "phase3": phase3,
           "rows": rows, "uri": uri, "root": root}
    mlflow.set_tracking_uri(None)


# ---------------------------------------------------------
# Reading the saved outputs (no MLflow store needed)
# ---------------------------------------------------------

def test_threshold_is_read_exactly_from_metrics_json(tmp_path):
    phase3 = write_phase3_dir(tmp_path / "phase3", "placeholder")
    _, _, model_path, threshold = load_selected(phase3)
    assert threshold == THRESHOLD
    assert model_path.name == "random_forest.joblib"


@pytest.mark.parametrize("bad", [None, 0.0, 1.0, "0.16"])
def test_invalid_threshold_is_rejected(tmp_path, bad):
    phase3 = write_phase3_dir(tmp_path / "phase3", "placeholder", threshold=bad)
    with pytest.raises(ValueError, match="selected_threshold"):
        load_selected(phase3)


def test_missing_model_file_is_reported(tmp_path):
    phase3 = write_phase3_dir(tmp_path / "phase3", "placeholder")
    (phase3 / "models" / "random_forest.joblib").unlink()
    with pytest.raises(FileNotFoundError, match="random_forest.joblib"):
        load_selected(phase3)


def test_registration_script_does_not_import_training_code():
    code = ("import sys, src.ml.register_model; "
            "assert 'src.ml.train' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True,
                   cwd=config.PROJECT_ROOT)


# ---------------------------------------------------------
# Registration in a temporary MLflow store
# ---------------------------------------------------------

def test_model_is_registered_with_version_and_alias(registered):
    client, summary = registered["client"], registered["summary"]
    assert summary["version"] == "1"
    version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    assert str(version.version) == "1"
    assert version.status == "READY"
    assert version.run_id == summary["run_id"]


def test_threshold_is_stored_exactly_on_the_version(registered):
    client = registered["client"]
    version = client.get_model_version(MODEL_NAME, "1")
    assert float(version.tags["selected_threshold"]) == THRESHOLD
    assert version.tags["threshold_source"].startswith("validation set")
    info = mlflow.models.get_model_info(f"models:/{MODEL_NAME}/1")
    assert info.metadata["selected_threshold"] == THRESHOLD


def test_version_carries_provenance_of_the_saved_file(registered):
    tags = registered["client"].get_model_version(MODEL_NAME, "1").tags
    phase3 = registered["phase3"]
    assert tags["model_sha256"] == sha256_file(phase3 / "models" / "random_forest.joblib")
    assert tags["metrics_json_sha256"] == sha256_file(phase3 / "metrics.json")
    assert tags["split_fingerprint_sha256"] == "ab" * 32
    assert tags["record_type"] == RECORD_TYPE
    assert tags["retrained"] == "false"
    assert tags["model_class"] == "sklearn.calibration.CalibratedClassifierCV"
    assert "Phase 3 MLflow run" in tags["model_sha256_verified_against"]


def test_registry_copy_reproduces_saved_probabilities(registered):
    rows = toy_rows(n=150, seed=5)
    saved = joblib.load(registered["phase3"] / "models" / "random_forest.joblib")
    mlflow.set_tracking_uri(registered["uri"])
    loaded = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    assert np.allclose(loaded.predict_proba(rows), saved.predict_proba(rows))
    pyfunc = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/1")
    assert np.allclose(np.asarray(pyfunc.predict(rows)), saved.predict_proba(rows))
    assert registered["summary"]["check"]["sklearn_allclose"] is True


def test_same_model_file_is_not_registered_twice(registered):
    with pytest.raises(RuntimeError, match="already registered"):
        register(registered["phase3"], registered["uri"], "test-experiment",
                 registered["root"] / "mlartifacts", MODEL_NAME,
                 rows=registered["rows"])
    versions = registered["client"].search_model_versions(f"name = '{MODEL_NAME}'")
    assert len(versions) == 1


# ---------------------------------------------------------
# Hash verification before the model file is unpickled
# ---------------------------------------------------------

@pytest.fixture(scope="module")
def unverified_store(tmp_path_factory):
    """A separate store for the fail-closed cases; each test uses its own experiment."""
    root = tmp_path_factory.mktemp("unverified")
    uri = f"sqlite:///{(root / 'mlflow.db').as_posix()}"
    phase3 = write_phase3_dir(root / "phase3", toy_model(toy_rows()))
    yield {"uri": uri, "root": root, "phase3": phase3,
           "client": MlflowClient(tracking_uri=uri)}
    mlflow.set_tracking_uri(None)


@pytest.fixture
def no_unpickling(monkeypatch):
    """Fail the test if the model file is unpickled."""
    def refuse(*args, **kwargs):
        raise AssertionError("joblib.load was called")
    monkeypatch.setattr(register_model.joblib, "load", refuse)


def registration_runs(client, experiment):
    experiment_id = client.get_experiment_by_name(experiment).experiment_id
    return client.search_runs(
        [experiment_id], filter_string=f"tags.record_type = '{RECORD_TYPE}'"
    )


@pytest.mark.parametrize("case, recorded, message", [
    ("mismatch", "0" * 64, "does not match"),
    ("tag-missing", None, "no selected_model_sha256"),
    ("no-phase3-run", "skip", "No finished Phase 3 run"),
])
def test_unverified_model_file_is_neither_loaded_nor_registered(
        unverified_store, no_unpickling, case, recorded, message):
    store, experiment = unverified_store, f"exp-{case}"
    if recorded == "skip":
        setup_experiment(experiment, store["uri"], store["root"] / "mlartifacts")
    else:
        log_phase3_parent(store["uri"], experiment, store["root"] / "mlartifacts",
                          store["phase3"], model_sha256=recorded)
    with pytest.raises(RuntimeError, match=message):
        register(store["phase3"], store["uri"], experiment,
                 store["root"] / "mlartifacts", f"model-{case}", rows=toy_rows())
    assert registration_runs(store["client"], experiment) == []
    assert store["client"].search_registered_models(f"name = 'model-{case}'") == []


def test_hash_from_a_different_metrics_json_is_not_trusted(unverified_store, no_unpickling):
    """The Phase 3 run must describe the same saved results (metrics.json SHA-256)."""
    store, experiment = unverified_store, "exp-other-metrics"
    experiment_id = setup_experiment(experiment, store["uri"], store["root"] / "mlartifacts")
    with mlflow.start_run(experiment_id=experiment_id, run_name=PHASE3_PARENT_RUN_NAME,
                          tags={"record_type": PHASE3_RECORD_TYPE,
                                "metrics_json_sha256": "f" * 64,
                                "selected_model_sha256": sha256_file(
                                    store["phase3"] / "models" / "random_forest.joblib")}):
        pass
    with pytest.raises(RuntimeError, match="No finished Phase 3 run"):
        register(store["phase3"], store["uri"], experiment,
                 store["root"] / "mlartifacts", "model-other-metrics", rows=toy_rows())


def test_load_verified_model_checks_hash_before_unpickling(tmp_path, no_unpickling):
    path = tmp_path / "model.joblib"
    path.write_bytes(b"not a trusted model")
    with pytest.raises(RuntimeError, match="not loaded"):
        load_verified_model(path, "0" * 64)


def test_load_verified_model_loads_matching_file(tmp_path):
    path = tmp_path / "model.joblib"
    joblib.dump({"a": 1}, path)
    model, digest = load_verified_model(path, sha256_file(path))
    assert model == {"a": 1}
    assert digest == sha256_file(path)
