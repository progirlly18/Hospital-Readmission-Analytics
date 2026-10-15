"""
Phase 6: export the verified registered model as a bundle for the container.

The MLflow store (mlflow.db) records artifact locations as absolute paths on
the machine that created it, so the store cannot simply be mounted into a
Linux container. Instead, the registered version is exported once, on the
machine that has the store, into a plain folder:

    model_bundle/
        manifest.json   registry details and the SHA-256 of every model file
        model/          the registered version's MLflow files, unchanged

Run from the project root:

    python -m src.api.bundle            # writes model_bundle/ (gitignored)
    python -m src.api.bundle --verify model_bundle
                                        # checks a bundle exactly as the API does
    python -m src.api.bundle --archive dist/model_bundle_v1.zip
                                        # verified, reproducible zip + .sha256

The archive is how the bundle reaches a machine without the MLflow store (for
example a marker's fresh clone): it is verified before it is written, its
entries have fixed timestamps so the same bundle always gives the same zip
hash, and a .sha256 file is written next to it. After unzipping, --verify
repeats every check. See docs/part2/model_bundle.md.

The export runs the same checks as the API before writing anything, and
refuses a model.skops that differs from the trusted registered artifact. The
API loads a bundle when READMISSION_MODEL_BUNDLE is set, repeating every check
plus the file hashes, and refuses the bundle if any file was added, removed
or changed. The bundle holds the model only: no data or credentials
(MLmodel keeps MLflow's informational artifact_path, which loading does not
use).
"""

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from src.ml import config
from src.api.model import (
    ModelNotReady,
    Settings,
    load_verified,
    resolve,
    verify,
)

import mlflow  # noqa: E402
from mlflow.models import Model  # noqa: E402


DEFAULT_BUNDLE_DIR = config.PROJECT_ROOT / "model_bundle"
MANIFEST = "manifest.json"
MODEL_DIR = "model"
MODEL_FILE = "model.skops"
FORMAT_VERSION = 1
# Only the tags verify() reads are exported.
VERSION_TAGS = ("selected_threshold", "model_sha256", "record_type", "retrained")
RUN_TAGS = ("model_sha256",)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_hashes(model_dir):
    """{relative posix path: sha256} for every file under model_dir."""
    return {
        path.relative_to(model_dir).as_posix(): sha256_file(path)
        for path in sorted(Path(model_dir).rglob("*")) if path.is_file()
    }


def check_trusted_file(settings, hashes):
    if hashes.get(MODEL_FILE) != settings.trusted_artifact_sha256:
        raise ModelNotReady("model file hash does not match the trusted registered artifact")


# ---------------------------------------------------------
# Export (host, with the MLflow store)
# ---------------------------------------------------------

def export_bundle(settings, out_dir):
    """Verify the registered model and write it as a bundle; return the manifest."""
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty; remove it or choose another folder")

    version, uri, info, run_tags = resolve(settings)
    verify(settings, version, info, run_tags)

    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "bundle"
        model_dir = staged / MODEL_DIR
        mlflow.artifacts.download_artifacts(
            artifact_uri=uri, dst_path=str(model_dir), tracking_uri=settings.tracking_uri,
        )
        hashes = file_hashes(model_dir)
        check_trusted_file(settings, hashes)

        manifest = {
            "format_version": FORMAT_VERSION,
            "model_name": settings.model_name,
            "alias": settings.alias,
            "version": str(version.version),
            "status": version.status,
            "version_tags": {key: version.tags[key] for key in VERSION_TAGS
                             if key in version.tags},
            "run_tags": {key: run_tags[key] for key in RUN_TAGS if key in run_tags},
            "files": hashes,
            "mlflow_version": mlflow.__version__,
        }
        (staged / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n",
                                       encoding="utf-8")

        # Load the staged bundle exactly as the API will before publishing it.
        load_bundle(replace(settings, bundle_dir=str(staged)))

        out_dir.mkdir(parents=True, exist_ok=True)
        for path in staged.iterdir():
            shutil.move(str(path), str(out_dir / path.name))
    return manifest


# ---------------------------------------------------------
# Load (API, typically inside the container)
# ---------------------------------------------------------

def load_bundle(settings):
    """Check a bundle against its manifest and the trusted values, then load it."""
    bundle = Path(settings.bundle_dir)
    try:
        manifest = json.loads((bundle / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ModelNotReady(f"model bundle is unavailable ({type(error).__name__})") from None

    if manifest.get("format_version") != FORMAT_VERSION:
        raise ModelNotReady("model bundle format is not supported")
    if (manifest.get("model_name"), manifest.get("alias")) != (settings.model_name,
                                                               settings.alias):
        raise ModelNotReady("model bundle is for a different registered model or alias")

    model_dir = bundle / MODEL_DIR
    hashes = file_hashes(model_dir) if model_dir.is_dir() else {}
    if not hashes or hashes != manifest.get("files"):
        raise ModelNotReady("model bundle files do not match its manifest")
    check_trusted_file(settings, hashes)

    version = SimpleNamespace(
        status=manifest.get("status"),
        version=manifest.get("version"),
        tags=manifest.get("version_tags") or {},
    )
    try:
        # Model.load reads MLmodel only; get_model_info would also query a
        # tracking store, which the container does not have.
        info = Model.load(str(model_dir))
    except Exception as error:
        raise ModelNotReady(f"model bundle could not be read ({type(error).__name__})") from None
    threshold = verify(settings, version, info, manifest.get("run_tags") or {})
    return load_verified(settings, version, threshold, str(model_dir))


# ---------------------------------------------------------
# Archive and verify (moving a bundle to another machine)
# ---------------------------------------------------------

ZIP_DATE = (2026, 1, 1, 0, 0, 0)  # fixed, so the zip hash depends on content only


def verify_bundle(settings, bundle_dir):
    """Run every API check on a bundle folder; return the loaded model."""
    return load_bundle(replace(settings, bundle_dir=str(bundle_dir)))


def archive_bundle(settings, bundle_dir, zip_path):
    """Verify a bundle, then write it as a reproducible zip; return its SHA-256."""
    bundle_dir, zip_path = Path(bundle_dir), Path(zip_path)
    verify_bundle(settings, bundle_dir)
    if zip_path.exists():
        raise FileExistsError(f"{zip_path} already exists; remove it or choose another name")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    files = [MANIFEST] + [f"{MODEL_DIR}/{name}" for name in
                          sorted(file_hashes(bundle_dir / MODEL_DIR))]
    with zipfile.ZipFile(zip_path, "w") as archive:
        for name in files:
            entry = zipfile.ZipInfo(f"model_bundle/{name}", date_time=ZIP_DATE)
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o644 << 16
            archive.writestr(entry, (bundle_dir / name).read_bytes())
    digest = sha256_file(zip_path)
    Path(f"{zip_path}.sha256").write_text(f"{digest}  {zip_path.name}\n", encoding="utf-8")
    return digest


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_BUNDLE_DIR,
                        help="export: folder to write the bundle to")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--verify", type=Path, metavar="BUNDLE_DIR",
                        help="check an existing bundle folder and exit")
    action.add_argument("--archive", type=Path, metavar="ZIP",
                        help="verify --bundle and write it as a zip with a .sha256 file")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE_DIR,
                        help="archive: the bundle folder to pack (default model_bundle)")
    args = parser.parse_args()

    settings = Settings.from_env()
    if args.verify or args.archive:
        try:
            if args.verify:
                loaded = verify_bundle(settings, args.verify)
                print(f"OK: {args.verify} holds {loaded.name} version {loaded.version} "
                      f"(@{loaded.alias}), threshold {loaded.threshold!r}, "
                      f"model sha256 {loaded.model_sha256}")
            else:
                digest = archive_bundle(settings, args.bundle, args.archive)
                print(f"Wrote {args.archive} (sha256 {digest}) and {args.archive}.sha256")
        except ModelNotReady as error:
            raise SystemExit(f"Bundle check failed: {error}") from None
        return

    if settings.bundle_dir:
        parser.error("unset READMISSION_MODEL_BUNDLE; the export reads the registry")
    manifest = export_bundle(settings, args.out)
    print(f"Exported {manifest['model_name']} version {manifest['version']} "
          f"(@{manifest['alias']}) to {args.out}")
    print(f"Threshold {manifest['version_tags']['selected_threshold']}; "
          f"{MODEL_FILE} sha256 {manifest['files'][MODEL_FILE]}")


if __name__ == "__main__":
    main()
