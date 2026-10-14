"""
Phase 4 helpers for local MLflow experiment tracking.

Tracking data goes to a local SQLite database (mlflow.db) and run artifacts
to a local folder (mlartifacts/), both at the project root and gitignored.
SQLite is used because the MLflow model registry needs a database backend.

View the runs from the project root with:

    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import hashlib
import json
import os
import re
from pathlib import Path

# MLflow 3.17 prints an assistant hint on import; it is not relevant here.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402
from mlflow.utils.validation import (  # noqa: E402
    MAX_ENTITY_KEY_LENGTH,
    MAX_PARAM_VAL_LENGTH,
    MAX_TAG_VAL_LENGTH,
)

from src.ml import config  # noqa: E402


TRACKING_DB = config.PROJECT_ROOT / "mlflow.db"
TRACKING_URI = f"sqlite:///{TRACKING_DB.as_posix()}"
ARTIFACT_ROOT = config.PROJECT_ROOT / "mlartifacts"
EXPERIMENT_NAME = "hospital-readmission-30d"

# MLflow keys allow letters, digits, "_", "-", ".", " " and "/".
_INVALID_KEY_CHARS = re.compile(r"[^A-Za-z0-9_\-. /]")


# ---------------------------------------------------------
# Experiment setup
# ---------------------------------------------------------

def setup_experiment(name=EXPERIMENT_NAME, tracking_uri=TRACKING_URI,
                     artifact_root=ARTIFACT_ROOT):
    """
    Point MLflow at the tracking store and return the experiment id,
    creating the experiment (with a local artifact folder) if needed.
    """
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        location = (Path(artifact_root).resolve() / name).as_uri()
        return mlflow.create_experiment(name, artifact_location=location)
    if experiment.lifecycle_stage == "deleted":
        raise RuntimeError(
            f"MLflow experiment '{name}' is deleted. Restore it or use "
            "another experiment name."
        )
    return experiment.experiment_id


# ---------------------------------------------------------
# Safe logging
# ---------------------------------------------------------

def metric_key(*parts):
    """Join key parts with "/" and replace characters MLflow rejects."""
    key = "/".join(_INVALID_KEY_CHARS.sub("_", str(p)) for p in parts if p != "")
    if len(key) > MAX_ENTITY_KEY_LENGTH:
        raise ValueError(f"MLflow key longer than {MAX_ENTITY_KEY_LENGTH}: {key}")
    return key


def flatten_metrics(values, prefix="", skip=("classification_report",)):
    """
    Flatten nested metric dictionaries into {"a/b/c": float}.

    Only real numbers are kept: strings, None and booleans are left out.
    Keys listed in `skip` are dropped with everything below them.
    """
    flat = {}
    for key, value in values.items():
        if key in skip:
            continue
        name = metric_key(prefix, key)
        if isinstance(value, dict):
            flat.update(flatten_metrics(value, name, skip))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            flat[name] = float(value)
    return flat


def _as_text(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def log_params(params):
    """Log parameters; fails instead of letting MLflow truncate a value."""
    clean = {}
    for key, value in params.items():
        text = _as_text(value)
        if len(text) > MAX_PARAM_VAL_LENGTH:
            raise ValueError(f"Parameter '{key}' is longer than {MAX_PARAM_VAL_LENGTH}")
        clean[metric_key(key)] = text
    mlflow.log_params(clean)


def set_tags(tags):
    """Set tags; fails instead of letting MLflow truncate a value."""
    clean = {}
    for key, value in tags.items():
        text = _as_text(value)
        if len(text) > MAX_TAG_VAL_LENGTH:
            raise ValueError(f"Tag '{key}' is longer than {MAX_TAG_VAL_LENGTH}")
        clean[key] = text
    mlflow.set_tags(clean)


def log_metrics(metrics):
    mlflow.log_metrics({metric_key(k): float(v) for k, v in metrics.items()})


def log_files(paths, artifact_path=None):
    """Log existing files as run artifacts; a missing file is an error."""
    for path in paths:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        mlflow.log_artifact(str(path), artifact_path=artifact_path)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
