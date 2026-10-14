"""
Phase 4: log the SAVED Phase 3 Random Forest as an MLflow model and register it.

Nothing is trained, re-trained or re-evaluated. The script:

1. reads artifacts/phase3/metrics.json and models/<selected>.joblib
   (the CalibratedClassifierCV pipeline saved by `python -m src.ml.train`);
   the model file is loaded only if its SHA-256 equals the
   `selected_model_sha256` recorded by the Phase 3 MLflow run that
   `log_phase3_to_mlflow` logged for the same metrics.json (fail closed);
2. logs that object unchanged with the MLflow sklearn flavour in one run,
   `register-phase3-<selected>`, and registers it as a new model version;
3. stores the decision threshold chosen on validation in Phase 3 (the exact
   value from metrics.json) and the provenance hashes as version tags;
4. loads the version back from the registry (models:/<name>/<version>) and
   checks that its predict_proba equals the joblib model's on a sample of
   real feature rows from the validation split (np.allclose). Only
   probabilities are compared; no metric is computed.

Run from the project root, after `python -m src.ml.log_phase3_to_mlflow`:

    python -m src.ml.register_model

The same saved model file is registered only once (matched on its SHA-256);
pass --allow-duplicate to register it again as a new version.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path

# Imported before mlflow: src.ml.tracking sets MLflow's environment options.
from src.ml import config
from src.ml.tracking import (
    ARTIFACT_ROOT,
    EXPERIMENT_NAME,
    TRACKING_URI,
    log_params,
    set_tags,
    setup_experiment,
    sha256_file,
)
from src.ml.log_phase3_to_mlflow import RECORD_TYPE as PHASE3_RECORD_TYPE

import joblib  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow import MlflowClient  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402


PHASE3_DIR = config.ARTIFACTS_DIR / "phase3"
REGISTERED_MODEL_NAME = "hospital-readmission-30d-classifier"
MODEL_ALIAS = "candidate"
RECORD_TYPE = "phase3_model_registration"
PHASE3_PARENT_RUN_NAME = "phase3-historical-comparison"
SAMPLE_SIZE = 500
# MLflow saves sklearn models with skops, which loads only types it is told to
# trust. These are the types inside the project's own calibrated Random Forest
# (created by src.ml.train, not downloaded), listed one by one.
SKOPS_TRUSTED_TYPES = [
    "numpy.dtype",
    "sklearn.calibration._CalibratedClassifier",
    "sklearn.calibration._SigmoidCalibration",
    "sklearn.tree._tree.Tree",
]
REGISTRATION_NOTE = (
    "The saved Phase 3 model file, logged unchanged. No model was trained, "
    "re-trained or re-evaluated when it was registered."
)
DISCLAIMER = (
    "Academic decision-support prototype; not clinically validated and not "
    "for real patient decisions."
)


# ---------------------------------------------------------
# Reading the saved Phase 3 outputs
# ---------------------------------------------------------

def load_selected(phase3_dir):
    """Return (metrics, model path, threshold) for the selected Phase 3 model."""
    phase3_dir = Path(phase3_dir)
    metrics_path = phase3_dir / "metrics.json"
    with open(metrics_path, encoding="utf-8") as handle:
        metrics = json.load(handle)
    selected = metrics.get("selected_model")
    if selected not in metrics.get("models", {}):
        raise ValueError(f"metrics.json has no entry for selected model '{selected}'")
    threshold = metrics["models"][selected].get("selected_threshold")
    if not isinstance(threshold, float) or not 0.0 < threshold < 1.0:
        raise ValueError(f"Invalid selected_threshold for '{selected}': {threshold!r}")
    model_path = phase3_dir / "models" / f"{selected}.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    return metrics, metrics_path, model_path, threshold


def sample_feature_rows(phase3_dir, n=SAMPLE_SIZE, seed=config.RANDOM_SEED):
    """
    A fixed sample of real model-input rows from the validation split.

    Rows are matched on the encounter ids in the saved split_assignments.csv,
    so the split is the one Phase 3 used. Only the feature columns are
    returned (no identifiers, target or subgroup columns).
    """
    from src.ml.data import load_modelling_data
    from src.ml.features import build_feature_frame

    assignments = pd.read_csv(Path(phase3_dir) / "split_assignments.csv", dtype=str)
    validation_ids = set(
        assignments.loc[assignments["split"] == "validation",
                        config.ENCOUNTER_ID_COLUMN]
    )
    cohort, _ = load_modelling_data()
    mask = cohort[config.ENCOUNTER_ID_COLUMN].astype(str).isin(validation_ids)
    features = build_feature_frame(cohort.loc[mask])
    if features.empty:
        raise ValueError("No validation rows matched split_assignments.csv")
    return features.sample(n=min(n, len(features)), random_state=seed)


def find_existing_versions(client, name, model_sha256):
    """Registered versions of `name` that hold this exact saved model file."""
    try:
        versions = client.search_model_versions(f"name = '{name}'")
    except mlflow.exceptions.MlflowException:
        return []
    return [v for v in versions if v.tags.get("model_sha256") == model_sha256]


def trusted_model_sha256(client, experiment_id, metrics_sha256):
    """
    The `selected_model_sha256` recorded by the Phase 3 MLflow run for these
    saved results (same metrics.json SHA-256), read from the tracking store.

    Only active, FINISHED parent runs logged by log_phase3_to_mlflow count.
    Raises RuntimeError if there is no such run, the tag is missing, or
    several runs disagree, so the caller never loads an unverified file.
    """
    try:
        runs = client.search_runs(
            [experiment_id],
            filter_string=(
                f"tags.record_type = '{PHASE3_RECORD_TYPE}' and "
                f"tags.metrics_json_sha256 = '{metrics_sha256}' and "
                f"attributes.run_name = '{PHASE3_PARENT_RUN_NAME}' and "
                "attributes.status = 'FINISHED'"
            ),
        )
    except mlflow.exceptions.MlflowException as error:
        raise RuntimeError(f"Could not read the Phase 3 run: {error}") from error
    if not runs:
        raise RuntimeError(
            "No finished Phase 3 run for this metrics.json in the tracking store; "
            "run `python -m src.ml.log_phase3_to_mlflow` first."
        )
    hashes = {run.data.tags.get("selected_model_sha256") for run in runs}
    if None in hashes or "" in hashes:
        raise RuntimeError("The Phase 3 run has no selected_model_sha256 tag.")
    if len(hashes) != 1:
        raise RuntimeError(f"Phase 3 runs disagree on selected_model_sha256: {sorted(hashes)}")
    return hashes.pop()


def load_verified_model(model_path, expected_sha256):
    """
    Read the model file once, check its SHA-256 and only then unpickle the
    same bytes. joblib.load runs pickle code, so a file that does not match
    the trusted hash is never loaded.
    """
    data = Path(model_path).read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(
            f"SHA-256 of {model_path} ({actual}) does not match the Phase 3 "
            f"run's selected_model_sha256 ({expected_sha256}); not loaded."
        )
    return joblib.load(io.BytesIO(data)), actual


# ---------------------------------------------------------
# Logging, registration and the consistency check
# ---------------------------------------------------------

def check_registry_copy(model_uri, joblib_model, rows):
    """
    Load the registered model and compare its probabilities with the joblib
    model's on `rows`. Returns a dict with the result; raises on mismatch.
    """
    registered = mlflow.sklearn.load_model(model_uri)
    expected = joblib_model.predict_proba(rows)
    actual = registered.predict_proba(rows)
    pyfunc_out = np.asarray(mlflow.pyfunc.load_model(model_uri).predict(rows))
    result = {
        "rows": len(rows),
        "sklearn_allclose": bool(np.allclose(actual, expected)),
        "pyfunc_allclose": bool(np.allclose(pyfunc_out, expected)),
        "max_abs_diff": float(np.max(np.abs(actual - expected))),
    }
    if not (result["sklearn_allclose"] and result["pyfunc_allclose"]):
        raise RuntimeError(f"Registered model does not reproduce the saved one: {result}")
    return result


def register(phase3_dir=PHASE3_DIR, tracking_uri=TRACKING_URI,
             experiment_name=EXPERIMENT_NAME, artifact_root=ARTIFACT_ROOT,
             model_name=REGISTERED_MODEL_NAME, rows=None, allow_duplicate=False):
    """Log, register and check the saved model; return a summary dict."""
    metrics, metrics_path, model_path, threshold = load_selected(phase3_dir)
    selected = metrics["selected_model"]
    metrics_sha256 = sha256_file(metrics_path)

    experiment_id = setup_experiment(experiment_name, tracking_uri, artifact_root)
    client = MlflowClient(tracking_uri=tracking_uri)
    # Fail closed before anything is unpickled or registered.
    trusted_sha256 = trusted_model_sha256(client, experiment_id, metrics_sha256)
    existing = find_existing_versions(client, model_name, trusted_sha256)
    if existing and not allow_duplicate:
        raise RuntimeError(
            f"This saved model is already registered as {model_name} "
            f"version {existing[0].version}. Use --allow-duplicate to register again."
        )
    model, model_sha256 = load_verified_model(model_path, trusted_sha256)

    if rows is None:
        rows = sample_feature_rows(phase3_dir)
    signature = infer_signature(rows, model.predict_proba(rows))

    provenance = {
        "record_type": RECORD_TYPE,
        "retrained": "false",
        "source_phase": "phase3",
        "model_name": selected,
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "model_sha256": model_sha256,
        "model_sha256_verified_against": "selected_model_sha256 of the Phase 3 MLflow run",
        "metrics_json_sha256": metrics_sha256,
        "split_fingerprint_sha256": metrics["split_fingerprint_sha256"],
    }
    version_tags = {
        **provenance,
        "selected_threshold": repr(threshold),
        "threshold_source": "validation set, maximise F1 (Phase 3)",
        "positive_class": metrics["positive_class"],
        "prediction_output": "predict_proba; column 1 = P(readmitted <30 days)",
        "test_roc_auc_phase3": repr(metrics["test_selected_model"]["roc_auc"]),
        "disclaimer": DISCLAIMER,
    }

    with mlflow.start_run(experiment_id=experiment_id,
                          run_name=f"register-phase3-{selected}",
                          description=REGISTRATION_NOTE) as run:
        set_tags({**provenance, "registration_note": REGISTRATION_NOTE})
        log_params({"selected_threshold": threshold, "registered_model_name": model_name})
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            pyfunc_predict_fn="predict_proba",
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
            registered_model_name=model_name,
            metadata={"selected_threshold": threshold, "model_sha256": model_sha256},
        )
        version = str(info.registered_model_version)

    for key, value in version_tags.items():
        client.set_model_version_tag(model_name, version, key, value)
    client.update_model_version(
        model_name, version,
        description=(f"Phase 3 {selected} (sigmoid-calibrated), saved file logged "
                     f"unchanged. Decision threshold {threshold!r} from validation. "
                     + DISCLAIMER),
    )
    client.update_registered_model(
        model_name,
        description="30-day readmission risk classifier (academic prototype). "
                    + DISCLAIMER,
    )
    client.set_registered_model_alias(model_name, MODEL_ALIAS, version)

    model_uri = f"models:/{model_name}/{version}"
    check = check_registry_copy(model_uri, model, rows)
    client.set_model_version_tag(model_name, version, "registry_load_check",
                                 f"predict_proba allclose on {check['rows']} "
                                 "validation rows: passed")

    return {"run_id": run.info.run_id, "model_name": model_name,
            "version": version, "alias": MODEL_ALIAS, "model_uri": model_uri,
            "threshold": threshold, "check": check}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase3-dir", type=Path, default=PHASE3_DIR)
    parser.add_argument("--tracking-uri", default=TRACKING_URI)
    parser.add_argument("--experiment", default=EXPERIMENT_NAME)
    parser.add_argument("--model-name", default=REGISTERED_MODEL_NAME)
    parser.add_argument("--allow-duplicate", action="store_true",
                        help="register again even if this model file is registered")
    args = parser.parse_args()

    try:
        summary = register(args.phase3_dir, args.tracking_uri, args.experiment,
                           ARTIFACT_ROOT, args.model_name,
                           allow_duplicate=args.allow_duplicate)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Nothing registered: {error}")
    check = summary["check"]
    print("Registered the saved Phase 3 model (nothing retrained).")
    print(f"Run id:      {summary['run_id']}")
    print(f"Model:       {summary['model_name']} version {summary['version']} "
          f"(alias '{summary['alias']}')")
    print(f"Threshold:   {summary['threshold']!r}")
    print(f"Load check:  {summary['model_uri']} predict_proba vs joblib on "
          f"{check['rows']} validation rows: sklearn allclose={check['sklearn_allclose']}, "
          f"pyfunc allclose={check['pyfunc_allclose']}, "
          f"max |diff|={check['max_abs_diff']:.3g}")


if __name__ == "__main__":
    main()
