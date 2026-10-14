"""
Phase 4: log the SAVED Phase 3 results to MLflow.

This script trains nothing and re-evaluates nothing. It reads the files that
`python -m src.ml.train` already wrote to artifacts/phase3/ and records them
in MLflow as historical results:

    phase3-historical-comparison              parent run: provenance + all artifacts
      historical-validation-<model>  (x3)     validation metrics per model
      historical-heldout-test-<selected>      the selected model's test metrics

Run from the project root:

    python -m src.ml.log_phase3_to_mlflow

The same saved results are logged only once per experiment (matched on the
SHA-256 of metrics.json); pass --allow-duplicate to log them again.
"""

import argparse
import json
from pathlib import Path

# Imported before mlflow: src.ml.tracking sets MLflow's environment options.
from src.ml import config
from src.ml.tracking import (
    ARTIFACT_ROOT,
    EXPERIMENT_NAME,
    TRACKING_URI,
    flatten_metrics,
    log_files,
    log_metrics,
    log_params,
    set_tags,
    setup_experiment,
    sha256_file,
)

import mlflow  # noqa: E402
from mlflow import MlflowClient  # noqa: E402


PHASE3_DIR = config.ARTIFACTS_DIR / "phase3"
RECORD_TYPE = "historical_phase3_results"
HISTORICAL_NOTE = (
    "Historical Phase 3 results read from saved files. No model was trained, "
    "re-trained or re-evaluated when this run was logged."
)
REQUIRED_KEYS = (
    "split_fingerprint_sha256", "random_seed", "sizes", "positive_class",
    "train_negatives_per_positive", "cv", "pre_training_checks_passed",
    "models", "selected_model", "selection_reasons", "test_selected_model",
)
REQUIRED_MODEL_KEYS = (
    "best_params", "cv_roc_auc_mean", "cv_roc_auc_std", "cv_pr_auc_mean",
    "selected_threshold", "training_seconds", "validation",
    "validation_uncalibrated",
)
SHARED_TABLES = (
    "model_comparison.csv", "subgroups_validation.csv", "subgroups_test.csv",
    "permutation_importance.csv", "metrics.json",
)
TEST_ARTIFACTS = (
    "subgroups_test.csv",
    "plots/confusion_test_selected.png",
    "plots/reliability_test_selected.png",
)


# ---------------------------------------------------------
# Reading and checking the saved Phase 3 outputs
# ---------------------------------------------------------

def load_phase3(phase3_dir):
    """
    Read metrics.json and check that every file to be logged exists,
    before any MLflow run is created.
    """
    phase3_dir = Path(phase3_dir)
    metrics_path = phase3_dir / "metrics.json"
    with open(metrics_path, encoding="utf-8") as handle:
        metrics = json.load(handle)

    missing = [key for key in REQUIRED_KEYS if key not in metrics]
    for name, model in metrics.get("models", {}).items():
        missing += [f"models.{name}.{key}" for key in REQUIRED_MODEL_KEYS
                    if key not in model]
    if missing:
        raise ValueError(f"metrics.json is missing: {missing}")
    selected = metrics["selected_model"]
    if selected not in metrics["models"]:
        raise ValueError(f"Selected model '{selected}' has no metrics entry")

    files = {
        "tables": [phase3_dir / name for name in SHARED_TABLES],
        "cv_results": {
            name: phase3_dir / f"cv_results_{name}.csv"
            for name in metrics["models"]
        },
        "plots": sorted((phase3_dir / "plots").glob("*.png")),
        "test": [phase3_dir / name for name in TEST_ARTIFACTS],
        "selected_model": phase3_dir / "models" / f"{selected}.joblib",
    }
    expected = (
        files["tables"] + list(files["cv_results"].values()) + files["test"]
        + [files["selected_model"]]
    )
    absent = [str(path) for path in expected if not path.is_file()]
    if not files["plots"]:
        absent.append(str(phase3_dir / "plots" / "*.png"))
    if absent:
        raise FileNotFoundError(f"Saved Phase 3 files not found: {absent}")

    return metrics, files, metrics_path


def find_existing(experiment_id, metrics_sha256):
    """Finished runs that already hold these exact saved results."""
    return MlflowClient().search_runs(
        [experiment_id],
        filter_string=(
            f"tags.record_type = '{RECORD_TYPE}' and "
            f"tags.metrics_json_sha256 = '{metrics_sha256}' and "
            "attributes.status = 'FINISHED'"
        ),
    )


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

def log_phase3(phase3_dir=PHASE3_DIR, tracking_uri=TRACKING_URI,
               experiment_name=EXPERIMENT_NAME, artifact_root=ARTIFACT_ROOT,
               allow_duplicate=False):
    """Log the saved Phase 3 results; return the created run ids."""
    metrics, files, metrics_path = load_phase3(phase3_dir)
    metrics_sha256 = sha256_file(metrics_path)
    model_sha256 = sha256_file(files["selected_model"])
    selected = metrics["selected_model"]

    experiment_id = setup_experiment(experiment_name, tracking_uri, artifact_root)
    existing = find_existing(experiment_id, metrics_sha256)
    if existing and not allow_duplicate:
        raise RuntimeError(
            "These saved Phase 3 results are already logged "
            f"(run {existing[0].info.run_id}). Use --allow-duplicate to log again."
        )

    common_tags = {
        "record_type": RECORD_TYPE,
        "historical_note": HISTORICAL_NOTE,
        "retrained": "false",
        "source_phase": "phase3",
        "metrics_json_sha256": metrics_sha256,
        "split_fingerprint_sha256": metrics["split_fingerprint_sha256"],
    }
    run_ids = {}

    with mlflow.start_run(experiment_id=experiment_id,
                          run_name="phase3-historical-comparison",
                          description=HISTORICAL_NOTE) as parent:
        set_tags({
            **common_tags,
            "data_split": "validation and test (see child runs)",
            "selected_model": selected,
            "selected_model_artifact": f"models/{selected}.joblib",
            "selected_model_sha256": model_sha256,
            "selection_reasons": "\n".join(metrics["selection_reasons"]),
        })
        log_params({
            "random_seed": metrics["random_seed"],
            "cv": metrics["cv"],
            "positive_class": metrics["positive_class"],
            "train_size": metrics["sizes"]["train"],
            "validation_size": metrics["sizes"]["validation"],
            "test_size": metrics["sizes"]["test"],
            "train_negatives_per_positive": metrics["train_negatives_per_positive"],
            "pre_training_checks_passed": metrics["pre_training_checks_passed"],
            "selected_model": selected,
        })
        log_files(files["tables"], "tables")
        log_files(files["cv_results"].values(), "cv_results")
        log_files(files["plots"], "plots")
        run_ids["parent"] = parent.info.run_id

        for name, model in metrics["models"].items():
            with mlflow.start_run(experiment_id=experiment_id,
                                  run_name=f"historical-validation-{name}",
                                  nested=True) as child:
                set_tags({
                    **common_tags,
                    "data_split": "validation",
                    "model_name": name,
                    "selected": str(name == selected).lower(),
                })
                log_params({
                    **model["best_params"],
                    "selected_threshold": model["selected_threshold"],
                    "threshold_rule": "maximise F1 on validation",
                    "calibration": "sigmoid, training split only",
                })
                log_metrics({
                    "cv_roc_auc_mean": model["cv_roc_auc_mean"],
                    "cv_roc_auc_std": model["cv_roc_auc_std"],
                    "cv_pr_auc_mean": model["cv_pr_auc_mean"],
                    "training_seconds": model["training_seconds"],
                    **flatten_metrics(model["validation"], "validation"),
                    **flatten_metrics(
                        model["validation_uncalibrated"], "validation_uncalibrated"
                    ),
                })
                log_files([files["cv_results"][name]], "cv_results")
                run_ids[name] = child.info.run_id

        with mlflow.start_run(experiment_id=experiment_id,
                              run_name=f"historical-heldout-test-{selected}",
                              nested=True) as test_run:
            set_tags({
                **common_tags,
                "data_split": "test",
                "evaluation": "held-out test set, selected model only, "
                              "evaluated once in Phase 3",
                "model_name": selected,
                "selected_model_sha256": model_sha256,
            })
            log_params({
                "selected_threshold": metrics["models"][selected]["selected_threshold"],
                "threshold_source": "validation set (Phase 3)",
            })
            log_metrics(flatten_metrics(metrics["test_selected_model"], "test"))
            log_files(files["test"], "test")
            run_ids["test"] = test_run.info.run_id

    return run_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase3-dir", type=Path, default=PHASE3_DIR)
    parser.add_argument("--tracking-uri", default=TRACKING_URI)
    parser.add_argument("--experiment", default=EXPERIMENT_NAME)
    parser.add_argument("--allow-duplicate", action="store_true",
                        help="log again even if these results are already logged")
    args = parser.parse_args()

    try:
        run_ids = log_phase3(args.phase3_dir, args.tracking_uri, args.experiment,
                             ARTIFACT_ROOT, allow_duplicate=args.allow_duplicate)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Nothing logged: {error}")
    print("Logged saved Phase 3 results to MLflow (historical, nothing retrained).")
    print(f"Tracking URI: {args.tracking_uri}")
    print(f"Experiment:   {args.experiment}")
    for label, run_id in run_ids.items():
        print(f"  {label:<20} {run_id}")


if __name__ == "__main__":
    main()
