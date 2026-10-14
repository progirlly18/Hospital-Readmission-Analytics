import json
import subprocess
import sys

import numpy as np
import pytest

from src.ml import config
from src.ml.metrics import evaluate, probability_metrics
from src.ml.tracking import flatten_metrics, metric_key, sha256_file

mlflow = pytest.importorskip("mlflow")

from mlflow import MlflowClient  # noqa: E402

from src.ml.log_phase3_to_mlflow import (  # noqa: E402
    RECORD_TYPE,
    load_phase3,
    log_phase3,
)


MODELS = ("logistic_regression", "random_forest", "xgboost")


def synthetic_metrics(seed=0):
    """A small metrics.json with the same structure train.py writes."""
    rng = np.random.default_rng(seed)
    y = (rng.random(300) < 0.15).astype(int)

    def scored(shift):
        return np.clip(0.1 + 0.2 * y + shift * rng.random(300), 0.0, 1.0)

    models = {}
    for i, name in enumerate(MODELS):
        prob = scored(0.3 + 0.1 * i)
        models[name] = {
            "best_params": {"model__depth": i + 1},
            "cv_roc_auc_mean": 0.66 + i / 100,
            "cv_roc_auc_std": 0.005,
            "cv_pr_auc_mean": 0.21,
            "selected_threshold": 0.2 + i / 100,
            "training_seconds": 1.5,
            "validation": evaluate(y, prob, 0.2 + i / 100),
            "validation_uncalibrated": probability_metrics(y, np.sqrt(prob)),
        }
    return {
        "split_fingerprint_sha256": "ab" * 32,
        "random_seed": 42,
        "sizes": {"train": 700, "validation": 300, "test": 300},
        "positive_class": "readmitted <30 days",
        "train_negatives_per_positive": 5.7,
        "cv": "GroupKFold(5) by patient on training split",
        "pre_training_checks_passed": True,
        "models": models,
        "selected_model": "random_forest",
        "selection_reasons": ["reason one", "reason two"],
        "test_selected_model": evaluate(y, scored(0.4), 0.21),
    }


def write_phase3_dir(root, metrics):
    root.mkdir(parents=True)
    (root / "plots").mkdir()
    (root / "models").mkdir()
    (root / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    for name in ("model_comparison.csv", "subgroups_validation.csv",
                 "subgroups_test.csv", "permutation_importance.csv"):
        (root / name).write_text("a,b\n1,2\n", encoding="utf-8")
    for name in MODELS:
        (root / f"cv_results_{name}.csv").write_text("x\n1\n", encoding="utf-8")
    for name in ("roc_pr_validation.png", "confusion_test_selected.png",
                 "reliability_test_selected.png"):
        (root / "plots" / name).write_bytes(b"\x89PNG fake")
    (root / "models" / "random_forest.joblib").write_bytes(b"saved model bytes")
    return root


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    """One temporary SQLite tracking store for the module (creation is slow)."""
    root = tmp_path_factory.mktemp("mlflow")
    uri = f"sqlite:///{(root / 'mlflow.db').as_posix()}"
    yield {"uri": uri, "artifacts": root / "mlartifacts"}
    mlflow.set_tracking_uri(None)


@pytest.fixture(scope="module")
def logged(store, tmp_path_factory):
    phase3 = write_phase3_dir(
        tmp_path_factory.mktemp("phase3") / "phase3", synthetic_metrics()
    )
    run_ids = log_phase3(phase3, store["uri"], "test-experiment", store["artifacts"])
    client = MlflowClient(tracking_uri=store["uri"])
    runs = {label: client.get_run(run_id) for label, run_id in run_ids.items()}
    return {"phase3": phase3, "run_ids": run_ids, "runs": runs, "client": client}


# ---------------------------------------------------------
# Helpers (no MLflow store needed)
# ---------------------------------------------------------

def test_flatten_keeps_numbers_and_drops_reports_text_and_booleans():
    flat = flatten_metrics({
        "roc_auc": 0.7,
        "n": 10,
        "flag": True,
        "label": "text",
        "at_0_5": {"precision": 0.5, "confusion_matrix": {"tp": 3},
                   "classification_report": {"a": 1.0}},
    }, "validation")
    assert flat == {
        "validation/roc_auc": 0.7,
        "validation/n": 10.0,
        "validation/at_0_5/precision": 0.5,
        "validation/at_0_5/confusion_matrix/tp": 3.0,
    }


def test_metric_key_replaces_characters_mlflow_rejects():
    assert metric_key("precision@0.5", "<30") == "precision_0.5/_30"


def test_missing_saved_file_is_reported_before_any_run(tmp_path):
    phase3 = write_phase3_dir(tmp_path / "phase3", synthetic_metrics())
    (phase3 / "cv_results_xgboost.csv").unlink()
    with pytest.raises(FileNotFoundError, match="cv_results_xgboost"):
        load_phase3(phase3)


def test_missing_metrics_key_is_reported(tmp_path):
    metrics = synthetic_metrics()
    del metrics["models"]["xgboost"]["validation"]
    phase3 = write_phase3_dir(tmp_path / "phase3", metrics)
    with pytest.raises(ValueError, match="models.xgboost.validation"):
        load_phase3(phase3)


def test_logging_script_does_not_import_training_code():
    code = ("import sys, src.ml.log_phase3_to_mlflow; "
            "assert 'src.ml.train' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True,
                   cwd=config.PROJECT_ROOT)


# ---------------------------------------------------------
# Logging to a temporary MLflow store
# ---------------------------------------------------------

def test_one_parent_three_model_runs_and_one_test_run(logged):
    runs, parent_id = logged["runs"], logged["run_ids"]["parent"]
    assert set(runs) == {"parent", *MODELS, "test"}
    for label, run in runs.items():
        if label != "parent":
            assert run.data.tags["mlflow.parentRunId"] == parent_id


def test_every_run_is_labelled_historical(logged):
    for run in logged["runs"].values():
        assert run.data.tags["record_type"] == RECORD_TYPE
        assert run.data.tags["retrained"] == "false"
        assert "No model was trained" in run.data.tags["historical_note"]


def test_provenance_tags_match_saved_files(logged):
    tags = logged["runs"]["parent"].data.tags
    phase3 = logged["phase3"]
    assert tags["metrics_json_sha256"] == sha256_file(phase3 / "metrics.json")
    assert tags["selected_model_sha256"] == sha256_file(
        phase3 / "models" / "random_forest.joblib"
    )
    assert tags["split_fingerprint_sha256"] == "ab" * 32
    assert tags["selected_model"] == "random_forest"
    assert tags["selection_reasons"] == "reason one\nreason two"
    assert logged["runs"]["parent"].data.params["random_seed"] == "42"


def test_model_runs_hold_required_validation_metrics(logged):
    saved = synthetic_metrics()["models"]
    for name in MODELS:
        run = logged["runs"][name]
        metrics, v = run.data.metrics, saved[name]["validation"]
        assert run.data.tags["data_split"] == "validation"
        assert metrics["validation/roc_auc"] == pytest.approx(v["roc_auc"])
        assert metrics["validation/brier"] == pytest.approx(v["brier"])
        assert metrics["validation/ece"] == pytest.approx(v["ece"])
        for key in ("precision", "recall", "f1"):
            assert metrics[f"validation/at_selected/{key}"] == pytest.approx(
                v["at_selected"][key]
            )
        assert run.data.params["model__depth"] == str(saved[name]["best_params"]["model__depth"])
        assert not any(key.startswith("test") for key in metrics)
    assert logged["runs"]["random_forest"].data.tags["selected"] == "true"
    assert logged["runs"]["xgboost"].data.tags["selected"] == "false"


def test_test_metrics_are_only_in_the_labelled_test_run(logged):
    run = logged["runs"]["test"]
    saved = synthetic_metrics()["test_selected_model"]
    assert run.data.tags["data_split"] == "test"
    assert "held-out test" in run.data.tags["evaluation"]
    assert run.data.metrics["test/roc_auc"] == pytest.approx(saved["roc_auc"])
    assert run.data.metrics["test/at_selected/recall"] == pytest.approx(
        saved["at_selected"]["recall"]
    )
    assert all(key.startswith("test/") for key in run.data.metrics)


def test_artifacts_are_logged(logged):
    client, ids = logged["client"], logged["run_ids"]

    def paths(run_id, folder):
        return {a.path for a in client.list_artifacts(run_id, folder)}

    assert paths(ids["parent"], "tables") == {
        "tables/model_comparison.csv", "tables/subgroups_validation.csv",
        "tables/subgroups_test.csv", "tables/permutation_importance.csv",
        "tables/metrics.json",
    }
    assert paths(ids["parent"], "cv_results") == {
        f"cv_results/cv_results_{name}.csv" for name in MODELS
    }
    assert "plots/roc_pr_validation.png" in paths(ids["parent"], "plots")
    assert paths(ids["xgboost"], "cv_results") == {"cv_results/cv_results_xgboost.csv"}
    assert "test/subgroups_test.csv" in paths(ids["test"], "test")


def test_no_model_is_logged_or_registered(logged):
    client = logged["client"]
    for run_id in logged["run_ids"].values():
        assert not any(a.path.endswith(".joblib")
                       for a in client.list_artifacts(run_id))
    assert client.search_registered_models() == []


def test_same_saved_results_are_not_logged_twice(logged, store):
    with pytest.raises(RuntimeError, match="already logged"):
        log_phase3(logged["phase3"], store["uri"], "test-experiment",
                   store["artifacts"])
