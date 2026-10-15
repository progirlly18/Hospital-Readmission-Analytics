import json

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.ml import config
from src.ml.features import build_feature_frame
from src.ml.split import assign_splits
from src.ml.train import split_frames
from src.monitoring import retraining
from src.monitoring.drift import (
    UNSEEN,
    DriftBaseline,
    numeric_edges,
    prediction_psi,
    psi_from_shares,
    status,
)
from src.monitoring.report import (
    SIMULATION_NOTE,
    build_report,
    load_splits,
    markdown,
    simulate_shift,
    write_report,
)
from tests.conftest import build_synthetic_cohort

THRESHOLD = 0.1617830043681171


# ---------------------------------------------------------
# PSI
# ---------------------------------------------------------

def test_psi_is_zero_for_identical_shares_and_grows_with_shift():
    assert psi_from_shares([0.5, 0.5], [0.5, 0.5]) == 0.0
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 20000)
    assert prediction_psi(base, rng.normal(0, 1, 20000)) < 0.01
    assert 0.10 <= prediction_psi(base, rng.normal(0.4, 1, 20000)) <= 0.25
    assert prediction_psi(base, rng.normal(1.0, 1, 20000)) > 0.25


def test_psi_is_finite_when_a_bin_is_empty():
    assert np.isfinite(psi_from_shares([1.0, 0.0], [0.0, 1.0]))


def test_status_bands():
    assert [status(v) for v in (0.05, 0.10, 0.25, 0.26)] == [
        "stable", "moderate", "moderate", "significant"]


def test_numeric_edges_merge_ties():
    edges = numeric_edges([1] * 90 + [2] * 10)
    assert len(edges) < 9 and all(a < b for a, b in zip(edges, edges[1:]))


@pytest.fixture(scope="module")
def splits():
    cohort = build_synthetic_cohort(n_patients=700, seed=3)
    return split_frames(cohort, build_feature_frame(cohort), assign_splits(cohort))


def test_feature_psi_flags_only_shifted_features(splits):
    train_x, test_x = splits["train"][0], splits["test"][0]
    baseline = DriftBaseline(train_x)
    unchanged = baseline.feature_psi(test_x)
    assert len(unchanged) == len(config.FEATURE_COLUMNS)
    assert set(unchanged["status"]) == {"stable"}

    shifted = baseline.feature_psi(simulate_shift(test_x)).set_index("feature")
    assert shifted.loc["number_inpatient", "status"] == "significant"
    assert shifted.loc["gender", "status"] == "stable"


def test_unseen_categories_are_counted_not_dropped(splits):
    train_x, test_x = splits["train"][0], splits["test"][0].copy()
    baseline = DriftBaseline(train_x)
    test_x["medical_specialty"] = "NeverSeen"
    row = baseline.feature_psi(test_x).set_index("feature").loc["medical_specialty"]
    assert row["unseen_share"] == 1.0 and row["status"] == "significant"
    assert UNSEEN not in train_x["medical_specialty"].values


# ---------------------------------------------------------
# Retraining criteria
# ---------------------------------------------------------

REFERENCE = {"roc_auc": 0.67, "ece": 0.007, "flagged_rate": 0.21}


def batch(name, psi=0.0, prediction=0.0, roc_auc=0.67, ece=0.007, flagged=0.21,
          subgroups=()):
    return {
        "name": name,
        "feature_drift": [{"feature": "number_inpatient", "psi": psi}],
        "prediction_psi": prediction,
        "performance": {"roc_auc": roc_auc, "ece": ece, "flagged_rate": flagged},
        "subgroups": list(subgroups),
    }


def recommend(*batches):
    return retraining.recommendation(list(batches), REFERENCE, ["number_inpatient"])


def test_stable_batches_recommend_nothing():
    result = recommend(batch("a"), batch("b"))
    assert not result["retraining_recommended"]
    assert result["warnings"] == [] and result["fairness_review"] == []


def test_feature_drift_must_persist_for_two_batches():
    once = recommend(batch("a", psi=0.4), batch("b"))
    assert not once["retraining_recommended"] and once["warnings"]
    twice = recommend(batch("a", psi=0.4), batch("b", psi=0.4))
    assert twice["retraining_recommended"]
    assert "number_inpatient drifted in 2 consecutive batches" in twice["retrain_reasons"][0]


def test_drift_on_an_unimportant_feature_is_ignored():
    result = retraining.recommendation([batch("a", psi=0.9), batch("b", psi=0.9)],
                                       REFERENCE, ["age_years"])
    assert not result["retraining_recommended"]


@pytest.mark.parametrize("changes, criterion", [
    ({"prediction": 0.3}, "prediction_drift"),
    ({"roc_auc": 0.60}, "roc_auc_drop"),
    ({"ece": 0.05}, "calibration"),
    ({"flagged": 0.40}, "flagged_rate"),
])
def test_each_performance_criterion_recommends_retraining(changes, criterion):
    result = recommend(batch("a", **changes))
    assert result["retraining_recommended"]
    assert result["retrain_reasons"] == [f"a: {criterion}"]


def test_limits_are_not_triggered_at_the_boundary():
    result = recommend(batch("a", prediction=0.25, roc_auc=0.621, ece=0.03, flagged=0.309))
    assert not result["retraining_recommended"]


def test_subgroup_gap_asks_for_review_not_retraining():
    groups = [{"attribute": "race", "group": "X", "positives": 40, "roc_auc": 0.50},
              {"attribute": "race", "group": "Y", "positives": 10, "roc_auc": 0.40},
              {"attribute": "race", "group": "Z", "encounters": "<30"}]
    result = recommend(batch("a", subgroups=groups))
    assert not result["retraining_recommended"]
    assert result["fairness_review"] == ["a: subgroup gap ['race=X']"]


# ---------------------------------------------------------
# Report (synthetic cohort and model; no project data)
# ---------------------------------------------------------

@pytest.fixture(scope="module")
def report(splits):
    train_x, train_y, _ = splits["train"]
    model = Pipeline([
        ("preprocess", ColumnTransformer(
            [("categorical", OneHotEncoder(handle_unknown="ignore"),
              config.CATEGORICAL_FEATURES)], remainder="passthrough")),
        ("model", RandomForestClassifier(n_estimators=20, max_depth=4, random_state=0)),
    ]).fit(train_x, train_y)
    return build_report(splits, model, THRESHOLD, "synthetic model", importance_rows=200)


def test_report_has_real_and_clearly_labelled_simulated_batches(report):
    names = [b["name"] for b in report["batches"]]
    assert names == ["test", "simulated_shift_a", "simulated_shift_b"]
    assert all(b["note"] == SIMULATION_NOTE for b in report["batches"][1:])
    assert "subgroups" in report["batches"][0]
    assert "subgroups" not in report["batches"][1]


def test_real_and_simulated_batches_are_judged_separately(report):
    real = report["retraining"]["real_batches"]
    simulated = report["retraining"]["simulated_demonstration"]
    assert [b["batch"] for b in real["batches"]] == ["test"]
    assert [b["batch"] for b in simulated["batches"]] == ["simulated_shift_a",
                                                          "simulated_shift_b"]
    assert simulated["retraining_recommended"]


def test_report_holds_aggregates_only(report, splits, tmp_path):
    out = write_report(report, tmp_path)
    text = (out / "monitoring_report.json").read_text(encoding="utf-8")
    json.loads(text)
    for column in ("encounter_id", "patient_nbr", "patient_sk"):
        assert f'"{column}"' not in text
    patients = splits["test"][2][config.GROUP_COLUMN].astype(str)
    assert not any(f'"{patient}"' in text or f" {patient}," in text
                   for patient in patients if len(patient) > 4)
    assert all(set(b["performance"]) >= {"roc_auc", "ece", "flagged_rate"}
               for b in report["batches"])
    subgroups = report["batches"][0]["subgroups"]
    assert all(isinstance(row["encounters"], int) and row["encounters"] >= 30
               or row["encounters"] == "<30" for row in subgroups)


def test_explainability_ranks_all_39_inputs(report):
    importance = report["explainability"]["importance"]
    assert sorted(row["feature"] for row in importance) == sorted(config.FEATURE_COLUMNS)
    assert len(report["explainability"]["key_features"]) == retraining.KEY_FEATURES


def test_markdown_separates_real_and_simulated_verdicts(report):
    text = markdown(report)
    assert "Real data (test split): retraining recommended" in text
    assert "Simulated shift (demonstration only)" in text


def test_every_batch_reports_input_quality(report):
    for batch in report["batches"]:
        quality = batch["input_quality"]
        assert quality["rows"] == batch["rows"] and quality["passed"]
        assert len(quality["columns"]) == len(config.FEATURE_COLUMNS)
    assert "## Input-data quality" in markdown(report)


def test_load_splits_refuses_a_different_split(tmp_path):
    cohort = build_synthetic_cohort(n_patients=50, seed=1).drop(
        columns=[config.TARGET_COLUMN])
    path = tmp_path / "cleaned.csv"
    cohort.to_csv(path, index=False)
    with pytest.raises(RuntimeError, match="fingerprint"):
        load_splits(path)
    assert set(load_splits(path, expected=None)) == {"train", "validation", "test"}


@pytest.mark.skipif(not config.CLEANED_FILE.is_file(), reason="cleaned data not available")
def test_real_split_matches_the_trained_fingerprint():
    splits = load_splits()
    assert [len(splits[name][0]) for name in ("train", "validation", "test")] == [
        69538, 14901, 14901]
