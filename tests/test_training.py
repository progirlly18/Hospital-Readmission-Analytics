"""
Training pipeline tests on small synthetic data (fast settings).
"""

import numpy as np
import pytest

from src.ml import config
from src.ml.features import build_feature_frame
from src.ml.models import build_model_specs, negatives_per_positive
from src.ml.split import assign_splits
from src.ml.train import run
from tests.conftest import build_synthetic_cohort


def run_quick(cohort, split=None):
    features = build_feature_frame(cohort)
    if split is None:
        split = assign_splits(cohort)
    return run(cohort, features, split, quick=True, cv_folds=3,
               importance_repeats=1)


@pytest.fixture(scope="module")
def cohort():
    return build_synthetic_cohort(n_patients=500, seed=7)


@pytest.fixture(scope="module")
def outcome(cohort):
    return run_quick(cohort)


def test_all_three_models_trained_and_evaluated(outcome):
    assert set(outcome["results"]) == {
        "logistic_regression", "random_forest", "xgboost"
    }
    assert outcome["selected"] in outcome["results"]
    for result in outcome["results"].values():
        for key in ("roc_auc", "pr_auc", "brier", "ece", "at_0_5", "at_selected"):
            assert key in result["validation"]


def test_models_compared_on_the_same_validation_rows(outcome):
    sizes = {len(r["validation_probabilities"]) for r in outcome["results"].values()}
    assert sizes == {outcome["sizes"]["validation"]}
    positives = {r["validation"]["positives"] for r in outcome["results"].values()}
    assert len(positives) == 1


def test_probabilities_are_valid(outcome):
    for result in outcome["results"].values():
        prob = result["validation_probabilities"]
        assert np.all((prob >= 0) & (prob <= 1))
    assert np.all((outcome["test_probabilities"] >= 0)
                  & (outcome["test_probabilities"] <= 1))


def test_only_selected_model_is_scored_on_test(outcome):
    for name, result in outcome["results"].items():
        assert "test" not in result
    assert outcome["test"]["n"] == outcome["sizes"]["test"]


def test_training_is_reproducible(cohort, outcome):
    again = run_quick(cohort)
    assert again["selected"] == outcome["selected"]
    for name, result in outcome["results"].items():
        # Equal up to floating-point rounding from multi-threaded training.
        np.testing.assert_allclose(
            again["results"][name]["validation_probabilities"],
            result["validation_probabilities"], rtol=1e-6,
        )
        assert again["results"][name]["selected_threshold"] == pytest.approx(
            result["selected_threshold"], rel=1e-6
        )


def test_test_labels_do_not_affect_selection_or_threshold(cohort, outcome):
    """
    Flip every test-set label: selection and thresholds must not change.

    The split is computed once from the original data and reused, because the
    split itself is stratified on labels (flipping them would move patients).
    """
    split = assign_splits(cohort)
    tampered = cohort.copy()
    test_rows = (split == "test").to_numpy()
    tampered.loc[test_rows, config.TARGET_COLUMN] = (
        1 - tampered.loc[test_rows, config.TARGET_COLUMN]
    )
    changed = run_quick(tampered, split=split)

    assert changed["selected"] == outcome["selected"]
    for name, result in outcome["results"].items():
        # Equal up to floating-point rounding from multi-threaded training.
        assert changed["results"][name]["selected_threshold"] == pytest.approx(
            result["selected_threshold"], rel=1e-6
        )
        assert changed["results"][name]["validation"]["roc_auc"] == pytest.approx(
            result["validation"]["roc_auc"], rel=1e-6
        )
    # The test metrics themselves must change (labels were flipped).
    assert changed["test"]["roc_auc"] != outcome["test"]["roc_auc"]


def test_preprocessing_fitted_on_training_rows_only(cohort, outcome):
    features = build_feature_frame(cohort)
    train = features.loc[(assign_splits(cohort) == "train").to_numpy()]
    calibrated = outcome["results"]["logistic_regression"]["pipeline"]
    pipeline = calibrated.calibrated_classifiers_[0].estimator
    scaler = pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps["scale"]
    position = config.NUMERIC_FEATURES.index("time_in_hospital")
    assert scaler.mean_[position] == pytest.approx(train["time_in_hospital"].mean())
    assert scaler.n_samples_seen_ == len(train)


def test_class_imbalance_handled_with_training_weights_only():
    y_train = np.array([1] * 10 + [0] * 90)
    specs = {spec.name: spec for spec in build_model_specs(y_train)}
    assert negatives_per_positive(y_train) == 9
    assert specs["xgboost"].pipeline.named_steps["model"].scale_pos_weight == 9
    assert specs["logistic_regression"].pipeline.named_steps["model"].class_weight == "balanced"
    assert specs["random_forest"].pipeline.named_steps["model"].class_weight == "balanced_subsample"


def test_calibrated_probabilities_close_to_observed_rate(outcome):
    for result in outcome["results"].values():
        v = result["validation"]
        # Uncalibrated class-weighted scores sit near 0.5; calibrated ones
        # should be much closer to the real positive rate.
        assert (abs(v["mean_predicted_probability"] - v["observed_positive_rate"])
                < abs(result["validation_uncalibrated"]["mean_predicted_probability"]
                      - v["observed_positive_rate"]))


def test_subgroup_and_importance_outputs(outcome):
    assert set(outcome["subgroups_test"]["attribute"]) == {"gender", "age_group", "race"}
    assert list(outcome["importance"].columns) == [
        "feature", "importance_mean", "importance_std"
    ]
    assert len(outcome["importance"]) == len(config.FEATURE_COLUMNS)
    assert "race" not in set(outcome["importance"]["feature"])
