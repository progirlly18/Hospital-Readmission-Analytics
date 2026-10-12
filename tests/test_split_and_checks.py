import pandas as pd
import pytest

from src.ml import config
from src.ml.checks import (
    check_forbidden_columns,
    check_prediction_time_availability,
    run_all_checks,
)
from src.ml.features import build_feature_frame
from src.ml.split import assign_splits, patient_overlap, summarize_splits


def test_split_has_zero_patient_overlap(synthetic_cohort):
    split = assign_splits(synthetic_cohort)
    assert patient_overlap(synthetic_cohort, split) == {
        "train_validation": 0, "train_test": 0, "validation_test": 0,
    }


def test_every_row_gets_exactly_one_split(synthetic_cohort):
    split = assign_splits(synthetic_cohort)
    assert split.notna().all()
    assert set(split) == {"train", "validation", "test"}


def test_split_proportions_close_to_70_15_15(synthetic_cohort):
    summary = summarize_splits(
        synthetic_cohort, assign_splits(synthetic_cohort)
    ).set_index("split")
    assert summary.loc["train", "share_of_encounters"] == pytest.approx(0.70, abs=0.05)
    assert summary.loc["validation", "share_of_encounters"] == pytest.approx(0.15, abs=0.04)
    assert summary.loc["test", "share_of_encounters"] == pytest.approx(0.15, abs=0.04)


def test_split_is_reproducible(synthetic_cohort):
    first = assign_splits(synthetic_cohort)
    second = assign_splits(synthetic_cohort)
    pd.testing.assert_series_equal(first, second)


def test_different_seed_changes_split(synthetic_cohort):
    assert not assign_splits(synthetic_cohort, seed=1).equals(
        assign_splits(synthetic_cohort, seed=2)
    )


def test_forbidden_column_check_catches_leaks():
    leaked = pd.DataFrame(columns=["age_years", "readmitted", "patient_sk"])
    result = check_forbidden_columns(leaked)
    assert not result["passed"]
    assert result["detail"] == ["patient_sk", "readmitted"]


def test_availability_check_catches_undocumented_feature():
    frame = pd.DataFrame(columns=["age_years", "days_until_next_visit"])
    result = check_prediction_time_availability(frame)
    assert not result["passed"]
    assert result["detail"] == ["days_until_next_visit"]


def test_single_feature_leakage_check_flags_target_copy(synthetic_cohort):
    features = build_feature_frame(synthetic_cohort)
    # Plant a leak: replace a numeric feature with the target itself.
    features["number_inpatient"] = synthetic_cohort[config.TARGET_COLUMN]
    split = assign_splits(synthetic_cohort)
    checks, _ = run_all_checks(synthetic_cohort, features, split)
    leak_check = [c for c in checks if "single feature" in c["check"]][0]
    assert not leak_check["passed"]
    assert "number_inpatient" in leak_check["detail"]


def test_all_checks_pass_on_clean_synthetic_data(synthetic_cohort):
    features = build_feature_frame(synthetic_cohort)
    split = assign_splits(synthetic_cohort)
    checks, balance = run_all_checks(synthetic_cohort, features, split)
    assert all(check["passed"] for check in checks), checks
    assert set(balance) == {"train", "validation", "test"}


def test_split_does_not_depend_on_row_order(synthetic_cohort):
    shuffled = synthetic_cohort.sample(frac=1, random_state=3)
    original = assign_splits(synthetic_cohort)
    reordered = assign_splits(shuffled).reindex(original.index)
    pd.testing.assert_series_equal(original, reordered)


def test_split_keeps_positive_rates_similar(synthetic_cohort):
    summary = summarize_splits(synthetic_cohort, assign_splits(synthetic_cohort))
    rates = summary["positive_rate"]
    assert rates.max() - rates.min() < 0.05


def test_split_rejects_bad_fractions(synthetic_cohort):
    with pytest.raises(ValueError, match="add up to 1"):
        assign_splits(synthetic_cohort, fractions=(0.7, 0.2, 0.2))
