"""
Integration test on the real Part 1 cleaned CSV.
Skipped automatically when the file is not present.
"""

import pytest

from src.ml import config
from src.ml.prepare_data import prepare

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not config.CLEANED_FILE.exists(),
        reason="Part 1 cleaned CSV not found",
    ),
]


@pytest.fixture(scope="module")
def prepared():
    return prepare()


def test_cohort_counts(prepared):
    cohort, _, _, report = prepared
    assert report["cohort"]["source_rows"] == 101_766
    removed = [step["encounters_removed"] for step in report["cohort"]["exclusions"]]
    assert removed == [1_652, 771, 3]
    assert len(cohort) == 99_340


def test_target_definition_matches_part1(prepared):
    cohort, _, _, _ = prepared
    assert (cohort[config.TARGET_COLUMN] == (cohort["readmitted"] == "<30")).all()


def test_all_pre_training_checks_pass(prepared):
    _, _, _, report = prepared
    failed = [check for check in report["checks"] if not check["passed"]]
    assert not failed, failed


def test_split_positive_rates_within_one_point(prepared):
    _, _, _, report = prepared
    rates = [row["positive_rate"] for row in report["split_summary"]]
    assert max(rates) - min(rates) < 0.01
