import pandas as pd
import pytest

from src.ml import config
from src.ml.data import apply_cohort_filters, build_target, load_cleaned_data
from tests.conftest import make_encounter


def test_target_positive_only_for_under_30():
    df = pd.DataFrame([
        make_encounter(1, 1, "<30"),
        make_encounter(2, 2, ">30"),
        make_encounter(3, 3, "NO"),
    ])
    assert build_target(df)[config.TARGET_COLUMN].tolist() == [1, 0, 0]


def test_target_rejects_unexpected_labels():
    df = pd.DataFrame([make_encounter(1, 1, "MAYBE")])
    with pytest.raises(ValueError, match="unexpected"):
        build_target(df)


def test_target_does_not_modify_input():
    df = pd.DataFrame([make_encounter(1, 1, "<30")])
    build_target(df)
    assert config.TARGET_COLUMN not in df.columns


def test_cohort_excludes_expired_hospice_and_invalid_gender():
    df = build_target(pd.DataFrame([
        make_encounter(1, 1, "NO", discharge_disposition_id="11"),   # expired
        make_encounter(2, 2, "NO", discharge_disposition_id="20"),   # expired
        make_encounter(3, 3, "<30", discharge_disposition_id="13"),  # hospice
        make_encounter(4, 4, "NO", discharge_disposition_id="14"),   # hospice
        make_encounter(5, 5, "NO", gender="Unknown/Invalid"),
        make_encounter(6, 6, "<30", discharge_disposition_id="1"),   # kept
        make_encounter(7, 7, "NO", discharge_disposition_id="6"),    # kept
    ]))

    cohort, exclusions = apply_cohort_filters(df)

    assert cohort["encounter_id"].tolist() == ["6", "7"]
    removed = {step["rule"].split(" (")[0]: step for step in exclusions}
    assert removed["Expired"]["encounters_removed"] == 2
    assert removed["Hospice"]["encounters_removed"] == 2
    assert removed["Hospice"]["positives_removed"] == 1
    assert removed["Invalid gender"]["encounters_removed"] == 1


def test_cohort_counts_each_encounter_once():
    # Expired AND invalid gender: counted under the first rule only.
    df = build_target(pd.DataFrame([
        make_encounter(1, 1, "NO", discharge_disposition_id="11",
                       gender="Unknown/Invalid"),
    ]))
    _, exclusions = apply_cohort_filters(df)
    assert sum(step["encounters_removed"] for step in exclusions) == 1


def test_missing_file_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Part 1 pipeline"):
        load_cleaned_data(tmp_path / "missing.csv")
