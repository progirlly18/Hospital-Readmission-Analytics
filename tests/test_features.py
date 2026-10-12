import numpy as np
import pandas as pd
import pytest

from src.ml import config
from src.ml.features import (
    age_band_to_years,
    build_feature_frame,
    build_preprocessor,
    map_icd9_group,
)
from tests.conftest import make_encounter


@pytest.mark.parametrize("code, expected", [
    ("428", "Circulatory"),
    ("414", "Circulatory"),
    ("250", "Diabetes"),
    ("250.01", "Diabetes"),
    ("250.83", "Diabetes"),          # Part 1 groups this as "Other"
    ("486", "Respiratory"),
    ("996", "Injury/Poisoning"),
    ("715", "Musculoskeletal"),
    ("38", "Infectious"),
    ("8", "Infectious"),
    ("V57", "Supplementary (V)"),
    ("E909", "External cause (E)"),
    ("239.99", "Neoplasms"),
    ("240", "Endocrine/Metabolic"),
    ("Unknown", "Missing"),
    ("?", "Missing"),
    (None, "Missing"),
    (pd.NA, "Missing"),
])
def test_icd9_grouping(code, expected):
    assert map_icd9_group(code) == expected


def test_decimal_codes_use_their_numeric_value():
    # "427.31" (atrial fibrillation) must stay circulatory, not become 42731.
    assert map_icd9_group("427.31") == "Circulatory"
    assert map_icd9_group("599.0") == "Genitourinary"


@pytest.mark.parametrize("band, years", [
    ("0-10", 5), ("50-60", 55), ("[90-100)", 95),
])
def test_age_band_midpoint(band, years):
    assert age_band_to_years(band) == years


def test_feature_frame_has_exactly_the_configured_columns(synthetic_cohort):
    features = build_feature_frame(synthetic_cohort)
    assert list(features.columns) == config.FEATURE_COLUMNS
    assert len(features) == len(synthetic_cohort)


def test_feature_frame_excludes_forbidden_columns(synthetic_cohort):
    features = build_feature_frame(synthetic_cohort)
    assert not set(features.columns) & config.FORBIDDEN_FEATURE_COLUMNS
    for forbidden in ["race", "payer_code", "weight", "readmitted",
                      "patient_sk", "patient_nbr", "encounter_id"]:
        assert forbidden not in features.columns


def test_features_do_not_change_when_race_or_payer_change():
    base = pd.DataFrame([make_encounter(1, 1)])
    other = pd.DataFrame([make_encounter(1, 1, race="Asian", payer_code="BC")])
    pd.testing.assert_frame_equal(
        build_feature_frame(base), build_feature_frame(other)
    )


def test_features_do_not_depend_on_target():
    yes = pd.DataFrame([make_encounter(1, 1, "<30")])
    no = pd.DataFrame([make_encounter(1, 1, "NO")])
    pd.testing.assert_frame_equal(build_feature_frame(yes), build_feature_frame(no))


def test_feature_values():
    row = pd.DataFrame([make_encounter(
        1, 1, age="70-80", metformin="Up", A1Cresult=pd.NA,
        diag_1="250.83", medical_specialty="Unknown",
    )])
    features = build_feature_frame(row).iloc[0]
    assert features["age_years"] == 75
    assert features["medication_count"] == 2          # insulin + metformin
    assert features["A1Cresult"] == "None"            # test not performed
    assert features["max_glu_serum"] == "None"
    assert features["diag_1_group"] == "Diabetes"
    assert features["medical_specialty"] == "Missing"
    assert features["admission_type_id"] == "1"


def test_feature_frame_is_row_wise(synthetic_cohort):
    """A row's features must not depend on which other rows are present."""
    full = build_feature_frame(synthetic_cohort)
    subset = build_feature_frame(synthetic_cohort.iloc[:10])
    pd.testing.assert_frame_equal(full.iloc[:10], subset)


def test_preprocessor_learns_only_from_training_rows(synthetic_cohort):
    features = build_feature_frame(synthetic_cohort)
    train = features.iloc[:300]
    holdout = features.iloc[300:].copy()
    holdout["time_in_hospital"] = 1000  # extreme values outside training

    preprocessor = build_preprocessor().fit(train)
    numeric = preprocessor.named_transformers_["numeric"]
    position = config.NUMERIC_FEATURES.index("time_in_hospital")

    assert numeric.named_steps["scale"].mean_[position] == pytest.approx(
        train["time_in_hospital"].mean()
    )
    # Refitting on the holdout would change the mean; the train fit must not.
    assert numeric.named_steps["scale"].mean_[position] < 100
    preprocessor.transform(holdout)


def test_preprocessor_handles_unseen_categories(synthetic_cohort):
    features = build_feature_frame(synthetic_cohort)
    preprocessor = build_preprocessor().fit(features)

    new = features.iloc[:1].copy()
    new["medical_specialty"] = "Never-Seen-Specialty"
    new["diag_1_group"] = "Never-Seen-Group"
    output = preprocessor.transform(new)

    assert output.shape == (1, len(preprocessor.get_feature_names_out()))
    assert np.isfinite(output).all()


def test_preprocessor_imputes_missing_numeric(synthetic_cohort):
    features = build_feature_frame(synthetic_cohort)
    preprocessor = build_preprocessor().fit(features)
    new = features.iloc[:1].copy()
    new["num_lab_procedures"] = np.nan
    assert np.isfinite(preprocessor.transform(new)).all()
