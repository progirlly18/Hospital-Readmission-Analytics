import json

import numpy as np
import pytest

from src.ml import config
from src.ml.features import build_feature_frame
from src.monitoring.data_quality import QUALITY_LIMIT, input_quality
from tests.conftest import build_synthetic_cohort


@pytest.fixture(scope="module")
def features():
    return build_feature_frame(build_synthetic_cohort(n_patients=300, seed=5))


def by_feature(result):
    return {row["feature"]: row for row in result["columns"]}


def test_clean_batch_passes_with_every_input_checked(features):
    result = input_quality(features)
    assert result["passed"] and result["failing_features"] == []
    assert result["missing_values"] == result["invalid_values"] == 0
    assert [row["feature"] for row in result["columns"]] == config.FEATURE_COLUMNS


def test_missing_and_invalid_values_are_counted_per_column(features):
    bad = features.copy()
    bad.loc[bad.index[:5], "num_lab_procedures"] = np.nan
    bad.loc[bad.index[:3], "time_in_hospital"] = 30          # API allows 1-14
    bad.loc[bad.index[:4], "age_years"] = 42.0               # not a band midpoint
    bad["admission_type_id"] = bad["admission_type_id"].astype(object)
    bad.loc[bad.index[:2], "admission_type_id"] = "99"       # unknown category
    rows = by_feature(input_quality(bad))
    assert rows["num_lab_procedures"]["missing"] == 5
    assert rows["time_in_hospital"]["invalid"] == 3
    assert rows["age_years"]["invalid"] == 4
    assert rows["admission_type_id"]["invalid"] == 2
    assert rows["gender"]["missing"] == rows["gender"]["invalid"] == 0


def test_excluded_discharge_codes_are_invalid(features):
    bad = features.copy()
    bad["discharge_disposition_id"] = bad["discharge_disposition_id"].astype(object)
    bad.loc[bad.index[:6], "discharge_disposition_id"] = "11"  # expired
    assert by_feature(input_quality(bad))["discharge_disposition_id"]["invalid"] == 6


def test_batch_fails_only_above_the_limit(features):
    bad = features.copy()
    at_limit = int(len(bad) * QUALITY_LIMIT)
    bad.loc[bad.index[:at_limit], "number_inpatient"] = np.nan
    assert input_quality(bad)["passed"]
    bad.loc[bad.index[:at_limit + 1], "number_inpatient"] = np.nan
    result = input_quality(bad)
    assert not result["passed"] and result["failing_features"] == ["number_inpatient"]


def test_absent_column_fails(features):
    result = input_quality(features.drop(columns=["gender"]))
    assert not result["passed"] and "gender" in result["failing_features"]
    assert by_feature(result)["gender"]["absent"]


def test_report_holds_counts_not_values(features):
    bad = features.copy()
    bad["admission_type_id"] = bad["admission_type_id"].astype(object)
    bad.loc[bad.index[:1], "admission_type_id"] = "SECRET-VALUE"
    text = json.dumps(input_quality(bad))
    assert "SECRET-VALUE" not in text
