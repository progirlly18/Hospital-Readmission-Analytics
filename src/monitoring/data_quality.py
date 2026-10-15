"""
Phase 7 follow-up: input-data quality checks for a batch of model inputs.

The API rejects a bad request one at a time (HTTP 422, src/api/schemas.py).
This module applies the same rules to a whole batch, so a monitoring run can
say how many values in it would have been rejected, column by column:

- missing: the value is empty (NaN/None);
- invalid: a numeric value outside the API's bounds (age must be a 10-year
  band midpoint), a category the API does not accept, or an expired or
  hospice discharge code (outside the model's cohort).

A batch fails the check when any column has more than QUALITY_LIMIT of its
rows missing or invalid. A failed check calls for the data pipeline to be
investigated before drift or performance results are trusted; it is not a
retraining criterion by itself. Only counts and rates are reported, never
values.
"""

import pandas as pd

from src.ml import config
from src.api.allowed_values import ALLOWED_VALUES
from src.api.schemas import AGE_BAND_MIDPOINTS, EXCLUDED_DISCHARGE_CODES, INTEGER_BOUNDS

QUALITY_LIMIT = 0.01  # share of rows missing or invalid in any one column


def _invalid_mask(column, values):
    """True where a present value would be rejected by the API."""
    present = values.notna()
    if column == "age_years":
        numeric = pd.to_numeric(values, errors="coerce")
        return present & ~numeric.isin(AGE_BAND_MIDPOINTS)
    if column in INTEGER_BOUNDS:
        low, high = INTEGER_BOUNDS[column]
        numeric = pd.to_numeric(values, errors="coerce")
        whole = numeric.notna() & (numeric == numeric.round())
        return present & ~(whole & numeric.between(low, high))
    text = values.astype(str)
    invalid = present & ~text.isin(ALLOWED_VALUES[column])
    if column == "discharge_disposition_id":
        invalid |= present & text.isin(EXCLUDED_DISCHARGE_CODES)
    return invalid


def input_quality(features, columns=config.FEATURE_COLUMNS, limit=QUALITY_LIMIT):
    """Missing and invalid counts per input column, and an overall verdict."""
    rows = len(features)
    report = []
    for column in columns:
        if column not in features:
            report.append({"feature": column, "missing": rows, "invalid": 0,
                           "problem_rate": 1.0 if rows else 0.0, "absent": True})
            continue
        values = features[column]
        missing = int(values.isna().sum())
        invalid = int(_invalid_mask(column, values).sum())
        rate = (missing + invalid) / rows if rows else 0.0
        report.append({"feature": column, "missing": missing, "invalid": invalid,
                       "problem_rate": round(rate, 6), "absent": False})
    failing = [row["feature"] for row in report if row["problem_rate"] > limit]
    return {
        "rows": int(rows),
        "limit": limit,
        "passed": not failing,
        "failing_features": failing,
        "missing_values": int(sum(row["missing"] for row in report)),
        "invalid_values": int(sum(row["invalid"] for row in report)),
        "columns": report,
    }
