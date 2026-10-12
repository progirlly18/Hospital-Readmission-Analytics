"""
Load the Part 1 cleaned dataset, build the binary target and apply the
modelling cohort rules.
"""

import pandas as pd

from src.ml import config


def load_cleaned_data(path=config.CLEANED_FILE):
    """
    Read the Part 1 cleaned CSV.

    All columns are read as text so diagnosis codes such as "250.83" or
    "V57" keep their exact form; numeric columns are converted later.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {path}. "
            "Run the Part 1 pipeline (ingest.py, clean.py) first."
        )

    df = pd.read_csv(path, dtype=str, keep_default_na=False)

    # Part 1 writes missing values as empty strings.
    df = df.replace("", pd.NA)

    return df


def build_target(df):
    """
    Add the binary target column.

    1 = readmitted within 30 days ("<30")
    0 = readmitted after 30 days (">30") or not readmitted ("NO")
    """
    labels = df[config.TARGET_SOURCE_COLUMN].astype("string").str.strip()

    invalid = ~labels.isin(config.VALID_READMISSION_LABELS)
    if invalid.any():
        raise ValueError(
            f"{int(invalid.sum())} rows have an unexpected "
            f"'{config.TARGET_SOURCE_COLUMN}' value; expected one of "
            f"{sorted(config.VALID_READMISSION_LABELS)}."
        )

    df = df.copy()
    df[config.TARGET_COLUMN] = labels.eq(config.POSITIVE_LABEL).astype(int)
    return df


def apply_cohort_filters(df):
    """
    Remove encounters outside the modelling population.

    Returns the filtered data and a list of exclusion steps with counts,
    applied in order (an encounter is counted under the first rule it hits).
    """
    disposition = pd.to_numeric(
        df["discharge_disposition_id"], errors="coerce"
    )

    rules = [
        (
            "Expired (discharge_disposition_id 11, 19, 20, 21)",
            disposition.isin(config.EXPIRED_DISPOSITION_IDS),
        ),
        (
            "Hospice (discharge_disposition_id 13, 14)",
            disposition.isin(config.HOSPICE_DISPOSITION_IDS),
        ),
        (
            "Invalid gender (Unknown/Invalid)",
            df["gender"].isin(config.INVALID_GENDER_VALUES),
        ),
    ]

    keep = pd.Series(True, index=df.index)
    exclusions = []

    for name, mask in rules:
        newly_excluded = keep & mask
        removed = df.loc[newly_excluded]
        exclusions.append({
            "rule": name,
            "encounters_removed": int(newly_excluded.sum()),
            "positives_removed": (
                int(removed[config.TARGET_COLUMN].sum())
                if config.TARGET_COLUMN in removed
                else None
            ),
        })
        keep &= ~mask

    return df.loc[keep].reset_index(drop=True), exclusions


def load_modelling_data(path=config.CLEANED_FILE):
    """Load cleaned data, build the target and apply the cohort rules."""
    raw = load_cleaned_data(path)
    with_target = build_target(raw)
    cohort, exclusions = apply_cohort_filters(with_target)

    summary = {
        "source_rows": len(raw),
        "cohort_rows": len(cohort),
        "exclusions": exclusions,
    }
    return cohort, summary
