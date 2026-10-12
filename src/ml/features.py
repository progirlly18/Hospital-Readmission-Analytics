"""
Feature engineering for the Part 2 model.

Two separate steps keep preprocessing leakage out:

1. build_feature_frame(): row-by-row transformations that use no
   statistics from the data (safe to apply before splitting).
2. build_preprocessor(): imputation, scaling and one-hot encoding that
   LEARN from data; it is only ever fitted on the training split.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.ml import config


MISSING_CATEGORY = "Missing"
NOT_MEASURED = "None"  # lab test not performed during the stay


# ---------------------------------------------------------
# ICD-9 diagnosis grouping
# ---------------------------------------------------------
# Part 2 uses its own grouping instead of Part 1's
# categorize_diagnosis(). Part 1 removes the decimal point before comparing
# ranges, so codes such as "250.83" become 25083 and fall into "Other"
# (8,522 primary diagnoses are affected). Part 1 is left unchanged.
#
# Groups follow the ICD-9-CM chapters. Diabetes (250.xx) gets its own group
# because every patient in this dataset is diabetic.

ICD9_CHAPTERS = [
    (1, 140, "Infectious"),
    (140, 240, "Neoplasms"),
    (240, 280, "Endocrine/Metabolic"),
    (280, 290, "Blood"),
    (290, 320, "Mental"),
    (320, 390, "Nervous"),
    (390, 460, "Circulatory"),
    (460, 520, "Respiratory"),
    (520, 580, "Digestive"),
    (580, 630, "Genitourinary"),
    (630, 680, "Pregnancy"),
    (680, 710, "Skin"),
    (710, 740, "Musculoskeletal"),
    (740, 760, "Congenital"),
    (760, 780, "Perinatal"),
    (780, 800, "Symptoms/Ill-defined"),
    (800, 1000, "Injury/Poisoning"),
]


def map_icd9_group(code):
    """Map one ICD-9 code (e.g. "428", "250.83", "V57") to a broad group."""
    if code is None or pd.isna(code):
        return MISSING_CATEGORY

    code = str(code).strip().upper()

    if code in ("", "?", "UNKNOWN"):
        return MISSING_CATEGORY
    if code.startswith("V"):
        return "Supplementary (V)"
    if code.startswith("E"):
        return "External cause (E)"

    try:
        value = float(code)  # keeps the decimal point: 250.83 -> 250.83
    except ValueError:
        return "Other"

    if 250 <= value < 251:
        return "Diabetes"

    for low, high, name in ICD9_CHAPTERS:
        if low <= value < high:
            return name

    return "Other"


# ---------------------------------------------------------
# Row-wise feature construction
# ---------------------------------------------------------

def age_band_to_years(age_band):
    """Convert "50-60" (or raw "[50-60)") to the band midpoint, 55."""
    if age_band is None or pd.isna(age_band):
        return float("nan")
    text = str(age_band).strip().strip("[)")
    low, high = text.split("-")
    return (float(low) + float(high)) / 2


def build_feature_frame(df):
    """
    Return a DataFrame containing exactly config.FEATURE_COLUMNS.

    Only row-wise rules are used here, so the result for one encounter never
    depends on any other encounter.
    """
    features = pd.DataFrame(index=df.index)

    features["age_years"] = df["age"].map(age_band_to_years)

    for column in [
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
    ]:
        features[column] = pd.to_numeric(df[column], errors="coerce")

    medication_status = df[config.ALL_MEDICATION_COLUMNS].fillna("No")
    features["medication_count"] = (
        medication_status.ne("No").sum(axis=1).astype(int)
    )

    # Codes are treated as categories, not numbers.
    for column in [
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id",
    ]:
        features[column] = (
            pd.to_numeric(df[column], errors="coerce")
            .astype("Int64")
            .astype("string")
            .fillna(MISSING_CATEGORY)
        )

    features["gender"] = df["gender"].astype("string").fillna(MISSING_CATEGORY)

    features["medical_specialty"] = (
        df["medical_specialty"]
        .astype("string")
        .replace("Unknown", MISSING_CATEGORY)
        .fillna(MISSING_CATEGORY)
    )

    for position in (1, 2, 3):
        features[f"diag_{position}_group"] = (
            df[f"diag_{position}"].map(map_icd9_group)
        )

    # Missing lab result means "test not performed", which is informative.
    for column in ["max_glu_serum", "A1Cresult"]:
        features[column] = df[column].astype("string").fillna(NOT_MEASURED)

    for column in ["change", "diabetesMed"] + config.MEDICATION_FEATURES:
        features[column] = df[column].astype("string").fillna(MISSING_CATEGORY)

    # Plain object dtype for categoricals keeps sklearn behaviour identical
    # across pandas 2.x and 3.x.
    for column in config.CATEGORICAL_FEATURES:
        features[column] = features[column].astype(object)

    return features[config.FEATURE_COLUMNS]


# ---------------------------------------------------------
# Learned preprocessing (fit on training data only)
# ---------------------------------------------------------

def build_preprocessor(scale_numeric=True):
    """
    Build the sklearn preprocessing step.

    - Numeric: median imputation (+ standard scaling for linear models).
    - Categorical: one-hot encoding; categories seen fewer than
      MIN_CATEGORY_FREQUENCY times in training are grouped, and categories
      never seen in training are mapped to that group at prediction time.
    """
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    categorical_steps = [
        (
            "impute",
            SimpleImputer(strategy="constant", fill_value=MISSING_CATEGORY),
        ),
        (
            "encode",
            OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=config.MIN_CATEGORY_FREQUENCY,
                sparse_output=False,
            ),
        ),
    ]

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), config.NUMERIC_FEATURES),
            (
                "categorical",
                Pipeline(categorical_steps),
                config.CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
