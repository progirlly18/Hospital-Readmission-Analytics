"""
Central configuration for the Part 2 modelling pipeline.

Every decision that affects the modelling data lives here so it can be
reviewed in one place and changed without editing the pipeline code.
"""

from pathlib import Path


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Part 1 output used as the modelling source (approved in Phase 1).
CLEANED_FILE = PROJECT_ROOT / "data" / "cleaned" / "diabetic_data_cleaned.csv"

# Generated outputs (gitignored).
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_PREP_DIR = ARTIFACTS_DIR / "data_prep"


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------

RANDOM_SEED = 42


# ---------------------------------------------------------
# Target
# ---------------------------------------------------------
# Positive class (1) = readmitted within 30 days ("<30").
# Negative class (0) = readmitted after 30 days (">30") or not readmitted ("NO").

TARGET_SOURCE_COLUMN = "readmitted"
TARGET_COLUMN = "readmitted_30d"
POSITIVE_LABEL = "<30"
VALID_READMISSION_LABELS = {"<30", ">30", "NO"}


# ---------------------------------------------------------
# Cohort exclusions (IDS_mapping.csv discharge_disposition_id)
# ---------------------------------------------------------
# Patients who died cannot be readmitted; hospice discharges are end-of-life
# care where readmission risk scoring is not meaningful.

EXPIRED_DISPOSITION_IDS = {11, 19, 20, 21}
HOSPICE_DISPOSITION_IDS = {13, 14}
INVALID_GENDER_VALUES = {"Unknown/Invalid"}


# ---------------------------------------------------------
# Identifiers and grouping
# ---------------------------------------------------------

ENCOUNTER_ID_COLUMN = "encounter_id"
GROUP_COLUMN = "patient_sk"  # patient surrogate key from Part 1


# ---------------------------------------------------------
# Columns that must never be used as model features
# ---------------------------------------------------------

FORBIDDEN_FEATURE_COLUMNS = {
    # identifiers
    "encounter_id",
    "patient_nbr",
    "patient_sk",
    # target and target-derived columns
    "readmitted",
    "readmitted_30d",
    "readmission_status",
    "readmitted_within_30_days",
    # excluded by decision (Phase 1 approval)
    "race",          # retained only for subgroup / fairness analysis
    "payer_code",    # insurance proxy, 40% missing
    "weight",        # 97% missing
    # Part 1 patient_summary columns aggregate future encounters (leakage)
    "encounter_count",
    "total_length_of_stay",
    "average_length_of_stay",
    "total_diagnoses",
    "total_medications",
    "prior_admission_count",
}

# Columns kept beside the features for subgroup analysis only.
SUBGROUP_COLUMNS = ["race", "gender", "age"]


# ---------------------------------------------------------
# Model features
# ---------------------------------------------------------
# Assumed prediction time: at discharge. Every feature below is recorded
# before or during the index stay. PREDICTION_TIME_AVAILABILITY documents why.

NUMERIC_FEATURES = [
    "age_years",            # midpoint of the 10-year age band
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "medication_count",
]

# Diabetes medications with enough variation to be informative.
# examide and citoglipton are constant ("No" for every encounter);
# acetohexamide, troglitazone, glimepiride-pioglitazone and
# metformin-rosiglitazone have 3 or fewer non-"No" values and are dropped.
MEDICATION_FEATURES = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "tolazamide",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "metformin-pioglitazone",
]

# All 23 medication columns, used for medication_count (same definition as Part 1).
ALL_MEDICATION_COLUMNS = MEDICATION_FEATURES + [
    "acetohexamide",
    "troglitazone",
    "examide",
    "citoglipton",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
]

CATEGORICAL_FEATURES = [
    "gender",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "medical_specialty",
    "diag_1_group",
    "diag_2_group",
    "diag_3_group",
    "max_glu_serum",
    "A1Cresult",
    "change",
    "diabetesMed",
] + MEDICATION_FEATURES

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

PREDICTION_TIME_AVAILABILITY = {
    "age_years": "Demographic, known at admission",
    "gender": "Demographic, known at admission",
    "admission_type_id": "Recorded at admission",
    "admission_source_id": "Recorded at admission",
    "discharge_disposition_id": "Recorded at discharge (prediction time)",
    "medical_specialty": "Admitting physician specialty, recorded during stay",
    "time_in_hospital": "Known at discharge",
    "num_lab_procedures": "Counted during the stay",
    "num_procedures": "Counted during the stay",
    "num_medications": "Counted during the stay",
    "number_outpatient": "Visits in the year BEFORE the encounter",
    "number_emergency": "Visits in the year BEFORE the encounter",
    "number_inpatient": "Visits in the year BEFORE the encounter",
    "number_diagnoses": "Diagnoses entered for the stay",
    "diag_1_group": "Primary diagnosis of the stay",
    "diag_2_group": "Secondary diagnosis of the stay",
    "diag_3_group": "Additional secondary diagnosis of the stay",
    "max_glu_serum": "Lab result during the stay (or not measured)",
    "A1Cresult": "Lab result during the stay (or not measured)",
    "change": "Diabetes medication change during the stay",
    "diabetesMed": "Diabetes medication prescribed during the stay",
    "medication_count": "Derived from the stay's medication columns",
    **{
        drug: "Medication status during the stay"
        for drug in MEDICATION_FEATURES
    },
}


# ---------------------------------------------------------
# Split
# ---------------------------------------------------------
# Patient-grouped, stratified split (see src/ml/split.py).
# Target proportions train / validation / test. Actual proportions are
# measured and reported, not forced.

SPLIT_FRACTIONS = (0.70, 0.15, 0.15)

# Categories seen fewer times than this in TRAINING data are merged into
# an "infrequent" bucket by the one-hot encoder.
MIN_CATEGORY_FREQUENCY = 50

# Any single feature with a training ROC-AUC above this is flagged as a
# possible leak before any model is trained.
SINGLE_FEATURE_LEAKAGE_AUC = 0.80

# Subgroups smaller than this are suppressed in reports (privacy precaution).
MIN_SUBGROUP_SIZE = 30
