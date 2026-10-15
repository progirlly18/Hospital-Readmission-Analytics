from pathlib import Path
import hashlib
import pandas as pd


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STAGING_DIR = PROJECT_ROOT / "data" / "staging"
CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"
LOG_DIR = PROJECT_ROOT / "logs" / "validation"

INPUT_FILE = STAGING_DIR / "diabetic_data.csv"
OUTPUT_FILE = CLEANED_DIR / "diabetic_data_cleaned.csv"
REJECTED_FILE = LOG_DIR / "rejected_records.csv"


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def create_patient_surrogate(patient_nbr):
    """Create a deterministic, patient-safe surrogate key."""
    value = str(patient_nbr)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def clean_question_marks(df):
    """Replace the dataset's special '?' missing-value code."""
    return df.replace("?", pd.NA)


def standardize_age(df):
    """Standardize age values such as [50-60) to 50-60."""
    df["age"] = (
        df["age"]
        .astype("string")
        .str.replace("[", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("-", "-", regex=False)
    )
    return df


def standardize_readmission(df):
    """Standardize the readmission categories."""
    df["readmitted"] = (
        df["readmitted"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    valid_values = {"NO", ">30", "<30"}

    df.loc[~df["readmitted"].isin(valid_values), "readmitted"] = pd.NA

    return df


def standardize_categorical_missing_values(df):
    """
    Replace missing categorical values with 'Unknown'
    where retaining the encounter is appropriate.
    """

    categorical_columns = [
        "race",
        "payer_code",
        "medical_specialty",
        "diag_1",
        "diag_2",
        "diag_3",
    ]

    for column in categorical_columns:
        if column in df.columns:
            df[column] = (
                df[column]
                .astype("string")
                .fillna("Unknown")
                .str.strip()
            )

    return df


# ---------------------------------------------------------
# Main cleaning function
# ---------------------------------------------------------

def clean_data():
    """Clean the staged dataset and create the cleaned layer."""

    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Check input
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Staging dataset not found: {INPUT_FILE}"
        )

    # Read staged data
    df = pd.read_csv(INPUT_FILE)

    original_rows = len(df)

    print(f"Input rows: {original_rows}")

    # -----------------------------------------------------
    # 1. Replace special missing-value codes
    # -----------------------------------------------------

    df = clean_question_marks(df)

    # -----------------------------------------------------
    # 2. Validate required encounter identifiers
    # -----------------------------------------------------

    required_columns = ["encounter_id", "patient_nbr"]

    invalid_id_mask = df[required_columns].isna().any(axis=1)

    rejected_records = df.loc[invalid_id_mask].copy()

    # Remove invalid records from cleaned dataset
    df = df.loc[~invalid_id_mask].copy()

    # -----------------------------------------------------
    # 3. Remove duplicate encounters
    # -----------------------------------------------------

    duplicate_mask = df["encounter_id"].duplicated(keep="first")

    duplicate_records = df.loc[duplicate_mask].copy()

    rejected_records = pd.concat(
        [rejected_records, duplicate_records],
        ignore_index=True
    )

    df = df.loc[~duplicate_mask].copy()

    # -----------------------------------------------------
    # 4. Create patient-safe surrogate key
    # -----------------------------------------------------

    df["patient_sk"] = df["patient_nbr"].apply(
        create_patient_surrogate
    )

    # -----------------------------------------------------
    # 5. Standardize categorical fields
    # -----------------------------------------------------

    df = standardize_categorical_missing_values(df)

    # -----------------------------------------------------
    # 6. Standardize age
    # -----------------------------------------------------

    df = standardize_age(df)

    # -----------------------------------------------------
    # 7. Standardize readmission label
    # -----------------------------------------------------

    df = standardize_readmission(df)

    # -----------------------------------------------------
    # 8. Standardize selected numeric columns
    # -----------------------------------------------------

    numeric_columns = [
        "encounter_id",
        "patient_nbr",
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id",
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # -----------------------------------------------------
    # 9. Save cleaned data
    # -----------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # -----------------------------------------------------
    # 10. Save rejected records
    # -----------------------------------------------------

    # Always written, so the file reflects this run (header only when
    # no records are rejected).
    rejected_records.to_csv(
        REJECTED_FILE,
        index=False
    )

    # -----------------------------------------------------
    # 11. Write validation/cleaning log
    # -----------------------------------------------------

    log_file = LOG_DIR / "cleaning_log.txt"

    with open(log_file, "w", encoding="utf-8") as log:
        log.write(
            f"Input rows: {original_rows}\n"
            f"Output rows: {len(df)}\n"
            f"Rejected rows: {len(rejected_records)}\n"
            f"Duplicate encounters rejected: "
            f"{len(duplicate_records)}\n"
            f"Invalid identifier records rejected: "
            f"{len(rejected_records) - len(duplicate_records)}\n"
            f"Special '?' values converted to missing values: YES\n"
            f"Patient surrogate key created: YES\n"
            f"Readmission categories standardized: YES\n"
            f"Status: SUCCESS\n"
        )

    print("Cleaning completed successfully.")
    print(f"Input rows: {original_rows}")
    print(f"Cleaned rows: {len(df)}")
    print(f"Rejected rows: {len(rejected_records)}")
    print(f"Cleaned file: {OUTPUT_FILE}")
    print(f"Cleaning log: {log_file}")


if __name__ == "__main__":
    clean_data()