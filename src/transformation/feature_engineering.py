from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"
ANALYTICAL_DIR = PROJECT_ROOT / "data" / "analytical"

INPUT_FILE = CLEANED_DIR / "diabetic_data_cleaned.csv"
OUTPUT_FILE = ANALYTICAL_DIR / "diabetic_data_features.csv"


# ---------------------------------------------------------
# Diagnosis categorization
# ---------------------------------------------------------

def categorize_diagnosis(code):
    """
    Categorize an ICD-9 diagnosis code into a broad
    analytical category.

    This is intended for dashboard-level analysis,
    not clinical diagnosis.
    """

    if pd.isna(code) or str(code).strip() == "" or code == "Unknown":
        return "Unknown"

    code = str(code).strip()

    # V and E codes
    if code.startswith("V"):
        return "Supplementary / Other"
    if code.startswith("E"):
        return "External Cause"

    # Remove decimal point for numeric comparison
    numeric_code = code.replace(".", "")

    try:
        value = float(numeric_code)
    except ValueError:
        return "Other"

    if 1 <= value < 140:
        return "Infectious Diseases"
    elif 140 <= value < 240:
        return "Neoplasms"
    elif 240 <= value < 280:
        return "Endocrine / Metabolic"
    elif 280 <= value < 290:
        return "Blood / Immune"
    elif 290 <= value < 320:
        return "Mental Disorders"
    elif 320 <= value < 390:
        return "Nervous System"
    elif 390 <= value < 460:
        return "Circulatory"
    elif 460 <= value < 520:
        return "Respiratory"
    elif 520 <= value < 580:
        return "Digestive"
    elif 580 <= value < 630:
        return "Genitourinary"
    elif 630 <= value < 680:
        return "Pregnancy-related"
    elif 680 <= value < 710:
        return "Skin / Subcutaneous"
    elif 710 <= value < 740:
        return "Musculoskeletal"
    elif 740 <= value < 760:
        return "Congenital"
    elif 760 <= value < 780:
        return "Perinatal"
    elif 780 <= value < 800:
        return "Symptoms / Signs"
    elif 800 <= value < 1000:
        return "Injury / Poisoning"
    else:
        return "Other"


# ---------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------

def engineer_features(df):
    """Create analytical features required for Part 1."""

    # -----------------------------------------------------
    # 1. Length of stay
    # -----------------------------------------------------

    df["length_of_stay_days"] = pd.to_numeric(
        df["time_in_hospital"],
        errors="coerce"
    )

    # -----------------------------------------------------
    # 2. Prior admission
    # -----------------------------------------------------
    #
    # number_inpatient represents prior inpatient visits
    # in the available dataset history.
    #
    # > 0 means the patient had at least one prior
    # inpatient visit.
    # -----------------------------------------------------

    df["prior_admission"] = (
        pd.to_numeric(
            df["number_inpatient"],
            errors="coerce"
        )
        .fillna(0)
        .gt(0)
        .astype(int)
    )

    # -----------------------------------------------------
    # 3. Diagnosis count
    # -----------------------------------------------------
    #
    # Count actual diagnosis fields among diag_1,
    # diag_2 and diag_3.
    # -----------------------------------------------------

    diagnosis_columns = [
        "diag_1",
        "diag_2",
        "diag_3"
    ]

    df["diagnosis_count"] = (
        df[diagnosis_columns]
        .replace("Unknown", pd.NA)
        .notna()
        .sum(axis=1)
    )

    # -----------------------------------------------------
    # 4. Medication count
    # -----------------------------------------------------
    #
    # Medication columns contain values such as:
    # No, Steady, Up, Down.
    #
    # We count medications with a recorded status other
    # than "No".
    # -----------------------------------------------------

    medication_columns = [
        "metformin",
        "repaglinide",
        "nateglinide",
        "chlorpropamide",
        "glimepiride",
        "acetohexamide",
        "glipizide",
        "glyburide",
        "tolbutamide",
        "pioglitazone",
        "rosiglitazone",
        "acarbose",
        "miglitol",
        "troglitazone",
        "tolazamide",
        "examide",
        "citoglipton",
        "insulin",
        "glyburide-metformin",
        "glipizide-metformin",
        "glimepiride-pioglitazone",
        "metformin-rosiglitazone",
        "metformin-pioglitazone",
    ]

    medication_status = df[medication_columns].fillna("No")

    df["medication_count"] = (
        medication_status
        .ne("No")
        .sum(axis=1)
    )

    # -----------------------------------------------------
    # 5. Primary diagnosis category
    # -----------------------------------------------------

    df["primary_diagnosis_category"] = (
        df["diag_1"].apply(categorize_diagnosis)
    )

    return df


# ---------------------------------------------------------
# Main function
# ---------------------------------------------------------

def run_feature_engineering():
    """Read cleaned data, engineer features and save output."""

    ANALYTICAL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {INPUT_FILE}"
        )

    # Read cleaned data
    df = pd.read_csv(INPUT_FILE)

    original_rows = len(df)

    print(f"Input rows: {original_rows}")

    # Create features
    df = engineer_features(df)

    # Save analytical feature dataset
    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("Feature engineering completed successfully.")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Output file: {OUTPUT_FILE}")

    print("\nEngineered columns:")
    print(" - length_of_stay_days")
    print(" - prior_admission")
    print(" - diagnosis_count")
    print(" - medication_count")
    print(" - primary_diagnosis_category")


if __name__ == "__main__":
    run_feature_engineering()