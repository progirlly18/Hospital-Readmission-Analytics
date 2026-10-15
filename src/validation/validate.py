import sys
import pandas as pd
from pathlib import Path
from datetime import datetime


# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLEANED_FILE = PROJECT_ROOT / "data" / "cleaned" / "diabetic_data_cleaned.csv"
ANALYTICAL_FILE = PROJECT_ROOT / "data" / "analytical" / "diabetic_data_features.csv"

LOG_DIR = PROJECT_ROOT / "logs" / "validation"

# Cleaning writes its rejected rows to logs/validation/rejected_records.csv.
# Validation writes its per-check summary to a separate file so it never
# overwrites the cleaning output.
VALIDATION_LOG_NAME = "validation_log.txt"
VALIDATION_ISSUES_NAME = "validation_issues.csv"

ISSUE_COLUMNS = ["check", "message", "count"]


# ---------------------------------------------------------
# Data-quality checks
# ---------------------------------------------------------

def run_checks(cleaned_df, analytical_df):
    """Run the configured checks and return a list of detected issues."""

    errors = []

    def add_error(check_name, message, count=1):
        errors.append({
            "check": check_name,
            "message": message,
            "count": count
        })

    # 1. Required ID checks

    if cleaned_df["encounter_id"].isna().any():
        count = cleaned_df["encounter_id"].isna().sum()
        add_error(
            "Missing encounter_id",
            "Encounter ID is missing.",
            count
        )

    if cleaned_df["patient_sk"].isna().any():
        count = cleaned_df["patient_sk"].isna().sum()
        add_error(
            "Missing patient_sk",
            "Patient surrogate key is missing.",
            count
        )

    # 2. Duplicate encounter check

    duplicate_count = cleaned_df["encounter_id"].duplicated().sum()

    if duplicate_count > 0:
        add_error(
            "Duplicate encounter_id",
            "Duplicate encounter IDs detected.",
            duplicate_count
        )

    # 3. Special missing-value check

    question_mark_count = (
        cleaned_df.astype(str) == "?"
    ).sum().sum()

    if question_mark_count > 0:
        add_error(
            "Special missing values",
            "The '?' missing-value code is still present.",
            question_mark_count
        )

    # 4. Readmission category check

    valid_readmission = {"NO", "<30", ">30"}

    invalid_readmission = ~cleaned_df["readmitted"].isin(valid_readmission)

    if invalid_readmission.any():
        count = invalid_readmission.sum()
        add_error(
            "Invalid readmission category",
            "Readmission contains values outside NO, <30, >30.",
            count
        )

    # 5. Length-of-stay validation

    invalid_los = ~cleaned_df["time_in_hospital"].between(1, 14)

    if invalid_los.any():
        count = invalid_los.sum()
        add_error(
            "Invalid length of stay",
            "time_in_hospital must be between 1 and 14 days.",
            count
        )

    # 6. Prior admission validation

    if "prior_admission" in analytical_df.columns:

        invalid_prior = ~analytical_df["prior_admission"].isin([0, 1])

        if invalid_prior.any():
            count = invalid_prior.sum()
            add_error(
                "Invalid prior_admission",
                "prior_admission must contain only 0 or 1.",
                count
            )

    # 7. Diagnosis count validation

    if "diagnosis_count" in analytical_df.columns:

        invalid_diagnosis = ~analytical_df["diagnosis_count"].between(0, 3)

        if invalid_diagnosis.any():
            count = invalid_diagnosis.sum()
            add_error(
                "Invalid diagnosis_count",
                "diagnosis_count must be between 0 and 3.",
                count
            )

    # 8. Medication count validation

    if "medication_count" in analytical_df.columns:

        invalid_medication = ~analytical_df["medication_count"].between(0, 23)

        if invalid_medication.any():
            count = invalid_medication.sum()
            add_error(
                "Invalid medication_count",
                "medication_count must be between 0 and 23.",
                count
            )

    # 9. Cleaned vs analytical row count

    if len(cleaned_df) != len(analytical_df):
        add_error(
            "Row count mismatch",
            "Cleaned and analytical datasets have different row counts.",
            abs(len(cleaned_df) - len(analytical_df))
        )

    return errors


# ---------------------------------------------------------
# Validation result files
# ---------------------------------------------------------

def write_results(errors, cleaned_rows, analytical_rows, log_dir):
    """Write the validation log and the per-check issue summary."""

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    validation_log = log_dir / VALIDATION_LOG_NAME
    issues_file = log_dir / VALIDATION_ISSUES_NAME

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(validation_log, "w", encoding="utf-8") as log:

        log.write("HOSPITAL READMISSION DATA VALIDATION\n")
        log.write("=" * 50 + "\n")
        log.write(f"Validation time: {timestamp}\n")
        log.write(f"Cleaned rows: {cleaned_rows}\n")
        log.write(f"Analytical rows: {analytical_rows}\n")
        log.write("\n")

        if not errors:
            log.write("VALIDATION STATUS: PASSED\n")
            log.write("All configured data-quality checks passed.\n")
        else:
            log.write("VALIDATION STATUS: FAILED\n")
            log.write("\nDetected issues:\n")

            for error in errors:
                log.write(
                    f"- {error['check']}: "
                    f"{error['message']} "
                    f"Count={error['count']}\n"
                )

    # An empty summary (header only) is written when no issues are found,
    # so a stale summary from an earlier failed run never remains.
    pd.DataFrame(errors, columns=ISSUE_COLUMNS).to_csv(
        issues_file,
        index=False
    )

    return validation_log, issues_file


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main(cleaned_file=CLEANED_FILE,
         analytical_file=ANALYTICAL_FILE,
         log_dir=LOG_DIR):
    """
    Validate the cleaned and analytical datasets.

    Returns 0 when every check passes and 1 when any check fails, so the
    Airflow validation task fails and the loading task does not run.
    """

    cleaned_df = pd.read_csv(cleaned_file)
    analytical_df = pd.read_csv(analytical_file)

    errors = run_checks(cleaned_df, analytical_df)

    validation_log, issues_file = write_results(
        errors,
        len(cleaned_df),
        len(analytical_df),
        log_dir
    )

    print("Validation completed.")
    print(f"Cleaned rows: {len(cleaned_df)}")
    print(f"Analytical rows: {len(analytical_df)}")

    if not errors:
        print("VALIDATION STATUS: PASSED")
    else:
        print("VALIDATION STATUS: FAILED")
        print(f"Issues detected: {len(errors)}")

        for error in errors:
            print(
                f"- {error['check']}: "
                f"{error['message']} "
                f"Count={error['count']}"
            )

    print(f"Validation log: {validation_log}")
    print(f"Validation issues: {issues_file}")

    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
