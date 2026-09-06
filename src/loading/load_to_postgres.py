from pathlib import Path
import os

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv



# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ANALYTICAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "analytical"
    / "diabetic_data_features.csv"
)

MAPPING_FILE = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "IDS_mapping.csv"
)


# ---------------------------------------------------------
# Load database configuration
# ---------------------------------------------------------

load_dotenv(PROJECT_ROOT / ".env")

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError(
        "DATABASE_URL is not set in the .env file."
    )
def get_postgres_connection():
    """Create a PostgreSQL connection using psycopg2."""

    postgres_url = DATABASE_URL.replace(
        "postgresql+psycopg2://",
        "postgresql://",
        1
    )

    return psycopg2.connect(postgres_url)

# ---------------------------------------------------------
# Read mapping file
# ---------------------------------------------------------

def load_mapping_file():
    """
    Read IDS_mapping.csv and create dictionaries for:
    admission type, discharge disposition,
    and admission source.
    """

    mapping = pd.read_csv(
        MAPPING_FILE,
        dtype=str
    )

    admission_type_map = {}
    discharge_map = {}
    admission_source_map = {}

    current_section = None

    for _, row in mapping.iterrows():

        first_value = str(row.iloc[0]).strip()
        second_value = row.iloc[1]

        # Detect section headers
        if first_value == "admission_type_id":
            current_section = "admission_type"
            continue

        if first_value == "discharge_disposition_id":
            current_section = "discharge"
            continue

        if first_value == "admission_source_id":
            current_section = "admission_source"
            continue

        # Ignore blank separator rows
        if first_value in ("nan", ""):
            continue

        # Ignore rows without descriptions
        if pd.isna(second_value):
            continue

        if current_section == "admission_type":
            admission_type_map[first_value] = str(second_value).strip()

        elif current_section == "discharge":
            discharge_map[first_value] = str(second_value).strip()

        elif current_section == "admission_source":
            admission_source_map[first_value] = str(second_value).strip()

    return (
        admission_type_map,
        discharge_map,
        admission_source_map
    )


# ---------------------------------------------------------
# Main loading function
# ---------------------------------------------------------

def load_data():
    """Load analytical data into PostgreSQL."""

    if not ANALYTICAL_FILE.exists():
        raise FileNotFoundError(
            f"Analytical file not found: {ANALYTICAL_FILE}"
        )

    if not MAPPING_FILE.exists():
        raise FileNotFoundError(
            f"Mapping file not found: {MAPPING_FILE}"
        )

    print("Reading analytical dataset...")

    df = pd.read_csv(ANALYTICAL_FILE)

    print(f"Analytical rows: {len(df)}")

    # -----------------------------------------------------
    # Load mapping dictionaries
    # -----------------------------------------------------

    (
        admission_type_map,
        discharge_map,
        admission_source_map
    ) = load_mapping_file()

    # -----------------------------------------------------
    # Add descriptive admission fields
    # -----------------------------------------------------

    df["admission_type"] = (
        df["admission_type_id"]
        .astype("Int64")
        .astype("string")
        .map(admission_type_map)
        .fillna("Unknown")
    )

    df["discharge_disposition"] = (
        df["discharge_disposition_id"]
        .astype("Int64")
        .astype("string")
        .map(discharge_map)
        .fillna("Unknown")
    )

    df["admission_source"] = (
        df["admission_source_id"]
        .astype("Int64")
        .astype("string")
        .map(admission_source_map)
        .fillna("Unknown")
    )

    

    # -----------------------------------------------------
    # 1. Patient Summary
    # -----------------------------------------------------

    patient_summary = (
        df.groupby("patient_sk")
        .agg(
            encounter_count=("encounter_id", "count"),
            total_length_of_stay=(
                "length_of_stay_days",
                "sum"
            ),
            average_length_of_stay=(
                "length_of_stay_days",
                "mean"
            ),
            total_diagnoses=(
                "diagnosis_count",
                "sum"
            ),
            total_medications=(
                "medication_count",
                "sum"
            ),
            prior_admission_count=(
                "prior_admission",
                "sum"
            ),
        )
        .reset_index()
    )

    # -----------------------------------------------------
    # 2. Admissions
    # -----------------------------------------------------

    admissions_columns = [
        "encounter_id",
        "patient_sk",
        "age",
        "gender",
        "race",
        "admission_type_id",
        "admission_type",
        "discharge_disposition_id",
        "discharge_disposition",
        "admission_source_id",
        "admission_source",
        "length_of_stay_days",
        "prior_admission",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
    ]

    admissions = df[admissions_columns].copy()

    # -----------------------------------------------------
    # 3. Diagnoses
    # -----------------------------------------------------

    diagnosis_records = []

    diagnosis_columns = [
        "diag_1",
        "diag_2",
        "diag_3"
    ]

    for position, column in enumerate(
        diagnosis_columns,
        start=1
    ):
        temp = df[
            [
                "encounter_id",
                "patient_sk",
                column,
                "primary_diagnosis_category"
            ]
        ].copy()

        temp = temp.rename(
            columns={
                column: "diagnosis_code",
                "primary_diagnosis_category":
                    "diagnosis_category"
            }
        )

        temp["diagnosis_position"] = position

        temp = temp[
            [
                "encounter_id",
                "patient_sk",
                "diagnosis_position",
                "diagnosis_code",
                "diagnosis_category"
            ]
        ]

        # Remove unknown diagnosis records
        temp = temp[
            temp["diagnosis_code"].notna()
            & (temp["diagnosis_code"] != "Unknown")
        ]

        diagnosis_records.append(temp)

    diagnoses = pd.concat(
        diagnosis_records,
        ignore_index=True
    )

    # -----------------------------------------------------
    # 4. Readmissions
    # -----------------------------------------------------

    readmissions = df[
        [
            "encounter_id",
            "patient_sk",
            "readmitted"
        ]
    ].copy()

    readmissions = readmissions.rename(
        columns={
            "readmitted": "readmission_status"
        }
    )

    readmissions["readmitted_within_30_days"] = (
        readmissions["readmission_status"]
        .eq("<30")
        .astype(int)
    )

    # -----------------------------------------------------
    # PostgreSQL bulk loading
    # -----------------------------------------------------

    def insert_dataframe(cursor, table_name, dataframe, page_size=5000):
        """Insert a DataFrame into an existing PostgreSQL table."""

        columns = list(dataframe.columns)

        column_sql = ", ".join(
            f'"{column}"'
            for column in columns
        )

        insert_sql = f'''
            INSERT INTO "{table_name}" ({column_sql})
            VALUES %s
        '''

        records = [
            tuple(
                None if pd.isna(value) else value
                for value in row
            )
            for row in dataframe.itertuples(
                index=False,
                name=None
            )
        ]

        execute_values(
            cursor,
            insert_sql,
            records,
            page_size=page_size
        )


    print("Connecting to PostgreSQL...")

    connection = get_postgres_connection()

    try:

        cursor = connection.cursor()

        # -------------------------------------------------
        # Clear previous pipeline data
        # -------------------------------------------------

        print("Clearing existing analytical data...")

        cursor.execute("""
            TRUNCATE TABLE
                diagnoses,
                readmissions,
                admissions,
                patient_summary
            RESTART IDENTITY CASCADE
        """)

        # -------------------------------------------------
        # Load patient summary
        # -------------------------------------------------

        print("Loading patient_summary...")

        insert_dataframe(
            cursor,
            "patient_summary",
            patient_summary
        )

        # -------------------------------------------------
        # Load admissions
        # -------------------------------------------------

        print("Loading admissions...")

        insert_dataframe(
            cursor,
            "admissions",
            admissions
        )

        # -------------------------------------------------
        # Load diagnoses
        # -------------------------------------------------

        print("Loading diagnoses...")

        insert_dataframe(
            cursor,
            "diagnoses",
            diagnoses
        )

        # -------------------------------------------------
        # Load readmissions
        # -------------------------------------------------

        print("Loading readmissions...")

        insert_dataframe(
            cursor,
            "readmissions",
            readmissions
        )

        # -------------------------------------------------
        # Commit transaction
        # -------------------------------------------------

        connection.commit()

        print("\nDatabase loading completed successfully.")
        print(f"Patients loaded: {len(patient_summary)}")
        print(f"Admissions loaded: {len(admissions)}")
        print(f"Diagnoses loaded: {len(diagnoses)}")
        print(f"Readmissions loaded: {len(readmissions)}")

    except Exception:

        connection.rollback()

        print("\nDatabase loading failed.")
        print("All database changes have been rolled back.")

        raise

    finally:

        cursor.close()
        connection.close()
if __name__ == "__main__":
    load_data()