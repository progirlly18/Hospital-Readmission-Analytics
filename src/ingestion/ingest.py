from pathlib import Path
from datetime import datetime
import shutil
import pandas as pd


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
STAGING_DIR = PROJECT_ROOT / "data" / "staging"
LOG_DIR = PROJECT_ROOT / "logs" / "ingestion"


# Input files
DIABETIC_DATA = RAW_DIR / "diabetic_data.csv"
IDS_MAPPING = RAW_DIR / "IDS_mapping.csv"


def run_ingestion():
    """Load raw UCI files and create the staging copies."""

    # Create required directories
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Check that the raw files exist
    if not DIABETIC_DATA.exists():
        raise FileNotFoundError(
            f"Missing dataset file: {DIABETIC_DATA}"
        )

    if not IDS_MAPPING.exists():
        raise FileNotFoundError(
            f"Missing mapping file: {IDS_MAPPING}"
        )

    # Read the raw files
    df = pd.read_csv(DIABETIC_DATA)
    mapping_df = pd.read_csv(IDS_MAPPING)

    # Create staging copies
    staging_data = STAGING_DIR / "diabetic_data.csv"
    staging_mapping = STAGING_DIR / "IDS_mapping.csv"

    shutil.copy2(DIABETIC_DATA, staging_data)
    shutil.copy2(IDS_MAPPING, staging_mapping)

    # Record ingestion metadata
    extraction_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_file = LOG_DIR / "ingestion_log.txt"

    with open(log_file, "a", encoding="utf-8") as log:
        log.write(
            f"Extraction date/time: {extraction_time}\n"
            f"Source: UCI Diabetes 130-US Hospitals dataset\n"
            f"Main file: diabetic_data.csv\n"
            f"Main row count: {len(df)}\n"
            f"Main column count: {len(df.columns)}\n"
            f"Mapping file: IDS_mapping.csv\n"
            f"Mapping row count: {len(mapping_df)}\n"
            f"Status: SUCCESS\n"
            f"{'-' * 60}\n"
        )

    print("Ingestion completed successfully.")
    print(f"Main dataset rows: {len(df)}")
    print(f"Main dataset columns: {len(df.columns)}")
    print(f"Staging directory: {STAGING_DIR}")
    print(f"Log file: {log_file}")


if __name__ == "__main__":
    run_ingestion()