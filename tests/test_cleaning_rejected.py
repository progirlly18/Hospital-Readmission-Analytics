"""
Tests for cleaning's rejected-record file (src/transformation/clean.py).

Cleaning runs on a small synthetic staging CSV in a temporary folder; the
project's data/ and logs/ folders are never read or written.
"""

import pandas as pd
import pytest

from src.transformation import clean
from tests.conftest import make_encounter


@pytest.fixture
def cleaning_paths(tmp_path, monkeypatch):
    """Point every cleaning path at a temporary folder."""
    staging = tmp_path / "staging"
    staging.mkdir()
    log_dir = tmp_path / "logs" / "validation"

    monkeypatch.setattr(clean, "INPUT_FILE", staging / "diabetic_data.csv")
    monkeypatch.setattr(clean, "CLEANED_DIR", tmp_path / "cleaned")
    monkeypatch.setattr(
        clean, "OUTPUT_FILE", tmp_path / "cleaned" / "diabetic_data_cleaned.csv"
    )
    monkeypatch.setattr(clean, "LOG_DIR", log_dir)
    monkeypatch.setattr(clean, "REJECTED_FILE", log_dir / "rejected_records.csv")
    return clean.INPUT_FILE, log_dir / "rejected_records.csv"


def write_staging(path, encounter_ids):
    rows = [make_encounter(eid, patient) for patient, eid in enumerate(encounter_ids)]
    staged = pd.DataFrame(rows).drop(columns=["patient_sk"])
    staged.to_csv(path, index=False)
    return list(staged.columns)


def test_zero_rejections_writes_header_only_file(cleaning_paths):
    input_file, rejected_file = cleaning_paths
    staged_columns = write_staging(input_file, [1, 2, 3])

    # A stale file from an earlier run must be replaced, not kept.
    rejected_file.parent.mkdir(parents=True)
    rejected_file.write_text("check,message,count\n", encoding="utf-8")

    clean.clean_data()

    rejected = pd.read_csv(rejected_file)
    assert rejected.empty
    assert list(rejected.columns) == staged_columns
    assert rejected_file.read_text(encoding="utf-8").strip() == ",".join(
        staged_columns
    )


def test_rejected_rows_keep_staging_schema(cleaning_paths):
    input_file, rejected_file = cleaning_paths
    staged_columns = write_staging(input_file, [1, 2, 2, 3])

    clean.clean_data()

    rejected = pd.read_csv(rejected_file)
    assert list(rejected.columns) == staged_columns
    assert rejected["encounter_id"].tolist() == [2]

    cleaned = pd.read_csv(clean.OUTPUT_FILE)
    assert cleaned["encounter_id"].tolist() == [1, 2, 3]
