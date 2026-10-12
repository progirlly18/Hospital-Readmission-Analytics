"""
Regression test: the split fingerprint must not depend on the operating
system. On Windows, pandas writes "\r\n" line endings by default, which
changed the fingerprint even though the split was identical.
"""

import hashlib
import os

import pandas as pd

from src.ml.prepare_data import write_split_assignments


def make_assignments():
    return pd.DataFrame({
        "encounter_id": ["1", "2", "3"],
        "patient_sk": ["p00001", "p00001", "p00002"],
        "split": ["train", "train", "test"],
    })


def test_file_uses_unix_line_endings(tmp_path):
    path = tmp_path / "split_assignments.csv"
    write_split_assignments(make_assignments(), path)
    content = path.read_bytes()
    assert b"\r" not in content
    assert content.count(b"\n") == 4  # header + 3 rows


def test_fingerprint_same_on_windows_and_linux(tmp_path, monkeypatch):
    assignments = make_assignments()

    monkeypatch.setattr(os, "linesep", "\n")       # Linux / macOS
    linux = write_split_assignments(assignments, tmp_path / "linux.csv")

    monkeypatch.setattr(os, "linesep", "\r\n")     # Windows
    windows = write_split_assignments(assignments, tmp_path / "windows.csv")

    assert linux == windows


def test_fingerprint_known_value(tmp_path):
    # Fixed expected hash guards against any change in CSV formatting.
    fingerprint = write_split_assignments(
        make_assignments(), tmp_path / "split_assignments.csv"
    )
    expected_bytes = (
        b"encounter_id,patient_sk,split\n"
        b"1,p00001,train\n"
        b"2,p00001,train\n"
        b"3,p00002,test\n"
    )
    assert fingerprint == hashlib.sha256(expected_bytes).hexdigest()
