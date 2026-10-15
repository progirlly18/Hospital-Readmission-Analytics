"""
Tests for the Part 1 validation step (src/validation/validate.py).

All tests use small synthetic CSVs in a temporary folder. They never read
or write the project's data/ or logs/ folders and never touch PostgreSQL.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.validation import validate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DAG_FILE = PROJECT_ROOT / "dags" / "hospital_readmission_dag.py"


def write_datasets(folder, cleaned_overrides=None, analytical_rows=None):
    """Write a tiny cleaned/analytical pair that passes every check."""
    cleaned = pd.DataFrame({
        "encounter_id": [1, 2, 3],
        "patient_sk": ["a1", "b2", "c3"],
        "readmitted": ["NO", "<30", ">30"],
        "time_in_hospital": [1, 5, 14],
    })
    for column, values in (cleaned_overrides or {}).items():
        cleaned[column] = values

    analytical = pd.DataFrame({
        "encounter_id": [1, 2, 3],
        "prior_admission": [0, 1, 0],
        "diagnosis_count": [3, 2, 0],
        "medication_count": [0, 4, 23],
    })
    if analytical_rows is not None:
        analytical = analytical.head(analytical_rows)

    cleaned_file = folder / "cleaned.csv"
    analytical_file = folder / "analytical.csv"
    cleaned.to_csv(cleaned_file, index=False)
    analytical.to_csv(analytical_file, index=False)
    return cleaned_file, analytical_file


def run_validate_process(cleaned_file, analytical_file, log_dir):
    """Run validation in a separate Python process and return its exit code."""
    code = (
        "import sys\n"
        "from src.validation import validate\n"
        f"sys.exit(validate.main({str(cleaned_file)!r}, "
        f"{str(analytical_file)!r}, {str(log_dir)!r}))\n"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )


# ---------------------------------------------------------
# Exit codes
# ---------------------------------------------------------

def test_passing_validation_exits_zero(tmp_path):
    cleaned, analytical = write_datasets(tmp_path)
    result = run_validate_process(cleaned, analytical, tmp_path / "logs")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "VALIDATION STATUS: PASSED" in result.stdout


@pytest.mark.parametrize("overrides, analytical_rows, expected_check", [
    ({"encounter_id": [1, 1, 3]}, None, "Duplicate encounter_id"),
    ({"patient_sk": ["a1", None, "c3"]}, None, "Missing patient_sk"),
    ({"readmitted": ["NO", "YES", ">30"]}, None, "Invalid readmission category"),
    ({"time_in_hospital": [0, 5, 14]}, None, "Invalid length of stay"),
    ({"patient_sk": ["a1", "?", "c3"]}, None, "Special missing values"),
    ({}, 2, "Row count mismatch"),
])
def test_failed_validation_exits_nonzero(tmp_path, overrides, analytical_rows,
                                         expected_check):
    cleaned, analytical = write_datasets(tmp_path, overrides, analytical_rows)
    result = run_validate_process(cleaned, analytical, tmp_path / "logs")

    assert result.returncode != 0
    assert "VALIDATION STATUS: FAILED" in result.stdout
    assert expected_check in result.stdout


def test_main_returns_status_code(tmp_path):
    cleaned, analytical = write_datasets(tmp_path)
    assert validate.main(cleaned, analytical, tmp_path / "ok") == 0

    cleaned, analytical = write_datasets(tmp_path, {"encounter_id": [1, 1, 3]})
    assert validate.main(cleaned, analytical, tmp_path / "bad") == 1


def test_script_entry_point_exits_with_main_status():
    """Running validate.py as a script passes main()'s status to sys.exit."""
    tree = ast.parse((PROJECT_ROOT / "src/validation/validate.py").read_text())
    main_guard = [
        node for node in tree.body
        if isinstance(node, ast.If) and "__main__" in ast.unparse(node.test)
    ]
    assert len(main_guard) == 1
    assert ast.unparse(main_guard[0].body[0]) == "sys.exit(main())"


# ---------------------------------------------------------
# Output files
# ---------------------------------------------------------

def test_issues_written_to_separate_summary_file(tmp_path):
    cleaned, analytical = write_datasets(
        tmp_path, {"encounter_id": [1, 1, 3], "time_in_hospital": [0, 5, 14]}
    )
    log_dir = tmp_path / "logs"
    validate.main(cleaned, analytical, log_dir)

    issues = pd.read_csv(log_dir / "validation_issues.csv")
    assert list(issues.columns) == ["check", "message", "count"]
    assert set(issues["check"]) == {
        "Duplicate encounter_id", "Invalid length of stay"
    }
    assert not (log_dir / "rejected_records.csv").exists()
    assert "VALIDATION STATUS: FAILED" in (
        log_dir / "validation_log.txt"
    ).read_text(encoding="utf-8")


def test_no_issues_writes_empty_summary(tmp_path):
    log_dir = tmp_path / "logs"

    # A failed run first, so a stale summary would be visible.
    cleaned, analytical = write_datasets(tmp_path, {"encounter_id": [1, 1, 3]})
    validate.main(cleaned, analytical, log_dir)
    assert len(pd.read_csv(log_dir / "validation_issues.csv")) == 1

    cleaned, analytical = write_datasets(tmp_path)
    assert validate.main(cleaned, analytical, log_dir) == 0

    issues = pd.read_csv(log_dir / "validation_issues.csv")
    assert list(issues.columns) == ["check", "message", "count"]
    assert issues.empty
    assert "VALIDATION STATUS: PASSED" in (
        log_dir / "validation_log.txt"
    ).read_text(encoding="utf-8")


@pytest.mark.parametrize("overrides", [{}, {"encounter_id": [1, 1, 3]}])
def test_cleaning_rejected_records_unchanged(tmp_path, overrides):
    """Validation must not touch cleaning's rejected_records.csv."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    rejected = log_dir / "rejected_records.csv"
    rejected.write_text(
        "encounter_id,patient_nbr,readmitted\n"
        "12522,48330783,NO\n"
        "12522,48330783,NO\n",
        encoding="utf-8",
    )
    before = rejected.read_bytes()

    cleaned, analytical = write_datasets(tmp_path, overrides)
    validate.main(cleaned, analytical, log_dir)

    assert rejected.read_bytes() == before


def test_cleaning_and_validation_use_different_files():
    from src.transformation import clean

    assert clean.REJECTED_FILE.name == "rejected_records.csv"
    assert validate.VALIDATION_ISSUES_NAME != clean.REJECTED_FILE.name
    assert validate.LOG_DIR == clean.LOG_DIR


# ---------------------------------------------------------
# Airflow chain
# ---------------------------------------------------------

def read_dag():
    """Return the DAG's task commands and dependency chain without Airflow."""
    tree = ast.parse(DAG_FILE.read_text(encoding="utf-8"))
    tasks, chain = {}, []

    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and ast.unparse(node.value.func) == "BashOperator"):
            keywords = {k.arg: k.value for k in node.value.keywords}
            tasks[node.targets[0].id] = {
                "command": ast.unparse(keywords["bash_command"]),
                "trigger_rule": keywords.get("trigger_rule"),
            }
        if (isinstance(node, ast.Expr)
                and isinstance(node.value, ast.BinOp)
                and isinstance(node.value.op, ast.RShift)):
            chain = [part.strip() for part in ast.unparse(node.value).split(">>")]

    return tasks, chain


def test_dag_runs_loading_only_after_validation():
    tasks, chain = read_dag()

    assert chain == [
        "ingestion", "cleaning", "feature_engineering", "validation", "loading"
    ]
    assert "src/validation/validate.py" in tasks["validation"]["command"]
    assert "src/loading/load_to_postgres.py" in tasks["loading"]["command"]
    # No task overrides Airflow's default all_success rule, so a failed
    # validation task leaves loading as upstream_failed.
    assert all(task["trigger_rule"] is None for task in tasks.values())


def test_failed_validation_stops_simulated_dag_before_loading(tmp_path):
    """
    Mimic BashOperator + all_success: each task runs only if the previous
    one exited 0. Validation runs for real on bad data; loading is a stub
    that records whether it was reached, so no database is touched.
    """
    _, chain = read_dag()
    cleaned, analytical = write_datasets(tmp_path, {"encounter_id": [1, 1, 3]})
    reached = []

    def run_task(name):
        reached.append(name)
        if name == "validation":
            return run_validate_process(
                cleaned, analytical, tmp_path / "logs"
            ).returncode
        return 0

    for task in chain:
        if run_task(task) != 0:
            break

    assert reached == [
        "ingestion", "cleaning", "feature_engineering", "validation"
    ]
    assert "loading" not in reached
