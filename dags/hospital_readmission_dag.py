from datetime import datetime

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator


# ---------------------------------------------------------
# Project configuration
# ---------------------------------------------------------

PROJECT_DIR = "/mnt/d/DOWNLOADS/Hospital-Readmission-Analytics"

PYTHON = "/home/vidya/airflow_venv/bin/python"


# ---------------------------------------------------------
# DAG definition
# ---------------------------------------------------------

with DAG(
    dag_id="hospital_readmission_etl",
    start_date=datetime(2026, 8, 29),
    schedule=None,
    catchup=False,
    tags=["hospital", "readmission", "etl", "part1"],
    description="Hospital Readmission Analytics ETL pipeline",
) as dag:

    # Step 1: Ingest raw data into staging
    ingestion = BashOperator(
        task_id="ingestion",
        bash_command=(
            f'cd "{PROJECT_DIR}" && '
            f'"{PYTHON}" src/ingestion/ingest.py'
        ),
    )

    # Step 2: Clean the dataset
    cleaning = BashOperator(
        task_id="cleaning",
        bash_command=(
            f'cd "{PROJECT_DIR}" && '
            f'"{PYTHON}" src/transformation/clean.py'
        ),
    )

    # Step 3: Engineer analytical features
    feature_engineering = BashOperator(
        task_id="feature_engineering",
        bash_command=(
            f'cd "{PROJECT_DIR}" && '
            f'"{PYTHON}" src/transformation/feature_engineering.py'
        ),
    )

    # Step 4: Validate transformed data
    validation = BashOperator(
        task_id="validation",
        bash_command=(
            f'cd "{PROJECT_DIR}" && '
            f'"{PYTHON}" src/validation/validate.py'
        ),
    )

    # Step 5: Load analytical data into PostgreSQL
    loading = BashOperator(
        task_id="postgresql_loading",
        bash_command=(
            f'cd "{PROJECT_DIR}" && '
            f'"{PYTHON}" src/loading/load_to_postgres.py'
        ),
    )


    # -----------------------------------------------------
    # Task dependency
    # -----------------------------------------------------

    ingestion >> cleaning >> feature_engineering >> validation >> loading