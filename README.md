# Hospital Readmission Analytics and Prediction

## Part 1 – Data Engineering Pipeline

### Project Overview

This project develops a reproducible healthcare data engineering pipeline for analysing hospital readmissions using the publicly available Diabetes 130-US Hospitals dataset.

The project focuses on data ingestion, cleaning, transformation, validation, PostgreSQL storage, analytical data preparation, workflow orchestration using Apache Airflow, and an interactive Streamlit dashboard.

The dataset is de-identified public healthcare data and does not contain personally identifiable patient information used by this project.

---

## Dataset

### Source

Diabetes 130-US Hospitals for Years 1999–2008

The dataset contains hospital encounter information for patients with diabetes, including admission details, diagnoses, procedures, medications, and readmission outcomes.

### Access instructions

- Publisher: UCI Machine Learning Repository, dataset 296, "Diabetes 130-US
  Hospitals for Years 1999–2008" (Strack et al., 2014), licensed CC BY 4.0.
- Page: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
- Download the zip from that page (free, no account needed) and copy
  `diabetic_data.csv` and `IDS_mapping.csv` into `data/raw/`.
- The repository already contains these two files in `data/raw/`, so a fresh
  clone can run the pipeline without downloading anything. They are the
  unmodified public files; the pipeline never edits them.

### Main Files

- `diabetic_data.csv` – primary hospital encounter dataset (101,766 encounters, 50 columns)
- `IDS_mapping.csv` – identifier/code mapping information

---

## Project Architecture

The implemented pipeline follows this flow:

Public Dataset
→ Raw Data
→ Python/Pandas ETL
→ Cleaning
→ Feature Engineering
→ Validation
→ PostgreSQL Loading
→ Streamlit Dashboard

Apache Airflow is used to orchestrate the ETL workflow. Its five tasks run in
order (ingestion → cleaning → feature engineering → validation →
PostgreSQL loading), and a failed task stops every task after it.

---

## Project Structure

```text
Hospital-Readmission-Analytics/
├── data/
│   ├── raw/                 original UCI files (unchanged)
│   ├── staging/             copies made by ingestion
│   ├── cleaned/             cleaned encounters (also the Part 2 modelling source)
│   └── analytical/          engineered features loaded into PostgreSQL
├── dags/
│   └── hospital_readmission_dag.py
├── src/
│   ├── ingestion/  transformation/  validation/  loading/     Part 1 ETL
│   ├── ml/                  Part 2 cohort, features, split, training, MLflow
│   ├── api/                 Part 2 FastAPI service and model bundle
│   └── monitoring/          Part 2 drift, input quality, retraining criteria
├── sql/schema.sql
├── dashboard/
│   ├── app.py               Part 1 dashboard
│   └── pages/               Part 2 prediction page
├── tests/                   Part 2 tests
├── docs/
│   ├── architecture.png, Schema_Diagram.png, data_dictionary.xlsx   Part 1
│   ├── part2/               Part 2 phase docs, model card, lifecycle, bundle
│   └── report/              project report (PDF and HTML source)
├── screenshots/             Part 1 evidence (01–07) and part2/ evidence
├── Dockerfile, requirements*.txt, pytest.ini
└── README.md
```

Generated locally and not committed (see `.gitignore`): `.env` (database
credentials), `logs/` (ingestion, cleaning, validation and rejected-record
logs, written on every run), `artifacts/`, `mlflow.db`, `mlartifacts/`,
`model_bundle/` and `dist/`.

---

## ETL Pipeline

### 1. Ingestion

The ingestion stage reads the source dataset using Python and Pandas and maintains the raw data separately from processed data.

### 2. Cleaning and Transformation

The cleaning stage handles:

* Special missing-value codes
* Standardization of categorical fields
* Diagnosis-related standardization
* Admission and discharge field standardization
* Medication-related fields
* Duplicate records
* Invalid encounters

### 3. Feature Engineering

The pipeline creates analytical features including:

* Length of stay
* Prior admission
* Diagnosis count
* Medication count

### 4. Validation

Data-quality checks are performed before loading the curated data into PostgreSQL.

`src/validation/validate.py` runs nine checks on the cleaned and analytical files:

1. Required identifiers (`encounter_id`, `patient_sk`) are present
2. No duplicate `encounter_id`
3. No remaining `?` missing-value codes
4. `readmitted` is one of `NO`, `<30`, `>30`
5. `time_in_hospital` is between 1 and 14 days
6. `prior_admission` is 0 or 1
7. `diagnosis_count` is between 0 and 3
8. `medication_count` is between 0 and 23
9. Cleaned and analytical files have the same row count

If any check fails, `validate.py` exits with code 1, so Airflow marks the
`validation` task failed and does not run `postgresql_loading`. A clean run
exits with code 0.

Cleaning and validation write separate outputs:

* `logs/validation/rejected_records.csv`: the rows removed by cleaning
  (invalid identifiers and duplicate encounters). Written only by `clean.py`;
  header-only when nothing is rejected.
* `logs/validation/validation_issues.csv`: one row per failed check
  (`check`, `message`, `count`). Written only by `validate.py`; header-only
  when every check passes.

Each run also writes `logs/ingestion/ingestion_log.txt`, `logs/validation/cleaning_log.txt` and `logs/validation/validation_log.txt` (extraction time, source files, status and row counts). On the current data the validation status is PASSED with 0 rejected records.

### 5. PostgreSQL Loading

The validated data is loaded into PostgreSQL by `src/loading/load_to_postgres.py`.
This step adds the admission type, discharge disposition and admission source
descriptions from `IDS_mapping.csv`, unpivots `diag_1`–`diag_3` into one
`diagnoses` row per encounter diagnosis, and reloads all four tables with
`TRUNCATE` and insert inside one transaction.

The database contains the following analytical tables:

* `patient_summary`
* `admissions`
* `diagnoses`
* `readmissions`

---

## Database Results

Current populated database (loaded on 6 September 2026; counts re-confirmed
with a read-only query on 15 October 2026, which also confirmed that the live
columns, keys and constraints match `sql/schema.sql`):

| Table             | Records |
| ----------------- | ------: |
| `patient_summary` |  71,518 |
| `admissions`      | 101,766 |
| `diagnoses`       | 303,496 |
| `readmissions`    | 101,766 |

Validation results:

* Total encounters: 101,766
* 30-day readmissions: 11,357
* 30-day readmission rate: 11.16%
* Unique patient surrogate keys: 71,518
* Duplicate patient surrogate keys: 0

---

## Airflow

The Apache Airflow DAG is named:

`hospital_readmission_etl`

The workflow contains the following stages:

```text
ingestion
    ↓
cleaning
    ↓
feature_engineering
    ↓
validation
    ↓
postgresql_loading
```

Airflow is used to orchestrate the execution order of the ETL pipeline.

Airflow 3.0.4 runs in WSL (Ubuntu) from `~/airflow_venv`. The DAG runs each
step with `BashOperator` inside the folder set by `PROJECT_DIR` in
`dags/hospital_readmission_dag.py` (currently
`/mnt/d/DOWNLOADS/Hospital-Readmission-Analytics`); change that line if the
project lives elsewhere. Airflow reads DAGs from `~/airflow/dags`, so copy the
file there after any change:

```bash
cp dags/hospital_readmission_dag.py ~/airflow/dags/
airflow dags list-import-errors
airflow tasks list hospital_readmission_etl
```

The loading step reads `DATABASE_URL` from `.env` and replaces the contents of
all four tables (TRUNCATE + insert in one transaction). Trigger it from the
Airflow UI or with `airflow dags trigger hospital_readmission_etl`.

Execution evidence: the last full run of `hospital_readmission_etl` succeeded
on 6 September 2026 (`screenshots/04_airflow_dag_structure.png`). On
15 October 2026 the DAG was redeployed to `~/airflow/dags` and parsed with no
import errors, and the validation gate was tested in a separate, throwaway
Airflow 3.0.4 setup: a failing validation left `postgresql_loading` as
`upstream_failed`, and a passing one let it run. The deployed DAG was then run
end to end (run `manual__2026-10-15T09:57:34`): all five tasks succeeded,
validation PASSED, and the loader wrote 101,766 admissions, 303,496 diagnoses,
71,518 patient summaries and 101,766 readmissions. For that run the
connection's `search_path` was set to a separate test schema, so the
production tables were not modified.

---

## Dashboard

The project uses Streamlit for interactive analytics.

The dashboard provides analytical views including:

1. Readmission rate by age group
2. Readmission by diagnosis category
3. Length-of-stay analysis
4. Admission and discharge patterns
5. Prior admission and readmission analysis

The dashboard reads analytical data from PostgreSQL.

---

## Running the Dashboard

Activate the project Python environment and run:

```bash
pip install -r requirements.txt
python src/ingestion/ingest.py
python src/transformation/clean.py
python src/transformation/feature_engineering.py
python src/validation/validate.py
python src/loading/load_to_postgres.py      # needs DATABASE_URL in .env
streamlit run dashboard/app.py
```

The steps above are the same ones the Airflow DAG runs. The dashboard will
open in the browser.

---

## Environment Variables

Database credentials are stored using environment variables.

Example:

```text
DATABASE_URL=postgresql+psycopg2://username:password@host:port/hospital_readmission
```

The `.env` file must not be committed to Git.

---

## Dependencies

Project dependencies are listed in:

```text
requirements.txt
```

Install them using:

```bash
pip install -r requirements.txt
```

---

## Documentation

Supporting project documentation is available in:

```text
docs/
```

including:

* `architecture.png` (architecture diagram)
* `Schema_Diagram.png` (PostgreSQL schema, drawn from `sql/schema.sql`)
* `data_dictionary.xlsx`

The `screenshots/` directory contains implementation and validation evidence.

---

## Privacy and Data Handling

The project uses a publicly available, de-identified healthcare dataset.

No private patient records, personally identifiable information, or confidential healthcare information are used.

The project follows a patient-safe approach by using surrogate patient keys in the analytical database.

---

## Project Scope

The sections above document Part 1 of the project, which focuses on data engineering, ETL, orchestration, analytical storage, validation, and visualization.

Part 2, below, adds machine-learning-based readmission prediction and the MLOps components around it. It reads Part 1's cleaned data and leaves the Part 1 pipeline, database and dashboard pages unchanged.

---

## Part 2 – Machine Learning and MLOps

> Academic decision-support prototype. The model is not clinically validated and must not be used for real patient decisions.

### What it does

Part 2 predicts whether a diabetic inpatient encounter will be followed by a readmission within 30 days of discharge, and wraps the model in a reproducible workflow:

| Phase | What was built | Documentation |
|---|---|---|
| 2 | Modelling cohort (99,340 encounters), 39 features known at discharge, patient-grouped train/validation/test split | `docs/part2/phase2_data_preparation.md` |
| 3 | Logistic Regression, Random Forest and XGBoost compared; calibrated Random Forest selected (test ROC-AUC 0.6715) | `docs/part2/phase3_model_training.md` |
| 4 | MLflow tracking (logged retrospectively from saved Phase 3 results) and registry: `hospital-readmission-30d-classifier` version 1, alias `@candidate` | `docs/part2/phase4_mlflow_tracking.md` |
| 5 | FastAPI service: `/health`, `/model-info`, `/predict` | `docs/part2/phase5_api.md` |
| 6 | Docker image serving an exported model bundle (container tested with a synthetic bundle only) | `docs/part2/phase6_docker.md` |
| 7 | Monitoring: drift (PSI), performance, explainability, retraining criteria, `/metrics` | `docs/part2/phase7_monitoring.md` |
| 8 | Readmission Risk Prediction page in the Streamlit dashboard, demo and submission evidence | `docs/part2/phase8_submission.md` |

Overview documents:

- `docs/report/Project_Report.pdf`: the project report (Part 1 and Part 2)
- `docs/part2/architecture.md`: architecture diagram
- `docs/part2/model_card.md`: model card
- `docs/part2/model_lifecycle.md`: registration, `@candidate`, approval, release, monitoring, retraining and rollback
- `docs/part2/model_bundle.md`: serving the registered model from a fresh clone
- `screenshots/part2/README.md`: index of the Part 2 screenshots

### Known limitations

- The real registered model has not been run inside a Docker container
  (Docker is not installed on the project machine). The image was built and
  tested with a synthetic bundle; the real bundle was served with Uvicorn.
- MLflow runs were logged after training from the saved Phase 3 results, not
  during training. `@candidate` means registered, not approved.
- Discrimination is modest (test ROC-AUC 0.6715), and there is no
  time-based evaluation because the dataset has no dates. The data are
  historical (US hospitals, 1999–2008).
- Monitoring is a report run on demand plus the API's in-memory `/metrics`;
  there is no continuous production alerting.

### Testing

On 15 October 2026 the full suite was run on the final code: **260 passed,
0 failed, 0 skipped** in 6 min 45 s. The 18 validation and cleaning tests also
pass on their own.

`tests/test_telemetry.py::test_metrics_never_contain_request_values` used to
be flaky: it searched the raw `/metrics` text for the input value `77`, which
a measured latency such as `12.77` could also contain. It now checks keys and
values exactly (latencies are excluded from the numeric check), and two tests
were added: `/metrics` may contain only its fixed aggregate fields, and its
contents must not change when the request values change. The telemetry tests
then passed 25 of 25 repeated runs, including a run with a forced latency of
77.77 ms.

### Quick start

```bash
pip install -r requirements.txt -r requirements-ml.txt
python -m pytest -q                       # full test suite
uvicorn src.api.main:app                  # prediction API on http://127.0.0.1:8000
streamlit run dashboard/app.py            # dashboard; open "Readmission Risk Prediction"
python -m src.monitoring.report           # monitoring report in artifacts/monitoring/
```

On a fresh clone, the registered model is not in Git: unzip the submitted
`model_bundle_v1.zip`, check it with `python -m src.api.bundle --verify
model_bundle` and start the API with `READMISSION_MODEL_BUNDLE=model_bundle`
(`docs/part2/model_bundle.md`). Retraining does not reproduce the served
model, because the API only accepts the pinned model hash.

The API and the monitoring report need the local MLflow store from Phase 4 (`mlflow.db` and `mlartifacts/`, not committed) or an exported model bundle (`READMISSION_MODEL_BUNDLE`). The prediction page calls the API at `http://localhost:8000` (or `READMISSION_API_URL`) and shows a message instead of a result when the API is not running. The full demo steps are in `docs/part2/phase8_submission.md`.

### Part 2 code layout

```text
src/ml/           cohort, features, split, training, selection, MLflow registration
src/api/          FastAPI service, request validation, model loading, telemetry, dashboard client
src/monitoring/   drift, retraining criteria, monitoring report
dashboard/pages/  Readmission Risk Prediction page
tests/            Part 2 tests
docs/part2/       phase documentation, architecture, model card
```

### Responsible use

- Inputs exclude identifiers, race and payer; race is used only to check performance across groups.
- The API does not log request bodies, and `/metrics` keeps aggregate counts only.
- The model does not apply to patients who died in hospital or were discharged to hospice; the API rejects those discharge codes.
- Discrimination is modest (ROC-AUC about 0.67). See the model card for limitations.

```

