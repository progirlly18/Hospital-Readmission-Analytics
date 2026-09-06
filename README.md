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

### Main Files

- `diabetic_data.csv` – primary hospital encounter dataset
- `IDS_mapping.csv` – identifier/code mapping information

The original dataset is maintained separately in the project's raw data area.

---

## Project Architecture

The implemented pipeline follows this flow:

Public Dataset
→ Raw Data
→ Python/Pandas ETL
→ Cleaning
→ Feature Engineering
→ Validation
→ PostgreSQL
→ Streamlit Dashboard

Apache Airflow is used to orchestrate the ETL workflow.

---

## Project Structure

```text
Hospital-Readmission-Analytics/
│
├── data/
│   ├── raw/
│   ├── staging/
│   ├── cleaned/
│   └── analytical/
│
├── dags/
│   └── hospital_readmission_dag.py
│
├── src/
│   ├── ingestion/
│   ├── loading/
│   ├── transformation/
│   └── validation/
│
├── sql/
│   └── schema.sql
│
├── dashboard/
│   └── app.py
│
├── logs/
│   ├── ingestion/
│   └── validation/
│
├── notebooks/
│
├── docs/
│   ├── architecture.png
│   └── data_dictionary.xlsx
│
├── screenshots/
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
````

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

Validation includes checks for:

* Missing or invalid records
* Duplicate encounters
* Unique patient surrogate keys
* Valid relationships between tables
* Non-negative numerical/count fields
* Readmission indicators

Rejected or invalid records are recorded through the project's error/rejection logging process.

### 5. PostgreSQL Loading

The validated data is loaded into PostgreSQL.

The database contains the following analytical tables:

* `patient_summary`
* `admissions`
* `diagnoses`
* `readmissions`

---

## Database Results

Current populated database:

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
streamlit run dashboard/app.py
```

The dashboard will open in the browser.

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

* `architecture.png`
* `data_dictionary.xlsx`

The `screenshots/` directory contains implementation and validation evidence.

---

## Privacy and Data Handling

The project uses a publicly available, de-identified healthcare dataset.

No private patient records, personally identifiable information, or confidential healthcare information are used.

The project follows a patient-safe approach by using surrogate patient keys in the analytical database.

---

## Project Scope

This README documents Part 1 of the project, which focuses on data engineering, ETL, orchestration, analytical storage, validation, and visualization.

Machine-learning-based readmission prediction and subsequent MLOps components are outside the current Part 1 implementation scope.

```

