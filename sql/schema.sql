-- =========================================================
-- Hospital Readmission Analytics
-- PostgreSQL Database Schema
-- =========================================================


-- ---------------------------------------------------------
-- 1. Patient Summary
-- Grain: one row per patient
-- ---------------------------------------------------------

CREATE TABLE IF NOT EXISTS patient_summary (
    patient_sk VARCHAR(16) PRIMARY KEY,

    encounter_count INTEGER,
    total_length_of_stay INTEGER,
    average_length_of_stay NUMERIC(10,2),

    total_diagnoses INTEGER,
    total_medications INTEGER,

    prior_admission_count INTEGER
);


-- ---------------------------------------------------------
-- 2. Admissions
-- Grain: one row per hospital encounter
-- ---------------------------------------------------------

CREATE TABLE IF NOT EXISTS admissions (
    encounter_id BIGINT PRIMARY KEY,
    patient_sk VARCHAR(16) NOT NULL,

    age VARCHAR(20),
    gender VARCHAR(20),
    race VARCHAR(50),

    admission_type_id INTEGER,
    admission_type VARCHAR(100),

    discharge_disposition_id INTEGER,
    discharge_disposition VARCHAR(255),

    admission_source_id INTEGER,
    admission_source VARCHAR(255),

    length_of_stay_days INTEGER,
    prior_admission INTEGER,

    num_lab_procedures INTEGER,
    num_procedures INTEGER,
    num_medications INTEGER,

    number_outpatient INTEGER,
    number_emergency INTEGER,
    number_inpatient INTEGER,

    CONSTRAINT fk_admission_patient
        FOREIGN KEY (patient_sk)
        REFERENCES patient_summary(patient_sk)
);


-- ---------------------------------------------------------
-- 3. Diagnoses
-- Grain: one row per diagnosis per encounter
-- ---------------------------------------------------------

CREATE TABLE IF NOT EXISTS diagnoses (
    diagnosis_id BIGSERIAL PRIMARY KEY,

    encounter_id BIGINT NOT NULL,
    patient_sk VARCHAR(16) NOT NULL,

    diagnosis_position INTEGER NOT NULL,
    diagnosis_code VARCHAR(20),
    diagnosis_category VARCHAR(100),

    CONSTRAINT fk_diagnosis_encounter
        FOREIGN KEY (encounter_id)
        REFERENCES admissions(encounter_id),

    CONSTRAINT fk_diagnosis_patient
        FOREIGN KEY (patient_sk)
        REFERENCES patient_summary(patient_sk)
);


-- ---------------------------------------------------------
-- 4. Readmissions
-- Grain: one row per encounter
-- ---------------------------------------------------------

CREATE TABLE IF NOT EXISTS readmissions (
    encounter_id BIGINT PRIMARY KEY,
    patient_sk VARCHAR(16) NOT NULL,

    readmission_status VARCHAR(10),
    readmitted_within_30_days INTEGER,

    CONSTRAINT fk_readmission_encounter
        FOREIGN KEY (encounter_id)
        REFERENCES admissions(encounter_id),

    CONSTRAINT fk_readmission_patient
        FOREIGN KEY (patient_sk)
        REFERENCES patient_summary(patient_sk)
);