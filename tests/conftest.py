import numpy as np
import pandas as pd
import pytest

from src.ml import config


def make_encounter(encounter_id, patient, readmitted="NO", **overrides):
    """One synthetic encounter shaped like a row of the Part 1 cleaned CSV."""
    row = {
        "encounter_id": str(encounter_id),
        "patient_nbr": str(patient),
        "patient_sk": f"p{patient:05d}",
        "race": "Caucasian",
        "gender": "Female",
        "age": "60-70",
        "weight": pd.NA,
        "admission_type_id": "1",
        "discharge_disposition_id": "1",
        "admission_source_id": "7",
        "time_in_hospital": "3",
        "payer_code": "MC",
        "medical_specialty": "InternalMedicine",
        "num_lab_procedures": "40",
        "num_procedures": "1",
        "num_medications": "15",
        "number_outpatient": "0",
        "number_emergency": "0",
        "number_inpatient": "0",
        "diag_1": "428",
        "diag_2": "250.01",
        "diag_3": "V45",
        "number_diagnoses": "9",
        "max_glu_serum": pd.NA,
        "A1Cresult": ">8",
        "change": "Ch",
        "diabetesMed": "Yes",
        "readmitted": readmitted,
    }
    for drug in config.ALL_MEDICATION_COLUMNS:
        row[drug] = "No"
    row["insulin"] = "Steady"
    row.update(overrides)
    return row


def build_synthetic_cohort(n_patients=600, seed=0):
    """
    Synthetic patients with 1-4 encounters each, ~12% positives.
    Fully deterministic for a given seed.
    """
    rng = np.random.default_rng(seed)
    rows = []
    encounter_id = 1
    for patient in range(n_patients):
        for _ in range(rng.integers(1, 5)):
            label = "<30" if rng.random() < 0.12 else rng.choice([">30", "NO"])
            rows.append(make_encounter(
                encounter_id,
                patient,
                readmitted=label,
                time_in_hospital=str(rng.integers(1, 15)),
                number_inpatient=str(rng.integers(0, 5)),
                medical_specialty=rng.choice(
                    ["InternalMedicine", "Cardiology", "Surgery-General"]
                ),
            ))
            encounter_id += 1
    df = pd.DataFrame(rows)
    df[config.TARGET_COLUMN] = (df["readmitted"] == "<30").astype(int)
    return df


@pytest.fixture
def synthetic_cohort():
    return build_synthetic_cohort()
