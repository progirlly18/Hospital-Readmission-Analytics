"""
Part 2: Readmission Risk Prediction page of the Streamlit dashboard.

Streamlit adds files in dashboard/pages/ as extra pages next to the Part 1
dashboard (dashboard/app.py), which is unchanged. This page needs no database:
it sends the 39 model inputs to the prediction API's POST /predict and shows
the result. Set READMISSION_API_URL if the API is not at http://localhost:8000.

Inputs are typed in for a hypothetical encounter. Nothing is stored or
logged here; race and payer are not inputs of the model.
"""

import csv
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api import client  # noqa: E402
from src.api.allowed_values import ALLOWED_VALUES  # noqa: E402
from src.ml import config  # noqa: E402

DISCLAIMER = (
    "Academic decision-support prototype. The model is not clinically "
    "validated and must not be used for real patient decisions."
)
AGE_BANDS = {f"{low}-{low + 10}": float(low + 5) for low in range(0, 100, 10)}
# Expired and hospice codes are outside the model's cohort; the API rejects them.
EXCLUDED_DISCHARGE = {str(code) for code in
                      config.EXPIRED_DISPOSITION_IDS | config.HOSPICE_DISPOSITION_IDS}
ID_COLUMNS = ("admission_type_id", "discharge_disposition_id", "admission_source_id")
SHOWN_MEDICATIONS = ("insulin", "metformin")
DEFAULTS = {
    "admission_type_id": "1", "discharge_disposition_id": "1",
    "admission_source_id": "7", "medical_specialty": "InternalMedicine",
    "diag_1_group": "Circulatory", "diag_2_group": "Circulatory",
    "diag_3_group": "Diabetes", "max_glu_serum": "None", "A1Cresult": "None",
    "change": "No", "diabetesMed": "Yes", "insulin": "Steady",
}


@st.cache_data
def id_descriptions():
    """{column: {code: description}} from the dataset's IDS_mapping.csv."""
    path = PROJECT_ROOT / "data" / "raw" / "IDS_mapping.csv"
    labels, column = {}, None
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.reader(handle):
                if not row or not row[0].strip():
                    column = None
                elif row[0] in ID_COLUMNS:
                    column = row[0]
                    labels[column] = {}
                elif column:
                    labels[column][row[0].strip()] = row[1].strip()
    except OSError:
        pass
    return labels


def options(column):
    values = list(ALLOWED_VALUES[column])
    if column == "discharge_disposition_id":
        values = [value for value in values if value not in EXCLUDED_DISCHARGE]
    if column in ID_COLUMNS:
        values.sort(key=int)
    return values


def select(column, label, help_text=None):
    values = options(column)
    default = DEFAULTS.get(column, "No" if "No" in values else values[0])
    names = id_descriptions().get(column, {})
    return st.selectbox(
        label, values, index=values.index(default), key=column, help=help_text,
        format_func=lambda code: f"{code}: {names[code]}" if code in names else code,
    )


def number(column, label, low, high, default):
    return st.number_input(label, min_value=low, max_value=high, value=default,
                           step=1, key=column)


st.set_page_config(page_title="Readmission Risk Prediction", page_icon="🏥",
                   layout="wide")
st.title("Readmission Risk Prediction")
st.caption("Part 2 · Machine learning model served by the prediction API")
st.warning(DISCLAIMER)

status = client.health()
if status.ok:
    st.success(f"Prediction API connected (model version "
               f"{status.data.get('model_version')}).")
elif status.status == "model_unavailable":
    st.error(f"The prediction API is running but its model is not loaded: "
             f"{status.message} Predictions are unavailable.")
else:
    st.error(f"{status.message} Start it with `uvicorn src.api.main:app` from "
             f"the project root (see docs/part2/phase8_submission.md); the "
             f"Part 1 dashboard pages still work without it.")

st.markdown("Enter a hypothetical encounter. The model estimates the probability "
            "of readmission within 30 days of discharge.")

with st.form("prediction"):
    values = {}
    left, middle, right = st.columns(3)
    with left:
        st.subheader("Patient and stay")
        age_band = st.selectbox("Age band (years)", list(AGE_BANDS), index=6, key="age_band")
        values["age_years"] = AGE_BANDS[age_band]
        values["gender"] = select("gender", "Gender")
        values["time_in_hospital"] = number("time_in_hospital", "Length of stay (days)", 1, 14, 4)
        values["admission_type_id"] = select("admission_type_id", "Admission type")
        values["admission_source_id"] = select("admission_source_id", "Admission source")
        values["discharge_disposition_id"] = select(
            "discharge_disposition_id", "Discharge disposition",
            "Expired and hospice codes are excluded: the model was not trained on them.")
    with middle:
        st.subheader("Prior visits and care")
        values["number_inpatient"] = number("number_inpatient", "Inpatient visits, past year", 0, 1000, 0)
        values["number_emergency"] = number("number_emergency", "Emergency visits, past year", 0, 1000, 0)
        values["number_outpatient"] = number("number_outpatient", "Outpatient visits, past year", 0, 1000, 0)
        values["num_lab_procedures"] = number("num_lab_procedures", "Lab procedures", 0, 1000, 40)
        values["num_procedures"] = number("num_procedures", "Other procedures", 0, 1000, 1)
        values["num_medications"] = number("num_medications", "Medications given", 0, 1000, 15)
        values["number_diagnoses"] = number("number_diagnoses", "Number of diagnoses", 0, 1000, 8)
    with right:
        st.subheader("Diagnoses and diabetes care")
        values["medical_specialty"] = select("medical_specialty", "Admitting specialty")
        values["diag_1_group"] = select("diag_1_group", "Primary diagnosis group")
        values["diag_2_group"] = select("diag_2_group", "Secondary diagnosis group")
        values["diag_3_group"] = select("diag_3_group", "Additional diagnosis group")
        values["A1Cresult"] = select("A1Cresult", "HbA1c result")
        values["max_glu_serum"] = select("max_glu_serum", "Max glucose serum")
        values["change"] = select("change", "Diabetes medication changed")
        values["diabetesMed"] = select("diabetesMed", "Diabetes medication prescribed")
        for drug in SHOWN_MEDICATIONS:
            values[drug] = select(drug, drug.capitalize())

    with st.expander("Other diabetes medications (default: not prescribed)"):
        columns = st.columns(4)
        others = [drug for drug in config.MEDICATION_FEATURES if drug not in SHOWN_MEDICATIONS]
        for index, drug in enumerate(others):
            with columns[index % 4]:
                values[drug] = select(drug, drug)
        other_medications = st.number_input(
            "Other diabetes medications in use that are not model inputs", 0,
            len(config.ALL_MEDICATION_COLUMNS) - len(config.MEDICATION_FEATURES), 0,
            key="other_medications",
            help="Counts toward the number of diabetes medications only.")

    submitted = st.form_submit_button("Estimate readmission risk", type="primary")

if submitted:
    result = client.predict(client.request_body(values, other_medications))
    if result.ok:
        probability = result.data["readmission_probability"]
        threshold = result.data["threshold"]
        level = "Higher risk" if result.data["high_risk"] else "Lower risk"
        one, two, three = st.columns(3)
        one.metric("Estimated 30-day readmission probability", f"{probability:.1%}")
        two.metric("Risk group", level)
        three.metric("Decision threshold", f"{threshold:.1%}")
        st.caption(
            f"{level} means the probability is "
            f"{'at or above' if result.data['high_risk'] else 'below'} the threshold "
            f"chosen on validation data. Model: {result.data['model_name']} version "
            f"{result.data['model_version']}. {result.data['disclaimer']}")
    elif result.status == "invalid":
        st.error(f"The API rejected the input: {result.message}")
    elif result.status == "model_unavailable":
        st.error(f"The model is not available right now: {result.message}")
    else:
        st.error(f"No prediction: {result.message}")
