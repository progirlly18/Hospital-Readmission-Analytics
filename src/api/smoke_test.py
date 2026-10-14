"""
Phase 6 smoke test for a running prediction API (for example the container).

    python -m src.api.smoke_test --url http://127.0.0.1:8000

Calls /health and /model-info, sends one valid request and one invalid
request to /predict, and exits with status 1 if any answer is unexpected.
The valid request is made-up example values, not a patient record. Uses the
standard library only, so it also runs inside the container.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

from src.ml import config
from src.api.allowed_values import ALLOWED_VALUES

EXAMPLE_REQUEST = {
    **{column: ALLOWED_VALUES[column][0] for column in config.CATEGORICAL_FEATURES},
    "age_years": 65, "time_in_hospital": 4, "num_lab_procedures": 40,
    "num_procedures": 1, "num_medications": 15, "number_outpatient": 0,
    "number_emergency": 0, "number_inpatient": 1, "number_diagnoses": 9,
    "medication_count": 2, "gender": "Female", "admission_type_id": "1",
    "discharge_disposition_id": "1", "admission_source_id": "7",
    "medical_specialty": "InternalMedicine",
}


def call(url, body=None):
    """(status, parsed JSON) for a GET, or a POST when body is given."""
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"{}")


def run(base_url):
    failures = []
    status, health = call(f"{base_url}/health")
    print(f"GET  /health      {status} {health}")
    if status != 200:
        failures.append("model is not loaded")
        return failures

    status, info = call(f"{base_url}/model-info")
    print(f"GET  /model-info  {status} version={info.get('model_version')} "
          f"threshold={info.get('threshold')} sha256={info.get('model_sha256')}")
    if status != 200 or len(info.get("features", [])) != len(config.FEATURE_COLUMNS):
        failures.append("unexpected /model-info answer")

    status, prediction = call(f"{base_url}/predict", EXAMPLE_REQUEST)
    print(f"POST /predict     {status} probability={prediction.get('readmission_probability')} "
          f"high_risk={prediction.get('high_risk')}")
    probability = prediction.get("readmission_probability")
    if status != 200 or not isinstance(probability, float) or not 0.0 <= probability <= 1.0:
        failures.append("valid request did not return a probability")
    elif prediction["high_risk"] != (probability >= prediction["threshold"]):
        failures.append("high_risk does not follow the threshold")

    status, _ = call(f"{base_url}/predict", {**EXAMPLE_REQUEST, "race": "x"})
    print(f"POST /predict     {status} (request with an extra 'race' field)")
    if status != 422:
        failures.append("invalid request was not rejected with 422")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    failures = run(parser.parse_args().url.rstrip("/"))
    print("SMOKE TEST " + ("FAILED: " + "; ".join(failures) if failures else "PASSED"))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
