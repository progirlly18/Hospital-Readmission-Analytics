# Part 2 – Phase 8: Dashboard integration, demo and submission evidence

> Academic decision-support prototype. The model is not clinically validated
> and must not be used for real patient decisions.

This phase adds a **Readmission Risk Prediction** page to the Streamlit
dashboard and collects the documentation and evidence for submission. Nothing
is trained or re-registered; Part 1 and the registered model (version 1,
`@candidate`, threshold 0.1617830043681171) are unchanged.

## What was added

| File | Purpose |
|---|---|
| `dashboard/pages/1_Readmission_Risk_Prediction.py` | The prediction page. Streamlit shows files in `dashboard/pages/` as extra pages in the sidebar, so `dashboard/app.py` (Part 1) is unchanged |
| `src/api/client.py` | Calls the API's `/health` and `/predict` (standard library only) and turns every failure into a message instead of an error |
| `tests/test_dashboard_prediction.py` | Client and page tests against the real API app over HTTP |
| `docs/part2/architecture.md`, `docs/part2/model_card.md` | Architecture diagram and model card |
| `README.md` | Part 2 section |

## The prediction page

- Shows the academic-prototype disclaimer at the top, and the API's own
  disclaimer with each result.
- Checks `/health` when it opens: connected (with the model version), API
  running without a model (503), or API not reachable. Predictions are only
  attempted when the user submits the form, and each failure gives a plain
  message; the page never crashes and the Part 1 pages do not depend on it.
- The form has the 39 model inputs, with the API's allowed values and the
  dataset's code descriptions (`data/raw/IDS_mapping.csv`). Expired and
  hospice discharge codes are not offered, because the model does not apply to
  them. Race and payer are not inputs. The number of diabetes medications is
  derived from the medication choices, as in training.
- The result shows the probability, the risk group (at or above the threshold
  is "Higher risk") and the threshold.
- Nothing is stored: the inputs go only to the API, which does not log them.
- The API address is `http://localhost:8000`, or `READMISSION_API_URL`.

## Running the demo

From the project root, with the project environment active
(`pip install -r requirements.txt -r requirements-ml.txt`) and the Phase 4
MLflow store in place (`mlflow.db`, `mlartifacts/`).

1. Start the prediction API (terminal 1):

   ```bash
   uvicorn src.api.main:app
   ```

   Or serve the exported bundle (Phase 6):
   `READMISSION_MODEL_BUNDLE=model_bundle uvicorn src.api.main:app` (bash) or
   `$env:READMISSION_MODEL_BUNDLE="model_bundle"; uvicorn src.api.main:app`
   (PowerShell).

2. Check it from the command line (terminal 2):

   ```bash
   python -m src.api.smoke_test --url http://127.0.0.1:8000
   ```

3. Start the dashboard (terminal 2, PostgreSQL running for the Part 1 pages):

   ```bash
   streamlit run dashboard/app.py
   ```

   Open **Readmission Risk Prediction** in the sidebar.

4. Demo with made-up inputs only (never real patient data):
   - **Scenario A, form defaults:** age 60–70, 4-day emergency admission
     from the emergency room, discharged home, on insulin, no prior visits.
     Expect a low probability, "Lower risk".
   - **Scenario B:** change *Inpatient visits, past year* to 4 and *Discharge
     disposition* to 3 (skilled nursing facility). Expect a higher
     probability; prior inpatient visits and discharge destination are the
     model's two most important inputs.
   - **Failure handling:** stop the API (Ctrl+C) and submit again. The page
     says the API is not reachable; the Part 1 pages still work.

5. Optional: `python -m src.monitoring.report` writes the monitoring report
   to `artifacts/monitoring/`, and `http://127.0.0.1:8000/metrics` shows the
   API's request counts.

## Tests

| File | Tests | What they cover |
|---|---:|---|
| `tests/test_dashboard_prediction.py` | 10 | Client against the real API app on a local port: health and prediction, rejected fields named without their values, model not loaded, API not reachable, address from the environment, request body matches the API schema. Page (Streamlit's test runner): disclaimer and message without the API and no crash on submit, prediction shown with the API, model-not-loaded message, no race or payer inputs and no excluded discharge codes |

## Submission evidence

| Phase | Pull request | Evidence |
|---|---|---|
| 2 Data preparation | #1 | `phase2_data_preparation.md`: cohort 99,340, patient-grouped split with zero overlap |
| 3 Model training | #2 | `phase3_model_training.md`: comparison of three models, test ROC-AUC 0.6715 |
| 4 MLflow tracking | #3 | `phase4_mlflow_tracking.md`: runs and registered version 1 `@candidate` |
| 5 FastAPI | #4 | `phase5_api.md`: validated `/predict`, 422 and 503 handling |
| 6 Docker | #5 | `phase6_docker.md`: image build and container checks |
| 7 Monitoring | #6 | `phase7_monitoring.md`: report on the registered model, no retraining needed on real data |
| 8 Dashboard and submission | this PR | this file, `model_card.md`, `architecture.md` |

Results on the project machine for this phase are recorded below.

### Actual results

_To be added from the run on the project machine._

## Limitations

- The page is for demonstration with made-up inputs; it has no login and no
  link to real records.
- The API must be started separately; the page does not start it.
- The real model has not been run inside a Docker container (no Docker on the
  project machine; see Phase 6).
