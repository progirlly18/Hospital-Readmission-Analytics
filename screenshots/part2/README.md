# Part 2 evidence

Captured on the project machine on 2026-10-15 from the running applications
(MLflow 3.17.1 UI, FastAPI, Streamlit) with a headless Microsoft Edge, or
copied unchanged from the Phase 3 training outputs (`artifacts/phase3/plots/`).
All inputs are made-up example values; no patient record was entered.
Registered model: `hospital-readmission-30d-classifier` version 1
(`@candidate`), threshold 0.1617830043681171. The MLflow database was
byte-identical before and after the captures.

| File | What it shows |
|---|---|
| `01_mlflow_experiment_runs.png` | Experiment `hospital-readmission-30d`: 6 matching runs, shown as the registration run (linked to the registered model) and the Phase 3 comparison parent run, whose 4 child runs (LR, RF and XGBoost validation, RF test) are collapsed |
| `02_mlflow_model_comparison_params.png` | MLflow run comparison of the three models: tuned parameters, calibration, selected thresholds |
| `03_mlflow_model_comparison_metrics.png` | The three validation runs plus the selected model's test run: metrics and tags, including `historical_note` (runs logged retrospectively from saved Phase 3 results) and `retrained=false` |
| `04_mlflow_registered_model_candidate.png` | Model registry: version 1 with alias `@candidate` and its tags (threshold, model hash, split fingerprint) |
| `05_mlflow_model_version1_details.png` | Version 1 details: source run, description with disclaimer, schema of 39 inputs |
| `06_fastapi_docs.png` | FastAPI interactive docs: `/health`, `/model-info`, `/predict`, `/metrics` |
| `07_fastapi_predict_200.png` | `POST /predict` with the smoke-test example: HTTP 200, probability 0.0671, `high_risk` false |
| `08_fastapi_predict_422_extra_field.png` | Same request plus a `race` field: HTTP 422 `extra_forbidden`, value not echoed |
| `09_fastapi_model_info.png` | `/model-info`: version, alias, threshold, model hash, 39 features |
| `10_fastapi_metrics.png` | `/metrics` after the demo calls: counts, failure rate, latency percentiles, no input values |
| `11_streamlit_prediction_scenario_A.png` | Dashboard prediction page, Scenario A (form defaults): 9.2%, Lower risk, threshold 16.2% |
| `12_streamlit_prediction_scenario_B.png` | Scenario B (4 prior inpatient visits, discharged to a skilled nursing facility): 31.6%, Higher risk |
| `13_streamlit_prediction_api_unavailable.png` | Same page with the API stopped: clear message, no crash |
| `14_monitoring_report.png` | `artifacts/monitoring/monitoring_report.md` rendered as a page: drift, input-data quality, importance, retraining verdicts (real vs simulated) |
| `15_streamlit_part1_dashboard_still_working.png` | Part 1 dashboard page with live PostgreSQL figures (101,766 encounters, 11.16%, 4.40 days) |
| `16_eval_roc_pr_validation.png` | ROC and precision-recall curves of the three models on validation |
| `17_eval_calibration_validation.png` | Reliability diagram of the three calibrated models on validation |
| `18_eval_calibration_test_selected.png` | Reliability diagram of the selected model on the held-out test split |
| `19_eval_confusion_matrix_test.png` | Confusion matrix of the selected model on test at the chosen threshold |
| `20_eval_permutation_importance.png` | Permutation importance of the 39 inputs |

Screenshot 14 includes the input-data quality section added in pull request
#8.
