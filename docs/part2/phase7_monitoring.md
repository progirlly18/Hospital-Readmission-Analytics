# Part 2 – Phase 7: Monitoring, drift, explainability and retraining criteria

This phase adds monitoring around the registered model without changing it:
an offline monitoring report (input and prediction drift, labelled
performance, subgroups, explainability, retraining criteria) and a live
`/metrics` endpoint on the prediction API.

> Academic decision-support prototype. The model is not clinically validated
> and must not be used for real patient decisions.

Nothing is trained or re-registered. Part 1, the registered model (version 1,
alias `@candidate`), its MLflow runs and the Phase 2 split are unchanged.
Streamlit integration is Phase 8.

## What was added

| File | Purpose |
|---|---|
| `src/monitoring/drift.py` | Population Stability Index (PSI) per input and for predicted probabilities |
| `src/monitoring/retraining.py` | Documented retraining and review criteria, applied to monitored batches |
| `src/monitoring/report.py` | `python -m src.monitoring.report`: builds the monitoring report |
| `src/api/telemetry.py` | Aggregate-only request and prediction counters |
| `src/api/main.py` | New `GET /metrics` endpoint and a middleware that feeds it |
| `tests/test_monitoring.py`, `tests/test_telemetry.py` | Tests (below) |

## How to run the report

From the project root, with the model available exactly as for the API (the
MLflow registry by default, or the Phase 6 bundle):

```
python -m src.monitoring.report
# or, from the exported bundle:
READMISSION_MODEL_BUNDLE=model_bundle python -m src.monitoring.report
```

The model is loaded through `src.api.model.load_model`, so the same checks
apply (version, alias, threshold, feature list, model hash); a model that
fails them stops the report. The report also rebuilds the Phase 2 split and
stops unless its fingerprint equals the training split's (`45788b05…a941`).

Outputs go to `artifacts/monitoring/monitoring_report.json` and `.md`, which
are gitignored. Options: `--out DIR`, `--importance-rows N` (default 5,000).

## Baselines and monitored batches

| Role | Data |
|---|---|
| Input baseline | Training split (69,538 encounters) |
| Prediction and performance reference | Validation split (14,901), the data the threshold was chosen on |
| Real monitored batch | Held-out test split (14,901), unchanged |
| Simulated batches | The two halves of the test split with an **artificial** shift |

There is no production traffic, so the test split stands in for a later
batch of real encounters. The simulated batches (age +10 years capped at 95,
prior inpatient visits +2, length of stay +2 days capped at 14; labels
unchanged) exist only to show that the checks react. They are labelled
`SIMULATED` in the report and judged separately from the real batch, so they
can never produce a recommendation about the real model.

## Drift (PSI)

PSI = Σ (current share − baseline share) × ln(current share / baseline share).

- Numeric inputs: 10 bins at the baseline's quantiles (tied edges merged).
- Categorical inputs: one bin per baseline category, plus an "unseen" bin, so
  a new category counts as drift instead of being dropped. Its share is
  reported as `unseen_share`.
- Predictions: the same numeric binning of predicted probabilities against
  validation predictions.
- Empty bins are floored at 0.0001 so PSI stays finite.

| PSI | Status |
|---|---|
| < 0.10 | stable |
| 0.10 – 0.25 | moderate |
| > 0.25 | significant |

## Performance and subgroups

For every batch with labels: ROC-AUC, PR-AUC, Brier score, expected
calibration error (ECE), and at the registered threshold (0.1618) precision,
recall, F1 and the share flagged high risk. For the real batch, subgroup
metrics by gender, age band and race reuse Phase 3's `subgroup_metrics`
(race is reported only for this fairness check; it is never a model input).
Subgroups under 30 encounters are suppressed.

## Explainability

Permutation importance on a 5,000-row validation sample (3 repeats): the drop
in ROC-AUC when one input is shuffled, reusing Phase 3's
`feature_importance`. It covers all 39 inputs through the full pipeline, so
importance is reported per original input rather than per one-hot column.
The top 10 become the report's key features for the drift criterion.

SHAP is not used. It is not installed in the project environment, and for a
calibrated random forest wrapped in a preprocessing pipeline it would explain
one-hot columns of the inner trees rather than the calibrated output. This is
global importance only; the API does not return per-patient explanations.

## Retraining criteria

Defined in `src/monitoring/retraining.py`. A triggered criterion is a
recommendation for a person to review the data, retrain through the Phase 3
pipeline and register a new version; nothing retrains automatically.

| Criterion | Rule | Outcome |
|---|---|---|
| `feature_drift` | PSI > 0.25 on one of the 10 most important inputs for 2 consecutive batches | Retrain (a single batch is a warning) |
| `prediction_drift` | PSI of predicted probabilities > 0.25 | Retrain |
| `roc_auc_drop` | ROC-AUC more than 0.05 below validation | Retrain |
| `calibration` | ECE above 0.03 (validation: 0.0070) | Retrain or recalibrate |
| `flagged_rate` | Share flagged high risk changes by more than 0.10 | Retrain or review threshold |
| `subgroup_gap` | A subgroup with at least 30 positives has ROC-AUC more than 0.10 below the batch's | Fairness review |

Performance criteria need outcomes, which arrive 30 days after discharge, so
in practice they run on batches at least a month old; the drift criteria
need no labels and can run earlier.

## `/metrics` endpoint

`GET /metrics` on the prediction API returns, since the process started:

- `model_loaded` and `model_error`;
- request counts by endpoint and status code (unknown paths are counted
  together as `other`, so arbitrary URLs cannot grow memory);
- for `/predict`: requests, succeeded, rejected as invalid (422), model
  unavailable (503), server errors and failure rate;
- `/predict` latency p50, p95 and max over the last 1,000 requests;
- predictions: count, share flagged high risk, and counts in fixed
  probability bins (0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 1).

Only counters are kept, in memory. No request body, feature value, gender or
individual prediction is stored or returned. Counters reset on restart.

## Tests

| File | Tests | What they cover |
|---|---:|---|
| `tests/test_monitoring.py` | 22 | PSI is zero when unchanged and grows with shift, stays finite with empty bins, status bands, tied edges; only shifted inputs flagged; unseen categories counted. Retraining: stable batches, drift must persist two batches, unimportant inputs ignored, each performance criterion, no trigger at the limits, subgroup gap asks for review only. Report on a synthetic cohort and model: real and simulated batches labelled and judged separately, aggregates only (no identifiers, subgroups under 30 suppressed), importance for all 39 inputs, Markdown verdicts, refuses a different split; the real split matches the training fingerprint (when the cleaned data is present) |
| `tests/test_telemetry.py` | 6 | Counts, failure rate, latency and probability bins; 503 recorded with the model error; no request values in `/metrics`; unknown paths share one counter; server errors counted; bins cover 0 to 1 |

## Actual results

Registered model, on the project machine: `hospital-readmission-30d-classifier`
version 1 (`@candidate`), loaded from the MLflow registry, threshold
0.1617830043681171, model sha256 `1d17bf4a…62f9`. Runtime 59 seconds.

| Batch | Rows | ROC-AUC | ECE | Flagged | Prediction PSI | Inputs with PSI > 0.25 |
|---|---:|---:|---:|---:|---:|---|
| validation (reference) | 14,901 | 0.6673 | 0.0070 | 0.2068 | – | – |
| test (real) | 14,901 | 0.6715 | 0.0046 | 0.2049 | 0.0021 | none |
| simulated_shift_a | 7,451 | 0.6457 | 0.0566 | 0.5842 | 2.0086 | number_inpatient, time_in_hospital, age_years |
| simulated_shift_b | 7,450 | 0.6486 | 0.0794 | 0.6681 | 2.3647 | number_inpatient, time_in_hospital, age_years |

The validation and test ROC-AUC match the Phase 3 results, as expected for the
unchanged model and split.

- Real data (test split): **no retraining recommended**; no criterion met.
- Simulated shift: retraining recommended, as intended for the demonstration:
  prediction drift, calibration and flagged rate in both halves; the three
  shifted inputs drifted in the first half (warning) and again in the second
  (persistent, so retraining).

Most important inputs (ROC-AUC drop when shuffled): number_inpatient 0.0582,
discharge_disposition_id 0.0371, number_diagnoses 0.0064, medical_specialty
0.0039, diag_1_group 0.0039, time_in_hospital 0.0038, age_years 0.0035,
number_emergency 0.0035, num_medications 0.0031, A1Cresult 0.0024.

Tests on the project machine: full suite 219 passed (191 before Phase 7 plus
28 new). In the cloud workspace, which has the data but not the registered
model, the new tests also pass.

The registry was unchanged afterwards: version 1 only, `@candidate` on
version 1, same threshold and source run.

## Limitations

- The "real" batch is the held-out test split, not production traffic; the
  report shows how monitoring would run, not how the model behaves later.
  Each run re-reads the test labels, which Phase 3 used only once for the
  final evaluation; the report only measures and never tunes anything on
  them, but new labelled data should replace the test split once it exists.
- The simulated shift is invented and its performance figures are only
  illustrative.
- PSI limits (0.10, 0.25) are common conventions, not values tuned for this
  dataset; small batches give noisy PSI.
- Permutation importance is global and correlated inputs share credit; it is
  not a per-patient explanation.
- `/metrics` is in-memory and per process; a deployment would export these
  counters to a metrics system and alert on them.
