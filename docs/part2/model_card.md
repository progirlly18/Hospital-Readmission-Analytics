# Model card: hospital-readmission-30d-classifier, version 1

> Academic decision-support prototype. The model is not clinically validated
> and must not be used for real patient decisions.

## Summary

| Item | Value |
|---|---|
| Registered name | `hospital-readmission-30d-classifier`, version 1, alias `@candidate` (MLflow registry) |
| Task | Probability that a diabetic inpatient encounter is followed by a readmission within 30 days of discharge |
| Model | Random Forest (200 trees, depth 16, min leaf 10, `balanced_subsample`) with sigmoid calibration (`CalibratedClassifierCV`) |
| Decision threshold | 0.1617830043681171 (maximises F1 on validation); `high_risk` = probability ≥ threshold |
| Inputs | 39 features, all known at discharge (`src/ml/config.py`) |
| Serialization | skops (no pickle); model SHA-256 `1d17bf4a…62f9` checked on every load |
| Serving | FastAPI `POST /predict` (Phase 5), Docker image (Phase 6), Streamlit page (Phase 8) |

## Intended use

- Coursework demonstration of an end-to-end MLOps workflow: data preparation,
  model comparison, tracking, serving, monitoring and a dashboard.
- Scoring made-up or historical encounters from the same dataset.

**Out of scope:** any real clinical decision, patients outside the cohort
(expired or hospice discharges, which the API rejects), other hospitals,
countries or time periods without re-validation.

## Data

- UCI "Diabetes 130-US Hospitals 1999–2008", public and de-identified, via
  the Part 1 cleaned file `data/cleaned/diabetic_data_cleaned.csv`.
- Cohort: 99,340 encounters after removing expired (1,652) and hospice (771)
  discharges and 3 invalid genders. Positive rate 11.39%.
- Target: `readmitted == "<30"` is 1; `">30"` and `"NO"` are 0.
- Split by patient (no patient in two splits), stratified, fixed hash order:
  train 69,538 / validation 14,901 / test 14,901. Split fingerprint
  `45788b05…a941`.

## Features

10 numeric (age band midpoint, length of stay, lab and other procedures,
medications given, outpatient, emergency and inpatient visits in the prior
year, number of diagnoses, number of diabetes medications in use) and 29
categorical (gender, admission type, discharge disposition, admission source,
admitting specialty, three diagnosis groups, glucose and HbA1c results,
medication change, diabetes medication prescribed, 17 diabetes medications).

Not used: identifiers, `race` (kept only for the fairness analysis below),
`payer_code`, `weight` (97% missing), near-constant medications. Gender is an
input; the API never logs or returns it.

## Model selection

Logistic Regression, Random Forest and XGBoost were tuned with grouped 5-fold
cross-validation on the training split, calibrated, and compared on
validation with a rule fixed in advance (Phase 3):

| Validation | Logistic Regression | Random Forest | XGBoost |
|---|---:|---:|---:|
| ROC-AUC | 0.6618 | 0.6673 | 0.6682 |
| PR-AUC | 0.2139 | 0.2192 | 0.2271 |
| Brier score | 0.0984 | 0.0979 | 0.0976 |
| ECE | 0.0099 | 0.0070 | 0.0056 |
| F1 at its threshold | 0.2790 | 0.2857 | 0.2887 |

Random Forest and XGBoost were both within the tolerances, and the rule
chose the simpler one. XGBoost is slightly better on most metrics; the choice
is documented in `docs/part2/phase3_model_training.md`.

## Performance (held-out test split, evaluated once)

14,901 encounters, 1,704 readmitted (11.44%).

| Metric | Value |
|---|---:|
| ROC-AUC | 0.6715 |
| PR-AUC (base rate 0.114) | 0.2237 |
| Brier score | 0.0967 |
| Expected calibration error | 0.0046 |
| Precision / recall / F1 at threshold | 0.2129 / 0.3815 / 0.2733 |
| Share flagged high risk | about 20% |

In plain terms, the model flags about 1 in 5 encounters, catches about 38% of
real 30-day readmissions, and about 1 in 5 flagged encounters is readmitted.

## Fairness (test split, threshold 0.1618)

| Group | ROC-AUC | Recall | Precision |
|---|---:|---:|---:|
| Female / Male | 0.671 / 0.672 | 0.392 / 0.369 | 0.214 / 0.211 |
| Age <50 / 50–69 / 70+ | 0.729 / 0.682 / 0.635 | 0.427 / 0.370 / 0.378 | 0.249 / 0.224 / 0.196 |
| Caucasian / AfricanAmerican / Hispanic | 0.673 / 0.654 / 0.784 | 0.380 / 0.384 / 0.512 | 0.213 / 0.197 / 0.339 |

Calibration holds across gender and age groups. Ranking is weaker for
patients aged 70 and over. Asian, Other and Unknown race groups have fewer
than 30 positives each, so their metrics are suppressed as unreliable.

## Explainability

Permutation importance (drop in ROC-AUC when one input is shuffled). The
largest effects are prior inpatient visits (about 0.05–0.06) and discharge
disposition (about 0.04); every other input adds less than 0.01 on its own.
SHAP is not used: it is not installed in the project environment and, for this
calibrated pipeline, would explain encoded columns of the inner trees rather
than the 39 inputs. Importance shows association, not cause, and there are no
per-patient explanations.

## Monitoring and retraining

Phase 7 (`docs/part2/phase7_monitoring.md`): PSI drift on every input and on
predictions, labelled performance, calibration and subgroup checks, and
documented retraining criteria (for example PSI above 0.25 on a top-10 input
for two consecutive batches, ROC-AUC more than 0.05 below validation, or ECE
above 0.03). On the test split no criterion is met. The API's `/metrics`
endpoint reports aggregate request, failure, latency and probability counts.

## Limitations and risks

- Modest discrimination (ROC-AUC about 0.67): most readmissions are missed at
  any threshold that keeps false alarms manageable.
- The threshold is a policy choice; F1 was used as a simple default.
- The dataset has no dates, so performance on future patients is not measured.
- One dataset and period (US, 1999–2008); results do not transfer without
  re-validation.
- The model must not be used for patients who died in hospital or were
  discharged to hospice.
- A risk score could be misused to deny care; any real use would need clinical
  validation, governance and human review.
