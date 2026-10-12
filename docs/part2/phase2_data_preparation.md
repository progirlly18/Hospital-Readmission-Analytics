# Part 2 – Phase 2: Feature Engineering and Data Splitting

This phase builds the modelling dataset, the features and the patient-grouped
train/validation/test split, then runs the pre-training checks.
**No model is trained in this phase.**

Part 1 files (ETL scripts, SQL, DAG, dashboard) are unchanged; the only edit to
an existing file is the new Part 2 section appended to `.gitignore`.

## How to run

From the project root, with the Part 1 environment activated:

```bash
pip install -r requirements-ml.txt
python -m src.ml.prepare_data     # builds cohort, features, split, runs checks
python -m pytest                  # 53 tests
```

`prepare_data` writes (gitignored, regenerated each run):

- `artifacts/data_prep/split_assignments.csv`: `encounter_id`, `patient_sk`, `split`
- `artifacts/data_prep/data_prep_report.json`: every number below

It exits with an error if any pre-training check fails.

## Source and target

- Source: `data/cleaned/diabetic_data_cleaned.csv` (Part 1 cleaning output,
  101,766 encounters). It is used instead of PostgreSQL because the database
  tables do not keep the medication, A1C, glucose and specialty columns.
- Target `readmitted_30d`: **1 = `<30`** (readmitted within 30 days),
  **0 = `>30` or `NO`**. Same rule as Part 1's `readmitted_within_30_days`.

## Modelling cohort

| Step | Encounters removed | Of which `<30` |
|---|---:|---:|
| Source rows | 101,766 (start) | 11,357 |
| Expired (discharge disposition 11, 19, 20, 21) | 1,652 | 0 |
| Hospice (discharge disposition 13, 14) | 771 | 43 |
| Gender `Unknown/Invalid` | 3 | 0 |
| **Modelling cohort** | **99,340** | **11,314 (11.39%)** |

**What this means for who the model applies to.** The model is only intended
for encounters where the patient is discharged alive and not to hospice care.
It must not be used to score patients who died in hospital or who are
discharged to hospice. Expired encounters are always `NO` by definition, so
including them would teach the model a trivial rule. Hospice patients receive
end-of-life care, where readmission risk scores are not a meaningful decision
aid. The prediction API (Phase 5) is planned to reject these discharge codes.
The positive rate rises from 11.16% to 11.39% because the removed encounters
were mostly negative.

## Features (39) and prediction time

Assumed prediction time: **at discharge**. Each feature has a documented
availability reason in `src/ml/config.py` (`PREDICTION_TIME_AVAILABILITY`),
and a check fails if any feature lacks one.

| Type | Features |
|---|---|
| Numeric (10) | `age_years` (age band midpoint), `time_in_hospital`, `num_lab_procedures`, `num_procedures`, `num_medications`, `number_outpatient`, `number_emergency`, `number_inpatient` (all three: visits in the year *before* the stay), `number_diagnoses`, `medication_count` |
| Categorical (29) | `gender`, `admission_type_id`, `discharge_disposition_id`, `admission_source_id`, `medical_specialty`, `diag_1_group`, `diag_2_group`, `diag_3_group`, `max_glu_serum`, `A1Cresult`, `change`, `diabetesMed`, 17 diabetes medications |

**Excluded on purpose:** identifiers (`encounter_id`, `patient_nbr`,
`patient_sk`), the target, `race` (kept only for subgroup analysis),
`payer_code`, `weight` (97% missing), the six constant or near-constant
medications, and every Part 1 `patient_summary` column (they aggregate a
patient's *later* encounters too).

Missing lab results (`A1Cresult`, `max_glu_serum`) become the category
`None` ("test not performed"), which is informative in itself.

### Diagnosis features, verified independently

Part 1's `categorize_diagnosis()` removes the decimal point before comparing
code ranges, so `250.83` becomes `25083` and is grouped as `Other`. On the real
data this affects **all 8,522 primary diagnoses that contain a decimal point**,
including most diabetes codes (only 235 of 8,757 `250.x` primary diagnoses
reach `Endocrine / Metabolic`). As agreed, Part 1 is **not** changed.

Part 2 uses its own `map_icd9_group()` in `src/ml/features.py`:

- ICD-9-CM chapter ranges compared on the full numeric value (`250.83` stays 250.83)
- `250.xx` gets its own `Diabetes` group
- `V` and `E` codes get their own groups; missing becomes `Missing`
- 19 unit tests cover integer, decimal, V, E and missing codes

## Preprocessing without leakage

Preprocessing is split into two steps:

1. `build_feature_frame()`: row-by-row rules only (age midpoint, ICD-9
   grouping, medication count, filling in categories). One encounter's features
   never depend on other rows, and a test confirms this.
2. `build_preprocessor()`: median imputation, standard scaling and one-hot
   encoding. Categories seen fewer than 50 times *in training* are grouped, and
   categories never seen in training are handled at prediction time. **It is
   fitted on the training split only**, and the pipeline checks that the
   learned medians and means equal the training statistics.

## Split

There are no dates in the dataset, so a chronological split is not possible.
A **patient-grouped, stratified** split is used instead (`src/ml/split.py`):

1. Every patient gets a fixed pseudo-random position from a SHA-256 hash of
   `"42:<patient_sk>"`.
2. Patients are split separately in two strata: with at least one `<30`
   encounter, and without.
3. Within each stratum, patients fill train up to 70% of the encounters, then
   validation up to 85%, then test.

Whole patients are moved, so the proportions are approximate. They are not
forced.

`sklearn.model_selection.StratifiedGroupKFold` was tried first and rejected:
scikit-learn 1.5.2 and 1.9.1 produced **different** splits from the same seed.
The hash-based method gave byte-identical output on both
(pandas 2.2.3 / scikit-learn 1.5.2 and pandas 3.0.6 / scikit-learn 1.9.1).

### Actual split (seed 42)

| Split | Encounters | Share | Patients | Positives | Negatives | Positive rate |
|---|---:|---:|---:|---:|---:|---:|
| Train | 69,538 | 70.00% | 49,013 | 7,888 | 61,650 | 11.34% |
| Validation | 14,901 | 15.00% | 10,497 | 1,722 | 13,179 | 11.56% |
| Test | 14,901 | 15.00% | 10,477 | 1,704 | 13,197 | 11.44% |
| **Total** | **99,340** | | **69,987** | **11,314** | **88,026** | **11.39%** |

**Patient overlap: train/validation 0, train/test 0, validation/test 0.**

Split fingerprint (SHA-256 of `split_assignments.csv`):
`45788b051270260de0dc1bb3eefc80b81f94b76fccfa0d280449631f388aa941`.
If you run `prepare_data` locally and get the same fingerprint, your split is
identical to this one.

### Class imbalance

About 7.8 negatives per positive in training. Nothing is resampled in this
phase. In Phase 3, imbalance will be handled with class weights on the
**training data only**; validation and test keep their natural distribution.

## Pre-training checks (actual output)

```text
[PASS] No identifier, target or excluded columns in features: none found
[PASS] Every feature is documented as available at discharge: 39 features documented
[PASS] No single feature has training ROC-AUC above 0.8:
       top 5: number_inpatient 0.608, discharge_disposition_id 0.578,
       time_in_hospital 0.548, number_diagnoses 0.544, num_medications 0.542
[PASS] Zero patients shared between splits
[PASS] Preprocessing statistics learned from training rows only:
       fitted on 69,538 rows, 200 output columns
```

The strongest single feature reaches a training ROC-AUC of only 0.61, which is
consistent with no feature directly revealing the outcome.

## Subgroups (for fairness analysis; race is not a model feature)

Race and gender are kept beside the features, never inside them. Privacy
precautions: they are used only in aggregate, groups with fewer than 30
encounters are suppressed in reports, and they will never be accepted or
logged by the API.

| Race | Train enc. (pos.) | Validation enc. (pos.) | Test enc. (pos.) |
|---|---|---|---|
| Caucasian | 52,023 (5,992) | 11,021 (1,271) | 11,176 (1,293) |
| AfricanAmerican | 13,122 (1,500) | 2,866 (342) | 2,784 (307) |
| Hispanic | 1,393 (145) | 307 (26) | 317 (41) |
| Unknown | 1,565 (123) | 346 (36) | 321 (29) |
| Other | 1,016 (95) | 240 (24) | 215 (25) |
| Asian | 419 (33) | 121 (23) | 88 (9) |

| Gender | Train enc. (pos.) | Validation enc. (pos.) | Test enc. (pos.) |
|---|---|---|---|
| Female | 37,465 (4,272) | 7,957 (909) | 8,032 (947) |
| Male | 32,073 (3,616) | 6,944 (813) | 6,869 (757) |

Small groups (Asian, Other, Hispanic, Unknown) have fewer than 50 positives
in validation and test, so their subgroup metrics will be statistically
unstable. Phase 3 will report confidence intervals or flag them.

## Tests

`python -m pytest`: **53 passed** on both environments above.

| File | What it covers |
|---|---|
| `tests/test_target_and_cohort.py` | Target mapping, invalid labels, cohort exclusions and counts |
| `tests/test_features.py` | ICD-9 grouping (incl. decimals), age, feature list, no forbidden columns, race/payer/target do not affect features, row-wise construction, train-only fitting, unseen categories, missing values |
| `tests/test_split_and_checks.py` | Zero overlap, proportions, reproducibility, seed sensitivity, row-order independence, positive-rate balance, leak detection on a planted target copy |
| `tests/test_real_data.py` | On the real CSV: exact cohort counts, target equals Part 1 rule, all checks pass, split positive rates within 1 point (skipped if the CSV is missing) |

## Limitations

- No dates, so the split is not chronological, and performance on *future*
  patients is not directly measured.
- A patient's earlier encounters are in the same split as their later ones,
  so the history counts (`number_inpatient` etc.) come from the dataset
  itself, not from a separate history.
- The model is limited to the cohort above (alive, non-hospice discharges,
  diabetic inpatients in 130 US hospitals, 1999–2008).
