# Part 2 – Phase 3: Model Training and Evaluation

This phase trains and compares the three required models (Logistic
Regression, Random Forest, XGBoost), selects one candidate on the validation
set, and evaluates **only that candidate, once,** on the test set.

> Academic decision-support prototype. These results are not a clinically
> validated model and must not be used for real patient decisions.

MLflow, model registry, API, Docker, monitoring and dashboard work belong to
later phases and are not part of this phase. Part 1 and the Phase 2 cohort,
features, target and split are unchanged.

## How to run

From the project root:

```bash
pip install -r requirements-ml.txt
python -m src.ml.train            # full run: about 8 minutes on 4 CPU cores
python -m src.ml.train --quick    # smoke run: one setting per model
python -m pytest                  # 78 tests, about 1 minute
```

The run rebuilds the Phase 2 data, stops if any pre-training check fails, and
prints the split fingerprint (expected `45788b05…aa941`). Outputs go to
`artifacts/phase3/`, which is gitignored:

| File | Contents |
|---|---|
| `metrics.json` | All metrics, best parameters, thresholds, selection reasons, test results |
| `model_comparison.csv` | One row per model, validation metrics |
| `cv_results_<model>.csv` | Every grid-search setting with its cross-validation scores |
| `subgroups_validation.csv`, `subgroups_test.csv` | Metrics by gender, age group and race (selected model) |
| `permutation_importance.csv` | Feature importance (selected model) |
| `plots/*.png` | Reliability, ROC/PR curves, confusion matrices, importance |
| `models/random_forest.joblib` | Selected pipeline (26 MB, not committed) |

## Approach

### Same data for every model

All three models use the 39 Phase 2 features and the same patient-grouped
split: train 69,538, validation 14,901 and test 14,901 encounters, with zero
patient overlap.

### Pipelines and preprocessing

Each model is one sklearn `Pipeline`: the Phase 2 preprocessor (imputation,
one-hot encoding, plus scaling for Logistic Regression only) followed by the
classifier. The whole pipeline is fitted on training rows only. A test checks
that the fitted scaler's means equal the training means and that it saw
exactly the training row count.

### Class imbalance (training data only)

There are 7.8 negatives per positive in training. Nothing is resampled.

- Logistic Regression: `class_weight="balanced"`
- Random Forest: `class_weight="balanced_subsample"`
- XGBoost: `scale_pos_weight` = training negatives ÷ training positives

### Hyperparameter search (small, documented)

`GridSearchCV` uses **patient-grouped 5-fold cross-validation** (`GroupKFold`
by `patient_sk`) on the training split only. The best setting has the highest
mean CV ROC-AUC; PR-AUC is also recorded.

| Model | Fixed settings | Grid | Best |
|---|---|---|---|
| Logistic Regression | `max_iter=2000` | `C` ∈ {0.01, 0.1, 1.0} | `C=0.01` |
| Random Forest | 200 trees, `max_features="sqrt"` | `max_depth` ∈ {8, 16} × `min_samples_leaf` ∈ {10, 50} | depth 16, leaf 10 |
| XGBoost | 300 trees, learning rate 0.05, subsample 0.8, colsample 0.8, `hist` | `max_depth` ∈ {3, 5} × `min_child_weight` ∈ {1, 10} | depth 3, child weight 10 |

All seeds are 42. The same command gives the same results up to
floating-point rounding (about 1e-9) caused by multi-threaded training; the
tests allow a relative tolerance of 1e-6.

### Probability calibration (training data only)

Class weights make the models focus on the rare positive class, but they also
inflate the predicted probabilities. Before calibration, the average predicted
probability on validation was 0.43–0.47, against a real rate of 0.116.

Each tuned model is therefore wrapped in `CalibratedClassifierCV`
(sigmoid/Platt scaling, `ensemble=False`). It uses out-of-fold predictions
from the same patient-grouped folds on the **training split**, so validation
and test stay untouched. The calibration step is monotonic, so ROC-AUC and the
ranking of patients do not change, but the probabilities become meaningful.

| Validation | LR raw | LR calibrated | RF raw | RF calibrated | XGB raw | XGB calibrated |
|---|---:|---:|---:|---:|---:|---:|
| Mean predicted | 0.467 | 0.114 | 0.435 | 0.116 | 0.462 | 0.115 |
| Brier | 0.2259 | 0.0984 | 0.2015 | 0.0979 | 0.2230 | 0.0975 |
| ECE | 0.351 | 0.010 | 0.319 | 0.007 | 0.346 | 0.005 |

For reference, always predicting the training positive rate gives a Brier
score of about 0.102 on validation.

ECE (expected calibration error) is the size-weighted average gap between
mean predicted probability and observed rate across 10 equal-frequency bins.
Tied predictions always share a bin.

### Threshold selection (validation only)

For each model, the threshold that maximises F1 on the validation set is
selected. Metrics are reported at 0.5 and at that threshold. With calibrated
probabilities very few encounters exceed 0.5, so at 0.5 the models flag
almost nobody. A threshold is a policy choice; F1 is used as a simple,
documented default.

### Model selection rule (validation only)

Implemented in `src/ml/selection.py`:

1. Eligible models are within 0.01 of the best validation ROC-AUC **and**
   within 0.01 of the best validation PR-AUC.
2. Among eligible models, keep those within 0.002 of the lowest Brier score.
3. Choose the simplest: Logistic Regression, then Random Forest, then XGBoost.

The selection function only receives validation metrics. A test flips every
test-set label and confirms that the selected model, the thresholds and the
validation metrics are unchanged.

## Results

### Validation comparison (all three models, same 14,901 encounters, 1,722 positives)

| Metric | Logistic Regression | Random Forest | XGBoost |
|---|---:|---:|---:|
| CV ROC-AUC (train, mean ± sd) | 0.6635 ± 0.0060 | 0.6683 ± 0.0064 | 0.6709 ± 0.0054 |
| **Validation ROC-AUC** | 0.6618 | 0.6673 | **0.6695** |
| **Validation PR-AUC** | 0.2139 | 0.2192 | **0.2288** |
| Brier score | 0.0984 | 0.0979 | **0.0975** |
| ECE | 0.0099 | 0.0070 | **0.0047** |
| Precision / recall / F1 @ 0.5 | 0.535 / 0.022 / 0.042 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |
| Selected threshold | 0.1349 | 0.1618 | 0.1403 |
| Precision @ selected | 0.2108 | **0.2227** | 0.2108 |
| Recall @ selected | 0.4123 | 0.3984 | **0.4686** |
| F1 @ selected | 0.2790 | 0.2857 | **0.2908** |
| Confusion @ selected (TN / FP / FN / TP) | 10,521 / 2,658 / 1,012 / 710 | 10,784 / 2,395 / 1,036 / 686 | 10,157 / 3,022 / 915 / 807 |

PR-AUC should be compared with the base rate of 0.116, which is what a random
model would score.

### Selected candidate: Random Forest, threshold 0.1618

The rule's output:

```text
Best validation ROC-AUC 0.6695, best PR-AUC 0.2288.
Eligible (within 0.01 ROC-AUC and 0.01 PR-AUC of the best): random_forest, xgboost.
Lowest Brier among eligible 0.0975; within 0.002: random_forest, xgboost.
Selected the simplest of these: random_forest.
```

**Note for review:** Random Forest passed the PR-AUC rule by a narrow margin
(0.0096 behind XGBoost, against a tolerance of 0.01). XGBoost is slightly
better on every validation metric except precision. A slightly tighter
tolerance would have selected XGBoost. The rule was fixed before the results
were seen, and it was applied as written.

### Test set (selected model only, evaluated once)

14,901 encounters, 1,704 positives (11.44%).

| Metric | Value |
|---|---:|
| ROC-AUC | 0.6715 |
| PR-AUC | 0.2237 |
| Brier score | 0.0967 |
| ECE | 0.0046 |
| Mean predicted / observed rate | 0.1157 / 0.1144 |
| Precision @ 0.1618 | 0.2129 |
| Recall @ 0.1618 | 0.3815 |
| F1 @ 0.1618 | 0.2733 |
| Confusion @ 0.1618 (TN / FP / FN / TP) | 10,794 / 2,403 / 1,054 / 650 |
| At 0.5 | no encounter flagged (precision, recall and F1 = 0) |

The test results are close to validation (ROC-AUC 0.6715 vs 0.6673), so there
is no sign of overfitting to the validation set. In plain terms, at the chosen
threshold the model flags about 20% of encounters. It catches about 38% of
real 30-day readmissions, and about 1 in 5 flagged encounters is actually
readmitted.

### Subgroups (selected model, test set, threshold 0.1618)

Race is not a model input; it is used here only for fairness analysis.

| Group | Encounters | Positives | Observed | Mean predicted | ROC-AUC | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| Female | 8,032 | 947 | 0.118 | 0.118 | 0.671 | 0.392 | 0.214 |
| Male | 6,869 | 757 | 0.110 | 0.113 | 0.672 | 0.369 | 0.211 |
| Age <50 | 2,329 | 239 | 0.103 | 0.102 | 0.729 | 0.427 | 0.249 |
| Age 50–69 | 5,970 | 660 | 0.111 | 0.109 | 0.682 | 0.370 | 0.224 |
| Age 70+ | 6,602 | 805 | 0.122 | 0.126 | 0.635 | 0.378 | 0.196 |
| Caucasian | 11,176 | 1,293 | 0.116 | 0.117 | 0.673 | 0.380 | 0.213 |
| AfricanAmerican | 2,784 | 307 | 0.110 | 0.115 | 0.654 | 0.384 | 0.197 |
| Hispanic | 317 | 41 | 0.129 | 0.112 | 0.784 | 0.512 | 0.339 |
| Asian, Other, Unknown | 88 / 215 / 321 | 9 / 25 / 29 | | | | | unstable (<30 positives) |

Calibration holds across gender and age groups (mean predicted close to
observed). Ranking is weaker for patients aged 70+ (ROC-AUC 0.635). Small
racial groups have too few positives for reliable conclusions. Validation
subgroup results are in `subgroups_validation.csv`.

### Explainability: permutation importance (selected model, validation)

This measures how much validation ROC-AUC drops when one feature's values are
shuffled (5 repeats, on the 39 original features):

| Rank | Feature | ROC-AUC drop |
|---:|---|---:|
| 1 | `number_inpatient` (inpatient visits in the prior year) | 0.0528 |
| 2 | `discharge_disposition_id` | 0.0419 |
| 3 | `diag_1_group` | 0.0042 |
| 4 | `medical_specialty` | 0.0031 |
| 5 | `num_medications` | 0.0029 |
| 6 | `number_diagnoses` | 0.0028 |
| 7 | `time_in_hospital` | 0.0025 |
| 8 | `age_years` | 0.0021 |
| 9 | `number_emergency` | 0.0021 |
| 10 | `insulin` | 0.0015 |

Prior inpatient use and discharge destination dominate; every other feature
adds little on its own. Permutation importance shows association, not cause.
SHAP is planned for Phase 8.

## Tests added (22, total 78)

| File | What it covers |
|---|---|
| `tests/test_metrics.py` | ECE (perfect, overconfident, row-order independent), Brier/ROC-AUC known values, confusion matrix, best-F1 threshold reproduces its F1 and beats a fine grid, both thresholds reported |
| `tests/test_selection.py` | Clear winner, tie goes to the simplest model, calibration breaks ties, PR-AUC gap excludes a model, selection functions have no test-set parameters |
| `tests/test_training.py` | All three models trained on the same validation rows, valid probabilities, only the selected model scored on test, reproducibility, **flipping test labels does not change selection or thresholds**, preprocessing fitted on training rows only, class weights from training labels, calibration moves probabilities towards the observed rate, subgroup and importance outputs (race never a feature) |

Training tests use small synthetic data with one setting per model so they
run quickly. The full real-data run is the `python -m src.ml.train` command
above.

## Limitations

- **Modest discrimination.** ROC-AUC is about 0.67 and PR-AUC about 0.22. The
  model separates higher-risk from lower-risk encounters better than chance,
  but it misses most readmissions at any threshold that keeps false alarms
  manageable.
- **Threshold is a policy choice.** The F1-maximising threshold flags about
  20% of encounters. A real service would choose it based on the cost of
  missed readmissions against unnecessary follow-ups.
- **Selection was close.** Random Forest and XGBoost are practically tied;
  see the note above.
- **Grid edges.** The best Logistic Regression `C` (0.01) and Random Forest
  depth (16) are at the edges of their grids, so slightly better settings may
  exist outside the small search.
- **No chronological validation.** The dataset has no dates (see Phase 2), so
  performance on future patients is not directly measured.
- **Single dataset and period.** Diabetic inpatients in 130 US hospitals,
  1999–2008, excluding deaths and hospice discharges. Results do not transfer
  to other populations or eras without re-validation.
- **Small subgroups.** Metrics for the Asian, Other and Unknown race groups
  rest on fewer than 30 positives each and are unreliable.

## Dependencies added (`requirements-ml.txt`)

`xgboost==3.2.0`, `matplotlib==3.11.2`, `joblib==1.6.0`. All have Windows
wheels for Python 3.11. Tests and the full run were done on Python 3.11 with
exactly the pinned versions.
