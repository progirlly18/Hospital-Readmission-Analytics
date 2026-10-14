# Part 2 – Phase 4: MLflow Experiment Tracking and Model Registry

This phase records the Phase 3 model comparison in MLflow: parameters,
validation metrics for all three models, the selected model's held-out test
metrics, provenance, and the saved tables and plots. It then logs the saved
Random Forest as an MLflow model and registers it, with its decision
threshold stored on the model version.

> Academic decision-support prototype. These results are not a clinically
> validated model and must not be used for real patient decisions.

**Scope:** experiment tracking (first increment) and model registration
(second increment, see [Model registry](#model-registry)). Part 1, the
Phase 2 data preparation and the Phase 3 training code are unchanged.

## Historical results, not a new training run

The runs are logged from the files that `python -m src.ml.train` already
wrote to `artifacts/phase3/`. The logging script trains nothing, re-evaluates
nothing and does not import the training code (a test checks this).

Logging these files does **not** show that training reproduces them. Phase 3
documents that a repeated full run gave different XGBoost results (about
1e-3); the logged values are those of the single saved run.

Every run carries these tags so this is visible in the MLflow UI:

| Tag | Value |
|---|---|
| `record_type` | `historical_phase3_results` |
| `retrained` | `false` |
| `historical_note` | Results read from saved files; no model was trained, re-trained or re-evaluated |
| `source_phase` | `phase3` |
| `metrics_json_sha256` | SHA-256 of the `metrics.json` that was read |
| `split_fingerprint_sha256` | Phase 2/3 split fingerprint from `metrics.json` |

## How to run

From the project root, after `pip install -r requirements-ml.txt`:

```bash
python -m src.ml.log_phase3_to_mlflow                       # log the saved results
python -m src.ml.register_model                             # log + register the saved Random Forest
mlflow ui --backend-store-uri sqlite:///mlflow.db           # browse at http://127.0.0.1:5000
python -m pytest tests/test_tracking.py tests/test_registry.py   # 31 tests, about 3 minutes
```

The same saved results are logged only once per experiment: the script looks
for a finished run with the same `metrics.json` SHA-256 and stops with
"Nothing logged" if it finds one. `--allow-duplicate` overrides this.
Missing or incomplete Phase 3 files are reported before any run is created.

## Configuration

| Setting | Value |
|---|---|
| MLflow | `mlflow==3.17.1` (`requirements-ml.txt`) |
| Tracking store | SQLite, `sqlite:///<project root>/mlflow.db` |
| Artifact store | Local folder, `<project root>/mlartifacts/<experiment>/` |
| Experiment | `hospital-readmission-30d` |

SQLite is used because the MLflow model registry requires a database backend. `mlflow.db`, `mlartifacts/` and `mlruns/` are
already gitignored. This configuration was checked on Windows 11 with
Python 3.11.9: database creation, nested runs, metrics, tags and artifacts
all work with `file://` artifact locations. Creating a new database takes
about 10–15 seconds the first time.

`mlflow ui` also starts on Windows and serves the runs. At start-up it logs a
warning that the MLflow job backend does not support Windows. That feature
is not used here, and tracking is not affected. Stop the UI with Ctrl+C in
its terminal: it runs several worker processes.

The settings live in `src/ml/tracking.py`. The script accepts
`--tracking-uri`, `--experiment` and `--phase3-dir` to log elsewhere.

## What is logged

```text
hospital-readmission-30d
└── phase3-historical-comparison           parent run
    ├── historical-validation-logistic_regression
    ├── historical-validation-random_forest
    ├── historical-validation-xgboost
    └── historical-heldout-test-random_forest
```

### Parent run: `phase3-historical-comparison`

- **Tags:** `selected_model`, `selection_reasons`, `selected_model_artifact`
  (`models/random_forest.joblib`), `selected_model_sha256` (SHA-256 of the
  saved joblib file), plus the shared tags above.
- **Parameters:** `random_seed`, `cv`, `positive_class`, train / validation /
  test sizes, `train_negatives_per_positive`, `pre_training_checks_passed`,
  `selected_model`.
- **Artifacts:**
  - `tables/`: `model_comparison.csv`, `subgroups_validation.csv`,
    `subgroups_test.csv`, `permutation_importance.csv`, `metrics.json`
  - `cv_results/`: `cv_results_<model>.csv` for all three models
  - `plots/`: every PNG in `artifacts/phase3/plots/`

### Model runs: `historical-validation-<model>` (validation only)

- **Tags:** `data_split=validation`, `model_name`, `selected` (`true` only
  for Random Forest).
- **Parameters:** the best grid-search setting (e.g. `model__max_depth`),
  `selected_threshold`, `threshold_rule`, `calibration`.
- **Metrics:**
  - `cv_roc_auc_mean`, `cv_roc_auc_std`, `cv_pr_auc_mean`, `training_seconds`
  - `validation/roc_auc`, `validation/pr_auc`
  - calibration: `validation/brier`, `validation/ece`,
    `validation/mean_predicted_probability`, `validation/observed_positive_rate`
  - `validation/at_0_5/…` and `validation/at_selected/…`: `precision`,
    `recall`, `f1`, `predicted_positive_rate`, `threshold` and the
    confusion-matrix counts (`confusion_matrix/tn`, `fp`, `fn`, `tp`)
  - `validation_uncalibrated/…`: the same model before calibration
- **Artifact:** that model's `cv_results/cv_results_<model>.csv`.

These runs hold no test-set metrics, so comparing them in the MLflow UI
compares validation results only.

### Held-out test run: `historical-heldout-test-random_forest`

- **Tags:** `data_split=test`, `evaluation` ("held-out test set, selected
  model only, evaluated once in Phase 3"), `model_name`,
  `selected_model_sha256`.
- **Parameters:** `selected_threshold` and `threshold_source` (the
  validation set).
- **Metrics:** `test/…` with the same structure as the validation metrics.
- **Artifacts:** `test/subgroups_test.csv`, `test/confusion_test_selected.png`,
  `test/reliability_test_selected.png`.

The per-class `classification_report` blocks in `metrics.json` are not logged
as metrics. They repeat precision, recall and F1, and their labels contain
characters MLflow does not allow in metric names. They remain available in
the logged `tables/metrics.json`.

## Code

| File | Purpose |
|---|---|
| `src/ml/tracking.py` | Tracking store and experiment setup; flattening nested metrics into MLflow names; parameter and tag logging that fails instead of truncating; file logging that fails on a missing file; SHA-256 helper |
| `src/ml/log_phase3_to_mlflow.py` | One-off script that checks and logs the saved Phase 3 outputs |
| `src/ml/register_model.py` | Checks the saved Random Forest's SHA-256 against the Phase 3 run, logs it with the sklearn flavour, registers it, tags the version, and checks the registry copy against the joblib file |
| `tests/test_tracking.py` | 13 tests on a temporary SQLite store with small synthetic metrics |
| `tests/test_registry.py` | 18 tests on temporary SQLite stores with a small synthetic calibrated Random Forest |

## Model registry

`python -m src.ml.register_model` logs the existing
`artifacts/phase3/models/random_forest.joblib` (a sigmoid
`CalibratedClassifierCV` around the Random Forest pipeline) **unchanged**
with the MLflow sklearn flavour, and registers it. Nothing is trained or
re-evaluated, and the script does not import the training code (a test
checks this).

| Item | Value |
|---|---|
| Run | `register-phase3-random_forest` in `hospital-readmission-30d` |
| Registered model | `hospital-readmission-30d-classifier` |
| Version | `1`, alias `candidate` |
| Flavours | `sklearn` (skops format) and `python_function` |
| pyfunc output | `predict_proba` (column 1 = P(readmitted <30 days)) |
| Signature | 39 feature columns in, probability array out, inferred from real validation rows |

**Decision threshold.** The threshold chosen on validation in Phase 3 is read
from `metrics.json` (`models.random_forest.selected_threshold`) and stored,
unrounded, in three places:

- model version tag `selected_threshold` = `0.1617830043681171`
  (with `threshold_source` = "validation set, maximise F1 (Phase 3)");
- the MLmodel `metadata` (`selected_threshold`, `model_sha256`);
- the registration run's parameter `selected_threshold`.

The model returns probabilities; applying the threshold is up to the caller.

**Provenance tags on the version:** `model_sha256` (SHA-256 of the joblib
file), `metrics_json_sha256`, `split_fingerprint_sha256`, `model_class`,
`retrained=false`, `record_type=phase3_model_registration`,
`test_roc_auc_phase3` (the saved Phase 3 value, not recomputed) and a
`disclaimer`. The same joblib file (same SHA-256) is registered only once;
`--allow-duplicate` registers it again as a new version.

**Model file hash check (fail closed).** `joblib.load` unpickles the file and
can run code from it, so the script loads it only after checking its hash:

1. It looks up the active, `FINISHED` `phase3-historical-comparison` run
   logged by `log_phase3_to_mlflow` for the **same** `metrics.json`
   (matched on `metrics_json_sha256`) and reads its `selected_model_sha256`.
2. It reads `random_forest.joblib` once, computes the SHA-256 of those bytes
   and unpickles the same bytes only if the hash matches.

If no such run exists, the tag is missing, several runs disagree, or the hash
differs, the script stops with "Nothing registered" before anything is
unpickled, logged or registered. The duplicate check also uses the trusted
hash, so an already registered file is not unpickled either. New versions
carry the tag `model_sha256_verified_against`.

Version 1 was registered before this check was added. Its file was checked
afterwards: the joblib file's SHA-256, the version's `model_sha256` tag and the
Phase 3 run's `selected_model_sha256` are all
`1d17bf4ac345891a58b4566ca5b3265798fea2cc73aa2dde860015dfc25062f9`.

**Registry consistency check.** After registration the script loads
`models:/hospital-readmission-30d-classifier/1` back with both
`mlflow.sklearn.load_model` and `mlflow.pyfunc.load_model`, and compares
`predict_proba` with the joblib model on 500 real feature rows sampled from
the Phase 3 validation split (`split_assignments.csv`, seed 42). Only
probabilities are compared; no metric is computed. Result on 14 Oct 2026:
`np.allclose` true for both loaders, maximum absolute difference 3.3e-16.
A separate check in a fresh process through the alias
(`models:/hospital-readmission-30d-classifier@candidate`) on 1,000 other
validation rows also gave `np.allclose` true (max difference 3.9e-16). The
result is recorded on the version as the tag `registry_load_check`.

**skops trusted types.** MLflow 3.17 saves sklearn models with skops, which
only loads types it is told to trust. The script lists exactly the four
types inside the project's own model: `numpy.dtype`,
`sklearn.calibration._CalibratedClassifier`,
`sklearn.calibration._SigmoidCalibration` and `sklearn.tree._tree.Tree`.
Without this, `log_model` stops with an "untrusted types" error. Do not
extend this list for model files from outside the project.

## Tests (31)

### `tests/test_tracking.py` (13)

`tests/test_tracking.py` uses a temporary SQLite store and a synthetic
`metrics.json` built with the Phase 3 metric functions, so it has the same
structure as the real file. It does not use the real dataset. It checks:

- metric flattening and name cleaning
- missing files and missing `metrics.json` keys are reported before any run
- the logging script does not import `src.ml.train`
- one parent run with three model runs and one test run
- every run is labelled historical
- provenance tags equal the SHA-256 of the files that were read
- the required validation metrics per model, with no test metrics in them
- test metrics only in the labelled test run
- the expected artifacts are present
- the tracking script itself logs and registers no model
- the same saved results are not logged twice

### `tests/test_registry.py` (18)

Uses a temporary SQLite store and a tiny sigmoid-calibrated Random Forest
pipeline fitted on synthetic rows inside the test (a fixture, not a project
model). It does not use the real dataset or the real model file. It checks:

- the threshold is read exactly from `metrics.json`; missing or out-of-range
  thresholds and a missing model file are rejected
- the registration script does not import `src.ml.train`
- version 1 is created, `READY`, linked to the run, with alias `candidate`
- the threshold is stored exactly on the version tag and in the MLmodel metadata
- provenance tags equal the SHA-256 of the files that were read
- the registry copy (by alias and by version, sklearn and pyfunc) gives the
  same `predict_proba` as the joblib model (`np.allclose`)
- the same model file is not registered twice
- hash check, fail closed: when the Phase 3 run's `selected_model_sha256`
  does not match, is missing, or no Phase 3 run exists for this
  `metrics.json`, registration fails, `joblib.load` is never called (the test
  makes it raise), and no registration run or registered model is created
- a matching hash permits the normal workflow (the registration tests above
  run with a stand-in Phase 3 run that records the file's real hash)

## Limitations

- **Historical record only.** The runs describe one saved Phase 3 run. MLflow
  start and end times are the time of logging, not of training;
  `training_seconds` is the value saved by Phase 3.
- **Only the selected model was saved.** Logistic Regression and XGBoost have
  parameters, metrics and CV results in MLflow, but no model files, because
  Phase 3 did not save them.
- **Registration is not validation.** Registering the model records and
  versions the saved file; it does not re-check its performance. The alias
  `candidate` means "registered Phase 3 selection", not "approved for use".
- **Integer columns in the signature.** The signature was inferred from real
  rows, where count columns are integers. MLflow warns that a missing value
  in such a column at inference time would fail schema enforcement in the
  pyfunc loader. The Phase 2 features have no missing counts, so this does
  not affect the current data.
- **Failed first registration run, soft-deleted.** The first attempt on
  14 Oct 2026 stopped on the skops "untrusted types" error before any model
  was registered. Its run `register-phase3-random_forest`
  (`f944a5aa5e954b6cb4130d5f865eea37`, status `FAILED`) held no model version
  and no artifacts. It was soft-deleted (`lifecycle_stage = deleted`): it is
  no longer active and does not appear in the default UI view, but it is
  still in `mlflow.db` and was not purged. The experiment has 6 active runs
  (5 historical + 1 successful registration).
- **Hash trust comes from the local store.** The trusted hash is read from
  the local `mlflow.db`. Someone who can change both the model file and
  `mlflow.db` (or `metrics.json` and the run) can defeat the check; it
  protects against a replaced or corrupted model file, not against full
  control of the project folder.
- **Local store.** `mlflow.db` and `mlartifacts/` are local and gitignored;
  each machine logs its own copy from its own `artifacts/phase3/`.
