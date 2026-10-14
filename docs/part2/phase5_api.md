# Part 2 – Phase 5: FastAPI Inference

This phase serves the registered Phase 3 Random Forest through a small
FastAPI service: one encounter's 39 features in, the predicted probability of
readmission within 30 days and a high-risk flag out.

> Academic decision-support prototype. The model is not clinically validated
> and must not be used for real patient decisions.

Nothing is trained or re-evaluated. Part 1, the Phase 2–4 code and the
registered model (version 1, alias `@candidate`) are unchanged.

## How to run

From the project root, after `pip install -r requirements-ml.txt` and the
Phase 4 steps (`python -m src.ml.log_phase3_to_mlflow`,
`python -m src.ml.register_model`):

```bash
uvicorn src.api.main:app                      # http://127.0.0.1:8000/docs
python -m pytest tests/test_api.py            # 61 tests, about a minute
```

`READMISSION_TRACKING_URI` can point the service at another MLflow store;
by default it uses the project store, `sqlite:///<project root>/mlflow.db`.

## Endpoints

| Method | Path | Success | Failure |
|---|---|---|---|
| GET | `/health` | 200 `{"status": "ok", "model_loaded": true, "model_version": "1"}` | 503 with the reason |
| GET | `/model-info` | 200: name, version, alias, threshold, `model_sha256`, `serialization_format`, positive class, the 39 feature names, disclaimer | 503 |
| POST | `/predict` | 200 (below) | 422 invalid request, 503 model unavailable |

Example `/predict` response:

```json
{
  "readmission_probability": 0.0538,
  "high_risk": false,
  "threshold": 0.1617830043681171,
  "model_name": "hospital-readmission-30d-classifier",
  "model_version": "1",
  "disclaimer": "Academic decision-support prototype; ..."
}
```

`high_risk` is `readmission_probability >= 0.1617830043681171`, the same rule
Phase 3 used when it chose the threshold on the validation set
(`src/ml/metrics.py`). The response never contains feature values.

## Request schema

The body is one JSON object with exactly the 39 model features
(`config.FEATURE_COLUMNS`). Every field is required, `null` is not accepted,
and types are strict (no conversion of `"5"` to `5`). The field names are the
model's column names, including `glyburide-metformin`, `glipizide-metformin`
and `metformin-pioglitazone`.

| Fields | Type | Allowed values |
|---|---|---|
| `age_years` | number | a 10-year age band midpoint: 5, 15, …, 95 |
| `time_in_hospital` | integer | 1–14 (the dataset's definition) |
| `medication_count` | integer | 0–23 (number of medication columns) |
| `num_lab_procedures`, `num_procedures`, `num_medications`, `number_outpatient`, `number_emergency`, `number_inpatient`, `number_diagnoses` | integer | 0–1000 (non-negative, sanity cap) |
| 29 categorical features | string | the values in `src/api/allowed_values.py` |

The type of each field matches the registered model's signature: `double` for
`age_years`, `long` for the other nine numeric fields, `string` for the 29
categorical fields. The service checks at startup that the signature's input
names and order equal `config.FEATURE_COLUMNS`.

**Categorical values.** `src/api/allowed_values.py` was generated from the
project data, not written by hand. It holds every value that occurs in the
Phase 2 modelling cohort after `load_modelling_data()` and
`build_feature_frame()`, the same rules used for training. Examples:
`medical_specialty` has 73 values, `diag_*_group` 20 and `admission_source_id` 17.
Missing values follow the training convention. A missing or "Unknown"
specialty, diagnosis or other category is sent as `"Missing"`, and a lab test
that was not measured as `"None"`. Raw `"Unknown"` and `null` are rejected.

Five cohort values never occurred in the training split. Each is a real
cohort record (99,340 encounters) that fell in validation or test. They are
accepted because they are valid values of the cohort. The saved model's
encoder (`OneHotEncoder(handle_unknown="infrequent_if_exist",
min_frequency=50)`, fitted on training only) handles them as follows. This
was checked on the saved model, which was not changed or retrained.

| Value | Cohort rows (split) | Encoding in the saved model | Effect on the prediction |
|---|---|---|---|
| `medical_specialty` `Perinatology` | 1 (validation) | the "infrequent" specialty column, shared with 39 rare specialties | same as any other rare specialty; that column is used by the trees |
| `medical_specialty` `Psychiatry-Addictive` | 1 (validation) | same as above | same as above |
| `admission_source_id` `25` | 2 (validation, test) | the "infrequent" admission-source column, shared with 8, 10, 11, 13, 14, 22 | same as those codes; the trees never split on that column |
| `tolazamide` `Up` | 1 (validation) | the "infrequent" tolazamide column, shared with `Steady` | none: the trees never split on any tolazamide column |
| `metformin-pioglitazone` `Steady` | 1 (validation) | all zeros: training had only `No`, so there is no infrequent column | none: the single `metformin-pioglitazone_No` column has importance 0 and is never split on |

On 200 validation rows, swapping in each of these values (for example `No`
to `Steady`) changed the predicted probability by at most 3e-16, which is
floating-point noise. Two tests check the list. One compares it with the
cohort values. The other checks that every category the saved model learned
is accepted.

**Rejected inputs (422):**

- any field not in the 39: identifiers (`encounter_id`, `patient_nbr`,
  `patient_sk`), `race`, `payer_code`, the target, and so on;
- a missing field, `null`, a wrong type or an out-of-range number;
- a category that is not in the allowed list;
- `discharge_disposition_id` 11, 19, 20, 21 (expired) or 13, 14 (hospice).
  These encounters are outside the model's cohort (Phase 2) and get their own
  message: "Expired or hospice discharge codes (11, 13, 14, 19, 20, 21) are
  outside the model's cohort; no prediction is made for them."

Validation errors list the field (`loc`), a message and an error type, but
never the submitted value:

```json
{"detail": [{"loc": ["body", "race"], "msg": "Extra inputs are not permitted",
             "type": "extra_forbidden"}]}
```

A request is validated before the model is used, so an invalid request gets
422 even when the model is unavailable.

**Gender and race.** Gender is one of the 39 model features and is required.
Its value is never logged or returned. Race is not a model feature and is
rejected. (The Phase 2 report was corrected to say this.)

## Model loading and verification

At startup the service resolves `hospital-readmission-30d-classifier@candidate`
once and loads it by its version number, so the alias cannot move between the
checks and the load. Predictions are enabled only if every check passes:

| Check | Expected |
|---|---|
| Version status | `READY` |
| Storage format | `skops` in the MLmodel sklearn flavour. The model is loaded with `mlflow.sklearn.load_model`, which uses skops and the four trusted types recorded in Phase 4, never pickle. Any other format is refused. |
| Signature | inputs equal the 39 `config.FEATURE_COLUMNS` in order; output is the `(-1, 2)` probability array |
| Threshold | version tag `selected_threshold` = MLmodel metadata `selected_threshold` = `0.1617830043681171` |
| Provenance | `model_sha256` in the version tag, the MLmodel metadata and the registration run's tag all equal the trusted Phase 3 hash `1d17bf4ac345891a58b4566ca5b3265798fea2cc73aa2dde860015dfc25062f9`; version tagged `record_type=phase3_model_registration`, `retrained=false` |
| Loaded object | binary classifier with `predict_proba` and classes `[0, 1]` |

If the store, model or alias is unavailable, or any check fails, the service
still starts. `/health`, `/model-info` and `/predict` then return 503 with the
reason, for example "Model unavailable: model provenance hash does not match
the trusted Phase 3 model". It never predicts with an unverified model.

The trusted hash and threshold are constants in `src/api/model.py`. Serving
a different model version needs a deliberate code change, not only a registry
change.

## Privacy

- The service writes no logs of its own. Request bodies and feature values
  are never logged or returned, and validation errors leave out the submitted
  value. A test checks the response and the captured logs.
- Uvicorn's access log records only the method, path and status code.

## Code

| File | Purpose |
|---|---|
| `src/api/main.py` | FastAPI app (`create_app`), endpoints, 422 handler without input values, one-row DataFrame with training dtypes |
| `src/api/model.py` | Settings, registry lookup, verification (`verify`), skops loading |
| `src/api/schemas.py` | Request model built from `config.FEATURE_COLUMNS`; response models |
| `src/api/allowed_values.py` | Allowed categorical values, generated from the cohort |
| `tests/test_api.py` | 61 tests |

Dependencies, pinned in `requirements-ml.txt`: `fastapi==0.143.0` and
`uvicorn==0.52.3` (the versions already installed with MLflow, not upgraded),
and `httpx==0.28.1` for FastAPI's test client. `pip check` reports no broken
requirements.

## Tests (61)

`tests/test_api.py` registers small synthetic calibrated Random Forests on
the 39 real column names in a temporary SQLite store, the way Phase 3 and 4
do. It does not touch the real registered model. It checks:

- **valid requests:** `/health` and `/model-info`; a probability in [0, 1];
  the flag uses `>=` the threshold; the API's probability equals the loaded
  model's `predict_proba`; an integer age is accepted
- **no leakage:** no feature values (specialty, gender) in responses or
  captured logs
- **422:** missing field; extra fields (`race`, `encounter_id`, `patient_nbr`,
  `payer_code`, `readmitted`), with the value not echoed; wrong types (string
  or float for an integer, a boolean, a string age, a non-midpoint age, an
  integer code, `null`); out-of-range numbers; invalid categories (including
  raw `"Unknown"`); each of the six excluded discharge codes; malformed JSON
- **503:** missing registered model, missing alias, untrusted hash, wrong
  threshold, loader failure; an invalid request still gets 422 when the model
  is unavailable
- **`verify()` one check at a time:** not READY, a non-skops or missing
  format, reordered or missing features, wrong output shape, missing or wrong
  threshold, provenance hash mismatch or missing run tag, version not
  labelled as the unchanged Phase 3 model
- **against the project files**, skipped when they are absent: allowed values
  equal the cohort values (73 specialties); every category in the saved
  model is allowed (the joblib file is loaded only after its SHA-256 check);
  the real `@candidate` model is served read-only with version 1, the trusted
  hash and the threshold

## Limitations

- **One encounter per request.** There is no batch endpoint.
- **Inputs are model features, not raw records.** Callers send the 39
  engineered features (age band midpoint, diagnosis groups, medication
  count), not the raw Part 1 columns.
- **The skops file itself is not hashed.** Provenance rests on the hashes
  recorded in the registry, which were checked against the Phase 3 joblib
  file in Phase 4, and on the local MLflow store being trusted.
- **Test client deprecation.** Starlette 1.3.1 warns that using `httpx` with
  its test client is deprecated in favour of `httpx2`. The pinned
  `httpx==0.28.1` works; the warning appears once per test run.
- **Local only.** No authentication, rate limiting or HTTPS; not for
  deployment as is.
