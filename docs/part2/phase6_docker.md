# Part 2 – Phase 6: Docker

This phase packages the Phase 5 prediction API as a Docker image. The API
code is reused unchanged except for one addition: it can load the registered
model from an exported **model bundle** instead of the MLflow store.

> Academic decision-support prototype. The model is not clinically validated
> and must not be used for real patient decisions.

Nothing is trained or re-evaluated. Part 1 and the registered model
(version 1, alias `@candidate`) are unchanged.

## Why a model bundle

The MLflow store cannot be mounted into the container as it is.
`mlflow.db` records every artifact location as an absolute path on the
machine that created it, for example
`file:///D:/<project folder>/mlartifacts/...`. That
path does not exist inside a Linux container, and SQLite is not meant to be
shared between machines.

So the registered version is exported once, on the machine that has the
store, into a plain folder that is mounted **read-only** into the container:

```text
model_bundle/                 (gitignored)
    manifest.json             registry details and SHA-256 of every model file
    model/                    the registered version's MLflow files, unchanged
        MLmodel  model.skops  conda.yaml  python_env.yaml
        requirements.txt  registered_model_meta
```

The bundle contains the model only: no data, input examples or credentials
(about 32 MB, nearly all `model.skops`). `MLmodel` still holds MLflow's
informational `artifact_path`, which names the folder on the exporting
machine; it is not used for loading, and the file is left unchanged because
its hash is pinned in the manifest.

## How to run

1. Export the bundle (host, project environment, from the project root):

   ```bash
   python -m src.api.bundle                 # writes model_bundle/
   ```

   The export resolves `@candidate`, runs every Phase 5 check, downloads the
   version's files, refuses a `model.skops` that is not the trusted registered
   file, loads the result exactly as the API will, and only then writes
   `model_bundle/`. It refuses to overwrite a non-empty folder.

2. Build the image:

   ```bash
   docker build -t readmission-api:phase6 .
   ```

3. Run it with the bundle mounted read-only:

   ```bash
   # Linux / macOS / Git Bash
   docker run --rm -p 8000:8000 -v "$(pwd)/model_bundle:/model-bundle:ro" readmission-api:phase6
   # Windows PowerShell
   docker run --rm -p 8000:8000 -v "${PWD}\model_bundle:/model-bundle:ro" readmission-api:phase6
   ```

4. Smoke-test it from the host (standard library only):

   ```bash
   python -m src.api.smoke_test --url http://127.0.0.1:8000
   ```

   It calls `/health` and `/model-info`, sends one valid made-up request and
   one invalid request (extra `race` field) to `/predict`, and exits with
   status 1 if any answer is unexpected.

The same bundle also works without Docker, from the project root:

```bash
READMISSION_MODEL_BUNDLE=model_bundle uvicorn src.api.main:app        # bash
$env:READMISSION_MODEL_BUNDLE="model_bundle"; uvicorn src.api.main:app # PowerShell
```

## What the image contains

| | |
|---|---|
| Base | `python:3.11.9-slim-bookworm` (same Python as the project environment) |
| Code | `src/api/` and the five `src/ml` modules it imports (`config`, `tracking`, `register_model`, `log_phase3_to_mlflow`, `__init__`) |
| Dependencies | `requirements-api.txt`: FastAPI, Uvicorn, `mlflow-skinny`, skops, scikit-learn, SciPy, NumPy, pandas, joblib, at the same versions as `requirements-ml.txt`; `pip check` runs during the build |
| User | non-root `api` (uid 10001) |
| Port | 8000 (`EXPOSE 8000`) |
| Health check | Docker `HEALTHCHECK` calls `/health` every 30 s |
| Environment | `READMISSION_MODEL_BUNDLE=/model-bundle`; MLflow telemetry off |

`.dockerignore` is an allow-list: only `requirements-api.txt` and the code
above enter the build context. Data, `mlflow.db`, `mlartifacts/`,
`artifacts/`, `model_bundle/`, `.env` and all Part 1 code stay out, so no
data or secret can be copied into the image by accident. No credentials are
needed at build or run time.

`mlflow-skinny` is the MLflow client without the tracking server and
database packages; the container only reads the bundle. Running the image
against an MLflow store is therefore not supported.

## Verification in the container

Before predictions are enabled, the API (`load_bundle` in `src/api/bundle.py`)
checks, failing closed with HTTP 503 on the first problem:

1. `manifest.json` exists and is for `hospital-readmission-30d-classifier`
   `@candidate`.
2. The files under `model/` are exactly those listed in the manifest, with the
   same SHA-256 (nothing added, removed or changed).
3. `model.skops` has the SHA-256 of registered version 1's file,
   `13f79fe5…d1d5`, pinned in `src/api/model.py`. Editing the manifest to
   match a replaced file is not enough.
4. The same Phase 5 `verify()` checks on the version details recorded in the
   manifest and on `MLmodel`: READY, skops format, the 39 features in order,
   two-column output, threshold `0.1617830043681171`, the Phase 3 hash
   `1d17bf4a…62f9` in the version tag, `MLmodel` metadata and run tag, and
   `retrained=false`.
5. The loaded object is a binary classifier with `predict_proba`.

The trusted hashes and the threshold are constants in the code; only the
bundle location can be set from the environment.

## Tests

| File | Tests | What they cover |
|---|---:|---|
| `tests/test_bundle.py` | 18 | Export from a temporary synthetic registry: manifest, unchanged files, no data or local paths, refuses a non-empty folder, an untrusted file or a model failing verification. Serving: bundle and registry give the same `/model-info` and predictions; settings from the environment; trusted values cannot be overridden. Fail closed (503): missing bundle, changed, added or removed file, manifest rewritten for a replaced `model.skops`, wrong threshold, wrong run hash, other model, not READY |
| `tests/test_docker_config.py` | 3 | Dockerfile copies code only, runs as non-root, exposes 8000; `.dockerignore` keeps data, store and Part 1 out; `requirements-api.txt` pins match `requirements-ml.txt` |

## Actual results

Docker is not installed on the project machine, so the image was built and
run in the cloud workspace (Docker 29.8.2) with this Dockerfile. There the
base image had to come from Docker Hub's mirror (`mirror.gcr.io`) with the
workspace's proxy certificate added, because of the workspace's network
setup; the Dockerfile itself is unchanged.

- `docker build`: succeeded, `pip check` clean, image 828 MB.
- Container with no bundle: `/health` 503 "model bundle is unavailable".
- Container with a synthetic test bundle and the real trust values:
  `/health` 503 "model bundle is for a different registered model" (fails
  closed as intended).
- Container serving the synthetic bundle with test trust values: smoke test
  passed (`/health` 200, `/model-info` 200, `/predict` 200 with a probability
  identical to the host's `predict_proba`, extra `race` field 422). Docker's
  health check reported `healthy`; it also ran with `--read-only`.
- On the project machine (no Docker): the real bundle was exported
  (`model.skops` `13f79fe5…d1d5`) and served with Uvicorn from the bundle.
  Smoke test passed (`/health` 200, `/predict` 200 with probability 0.0671
  and `high_risk` false, extra `race` field 422), and 20 real validation rows
  gave the same probabilities as the registry (max difference 2.2e-16). Full
  suite: 191 passed.
- At that point the real model had not yet run inside a container; that test
  was completed on 15 October 2026 (next section).

### Real-model container test (15 October 2026)

Docker Desktop 29.8.2 (WSL2 backend) on the project machine. The image was
built from the unchanged Dockerfile, and the real registered model was served
from the existing exported bundle, mounted read-only. No retraining, and no
change to the registry, alias, threshold or pinned hashes.

- Bundle check before the run: `python -m src.api.bundle --verify model_bundle`
  → OK, `hospital-readmission-30d-classifier` version 1 (`@candidate`),
  threshold 0.1617830043681171, model sha256 `1d17bf4a…62f9`.
- `docker build -t readmission-api:real-model-test .`: succeeded in 156 s,
  `pip check` "No broken requirements found", image runs as user `api`.
- Run:

  ```powershell
  docker run -d --name readmission-real -p 127.0.0.1:8000:8000 `
    --read-only --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges `
    -v "${PWD}\model_bundle:/model-bundle:ro" readmission-api:real-model-test
  ```

  Docker's health check reported `healthy` after about 15 s; read-only root
  filesystem, read-only bundle mount, port bound to 127.0.0.1 only.
- `python -m src.api.smoke_test --url http://127.0.0.1:8000`: **passed**.
  `/health` 200 (`model_loaded` true, version 1); `/model-info` 200
  (version 1, alias `candidate`, threshold 0.1617830043681171, sha256
  `1d17bf4a…62f9`, `skops`, 39 features, disclaimer present); `/predict` 200
  with probability 0.0671 and `high_risk` false (the same value the bundle
  gave under Uvicorn); extra `race` field 422.
- Further checks, all as expected: extra `payer_code`, `encounter_id`,
  `patient_nbr` and `patient_sk` fields 422; a missing field, an out-of-range
  `time_in_hospital`, a non-numeric `num_medications` and a disallowed
  `gender` 422; discharge codes 11, 13 and 14 (death or hospice) 422. The
  `/predict` response carries the academic-prototype disclaimer, and none of
  race, payer or patient identifiers is a model input.
- The container log contains only start-up and access lines; no request
  values were logged. The container was then stopped and removed.

## Limitations

- The container serves one fixed model version. A new registered version
  needs a new export (and new pinned hashes in the code).
- The bundle's integrity rests on the pinned `model.skops` hash and the
  manifest; mount it read-only and keep it with the project.
- No authentication, HTTPS or rate limiting; local prototype only. Uvicorn's
  access log records method, path and status code, never request bodies.
