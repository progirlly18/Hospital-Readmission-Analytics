# Serving the registered model from a fresh clone

> Academic decision-support prototype. The model is not clinically validated
> and must not be used for real patient decisions.

The MLflow store (`mlflow.db`, `mlartifacts/`) and the exported bundle
(`model_bundle/`) are generated files and are not committed. A fresh clone
therefore has the code but not the model. Retraining is **not** a way to get
it back: the API only serves the model whose hashes are pinned in
`src/api/model.py` (`TRUSTED_MODEL_SHA256`, `TRUSTED_ARTIFACT_SHA256`), and a
retrained model would have a different file hash, so `/health` would report
503 by design.

Instead, the registered model travels as a verified archive of the bundle:

| Item | Value |
|---|---|
| File | `model_bundle_v1.zip` (about 10 MB) |
| SHA-256 of the zip | `5993ef3a3321d893855e406c0565ba1976109db1840241b6d3756f6859071b04` |
| Contents | `model_bundle/manifest.json` and `model_bundle/model/` (the registered version's MLflow files, unchanged) |
| Model | `hospital-readmission-30d-classifier` version 1 (`@candidate`), threshold 0.1617830043681171 |
| `model.skops` SHA-256 | `13f79fe55a2112ad17fc77af4b8cda65784a090fb9890bb8b50688f1a704d1d5` |
| Model provenance SHA-256 | `1d17bf4ac345891a58b4566ca5b3265798fea2cc73aa2dde860015dfc25062f9` |

The zip holds the model only: no data, identifiers, MLflow database or
credentials. It is delivered with the submission (not committed to Git).

## Creating the archive (project machine, which has the MLflow store)

```bash
python -m src.api.bundle                                  # export model_bundle/ (skip if it exists)
python -m src.api.bundle --archive dist/model_bundle_v1.zip
```

The archive command first runs every API check on the bundle, then writes
the zip with fixed timestamps, so the same bundle always produces the same
zip hash, plus `dist/model_bundle_v1.zip.sha256`. `dist/` is gitignored.

## Using it on a fresh clone (marker)

1. Clone the repository and install the API dependencies:

   ```bash
   git clone https://github.com/progirlly18/Hospital-Readmission-Analytics.git
   cd Hospital-Readmission-Analytics
   pip install -r requirements-api.txt
   ```

2. Put `model_bundle_v1.zip` in the project root and check its hash against
   the value above:

   ```bash
   sha256sum model_bundle_v1.zip                           # Linux / macOS / Git Bash
   ```

   ```powershell
   Get-FileHash model_bundle_v1.zip -Algorithm SHA256      # PowerShell
   ```

3. Unzip it (creates `model_bundle/`) and verify it exactly as the API will:

   ```bash
   python -m zipfile -e model_bundle_v1.zip .
   python -m src.api.bundle --verify model_bundle
   ```

   Expected: `OK: model_bundle holds hospital-readmission-30d-classifier
   version 1 (@candidate), threshold 0.1617830043681171, model sha256
   1d17bf4a…62f9`.

4. Serve it, either directly or in Docker:

   ```bash
   READMISSION_MODEL_BUNDLE=model_bundle uvicorn src.api.main:app      # bash
   ```

   ```powershell
   $env:READMISSION_MODEL_BUNDLE="model_bundle"; uvicorn src.api.main:app
   ```

   ```bash
   docker build -t readmission-api:phase6 .
   docker run --rm -p 8000:8000 -v "$(pwd)/model_bundle:/model-bundle:ro" readmission-api:phase6
   ```

5. Check it:

   ```bash
   python -m src.api.smoke_test --url http://127.0.0.1:8000
   ```

## Actual results (project machine, 2026-10-15)

- `python -m src.api.bundle --verify model_bundle`: OK, version 1,
  `@candidate`, threshold 0.1617830043681171.
- `--archive` run twice gave the same zip SHA-256 (`5993ef3a…1b04`),
  10,256,501 bytes.
- Fresh-clone test: the branch was cloned into an empty folder (no
  `mlflow.db`, no `mlartifacts/`), the zip was unzipped there and verified,
  and Uvicorn served it from `READMISSION_MODEL_BUNDLE=model_bundle`. Smoke
  test passed: `/health` 200 (model version 1), `/model-info` 200 (threshold
  0.1617830043681171, sha256 `1d17bf4a…62f9`), `/predict` 200 (probability
  0.0671, `high_risk` false), request with an extra `race` field 422.
- `tests/test_bundle.py`: 22 passed (4 new archive tests: reproducible zip
  with only the bundle files, unzipped archive passes every check, tampered
  bundle refused, existing file not overwritten).
- **Not done:** the Docker step (4, second option) with the real bundle.
  Docker is not installed on the project machine or its WSL Ubuntu, so the
  real model has not run inside a container. The image itself was built and
  tested with a synthetic bundle in Phase 6 (`phase6_docker.md`). Running
  steps 1–5 with Docker on any machine that has it completes this test.
