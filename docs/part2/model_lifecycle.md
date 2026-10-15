# Model lifecycle

> Academic decision-support prototype. The model is not clinically validated
> and must not be used for real patient decisions.

This page brings together, in one place, how a model moves from training to
serving, how it is watched, and how it is replaced or rolled back. It
describes the process as built in Phases 2–8; steps that are a written
procedure rather than code are marked **(procedure)**.

## Current state

| Item | Value |
|---|---|
| Registered model | `hospital-readmission-30d-classifier` |
| Version | 1 (the only version) |
| Alias | `@candidate` |
| Formal approval | **None.** No version has been approved for use; see below |
| Decision threshold | 0.1617830043681171 (stored as a version tag) |
| Pinned hashes | `TRUSTED_MODEL_SHA256` `1d17bf4a…62f9` and `TRUSTED_ARTIFACT_SHA256` `13f79fe5…d1d5` in `src/api/model.py` |
| Served by | FastAPI from the registry (`uvicorn src.api.main:app`) or from the exported bundle (`READMISSION_MODEL_BUNDLE`, Docker) |

## What `@candidate` means

`@candidate` marks the version that the selection rule chose and that passed
the automated checks. It is **not** an approval. In this project nobody has
signed off the model for any real use, and nobody could: it is a coursework
prototype that has never been validated clinically. The API serves the
`@candidate` version for demonstration only, and every response carries the
disclaimer. If the model were ever to be used for real, an `@approved` (or
`@production`) alias would be set by a named reviewer after the approval
step below, and the API would be pointed at that alias instead.

## Stages

```text
data (Part 1) -> prepare -> train & select -> log & register (@candidate)
   -> [approve] -> export bundle -> serve (API / Docker / dashboard)
   -> monitor -> retraining criteria -> retrain -> new version -> ...
                                     \-> rollback to previous version
```

1. **Prepare data** (`python -m src.ml.prepare_data`). Builds the cohort,
   the 39 features and the patient-grouped split; the split fingerprint
   (`45788b05…a941`) identifies the exact split.
2. **Train and select** (`python -m src.ml.train`). Tunes, calibrates and
   compares Logistic Regression, Random Forest and XGBoost on validation, and
   applies the selection rule fixed in advance. The test split is used once,
   for the selected model only.
3. **Log and register** (`python -m src.ml.log_phase3_to_mlflow`,
   `python -m src.ml.register_model`). Logs parameters, metrics and plots,
   then registers the saved model as a new version with its threshold, hash,
   split fingerprint and provenance tags, and points `@candidate` at it.
   For version 1 these runs were logged **after** training, from the saved
   Phase 3 files (tagged `historical_phase3_results`, `retrained=false`); the
   training script itself does not call MLflow.
4. **Approve (procedure).** A reviewer checks, for the candidate version:
   test metrics against the previous version (ROC-AUC, PR-AUC, Brier, ECE),
   the subgroup table (no group with 30 or more positives more than 0.10
   ROC-AUC below the overall), the monitoring report on the newest labelled
   data, and the model card. The decision and the reviewer are recorded as
   version tags (`approved_by`, `approval_date`, `approval_notes`) and an
   `@approved` alias is set. **Not done for version 1.**
5. **Release.** The trusted hashes in `src/api/model.py` are updated to the
   new version's values in a reviewed pull request (they cannot be changed
   from the environment, so an unreviewed model is never served). The bundle
   is exported and archived (`python -m src.api.bundle`,
   `--archive dist/model_bundle_vN.zip`, see `model_bundle.md`) and the
   container is started with the bundle mounted read-only.
6. **Serve.** On start-up the API resolves the alias, checks status, skops
   format, signature, threshold and every hash, and refuses to serve (HTTP
   503) if anything differs.
7. **Monitor.** `/metrics` (requests, failures, latency, prediction
   distribution, in memory) and `python -m src.monitoring.report` (input
   quality, PSI drift, labelled performance, calibration, subgroups,
   importance). See `phase7_monitoring.md`.
8. **Decide on retraining.** `src/monitoring/retraining.py` applies the
   documented criteria: persistent drift on a top-10 input, prediction drift,
   ROC-AUC drop above 0.05, ECE above 0.03, flagged-rate change above 0.10,
   and a subgroup gap (fairness review). A failed input-quality check sends
   the batch back to the data pipeline first. Real labelled data newer than
   the training data, or a change in the Part 1 data, is also a reason to
   retrain. A triggered criterion is a recommendation for a person; nothing
   retrains automatically.
9. **Retrain (procedure).** Rerun steps 1–3 on the new data. The new model
   becomes version N+1 with `@candidate`; version N keeps its files, tags and
   history. It then goes through steps 4–6.

## Rollback

Versions are never deleted or overwritten, so rolling back means pointing
the service at the previous version again:

1. Move the alias back:
   `MlflowClient().set_registered_model_alias("hospital-readmission-30d-classifier", "candidate", "<previous version>")`
   (and `@approved`, if used).
2. Restore the previous version's trusted hashes and threshold in
   `src/api/model.py` (revert the release pull request).
3. Re-export or reuse that version's archived bundle (its zip SHA-256 is
   recorded at release) and restart the API or container.
4. `python -m src.api.smoke_test` and `/model-info` confirm the version,
   threshold and hash being served.

## Responsibilities in this project

All roles (developer, reviewer, operator) are held by the student. The
separation above shows where they would be split in a real deployment.
