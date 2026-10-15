"""
Phase 7: monitoring report for the registered readmission model.

Run from the project root (the model is loaded and verified exactly as the
API does it: from the MLflow registry, or from the exported bundle when
READMISSION_MODEL_BUNDLE is set):

    python -m src.monitoring.report

What it does, without retraining anything:

1. Rebuilds the Phase 2 cohort, features and split, and refuses to continue
   unless the split fingerprint equals the one the model was trained on.
2. Baselines: input distributions from the training split; predictions and
   labelled performance from the validation split (the threshold's data).
3. Monitored batches:
   - "test": the held-out test split, real and unchanged;
   - "simulated_shift_a" and "simulated_shift_b": the two halves of the test
     split with an ARTIFICIAL, clearly labelled shift (older patients, more
     prior inpatient visits, longer stays). They exist only to show that
     drift detection and the retraining rules react; they are not real data.
4. Per batch: input-data quality (missing and invalid values per input,
   src/monitoring/data_quality.py), PSI per feature and for the predictions,
   labelled performance and calibration, and subgroup metrics (real batch
   only).
5. Explainability: permutation importance of the 39 inputs on validation.
6. Retraining criteria (src/monitoring/retraining.py), applied to the real
   batch and to the simulated batches separately.

Outputs (gitignored): artifacts/monitoring/monitoring_report.json and .md.
Only aggregates are written: no rows, identifiers or individual predictions,
and subgroups under 30 encounters are suppressed.
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml import config
from src.ml.data import load_modelling_data
from src.ml.features import build_feature_frame
from src.ml.metrics import probability_metrics, threshold_metrics
from src.ml.prepare_data import write_split_assignments
from src.ml.split import assign_splits
from src.ml.train import feature_importance, split_frames, subgroup_metrics
from src.monitoring.data_quality import input_quality
from src.monitoring.drift import DriftBaseline, prediction_psi
from src.monitoring.retraining import KEY_FEATURES, recommendation

EXPECTED_SPLIT_FINGERPRINT = (
    "45788b051270260de0dc1bb3eefc80b81f94b76fccfa0d280449631f388aa941"
)
OUTPUT_DIR = config.ARTIFACTS_DIR / "monitoring"
SIMULATION_NOTE = (
    "SIMULATED: test rows with an artificial shift (age +10 years, capped at "
    "95; prior inpatient visits +2; length of stay +2 days, capped at 14). "
    "Labels are unchanged, so performance on these batches is illustrative."
)


# ---------------------------------------------------------
# Data
# ---------------------------------------------------------

def split_fingerprint(cohort, split):
    assignments = cohort[[config.ENCOUNTER_ID_COLUMN, config.GROUP_COLUMN]].assign(
        split=split.to_numpy())
    with tempfile.TemporaryDirectory() as tmp:
        return write_split_assignments(assignments, Path(tmp) / "split.csv")


def load_splits(path=config.CLEANED_FILE, expected=EXPECTED_SPLIT_FINGERPRINT):
    """{name: (features, labels, cohort rows)}; fails if the split changed."""
    cohort, _ = load_modelling_data(path)
    split = assign_splits(cohort)
    if expected is not None and split_fingerprint(cohort, split) != expected:
        raise RuntimeError("Split fingerprint differs from the one the model was "
                           "trained on; refusing to report on a different split.")
    return split_frames(cohort, build_feature_frame(cohort), split)


def simulate_shift(features):
    """An artificial shift for demonstration only (see SIMULATION_NOTE)."""
    shifted = features.copy()
    shifted["age_years"] = np.minimum(shifted["age_years"] + 10, 95.0)
    shifted["number_inpatient"] = shifted["number_inpatient"] + 2
    shifted["time_in_hospital"] = np.minimum(shifted["time_in_hospital"] + 2, 14)
    return shifted


# ---------------------------------------------------------
# Measurements
# ---------------------------------------------------------

def performance(labels, probabilities, threshold):
    result = probability_metrics(labels, probabilities)
    at_threshold = threshold_metrics(labels, probabilities, threshold)
    result.update({key: at_threshold[key] for key in ("precision", "recall", "f1")})
    result["flagged_rate"] = at_threshold["predicted_positive_rate"]
    return result


def records(table):
    """DataFrame -> JSON-safe list of dicts (NaN -> None)."""
    return json.loads(table.to_json(orient="records"))


def measure_batch(name, features, labels, model, threshold, baseline,
                  reference_probabilities, cohort_rows=None, note=""):
    probabilities = model.predict_proba(features)[:, 1]
    batch = {
        "name": name,
        "note": note,
        "rows": int(len(features)),
        "input_quality": input_quality(features),
        "feature_drift": records(baseline.feature_psi(features)),
        "prediction_psi": prediction_psi(reference_probabilities, probabilities),
        "performance": performance(labels, probabilities, threshold),
    }
    if cohort_rows is not None:
        batch["subgroups"] = records(subgroup_metrics(cohort_rows, labels,
                                                      probabilities, threshold))
    return batch


def importance(model, features, labels, rows, seed=config.RANDOM_SEED):
    if rows and rows < len(features):
        sample = np.random.default_rng(seed).choice(len(features), rows, replace=False)
        features, labels = features.iloc[sample], labels[sample]
    return feature_importance(model, features, labels, n_repeats=3)


# ---------------------------------------------------------
# Report
# ---------------------------------------------------------

def build_report(splits, model, threshold, model_info, importance_rows=5000):
    train_x, _, _ = splits["train"]
    val_x, val_y, _ = splits["validation"]
    test_x, test_y, test_rows = splits["test"]

    baseline = DriftBaseline(train_x)
    val_probabilities = model.predict_proba(val_x)[:, 1]
    reference = performance(val_y, val_probabilities, threshold)

    halves = np.array_split(np.arange(len(test_x)), 2)
    batches = [measure_batch("test", test_x, test_y, model, threshold, baseline,
                             val_probabilities, cohort_rows=test_rows,
                             note="Held-out test split (real, unchanged).")]
    for label, index in zip("ab", halves):
        batches.append(measure_batch(
            f"simulated_shift_{label}", simulate_shift(test_x.iloc[index]),
            test_y[index], model, threshold, baseline, val_probabilities,
            note=SIMULATION_NOTE))

    ranked = importance(model, val_x, val_y, importance_rows)
    key_features = list(ranked["feature"].head(KEY_FEATURES))
    return {
        "model": model_info,
        "disclaimer": ("Academic decision-support prototype; not clinically "
                       "validated and not for real patient decisions."),
        "baseline": {"inputs": f"training split ({len(train_x)} encounters)",
                     "predictions_and_performance":
                         f"validation split ({len(val_x)} encounters)"},
        "reference_performance": reference,
        "batches": batches,
        "explainability": {
            "method": "permutation importance (ROC-AUC drop when one input is "
                      "shuffled) on validation rows",
            "rows": int(min(importance_rows or len(val_x), len(val_x))),
            "importance": records(ranked),
            "key_features": key_features,
        },
        # Real and simulated batches are judged separately, so the artificial
        # shift can never produce a recommendation about the real model.
        "retraining": {
            "real_batches": recommendation(batches[:1], reference, key_features),
            "simulated_demonstration": recommendation(batches[1:], reference,
                                                      key_features),
        },
    }


def markdown(report):
    lines = ["# Monitoring report", "", f"> {report['disclaimer']}", "",
             f"Model: {report['model']}", "",
             "## Batches", "",
             "| Batch | Rows | ROC-AUC | ECE | Flagged | Prediction PSI | "
             "Features with PSI > 0.25 |", "|---|---:|---:|---:|---:|---:|---|"]
    reference = report["reference_performance"]
    lines.append(f"| validation (reference) | {reference['n']} | {reference['roc_auc']:.4f} "
                 f"| {reference['ece']:.4f} | {reference['flagged_rate']:.4f} | - | - |")
    for batch in report["batches"]:
        perf = batch["performance"]
        drifted = ", ".join(row["feature"] for row in batch["feature_drift"]
                            if row["status"] == "significant") or "none"
        lines.append(f"| {batch['name']} | {batch['rows']} | {perf['roc_auc']:.4f} | "
                     f"{perf['ece']:.4f} | {perf['flagged_rate']:.4f} | "
                     f"{batch['prediction_psi']:.4f} | {drifted} |")
    lines += ["", f"Simulated batches: {SIMULATION_NOTE}", "",
              "## Input-data quality", "",
              "Values the API would reject (missing, out of range or unknown "
              "category), per batch. A batch fails when any input has more than "
              f"{report['batches'][0]['input_quality']['limit']:.0%} of its rows "
              "affected.", "",
              "| Batch | Rows | Missing values | Invalid values | Result |",
              "|---|---:|---:|---:|---|"]
    for batch in report["batches"]:
        quality = batch["input_quality"]
        result = "passed" if quality["passed"] else (
            "failed: " + ", ".join(quality["failing_features"]))
        lines.append(f"| {batch['name']} | {quality['rows']} | "
                     f"{quality['missing_values']} | {quality['invalid_values']} | "
                     f"{result} |")
    lines += ["",
              "## Most important inputs (permutation importance)", "",
              "| Feature | ROC-AUC drop |", "|---|---:|"]
    for row in report["explainability"]["importance"][:KEY_FEATURES]:
        lines.append(f"| {row['feature']} | {row['importance_mean']:.4f} |")
    real = report["retraining"]["real_batches"]
    lines += ["", "## Retraining criteria", ""]
    lines += [f"- `{c['criterion']}`: {c['rule']}" for c in real["criteria"]]
    for title, result in (("Real data (test split)", real),
                          ("Simulated shift (demonstration only)",
                           report["retraining"]["simulated_demonstration"])):
        verdict = "yes" if result["retraining_recommended"] else "no"
        lines += ["", f"**{title}: retraining recommended: {verdict}**", ""]
        lines += [f"- {reason}" for reason in result["retrain_reasons"]]
        lines += [f"- warning: {warning}" for warning in result["warnings"]]
        lines += [f"- fairness review: {item}" for item in result["fairness_review"]]
    return "\n".join(lines) + "\n"


def write_report(report, out_dir=OUTPUT_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "monitoring_report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    (out_dir / "monitoring_report.md").write_text(markdown(report), encoding="utf-8")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--importance-rows", type=int, default=5000)
    args = parser.parse_args()

    from src.api.model import ModelNotReady, Settings, load_model
    try:
        loaded = load_model(Settings.from_env())
    except ModelNotReady as error:
        raise SystemExit(f"Model not available: {error}") from None

    report = build_report(
        load_splits(), loaded.model, loaded.threshold,
        model_info=f"{loaded.name} version {loaded.version} (@{loaded.alias}), "
                   f"threshold {loaded.threshold!r}, sha256 {loaded.model_sha256}",
        importance_rows=args.importance_rows,
    )
    out_dir = write_report(report, args.out)
    print(markdown(report))
    print(f"Written to {out_dir}")


if __name__ == "__main__":
    main()
