"""
Phase 3 entry point: train, tune and compare the three required models,
select a candidate on the validation set, then evaluate ONLY that candidate
once on the test set.

Run from the project root:

    python -m src.ml.train            # full run
    python -m src.ml.train --quick    # small smoke run (one setting per model)

Outputs (gitignored) go to artifacts/phase3/.
"""

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, GroupKFold

from src.ml import config
from src.ml.metrics import best_f1_threshold, evaluate, probability_metrics
from src.ml.models import build_model_specs
from src.ml.plots import (
    confusion_matrix_plot,
    importance_plot,
    reliability_plot,
    roc_pr_plot,
)
from src.ml.prepare_data import prepare, write_split_assignments
from src.ml.selection import select_candidate


PHASE3_DIR = config.ARTIFACTS_DIR / "phase3"
CV_FOLDS = 5


# ---------------------------------------------------------
# Training and selection (never sees the test set)
# ---------------------------------------------------------

def tune_model(spec, X_train, y_train, groups, cv_folds=CV_FOLDS):
    """
    Grid search with patient-grouped cross-validation on the training split.

    GroupKFold keeps every patient's encounters in one fold, so tuning is
    never scored on a patient seen in that fold's training data. Best
    setting = highest mean CV ROC-AUC; it is then refitted on all training
    rows.
    """
    search = GridSearchCV(
        estimator=spec.pipeline,
        param_grid=spec.param_grid,
        scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"},
        refit="roc_auc",
        cv=GroupKFold(n_splits=cv_folds),
        n_jobs=1,
        error_score="raise",
    )
    search.fit(X_train, y_train, groups=groups)
    return search


def calibrate(pipeline, X_train, y_train, groups, cv_folds=CV_FOLDS):
    """
    Platt (sigmoid) calibration learned from the TRAINING split only.

    Class weights make the models pay attention to the rare positive class,
    but they also push predicted probabilities far above the real ~11%
    readmission rate. To fix the probabilities without touching validation
    or test data:

    1. Out-of-fold predictions are made on the training split with the same
       patient-grouped folds used for tuning.
    2. A logistic (sigmoid) curve is fitted that maps those scores to the
       observed readmission rate.
    3. The model itself is refitted on all training rows.

    Ranking (ROC-AUC) is almost unchanged; probabilities become usable.
    """
    folds = list(GroupKFold(n_splits=cv_folds).split(X_train, y_train, groups))
    calibrated = CalibratedClassifierCV(
        clone(pipeline), method="sigmoid", cv=folds, ensemble=False
    )
    return calibrated.fit(X_train, y_train)


def train_and_select(X_train, y_train, groups, X_val, y_val, specs,
                     cv_folds=CV_FOLDS):
    """
    Tune every model on training data, score it on validation, choose each
    model's threshold on validation, and select the candidate.

    The test set is not a parameter of this function.
    """
    results = {}
    for spec in specs:
        started = time.perf_counter()
        search = tune_model(spec, X_train, y_train, groups, cv_folds)
        raw_val_prob = search.best_estimator_.predict_proba(X_val)[:, 1]
        model = calibrate(
            search.best_estimator_, X_train, y_train, groups, cv_folds
        )
        val_prob = model.predict_proba(X_val)[:, 1]
        threshold, _ = best_f1_threshold(y_val, val_prob)

        cv = pd.DataFrame(search.cv_results_)
        best_row = cv.loc[search.best_index_]
        results[spec.name] = {
            "display_name": spec.display_name,
            "pipeline": model,
            "best_params": search.best_params_,
            "cv_results": cv,
            "cv_roc_auc_mean": float(best_row["mean_test_roc_auc"]),
            "cv_roc_auc_std": float(best_row["std_test_roc_auc"]),
            "cv_pr_auc_mean": float(best_row["mean_test_pr_auc"]),
            "validation_probabilities": val_prob,
            "selected_threshold": threshold,
            "validation": evaluate(y_val, val_prob, threshold),
            # Same model before calibration, reported for transparency.
            "validation_uncalibrated": probability_metrics(y_val, raw_val_prob),
            "training_seconds": round(time.perf_counter() - started, 1),
        }

    selected, reasons = select_candidate({
        name: result["validation"] for name, result in results.items()
    })
    return results, selected, reasons


# ---------------------------------------------------------
# Final test evaluation (candidate only, after selection)
# ---------------------------------------------------------

def evaluate_on_test(candidate, X_test, y_test):
    """Score the already-selected candidate with its validation threshold."""
    test_prob = candidate["pipeline"].predict_proba(X_test)[:, 1]
    return test_prob, evaluate(y_test, test_prob, candidate["selected_threshold"])


# ---------------------------------------------------------
# Subgroups and explainability
# ---------------------------------------------------------

def age_group(age_band):
    low = int(str(age_band).split("-")[0])
    if low < 50:
        return "<50"
    if low < 70:
        return "50-69"
    return "70+"


def subgroup_metrics(cohort_rows, y_true, y_prob, threshold):
    """
    Metrics by gender, age group and race. Race is used here only, never as
    a model input. Groups with fewer than MIN_SUBGROUP_SIZE encounters are
    suppressed; groups with fewer than 30 positives are marked unstable.
    """
    frame = pd.DataFrame({
        "gender": cohort_rows["gender"].to_numpy(),
        "age_group": cohort_rows["age"].map(age_group).to_numpy(),
        "race": cohort_rows["race"].fillna("Unknown").to_numpy(),
        "y": np.asarray(y_true),
        "p": np.asarray(y_prob),
    })
    rows = []
    for attribute in ("gender", "age_group", "race"):
        for value, part in frame.groupby(attribute):
            n, positives = len(part), int(part["y"].sum())
            row = {"attribute": attribute, "group": value}
            if n < config.MIN_SUBGROUP_SIZE:
                row.update({"encounters": f"<{config.MIN_SUBGROUP_SIZE}",
                            "note": "suppressed (small group)"})
                rows.append(row)
                continue
            predicted = part["p"] >= threshold
            true_positive = int((predicted & (part["y"] == 1)).sum())
            row.update({
                "encounters": n,
                "positives": positives,
                "observed_rate": round(part["y"].mean(), 4),
                "mean_predicted": round(part["p"].mean(), 4),
                "roc_auc": (
                    round(probability_metrics(part["y"], part["p"])["roc_auc"], 4)
                    if 0 < positives < n else None
                ),
                "recall": round(true_positive / positives, 4) if positives else None,
                "precision": (
                    round(true_positive / predicted.sum(), 4)
                    if predicted.sum() else None
                ),
                "flagged_rate": round(predicted.mean(), 4),
                "note": "unstable (<30 positives)" if positives < 30 else "",
            })
            rows.append(row)
    return pd.DataFrame(rows)


def feature_importance(pipeline, X_val, y_val, seed=config.RANDOM_SEED,
                       n_repeats=5):
    """
    Permutation importance on the validation set: how much ROC-AUC drops
    when one original feature's values are shuffled. Works the same for all
    three model types and is computed on the 39 input features.
    """
    result = permutation_importance(
        pipeline, X_val, y_val, scoring="roc_auc",
        n_repeats=n_repeats, random_state=seed, n_jobs=1,
    )
    return (
        pd.DataFrame({
            "feature": X_val.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        })
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------
# Orchestration
# ---------------------------------------------------------

def split_frames(cohort, features, split):
    parts = {}
    for name in ("train", "validation", "test"):
        mask = (split == name).to_numpy()
        parts[name] = (
            features.loc[mask],
            cohort.loc[mask, config.TARGET_COLUMN].to_numpy(),
            cohort.loc[mask],
        )
    return parts


def quick_specs(y_train, seed=config.RANDOM_SEED):
    """One setting per model (first grid value) for smoke runs and tests."""
    specs = build_model_specs(y_train, seed=seed)
    for spec in specs:
        spec.param_grid = {key: values[:1] for key, values in spec.param_grid.items()}
    return specs


def run(cohort, features, split, quick=False, cv_folds=CV_FOLDS,
        importance_repeats=5):
    parts = split_frames(cohort, features, split)
    X_train, y_train, train_rows = parts["train"]
    X_val, y_val, val_rows = parts["validation"]
    X_test, y_test, test_rows = parts["test"]
    groups = train_rows[config.GROUP_COLUMN].to_numpy()

    specs = quick_specs(y_train) if quick else build_model_specs(y_train)
    results, selected, reasons = train_and_select(
        X_train, y_train, groups, X_val, y_val, specs, cv_folds
    )

    candidate = results[selected]
    test_prob, test_metrics = evaluate_on_test(candidate, X_test, y_test)

    return {
        "results": results,
        "selected": selected,
        "selection_reasons": reasons,
        "test_probabilities": test_prob,
        "test": test_metrics,
        "subgroups_validation": subgroup_metrics(
            val_rows, y_val, candidate["validation_probabilities"],
            candidate["selected_threshold"],
        ),
        "subgroups_test": subgroup_metrics(
            test_rows, y_test, test_prob, candidate["selected_threshold"]
        ),
        "importance": feature_importance(
            candidate["pipeline"], X_val, y_val, n_repeats=importance_repeats
        ),
        "y": {"validation": y_val, "test": y_test},
        "sizes": {name: len(parts[name][1]) for name in parts},
        "class_weight_ratio_train": round(
            (len(y_train) - y_train.sum()) / y_train.sum(), 4
        ),
    }


def comparison_table(outcome):
    rows = []
    for name, r in outcome["results"].items():
        v = r["validation"]
        rows.append({
            "model": r["display_name"],
            "best_params": json.dumps(r["best_params"]),
            "cv_roc_auc": round(r["cv_roc_auc_mean"], 4),
            "val_roc_auc": round(v["roc_auc"], 4),
            "val_pr_auc": round(v["pr_auc"], 4),
            "val_brier": round(v["brier"], 4),
            "val_ece": round(v["ece"], 4),
            "val_mean_pred": round(v["mean_predicted_probability"], 4),
            "uncal_val_roc_auc": round(r["validation_uncalibrated"]["roc_auc"], 4),
            "uncal_val_brier": round(r["validation_uncalibrated"]["brier"], 4),
            "uncal_val_ece": round(r["validation_uncalibrated"]["ece"], 4),
            "uncal_val_mean_pred": round(
                r["validation_uncalibrated"]["mean_predicted_probability"], 4
            ),
            "precision@0.5": round(v["at_0_5"]["precision"], 4),
            "recall@0.5": round(v["at_0_5"]["recall"], 4),
            "f1@0.5": round(v["at_0_5"]["f1"], 4),
            "selected_threshold": round(r["selected_threshold"], 4),
            "precision@sel": round(v["at_selected"]["precision"], 4),
            "recall@sel": round(v["at_selected"]["recall"], 4),
            "f1@sel": round(v["at_selected"]["f1"], 4),
            "selected": name == outcome["selected"],
        })
    return pd.DataFrame(rows)


def save_outputs(outcome, report, fingerprint, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)

    selected = outcome["selected"]
    candidate = outcome["results"][selected]
    y_val, y_test = outcome["y"]["validation"], outcome["y"]["test"]

    comparison = comparison_table(outcome)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    for name, r in outcome["results"].items():
        r["cv_results"].to_csv(output_dir / f"cv_results_{name}.csv", index=False)
    outcome["subgroups_validation"].to_csv(output_dir / "subgroups_validation.csv", index=False)
    outcome["subgroups_test"].to_csv(output_dir / "subgroups_test.csv", index=False)
    outcome["importance"].to_csv(output_dir / "permutation_importance.csv", index=False)

    metrics = {
        "split_fingerprint_sha256": fingerprint,
        "random_seed": config.RANDOM_SEED,
        "sizes": outcome["sizes"],
        "positive_class": "readmitted <30 days",
        "train_negatives_per_positive": outcome["class_weight_ratio_train"],
        "cv": f"GroupKFold({CV_FOLDS}) by patient on training split",
        "pre_training_checks_passed": all(c["passed"] for c in report["checks"]),
        "models": {
            name: {
                "best_params": r["best_params"],
                "cv_roc_auc_mean": r["cv_roc_auc_mean"],
                "cv_roc_auc_std": r["cv_roc_auc_std"],
                "cv_pr_auc_mean": r["cv_pr_auc_mean"],
                "selected_threshold": r["selected_threshold"],
                "training_seconds": r["training_seconds"],
                "validation": r["validation"],
                "validation_uncalibrated": r["validation_uncalibrated"],
            }
            for name, r in outcome["results"].items()
        },
        "selected_model": selected,
        "selection_reasons": outcome["selection_reasons"],
        "test_selected_model": outcome["test"],
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    val_curves = {
        r["display_name"]: (y_val, r["validation_probabilities"])
        for r in outcome["results"].values()
    }
    plots = output_dir / "plots"
    reliability_plot(val_curves, plots / "reliability_validation.png")
    roc_pr_plot(val_curves, plots / "roc_pr_validation.png")
    reliability_plot(
        {candidate["display_name"]: (y_test, outcome["test_probabilities"])},
        plots / "reliability_test_selected.png",
        title="Reliability (test, selected model)",
    )
    for name, r in outcome["results"].items():
        confusion_matrix_plot(
            r["validation"]["at_selected"]["confusion_matrix"],
            plots / f"confusion_validation_{name}.png",
            f"{r['display_name']} validation @ {r['selected_threshold']:.3f}",
        )
    confusion_matrix_plot(
        outcome["test"]["at_selected"]["confusion_matrix"],
        plots / "confusion_test_selected.png",
        f"{candidate['display_name']} test @ {candidate['selected_threshold']:.3f}",
    )
    importance_plot(
        outcome["importance"], plots / "permutation_importance.png",
        f"Permutation importance: {candidate['display_name']} (validation)",
    )

    # Candidate pipeline for later phases (gitignored, not committed).
    joblib.dump(candidate["pipeline"], output_dir / "models" / f"{selected}.joblib")
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="one hyperparameter setting per model")
    args = parser.parse_args()

    started = time.perf_counter()
    cohort, features, split, report = prepare()
    failed = [c["check"] for c in report["checks"] if not c["passed"]]
    if failed:
        raise SystemExit(f"Pre-training checks failed: {failed}")

    PHASE3_DIR.mkdir(parents=True, exist_ok=True)
    assignments = cohort[[config.ENCOUNTER_ID_COLUMN, config.GROUP_COLUMN]].assign(
        split=split.to_numpy()
    )
    fingerprint = write_split_assignments(
        assignments, PHASE3_DIR / "split_assignments.csv"
    )

    outcome = run(cohort, features, split, quick=args.quick)
    comparison = save_outputs(outcome, report, fingerprint, PHASE3_DIR)

    pd.set_option("display.width", 200)
    print("PHASE 3 MODEL TRAINING AND EVALUATION")
    print("=" * 70)
    print(f"Split fingerprint: {fingerprint}")
    print(f"Sizes: {outcome['sizes']}")
    print()
    print("Validation comparison:")
    print(comparison.drop(columns=["best_params"]).T.to_string(header=False))
    print()
    for line in outcome["selection_reasons"]:
        print(line)
    test = outcome["test"]
    sel = test["at_selected"]
    print()
    print(f"TEST ({outcome['selected']}, threshold {sel['threshold']}):")
    print(f"  ROC-AUC {test['roc_auc']:.4f}  PR-AUC {test['pr_auc']:.4f}  "
          f"Brier {test['brier']:.4f}  ECE {test['ece']:.4f}")
    print(f"  precision {sel['precision']:.4f}  recall {sel['recall']:.4f}  "
          f"F1 {sel['f1']:.4f}  confusion {sel['confusion_matrix']}")
    print()
    print(f"Outputs: {PHASE3_DIR}")
    print(f"Total time: {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
