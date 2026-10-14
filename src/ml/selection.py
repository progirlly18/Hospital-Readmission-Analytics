"""
Model selection rule (validation data only).

The test set is deliberately NOT an input to this function: selection can
only see validation metrics.

Rule:
1. Eligible models are within ROC_AUC_TOLERANCE of the best validation
   ROC-AUC AND within PR_AUC_TOLERANCE of the best validation PR-AUC.
2. Among eligible models, choose the lowest validation Brier score
   (better calibrated).
3. If Brier scores are within BRIER_TOLERANCE of the lowest, choose the
   simplest model: Logistic Regression, then Random Forest, then XGBoost.
"""

from src.ml.models import SIMPLICITY_ORDER


ROC_AUC_TOLERANCE = 0.01
PR_AUC_TOLERANCE = 0.01
BRIER_TOLERANCE = 0.002


def select_candidate(validation_metrics):
    """
    validation_metrics: {model_name: {"roc_auc", "pr_auc", "brier", ...}}
    Returns (selected_model_name, explanation_lines).
    """
    best_roc = max(m["roc_auc"] for m in validation_metrics.values())
    best_pr = max(m["pr_auc"] for m in validation_metrics.values())

    eligible = [
        name for name, m in validation_metrics.items()
        if m["roc_auc"] >= best_roc - ROC_AUC_TOLERANCE
        and m["pr_auc"] >= best_pr - PR_AUC_TOLERANCE
    ]
    reasons = [
        f"Best validation ROC-AUC {best_roc:.4f}, best PR-AUC {best_pr:.4f}.",
        f"Eligible (within {ROC_AUC_TOLERANCE} ROC-AUC and {PR_AUC_TOLERANCE} "
        f"PR-AUC of the best): {', '.join(eligible)}.",
    ]

    lowest_brier = min(validation_metrics[name]["brier"] for name in eligible)
    near_best_brier = [
        name for name in eligible
        if validation_metrics[name]["brier"] <= lowest_brier + BRIER_TOLERANCE
    ]
    reasons.append(
        f"Lowest Brier among eligible {lowest_brier:.4f}; within "
        f"{BRIER_TOLERANCE}: {', '.join(near_best_brier)}."
    )

    selected = min(near_best_brier, key=SIMPLICITY_ORDER.index)
    reasons.append(f"Selected the simplest of these: {selected}.")
    return selected, reasons
