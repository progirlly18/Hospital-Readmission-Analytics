"""
Evaluation metrics used to compare models.

All functions take true labels (0/1) and predicted probabilities of the
positive class (30-day readmission).
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def equal_frequency_bins(y_prob, n_bins=10):
    """
    Bin index (0..n_bins-1) per prediction, using quantiles of the predicted
    probabilities as bin edges. Identical predictions always share a bin, so
    the result does not depend on row order.
    """
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
    return np.searchsorted(edges[1:-1], y_prob, side="right")


def calibration_bins(y_true, y_prob, n_bins=10):
    """(mean predicted, observed rate, size) for each non-empty bin."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = equal_frequency_bins(y_prob, n_bins)
    return [
        (y_prob[bins == b].mean(), y_true[bins == b].mean(), int((bins == b).sum()))
        for b in np.unique(bins)
    ]


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Expected Calibration Error (ECE) with equal-frequency bins.

    Predictions are sorted into about n_bins groups of similar size. In each
    group we compare the average predicted probability with the observed
    readmission rate; ECE is the size-weighted average gap. 0 = perfectly
    calibrated. Equal-frequency bins are used because most predictions are
    low when only ~11% of encounters are positive.
    """
    total = 0.0
    count = 0
    for mean_predicted, observed, size in calibration_bins(y_true, y_prob, n_bins):
        total += abs(mean_predicted - observed) * size
        count += size
    return float(total / count)


def best_f1_threshold(y_true, y_prob):
    """
    Threshold that maximises F1 (used on VALIDATION data only).

    Returns (threshold, f1). The returned threshold is one of the predicted
    probabilities, so applying `y_prob >= threshold` reproduces that F1.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # The last precision/recall pair has no threshold.
    precision, recall = precision[:-1], recall[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(
            precision + recall > 0,
            2 * precision * recall / (precision + recall),
            0.0,
        )
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])


def threshold_metrics(y_true, y_prob, threshold):
    """Precision, recall, F1 and confusion matrix at one threshold."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "predicted_positive_rate": float(y_pred.mean()),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        },
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1],
            target_names=["not readmitted <30", "readmitted <30"],
            zero_division=0, output_dict=True,
        ),
    }


def probability_metrics(y_true, y_prob):
    """Threshold-free metrics: ranking (ROC-AUC, PR-AUC) and calibration."""
    y_prob = np.asarray(y_prob, dtype=float)
    return {
        "n": int(len(y_prob)),
        "positives": int(np.sum(y_true)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob),
        "mean_predicted_probability": float(y_prob.mean()),
        "observed_positive_rate": float(np.mean(y_true)),
    }


def evaluate(y_true, y_prob, selected_threshold):
    """Full evaluation at the default 0.5 and at the selected threshold."""
    return {
        **probability_metrics(y_true, y_prob),
        "at_0_5": threshold_metrics(y_true, y_prob, 0.5),
        "at_selected": threshold_metrics(y_true, y_prob, selected_threshold),
    }
