"""Plots for the Phase 3 evaluation report (saved as PNG files)."""

import matplotlib

matplotlib.use("Agg")  # no display needed

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from src.ml.metrics import calibration_bins  # noqa: E402


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def reliability_plot(curves, path, n_bins=10, title="Reliability (validation)"):
    """
    curves: {label: (y_true, y_prob)}. Equal-frequency bins; each point is
    the mean predicted probability vs. the observed readmission rate.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")
    for label, (y_true, y_prob) in curves.items():
        bins = calibration_bins(y_true, y_prob, n_bins)
        ax.plot(
            [mean_predicted for mean_predicted, _, _ in bins],
            [observed for _, observed, _ in bins],
            marker="o", label=label,
        )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed 30-day readmission rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper left")
    _save(fig, path)


def roc_pr_plot(curves, path, title_suffix="(validation)"):
    fig, (roc_ax, pr_ax) = plt.subplots(1, 2, figsize=(12, 5))
    for label, (y_true, y_prob) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_ax.plot(fpr, tpr, label=label)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_ax.plot(recall, precision, label=label)
    roc_ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    roc_ax.set_xlabel("False positive rate")
    roc_ax.set_ylabel("True positive rate (recall)")
    roc_ax.set_title(f"ROC curve {title_suffix}")
    roc_ax.legend(loc="lower right")
    baseline = float(np.mean(next(iter(curves.values()))[0]))
    pr_ax.axhline(baseline, linestyle="--", color="grey", label=f"Base rate {baseline:.3f}")
    pr_ax.set_xlabel("Recall")
    pr_ax.set_ylabel("Precision")
    pr_ax.set_title(f"Precision-recall curve {title_suffix}")
    pr_ax.legend(loc="upper right")
    _save(fig, path)


def confusion_matrix_plot(cm, path, title):
    """cm: {"tn", "fp", "fn", "tp"}."""
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(matrix, cmap="Blues")
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, f"{value:,}", ha="center", va="center",
                color="white" if value > matrix.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Predicted 0", "Predicted 1"])
    ax.set_yticks([0, 1], ["Actual 0", "Actual 1"])
    ax.set_title(title)
    _save(fig, path)


def importance_plot(importance, path, title, top_n=20):
    """importance: DataFrame with columns feature, importance_mean, importance_std."""
    top = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
    ax.set_xlabel("Drop in validation ROC-AUC when the feature is shuffled")
    ax.set_title(title)
    _save(fig, path)
