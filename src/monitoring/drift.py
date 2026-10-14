"""
Phase 7: data and prediction drift, measured with the Population Stability
Index (PSI) against a fixed baseline (the training split).

    PSI = sum over bins of (current% - baseline%) * ln(current% / baseline%)

Numeric features use 10 bins with edges at the baseline's quantiles; equal
quantile edges are merged, so a feature with few distinct values gets fewer
bins. Categorical features use the baseline's categories plus one bin for
values the baseline never had. Empty bins are given a small floor so the
logarithm stays finite.

Usual reading (and the bands used here): below 0.10 stable, 0.10-0.25
moderate shift, above 0.25 significant shift.
"""

import numpy as np
import pandas as pd

from src.ml import config

N_BINS = 10
FLOOR = 1e-4
MODERATE_PSI = 0.10
SIGNIFICANT_PSI = 0.25
UNSEEN = "__unseen__"


def psi_from_shares(baseline_share, current_share):
    baseline = np.clip(np.asarray(baseline_share, dtype=float), FLOOR, None)
    current = np.clip(np.asarray(current_share, dtype=float), FLOOR, None)
    return float(np.sum((current - baseline) * np.log(current / baseline)))


def numeric_edges(baseline, n_bins=N_BINS):
    """Inner bin edges at the baseline's quantiles (duplicates merged)."""
    values = pd.to_numeric(pd.Series(baseline), errors="coerce").dropna()
    quantiles = np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.unique(quantiles)


def numeric_shares(values, edges):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy()
    counts = np.bincount(np.searchsorted(edges, values, side="right"),
                         minlength=len(edges) + 1)
    return counts / max(counts.sum(), 1)


def categorical_shares(values, categories):
    values = pd.Series(values).astype(str)
    values = values.where(values.isin(categories), UNSEEN)
    counts = values.value_counts().reindex([*categories, UNSEEN], fill_value=0)
    return (counts / max(counts.sum(), 1)).to_numpy()


def status(psi):
    if psi > SIGNIFICANT_PSI:
        return "significant"
    if psi >= MODERATE_PSI:
        return "moderate"
    return "stable"


class DriftBaseline:
    """Bin edges and category shares learned once from the baseline rows."""

    def __init__(self, features, numeric=config.NUMERIC_FEATURES,
                 categorical=config.CATEGORICAL_FEATURES):
        self.numeric = {
            column: (edges := numeric_edges(features[column]),
                     numeric_shares(features[column], edges))
            for column in numeric
        }
        self.categorical = {}
        for column in categorical:
            categories = sorted(features[column].astype(str).unique())
            self.categorical[column] = (
                categories, categorical_shares(features[column], categories))

    def feature_psi(self, features):
        """One row per feature: PSI and its status for the current batch."""
        rows = []
        for column, (edges, baseline) in self.numeric.items():
            psi = psi_from_shares(baseline, numeric_shares(features[column], edges))
            rows.append({"feature": column, "type": "numeric", "psi": psi})
        for column, (categories, baseline) in self.categorical.items():
            current = categorical_shares(features[column], categories)
            rows.append({"feature": column, "type": "categorical",
                         "psi": psi_from_shares(baseline, current),
                         "unseen_share": float(current[-1])})
        table = pd.DataFrame(rows)
        table["status"] = table["psi"].map(status)
        return table.sort_values("psi", ascending=False).reset_index(drop=True)


def prediction_psi(baseline_probabilities, current_probabilities):
    """PSI of the predicted probabilities (no labels needed)."""
    edges = numeric_edges(baseline_probabilities)
    return psi_from_shares(numeric_shares(baseline_probabilities, edges),
                           numeric_shares(current_probabilities, edges))
