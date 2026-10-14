import numpy as np
import pytest
from sklearn.metrics import f1_score

from src.ml.metrics import (
    best_f1_threshold,
    evaluate,
    expected_calibration_error,
    probability_metrics,
    threshold_metrics,
)


def test_ece_zero_when_probabilities_match_observed_rates():
    # Two bins: predictions 0.0 with outcome 0, predictions 1.0 with outcome 1.
    y_true = np.array([0] * 50 + [1] * 50)
    y_prob = y_true.astype(float)
    assert expected_calibration_error(y_true, y_prob, n_bins=2) == pytest.approx(0.0)


def test_ece_detects_overconfident_probabilities():
    # Everyone predicted 0.5 but only 10% are readmitted -> gap of 0.4.
    y_true = np.array([1] * 10 + [0] * 90)
    y_prob = np.full(100, 0.5)
    assert expected_calibration_error(y_true, y_prob) == pytest.approx(0.4)


def test_brier_and_auc_known_values():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.35, 0.8])
    m = probability_metrics(y_true, y_prob)
    assert m["roc_auc"] == pytest.approx(0.75)
    expected_brier = np.mean((y_prob - y_true) ** 2)
    assert m["brier"] == pytest.approx(expected_brier)
    assert m["positives"] == 2


def test_threshold_metrics_confusion_matrix():
    y_true = np.array([0, 0, 1, 1, 1])
    y_prob = np.array([0.2, 0.7, 0.6, 0.4, 0.9])
    m = threshold_metrics(y_true, y_prob, 0.5)
    assert m["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 2}
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert "readmitted <30" in m["classification_report"]


def test_best_f1_threshold_reproduces_reported_f1():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, 300)
    y_prob = np.clip(y_true * 0.3 + rng.random(300) * 0.7, 0, 1)
    threshold, best_f1 = best_f1_threshold(y_true, y_prob)
    achieved = f1_score(y_true, (y_prob >= threshold).astype(int))
    assert achieved == pytest.approx(best_f1)
    # No other threshold on a fine grid does better.
    for t in np.linspace(0.01, 0.99, 99):
        assert f1_score(y_true, (y_prob >= t).astype(int)) <= best_f1 + 1e-9


def test_evaluate_reports_both_thresholds():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.3, 0.2, 0.9])
    result = evaluate(y_true, y_prob, selected_threshold=0.25)
    assert result["at_0_5"]["threshold"] == 0.5
    assert result["at_selected"]["threshold"] == 0.25
    assert result["at_selected"]["recall"] == 1.0
    assert result["at_0_5"]["recall"] == 0.5


def test_ece_does_not_depend_on_row_order():
    rng = np.random.default_rng(2)
    y_true = rng.integers(0, 2, 200)
    y_prob = np.round(rng.random(200), 1)  # many tied predictions
    order = rng.permutation(200)
    assert expected_calibration_error(y_true, y_prob) == pytest.approx(
        expected_calibration_error(y_true[order], y_prob[order])
    )
