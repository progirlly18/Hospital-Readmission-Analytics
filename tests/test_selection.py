import inspect

from src.ml.selection import select_candidate
from src.ml.train import train_and_select


def metrics(roc, pr, brier):
    return {"roc_auc": roc, "pr_auc": pr, "brier": brier}


def test_clearly_best_model_is_selected():
    selected, _ = select_candidate({
        "logistic_regression": metrics(0.62, 0.18, 0.098),
        "random_forest": metrics(0.64, 0.20, 0.098),
        "xgboost": metrics(0.67, 0.23, 0.097),
    })
    assert selected == "xgboost"


def test_tie_goes_to_simplest_model():
    selected, reasons = select_candidate({
        "logistic_regression": metrics(0.665, 0.225, 0.0980),
        "random_forest": metrics(0.668, 0.226, 0.0981),
        "xgboost": metrics(0.670, 0.229, 0.0979),
    })
    assert selected == "logistic_regression"
    assert any("simplest" in line for line in reasons)


def test_better_calibration_wins_among_eligible():
    selected, _ = select_candidate({
        "logistic_regression": metrics(0.665, 0.225, 0.1100),  # poorly calibrated
        "random_forest": metrics(0.668, 0.226, 0.0980),
        "xgboost": metrics(0.670, 0.229, 0.0985),
    })
    assert selected == "random_forest"


def test_high_pr_auc_gap_makes_model_ineligible():
    selected, _ = select_candidate({
        "logistic_regression": metrics(0.668, 0.20, 0.090),  # PR-AUC 0.03 behind
        "xgboost": metrics(0.670, 0.23, 0.098),
    })
    assert selected == "xgboost"


def test_selection_code_cannot_receive_test_data():
    """Model and threshold selection functions take no test-set arguments."""
    for function in (select_candidate, train_and_select):
        parameters = inspect.signature(function).parameters
        assert not any("test" in name for name in parameters), parameters
