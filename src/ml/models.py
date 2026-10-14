"""
The three required models and their small hyperparameter grids.

Every model is a full sklearn Pipeline: preprocessing (fitted on training
data only) followed by the classifier. Class imbalance is handled with
class weights computed from the training data only; nothing is resampled.
"""

from dataclasses import dataclass, field

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.ml import config
from src.ml.features import build_preprocessor


# Simpler models first. Used to break ties in model selection.
SIMPLICITY_ORDER = ["logistic_regression", "random_forest", "xgboost"]


@dataclass
class ModelSpec:
    name: str
    display_name: str
    pipeline: Pipeline
    param_grid: dict = field(default_factory=dict)


def negatives_per_positive(y_train):
    """Ratio used by XGBoost's scale_pos_weight (training labels only)."""
    positives = int(y_train.sum())
    return (len(y_train) - positives) / positives


def build_model_specs(y_train, seed=config.RANDOM_SEED, n_jobs=-1):
    """
    Create the three model pipelines.

    y_train is used only to set XGBoost's class weight; it must be the
    TRAINING labels.
    """
    logistic = ModelSpec(
        name="logistic_regression",
        display_name="Logistic Regression",
        pipeline=Pipeline([
            ("preprocess", build_preprocessor(scale_numeric=True)),
            ("model", LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=seed,
            )),
        ]),
        # C = inverse regularisation strength.
        param_grid={"model__C": [0.01, 0.1, 1.0]},
    )

    forest = ModelSpec(
        name="random_forest",
        display_name="Random Forest",
        pipeline=Pipeline([
            # Trees do not need scaled inputs.
            ("preprocess", build_preprocessor(scale_numeric=False)),
            ("model", RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced_subsample",
                max_features="sqrt",
                random_state=seed,
                n_jobs=n_jobs,
            )),
        ]),
        param_grid={
            "model__max_depth": [8, 16],
            "model__min_samples_leaf": [10, 50],
        },
    )

    xgboost = ModelSpec(
        name="xgboost",
        display_name="XGBoost",
        pipeline=Pipeline([
            ("preprocess", build_preprocessor(scale_numeric=False)),
            ("model", XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=negatives_per_positive(y_train),
                tree_method="hist",
                eval_metric="logloss",
                random_state=seed,
                n_jobs=n_jobs,
            )),
        ]),
        param_grid={
            "model__max_depth": [3, 5],
            "model__min_child_weight": [1, 10],
        },
    )

    return [logistic, forest, xgboost]
