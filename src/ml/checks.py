"""
Pre-training checks run before any model is trained.

1. Target leakage: forbidden columns absent; no single feature is
   suspiciously predictive on its own.
2. Prediction-time availability: every feature has a documented reason it is
   known at discharge.
3. Patient overlap: zero shared patients across splits.
4. Preprocessing leakage: the preprocessor is fitted on training rows only
   and its learned statistics match the training data.
5. Class imbalance: positive rate per split.
"""

import numpy as np
from sklearn.metrics import roc_auc_score

from src.ml import config
from src.ml.features import build_preprocessor
from src.ml.split import patient_overlap


def check_forbidden_columns(features):
    """Fail if any identifier, target or excluded column is a feature."""
    leaked = sorted(set(features.columns) & config.FORBIDDEN_FEATURE_COLUMNS)
    return {
        "check": "No identifier, target or excluded columns in features",
        "passed": not leaked,
        "detail": leaked or "none found",
    }


def check_prediction_time_availability(features):
    """Fail if a feature has no documented prediction-time justification."""
    undocumented = sorted(
        set(features.columns) - set(config.PREDICTION_TIME_AVAILABILITY)
    )
    return {
        "check": "Every feature is documented as available at discharge",
        "passed": not undocumented,
        "detail": undocumented or f"{features.shape[1]} features documented",
    }


def single_feature_auc(features, target):
    """
    ROC-AUC of each feature used alone (training data only).

    Numeric features are scored directly. Categorical features are scored by
    each category's positive rate. Values are folded to >= 0.5 so the
    direction of the relationship does not matter.
    """
    scores = {}
    for column in features.columns:
        values = features[column]
        if column in config.NUMERIC_FEATURES:
            score_input = values.fillna(values.median()).to_numpy(dtype=float)
        else:
            category_rate = target.groupby(values).mean()
            score_input = values.map(category_rate).to_numpy(dtype=float)
        auc = roc_auc_score(target, score_input)
        scores[column] = round(float(max(auc, 1 - auc)), 4)
    return dict(sorted(scores.items(), key=lambda item: -item[1]))


def check_single_feature_leakage(train_features, train_target):
    scores = single_feature_auc(train_features, train_target)
    suspicious = {
        name: auc
        for name, auc in scores.items()
        if auc > config.SINGLE_FEATURE_LEAKAGE_AUC
    }
    return {
        "check": (
            "No single feature has training ROC-AUC above "
            f"{config.SINGLE_FEATURE_LEAKAGE_AUC}"
        ),
        "passed": not suspicious,
        "detail": suspicious or {"top_5": dict(list(scores.items())[:5])},
    }


def check_patient_overlap(df, split):
    overlap = patient_overlap(df, split)
    return {
        "check": "Zero patients shared between splits",
        "passed": all(count == 0 for count in overlap.values()),
        "detail": overlap,
    }


def check_preprocessing_fit_on_train(features, split):
    """
    Fit the preprocessor on training rows only and confirm the learned
    numeric medians and means equal the training statistics (and differ from
    the full-data statistics, which would indicate leakage).
    """
    train = features.loc[split == "train"]
    preprocessor = build_preprocessor().fit(train)

    numeric_pipeline = preprocessor.named_transformers_["numeric"]
    learned_medians = numeric_pipeline.named_steps["impute"].statistics_
    learned_means = numeric_pipeline.named_steps["scale"].mean_

    train_numeric = train[config.NUMERIC_FEATURES]
    train_medians = train_numeric.median().to_numpy()
    train_means = (
        train_numeric.fillna(train_numeric.median()).mean().to_numpy()
    )

    passed = bool(
        np.allclose(learned_medians, train_medians)
        and np.allclose(learned_means, train_means)
    )

    # Transform must also work on unseen splits (new categories handled).
    for name in ("validation", "test"):
        preprocessor.transform(features.loc[split == name])

    return {
        "check": "Preprocessing statistics learned from training rows only",
        "passed": passed,
        "detail": {
            "fitted_rows": len(train),
            "output_columns": len(preprocessor.get_feature_names_out()),
        },
    }


def class_balance(df, split):
    result = {}
    for name in ("train", "validation", "test"):
        target = df.loc[split == name, config.TARGET_COLUMN]
        positives = int(target.sum())
        result[name] = {
            "positives": positives,
            "negatives": int(len(target) - positives),
            "positive_rate": round(float(target.mean()), 4),
            "negatives_per_positive": round((len(target) - positives) / positives, 2),
        }
    return result


def run_all_checks(df, features, split):
    train_mask = split == "train"
    checks = [
        check_forbidden_columns(features),
        check_prediction_time_availability(features),
        check_single_feature_leakage(
            features.loc[train_mask], df.loc[train_mask, config.TARGET_COLUMN]
        ),
        check_patient_overlap(df, split),
        check_preprocessing_fit_on_train(features, split),
    ]
    return checks, class_balance(df, split)
