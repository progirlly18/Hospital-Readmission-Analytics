"""
Phase 7: documented retraining and review criteria, checked against a
monitoring report. Nothing here retrains anything: a triggered criterion is
a recommendation for a person to review, retrain and re-register a model
through the Phase 3-4 pipeline.

Reference values come from the validation split, the data the decision
threshold was chosen on.
"""

from src.monitoring.drift import SIGNIFICANT_PSI

# Drift on an important input or on the predictions themselves.
FEATURE_PSI_LIMIT = SIGNIFICANT_PSI          # 0.25
PREDICTION_PSI_LIMIT = SIGNIFICANT_PSI       # 0.25
KEY_FEATURES = 10                            # top features by permutation importance
CONSECUTIVE_BATCHES = 2                      # drift must persist before retraining
# Labelled performance against the validation reference.
ROC_AUC_DROP_LIMIT = 0.05
ECE_LIMIT = 0.03                             # validation ECE is about 0.007
FLAGGED_RATE_CHANGE_LIMIT = 0.10             # absolute change in share flagged high risk
# Subgroups (reported groups with at least 30 positives only).
SUBGROUP_ROC_AUC_GAP_LIMIT = 0.10

CRITERIA = (
    ("feature_drift", f"PSI > {FEATURE_PSI_LIMIT} on one of the {KEY_FEATURES} most "
                      f"important features for {CONSECUTIVE_BATCHES} consecutive batches"),
    ("prediction_drift", f"PSI of predicted probabilities > {PREDICTION_PSI_LIMIT}"),
    ("roc_auc_drop", f"ROC-AUC more than {ROC_AUC_DROP_LIMIT} below the validation reference"),
    ("calibration", f"expected calibration error above {ECE_LIMIT}"),
    ("flagged_rate", f"share flagged high risk changes by more than "
                     f"{FLAGGED_RATE_CHANGE_LIMIT} from the reference"),
    ("subgroup_gap", f"a subgroup's ROC-AUC more than {SUBGROUP_ROC_AUC_GAP_LIMIT} "
                     "below the batch's overall ROC-AUC (review for fairness)"),
)


def batch_findings(batch, reference, key_features):
    """Criteria met by one batch on its own (drift persistence is checked later)."""
    findings = []
    drifted = [row["feature"] for row in batch["feature_drift"]
               if row["feature"] in key_features and row["psi"] > FEATURE_PSI_LIMIT]
    if drifted:
        findings.append({"criterion": "feature_drift", "features": drifted})
    if batch["prediction_psi"] > PREDICTION_PSI_LIMIT:
        findings.append({"criterion": "prediction_drift",
                         "value": round(batch["prediction_psi"], 4)})

    performance = batch.get("performance")
    if performance:
        drop = reference["roc_auc"] - performance["roc_auc"]
        if drop > ROC_AUC_DROP_LIMIT:
            findings.append({"criterion": "roc_auc_drop", "value": round(drop, 4)})
        if performance["ece"] > ECE_LIMIT:
            findings.append({"criterion": "calibration",
                             "value": round(performance["ece"], 4)})
        change = abs(performance["flagged_rate"] - reference["flagged_rate"])
        if change > FLAGGED_RATE_CHANGE_LIMIT:
            findings.append({"criterion": "flagged_rate", "value": round(change, 4)})
        gaps = [
            f"{row['attribute']}={row['group']}"
            for row in batch.get("subgroups", [])
            if isinstance(row.get("positives"), int) and row["positives"] >= 30
            and row.get("roc_auc") is not None
            and performance["roc_auc"] - row["roc_auc"] > SUBGROUP_ROC_AUC_GAP_LIMIT
        ]
        if gaps:
            findings.append({"criterion": "subgroup_gap", "groups": gaps})
    return findings


def recommendation(batches, reference, key_features):
    """
    Findings per batch, in order, and the overall recommendation.

    Feature drift recommends retraining only when the same key feature drifts
    in CONSECUTIVE_BATCHES batches in a row; a single drifted batch is a
    warning. Every other finding recommends retraining on its own, except a
    subgroup gap, which calls for a fairness review.
    """
    per_batch, retrain, review, warnings = [], [], [], []
    streaks = {}
    for batch in batches:
        findings = batch_findings(batch, reference, key_features)
        per_batch.append({"batch": batch["name"], "findings": findings})
        drifted = next((f["features"] for f in findings
                        if f["criterion"] == "feature_drift"), [])
        streaks = {feature: streaks.get(feature, 0) + 1 for feature in drifted}
        for feature, run in streaks.items():
            if run >= CONSECUTIVE_BATCHES:
                retrain.append(f"{batch['name']}: {feature} drifted in {run} consecutive batches")
            else:
                warnings.append(f"{batch['name']}: {feature} drifted (not yet persistent)")
        for finding in findings:
            if finding["criterion"] in ("prediction_drift", "roc_auc_drop",
                                        "calibration", "flagged_rate"):
                retrain.append(f"{batch['name']}: {finding['criterion']}")
            elif finding["criterion"] == "subgroup_gap":
                review.append(f"{batch['name']}: subgroup gap {finding['groups']}")
    return {
        "criteria": [{"criterion": name, "rule": rule} for name, rule in CRITERIA],
        "batches": per_batch,
        "retraining_recommended": bool(retrain),
        "retrain_reasons": retrain,
        "fairness_review": review,
        "warnings": warnings,
    }
