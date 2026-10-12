"""
Phase 2 entry point: build the modelling dataset and the patient-grouped
split, then run the pre-training checks.

Run from the project root:

    python -m src.ml.prepare_data

Outputs (gitignored, regenerated on every run):
    artifacts/data_prep/split_assignments.csv  encounter_id, patient_sk, split
    artifacts/data_prep/data_prep_report.json  counts, checks, class balance
"""

import hashlib
import json
import sys

import pandas as pd

from src.ml import config
from src.ml.checks import run_all_checks
from src.ml.data import load_modelling_data
from src.ml.features import build_feature_frame
from src.ml.split import assign_splits, summarize_splits


def subgroup_counts(df, split, column):
    """Encounters and positives per subgroup and split, small cells suppressed."""
    table = (
        df.assign(split=split)
        .groupby(["split", column])[config.TARGET_COLUMN]
        .agg(encounters="count", positives="sum")
        .reset_index()
    )
    small = table["encounters"] < config.MIN_SUBGROUP_SIZE
    table["encounters"] = table["encounters"].astype(object)
    table["positives"] = table["positives"].astype(object)
    table.loc[small, ["encounters", "positives"]] = (
        f"<{config.MIN_SUBGROUP_SIZE}"
    )
    return table


def prepare(path=config.CLEANED_FILE):
    cohort, cohort_summary = load_modelling_data(path)
    features = build_feature_frame(cohort)
    split = assign_splits(cohort)
    checks, balance = run_all_checks(cohort, features, split)

    report = {
        "source_file": str(path.relative_to(config.PROJECT_ROOT))
        if path.is_relative_to(config.PROJECT_ROOT) else str(path),
        "random_seed": config.RANDOM_SEED,
        "target": "1 = readmitted '<30'; 0 = '>30' or 'NO'",
        "cohort": cohort_summary,
        "cohort_positive_rate": round(
            float(cohort[config.TARGET_COLUMN].mean()), 4
        ),
        "n_features": features.shape[1],
        "numeric_features": config.NUMERIC_FEATURES,
        "categorical_features": config.CATEGORICAL_FEATURES,
        "split_summary": summarize_splits(cohort, split).to_dict("records"),
        "checks": checks,
        "class_balance": balance,
        "subgroups": {
            column: subgroup_counts(cohort, split, column).to_dict("records")
            for column in ("race", "gender")
        },
    }
    return cohort, features, split, report


def main():
    cohort, _, split, report = prepare()

    config.DATA_PREP_DIR.mkdir(parents=True, exist_ok=True)

    assignments = cohort[[config.ENCOUNTER_ID_COLUMN, config.GROUP_COLUMN]].assign(
        split=split.to_numpy()
    )
    assignments_file = config.DATA_PREP_DIR / "split_assignments.csv"
    assignments.to_csv(assignments_file, index=False)

    # Fingerprint so a split made on another machine can be compared.
    report["split_assignments_sha256"] = hashlib.sha256(
        assignments_file.read_bytes()
    ).hexdigest()

    with open(config.DATA_PREP_DIR / "data_prep_report.json", "w",
              encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)

    print("PHASE 2 DATA PREPARATION")
    print("=" * 60)
    print(f"Source rows:  {report['cohort']['source_rows']:,}")
    for step in report["cohort"]["exclusions"]:
        print(
            f"  excluded  {step['encounters_removed']:>6,}  "
            f"(positives {step['positives_removed']:,})  {step['rule']}"
        )
    print(f"Cohort rows:  {report['cohort']['cohort_rows']:,}")
    print(f"Cohort positive rate: {report['cohort_positive_rate']:.2%}")
    print(f"Features: {report['n_features']}")
    print()
    print(pd.DataFrame(report["split_summary"]).to_string(index=False))
    print()
    print("Pre-training checks:")
    for check in report["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"  [{status}] {check['check']}: {check['detail']}")
    print()
    print(f"Split fingerprint (SHA-256): {report['split_assignments_sha256']}")
    print(f"Outputs written to {config.DATA_PREP_DIR}")

    if not all(check["passed"] for check in report["checks"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
