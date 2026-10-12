"""
Patient-grouped, stratified train / validation / test split.

Why not chronological? The dataset has no admission or discharge dates, so
a true time-based split is not possible. Instead:

- Grouped: all encounters of a patient go to the same split, so the model is
  never evaluated on a patient it saw during training.
- Stratified: patients are split separately within two strata, "has at least
  one 30-day readmission" and "has none", so each split keeps a similar
  share of positives.

How it works (simple and identical on every machine):

1. Give every patient a pseudo-random but fixed position by hashing
   "<seed>:<patient_sk>" with SHA-256.
2. Within each stratum, walk through patients in that order and add up
   their encounters.
3. Patients whose cumulative position falls in the first 70% of the
   stratum's encounters go to train, the next 15% to validation and the
   rest to test.

Because whole patients are moved, the proportions are close to, but not
exactly, 70/15/15. The actual numbers are measured and reported.

This replaces sklearn's StratifiedGroupKFold, whose output changed between
scikit-learn versions during testing (1.5 vs 1.9 gave different splits).
"""

import hashlib

import numpy as np
import pandas as pd

from src.ml import config


SPLIT_NAMES = ("train", "validation", "test")


def patient_order_key(patient_id, seed):
    """Stable pseudo-random sort key for one patient."""
    text = f"{seed}:{patient_id}".encode("utf-8")
    return hashlib.sha256(text).hexdigest()


def assign_splits(
    df,
    target_column=config.TARGET_COLUMN,
    group_column=config.GROUP_COLUMN,
    seed=config.RANDOM_SEED,
    fractions=config.SPLIT_FRACTIONS,
):
    """Return a Series (same index as df) with 'train', 'validation' or 'test'."""
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("Split fractions must add up to 1.")

    patients = (
        df.groupby(group_column)[target_column]
        .agg(encounters="count", positives="sum")
        .reset_index()
    )
    patients["stratum"] = patients["positives"].gt(0).astype(int)
    patients["order_key"] = patients[group_column].map(
        lambda patient: patient_order_key(patient, seed)
    )
    patients = patients.sort_values(["stratum", "order_key"]).reset_index(drop=True)

    train_end = fractions[0]
    validation_end = fractions[0] + fractions[1]

    patient_split = pd.Series(index=patients.index, dtype=object)
    for _, stratum in patients.groupby("stratum"):
        cumulative = stratum["encounters"].cumsum()
        # Position of each patient's midpoint within the stratum (0..1).
        position = (cumulative - stratum["encounters"] / 2) / cumulative.iloc[-1]
        patient_split.loc[stratum.index] = np.where(
            position < train_end,
            "train",
            np.where(position < validation_end, "validation", "test"),
        )

    split_by_patient = dict(zip(patients[group_column], patient_split))
    return df[group_column].map(split_by_patient).rename("split")


def summarize_splits(df, split, target_column=config.TARGET_COLUMN,
                     group_column=config.GROUP_COLUMN):
    """Rows, share of rows, patients, positives and positive rate per split."""
    rows = []
    for name in SPLIT_NAMES:
        part = df.loc[split == name]
        rows.append({
            "split": name,
            "encounters": len(part),
            "share_of_encounters": round(len(part) / len(df), 4),
            "patients": int(part[group_column].nunique()),
            "positives": int(part[target_column].sum()),
            "negatives": int((part[target_column] == 0).sum()),
            "positive_rate": round(float(part[target_column].mean()), 4),
        })
    return pd.DataFrame(rows)


def patient_overlap(df, split, group_column=config.GROUP_COLUMN):
    """Number of patients shared by each pair of splits (all must be 0)."""
    patients = {
        name: set(df.loc[split == name, group_column]) for name in SPLIT_NAMES
    }
    return {
        "train_validation": len(patients["train"] & patients["validation"]),
        "train_test": len(patients["train"] & patients["test"]),
        "validation_test": len(patients["validation"] & patients["test"]),
    }
