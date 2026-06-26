"""Patient-level five-fold split generation and reporting."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
try:
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
except ImportError:
    StratifiedGroupKFold = None
    StratifiedShuffleSplit = None

from .csv_parser import CsvSchema, load_and_infer, normalize_path_string


def _patient_labels(df: pd.DataFrame, schema: CsvSchema) -> pd.DataFrame:
    labels = (
        df[[schema.patient_col, schema.label_col]]
        .dropna()
        .astype(str)
        .groupby(schema.patient_col)[schema.label_col]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
    )
    return labels


def create_patient_level_folds(
    csv_path: str,
    output: str,
    n_folds: int = 5,
    seed: int = 42,
    val_fraction: float = 0.15,
) -> pd.DataFrame:
    """Create paper-compatible patient-level stratified 5-fold assignments."""
    df, schema = load_and_infer(csv_path)
    labels = _patient_labels(df, schema)
    if labels[schema.patient_col].nunique() < n_folds:
        raise ValueError("Not enough unique patients to build patient-level folds.")

    fold_df = labels.copy()
    used_existing = False
    for col in schema.fold_cols:
        tmp = df[[schema.patient_col, col]].dropna().copy()
        if tmp.empty:
            continue
        per_patient = tmp.astype({schema.patient_col: str}).groupby(schema.patient_col)[col].nunique()
        unique_folds = sorted(pd.to_numeric(tmp[col], errors="coerce").dropna().astype(int).unique().tolist())
        if per_patient.max() == 1 and len(unique_folds) == n_folds:
            mapping = tmp.astype({schema.patient_col: str}).drop_duplicates(schema.patient_col)
            mapping["fold"] = pd.to_numeric(mapping[col], errors="coerce").astype(int)
            fold_df = fold_df.merge(mapping[[schema.patient_col, "fold"]], on=schema.patient_col, how="left")
            if not fold_df["fold"].isna().any():
                used_existing = True
                break
    if not used_existing and StratifiedGroupKFold is not None:
        patient_to_fold: Dict[str, int] = {}
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        x = np.zeros(len(labels))
        y = labels[schema.label_col].astype(str).to_numpy()
        groups = labels[schema.patient_col].astype(str).to_numpy()
        for fold, (_, test_idx) in enumerate(splitter.split(x, y, groups=groups)):
            for patient in groups[test_idx]:
                patient_to_fold[patient] = fold
        fold_df["fold"] = fold_df[schema.patient_col].astype(str).map(patient_to_fold)
    elif not used_existing:
        rng = np.random.default_rng(seed)
        fold_df["fold"] = -1
        for _, part in fold_df.groupby(schema.label_col):
            indices = part.index.to_numpy().copy()
            rng.shuffle(indices)
            for rank, idx in enumerate(indices):
                fold_df.loc[idx, "fold"] = rank % n_folds
    fold_df["val_in_fold"] = False
    for fold in range(n_folds):
        train_patients = fold_df[fold_df["fold"] != fold].copy()
        if StratifiedShuffleSplit is not None and train_patients[schema.label_col].nunique() > 1 and len(train_patients) > 2:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed + fold)
            idx = np.arange(len(train_patients))
            _, val_idx = next(sss.split(idx, train_patients[schema.label_col].astype(str).to_numpy()))
            val_patients = set(train_patients.iloc[val_idx][schema.patient_col].astype(str))
            fold_df.loc[fold_df[schema.patient_col].astype(str).isin(val_patients), f"val_for_fold_{fold}"] = True
        elif len(train_patients) > 2:
            val_patients = []
            rng = np.random.default_rng(seed + fold)
            for _, part in train_patients.groupby(schema.label_col):
                idxs = part.index.to_numpy().copy()
                rng.shuffle(idxs)
                n_val = max(1, int(round(len(idxs) * val_fraction)))
                val_patients.extend(train_patients.loc[idxs[:n_val], schema.patient_col].astype(str).tolist())
            fold_df.loc[fold_df[schema.patient_col].astype(str).isin(set(val_patients)), f"val_for_fold_{fold}"] = True
        else:
            fold_df[f"val_for_fold_{fold}"] = False
    for fold in range(n_folds):
        col = f"val_for_fold_{fold}"
        if col not in fold_df:
            fold_df[col] = False
        fold_df[col] = fold_df[col].fillna(False).astype(bool)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(output, index=False)
    return fold_df


def attach_split(df: pd.DataFrame, schema: CsvSchema, split_csv: str, fold: int) -> pd.DataFrame:
    """Attach train/val/test assignment for a selected fold."""
    split_df = pd.read_csv(split_csv)
    patient_col = schema.patient_col
    if patient_col not in split_df.columns:
        raise ValueError(f"Split file does not contain patient column {patient_col}.")
    merged = df.copy()
    merged[patient_col] = merged[patient_col].astype(str)
    split_df[patient_col] = split_df[patient_col].astype(str)
    merged = merged.merge(split_df[[patient_col, "fold", f"val_for_fold_{fold}"]], on=patient_col, how="left")
    if merged["fold"].isna().any():
        raise ValueError("Some patients in the CSV are missing from the split file.")
    val_col = f"val_for_fold_{fold}"
    merged["split"] = "train"
    merged.loc[merged["fold"].astype(int) == fold, "split"] = "test"
    merged.loc[(merged["fold"].astype(int) != fold) & merged[val_col].astype(bool), "split"] = "val"
    return merged


def log_fold_summary(df: pd.DataFrame, schema: CsvSchema) -> str:
    """Return a human-readable split summary with patients, CFP, FFA, and labels."""
    lines: List[str] = []
    for split in ("train", "val", "test"):
        part = df[df["split"] == split]
        ffa_count = 0
        for col in schema.ffa_cols:
            ffa_count += part[col].apply(lambda x: bool(normalize_path_string(x))).sum()
        lines.append(f"[{split}] patients={part[schema.patient_col].nunique()} cfp={len(part)} ffa_paths={ffa_count}")
        lines.append(str(part[schema.label_col].value_counts().to_dict()))
    return "\n".join(lines)
