"""Create patient-level 5-fold split CSV."""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.csv_parser import load_and_infer
from data.splits import attach_split, create_patient_level_folds, log_fold_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    split_df = create_patient_level_folds(args.csv, args.output, args.n_folds, args.seed)
    df, schema = load_and_infer(args.csv)
    for fold in range(args.n_folds):
        merged = attach_split(df, schema, args.output, fold)
        print(f"Fold {fold}")
        print(log_fold_summary(merged, schema))

