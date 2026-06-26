"""Run paper-style 5-fold patient-level teacher/student training."""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.csv_parser import load_and_infer
from data.splits import create_patient_level_folds, log_fold_summary, attach_split
from train.evaluate import run as evaluate_run
from train.train_cfp_student import run as train_student
from train.train_ffa_teacher import run as train_teacher
from utils.config import load_config
from utils.metrics import mean_std


def run(config_path: str, csv_path: str):
    cfg = load_config(config_path)
    cfg["paths"]["csv"] = csv_path
    split_path = Path(cfg["paths"]["split_csv"])
    if not split_path.exists():
        create_patient_level_folds(csv_path, str(split_path), cfg["data"]["n_folds"], cfg.get("seed", 42), cfg["data"]["val_fraction"])
    df, schema = load_and_infer(csv_path)
    for fold in range(cfg["data"]["n_folds"]):
        merged = attach_split(df, schema, str(split_path), fold)
        print(f"Fold {fold}\n{log_fold_summary(merged, schema)}")
        train_teacher(config_path, fold)
        train_student(
            config_path,
            fold,
            f"{cfg['paths']['checkpoints']}/fold{fold}_ffa_teacher.pth",
            cfg["paths"]["retinabert_ckpt"],
        )
    rows = []
    for fold in range(cfg["data"]["n_folds"]):
        metrics = evaluate_run(config_path, f"{cfg['paths']['checkpoints']}/fold{fold}_student_best.pth", fold)
        rows.append({"fold": fold, "acc": metrics["acc"], "auc": metrics["auc_macro_ovr"]})
    out_dir = Path(cfg["paths"]["results"])
    out_dir.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(rows)
    result_df.to_csv(out_dir / "retinadual_5fold_results.csv", index=False)
    acc_m, acc_s = mean_std(result_df["acc"])
    auc_m, auc_s = mean_std(result_df["auc"])
    lines = [*(f"fold {r.fold}: ACC={r.acc:.6f}, AUC={r.auc:.6f}" for r in result_df.itertuples()), f"mean ACC ± std: {acc_m:.6f} ± {acc_s:.6f}", f"mean AUC ± std: {auc_m:.6f} ± {auc_s:.6f}"]
    (out_dir / "summary_mean_std.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()
    run(args.config, args.csv)

