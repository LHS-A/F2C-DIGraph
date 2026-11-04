"""Evaluate a trained CFP-only student checkpoint on a selected fold."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.f2c_digraph import CfpInferenceModel
from train.common import build_loader
from utils.checkpoint import load_checkpoint
from utils.config import load_config
from utils.metrics import compute_metrics


@torch.no_grad()
def run(config_path: str, checkpoint: str, fold: int):
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.get("require_cuda", False) and device.type != "cuda":
        raise RuntimeError("CUDA is required by config but torch.cuda.is_available() is False.")
    ckpt = load_checkpoint(checkpoint, map_location="cpu")
    if "inference_model" not in ckpt:
        raise ValueError("Checkpoint predates the CFP-only export format; retrain or export the updated student.")
    inference_state = ckpt["inference_model"]
    semantic_basis = inference_state["semantic_basis"]
    student = CfpInferenceModel(
        len(cfg["data"]["classes"]),
        semantic_basis,
        embed_dim=cfg["model"]["embed_dim"],
        pretrained=False,
        heads=cfg["model"]["heads"],
        dropout=cfg["model"]["dropout"],
        backbone=cfg["model"].get("backbone", "resnet50"),
    ).to(device)
    student.load_state_dict(inference_state, strict=True)
    student.eval()
    loader = build_loader(cfg, "test", fold, train=False, require_masks=False)
    y_true, y_prob = [], []
    for batch in tqdm(loader, desc="evaluate"):
        logits = student(batch["cfp"].to(device))["logits"]
        y_true.extend(batch["label"].numpy().tolist())
        y_prob.append(torch.softmax(logits, dim=-1).cpu().numpy())
    metrics = compute_metrics(y_true, np.concatenate(y_prob, axis=0), cfg["data"]["classes"])
    out_dir = Path(cfg["paths"]["results"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"fold{fold}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(metrics["confusion_matrix"], index=cfg["data"]["classes"], columns=cfg["data"]["classes"]).to_csv(out_dir / f"fold{fold}_confusion_matrix.csv")
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()
    run(args.config, args.checkpoint, args.fold)
