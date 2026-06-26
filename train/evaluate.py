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

from models.f2c_digraph import CfpStudent
from train.common import build_loader, load_semantic_initialization
from utils.checkpoint import load_checkpoint
from utils.config import load_config
from utils.metrics import compute_metrics


@torch.no_grad()
def run(config_path: str, checkpoint: str, fold: int):
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.get("require_cuda", False) and device.type != "cuda":
        raise RuntimeError("CUDA is required by config but torch.cuda.is_available() is False.")
    retinabert_ckpt = cfg["paths"].get("retinabert_ckpt", "outputs/checkpoints/retinabert.pth")
    e0, a0 = load_semantic_initialization(retinabert_ckpt, cfg, device)
    from models.semantic_dynamic_encoder import SemanticDynamicEncoder

    sem_encoder = SemanticDynamicEncoder(cfg["model"]["embed_dim"], cfg["model"]["heads"], cfg["model"]["semantic_layers"], cfg["model"]["dropout"]).to(device)
    ckpt = load_checkpoint(checkpoint, map_location="cpu")
    state = ckpt["model"]
    sem_state = {k.replace("semantic_encoder.", ""): v for k, v in state.items() if k.startswith("semantic_encoder.")}
    if sem_state:
        sem_encoder.load_state_dict(sem_state, strict=False)
    h_sem, _ = sem_encoder(e0.to(device), a0.to(device))
    student = CfpStudent(
        len(cfg["data"]["classes"]),
        cfg["model"]["embed_dim"],
        pretrained=False,
        heads=cfg["model"]["heads"],
        backbone=cfg["model"].get("backbone", "resnet50"),
    ).to(device)
    student_state = {k.replace("student.", ""): v for k, v in state.items() if k.startswith("student.")}
    if not student_state:
        student_state = {k.replace("module.student.", ""): v for k, v in state.items() if k.startswith("module.student.")}
    student.load_state_dict(student_state, strict=False)
    student.eval()
    loader = build_loader(cfg, "test", fold, train=False, require_masks=False)
    y_true, y_prob = [], []
    for batch in tqdm(loader, desc="evaluate"):
        logits = student(batch["cfp"].to(device), h_sem)["logits"]
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
