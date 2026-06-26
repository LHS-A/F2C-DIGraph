"""Train CFP-only F2C-DIGraph student with semantic and topological CID."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.f2c_digraph import CfpStudent, F2CDIGraph, FfaTeacher
from models.semantic_dynamic_encoder import SemanticDynamicEncoder
from models.topological_dynamic_encoder import TopologicalDynamicEncoder
from train.common import build_loader, device_and_wrap, load_semantic_initialization
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.config import load_config
from utils.logger import setup_logger
from utils.metrics import compute_metrics
from utils.scheduler import build_warmup_cosine_scheduler, distill_weight
from utils.seed import seed_everything


@torch.no_grad()
def evaluate_student(model, loader, device, classes):
    model.eval()
    y_true, y_prob = [], []
    for batch in tqdm(loader, desc="student evaluate", leave=False):
        for key in ("cfp", "ffa", "label"):
            batch[key] = batch[key].to(device)
        out = model(batch)
        y_true.extend(batch["label"].cpu().numpy().tolist())
        y_prob.append(torch.softmax(out["logits"], dim=-1).cpu().numpy())
    return compute_metrics(y_true, np.concatenate(y_prob, axis=0), list(classes))


def run(config_path: str, fold: int, teacher_ckpt: str, retinabert_ckpt: str):
    cfg = load_config(config_path)
    seed_everything(cfg.get("seed", 42) + fold, cfg.get("deterministic", True))
    logger = setup_logger(f"student_fold{fold}", f"{cfg['paths']['logs']}/fold{fold}_student.log")
    require_masks = cfg.get("strict_reproduction", True)
    train_loader = build_loader(cfg, "train", fold, train=True, require_masks=require_masks)
    val_loader = build_loader(cfg, "val", fold, train=False, require_masks=require_masks)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.get("require_cuda", False) and device.type != "cuda":
        raise RuntimeError("CUDA is required by config but torch.cuda.is_available() is False.")

    teacher = FfaTeacher(
        len(cfg["data"]["classes"]),
        cfg["model"]["embed_dim"],
        pretrained=False,
        backbone=cfg["model"].get("backbone", "resnet50"),
    )
    teacher_state = load_checkpoint(teacher_ckpt, map_location="cpu")["model"]
    teacher.load_state_dict(teacher_state)
    for p in teacher.parameters():
        p.requires_grad = False
    student = CfpStudent(
        len(cfg["data"]["classes"]),
        cfg["model"]["embed_dim"],
        cfg["model"].get("pretrained", True),
        cfg["model"]["heads"],
        cfg["model"].get("backbone", "resnet50"),
    )
    sem_encoder = SemanticDynamicEncoder(cfg["model"]["embed_dim"], cfg["model"]["heads"], cfg["model"]["semantic_layers"], cfg["model"]["dropout"])
    topo_encoder = TopologicalDynamicEncoder(cfg["model"]["embed_dim"], cfg["model"]["topology_layers"], cfg["model"]["topology_bases"])
    e0, a0 = load_semantic_initialization(retinabert_ckpt, cfg, device)
    model = F2CDIGraph(teacher, student, sem_encoder, topo_encoder, e0, a0, cfg["model"]["embed_dim"])
    device, model = device_and_wrap(model, cfg.get("multi_gpu", True))
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    scheduler = build_warmup_cosine_scheduler(opt, cfg["training"]["warmup_epochs"], cfg["training"]["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.get("mixed_precision", True) and device.type == "cuda")
    best_auc = -1.0
    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        totals = {"loss": 0.0, "loss_cls": 0.0, "loss_sem": 0.0, "loss_topo": 0.0}
        lam_sem = distill_weight(epoch, cfg["training"]["epochs"], cfg["training"]["lambda_sem_max"], cfg["training"]["distill_warmup_fraction"])
        lam_topo = distill_weight(epoch, cfg["training"]["epochs"], cfg["training"]["lambda_topo_max"], cfg["training"]["distill_warmup_fraction"])
        for batch in tqdm(train_loader, desc=f"student fold {fold} epoch {epoch + 1}"):
            for key in ("cfp", "ffa", "label"):
                batch[key] = batch[key].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                outputs = model(batch)
                loss_dict = model.module.losses(outputs, batch["label"], lam_sem, lam_topo) if hasattr(model, "module") else model.losses(outputs, batch["label"], lam_sem, lam_topo)
            scaler.scale(loss_dict["loss"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, cfg["training"].get("grad_clip_norm", 1.0))
            scaler.step(opt)
            scaler.update()
            for key in totals:
                totals[key] += float(loss_dict[key].item())
        scheduler.step()
        metrics = evaluate_student(model, val_loader, device, cfg["data"]["classes"])
        avg_losses = {k: v / max(1, len(train_loader)) for k, v in totals.items()}
        logger.info("epoch=%d lambda_sem=%.4f lambda_topo=%.4f losses=%s val=%s", epoch + 1, lam_sem, lam_topo, json.dumps(avg_losses), json.dumps(metrics))
        auc = metrics.get("auc_macro_ovr", float("nan"))
        if auc == auc and auc > best_auc:
            best_auc = auc
            state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            save_checkpoint(f"{cfg['paths']['checkpoints']}/fold{fold}_student_best.pth", {"model": state, "metrics": metrics, "config": cfg})
    return best_auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--teacher_ckpt", required=True)
    parser.add_argument("--retinabert_ckpt", required=True)
    args = parser.parse_args()
    run(args.config, args.fold, args.teacher_ckpt, args.retinabert_ckpt)
