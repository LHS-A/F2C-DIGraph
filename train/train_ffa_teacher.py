"""Train the stacked-FFA ResNet-50 teacher."""

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.f2c_digraph import FfaTeacher
from train.common import build_loader, device_and_wrap, evaluate_classifier
from utils.checkpoint import save_checkpoint
from utils.config import load_config
from utils.logger import setup_logger
from utils.scheduler import build_warmup_cosine_scheduler
from utils.seed import seed_everything


def run(config_path: str, fold: int):
    cfg = load_config(config_path)
    seed_everything(cfg.get("seed", 42) + fold, cfg.get("deterministic", True))
    logger = setup_logger(f"teacher_fold{fold}", f"{cfg['paths']['logs']}/fold{fold}_teacher.log")
    train_loader = build_loader(cfg, "train", fold, train=True, require_masks=False)
    val_loader = build_loader(cfg, "val", fold, train=False, require_masks=False)
    model = FfaTeacher(
        len(cfg["data"]["classes"]),
        cfg["model"]["embed_dim"],
        cfg["model"].get("pretrained", True),
        cfg["model"].get("backbone", "resnet50"),
    )
    model.is_teacher = True
    device, model = device_and_wrap(model, cfg.get("multi_gpu", True))
    if cfg.get("require_cuda", False) and device.type != "cuda":
        raise RuntimeError("CUDA is required by config but torch.cuda.is_available() is False.")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    scheduler = build_warmup_cosine_scheduler(opt, cfg["training"]["warmup_epochs"], cfg["training"]["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.get("mixed_precision", True) and device.type == "cuda")
    best_auc = -1.0
    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        total = 0.0
        for batch in tqdm(train_loader, desc=f"teacher fold {fold} epoch {epoch + 1}"):
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(batch["ffa"].to(device))["logits"]
                loss = torch.nn.functional.cross_entropy(logits, batch["label"].to(device))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"].get("grad_clip_norm", 1.0))
            scaler.step(opt)
            scaler.update()
            total += float(loss.item())
        scheduler.step()
        metrics = evaluate_classifier(model, val_loader, device, cfg["data"]["classes"])
        logger.info("epoch=%d loss=%.6f val=%s", epoch + 1, total / max(1, len(train_loader)), json.dumps(metrics))
        auc = metrics.get("auc_macro_ovr", float("nan"))
        if auc == auc and auc > best_auc:
            best_auc = auc
            state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            save_checkpoint(f"{cfg['paths']['checkpoints']}/fold{fold}_ffa_teacher.pth", {"model": state, "metrics": metrics, "config": cfg})
    return best_auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()
    run(args.config, args.fold)
