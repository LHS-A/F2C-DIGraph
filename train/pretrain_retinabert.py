"""Pretrain RetinaBERT on prepared UMLS retinal concept samples."""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.retinabert import RetinaBERT, UMLSContrastiveDataset, load_umls_knowledge, umls_contrastive_loss
from utils.checkpoint import save_checkpoint
from utils.config import load_config
from utils.logger import setup_logger
from utils.seed import seed_everything


def run(config_path: str):
    cfg = load_config(config_path)
    seed_everything(cfg.get("seed", 42), cfg.get("deterministic", True))
    logger = setup_logger("retinabert", f"{cfg['paths']['logs']}/pretrain_retinabert.log")
    samples = load_umls_knowledge(cfg["paths"]["umls_concepts"], cfg["retinabert"]["semantic_types"])
    dataset = UMLSContrastiveDataset(samples, cfg["retinabert"]["pretrained_name"])
    loader = DataLoader(dataset, batch_size=cfg["retinabert"]["batch_size"], shuffle=True, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RetinaBERT(cfg["retinabert"]["pretrained_name"], cfg["model"]["embed_dim"]).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["retinabert"]["lr"],
        betas=(cfg["retinabert"]["beta1"], cfg["retinabert"]["beta2"]),
        eps=cfg["retinabert"]["eps"],
        weight_decay=cfg["retinabert"]["weight_decay"],
    )
    total_steps = len(loader) * cfg["retinabert"]["epochs"]
    scheduler = get_linear_schedule_with_warmup(opt, cfg["retinabert"]["warmup_steps"], total_steps)
    for epoch in range(cfg["retinabert"]["epochs"]):
        model.train()
        total = 0.0
        for batch in tqdm(loader, desc=f"RetinaBERT epoch {epoch + 1}"):
            opt.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].flatten(0, 1).to(device)
            attention_mask = batch["attention_mask"].flatten(0, 1).to(device)
            cui = batch["cui"].flatten(0, 1).to(device)
            emb = model(input_ids, attention_mask)
            loss = umls_contrastive_loss(emb, cui, cfg["retinabert"]["tau"])
            loss.backward()
            opt.step()
            scheduler.step()
            total += float(loss.item())
        logger.info("epoch=%d loss=%.6f", epoch + 1, total / max(1, len(loader)))
    save_checkpoint(cfg["paths"]["retinabert_ckpt"], {"model": model.state_dict(), "config": cfg, "pretrained_name": cfg["retinabert"]["pretrained_name"]})
    logger.info("saved %s", cfg["paths"]["retinabert_ckpt"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)
