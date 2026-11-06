"""Shared training helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import RetinaDualDataset, build_weighted_sampler, collate_with_tpg
from models.retinabert import RetinaBERT
from models.semantic_chain_graph import build_default_scg, load_sgc_nodes
from utils.metrics import compute_metrics


def device_and_wrap(model: torch.nn.Module, multi_gpu: bool = True) -> Tuple[torch.device, torch.nn.Module]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if multi_gpu and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    return device, model


def build_loader(cfg: Dict, split: str, fold: int, train: bool, require_masks: bool = False) -> DataLoader:
    ds = RetinaDualDataset(
        csv_path=cfg["paths"]["csv"],
        split_csv=cfg["paths"]["split_csv"],
        fold=fold,
        split=split,
        classes=cfg["data"]["classes"],
        image_size=cfg["data"]["image_size"],
        train=train,
        require_masks=require_masks,
        mask_csv=cfg["paths"].get("mask_csv"),
        rho_cross=cfg["data"].get("rho_cross", 0.1),
        kappa=cfg["data"].get("kappa", 8),
    )
    sampler = build_weighted_sampler(ds) if train else None
    return DataLoader(
        ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=train and sampler is None,
        sampler=sampler,
        num_workers=cfg["data"].get("num_workers", 4),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_with_tpg,
    )


@torch.no_grad()
def evaluate_classifier(model: torch.nn.Module, loader: DataLoader, device: torch.device, classes) -> Dict[str, object]:
    model.eval()
    y_true = []
    y_prob = []
    is_teacher = getattr(model, "is_teacher", False) or getattr(getattr(model, "module", None), "is_teacher", False)
    for batch in tqdm(loader, desc="evaluate", leave=False):
        if is_teacher:
            batch = move_batch_to_device(batch, device)
            logits = model(batch)["logits"]
        else:
            logits = model(batch["cfp"].to(device))["logits"]
        y_true.extend(batch["label"].cpu().numpy().tolist())
        y_prob.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return compute_metrics(y_true, np.concatenate(y_prob, axis=0), list(classes))


def move_batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    """Recursively move tensor-valued batch fields without touching metadata."""
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        elif isinstance(value, dict):
            moved[key] = move_batch_to_device(value, device)
        else:
            moved[key] = value
    return moved


def load_semantic_initialization(retinabert_ckpt: str, cfg: Dict, device: torch.device):
    """Encode SCG nodes with RetinaBERT for strict reproduction."""
    scg_path = cfg.get("paths", {}).get("scg_nodes")
    target_k = cfg["model"].get("semantic_nodes", 66)
    nodes, a0, _ = build_default_scg(scg_path, target_k=target_k)
    _, node_texts = load_sgc_nodes(scg_path, target_k=target_k)

    from transformers import AutoTokenizer

    if not Path(retinabert_ckpt).exists():
        raise FileNotFoundError(
            f"RetinaBERT checkpoint not found: {retinabert_ckpt}. "
            "Run train/pretrain_retinabert.py first or provide a trained RetinaBERT checkpoint."
        )
    ckpt = torch.load(retinabert_ckpt, map_location=device)
    tokenizer_name = ckpt.get("pretrained_name", cfg["retinabert"]["pretrained_name"])
    local_tokenizer = Path("RetinaBERT/Bio_ClinicalBERT_tokenizer")
    tokenizer_source = str(local_tokenizer) if local_tokenizer.exists() else tokenizer_name
    model = RetinaBERT(tokenizer_name, embed_dim=cfg["model"]["embed_dim"]).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    with torch.no_grad():
        e0 = model.encode_texts(node_texts, tokenizer, device)
    return e0.detach().cpu(), a0
