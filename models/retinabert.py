"""RetinaBERT: relation-aware retinal language encoder pretraining."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import random

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer


REQUIRED_UMLS_COLUMNS = {"cui", "concept", "definition", "semantic_type"}


@dataclass
class KnowledgeSample:
    text: str
    cui: str


def load_umls_knowledge(path: str, semantic_types: Iterable[str]) -> List[KnowledgeSample]:
    """Load prepared UMLS concept-definition and relation-triplet samples."""
    if not Path(path).exists():
        raise FileNotFoundError(
            "Strict RetinaBERT pretraining requires a prepared UMLS concept file. "
            f"Expected {path}. Required columns: cui, concept, definition, semantic_type, "
            "and optional relation, tail_concept."
        )
    df = pd.read_csv(path)
    missing = REQUIRED_UMLS_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"UMLS file is missing required columns: {sorted(missing)}")
    df = df[df["semantic_type"].astype(str).isin(set(semantic_types))]
    samples: List[KnowledgeSample] = []
    for _, row in df.iterrows():
        cui = str(row["cui"]).strip()
        concept = str(row["concept"]).strip()
        definition = str(row["definition"]).strip()
        if cui and concept and definition:
            samples.append(KnowledgeSample(f"{concept}. Definition: {definition}", cui))
        if {"relation", "tail_concept"}.issubset(df.columns):
            rel = str(row.get("relation", "")).strip()
            tail = str(row.get("tail_concept", "")).strip()
            if rel and tail and rel.lower() != "nan" and tail.lower() != "nan":
                samples.append(KnowledgeSample(f"{concept} {rel} {tail}", cui))
    if not samples:
        raise ValueError("No usable UMLS samples after semantic-type filtering.")
    return samples


class UMLSContrastiveDataset(Dataset):
    """Dataset of UMLS textual samples grouped by CUI."""

    def __init__(self, samples: List[KnowledgeSample], tokenizer_name: str, max_length: int = 96) -> None:
        self.samples = samples
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.cuis = sorted({s.cui for s in samples})
        self.cui_to_idx = {cui: i for i, cui in enumerate(self.cuis)}
        self.by_cui: Dict[str, List[str]] = {cui: [] for cui in self.cuis}
        for sample in samples:
            self.by_cui[sample.cui].append(sample.text)
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.cuis)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cui = self.cuis[idx]
        texts = self.by_cui[cui]
        pair = random.sample(texts, 2) if len(texts) >= 2 else [texts[0], texts[0]]
        enc = self.tokenizer(
            pair,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "cui": torch.tensor([self.cui_to_idx[cui], self.cui_to_idx[cui]], dtype=torch.long),
        }


class RetinaBERT(nn.Module):
    """BioClinicalBERT initialized encoder with projection for contrastive learning."""

    def __init__(self, pretrained_name: str = "emilyalsentzer/Bio_ClinicalBERT", embed_dim: int = 768) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_name)
        hidden = self.bert.config.hidden_size
        self.proj = nn.Linear(hidden, embed_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return F.normalize(self.proj(cls), dim=-1)

    @torch.no_grad()
    def encode_texts(self, texts: List[str], tokenizer, device: torch.device, max_length: int = 96) -> torch.Tensor:
        self.eval()
        enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        return self(enc["input_ids"], enc["attention_mask"])


def umls_contrastive_loss(emb: torch.Tensor, cui: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """Supervised contrastive loss over samples sharing the same CUI."""
    sim = emb @ emb.t() / tau
    logits_mask = ~torch.eye(len(emb), dtype=torch.bool, device=emb.device)
    same = (cui[:, None] == cui[None, :]) & logits_mask
    log_prob = sim - torch.logsumexp(sim.masked_fill(~logits_mask, -1e9), dim=1, keepdim=True)
    denom = same.sum(dim=1).clamp_min(1)
    loss = -(log_prob * same.float()).sum(dim=1) / denom
    valid = same.any(dim=1)
    if not valid.any():
        raise ValueError("RetinaBERT batch has no positive CUI pairs. Increase batch size or use grouped sampling.")
    return loss[valid].mean()
