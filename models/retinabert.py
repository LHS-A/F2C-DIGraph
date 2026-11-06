"""RetinaBERT: relation-aware retinal language encoder pretraining."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import random

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer


REQUIRED_UMLS_COLUMNS = {
    "cui", "concept", "definition", "synonyms", "semantic_type", "relation", "tail_concept"
}


@dataclass
class KnowledgeSample:
    text: str
    cui: str
    form: str


def load_umls_knowledge(path: str) -> List[KnowledgeSample]:
    """Load prepared UMLS concept-definition and relation-triplet samples."""
    if not Path(path).exists():
        raise FileNotFoundError(
            "Strict RetinaBERT pretraining requires a prepared UMLS concept file. "
            f"Expected {path}. Required columns: cui, concept, definition, synonyms, "
            "semantic_type, relation, and tail_concept."
        )
    df = pd.read_csv(path)
    missing = REQUIRED_UMLS_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"UMLS file is missing required columns: {sorted(missing)}")
    samples: List[KnowledgeSample] = []
    seen_definitions = set()
    seen_relations = set()
    for _, row in df.iterrows():
        cui = str(row["cui"]).strip()
        concept = str(row["concept"]).strip()
        definition = str(row["definition"]).strip()
        definition_key = (cui, concept, definition)
        if cui and concept and definition and definition.lower() != "nan" and definition_key not in seen_definitions:
            samples.append(KnowledgeSample(f"{concept}. Definition: {definition}", cui, "definition"))
            seen_definitions.add(definition_key)
        relation = str(row["relation"]).strip()
        tail = str(row["tail_concept"]).strip()
        relation_key = (cui, concept, relation, tail)
        if relation and tail and relation.lower() != "nan" and tail.lower() != "nan" and relation_key not in seen_relations:
            samples.append(KnowledgeSample(f"{concept} {relation} {tail}", cui, "relation"))
            seen_relations.add(relation_key)
    counts: Dict[str, int] = {}
    for sample in samples:
        counts[sample.cui] = counts.get(sample.cui, 0) + 1
    eligible = {cui for cui, count in counts.items() if count >= 2}
    if not eligible:
        raise ValueError("RetinaBERT pretraining requires at least two descriptions for each sampled CUI.")
    return [sample for sample in samples if sample.cui in eligible]


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
        pair = random.sample(self.by_cui[cui], 2)
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
        weights = attention_mask.unsqueeze(-1).to(out.last_hidden_state.dtype)
        mean_pooled = (out.last_hidden_state * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return F.normalize(self.proj(mean_pooled), dim=-1)

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
