"""Dynamic Semantic Chain Graph Encoder (S-DynEnc)."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


class SemanticDynamicLayer(nn.Module):
    """One S-DynEnc layer with adjacency-constrained self-attention."""

    def __init__(self, dim: int = 768, heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.alpha = nn.Parameter(torch.zeros(()))
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, e: torch.Tensor, a0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.wq(e)
        k = self.wk(e)
        s_dyn = torch.softmax((q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1]), dim=-1)
        a_hat = F.relu(a0 + self.alpha * (a0 * s_dyn))
        mask = a_hat <= 0
        attn_out, _ = self.attn(e.unsqueeze(0), e.unsqueeze(0), e.unsqueeze(0), attn_mask=mask)
        e_rsa = self.ln1(attn_out.squeeze(0) + e)
        out = self.ln2(self.ffn(e_rsa) + e_rsa)
        return out, a_hat


class SemanticDynamicEncoder(nn.Module):
    """Stack three dynamic semantic graph layers by default."""

    def __init__(self, dim: int = 768, heads: int = 8, layers: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.ModuleList([SemanticDynamicLayer(dim, heads, dropout) for _ in range(layers)])

    def forward(self, e0: torch.Tensor, a0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = e0
        a_hat = a0
        for layer in self.layers:
            h, a_hat = layer(h, a0)
        return h, a_hat

