"""Dynamic Semantic Chain Graph Encoder (S-DynEnc)."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


class RelationAwareSelfAttention(nn.Module):
    """Multi-head self-attention whose probabilities are weighted by SCG edges."""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("The semantic embedding dimension must be divisible by the number of heads.")
        self.heads = heads
        self.head_dim = dim // heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        n, dim = x.shape
        q = self.q(x).view(n, self.heads, self.head_dim).transpose(0, 1)
        k = self.k(x).view(n, self.heads, self.head_dim).transpose(0, 1)
        v = self.v(x).view(n, self.heads, self.head_dim).transpose(0, 1)
        logits = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        connected = edge_weight > 0
        logits = logits + edge_weight.clamp_min(1e-12).log().unsqueeze(0)
        logits = logits.masked_fill(~connected.unsqueeze(0), torch.finfo(logits.dtype).min)
        attn = self.dropout(torch.softmax(logits, dim=-1))
        out = (attn @ v).transpose(0, 1).reshape(n, dim)
        return self.out(out)


class SemanticDynamicLayer(nn.Module):
    """One adjacency-constrained dynamic refinement and RSA layer."""

    def __init__(self, dim: int = 768, heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.affinity_q = nn.Linear(dim, dim)
        self.affinity_k = nn.Linear(dim, dim)
        self.rsa = RelationAwareSelfAttention(dim, heads, dropout)
        self.ln1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim)
        )
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, e: torch.Tensor, a0: torch.Tensor, alpha: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.affinity_q(e)
        k = self.affinity_k(e)
        s_sem = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
        a_hat = F.relu(a0 + alpha * (a0 * s_sem))
        message_adjacency = a_hat.clone()
        message_adjacency.diagonal().fill_(1.0)
        h = self.ln1(e + self.rsa(e, message_adjacency))
        return self.ln2(h + self.ffn(h)), a_hat


class SemanticDynamicEncoder(nn.Module):
    """The two-layer S-DynEnc described in the updated methodology."""

    def __init__(self, dim: int = 768, heads: int = 8, layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        if layers != 2:
            raise ValueError("The updated methodology specifies exactly two S-DynEnc layers.")
        self.alpha = nn.Parameter(torch.zeros(()))
        self.layers = nn.ModuleList([SemanticDynamicLayer(dim, heads, dropout) for _ in range(layers)])

    def forward(self, e0: torch.Tensor, a0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = e0
        a_hat = a0
        for layer in self.layers:
            h, a_hat = layer(h, a0, self.alpha)
        return h, a_hat

