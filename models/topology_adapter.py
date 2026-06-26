"""Topology Adapter for CFP topology response prediction."""

from torch import nn
import torch


class TopologyAdapter(nn.Module):
    """Multi-head self-attention plus FFN over semantic-aligned CFP tokens."""

    def __init__(self, dim: int = 768, heads: int = 8, n_topology: int = 5, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))
        self.ln2 = nn.LayerNorm(dim)
        self.pred = nn.Linear(dim, n_topology)

    def forward(self, x: torch.Tensor):
        attn_out, _ = self.attn(x, x, x)
        z = self.ln1(attn_out + x)
        z = self.ln2(self.ffn(z) + z)
        return z, torch.softmax(self.pred(z), dim=-1)

