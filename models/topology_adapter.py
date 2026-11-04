"""CFP topology adapter retained at inference."""

from torch import nn
import torch


class TopologyAdapter(nn.Module):
    """Residual semantic cross-attention followed by token self-attention."""

    def __init__(self, dim: int = 768, heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.cross_ln = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.self_ln = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim)
        )
        self.ffn_ln = nn.LayerNorm(dim)

    def forward(self, visual_tokens: torch.Tensor, semantic_basis: torch.Tensor) -> torch.Tensor:
        if semantic_basis.ndim == 2:
            semantic_basis = semantic_basis.unsqueeze(0).expand(visual_tokens.shape[0], -1, -1)
        aligned, _ = self.cross_attention(visual_tokens, semantic_basis, semantic_basis, need_weights=False)
        z = self.cross_ln(visual_tokens + aligned)
        contextual, _ = self.self_attention(z, z, z, need_weights=False)
        z = self.self_ln(z + contextual)
        return self.ffn_ln(z + self.ffn(z))

