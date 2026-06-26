"""Cross-modal invariance distillation losses."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


def graph_normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
    a = adj + torch.eye(adj.shape[0], device=adj.device, dtype=adj.dtype)
    deg = a.sum(dim=-1).clamp_min(1e-6)
    d_inv = deg.pow(-0.5)
    return d_inv[:, None] * a * d_inv[None, :]


class SemanticDistillationLoss(nn.Module):
    """Graph-aware semantic KL divergence from FFA teacher to CFP student."""

    def __init__(self, dim: int = 768) -> None:
        super().__init__()
        self.teacher_proj = nn.Linear(dim, dim)
        self.student_proj = nn.Linear(dim, dim)

    def relevance(self, v: torch.Tensor, h_sem: torch.Tensor, adj: torch.Tensor, mode: str) -> torch.Tensor:
        proj = self.teacher_proj(v) if mode == "teacher" else self.student_proj(v)
        scores = F.cosine_similarity(proj[:, None, :], h_sem[None, :, :], dim=-1)
        abar = graph_normalize_adjacency(adj)
        propagated = scores @ abar.t()
        return torch.softmax(propagated, dim=-1)

    def forward(self, v_teacher: torch.Tensor, v_student: torch.Tensor, h_sem: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        p_t = self.relevance(v_teacher, h_sem, adj, "teacher").detach()
        p_s = self.relevance(v_student, h_sem, adj, "student")
        return F.kl_div((p_s + 1e-8).log(), p_t, reduction="batchmean")


def topology_response_loss(a_teacher: torch.Tensor, a_student: torch.Tensor) -> torch.Tensor:
    """Row-wise topology KL divergence over five topology bases."""
    p_t = a_teacher.detach().reshape(-1, a_teacher.shape[-1])
    p_s = a_student.reshape(-1, a_student.shape[-1])
    return F.kl_div((p_s + 1e-8).log(), p_t, reduction="batchmean")
