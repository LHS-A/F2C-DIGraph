"""Cross-modal invariance distillation losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def graph_normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
    a = adj + torch.eye(adj.shape[0], device=adj.device, dtype=adj.dtype)
    deg = a.sum(dim=-1).clamp_min(1e-6)
    d_inv = deg.pow(-0.5)
    return d_inv[:, None] * a * d_inv[None, :]


def semantic_distribution(
    visual_feature: torch.Tensor,
    projection: nn.Module,
    semantic_basis: torch.Tensor,
    dynamic_adjacency: torch.Tensor,
) -> torch.Tensor:
    """Cosine node relevance followed by normalized-SCG propagation."""
    projected = projection(visual_feature)
    scores = F.cosine_similarity(projected[:, None, :], semantic_basis[None, :, :], dim=-1)
    propagated = scores @ graph_normalize_adjacency(dynamic_adjacency).t()
    return torch.softmax(propagated, dim=-1)


def semantic_distillation_loss(teacher_probability: torch.Tensor, student_probability: torch.Tensor) -> torch.Tensor:
    return F.kl_div(student_probability.clamp_min(1e-8).log(), teacher_probability.detach(), reduction="batchmean")


def _bernoulli_kl(target: torch.Tensor, prediction: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = target.detach().clamp(eps, 1.0 - eps)
    q = prediction.clamp(eps, 1.0 - eps)
    return p * (p.log() - q.log()) + (1.0 - p) * ((1.0 - p).log() - (1.0 - q).log())


def topology_response_loss(
    teacher_response: torch.Tensor,
    student_response: torch.Tensor,
    foreground_mask: torch.Tensor,
    structure_presence: torch.Tensor,
    valid_tokens: torch.Tensor,
) -> torch.Tensor:
    """Five-category Bernoulli KL on matched, valid CFP--FFA tokens."""
    losses = []
    for batch_index in range(teacher_response.shape[0]):
        valid = valid_tokens[batch_index].bool()
        if not valid.any():
            continue
        category_losses = []
        for category in range(teacher_response.shape[-1]):
            present = bool(structure_presence[batch_index, category].item())
            indices = torch.nonzero(valid, as_tuple=False).flatten()
            if present:
                foreground = foreground_mask[batch_index, :, category].bool() & valid
                background = (~foreground_mask[batch_index, :, category].bool()) & valid
                foreground_indices = torch.nonzero(foreground, as_tuple=False).flatten()
                background_indices = torch.nonzero(background, as_tuple=False).flatten()
                if foreground_indices.numel() and background_indices.numel():
                    count = min(foreground_indices.numel(), background_indices.numel())
                    foreground_indices = foreground_indices[torch.randperm(foreground_indices.numel(), device=indices.device)[:count]]
                    background_indices = background_indices[torch.randperm(background_indices.numel(), device=indices.device)[:count]]
                    indices = torch.cat((foreground_indices, background_indices))
            if indices.numel():
                category_losses.append(
                    _bernoulli_kl(
                        teacher_response[batch_index, indices, category],
                        student_response[batch_index, indices, category],
                    ).mean()
                )
        if category_losses:
            losses.append(torch.stack(category_losses).mean())
    if not losses:
        return student_response.sum() * 0.0
    return torch.stack(losses).mean()
