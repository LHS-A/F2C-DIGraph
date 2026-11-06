"""F2C-DIGraph teacher, CFP student, and cross-modal distillation wrapper."""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .backbones import ResNet50Encoder
from .cid_losses import semantic_distribution, semantic_distillation_loss, topology_response_loss
from .semantic_dynamic_encoder import SemanticDynamicEncoder
from .topological_dynamic_encoder import TopologicalDynamicEncoder
from .topology_adapter import TopologyAdapter


def _grid_masks(masks: torch.Tensor, grid_size: tuple[int, int]) -> torch.Tensor:
    """Occupancy-preserving mask projection to the common visual-token grid."""
    pooled = F.adaptive_max_pool2d(masks.float(), grid_size)
    return pooled.flatten(2).transpose(1, 2)


def _grid_valid_tokens(valid_overlap: torch.Tensor, grid_size: tuple[int, int]) -> torch.Tensor:
    """Select tokens whose complete receptive grid cell lies in the valid overlap."""
    coverage = F.adaptive_avg_pool2d(valid_overlap.float(), grid_size)
    return coverage.flatten(2).squeeze(1).ge(1.0 - 1e-6)


class FfaTeacher(nn.Module):
    """Three-channel grayscale FFA teacher with SIM and TIM supervision."""

    def __init__(
        self,
        num_classes: int,
        semantic_e0: torch.Tensor,
        semantic_a0: torch.Tensor,
        embed_dim: int = 768,
        pretrained: bool = True,
        heads: int = 8,
        semantic_layers: int = 2,
        topology_layers: int = 2,
        dropout: float = 0.1,
        backbone: str = "resnet50",
    ) -> None:
        super().__init__()
        if backbone != "resnet50":
            raise ValueError("Strict reproduction uses an ImageNet-pretrained ResNet-50.")
        self.is_teacher = True
        self.encoder = ResNet50Encoder(3, embed_dim, pretrained, project_global=True)
        self.semantic_encoder = SemanticDynamicEncoder(embed_dim, heads, semantic_layers, dropout)
        self.topology_encoder = TopologicalDynamicEncoder(embed_dim, topology_layers, 5)
        self.register_buffer("semantic_e0", semantic_e0.float())
        self.register_buffer("semantic_a0", semantic_a0.float())
        self.semantic_projection = nn.Linear(embed_dim, embed_dim)
        self.topology_q = nn.Linear(embed_dim, embed_dim)
        self.topology_k = nn.Linear(embed_dim, embed_dim)
        # FFA feature + K semantic responses + 5 pooled topology responses + 5 presence indicators.
        self.classifier = nn.Linear(embed_dim + semantic_e0.shape[0] + 5 + 5, num_classes)

    def semantic_basis(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.semantic_encoder(self.semantic_e0, self.semantic_a0)

    def topology_response(
        self,
        tokens: torch.Tensor,
        topology_basis: torch.Tensor,
        structure_masks: torch.Tensor,
        structure_presence: torch.Tensor,
        grid_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = torch.einsum(
            "bmd,bcd->bmc", self.topology_q(tokens), self.topology_k(topology_basis)
        ) / math.sqrt(tokens.shape[-1])
        preliminary = torch.sigmoid(logits)
        foreground = _grid_masks(structure_masks, grid_size)
        present = structure_presence.bool()[:, None, :]
        constrained = torch.where(present, foreground * (1.0 + preliminary) / 2.0, torch.zeros_like(preliminary))
        return constrained, foreground

    def forward(self, batch: Dict[str, object]) -> Dict[str, torch.Tensor]:
        visual = self.encoder(batch["ffa"])
        h_sem, a_sem = self.semantic_basis()
        semantic_probability = semantic_distribution(
            visual["global"], self.semantic_projection, h_sem, a_sem
        )
        h_topo = self.topology_encoder(batch["tpg"], visual["global"].device)
        grid_size = visual["feature_map"].shape[-2:]
        topology_response, foreground = self.topology_response(
            visual["tokens"], h_topo, batch["structure_masks"], batch["structure_presence"], grid_size
        )
        valid_tokens = _grid_valid_tokens(batch["valid_overlap"], grid_size)
        pooled_topology = (
            topology_response * valid_tokens[:, :, None]
        ).sum(dim=1) / valid_tokens.sum(dim=1, keepdim=True).clamp_min(1)
        classifier_input = torch.cat(
            (visual["global"], semantic_probability, pooled_topology, batch["structure_presence"].float()),
            dim=-1,
        )
        return {
            **visual,
            "logits": self.classifier(classifier_input),
            "h_sem": h_sem,
            "a_sem": a_sem,
            "semantic_probability": semantic_probability,
            "h_topo": h_topo,
            "topology_response": topology_response,
            "foreground_tokens": foreground,
            "valid_tokens": valid_tokens,
        }


class CfpInferenceModel(nn.Module):
    """Exactly the modules retained for CFP-only inference."""

    def __init__(
        self,
        num_classes: int,
        semantic_basis: torch.Tensor,
        embed_dim: int = 768,
        pretrained: bool = True,
        heads: int = 8,
        dropout: float = 0.1,
        backbone: str = "resnet50",
    ) -> None:
        super().__init__()
        if backbone != "resnet50":
            raise ValueError("Strict reproduction uses an ImageNet-pretrained ResNet-50.")
        # The inference encoder produces spatial tokens only. The global
        # projection used by semantic distillation belongs to CfpStudent and
        # is therefore absent from exported CFP-only checkpoints.
        self.encoder = ResNet50Encoder(3, embed_dim, pretrained, project_global=False)
        self.topology_adapter = TopologyAdapter(embed_dim, heads, dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)
        # It is fixed during training but retained and used at inference, so it
        # is represented as a non-trainable parameter and included in Params.
        self.semantic_basis = nn.Parameter(semantic_basis.detach().float().clone(), requires_grad=False)

    def forward(self, cfp: torch.Tensor) -> Dict[str, torch.Tensor]:
        visual = self.encoder(cfp)
        topology_tokens = self.topology_adapter(visual["tokens"], self.semantic_basis)
        return {
            **visual,
            "topology_tokens": topology_tokens,
            "logits": self.classifier(topology_tokens.mean(dim=1)),
        }


class CfpStudent(nn.Module):
    """Inference student plus projections used only while distilling."""

    def __init__(
        self,
        num_classes: int,
        semantic_basis: torch.Tensor,
        embed_dim: int = 768,
        pretrained: bool = True,
        heads: int = 8,
        dropout: float = 0.1,
        backbone: str = "resnet50",
    ) -> None:
        super().__init__()
        self.inference_model = CfpInferenceModel(
            num_classes, semantic_basis, embed_dim, pretrained, heads, dropout, backbone
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_projection = nn.Linear(2048, embed_dim)
        self.semantic_projection = nn.Linear(embed_dim, embed_dim)
        self.topology_projection = nn.Linear(embed_dim, 5)

    def forward(self, cfp: torch.Tensor, distill: bool = True) -> Dict[str, torch.Tensor]:
        out = self.inference_model(cfp)
        if distill:
            out["global"] = self.global_projection(self.global_pool(out["feature_map"]).flatten(1))
            out["topology_response"] = torch.sigmoid(self.topology_projection(out["topology_tokens"]))
        return out


class F2CDIGraph(nn.Module):
    """Student-training wrapper; the complete teacher is frozen."""

    def __init__(self, teacher: FfaTeacher, student: CfpStudent) -> None:
        super().__init__()
        self.teacher = teacher
        self.student = student
        for parameter in self.teacher.parameters():
            parameter.requires_grad = False
        self.teacher.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(self, batch: Dict[str, object]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            teacher = self.teacher(batch)
        student = self.student(batch["cfp"], distill=True)
        student_semantic_probability = semantic_distribution(
            student["global"],
            self.student.semantic_projection,
            self.student.inference_model.semantic_basis,
            teacher["a_sem"],
        )
        return {
            "logits": student["logits"],
            "teacher_semantic_probability": teacher["semantic_probability"],
            "student_semantic_probability": student_semantic_probability,
            "teacher_topology_response": teacher["topology_response"],
            "student_topology_response": student["topology_response"],
            "foreground_tokens": teacher["foreground_tokens"],
            "valid_tokens": teacher["valid_tokens"],
            "structure_presence": batch["structure_presence"],
        }

    def losses(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        lambda_sem: float,
        lambda_topo: float,
    ) -> Dict[str, torch.Tensor]:
        classification = F.cross_entropy(outputs["logits"], labels)
        semantic = semantic_distillation_loss(
            outputs["teacher_semantic_probability"], outputs["student_semantic_probability"]
        )
        topology = topology_response_loss(
            outputs["teacher_topology_response"],
            outputs["student_topology_response"],
            outputs["foreground_tokens"],
            outputs["structure_presence"],
            outputs["valid_tokens"],
        )
        total = classification + lambda_sem * semantic + lambda_topo * topology
        return {"loss": total, "loss_cls": classification, "loss_sem": semantic, "loss_topo": topology}
