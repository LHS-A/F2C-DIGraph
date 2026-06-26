"""Full F2C-DIGraph teacher, student, and distillation wrapper."""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .backbones import ImageClassifier
from .cid_losses import SemanticDistillationLoss, topology_response_loss
from .semantic_dynamic_encoder import SemanticDynamicEncoder
from .topological_dynamic_encoder import TopologicalDynamicEncoder
from .topology_adapter import TopologyAdapter


class FfaTeacher(nn.Module):
    """FFA teacher receives early/mid/late FFA frames stacked as 9 channels."""

    def __init__(self, num_classes: int, embed_dim: int = 768, pretrained: bool = True, backbone: str = "resnet50") -> None:
        super().__init__()
        self.is_teacher = True
        self.model = ImageClassifier(in_channels=9, num_classes=num_classes, embed_dim=embed_dim, pretrained=pretrained, backbone=backbone)

    def forward(self, ffa: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(ffa)


class CfpStudent(nn.Module):
    """CFP student with semantic alignment, topology adapter, and classifier."""

    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 768,
        pretrained: bool = True,
        heads: int = 8,
        backbone: str = "resnet50",
    ) -> None:
        super().__init__()
        self.model = ImageClassifier(in_channels=3, num_classes=num_classes, embed_dim=embed_dim, pretrained=pretrained, backbone=backbone)
        self.sem_q = nn.Linear(embed_dim, embed_dim)
        self.sem_k = nn.Linear(embed_dim, embed_dim)
        self.sem_v = nn.Linear(embed_dim, embed_dim)
        self.ta = TopologyAdapter(embed_dim, heads=heads, n_topology=5)
        self.fused_classifier = nn.Linear(embed_dim, num_classes)

    def semantic_align(self, tokens: torch.Tensor, h_sem: torch.Tensor) -> torch.Tensor:
        q = self.sem_q(tokens)
        k = self.sem_k(h_sem)
        v = self.sem_v(h_sem)
        attn = torch.softmax(q @ k.t() / math.sqrt(q.shape[-1]), dim=-1)
        return tokens + attn @ v

    def forward(self, cfp: torch.Tensor, h_sem: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        out = self.model(cfp)
        if h_sem is not None:
            sem_tokens = self.semantic_align(out["tokens"], h_sem)
            topo_tokens, topo_resp = self.ta(sem_tokens)
            pooled = topo_tokens.mean(dim=1)
            out["logits"] = self.fused_classifier(pooled)
            out["semantic_tokens"] = sem_tokens
            out["topology_tokens"] = topo_tokens
            out["topology_response"] = topo_resp
        return out


class F2CDIGraph(nn.Module):
    """Training-time wrapper for semantic and topological cross-modal distillation."""

    def __init__(
        self,
        teacher: FfaTeacher,
        student: CfpStudent,
        semantic_encoder: SemanticDynamicEncoder,
        topology_encoder: TopologicalDynamicEncoder,
        semantic_e0: torch.Tensor,
        semantic_a0: torch.Tensor,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.semantic_encoder = semantic_encoder
        self.topology_encoder = topology_encoder
        self.register_buffer("semantic_e0", semantic_e0.float())
        self.register_buffer("semantic_a0", semantic_a0.float())
        self.sem_loss = SemanticDistillationLoss(embed_dim)
        self.topo_q = nn.Linear(embed_dim, embed_dim)
        self.topo_k = nn.Linear(embed_dim, embed_dim)

    def teacher_topology_response(self, teacher_tokens: torch.Tensor, h_topo: torch.Tensor) -> torch.Tensor:
        q = self.topo_q(teacher_tokens)
        k = self.topo_k(h_topo)
        logits = torch.einsum("bmd,bnd->bmn", q, k) / math.sqrt(q.shape[-1])
        return torch.softmax(logits, dim=-1)

    def forward(self, batch: Dict[str, object]) -> Dict[str, torch.Tensor]:
        h_sem, a_sem = self.semantic_encoder(self.semantic_e0, self.semantic_a0)
        self.teacher.eval()
        with torch.no_grad():
            teacher_out = self.teacher(batch["ffa"])
        student_out = self.student(batch["cfp"], h_sem)
        out = {
            "logits": student_out["logits"],
            "teacher_logits": teacher_out["logits"],
            "h_sem": h_sem,
            "a_sem": a_sem,
            "student_global": student_out["global"],
            "teacher_global": teacher_out["global"],
            "student_topology_response": student_out["topology_response"],
        }
        if "tpg" in batch:
            h_topo = self.topology_encoder(batch["tpg"], device=batch["cfp"].device)
            out["h_topo"] = h_topo
            out["teacher_topology_response"] = self.teacher_topology_response(teacher_out["tokens"], h_topo)
        return out

    def losses(self, outputs: Dict[str, torch.Tensor], labels: torch.Tensor, lambda_sem: float, lambda_topo: float) -> Dict[str, torch.Tensor]:
        cls = F.cross_entropy(outputs["logits"], labels)
        sem = self.sem_loss(outputs["teacher_global"], outputs["student_global"], outputs["h_sem"], outputs["a_sem"])
        if "teacher_topology_response" not in outputs:
            raise RuntimeError("Topology distillation requires TPG masks in strict reproduction mode.")
        topo = topology_response_loss(outputs["teacher_topology_response"], outputs["student_topology_response"])
        total = cls + lambda_sem * sem + lambda_topo * topo
        return {"loss": total, "loss_cls": cls, "loss_sem": sem, "loss_topo": topo}
