"""Dynamic Topological Parameter Graph Encoder (T-DynEnc)."""

from __future__ import annotations

import math
from typing import Dict, List, Union

import torch
import torch.nn.functional as F
from torch import nn


class TypeBranch(nn.Module):
    """Category-specific FC--LN--ReLU projection of four measurements."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(4, dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.ln(self.fc(torch.log1p(x.clamp_min(0.0)))))


class TopologicalDynamicLayer(nn.Module):
    def __init__(self, dim: int = 768) -> None:
        super().__init__()
        self.affinity_q = nn.Linear(dim, dim)
        self.affinity_k = nn.Linear(dim, dim)
        self.message = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)

    def forward(
        self,
        z: torch.Tensor,
        b0: torch.Tensor,
        candidate_mask: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        q = self.affinity_q(z)
        k = self.affinity_k(z)
        signed = q @ k.t() / math.sqrt(q.shape[-1])
        signed = 0.5 * (signed + signed.t())
        b = F.relu(b0 + beta * (candidate_mask.to(z.dtype) * signed))
        b = 0.5 * (b + b.t())
        b = b.clone()
        b.fill_diagonal_(1.0)
        weights = b / b.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        aggregated = weights @ z
        return self.ln(z + F.relu(self.message(aggregated)))


class TopologicalDynamicEncoder(nn.Module):
    """Encode a variable-size TPG and pool RV/MA/NPA/NV/leakage bases."""

    def __init__(self, dim: int = 768, layers: int = 2, n_types: int = 5) -> None:
        super().__init__()
        if layers != 2:
            raise ValueError("The updated methodology specifies exactly two T-DynEnc layers.")
        if n_types != 5:
            raise ValueError("TPG must use RV, MA, NPA, NV, and leakage.")
        self.dim = dim
        self.n_types = n_types
        self.beta = nn.Parameter(torch.zeros(()))
        self.branches = nn.ModuleList([TypeBranch(dim) for _ in range(n_types)])
        self.layers = nn.ModuleList([TopologicalDynamicLayer(dim) for _ in range(layers)])

    def encode_one(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        candidate_mask: torch.Tensor,
        node_types: torch.Tensor,
    ) -> torch.Tensor:
        if x.shape[0] == 0:
            return x.new_zeros((self.n_types, self.dim))
        z = x.new_zeros((x.shape[0], self.dim))
        for type_index, branch in enumerate(self.branches):
            selected = node_types == type_index
            if selected.any():
                z[selected] = branch(x[selected]).to(z.dtype)
        for layer in self.layers:
            z = layer(z, adj, candidate_mask, self.beta)
        pooled = [
            z[node_types == t].mean(dim=0) if (node_types == t).any() else z.new_zeros(self.dim)
            for t in range(self.n_types)
        ]
        return torch.stack(pooled)

    def forward(
        self,
        tpg_list: Union[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]],
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(tpg_list, dict):
            iterable = []
            for i in range(tpg_list["x"].shape[0]):
                selected = tpg_list["node_mask"][i].to(device)
                adj = tpg_list["adj"][i].to(device)[selected][:, selected]
                candidate = tpg_list["candidate_mask"][i].to(device)[selected][:, selected]
                iterable.append(
                    {
                        "x": tpg_list["x"][i].to(device)[selected],
                        "adj": adj,
                        "candidate_mask": candidate,
                        "node_types": tpg_list["node_types"][i].to(device)[selected],
                    }
                )
        else:
            iterable = tpg_list
        bases = []
        for graph in iterable:
            bases.append(
                self.encode_one(
                    graph["x"].to(device),
                    graph["adj"].to(device),
                    graph["candidate_mask"].to(device),
                    graph["node_types"].to(device),
                )
            )
        if not bases:
            return torch.empty((0, self.n_types, self.dim), device=device)
        return torch.stack(bases)
