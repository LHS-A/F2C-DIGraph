"""Dynamic Topological Parameter Graph Encoder (T-DynEnc)."""

from __future__ import annotations

import math
from typing import Dict, List, Union

import torch
import torch.nn.functional as F
from torch import nn


class TypeBranch(nn.Module):
    """Category-specific FC + BatchNorm + ReLU branch for raw 4-D node features."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(4, dim)
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.fc(x)
        if z.shape[0] > 1:
            z = self.bn(z)
        return F.relu(z)


class TopologicalDynamicLayer(nn.Module):
    def __init__(self, dim: int = 768) -> None:
        super().__init__()
        self.wp = nn.Linear(dim, dim)
        self.wg = nn.Linear(dim, dim)
        self.beta = nn.Parameter(torch.zeros(()))
        self.msg = nn.Linear(dim * 2, dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, z: torch.Tensor, b0: torch.Tensor) -> torch.Tensor:
        p = self.wp(z)
        g = self.wg(z)
        s = torch.softmax((p @ g.t()) / math.sqrt(p.shape[-1]), dim=-1)
        b = F.relu(b0 + self.beta * s)
        m = (b @ z) / b.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return self.ln(F.relu(self.msg(torch.cat([z, m], dim=-1))) + z)


class TopologicalDynamicEncoder(nn.Module):
    """Encode variable-size TPGs and pool into five topology bases."""

    def __init__(self, dim: int = 768, layers: int = 2, n_types: int = 5) -> None:
        super().__init__()
        self.n_types = n_types
        self.branches = nn.ModuleList([TypeBranch(dim) for _ in range(n_types)])
        self.layers = nn.ModuleList([TopologicalDynamicLayer(dim) for _ in range(layers)])

    def encode_one(self, x: torch.Tensor, adj: torch.Tensor, node_types: torch.Tensor) -> torch.Tensor:
        z = torch.zeros(x.shape[0], self.branches[0].fc.out_features, device=x.device, dtype=x.dtype)
        for t, branch in enumerate(self.branches):
            mask = node_types == t
            if mask.any():
                z[mask] = branch(x[mask]).to(dtype=z.dtype)
        for layer in self.layers:
            z = layer(z, adj)
        pooled = []
        for t in range(self.n_types):
            mask = node_types == t
            if mask.any():
                pooled.append(z[mask].mean(dim=0))
            else:
                pooled.append(torch.zeros(z.shape[-1], device=z.device, dtype=z.dtype))
        return torch.stack(pooled, dim=0)

    def forward(self, tpg_list: Union[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]], device: torch.device) -> torch.Tensor:
        bases = []
        if isinstance(tpg_list, dict):
            x_batch = tpg_list["x"].to(device)
            adj_batch = tpg_list["adj"].to(device)
            type_batch = tpg_list["node_types"].to(device)
            mask_batch = tpg_list["node_mask"].to(device)
            iterable = []
            for i in range(x_batch.shape[0]):
                mask = mask_batch[i]
                iterable.append({"x": x_batch[i, mask], "adj": adj_batch[i][mask][:, mask], "node_types": type_batch[i, mask]})
        else:
            iterable = tpg_list
        for graph in iterable:
            x = graph["x"].to(device)
            adj = graph["adj"].to(device)
            node_types = graph["node_types"].to(device)
            bases.append(self.encode_one(x, adj, node_types))
        return torch.stack(bases, dim=0)
