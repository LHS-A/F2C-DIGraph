"""Semantic Chain Graph construction for disease-anatomy-finding reasoning."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


DISEASES = ["AMD", "CSC", "DR", "RVO", "Normal"]
ANATOMY = [
    "Retina",
    "Macula",
    "Vasculature",
    "Veins",
    "Capillary network",
    "Foveal avascular zone",
    "Choroid",
    "Retinal pigment epithelium",
    "Optic disc",
    "Arteriovenous transit",
    "Ischemic retina",
    "Fovea",
]
FINDINGS = [
    "Microaneurysms",
    "Dot and blot hemorrhages",
    "Hard exudates",
    "Cotton wool spots",
    "Venous beading",
    "Intraretinal microvascular abnormalities",
    "Retinal neovascularization",
    "Capillary nonperfusion",
    "Macular ischemia",
    "Retinal hemorrhage",
    "Vascular tortuosity",
    "Blood vessel dilation",
    "Blocked fluorescence",
    "Subretinal fluid",
    "Serous retinal detachment",
    "Inkblot leakage",
    "Fluorescein leakage",
    "Central scotoma",
    "Retinal distortion",
    "Choroidal neovascularization",
    "Pigment epithelial detachment",
    "Window defect",
    "Choroidal filling",
    "Geographic atrophy",
    "Drusen",
    "Leakage point",
    "Foveal leakage",
    "Retinal edema",
    "Avascular zone enlargement",
    "Ischemic area",
    "Neovascular leakage",
    "Vessel wall staining",
    "Delayed filling",
    "Normal retinal appearance",
]
OTHER_NODES = [f"Others {i + 1}" for i in range(14)]
SCG_NODES = ["[CLS]"] + DISEASES + ANATOMY + FINDINGS + OTHER_NODES


def load_sgc_nodes(path: Optional[str], target_k: int = 66) -> Tuple[List[str], List[str]]:
    """Load SCG node names and definitions from ``name: definition`` text."""
    if not path or not Path(path).exists():
        nodes = SCG_NODES[:target_k]
        texts = nodes
        return nodes, texts
    nodes = ["[CLS]"]
    texts = ["global retinal diagnostic root node"]
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        name, definition = line.split(":", 1)
        name = name.strip()
        definition = definition.strip()
        if name:
            nodes.append(name)
            texts.append(f"{name}. Definition: {definition}" if definition else name)
    while len(nodes) < target_k:
        i = len(nodes)
        nodes.append(f"Others {i}")
        texts.append(f"Others {i}. Definition: disease-specific residual retinal finding node")
    return nodes[:target_k], texts[:target_k]


def build_default_scg(nodes_path: Optional[str] = None, target_k: int = 66) -> Tuple[List[str], torch.Tensor, Dict[str, int]]:
    """Build the 66-node SCG adjacency from the paper's hierarchy rules."""
    nodes, _ = load_sgc_nodes(nodes_path, target_k)
    if len(nodes) != target_k:
        raise AssertionError(f"SCG must contain K={target_k} nodes, got {len(nodes)}.")
    idx = {name: i for i, name in enumerate(nodes)}
    adj = torch.zeros(len(nodes), len(nodes), dtype=torch.float32)
    cls = idx["[CLS]"]
    adj[cls, :] = 1.0
    adj[:, cls] = 1.0
    disease_to_anatomy = {
        "AMD": ["Macula", "Choroid", "Retinal pigment epithelium", "Fovea"],
        "CSC": ["Macula", "Choroid", "Retinal pigment epithelium", "Retina"],
        "DR": ["Retina", "Vasculature", "Capillary network", "Foveal avascular zone", "Ischemic retina"],
        "RVO": ["Veins", "Vasculature", "Arteriovenous transit", "Ischemic retina", "Retina"],
        "Normal": ["Retina", "Macula", "Optic disc"],
    }
    anatomy_to_findings = {
        "Retina": ["Retinal hemorrhage", "Retinal edema", "Normal retinal appearance"],
        "Macula": ["Central scotoma", "Subretinal fluid", "Serous retinal detachment", "Macular ischemia", "Drusen"],
        "Vasculature": ["Vascular tortuosity", "Blood vessel dilation", "Retinal neovascularization"],
        "Veins": ["Venous beading", "Delayed filling", "Vessel wall staining"],
        "Capillary network": ["Microaneurysms", "Capillary nonperfusion", "Dot and blot hemorrhages"],
        "Foveal avascular zone": ["Avascular zone enlargement", "Macular ischemia"],
        "Choroid": ["Choroidal neovascularization", "Choroidal filling", "Geographic atrophy"],
        "Retinal pigment epithelium": ["Pigment epithelial detachment", "Window defect"],
        "Optic disc": ["Normal retinal appearance"],
        "Arteriovenous transit": ["Delayed filling", "Blocked fluorescence"],
        "Ischemic retina": ["Ischemic area", "Neovascular leakage", "Capillary nonperfusion"],
        "Fovea": ["Foveal leakage", "Leakage point"],
    }
    if all(name in idx for name in DISEASES):
        for disease, anatomies in disease_to_anatomy.items():
            for anatomy in anatomies:
                if anatomy in idx:
                    adj[idx[disease], idx[anatomy]] = 1.0
                    adj[idx[anatomy], idx[disease]] = 1.0
        for anatomy, findings in anatomy_to_findings.items():
            if anatomy not in idx:
                continue
            ids = [idx[f] for f in findings if f in idx]
            for f in ids:
                adj[idx[anatomy], f] = 1.0
                adj[f, idx[anatomy]] = 1.0
            for a in ids:
                for b in ids:
                    if a != b:
                        adj[a, b] = 1.0
        for node in [n for n in nodes if n.lower().startswith("others")]:
            for disease in DISEASES:
                adj[idx[disease], idx[node]] = 1.0
                adj[idx[node], idx[disease]] = 1.0
    else:
        disease_ids = list(range(1, min(6, len(nodes))))
        anatomy_ids = list(range(6, min(14, len(nodes))))
        finding_ids = list(range(14, len(nodes)))
        for d in disease_ids:
            for a in anatomy_ids:
                adj[d, a] = adj[a, d] = 1.0
        for pos, f in enumerate(finding_ids):
            a = anatomy_ids[pos % max(1, len(anatomy_ids))]
            adj[a, f] = adj[f, a] = 1.0
        for a in anatomy_ids:
            linked = [f for f in finding_ids if adj[a, f] > 0]
            for i in linked:
                for j in linked:
                    if i != j:
                        adj[i, j] = 1.0
    adj.fill_diagonal_(1.0)
    return nodes, adj, idx
