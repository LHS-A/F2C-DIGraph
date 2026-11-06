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
OTHER_NODE_TO_DISEASE = {
    node: DISEASES[position % len(DISEASES)] for position, node in enumerate(OTHER_NODES)
}
SCG_NODES = ["[CLS]"] + DISEASES + ANATOMY + FINDINGS + OTHER_NODES


def load_sgc_nodes(path: Optional[str], target_k: int = 66) -> Tuple[List[str], List[str]]:
    """Return the fixed 66-node taxonomy and attach supplied descriptions by name."""
    if target_k != 66:
        raise ValueError("The updated SCG has exactly K=66 nodes.")
    definitions: Dict[str, str] = {}
    if path and Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            name, definition = (part.strip() for part in line.split(":", 1))
            definitions[name.casefold()] = definition
    aliases = {
        "AMD": "age-related macular degeneration",
        "CSC": "central serous chorioretinopathy",
        "DR": "diabetic retinopathy",
        "RVO": "retinal vein occlusion",
        "Vasculature": "retinal vasculature",
        "Veins": "retinal veins",
        "Capillary network": "retinal capillary network",
    }
    texts = []
    for node in SCG_NODES:
        key = node.casefold()
        definition = definitions.get(key) or definitions.get(aliases.get(node, "").casefold(), "")
        if node == "[CLS]":
            definition = "global retinal diagnostic aggregation node"
        elif node.startswith("Others") and not definition:
            disease = OTHER_NODE_TO_DISEASE[node]
            definition = (
                f"residual retinal finding associated with the {disease} diagnostic branch "
                "outside its predefined findings"
            )
        elif not definition and node in DISEASES:
            definition = f"expert-verified retinal diagnostic category: {node}"
        elif not definition and node in ANATOMY:
            definition = f"expert-verified retinal anatomical region: {node}"
        elif not definition and node in FINDINGS:
            definition = f"expert-verified retinal clinical finding: {node}"
        texts.append(f"{node}. Definition: {definition}" if definition else node)
    return list(SCG_NODES), texts


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
        others = [n for n in nodes if n.lower().startswith("others")]
        for node in others:
            disease = OTHER_NODE_TO_DISEASE[node]
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
    # A0 contains expert-defined clinical edges only. Self-connections are
    # introduced by the graph encoders/normalization exactly where specified.
    adj.fill_diagonal_(0.0)
    return nodes, adj, idx
