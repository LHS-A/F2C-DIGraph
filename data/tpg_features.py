"""Topological Parameter Graph feature extraction from FFA masks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from scipy.spatial.distance import cdist
from skimage import measure, morphology

MASK_TYPES = ("RV", "FAZ", "MA", "NPA", "NV")
TYPE_TO_INDEX = {name: i for i, name in enumerate(MASK_TYPES)}


@dataclass
class TPG:
    features: np.ndarray
    adjacency: np.ndarray
    node_types: np.ndarray


def _read_mask(path: str, image_size: int) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Required FFA segmentation mask not found: {path}")
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read mask: {path}")
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    return img > 0


def _merge_masks(paths: Sequence[str], image_size: int) -> np.ndarray:
    masks = [_read_mask(p, image_size) for p in paths if p]
    if not masks:
        return np.zeros((image_size, image_size), dtype=bool)
    out = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        out |= mask
    return out


def _fractal_dimension(mask: np.ndarray) -> float:
    mask = mask.astype(bool)
    sizes = 2 ** np.arange(1, int(np.log2(min(mask.shape))) - 1)
    counts = []
    for size in sizes:
        rows = mask.shape[0] // size
        cols = mask.shape[1] // size
        if rows == 0 or cols == 0:
            continue
        trimmed = mask[: rows * size, : cols * size]
        blocks = trimmed.reshape(rows, size, cols, size).any(axis=(1, 3))
        counts.append(blocks.sum())
    if len(counts) < 2 or any(c <= 0 for c in counts):
        return 0.0
    coeffs = np.polyfit(np.log(1.0 / sizes[: len(counts)]), np.log(counts), 1)
    return float(coeffs[0])


def _component_features(mask: np.ndarray, node_type: str) -> Tuple[List[np.ndarray], List[Tuple[float, float]], List[np.ndarray]]:
    label = measure.label(mask)
    feats: List[np.ndarray] = []
    centroids: List[Tuple[float, float]] = []
    boundaries: List[np.ndarray] = []
    for region in measure.regionprops(label):
        if region.area <= 2:
            continue
        comp = label == region.label
        perimeter = float(region.perimeter or 1.0)
        area = float(region.area)
        circularity = float(4.0 * np.pi * area / (perimeter * perimeter + 1e-6))
        solidity = float(region.solidity or 0.0)
        eccentricity = float(region.eccentricity or 0.0)
        eq_diameter = float(region.equivalent_diameter_area)
        hull_area = float(region.area_convex or area)
        minr, minc, maxr, maxc = region.bbox
        aspect = float((maxc - minc + 1) / (maxr - minr + 1))
        if node_type == "FAZ":
            acircularity = float(perimeter * perimeter / (4.0 * np.pi * area + 1e-6))
            vec = np.array([area, perimeter, acircularity, solidity], dtype=np.float32)
        elif node_type == "NPA":
            vec = np.array([area, eccentricity, hull_area, eq_diameter], dtype=np.float32)
        elif node_type == "MA":
            vec = np.array([area, perimeter, circularity, eccentricity], dtype=np.float32)
        elif node_type == "NV":
            vec = np.array([area, _fractal_dimension(comp), solidity, aspect], dtype=np.float32)
        else:
            vec = np.array([area, perimeter, circularity, solidity], dtype=np.float32)
        boundary = measure.find_contours(comp.astype(float), 0.5)
        boundary_points = np.concatenate(boundary, axis=0) if boundary else np.argwhere(comp)
        feats.append(vec)
        centroids.append(tuple(region.centroid))
        boundaries.append(boundary_points.astype(np.float32))
    return feats, centroids, boundaries


def _rv_features(mask: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[float, float]], List[np.ndarray], List[int]]:
    skeleton = morphology.skeletonize(mask)
    skel_u8 = (skeleton.astype(np.uint8) * 255)
    corners = cv2.goodFeaturesToTrack(skel_u8, maxCorners=512, qualityLevel=0.01, minDistance=3)
    endpoints = []
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
    for y, x in np.argwhere(skeleton & (neighbor_count <= 2)):
        endpoints.append((float(y), float(x)))
    keypoints = endpoints
    if corners is not None:
        for c in corners.reshape(-1, 2):
            x, y = c
            yy, xx = int(round(y)), int(round(x))
            if 0 <= yy < skeleton.shape[0] and 0 <= xx < skeleton.shape[1] and skeleton[yy, xx]:
                if neighbor_count[yy, xx] >= 4:
                    keypoints.append((float(yy), float(xx)))
    label = measure.label(skeleton)
    feats: List[np.ndarray] = []
    centroids: List[Tuple[float, float]] = []
    boundaries: List[np.ndarray] = []
    shared_keys: List[int] = []
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    for region in measure.regionprops(label):
        if region.area <= 3:
            continue
        coords = region.coords
        length = float(len(coords))
        minr, minc, maxr, maxc = region.bbox
        chord = float(np.linalg.norm(np.array([maxr - minr, maxc - minc], dtype=np.float32)) + 1e-6)
        tortuosity = length / chord
        radii = distance[coords[:, 0], coords[:, 1]]
        radius_var = float(np.std(radii))
        area = float(mask[coords[:, 0], coords[:, 1]].sum())
        feats.append(np.array([area, length, tortuosity, radius_var], dtype=np.float32))
        centroids.append(tuple(coords.mean(axis=0)))
        boundaries.append(coords.astype(np.float32))
        if keypoints:
            d = cdist(coords.astype(float), np.asarray(keypoints))
            shared_keys.append(int(d.min(axis=0).argmin()))
        else:
            shared_keys.append(len(shared_keys))
    return feats, centroids, boundaries, shared_keys


def _connect_ordered(adj: np.ndarray, indices: List[int], centroids: List[Tuple[float, float]]) -> None:
    if len(indices) < 2:
        return
    pts = np.asarray([centroids[i] for i in indices], dtype=np.float32)
    pts_centered = pts - pts.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(pts_centered, full_matrices=False)
    axis = vt[0]
    order = np.argsort(pts_centered @ axis)
    ordered = [indices[i] for i in order]
    for a, b in zip(ordered[:-1], ordered[1:]):
        adj[a, b] = adj[b, a] = 1.0


def _min_boundary_distance(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    return float(cdist(a, b).min())


def build_tpg_from_masks(mask_paths: Dict[str, Sequence[str]], image_size: int = 512, rho_cross: float = 0.1) -> TPG:
    """Build TPG nodes and clinically guided adjacency from phase-merged masks."""
    features: List[np.ndarray] = []
    centroids: List[Tuple[float, float]] = []
    boundaries: List[np.ndarray] = []
    node_types: List[int] = []
    rv_shared: List[int] = []
    by_type: Dict[str, List[int]] = {name: [] for name in MASK_TYPES}

    for mask_type in MASK_TYPES:
        merged = _merge_masks(mask_paths.get(mask_type, []), image_size)
        if mask_type == "RV":
            feats, cents, bounds, shared = _rv_features(merged)
            rv_shared = shared
        else:
            feats, cents, bounds = _component_features(merged, mask_type)
        for vec, cen, bound in zip(feats, cents, bounds):
            by_type[mask_type].append(len(features))
            features.append(vec)
            centroids.append(cen)
            boundaries.append(bound)
            node_types.append(TYPE_TO_INDEX[mask_type])

    if not features:
        raise ValueError("No TPG nodes were extracted from the provided masks.")

    x = np.stack(features).astype(np.float32)
    x = np.nan_to_num(x)
    x = np.log1p(np.maximum(x, 0.0))
    adj = np.eye(len(x), dtype=np.float32)
    for i, a in enumerate(by_type["RV"]):
        for j, b in enumerate(by_type["RV"]):
            if i < j and rv_shared[i] == rv_shared[j]:
                adj[a, b] = adj[b, a] = 1.0
    for mask_type in ("FAZ", "NPA", "MA", "NV"):
        _connect_ordered(adj, by_type[mask_type], centroids)

    diag = (2.0 ** 0.5) * image_size
    threshold = rho_cross * diag
    pairs = [("RV", "FAZ"), ("RV", "NPA"), ("RV", "MA"), ("RV", "NV"), ("MA", "NPA"), ("NPA", "NV"), ("FAZ", "NPA")]
    for left, right in pairs:
        for i in by_type[left]:
            for j in by_type[right]:
                if _min_boundary_distance(boundaries[i], boundaries[j]) < threshold:
                    adj[i, j] = adj[j, i] = 1.0
    return TPG(features=x, adjacency=adj, node_types=np.asarray(node_types, dtype=np.int64))
