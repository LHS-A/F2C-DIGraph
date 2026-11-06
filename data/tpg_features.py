"""Topological Parameter Graph construction from registered FFA masks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from scipy.spatial.distance import cdist
from skimage import measure, morphology


MASK_TYPES = ("RV", "MA", "NPA", "NV", "LEAKAGE")
TYPE_TO_INDEX = {name: index for index, name in enumerate(MASK_TYPES)}


@dataclass
class TPG:
    features: np.ndarray
    adjacency: np.ndarray
    candidate_mask: np.ndarray
    node_types: np.ndarray


def _read_mask(path: str, image_size: int) -> np.ndarray:
    image = cv2.imread(str(Path(path)), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read FFA structure mask: {path}")
    return cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_NEAREST) > 0


def merge_mask_paths(paths: Sequence[str], image_size: int) -> np.ndarray:
    merged = np.zeros((image_size, image_size), dtype=bool)
    for path in paths:
        if path:
            merged |= _read_mask(path, image_size)
    return merged


def _fractal_dimension(mask: np.ndarray) -> float:
    sizes = 2 ** np.arange(1, max(2, int(np.log2(min(mask.shape))) - 1))
    counts, used_sizes = [], []
    for size in sizes:
        rows, cols = mask.shape[0] // size, mask.shape[1] // size
        if rows == 0 or cols == 0:
            continue
        blocks = mask[: rows * size, : cols * size].reshape(rows, size, cols, size).any(axis=(1, 3))
        count = int(blocks.sum())
        if count:
            counts.append(count)
            used_sizes.append(size)
    if len(counts) < 2:
        return 0.0
    return float(np.polyfit(np.log(1.0 / np.asarray(used_sizes)), np.log(counts), 1)[0])


def _component_features(
    mask: np.ndarray, node_type: str
) -> Tuple[List[np.ndarray], List[Tuple[float, float]], List[np.ndarray]]:
    labels = measure.label(mask, connectivity=2)
    features, centroids, boundaries = [], [], []
    for region in measure.regionprops(labels):
        component = labels == region.label
        area = float(region.area)
        perimeter = float(region.perimeter)
        if node_type == "MA":
            vector = [area, perimeter, 4.0 * np.pi * area / (perimeter * perimeter + 1e-6), region.eccentricity]
        elif node_type == "NPA":
            vector = [area, region.eccentricity, region.area_convex, region.equivalent_diameter_area]
        elif node_type == "NV":
            min_row, min_col, max_row, max_col = region.bbox
            aspect = (max_col - min_col + 1.0) / (max_row - min_row + 1.0)
            vector = [area, _fractal_dimension(component), region.solidity, aspect]
        elif node_type == "LEAKAGE":
            vector = [area, perimeter, region.axis_major_length, region.axis_minor_length]
        else:
            raise ValueError(f"Unsupported TPG structure: {node_type}")
        contours = measure.find_contours(component.astype(np.float32), 0.5)
        boundary = np.concatenate(contours) if contours else np.argwhere(component)
        features.append(np.asarray(vector, dtype=np.float32))
        centroids.append(tuple(region.centroid))
        boundaries.append(boundary.astype(np.float32))
    return features, centroids, boundaries


def _neighbor_count(skeleton: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    return cv2.filter2D(skeleton.astype(np.uint8), -1, kernel, borderType=cv2.BORDER_CONSTANT)


def _rv_features(
    vessel_mask: np.ndarray,
) -> Tuple[List[np.ndarray], List[Tuple[float, float]], List[np.ndarray], List[set[int]]]:
    """Split the skeleton into keypoint-to-keypoint vessel segments.

    Degree-four keypoint clusters are treated as crossings, while degree-three
    (or higher non-crossing) clusters are retained as true bifurcations.
    """
    skeleton = morphology.skeletonize(vessel_mask)
    degree = _neighbor_count(skeleton)
    critical = skeleton & (degree != 2)
    critical_labels = measure.label(critical, connectivity=2)
    true_bifurcations: set[int] = set()
    crossing_centers: Dict[int, np.ndarray] = {}
    for region in measure.regionprops(critical_labels):
        cluster = critical_labels == region.label
        dilation_ring = morphology.binary_dilation(cluster, morphology.square(3)) & skeleton & ~cluster
        outgoing = measure.label(dilation_ring, connectivity=2).max()
        if outgoing == 3 or outgoing >= 5:
            true_bifurcations.add(region.label)
        elif outgoing == 4:
            crossing_centers[region.label] = np.asarray(region.centroid, dtype=np.float32)

    corridors = skeleton & ~critical
    corridor_labels = measure.label(corridors, connectivity=2)
    distance = cv2.distanceTransform(vessel_mask.astype(np.uint8), cv2.DIST_L2, 5)
    features, centroids, boundaries, shared = [], [], [], []
    segment_crossings: List[set[int]] = []
    for region in measure.regionprops(corridor_labels):
        coords = region.coords
        if coords.shape[0] < 2:
            continue
        corridor = corridor_labels == region.label
        adjacent = morphology.binary_dilation(corridor, morphology.square(3)) & critical
        key_ids = set(np.unique(critical_labels[adjacent]).tolist()) - {0}
        bif_ids = key_ids & true_bifurcations
        length = float(coords.shape[0])
        corridor_degree = _neighbor_count(corridor)
        endpoints = np.argwhere(corridor & (corridor_degree <= 1))
        if endpoints.shape[0] >= 2:
            chord = float(cdist(endpoints, endpoints).max())
        else:
            centered = coords.astype(np.float32) - coords.mean(axis=0, keepdims=True)
            axis = np.linalg.svd(centered, full_matrices=False)[2][0]
            projection = centered @ axis
            chord = float(projection.max() - projection.min())
        chord = max(chord, 1e-6)
        radii = distance[coords[:, 0], coords[:, 1]]
        area = float(np.maximum(2.0 * radii, 1.0).sum())
        features.append(np.asarray([area, length, length / chord, np.std(radii)], dtype=np.float32))
        centroids.append(tuple(coords.mean(axis=0)))
        boundaries.append(coords.astype(np.float32))
        shared.append(bif_ids)
        segment_crossings.append(key_ids & set(crossing_centers))

    # A four-way crossing contains two distinct vessels. Pair the most nearly
    # opposite outgoing segments and connect only each continuation pair.
    next_connection_id = int(critical_labels.max()) + 1
    for crossing_id, center in crossing_centers.items():
        members = [index for index, ids in enumerate(segment_crossings) if crossing_id in ids]
        if len(members) < 2:
            continue
        directions = {}
        for index in members:
            coords = boundaries[index]
            vector = coords[np.argmax(np.linalg.norm(coords - center, axis=1))] - center
            norm = float(np.linalg.norm(vector))
            directions[index] = vector / max(norm, 1e-6)
        if len(members) == 4:
            a, b, c, d = members
            pairings = [((a, b), (c, d)), ((a, c), (b, d)), ((a, d), (b, c))]
            pairs = min(
                pairings,
                key=lambda pairing: sum(float(np.dot(directions[left], directions[right])) for left, right in pairing),
            )
        else:
            remaining = set(members)
            pairs = []
            while len(remaining) >= 2:
                left, right = min(
                    ((left, right) for left in remaining for right in remaining if left < right),
                    key=lambda pair: float(np.dot(directions[pair[0]], directions[pair[1]])),
                )
                pairs.append((left, right))
                remaining.remove(left)
                remaining.remove(right)
        for left, right in pairs:
            shared[left].add(next_connection_id)
            shared[right].add(next_connection_id)
            next_connection_id += 1

    # A closed vessel loop has no keypoint and therefore no corridor after the
    # usual split; retain each such skeleton component as one segment.
    if not features and skeleton.any():
        for region in measure.regionprops(measure.label(skeleton, connectivity=2)):
            coords = region.coords
            radii = distance[coords[:, 0], coords[:, 1]]
            length = float(coords.shape[0])
            features.append(np.asarray([np.maximum(2.0 * radii, 1.0).sum(), length, 1.0, np.std(radii)], dtype=np.float32))
            centroids.append(tuple(coords.mean(axis=0)))
            boundaries.append(coords.astype(np.float32))
            shared.append(set())
    return features, centroids, boundaries, shared


def _minimum_boundary_distance(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return float("inf")
    return float(cdist(left, right).min())


def _ordered_edges(
    adjacency: np.ndarray,
    indices: List[int],
    centroids: List[Tuple[float, float]],
    threshold: float,
) -> None:
    if len(indices) < 2:
        return
    points = np.asarray([centroids[index] for index in indices], dtype=np.float32)
    centered = points - points.mean(axis=0, keepdims=True)
    first_axis = np.linalg.svd(centered, full_matrices=False)[2][0]
    order = np.argsort(centered @ first_axis)
    for left_position, right_position in zip(order[:-1], order[1:]):
        left, right = indices[left_position], indices[right_position]
        if np.linalg.norm(points[left_position] - points[right_position]) < threshold:
            adjacency[left, right] = adjacency[right, left] = 1.0


def build_tpg_from_arrays(
    masks: Dict[str, np.ndarray], rho_cross: float = 0.1, kappa: int = 8
) -> TPG:
    """Build B0 and the sparse dynamic candidate mask from aligned masks."""
    if set(masks) != set(MASK_TYPES):
        missing = set(MASK_TYPES) - set(masks)
        raise ValueError(f"TPG masks must contain {MASK_TYPES}; missing {sorted(missing)}")
    shape = next(iter(masks.values())).shape
    features: List[np.ndarray] = []
    centroids: List[Tuple[float, float]] = []
    boundaries: List[np.ndarray] = []
    node_types: List[int] = []
    rv_shared: List[set[int]] = []
    by_type: Dict[str, List[int]] = {name: [] for name in MASK_TYPES}

    for mask_type in MASK_TYPES:
        mask = masks[mask_type].astype(bool)
        if mask_type == "RV":
            type_features, type_centroids, type_boundaries, rv_shared = _rv_features(mask)
        else:
            type_features, type_centroids, type_boundaries = _component_features(mask, mask_type)
        for vector, centroid, boundary in zip(type_features, type_centroids, type_boundaries):
            by_type[mask_type].append(len(features))
            features.append(vector)
            centroids.append(centroid)
            boundaries.append(boundary)
            node_types.append(TYPE_TO_INDEX[mask_type])

    if not features:
        return TPG(
            np.zeros((0, 4), np.float32), np.zeros((0, 0), np.float32),
            np.zeros((0, 0), bool), np.zeros((0,), np.int64)
        )
    x = np.nan_to_num(np.stack(features).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    count = len(features)
    adjacency = np.eye(count, dtype=np.float32)
    for local_left, global_left in enumerate(by_type["RV"]):
        for local_right, global_right in enumerate(by_type["RV"]):
            if local_left < local_right and (rv_shared[local_left] & rv_shared[local_right]):
                adjacency[global_left, global_right] = adjacency[global_right, global_left] = 1.0

    threshold = rho_cross * float(np.hypot(*shape))
    for mask_type in MASK_TYPES[1:]:
        _ordered_edges(adjacency, by_type[mask_type], centroids, threshold)

    # Every node selects up to kappa nearest eligible nodes from each other type.
    for source_type in MASK_TYPES:
        for target_type in MASK_TYPES:
            if source_type == target_type:
                continue
            for source in by_type[source_type]:
                ranked = sorted(
                    ((_minimum_boundary_distance(boundaries[source], boundaries[target]), target)
                     for target in by_type[target_type]),
                    key=lambda pair: pair[0],
                )
                for distance, target in ranked[:kappa]:
                    if distance <= threshold:
                        adjacency[source, target] = adjacency[target, source] = 1.0

    candidate = adjacency.astype(bool)
    for source in range(count):
        ranked = sorted(
            ((_minimum_boundary_distance(boundaries[source], boundaries[target]), target)
             for target in range(count) if target != source),
            key=lambda pair: pair[0],
        )
        for distance, target in ranked[:kappa]:
            if distance <= 2.0 * threshold:
                candidate[source, target] = candidate[target, source] = True
    np.fill_diagonal(candidate, True)
    return TPG(x, adjacency, candidate, np.asarray(node_types, dtype=np.int64))


def build_tpg_from_masks(
    mask_paths: Dict[str, Sequence[str]], image_size: int = 512, rho_cross: float = 0.1, kappa: int = 8
) -> TPG:
    masks = {name: merge_mask_paths(mask_paths.get(name, []), image_size) for name in MASK_TYPES}
    return build_tpg_from_arrays(masks, rho_cross=rho_cross, kappa=kappa)
