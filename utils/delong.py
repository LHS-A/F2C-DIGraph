"""DeLong test for comparing two correlated binary ROC AUCs."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from scipy.stats import norm


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    ranks = np.zeros(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def _fast_delong(predictions: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels).astype(int)
    predictions = np.asarray(predictions, dtype=float)
    if predictions.ndim == 1:
        predictions = predictions[None, :]
    pos = labels == 1
    neg = labels == 0
    m = int(pos.sum())
    n = int(neg.sum())
    if m == 0 or n == 0:
        raise ValueError("DeLong test requires both positive and negative samples.")

    preds = np.concatenate([predictions[:, pos], predictions[:, neg]], axis=1)
    k = preds.shape[0]
    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)
    for r in range(k):
        tx[r] = _midrank(preds[r, :m])
        ty[r] = _midrank(preds[r, m:])
        tz[r] = _midrank(preds[r])

    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1.0) / 2.0) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01, bias=False)
    sy = np.cov(v10, bias=False)
    covariance = sx / m + sy / n
    covariance = np.atleast_2d(covariance)
    return aucs, covariance


def delong_roc_test(labels: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """Return the two-sided p-value for two correlated binary ROC AUCs.

    Args:
        labels: Binary ground-truth labels encoded as 0/1.
        scores_a: Continuous scores from model A for the positive class.
        scores_b: Continuous scores from model B for the positive class.
    """
    labels = np.asarray(labels).astype(int)
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("labels must be binary and encoded as 0/1.")
    preds = np.vstack([np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)])
    aucs, covariance = _fast_delong(preds, labels)
    contrast = np.array([[1.0, -1.0]])
    variance = float(contrast @ covariance @ contrast.T)
    if variance <= 0:
        return 1.0 if math.isclose(float(aucs[0]), float(aucs[1])) else 0.0
    z = abs(float(np.diff(aucs)[0])) / math.sqrt(variance)
    return float(2.0 * norm.sf(z))
