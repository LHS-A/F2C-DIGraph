"""Classification metrics required by the F2C-DIGraph experiments."""

from typing import Dict, Iterable, List, Tuple

import numpy as np
try:
    from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
    from sklearn.preprocessing import label_binarize
except ImportError:
    accuracy_score = None
    confusion_matrix = None
    roc_auc_score = None
    label_binarize = None


def _confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    mat = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            mat[t, p] += 1
    return mat


def _binary_auc_np(y_true: np.ndarray, scores: np.ndarray) -> float:
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    for p in pos:
        wins += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return float(wins / (len(pos) * len(neg)))


def _macro_auc_np(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int) -> float:
    vals = []
    for i in range(n_classes):
        vals.append(_binary_auc_np((y_true == i).astype(int), y_prob[:, i]))
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def compute_metrics(
    y_true: Iterable[int],
    y_prob: np.ndarray,
    class_names: List[str],
) -> Dict[str, object]:
    """Compute ACC, macro one-vs-rest AUC, confusion matrix, and per-class scores."""
    y_true_arr = np.asarray(list(y_true), dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = y_prob.argmax(axis=1)
    n_classes = len(class_names)
    acc = float(accuracy_score(y_true_arr, y_pred)) if accuracy_score is not None else float((y_true_arr == y_pred).mean())
    cm = confusion_matrix(y_true_arr, y_pred, labels=list(range(n_classes))) if confusion_matrix is not None else _confusion_matrix_np(y_true_arr, y_pred, n_classes)
    metrics: Dict[str, object] = {"acc": acc, "confusion_matrix": cm.tolist()}
    if label_binarize is not None and roc_auc_score is not None:
        y_bin = label_binarize(y_true_arr, classes=list(range(n_classes)))
        try:
            metrics["auc_macro_ovr"] = float(roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr"))
        except ValueError:
            metrics["auc_macro_ovr"] = float("nan")
    else:
        y_bin = np.eye(n_classes, dtype=int)[y_true_arr]
        metrics["auc_macro_ovr"] = _macro_auc_np(y_true_arr, y_prob, n_classes)
    per_class_acc = {}
    per_class_auc = {}
    for i, name in enumerate(class_names):
        mask = y_true_arr == i
        per_class_acc[name] = float((y_pred[mask] == i).mean()) if mask.any() else float("nan")
        try:
            per_class_auc[name] = float(roc_auc_score(y_bin[:, i], y_prob[:, i])) if roc_auc_score is not None else _binary_auc_np(y_bin[:, i], y_prob[:, i])
        except ValueError:
            per_class_auc[name] = float("nan")
    metrics["per_class_acc"] = per_class_acc
    metrics["per_class_auc"] = per_class_auc
    return metrics


def mean_std(values: Iterable[float]) -> Tuple[float, float]:
    arr = np.asarray([v for v in values if not np.isnan(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1 if arr.size > 1 else 0))
