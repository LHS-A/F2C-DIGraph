"""RetinaDUAL dataset for FFA teacher and CFP student training."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler

from .csv_parser import CsvSchema, infer_mask_schema, load_and_infer, normalize_path_string, read_csv_robust
from .splits import attach_split
from .tpg_features import build_tpg_from_masks
from .transforms import build_transforms


def _resolve_path(path_value: object, csv_dir: Path) -> str:
    raw = "" if pd.isna(path_value) else str(path_value).strip()
    if not raw:
        return ""
    p = Path(raw)
    if p.exists():
        return str(p)
    alt = Path(raw.replace("\\", "/"))
    if alt.exists():
        return str(alt)
    joined = csv_dir / raw
    if joined.exists():
        return str(joined)
    return raw


def _load_rgb(path: str) -> Image.Image:
    if not path or not Path(path).exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


class RetinaDualDataset(Dataset):
    """Dataset returning CFP image, three FFA frames, label, optional TPG, and metadata."""

    def __init__(
        self,
        csv_path: str,
        split_csv: str,
        fold: int,
        split: str,
        classes: Sequence[str],
        image_size: int = 512,
        train: bool = True,
        require_masks: bool = False,
        mask_csv: Optional[str] = None,
        tpg_feature_csv: Optional[str] = None,
    ) -> None:
        self.df, self.schema = load_and_infer(csv_path)
        self.df = attach_split(self.df, self.schema, split_csv, fold)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No rows found for split={split}, fold={fold}.")
        self.csv_dir = Path(csv_path).resolve().parent
        self.ffa_group_map = {}
        for _, r in self.df.iterrows():
            cfp_key = normalize_path_string(r[self.schema.cfp_cols[0]])
            paths = [_resolve_path(r[col], self.csv_dir) for col in self.schema.ffa_cols if normalize_path_string(r[col])]
            self.ffa_group_map.setdefault(cfp_key, [])
            self.ffa_group_map[cfp_key].extend(paths)
        self.df = self.df.drop_duplicates(subset=[self.schema.cfp_cols[0], self.schema.patient_col, self.schema.label_col]).reset_index(drop=True)
        self.classes = list(classes)
        self.class_to_idx = {name.lower(): i for i, name in enumerate(self.classes)}
        self.image_size = image_size
        self.require_masks = require_masks
        self.transform = build_transforms(image_size, train=train)
        self.mask_df = None
        self.mask_schema = None
        self.mask_csv_dir = Path(".").resolve()
        if mask_csv and Path(mask_csv).exists():
            self.mask_df, enc = read_csv_robust(mask_csv)
            self.mask_schema = infer_mask_schema(self.mask_df, enc)
            self.mask_df["_patient_key"] = self.mask_df[self.mask_schema.patient_col].astype(str)
            self.mask_csv_dir = Path(mask_csv).resolve().parent
        self.tpg_index = {}
        if tpg_feature_csv and Path(tpg_feature_csv).exists():
            tpg_df, _ = read_csv_robust(tpg_feature_csv)
            tpg_dir = Path(tpg_feature_csv).resolve().parent
            for _, r in tpg_df.iterrows():
                key = str(r.get("cfp_path", r.get("sample_id", ""))).replace("\\", "/")
                value = str(r.get("tpg_npz", ""))
                if key and value:
                    p = Path(value)
                    self.tpg_index[key] = str(p if p.is_absolute() else tpg_dir / p)

    def __len__(self) -> int:
        return len(self.df)

    def _label(self, row: pd.Series) -> int:
        value = str(row[self.schema.label_col]).strip()
        key = value.lower()
        if key not in self.class_to_idx:
            raise ValueError(f"Unknown disease label '{value}'. Expected one of {self.classes}.")
        return self.class_to_idx[key]

    def _ffa_paths(self, row: pd.Series) -> List[str]:
        cfp_key = normalize_path_string(row[self.schema.cfp_cols[0]])
        paths = list(dict.fromkeys(self.ffa_group_map.get(cfp_key, [])))
        if not paths:
            paths = [_resolve_path(row[col], self.csv_dir) for col in self.schema.ffa_cols if normalize_path_string(row[col])]
        if not paths:
            raise FileNotFoundError("No FFA paths found for row.")
        while len(paths) < 3:
            paths.append(paths[-1])
        return paths[:3]

    def _mask_paths(self, row: pd.Series) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for mask_type, cols in self.schema.mask_cols.items():
            values = [_resolve_path(row[col], self.csv_dir) for col in cols if col in row and normalize_path_string(row[col])]
            out[mask_type] = values
        if self.mask_df is not None and self.mask_schema is not None:
            patient = str(row[self.schema.patient_col])
            candidates = self.mask_df[self.mask_df["_patient_key"] == patient]
            if not candidates.empty:
                mask_row = candidates.iloc[0]
                for mask_type, cols in self.mask_schema.mask_cols.items():
                    values = [_resolve_path(mask_row[col], self.mask_csv_dir) for col in cols if col in mask_row and normalize_path_string(mask_row[col])]
                    if values:
                        out[mask_type] = values
        return out

    def _tpg(self, row: pd.Series):
        cfp_path = _resolve_path(row[self.schema.cfp_cols[0]], self.csv_dir)
        norm_cfp = cfp_path.replace("\\", "/")
        rel_cfp = normalize_path_string(row[self.schema.cfp_cols[0]])
        npz_path = self.tpg_index.get(norm_cfp) or self.tpg_index.get(rel_cfp)
        if npz_path and Path(npz_path).exists():
            data = np.load(npz_path)
            tpg = type("CachedTPG", (), {"features": data["features"], "adjacency": data["adjacency"], "node_types": data["node_types"]})
        else:
            mask_paths = self._mask_paths(row)
            required = ("RV", "FAZ", "MA", "NPA", "NV")
            missing = [m for m in required if not mask_paths.get(m)]
            if missing:
                raise FileNotFoundError(
                    "Strict reproduction requires FFA segmentation mask paths for "
                    f"{missing}. Add columns for RV, FAZ, MA, NPA, and NV masks, "
                    "provide paths through mask_csv, or provide precomputed TPG npz files through tpg_feature_csv."
                )
            tpg = build_tpg_from_masks(mask_paths, self.image_size)
        return {
            "x": torch.from_numpy(tpg.features).float(),
            "adj": torch.from_numpy(tpg.adjacency).float(),
            "node_types": torch.from_numpy(tpg.node_types).long(),
        }

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.df.iloc[index]
        cfp_path = _resolve_path(row[self.schema.cfp_cols[0]], self.csv_dir)
        ffa_paths = self._ffa_paths(row)
        cfp = self.transform(_load_rgb(cfp_path))
        ffa_frames = [self.transform(_load_rgb(path)) for path in ffa_paths]
        sample = {
            "cfp": cfp,
            "ffa": torch.cat(ffa_frames, dim=0),
            "label": torch.tensor(self._label(row), dtype=torch.long),
            "patient_id": str(row[self.schema.patient_col]),
            "cfp_path": cfp_path,
            "ffa_paths": ffa_paths,
        }
        if self.require_masks or self.tpg_index:
            sample["tpg"] = self._tpg(row)
        return sample


def collate_with_tpg(batch: List[Dict[str, object]]) -> Dict[str, object]:
    """Collate images and pad variable-size TPGs into tensor batches."""
    out = {
        "cfp": torch.stack([b["cfp"] for b in batch]),
        "ffa": torch.stack([b["ffa"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "patient_id": [b["patient_id"] for b in batch],
        "cfp_path": [b["cfp_path"] for b in batch],
        "ffa_paths": [b["ffa_paths"] for b in batch],
    }
    if "tpg" in batch[0]:
        graphs = [b["tpg"] for b in batch]
        max_nodes = max(g["x"].shape[0] for g in graphs)
        feat_dim = graphs[0]["x"].shape[1]
        x = torch.zeros(len(graphs), max_nodes, feat_dim, dtype=torch.float32)
        adj = torch.zeros(len(graphs), max_nodes, max_nodes, dtype=torch.float32)
        node_types = torch.zeros(len(graphs), max_nodes, dtype=torch.long)
        node_mask = torch.zeros(len(graphs), max_nodes, dtype=torch.bool)
        for i, g in enumerate(graphs):
            n = g["x"].shape[0]
            x[i, :n] = g["x"]
            adj[i, :n, :n] = g["adj"]
            node_types[i, :n] = g["node_types"]
            node_mask[i, :n] = True
        out["tpg"] = {"x": x, "adj": adj, "node_types": node_types, "node_mask": node_mask}
    return out


def build_weighted_sampler(dataset: RetinaDualDataset) -> WeightedRandomSampler:
    labels = [dataset._label(dataset.df.iloc[i]) for i in range(len(dataset))]
    counts = np.bincount(labels, minlength=len(dataset.classes)).astype(np.float32)
    weights = 1.0 / np.maximum(counts, 1.0)
    sample_weights = [weights[y] for y in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
