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
from .tpg_features import MASK_TYPES, build_tpg_from_arrays, merge_mask_paths
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


def _load_image(path: str, mode: str) -> Image.Image:
    if not path or not Path(path).exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert(mode)


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
        rho_cross: float = 0.1,
        kappa: int = 8,
        min_valid_overlap: float = 0.9,
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
        self.rho_cross = rho_cross
        self.kappa = kappa
        self.min_valid_overlap = min_valid_overlap
        self.transform = build_transforms(image_size, train=train)
        self.mask_df = None
        self.mask_schema = None
        self.mask_csv_dir = Path(".").resolve()
        if mask_csv and Path(mask_csv).exists():
            self.mask_df, enc = read_csv_robust(mask_csv)
            self.mask_schema = infer_mask_schema(self.mask_df, enc)
            self.mask_df["_patient_key"] = self.mask_df[self.mask_schema.patient_col].astype(str)
            self.mask_csv_dir = Path(mask_csv).resolve().parent
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
        phase_paths: List[Optional[str]] = []
        for phase in ("early", "mid", "late"):
            candidates = [
                _resolve_path(row[col], self.csv_dir)
                for col in self.schema.phase_cols.get(phase, [])
                if col in row and normalize_path_string(row[col])
            ]
            if not candidates:
                phase_tokens = {"early": ("early", "arterial"), "mid": ("mid", "venous"), "late": ("late",)}[phase]
                candidates = [path for path in paths if any(token in Path(path).stem.lower() for token in phase_tokens)]
            phase_paths.append(candidates[0] if candidates else None)
        missing = [phase for phase, path in zip(("early", "mid", "late"), phase_paths) if not path]
        if missing:
            raise FileNotFoundError(
                "The updated method requires registered grayscale early-, mid-, and late-phase FFA keyframes; "
                f"missing identifiable phases: {missing}. Use phase-named columns or filenames."
            )
        return [str(path) for path in phase_paths]

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

    @staticmethod
    def _parse_status(value: object) -> Optional[int]:
        if pd.isna(value):
            return None
        text = str(value).strip().lower().replace("_", " ")
        if text in {"1", "present", "positive", "observed"}:
            return 1
        if text in {"0", "absent", "verified absent", "negative", "verified negative"}:
            return 0
        if text in {"-1", "unavailable", "unknown", "not observed", "inadequate", "missing"}:
            return -1
        return None

    def _observation_status(self, row: pd.Series, mask_paths: Dict[str, List[str]]) -> np.ndarray:
        status = np.full(len(MASK_TYPES), -1, dtype=np.int64)
        aliases = {
            "RV": ("rv", "vessel"), "MA": ("ma", "microaneurysm"),
            "NPA": ("npa", "nonperfusion"), "NV": ("nv", "neovascular"),
            "LEAKAGE": ("leakage", "leak", "extravasation"),
        }
        sources = [row]
        if self.mask_df is not None and self.mask_schema is not None:
            patient = str(row[self.schema.patient_col])
            candidates = self.mask_df[self.mask_df["_patient_key"] == patient]
            if not candidates.empty:
                sources.append(candidates.iloc[0])
        for index, structure in enumerate(MASK_TYPES):
            found = False
            for source in sources:
                for column in source.index:
                    name = str(column).lower()
                    if ("status" in name or "observation" in name) and any(alias in name for alias in aliases[structure]):
                        parsed = self._parse_status(source[column])
                        if parsed is not None:
                            status[index] = parsed
                            found = True
                            break
                if found:
                    break
            else:
                # A non-empty verified mask proves presence. Missing masks never
                # prove absence and therefore default to unavailable.
                if mask_paths.get(structure):
                    status[index] = 1
        return status

    def _valid_overlap(self, row: pd.Series) -> np.ndarray:
        paths = []
        sources = [(row, self.csv_dir)]
        if self.mask_df is not None and self.mask_schema is not None:
            patient = str(row[self.schema.patient_col])
            candidates = self.mask_df[self.mask_df["_patient_key"] == patient]
            if not candidates.empty:
                sources.append((candidates.iloc[0], self.mask_csv_dir))
        for source, base in sources:
            for column in source.index:
                name = str(column).lower()
                if any(token in name for token in ("valid", "fov", "overlap")) and any(token in name for token in ("mask", "path")):
                    path = _resolve_path(source[column], base)
                    if path and Path(path).exists():
                        paths.append(path)
        if not paths:
            return np.ones((self.image_size, self.image_size), dtype=bool)
        overlap = np.ones((self.image_size, self.image_size), dtype=bool)
        for path in paths:
            mask = Image.open(path).convert("L").resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
            overlap &= np.asarray(mask) > 0
        return overlap

    def _tpg(self, masks: Dict[str, np.ndarray]):
        tpg = build_tpg_from_arrays(masks, self.rho_cross, self.kappa)
        return {
            "x": torch.from_numpy(tpg.features).float(),
            "adj": torch.from_numpy(tpg.adjacency).float(),
            "candidate_mask": torch.from_numpy(tpg.candidate_mask).bool(),
            "node_types": torch.from_numpy(tpg.node_types).long(),
        }

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.df.iloc[index]
        cfp_path = _resolve_path(row[self.schema.cfp_cols[0]], self.csv_dir)
        ffa_paths = self._ffa_paths(row)
        mask_paths = self._mask_paths(row)
        raw_masks = {name: merge_mask_paths(mask_paths.get(name, []), self.image_size) for name in MASK_TYPES}
        status = self._observation_status(row, mask_paths)
        cfp, ffa, masks, valid_overlap = self.transform(
            _load_image(cfp_path, "RGB"),
            [_load_image(path, "L") for path in ffa_paths],
            raw_masks,
            self._valid_overlap(row),
        )
        if float(valid_overlap.mean()) < self.min_valid_overlap:
            status[status == 1] = -1
        for structure_index, structure in enumerate(MASK_TYPES):
            if status[structure_index] == 1 and not masks[structure].any():
                status[structure_index] = -1
        sample = {
            "cfp": cfp,
            "ffa": ffa,
            "label": torch.tensor(self._label(row), dtype=torch.long),
            "patient_id": str(row[self.schema.patient_col]),
            "cfp_path": cfp_path,
            "ffa_paths": ffa_paths,
            "structure_masks": torch.from_numpy(np.stack([masks[name] for name in MASK_TYPES])).float(),
            "observation_status": torch.from_numpy(status).long(),
            "valid_overlap": valid_overlap,
        }
        if self.require_masks:
            sample["tpg"] = self._tpg(masks)
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
        "structure_masks": torch.stack([b["structure_masks"] for b in batch]),
        "observation_status": torch.stack([b["observation_status"] for b in batch]),
        "valid_overlap": torch.stack([b["valid_overlap"] for b in batch]),
    }
    if "tpg" in batch[0]:
        graphs = [b["tpg"] for b in batch]
        max_nodes = max(g["x"].shape[0] for g in graphs)
        feat_dim = graphs[0]["x"].shape[1]
        x = torch.zeros(len(graphs), max_nodes, feat_dim, dtype=torch.float32)
        adj = torch.zeros(len(graphs), max_nodes, max_nodes, dtype=torch.float32)
        candidate_mask = torch.zeros(len(graphs), max_nodes, max_nodes, dtype=torch.bool)
        node_types = torch.zeros(len(graphs), max_nodes, dtype=torch.long)
        node_mask = torch.zeros(len(graphs), max_nodes, dtype=torch.bool)
        for i, g in enumerate(graphs):
            n = g["x"].shape[0]
            x[i, :n] = g["x"]
            adj[i, :n, :n] = g["adj"]
            candidate_mask[i, :n, :n] = g["candidate_mask"]
            node_types[i, :n] = g["node_types"]
            node_mask[i, :n] = True
        out["tpg"] = {"x": x, "adj": adj, "candidate_mask": candidate_mask, "node_types": node_types, "node_mask": node_mask}
    return out


def build_weighted_sampler(dataset: RetinaDualDataset) -> WeightedRandomSampler:
    labels = [dataset._label(dataset.df.iloc[i]) for i in range(len(dataset))]
    counts = np.bincount(labels, minlength=len(dataset.classes)).astype(np.float32)
    weights = 1.0 / np.maximum(counts, 1.0)
    sample_weights = [weights[y] for y in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
