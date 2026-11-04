"""Robust schema inference for RetinaDUAL and bundled test CSV files."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import re

import pandas as pd


ENCODINGS = ("utf-8", "utf-8-sig", "gbk", "gb18030")
DISEASES = ("AMD", "CSC", "DR", "RVO", "Normal")


@dataclass
class CsvSchema:
    cfp_cols: List[str]
    ffa_cols: List[str]
    label_col: str
    patient_col: str
    report_col: Optional[str]
    split_cols: List[str]
    fold_cols: List[str]
    phase_cols: Dict[str, List[str]]
    mask_cols: Dict[str, List[str]]
    encoding: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def read_csv_robust(path: str) -> Tuple[pd.DataFrame, str]:
    """Load a CSV using the requested encoding fallbacks."""
    errors = []
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc), enc
        except UnicodeDecodeError as exc:
            errors.append(f"{enc}: {exc}")
    raise UnicodeDecodeError("csv", b"", 0, 1, "Unable to decode with " + "; ".join(errors))


def normalize_path_string(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().replace("\\", "/")


def _score_cols(columns: Sequence[str], include: Sequence[str], exclude: Sequence[str] = ()) -> List[str]:
    scored = []
    for col in columns:
        c = col.lower()
        if any(e in c for e in exclude):
            continue
        score = sum(1 for key in include if key in c)
        if score:
            scored.append((score, col))
    return [col for _, col in sorted(scored, key=lambda x: (-x[0], x[1].lower()))]


def _value_path_hits(df: pd.DataFrame, pattern: str) -> List[str]:
    hits = []
    for col in df.columns:
        vals = df[col].dropna().astype(str).head(100).str.replace("\\", "/", regex=False).str.lower()
        if not vals.empty and vals.str.contains(pattern, regex=True).mean() > 0.2:
            hits.append(col)
    return hits


def _infer_label_col(df: pd.DataFrame) -> str:
    candidates = _score_cols(df.columns, ("label", "disease", "diagnosis", "category", "class"))
    for col in candidates:
        vals = df[col].dropna().astype(str).str.strip()
        if vals.empty:
            continue
        hits = vals.str.upper().isin([d.upper() for d in DISEASES]).mean()
        if hits > 0.4 or vals.nunique() <= 20:
            return col
    raise ValueError("Could not infer disease label column. Expected AMD/CSC/DR/RVO/Normal labels.")


def _infer_patient_col(df: pd.DataFrame) -> str:
    candidates = _score_cols(df.columns, ("patient", "pid", "case", "subject", "name", "id"), ("image", "path", "file"))
    for col in candidates:
        if df[col].dropna().nunique() < len(df):
            return col
    if candidates:
        return candidates[0]
    path_like = set(_value_path_hits(df, r"(?:^|/)(?:cfp|ffa)(?:/|_)"))
    label_like = {c for c in df.columns if c.lower() == "label" or "report" in c.lower()}
    for col in df.columns:
        if col not in path_like and col not in label_like and df[col].dropna().nunique() < len(df):
            return col
    raise ValueError("Could not infer patient ID/name column.")


def _infer_mask_cols(columns: Sequence[str], allow_path_without_mask: bool = False) -> Dict[str, List[str]]:
    aliases = {
        "RV": ("rv", "vessel", "retinal_vessel"),
        "MA": ("ma", "microaneurysm"),
        "NPA": ("npa", "nonperfusion", "non_perfusion"),
        "NV": ("nv", "neovascular"),
        "LEAKAGE": ("leakage", "leak", "extravasation"),
    }
    mask_cols: Dict[str, List[str]] = {}
    for key, keys in aliases.items():
        selected = []
        for col in columns:
            c = col.lower()
            has_type = any(re.search(rf"(?:^|[_\-\s]){re.escape(alias)}(?:[_\-\s]|$)", c) or alias in c for alias in keys)
            has_mask = "mask" in c or "seg" in c or (allow_path_without_mask and "path" in c)
            if has_type and has_mask:
                selected.append(col)
        mask_cols[key] = selected
    return mask_cols


def infer_schema(df: pd.DataFrame, encoding: str = "") -> CsvSchema:
    """Infer CFP, FFA, label, patient, report, split/fold, phase, and mask columns."""
    columns = list(df.columns)
    cfp_value_cols = _value_path_hits(df, r"(?:^|/)cfp(?:/|_)")
    ffa_value_cols = _value_path_hits(df, r"(?:^|/)ffa(?:/|_)")
    path_cols = _score_cols(columns, ("path", "file", "image", "img"))
    cfp_name_cols = _score_cols(columns, ("cfp", "fundus", "color", "photo", "fc"), ("mask", "seg"))
    ffa_name_cols = _score_cols(columns, ("ffa", "fluorescein", "angiography"), ("mask", "seg"))
    cfp_cols = cfp_value_cols or [c for c in cfp_name_cols if c in path_cols or "path" in c.lower() or "image" in c.lower()] or [
        c for c in path_cols if "cfp" in c.lower() or "fundus" in c.lower()
    ]
    ffa_cols = ffa_value_cols or [c for c in ffa_name_cols if c in path_cols or "path" in c.lower() or "image" in c.lower()] or [
        c for c in path_cols if "ffa" in c.lower() or "fluorescein" in c.lower()
    ]
    if not cfp_cols:
        raise ValueError("Could not infer CFP image path column.")
    if not ffa_cols:
        raise ValueError("Could not infer FFA image path column(s).")
    return CsvSchema(
        cfp_cols=cfp_cols,
        ffa_cols=ffa_cols,
        label_col=_infer_label_col(df),
        patient_col=_infer_patient_col(df),
        report_col=(_score_cols(columns, ("report", "impression", "description", "text", "note")) or [None])[0],
        split_cols=_score_cols(columns, ("split", "set", "partition")),
        fold_cols=_score_cols(columns, ("fold", "cv")),
        phase_cols={
            "early": _score_cols(columns, ("early", "arterial", "phase1")),
            "mid": _score_cols(columns, ("mid", "middle", "venous", "phase2")),
            "late": _score_cols(columns, ("late", "phase3")),
        },
        mask_cols=_infer_mask_cols(columns),
        encoding=encoding,
    )


def infer_mask_schema(df: pd.DataFrame, encoding: str = "") -> CsvSchema:
    """Infer schema for an independent mask CSV without requiring disease labels."""
    columns = list(df.columns)
    try:
        patient_col = _infer_patient_col(df)
    except ValueError:
        patient_col = columns[0]
    return CsvSchema(
        cfp_cols=_value_path_hits(df, r"(?:^|/)cfp(?:/|_)") or _score_cols(columns, ("cfp", "fc", "fundus")),
        ffa_cols=_value_path_hits(df, r"(?:^|/)ffa(?:/|_)") or _score_cols(columns, ("ffa", "fluorescein", "angiography")),
        label_col="",
        patient_col=patient_col,
        report_col=None,
        split_cols=[],
        fold_cols=[],
        phase_cols={"early": [], "mid": [], "late": []},
        mask_cols=_infer_mask_cols(columns, allow_path_without_mask=True),
        encoding=encoding,
    )


def load_and_infer(csv_path: str) -> Tuple[pd.DataFrame, CsvSchema]:
    df, enc = read_csv_robust(csv_path)
    return df, infer_schema(df, enc)


def write_inspection_report(csv_path: str, output: str) -> CsvSchema:
    """Inspect a CSV and write a schema report."""
    df, schema = load_and_infer(csv_path)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"CSV: {csv_path}",
        f"Rows: {len(df)}",
        f"Encoding: {schema.encoding}",
        "Columns:",
        *[f"  - {c}" for c in df.columns],
        f"Inferred CFP image path columns: {schema.cfp_cols}",
        f"Inferred FFA image path columns: {schema.ffa_cols}",
        f"Inferred disease label column: {schema.label_col}",
        f"Inferred patient ID/name column: {schema.patient_col}",
        f"Inferred report/text column: {schema.report_col}",
        f"Inferred split columns: {schema.split_cols}",
        f"Inferred fold columns: {schema.fold_cols}",
        f"Inferred FFA phase columns: {schema.phase_cols}",
        f"Inferred mask columns: {schema.mask_cols}",
    ]
    Path(output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return schema

