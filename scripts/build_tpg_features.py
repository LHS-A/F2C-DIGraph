"""Precompute TPG features from an independent FFA mask CSV."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.csv_parser import infer_mask_schema, normalize_path_string, read_csv_robust
from data.tpg_features import build_tpg_from_masks


def resolve(value, base: Path) -> str:
    raw = "" if pd.isna(value) else str(value).strip()
    if not raw:
        return ""
    p = Path(raw)
    if p.exists() or p.is_absolute():
        return str(p)
    return str(base / raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_csv", required=True)
    parser.add_argument("--output_dir", default="outputs/tpg_features")
    parser.add_argument("--index_csv", default="outputs/tpg_features/tpg_feature_index.csv")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--rho_cross", type=float, default=0.1)
    args = parser.parse_args()

    df, enc = read_csv_robust(args.mask_csv)
    schema = infer_mask_schema(df, enc)
    base = Path(args.mask_csv).resolve().parent
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, row in df.iterrows():
        mask_paths = {}
        for mask_type, cols in schema.mask_cols.items():
            mask_paths[mask_type] = [resolve(row[c], base) for c in cols if c in row and normalize_path_string(row[c])]
        tpg = build_tpg_from_masks(mask_paths, args.image_size, args.rho_cross)
        sample_id = str(row.get(schema.patient_col, f"sample_{i}"))
        npz_name = f"{i:06d}_{abs(hash(sample_id)) % (10**8)}.npz"
        npz_path = out_dir / npz_name
        np.savez_compressed(npz_path, features=tpg.features, adjacency=tpg.adjacency, node_types=tpg.node_types)
        cfp_path = ""
        if schema.cfp_cols:
            cfp_path = normalize_path_string(row[schema.cfp_cols[0]])
        rows.append({"sample_id": sample_id, "cfp_path": cfp_path, "tpg_npz": str(npz_path)})
    Path(args.index_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.index_csv, index=False)
    print(f"Saved {len(rows)} TPG feature files and index: {args.index_csv}")


if __name__ == "__main__":
    main()
