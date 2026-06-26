#!/usr/bin/env bash
set -euo pipefail
python train/train_5fold.py \
  --config configs/default.yaml \
  --csv "${1:-data/metadata.csv}"
