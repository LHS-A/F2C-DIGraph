#!/usr/bin/env bash
set -euo pipefail
python train/train_ffa_teacher.py --config configs/train_teacher.yaml --fold "${1:-0}"

