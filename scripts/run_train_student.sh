#!/usr/bin/env bash
set -euo pipefail
FOLD="${1:-0}"
python train/train_cfp_student.py \
  --config configs/train_student.yaml \
  --fold "$FOLD" \
  --teacher_ckpt "outputs/checkpoints/fold${FOLD}_ffa_teacher.pth" \
  --retinabert_ckpt outputs/checkpoints/retinabert.pth

