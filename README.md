# F2C-DIGraph

This repository provides the PyTorch implementation of **F2C-DIGraph** for CFP-only retinal disease diagnosis with cross-modal semantic and topological invariance distillation from FFA.

The training pipeline contains three stages:

1. RetinaBERT pretraining for semantic concept encoding.
2. FFA teacher training.
3. CFP student training with semantic and topological distillation.

After training, inference uses only the CFP image branch.

## Environment

Install dependencies:

```bash
pip install -r requirements.txt
```

The implementation supports CUDA automatically when a GPU is available.

## Configuration

The main configuration files are:

- `configs/default.yaml`: common experiment settings.
- `configs/retinabert.yaml`: RetinaBERT pretraining.
- `configs/train_teacher.yaml`: FFA teacher training.
- `configs/train_student.yaml`: CFP student distillation training.
- `configs/inference.yaml`: evaluation and inference.

Before running, set the dataset and output paths in the YAML files:

```yaml
paths:
  csv: path/to/metadata.csv
  split_csv: outputs/folds/patient_level_5fold_split.csv
  mask_csv: path/to/mask_paths.csv
  tpg_feature_csv: outputs/tpg_features/tpg_feature_index.csv
  retinabert_ckpt: outputs/checkpoints/retinabert.pth
```

The metadata CSV should contain CFP image paths, paired FFA image paths, disease labels, and patient identifiers. The mask CSV stores the RV, FAZ, MA, NPA, and NV mask paths used for TPG construction.

## Step 1: Train RetinaBERT

```bash
python train/pretrain_retinabert.py \
  --config configs/retinabert.yaml
```

Output:

- `outputs/checkpoints/retinabert.pth`
- `outputs/logs/pretrain_retinabert.log`

The trained RetinaBERT checkpoint is used to initialize the Semantic Chain Graph nodes from `SGC_nodes.txt`.

## Step 2: Build TPG Features

```bash
python scripts/build_tpg_features.py \
  --mask_csv path/to/mask_paths.csv \
  --output_dir outputs/tpg_features \
  --index_csv outputs/tpg_features/tpg_feature_index.csv
```

Output:

- `outputs/tpg_features/*.npz`
- `outputs/tpg_features/tpg_feature_index.csv`

Set the generated index in the config:

```yaml
paths:
  tpg_feature_csv: outputs/tpg_features/tpg_feature_index.csv
```

## Step 3: Train FFA Teacher

```bash
python train/train_ffa_teacher.py \
  --config configs/train_teacher.yaml \
  --fold 0
```

Output:

- `outputs/checkpoints/fold0_ffa_teacher.pth`
- `outputs/logs/fold0_teacher.log`

Repeat with `--fold 1` to `--fold 4` for the remaining folds.

## Step 4: Train CFP Student

```bash
python train/train_cfp_student.py \
  --config configs/train_student.yaml \
  --fold 0 \
  --teacher_ckpt outputs/checkpoints/fold0_ffa_teacher.pth \
  --retinabert_ckpt outputs/checkpoints/retinabert.pth
```

Output:

- `outputs/checkpoints/fold0_student_best.pth`
- `outputs/logs/fold0_student.log`

This stage optimizes:

- disease classification loss,
- graph-aware semantic distillation loss,
- row-wise topology response distillation loss.

Repeat with `--fold 1` to `--fold 4` using the corresponding teacher checkpoints.

## Evaluation

Evaluate a trained student checkpoint:

```bash
python train/evaluate.py \
  --config configs/inference.yaml \
  --checkpoint outputs/checkpoints/fold0_student_best.pth \
  --fold 0
```

Output:

- `outputs/results/fold0_metrics.json`
- `outputs/results/fold0_confusion_matrix.csv`

Reported metrics include accuracy, one-vs-rest macro AUC, per-class accuracy, per-class AUC, and confusion matrix.

## Inference

For inference, use the trained CFP student checkpoint. The FFA teacher and FFA-only graph construction modules are not used during deployment.

```bash
python train/evaluate.py \
  --config configs/inference.yaml \
  --checkpoint outputs/checkpoints/fold0_student_best.pth \
  --fold 0
```

## Main Modules

- `data/csv_parser.py`: metadata parsing and column inference.
- `data/splits.py`: patient-level 5-fold split generation.
- `data/tpg_features.py`: TPG node, feature, and edge construction from FFA masks.
- `models/retinabert.py`: RetinaBERT semantic encoder.
- `models/semantic_chain_graph.py`: Semantic Chain Graph construction.
- `models/semantic_dynamic_encoder.py`: dynamic semantic graph encoder.
- `models/topological_dynamic_encoder.py`: dynamic topological graph encoder.
- `models/topology_adapter.py`: CFP topology adapter.
- `models/cid_losses.py`: semantic and topological distillation losses.
- `models/f2c_digraph.py`: teacher, student, and full F2C-DIGraph model.

## Outputs

All experiment artifacts are saved under `outputs/`:

- `outputs/folds/`: generated patient-level splits.
- `outputs/checkpoints/`: RetinaBERT, teacher, and student checkpoints.
- `outputs/logs/`: training and evaluation logs.
- `outputs/results/`: per-fold and averaged metrics.
- `outputs/tpg_features/`: optional precomputed TPG feature files.
