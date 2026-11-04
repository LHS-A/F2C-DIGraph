# RetinaBERT local assets

This directory contains only the local BioClinicalBERT configuration/tokenizer assets used by `models/retinabert.py` and `train/pretrain_retinabert.py`.

The previous standalone contrastive trainer was removed because it used pooled-CLS features and did not implement the updated mean-pooled, complementary two-description CUI objective. Use `scripts/build_retinabert_corpus.py` followed by `train/pretrain_retinabert.py`.
