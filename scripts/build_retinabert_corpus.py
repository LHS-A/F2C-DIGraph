"""Build the report-derived UMLS corpus used to pretrain RetinaBERT."""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.retinabert_corpus import build_knowledge_corpus


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--umls_mapping_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--spacy_model", default="en_core_web_sm")
    args = parser.parse_args()
    corpus = build_knowledge_corpus(
        args.metadata_csv,
        args.umls_mapping_csv,
        args.output_csv,
        args.spacy_model,
    )
    print(f"Saved {len(corpus)} UMLS knowledge rows to {args.output_csv}")
