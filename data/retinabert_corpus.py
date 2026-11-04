"""Construction of the retinal-domain RetinaBERT knowledge corpus."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Set

import pandas as pd

from .csv_parser import load_and_infer, read_csv_robust


REQUIRED_MAPPING_COLUMNS = {
    "entity", "cui", "concept", "definition", "synonyms", "semantic_type",
    "relation", "tail_concept",
}


def _aliases(row: pd.Series) -> Iterable[str]:
    yield str(row["entity"]).strip().casefold()
    yield str(row["concept"]).strip().casefold()
    if pd.notna(row["synonyms"]):
        for synonym in str(row["synonyms"]).replace("|", ";").split(";"):
            if synonym.strip():
                yield synonym.strip().casefold()


def build_knowledge_corpus(
    metadata_csv: str,
    umls_mapping_csv: str,
    output_csv: str,
    spacy_model: str = "en_core_web_sm",
) -> pd.DataFrame:
    """Extract report entities and join their supplied UMLS concept records."""
    import spacy

    metadata, schema = load_and_infer(metadata_csv)
    if schema.report_col is None:
        raise ValueError("A report/impression text column is required to construct the RetinaBERT corpus.")
    reports = metadata[schema.report_col].dropna().astype(str).drop_duplicates()
    if reports.empty:
        raise ValueError("No expert-annotated reports were found in the metadata CSV.")

    mapping, _ = read_csv_robust(umls_mapping_csv)
    missing = REQUIRED_MAPPING_COLUMNS - set(mapping.columns)
    if missing:
        raise ValueError(f"UMLS mapping is missing columns: {sorted(missing)}")

    nlp = spacy.load(spacy_model)
    extracted: Set[str] = set()
    for document in nlp.pipe(reports.tolist(), batch_size=64):
        extracted.update(entity.text.strip().casefold() for entity in document.ents if entity.text.strip())

    selected = mapping[
        mapping.apply(lambda row: any(alias in extracted for alias in _aliases(row)), axis=1)
    ].copy()
    selected = selected.drop_duplicates(
        subset=["cui", "concept", "definition", "relation", "tail_concept"], keep="first"
    )
    if selected.empty:
        raise ValueError("spaCy entities did not match any supplied UMLS concept record.")
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_csv, index=False)
    return selected
