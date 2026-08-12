from __future__ import annotations

import re
from typing import Iterable

import pandas as pd


def clean_text(text: object) -> str:
    """Normalize raw plot text for a consistent TF-IDF pipeline."""
    if pd.isna(text):
        return ""

    normalized = str(text).lower().strip()
    normalized = normalized.replace("\n", " ")
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def prepare_dataset(df: pd.DataFrame, text_column: str = "plot", label_column: str = "genre") -> pd.DataFrame:
    """Clean nulls, duplicates, and ensure the dataset has a consistent label/text schema."""
    cleaned = df.copy()

    required_columns = {text_column, label_column}
    missing_columns = required_columns - set(cleaned.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    cleaned[text_column] = cleaned[text_column].map(clean_text)
    cleaned[label_column] = cleaned[label_column].fillna("unknown").astype(str).str.strip()

    cleaned = cleaned.dropna(subset=[text_column]).copy()
    cleaned = cleaned.drop_duplicates(subset=[text_column, label_column], keep="first").reset_index(drop=True)
    return cleaned


def get_text_series(df: pd.DataFrame, text_column: str = "plot") -> Iterable[str]:
    """Return the cleaned plot text values as strings."""
    return df[text_column].fillna("").astype(str).tolist()
