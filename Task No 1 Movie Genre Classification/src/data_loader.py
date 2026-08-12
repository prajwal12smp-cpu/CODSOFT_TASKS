from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Tuple

import kagglehub
import pandas as pd


DEFAULT_DATASET = "hijest/genre-classification-dataset-imdb"


def download_dataset(dataset_name: str = DEFAULT_DATASET) -> Path:
    """Download the Kaggle dataset and return the dataset root directory."""
    try:
        dataset_path = Path(kagglehub.dataset_download(dataset_name))
        return dataset_path
    except Exception as exc:  # pragma: no cover - depends on KaggleHub availability
        raise RuntimeError(f"Unable to download Kaggle dataset '{dataset_name}'.") from exc


def find_dataset_files(root_dir: str | os.PathLike[str]) -> List[Path]:
    """Return all CSV/TSV/TXT files in the Kaggle dataset directory."""
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {root}")

    files = [
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".csv", ".tsv", ".txt"}
    ]
    return sorted(files)


def parse_text_dataset(file_path: str | os.PathLike[str]) -> pd.DataFrame:
    """Parse the Kaggle IMDb genre dataset stored in the ':::' delimited text format."""
    file_path = Path(file_path)
    records: List[dict] = []

    with file_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw_line = line.strip()
            if not raw_line:
                continue

            parts = [part.strip() for part in raw_line.split(" ::: ")]
            if len(parts) < 4:
                continue

            movie_id, title, genre, plot = parts[0], parts[1], parts[2], " ::: ".join(parts[3:])
            records.append({
                "movie_id": movie_id,
                "title": title,
                "genre": genre,
                "plot": plot,
            })

    if not records:
        raise ValueError(f"No valid records found in dataset file: {file_path}")

    return pd.DataFrame(records)


def auto_detect_training_data(dataset_root: str | os.PathLike[str]) -> pd.DataFrame:
    """Inspect all detected files and select the training dataset automatically."""
    dataset_root = Path(dataset_root)
    files = find_dataset_files(dataset_root)

    if not files:
        raise FileNotFoundError(f"No data files were found in {dataset_root}")

    candidate_paths = [
        path for path in files
        if path.name.lower().startswith("train") and path.suffix.lower() in {".txt", ".csv", ".tsv"}
    ]

    if not candidate_paths:
        candidate_paths = files

    for file_path in candidate_paths:
        ext = file_path.suffix.lower()
        try:
            if ext == ".csv":
                df = pd.read_csv(file_path)
            elif ext == ".tsv":
                df = pd.read_csv(file_path, sep="\t")
            else:
                df = parse_text_dataset(file_path)

            required = {"title", "genre", "plot"}
            if required.issubset(df.columns):
                return df

            if {"movie_id", "title", "genre", "plot"}.issubset(df.columns):
                return df

            # The Kaggle dataset is formatted as 'id ::: title ::: genre ::: plot'.
            if "movie_id" not in df.columns and "title" not in df.columns and "genre" in df.columns and "plot" in df.columns:
                return df

        except Exception:
            continue

    raise ValueError(
        "Could not identify a compatible training dataset. "
        "The downloaded Kaggle files may have an unexpected format."
    )


def load_dataset(dataset_name: str = DEFAULT_DATASET) -> Tuple[Path, pd.DataFrame, List[Path]]:
    """Download the dataset, discover files, and load the correct training data."""
    dataset_path = download_dataset(dataset_name)
    files = find_dataset_files(dataset_path)
    df = auto_detect_training_data(dataset_path)
    return dataset_path, df, files
