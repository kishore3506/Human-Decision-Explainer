"""Dataset discovery and schema inspection for the human decision project.

This phase is intentionally lightweight: it locates the raw behavioral dataset,
inspects its columns and sample rows, and validates expected fields before the
preprocessing pipeline is implemented.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

try:
    from datasets import Dataset
except ImportError:  # pragma: no cover - optional dependency fallback
    Dataset = None

from src.config import RAW_DATA_DIR, SEED


def normalize_column_name(name: str) -> str:
    """Normalize column names so the loader can match likely behavioral fields."""
    normalized = str(name).strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = normalized.strip("_")
    return normalized


def infer_behavioral_columns(df: pd.DataFrame) -> dict[str, str]:
    """Map flexible dataset schema names to a consistent internal representation."""
    normalized = {normalize_column_name(col): col for col in df.columns}
    aliases = {
        "option_a": ["option_a", "optiona", "a_option", "choice_a", "option_1", "option_1_text"],
        "option_b": ["option_b", "optionb", "b_option", "choice_b", "option_2", "option_2_text"],
        "human_choice": ["human_choice", "chosen_option", "selected_option", "choice", "response"],
        "actual_choice_probabilities": [
            "actual_choice_probabilities",
            "choice_distribution",
            "human_distribution",
            "probabilities",
            "choice_probabilities",
        ],
        "decision_text": ["decision_text", "scenario", "problem", "question", "task", "prompt"],
    }

    mapping: dict[str, str] = {}
    for canonical, candidates in aliases.items():
        for candidate in candidates:
            if candidate in normalized:
                mapping[canonical] = normalized[candidate]
                break
    return mapping


def find_dataset_path(raw_dir: Path | str | None = None) -> Path | None:
    """Locate the raw dataset in the data/raw directory.

    The project expects the dataset to be placed in data/raw/ by the user. If no
    compatible dataset is found, this function returns None instead of creating a
    fake dataset. For Choices13K, the main table is the CSV while the JSON file is
    a metadata companion that stores the gamble definitions.
    """
    raw_dir = Path(raw_dir) if raw_dir is not None else RAW_DATA_DIR
    candidates = [
        raw_dir,
        raw_dir / "choices13k",
        raw_dir / "Choices13K",
        raw_dir / "choices13k-main",
    ]

    for path in candidates:
        if path.exists():
            if path.is_dir():
                csv_files = sorted(path.glob("*.csv"), key=lambda p: p.name.lower())
                if csv_files:
                    return csv_files[0]
                for item in sorted(path.iterdir()):
                    if item.is_file() and item.suffix.lower() in {".parquet", ".jsonl", ".json"}:
                        return item
            elif path.is_file():
                return path

    return None


def load_dataset(dataset_path: str | Path | None = None) -> pd.DataFrame:
    """Load the dataset with pandas and optionally convert it to a Hugging Face dataset."""
    dataset_path = Path(dataset_path) if dataset_path is not None else find_dataset_path()
    if dataset_path is None:
        raise FileNotFoundError(
            "Choices13K dataset not found. Place the raw dataset under data/raw/ "
            "before running preprocessing."
        )

    file_path = Path(dataset_path)
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(file_path)
    elif suffix == ".parquet":
        df = pd.read_parquet(file_path)
    elif suffix in {".json", ".jsonl"}:
        # The project uses the CSV table as the primary dataset; if the JSON file is
        # selected instead, we still load it to inspect schema but warn the caller.
        df = pd.read_json(file_path, lines=suffix == ".jsonl")
    else:
        raise ValueError(f"Unsupported dataset format: {suffix}. Use CSV, JSON, JSONL, or Parquet.")

    if Dataset is not None:
        try:
            _ = Dataset.from_pandas(df)
        except Exception:
            pass

    return df


def inspect_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Print basic dataset diagnostics and return a summary dictionary."""
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("Sample records:")
    print(df.head(3).to_string(index=False))
    print(f"Missing values:\n{df.isna().sum()}")
    duplicate_rows = int(df.astype(str).duplicated().sum())
    print(f"Duplicate rows: {duplicate_rows}")

    stats = {
        "shape": df.shape,
        "columns": list(df.columns),
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": duplicate_rows,
    }
    return stats


def validate_behavioral_fields(df: pd.DataFrame) -> list[str]:
    """Check the dataset for expected human-choice behavioral fields.

    This function intentionally adapts to the real Choices13K schema instead of
    assuming the final LLM prompt schema.
    """
    normalized = {normalize_column_name(col): col for col in df.columns}
    actual_expected = [
        "problem",
        "feedback",
        "n",
        "block",
        "brate",
        "ha",
        "pha",
        "la",
        "hb",
        "phb",
        "lb",
        "lotshapeb",
        "lotnumb",
        "amb",
        "corr",
    ]
    available = [field for field in actual_expected if field in normalized]
    missing = [field for field in actual_expected if field not in normalized]

    if missing:
        print("Missing expected Choices13K behavioral fields:", missing)
    else:
        print("Expected Choices13K behavioral fields found.")

    print("Detected schema mapping:")
    for field in actual_expected:
        if field in normalized:
            print(f"  {field} -> {normalized[field]}")

    if "brate" in normalized:
        print("The dataset's human-choice measure is `bRate`, representing the share of subjects selecting Gamble B.")

    return missing


if __name__ == "__main__":
    dataset_path = find_dataset_path()
    if dataset_path is None:
        print("No raw dataset found in data/raw/. Place the Choices13K dataset there before preprocessing.")
        print("Expected location: data/raw/")
    else:
        df = load_dataset(dataset_path)
        inspect_dataframe(df)
        validate_behavioral_fields(df)
