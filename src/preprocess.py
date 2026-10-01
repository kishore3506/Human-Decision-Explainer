"""Data preprocessing pipeline for risky-choice decision modeling.

This phase converts the raw Choices13K risky-choice table and gamble metadata into a
standardized LLM-ready representation with human-choice probabilities, option text,
and train/test splits.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, SEED
from src.data_loader import find_dataset_path, load_dataset


def clean_text(text: str | None) -> str:
    """Normalize text while preserving meaning."""
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def safe_float(value, default: float = 0.0) -> float:
    """Convert a value to float safely."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def load_problem_metadata(raw_dir: Path | str | None = None) -> dict[str, dict]:
    """Load the Choices13K JSON metadata containing Gamble A/B definitions."""
    raw_dir = Path(raw_dir) if raw_dir is not None else RAW_DATA_DIR
    json_path = raw_dir / "c13k_problems.json"
    if not json_path.exists():
        return {}

    with open(json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {str(k): v for k, v in payload.items()}


def format_option_text(option_name: str, row: dict, problem_data: dict | None = None) -> str:
    """Format the gamble description into readable natural language."""
    if problem_data:
        option_values = problem_data.get(option_name, [])
        if isinstance(option_values, list) and option_values:
            pieces = []
            for outcome in option_values:
                if isinstance(outcome, (list, tuple)) and len(outcome) >= 2:
                    probability, value = outcome
                    pieces.append(f"{value:.2f} with probability {probability:.4f}")
            if pieces:
                return "; ".join(pieces)

    if option_name == "A":
        ha = safe_float(row.get("Ha", row.get("ha", 0.0)))
        pha = safe_float(row.get("pHa", row.get("pha", 0.0)))
        la = safe_float(row.get("La", row.get("la", 0.0)))
        return f"{ha:.2f} with probability {pha:.4f}; {la:.2f} with probability {1.0 - pha:.4f}"

    hb = safe_float(row.get("Hb", row.get("hb", 0.0)))
    phb = safe_float(row.get("pHb", row.get("phb", 0.0)))
    lb = safe_float(row.get("Lb", row.get("lb", 0.0)))
    return f"{hb:.2f} with probability {phb:.4f}; {lb:.2f} with probability {1.0 - phb:.4f}"


def build_decision_scenario(row: dict) -> str:
    """Create a concise description of the risky choice decision."""
    feedback = row.get("Feedback", row.get("feedback", "unknown"))
    block = row.get("Block", row.get("block", "unknown"))
    amb = row.get("Amb", row.get("amb", "unknown"))
    corr = row.get("Corr", row.get("corr", "unknown"))
    return (
        "Risky choice problem. "
        f"Feedback: {feedback}. Block: {block}. Ambiguity: {amb}. Correlation: {corr}. "
        "Choose the option that is most appealing to human decision-makers."
    )


def build_decision_prompt(row: dict, option_a: str, option_b: str) -> str:
    """Create the LLM instruction template for the risky-choice task."""
    scenario = clean_text(build_decision_scenario(row))
    option_a_text = clean_text(option_a)
    option_b_text = clean_text(option_b)

    return (
        "You are modeling human decision-making.\n\n"
        "Consider this risky-choice problem:\n\n"
        f"Scenario:\n{scenario}\n\n"
        f"Option A:\n{option_a_text}\n\n"
        f"Option B:\n{option_b_text}\n\n"
        "Predict the distribution of human choices.\n\n"
        "Provide:\n"
        "1. A brief explanation.\n"
        "2. Probability for Option A.\n"
        "3. Probability for Option B.\n"
    )


def extract_human_choice_distribution(row: dict) -> tuple[float, float, str]:
    """Map the raw human choice rate to a consistent probability distribution.

    Here, `bRate` denotes the share of participants choosing Gamble B in the
    original dataset. The final distribution is normalized to Option A and Option B.
    """
    b_rate = safe_float(row.get("bRate", row.get("brate", 0.0)))
    b_rate = max(0.0, min(1.0, b_rate))
    prob_a = 1.0 - b_rate
    prob_b = b_rate
    human_choice = "Option B" if b_rate >= 0.5 else "Option A"
    return prob_a, prob_b, human_choice


def format_decision_row(row: dict, problem_data: dict | None = None) -> dict:
    """Create a standardized record from a raw row with actual Choices13K fields."""
    scenario = build_decision_scenario(row)
    option_a = format_option_text("A", row, problem_data)
    option_b = format_option_text("B", row, problem_data)
    prob_a, prob_b, human_choice = extract_human_choice_distribution(row)

    formatted = {
        "problem_id": row.get("Problem", row.get("problem", "")),
        "scenario": scenario,
        "option_a": option_a,
        "option_b": option_b,
        "actual_prob_a": prob_a,
        "actual_prob_b": prob_b,
        "actual_choice_probabilities": {"Option A": prob_a, "Option B": prob_b},
        "human_choice": human_choice,
        "decision_text": scenario,
        "prompt": build_decision_prompt(row, option_a, option_b),
    }

    # Preserve both raw row fields and the processed representation.
    for key, value in row.items():
        if key not in formatted:
            formatted[key] = value

    return formatted


def split_data(df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create deterministic train/test splits with a fixed seed."""
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    split_idx = max(1, int(len(df) * (1 - test_size)))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def preprocess_dataset(raw_df: pd.DataFrame, metadata: dict | None = None) -> pd.DataFrame:
    """Apply cleaning, standardization, and prompt construction to the raw risky-choice data."""
    rows = []
    metadata = metadata or {}
    for index, row in raw_df.reset_index().iterrows():
        problem_data = metadata.get(str(index), {}) if isinstance(metadata, dict) else {}
        rows.append(format_decision_row(row.to_dict(), problem_data))
    return pd.DataFrame(rows)


def save_processed_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Write processed data to data/processed/."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    test_df.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)


if __name__ == "__main__":
    dataset_path = find_dataset_path()
    if dataset_path is None:
        raise FileNotFoundError(
            "No raw dataset found in data/raw/. Place the Choices13K CSV in data/raw/ before preprocessing."
        )

    raw_df = load_dataset(dataset_path)
    metadata = load_problem_metadata(dataset_path.parent)
    processed_df = preprocess_dataset(raw_df, metadata)
    train_df, test_df = split_data(processed_df, test_size=0.2)
    save_processed_data(train_df, test_df)

    print(f"Total samples: {len(processed_df)}")
    print(f"Training samples: {len(train_df)}")
    print(f"Testing samples: {len(test_df)}")
    print(f"Processed files saved to: {PROCESSED_DATA_DIR}")
