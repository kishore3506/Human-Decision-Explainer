"""Comparison utilities for SFT and RL model outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import RESULTS_DIR
from src.evaluate import evaluate_models


def compare_models(data_path: str | Path | None = None, predictions_dir: str | Path | None = None, output_path: str | Path | None = None):
    """Compare the SFT and RL-GRPO results after evaluation and save a summary table."""
    metrics = evaluate_models(data_path=data_path, predictions_dir=predictions_dir)
    if not metrics:
        return pd.DataFrame(columns=["model", "n_samples", "mse_a", "mse_b", "mae_a", "mae_b", "mse_total", "mae_total"])

    comparison_df = pd.DataFrame(metrics)
    comparison_df = comparison_df.sort_values(["model"], ascending=[True]).reset_index(drop=True)

    output_file = Path(output_path) if output_path is not None else RESULTS_DIR / "metrics" / "model_comparison.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_file, index=False)
    return comparison_df
