"""Visualization utilities for evaluation results."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import RESULTS_DIR


def generate_plots(metrics_path: str | Path | None = None, output_dir: str | Path | None = None):
    """Generate bar charts for SFT vs RL comparison and reward progression."""
    metrics_file = Path(metrics_path) if metrics_path is not None else RESULTS_DIR / "metrics" / "model_comparison.csv"
    output_dir_path = Path(output_dir) if output_dir is not None else RESULTS_DIR / "plots"
    output_dir_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metrics_file)
    if df.empty:
        raise ValueError(f"No comparison data found in {metrics_file}.")

    if "model" not in df.columns:
        raise ValueError(f"The metrics file must contain a 'model' column: {metrics_file}")

    for metric_name, file_name in [("mse_total", "mse_comparison.png"), ("mae_total", "mae_comparison.png")]:
        if metric_name not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(df["model"], df[metric_name], color=["#4c72b0", "#dd8452"][: len(df)])
        ax.set_title(f"{metric_name.replace('_', ' ').upper()} by model")
        ax.set_xlabel("Model")
        ax.set_ylabel(metric_name.replace("_", " ").upper())
        ax.bar_label(bars, fmt="%.4f")
        plt.tight_layout()
        plt.savefig(output_dir_path / file_name, dpi=150)
        plt.close(fig)

    return [output_dir_path / "mse_comparison.png", output_dir_path / "mae_comparison.png"]
