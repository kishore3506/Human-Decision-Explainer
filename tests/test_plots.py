from pathlib import Path

import pandas as pd

from evaluation.plots import generate_plots


def test_generate_plots_creates_png_outputs(tmp_path):
    metrics_path = tmp_path / "model_comparison.csv"
    pd.DataFrame(
        {
            "model": ["rl", "sft"],
            "mse_total": [0.05, 0.02],
            "mae_total": [0.2, 0.1],
        }
    ).to_csv(metrics_path, index=False)

    plots_dir = tmp_path / "plots"
    generate_plots(metrics_path=str(metrics_path), output_dir=str(plots_dir))

    assert (plots_dir / "mse_comparison.png").exists()
    assert (plots_dir / "mae_comparison.png").exists()
