from pathlib import Path

import pandas as pd

from evaluation.compare_models import compare_models


def test_compare_models_builds_summary(tmp_path):
    data_path = tmp_path / "test.csv"
    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir()

    pd.DataFrame(
        {
            "problem_id": [1, 2],
            "actual_prob_a": [0.0, 0.5],
            "actual_prob_b": [1.0, 0.5],
        }
    ).to_csv(data_path, index=False)

    pd.DataFrame(
        {
            "problem_id": [1, 2],
            "predicted_prob_a": [0.1, 0.6],
            "predicted_prob_b": [0.9, 0.4],
        }
    ).to_csv(predictions_dir / "sft_predictions.csv", index=False)

    pd.DataFrame(
        {
            "problem_id": [1, 2],
            "predicted_prob_a": [0.2, 0.4],
            "predicted_prob_b": [0.8, 0.6],
        }
    ).to_csv(predictions_dir / "rl_predictions.csv", index=False)

    summary = compare_models(data_path=data_path, predictions_dir=predictions_dir)

    assert {"sft", "rl"}.issubset(set(summary["model"]))
    assert list(summary["model"]) == sorted(summary["model"], key=lambda x: x.lower())
    assert (tmp_path / "comparison_summary.csv").exists() is False
