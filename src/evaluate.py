"""Evaluation entry point for SFT and RL models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import mae_metric, mse_metric
from src.config import PROCESSED_DATA_DIR, RESULTS_DIR
from src.inference import parse_prediction_text


def _read_prediction_file(path: Path) -> pd.DataFrame:
    """Read a prediction artifact from CSV or JSON into a DataFrame."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict):
            return pd.DataFrame([payload])
        return pd.DataFrame()
    return pd.DataFrame()


def _extract_probabilities_from_row(row: pd.Series):
    """Normalize the predicted distribution from a record into Option A/B probabilities."""
    for a_col, b_col in (
        ("predicted_prob_a", "predicted_prob_b"),
        ("prob_a", "prob_b"),
        ("option_a_prob", "option_b_prob"),
        ("pred_a", "pred_b"),
    ):
        if a_col in row.index and b_col in row.index:
            return float(row[a_col]), float(row[b_col])

    if "prediction" in row.index and isinstance(row["prediction"], str):
        pred_a, pred_b = parse_prediction_text(row["prediction"])
        return pred_a, pred_b

    if "prediction_text" in row.index and isinstance(row["prediction_text"], str):
        pred_a, pred_b = parse_prediction_text(row["prediction_text"])
        return pred_a, pred_b

    if "response" in row.index and isinstance(row["response"], str):
        pred_a, pred_b = parse_prediction_text(row["response"])
        return pred_a, pred_b

    if "Option A" in row.index and "Option B" in row.index:
        return float(row["Option A"]), float(row["Option B"])

    raise ValueError(f"Cannot extract probability columns from prediction row: {row.to_dict()}")


def _normalize_prediction_frame(frame: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Standardize prediction rows to a common probability schema."""
    if frame.empty:
        raise ValueError(f"Prediction file for {model_name} is empty.")

    standardized = frame.copy()
    if "problem_id" not in standardized.columns:
        standardized["problem_id"] = list(range(len(standardized)))

    if not {"predicted_prob_a", "predicted_prob_b"}.issubset(standardized.columns):
        prob_rows = []
        for _, row in standardized.iterrows():
            pred_a, pred_b = _extract_probabilities_from_row(row)
            prob_rows.append({"problem_id": row.get("problem_id", len(prob_rows)), "predicted_prob_a": pred_a, "predicted_prob_b": pred_b})
        standardized = pd.DataFrame(prob_rows)

    return standardized[["problem_id", "predicted_prob_a", "predicted_prob_b"]].copy()


def _evaluate_prediction_file(actual_df: pd.DataFrame, prediction_df: pd.DataFrame, model_name: str):
    """Compute MSE and MAE for one model against the actual test probabilities."""
    actual = actual_df[["problem_id", "actual_prob_a", "actual_prob_b"]].copy()
    pred = _normalize_prediction_frame(prediction_df, model_name)

    merged = actual.merge(pred, on="problem_id", how="inner")
    if merged.empty:
        raise ValueError(f"No overlapping problem_ids found between the test set and the {model_name} predictions.")

    actual_a = merged["actual_prob_a"].to_numpy(dtype=float)
    actual_b = merged["actual_prob_b"].to_numpy(dtype=float)
    pred_a = merged["predicted_prob_a"].to_numpy(dtype=float)
    pred_b = merged["predicted_prob_b"].to_numpy(dtype=float)

    return {
        "model": model_name,
        "n_samples": len(merged),
        "mse_a": mse_metric(actual_a, pred_a),
        "mse_b": mse_metric(actual_b, pred_b),
        "mae_a": mae_metric(actual_a, pred_a),
        "mae_b": mae_metric(actual_b, pred_b),
        "mse_total": mse_metric(np.concatenate([actual_a, actual_b]), np.concatenate([pred_a, pred_b])),
        "mae_total": mae_metric(np.concatenate([actual_a, actual_b]), np.concatenate([pred_a, pred_b])),
    }


def evaluate_models(data_path: str | Path | None = None, predictions_dir: str | Path | None = None):
    """Compare model predictions against the processed test set and save a metric summary."""
    dataset_path = Path(data_path) if data_path is not None else PROCESSED_DATA_DIR / "test.csv"
    predictions_path = Path(predictions_dir) if predictions_dir is not None else RESULTS_DIR / "predictions"

    actual_df = pd.read_csv(dataset_path)
    if not {"problem_id", "actual_prob_a", "actual_prob_b"}.issubset(actual_df.columns):
        raise ValueError(f"Expected probability columns in {dataset_path}, but found: {list(actual_df.columns)}")

    results_dir = RESULTS_DIR / "metrics"
    results_dir.mkdir(parents=True, exist_ok=True)

    candidate_files = sorted(predictions_path.glob("*.csv")) + sorted(predictions_path.glob("*.json"))
    if not candidate_files:
        print(f"No prediction files found in {predictions_path}. Skipping evaluation.")
        return []

    summaries = []
    for prediction_file in candidate_files:
        model_name = prediction_file.stem.replace("_predictions", "").replace("_prediction", "")
        try:
            prediction_df = _read_prediction_file(prediction_file)
            result = _evaluate_prediction_file(actual_df, prediction_df, model_name)
            summaries.append(result)
        except Exception as exc:  # pragma: no cover - protects the CLI from a single bad artifact
            print(f"Skipping {prediction_file.name}: {exc}")

    if not summaries:
        print(f"No valid prediction files were usable for evaluation in {predictions_path}.")
        return []

    summary_df = pd.DataFrame(summaries)
    summary_path = results_dir / "model_metrics.csv"
    summary_df.to_csv(summary_path, index=False)

    print(summary_df.to_string(index=False))
    print(f"Saved metrics to: {summary_path}")
    return summaries


if __name__ == "__main__":
    evaluate_models()
