"""Metric computation for model comparison."""

from __future__ import annotations

import numpy as np


def _as_1d_float_array(values):
    """Normalize numeric inputs to a flat float array."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.reshape(-1)


def mse_metric(actual, prediction):
    """Compute mean squared error across matched numeric arrays."""
    actual_arr = _as_1d_float_array(actual)
    prediction_arr = _as_1d_float_array(prediction)

    if actual_arr.shape[0] != prediction_arr.shape[0]:
        raise ValueError(f"MSE requires equal-length arrays; got {actual_arr.shape[0]} and {prediction_arr.shape[0]}.")

    return float(np.mean((actual_arr - prediction_arr) ** 2))


def mae_metric(actual, prediction):
    """Compute mean absolute error across matched numeric arrays."""
    actual_arr = _as_1d_float_array(actual)
    prediction_arr = _as_1d_float_array(prediction)

    if actual_arr.shape[0] != prediction_arr.shape[0]:
        raise ValueError(f"MAE requires equal-length arrays; got {actual_arr.shape[0]} and {prediction_arr.shape[0]}.")

    return float(np.mean(np.abs(actual_arr - prediction_arr)))
