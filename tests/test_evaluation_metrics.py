import numpy as np

from evaluation.metrics import mae_metric, mse_metric


def test_mse_metric_values():
    actual = np.array([0.0, 2.0])
    prediction = np.array([1.0, 2.0])
    assert mse_metric(actual, prediction) == 0.5


def test_mae_metric_values():
    actual = np.array([0.0, 2.0])
    prediction = np.array([1.0, 2.0])
    assert mae_metric(actual, prediction) == 0.5


def test_metric_length_mismatch_raises():
    try:
        mse_metric([0.0, 1.0], [1.0])
        raise AssertionError("Expected ValueError for mismatched lengths")
    except ValueError:
        pass
