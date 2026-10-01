"""Behavioral reward implementation for decision prediction.

The reward is intentionally transparent and based on prediction error. It is not an
arbitrary heuristic beyond the documented MSE-based formulation.
"""

from __future__ import annotations


def behavioral_reward(actual_a: float, actual_b: float, pred_a: float, pred_b: float) -> float:
    """Return the negative mean squared error between actual and predicted choice probabilities.

    The reward is defined as:
        mse = ((pred_a - actual_a)^2 + (pred_b - actual_b)^2) / 2
        reward = -mse

    This keeps the optimization signal aligned with behavioral fit quality.
    """
    mse = ((pred_a - actual_a) ** 2 + (pred_b - actual_b) ** 2) / 2.0
    return -mse


if __name__ == "__main__":
    reward = behavioral_reward(0.65, 0.35, 0.63, 0.37)
    print(f"Example reward: {reward}")
