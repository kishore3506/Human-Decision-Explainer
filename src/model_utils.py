"""Utility helpers for model loading and configuration."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.config import DEVICE, MODEL_NAME, USE_DEVELOPMENT_MODEL, DEVELOPMENT_MODEL_NAME


def get_model_name() -> str:
    """Return the user-selected model name without silently replacing it."""
    if USE_DEVELOPMENT_MODEL:
        return DEVELOPMENT_MODEL_NAME
    return MODEL_NAME


def get_device() -> str:
    """Return the active device and print a clear warning if CUDA is unavailable."""
    if torch.cuda.is_available():
        print(f"CUDA detected. Device: {torch.cuda.get_device_name(0)}")
        return "cuda"

    print("No CUDA GPU detected. Falling back to CPU. Training may be slow or unsupported for large models.")
    return "cpu"


def summarize_trainable_parameters(model) -> tuple[int, int, float]:
    """Return total, trainable, and percentage parameters for a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_pct = (trainable_params / total_params * 100.0) if total_params else 0.0
    return total_params, trainable_params, trainable_pct
