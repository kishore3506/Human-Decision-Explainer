"""Inference utilities for the human decision model.

This module is intentionally written to support both development work and the
full Qwen2.5-based research setup. It does not start fine-tuning; it only loads
an LLM, generates a response to a risky-choice prompt, and parses the resulting
probabilities safely.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.config import MODEL_NAME, USE_DEVELOPMENT_MODEL, DEVELOPMENT_MODEL_NAME, DEVICE


def get_inference_model_name() -> str:
    """Return the configured model name without silently replacing it."""
    if USE_DEVELOPMENT_MODEL:
        return DEVELOPMENT_MODEL_NAME
    return MODEL_NAME


def build_quantization_config(load_in_4bit: bool = False, load_in_8bit: bool = False):
    """Return a BitsAndBytes quantization config when requested."""
    if load_in_4bit and load_in_8bit:
        raise ValueError("Only one of load_in_4bit or load_in_8bit can be enabled at a time.")

    if not load_in_4bit and not load_in_8bit:
        return None

    if not torch.cuda.is_available():
        raise RuntimeError("Quantized loading requires CUDA. Use a GPU-enabled environment or disable 4-bit/8-bit loading.")

    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )


def load_model_and_tokenizer(
    model_name: str | None = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    local_files_only: bool = False,
):
    """Load a Hugging Face causal LM and tokenizer for inference."""
    model_name = model_name or get_inference_model_name()
    device = DEVICE

    if device != "cuda" and (load_in_4bit or load_in_8bit):
        raise RuntimeError("Quantized loading requires CUDA; the current device is CPU.")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=False,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = build_quantization_config(
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
    )

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": False,
        "local_files_only": local_files_only,
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    elif device == "cuda":
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()
    if device == "cuda" and quantization_config is None:
        model.to(device)

    return model, tokenizer


def generate_response(
    prompt: str,
    model=None,
    tokenizer=None,
    model_name: str | None = None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    do_sample: bool = True,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    local_files_only: bool = False,
) -> str:
    """Generate a model response for a prompt and return the decoded text."""
    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer(
            model_name=model_name,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            local_files_only=local_files_only,
        )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    inputs = {k: v.to(model.device if hasattr(model, "device") else DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return decoded.strip()


def _find_probability_values(text: str, label: str) -> list[float]:
    """Find numeric probability values associated with an option label."""
    pattern = rf"{label}\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%?"
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    if matches:
        return [float(value) for value in matches]

    # Handle forms like "Option A = 0.6125" and "A: 61.25%" in a more forgiving way.
    alt_pattern = rf"(?:Option\s*|Option\s*{label.upper()}\s*|{label.upper()}\s*[:=]\s*)(\d+(?:\.\d+)?)\s*%?"
    alt_matches = re.findall(alt_pattern, text, flags=re.IGNORECASE)
    if alt_matches:
        return [float(value) for value in alt_matches]

    return []


def parse_prediction_text(text: str) -> tuple[float, float]:
    """Parse a model response into Option A and Option B probabilities.

    Accepted shapes include:
      - Option A: 61.25%
      - Option B: 38.75%
      - PREDICTION: A=0.6125, B=0.3875
    Invalid or malformed outputs return safe defaults instead of crashing.
    """
    if text is None:
        return 50.0, 50.0

    cleaned = text.strip()
    if not cleaned:
        return 50.0, 50.0

    a_values = _find_probability_values(cleaned, "A")
    b_values = _find_probability_values(cleaned, "B")

    if not a_values and not b_values:
        # Handle output like "A 0.61, B 0.39" without explicit labels.
        nums = re.findall(r"(\d+(?:\.\d+)?)\s*%?", cleaned)
        if len(nums) >= 2:
            a_values = [float(nums[0])]
            b_values = [float(nums[1])]

    if not a_values and b_values:
        a_values = [100.0 - float(b_values[0])]
    if not b_values and a_values:
        b_values = [100.0 - float(a_values[0])]

    if not a_values or not b_values:
        return 50.0, 50.0

    prob_a = float(a_values[0])
    prob_b = float(b_values[0])

    if prob_a > 1.0 and prob_b > 1.0:
        total = prob_a + prob_b
        if total > 0:
            prob_a = prob_a / total * 100.0
            prob_b = prob_b / total * 100.0
    elif prob_a <= 1.0 and prob_b <= 1.0:
        prob_a *= 100.0
        prob_b *= 100.0

    if prob_a < 0.0:
        prob_a = 0.0
    if prob_b < 0.0:
        prob_b = 0.0

    total = prob_a + prob_b
    if total <= 0.0:
        return 50.0, 50.0

    if abs(total - 100.0) > 1e-6:
        prob_a = prob_a / total * 100.0
        prob_b = prob_b / total * 100.0

    return float(prob_a), float(prob_b)


if __name__ == "__main__":
    sample = """REASONING:
The expected utility is higher for Option A because it has a better chance of a large gain.

PREDICTION:
Option A: 61.25%
Option B: 38.75%"""
    print(parse_prediction_text(sample))
