"""GRPO reinforcement learning for the human decision model.

This phase wires the reward signal into a real TRL GRPO loop. It keeps the
training configuration small and explicit, while still using the project's
actual prompt dataset and the behavioral MSE reward defined in src/reward.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from src.config import (
    BATCH_SIZE,
    DEV_BATCH_SIZE,
    DEV_RL_STEPS,
    EPOCHS,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    MODEL_NAME,
    PROCESSED_DATA_DIR,
    RL_STEPS,
    SEED,
    USE_DEVELOPMENT_MODEL,
    DEVELOPMENT_MODEL_NAME,
)
from src.inference import parse_prediction_text
from src.model_utils import get_model_name
from src.reward import behavioral_reward


def load_rl_dataset(data_dir: str | Path | None = None, max_samples: int | None = None) -> Dataset:
    """Load processed prompts and targets for GRPO optimization."""
    data_path = Path(data_dir) if data_dir is not None else PROCESSED_DATA_DIR / "train.csv"
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["prompt", "actual_prob_a", "actual_prob_b"]).copy()
    if max_samples is not None:
        df = df.head(max_samples)

    return Dataset.from_pandas(
        df[["prompt", "actual_prob_a", "actual_prob_b"]].reset_index(drop=True),
    )


def build_behavioral_reward(actual_lookup: dict[str, tuple[float, float]]):
    """Create a reward callable that compares parsed completion probabilities to the true empirical distribution."""

    def _reward_func(prompts: list[str], completions: list[str], **kwargs):
        rewards: list[float] = []
        for prompt, completion in zip(prompts, completions):
            actual = actual_lookup.get(prompt)
            if actual is None:
                rewards.append(-1.0)
                continue

            actual_a, actual_b = actual
            pred_a, pred_b = parse_prediction_text(completion)
            pred_a_norm = pred_a / 100.0 if pred_a > 1.0 else pred_a
            pred_b_norm = pred_b / 100.0 if pred_b > 1.0 else pred_b
            rewards.append(behavioral_reward(actual_a, actual_b, pred_a_norm, pred_b_norm))

        return rewards

    return _reward_func


def apply_lora(model):
    """Attach LoRA adapters to the GRPO model."""
    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )
    return get_peft_model(model, config)


def train_rl(
    model_name: str | None = None,
    train: bool = False,
    steps: int | None = None,
    batch_size: int | None = None,
    epochs: int | None = None,
    output_dir: str | Path | None = None,
    max_samples: int | None = None,
):
    """Dry-run the RL-GRPO configuration or run a small GRPO training step."""
    model_name = model_name or get_model_name()
    effective_steps = steps or (DEV_RL_STEPS if USE_DEVELOPMENT_MODEL else RL_STEPS)
    effective_batch = batch_size or (DEV_BATCH_SIZE if USE_DEVELOPMENT_MODEL else BATCH_SIZE)
    effective_epochs = epochs or EPOCHS
    output_dir = Path(output_dir) if output_dir is not None else Path("models/rl")
    effective_max_samples = max_samples or (32 if USE_DEVELOPMENT_MODEL else 128)

    if not train:
        print("GRPO dry run only. No RL training was started.")
        print(f"Model: {model_name}")
        print(f"Mode: {'development' if USE_DEVELOPMENT_MODEL else 'research'}")
        print(f"Reward: behavioral_reward(actual_a, actual_b, pred_a, pred_b) = -MSE")
        print(f"Batch size: {effective_batch}")
        print(f"Epochs: {effective_epochs}")
        print(f"RL steps: {effective_steps}")
        print(f"LoRA config: r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
        print(f"Seed: {SEED}")
        print(f"Output directory: {output_dir}")
        print(f"Max GRPO samples: {effective_max_samples}")
        return None

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=False,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if torch.cuda.is_available():
        base_model.to("cuda")

    model = apply_lora(base_model)
    dataset = load_rl_dataset(max_samples=effective_max_samples)
    target_lookup = {
        str(row["prompt"]): (float(row["actual_prob_a"]), float(row["actual_prob_b"]))
        for row in dataset.to_pandas().to_dict(orient="records")
    }
    reward_fn = build_behavioral_reward(target_lookup)

    args = GRPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=effective_batch,
        gradient_accumulation_steps=4,
        num_train_epochs=effective_epochs,
        max_steps=effective_steps,
        learning_rate=1e-5,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        report_to=[],
        seed=SEED,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"GRPO adapter saved to: {output_dir}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Behavioral GRPO training for human decision modeling")
    parser.add_argument("--train", action="store_true", help="Launch the actual RL training loop")
    parser.add_argument("--model-name", type=str, default=None, help="Override the base model name")
    parser.add_argument("--steps", type=int, default=None, help="Number of GRPO steps")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="models/rl")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit RL samples for a quick dry run")
    args = parser.parse_args()

    train_rl(
        model_name=args.model_name,
        train=args.train,
        steps=args.steps,
        batch_size=args.batch_size,
        epochs=args.epochs,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
    )
