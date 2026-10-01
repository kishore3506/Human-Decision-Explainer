"""LoRA-based Supervised Fine-Tuning for the risky-choice decision model.

This script is designed to train against the processed Choices13K data with a
small, configurable dataset by default. Training is not launched automatically;
use the explicit --train flag when you want to run the fine-tuning loop.
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
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

from src.config import (
    BATCH_SIZE,
    DEV_BATCH_SIZE,
    DEV_EPOCHS,
    DEV_MAX_SAMPLES,
    EPOCHS,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    MAX_LENGTH,
    MODEL_NAME,
    PROCESSED_DATA_DIR,
    SEED,
    SFT_DEV_MAX_SAMPLES,
    SFT_MAX_SAMPLES,
    USE_DEVELOPMENT_MODEL,
    DEVELOPMENT_MODEL_NAME,
)
from src.model_utils import get_model_name, summarize_trainable_parameters


def build_target_response(actual_prob_a: float, actual_prob_b: float) -> str:
    """Create a structured supervised target aligned with the project output format."""
    a_pct = float(actual_prob_a) * 100.0
    b_pct = float(actual_prob_b) * 100.0
    return (
        "REASONING:\n"
        "The observed human decision data suggests that the distribution should remain close to the empirical choice rate for this risky choice problem.\n\n"
        "PREDICTION:\n"
        f"Option A: {a_pct:.2f}%\n"
        f"Option B: {b_pct:.2f}%\n"
    )


def load_training_dataframe(data_dir: str | Path | None = None, max_samples: int | None = None) -> pd.DataFrame:
    """Load the processed training CSV and keep only the columns needed for SFT."""
    data_path = Path(data_dir) if data_dir is not None else PROCESSED_DATA_DIR / "train.csv"
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["prompt", "actual_prob_a", "actual_prob_b"]).copy()
    if max_samples is not None:
        df = df.head(max_samples)
    return df.reset_index(drop=True)


def create_sft_dataset(df: pd.DataFrame, tokenizer, max_length: int = MAX_LENGTH):
    """Turn prompt/target pairs into tokenized training examples."""
    records = []

    for row in df.to_dict(orient="records"):
        prompt = str(row["prompt"])
        target = build_target_response(float(row["actual_prob_a"]), float(row["actual_prob_b"]))
        full_text = f"{prompt}\n\n{target}"

        full_encoding = tokenizer(full_text, truncation=True, max_length=max_length, add_special_tokens=False)
        prompt_encoding = tokenizer(prompt, truncation=True, max_length=max_length, add_special_tokens=False)

        full_ids = full_encoding["input_ids"]
        prompt_ids = prompt_encoding["input_ids"]
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        labels = [-100] * len(full_ids)
        if len(full_ids) > len(prompt_ids):
            labels[len(prompt_ids):] = full_ids[len(prompt_ids):]

        attention_mask = full_encoding.get("attention_mask", [1] * len(full_ids))
        if len(attention_mask) < len(full_ids):
            attention_mask.extend([0] * (len(full_ids) - len(attention_mask)))
        if len(full_ids) < max_length:
            pad_len = max_length - len(full_ids)
            full_ids = full_ids + [pad_token_id] * pad_len
            labels = labels + [-100] * pad_len
            attention_mask = attention_mask + [0] * pad_len
        else:
            full_ids = full_ids[:max_length]
            labels = labels[:max_length]
            attention_mask = attention_mask[:max_length]

        records.append({
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        })

    return Dataset.from_list(records)


def apply_lora(model, target_modules: list[str] | None = None):
    """Attach LoRA adapters to the base model."""
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules or LORA_TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    return model


def train_sft(
    model_name: str | None = None,
    train: bool = False,
    max_samples: int | None = None,
    batch_size: int | None = None,
    epochs: int | None = None,
    output_dir: str | Path | None = None,
):
    """Train a LoRA-adapted baseline or print configuration in dry-run mode."""
    model_name = model_name or get_model_name()
    effective_batch = batch_size or (DEV_BATCH_SIZE if USE_DEVELOPMENT_MODEL else BATCH_SIZE)
    effective_epochs = epochs or (DEV_EPOCHS if USE_DEVELOPMENT_MODEL else EPOCHS)
    effective_max_samples = max_samples or (SFT_DEV_MAX_SAMPLES if USE_DEVELOPMENT_MODEL else SFT_MAX_SAMPLES)
    output_dir = Path(output_dir) if output_dir is not None else Path("models/sft")

    if not train:
        print("SFT dry run only. No training was started.")
        print(f"Model: {model_name}")
        print(f"Mode: {'development' if USE_DEVELOPMENT_MODEL else 'research'}")
        print(f"Batch size: {effective_batch}")
        print(f"Epochs: {effective_epochs}")
        print(f"Max samples for SFT: {effective_max_samples}")
        print(f"LoRA config: r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
        return None

    if not torch.cuda.is_available() and model_name.startswith("Qwen"):
        print("Warning: Qwen2.5-7B-Instruct is large and may be slow or unsupported on CPU. Consider setting USE_DEVELOPMENT_MODEL = True for a smaller debug run.")

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
    total_params, trainable_params, trainable_pct = summarize_trainable_parameters(model)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Trainable percentage: {trainable_pct:.2f}%")

    train_df = load_training_dataframe(max_samples=effective_max_samples)
    dataset = create_sft_dataset(train_df, tokenizer, max_length=MAX_LENGTH)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=effective_batch,
        gradient_accumulation_steps=4,
        num_train_epochs=effective_epochs,
        learning_rate=2e-4,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"LoRA adapter saved to: {output_dir}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA SFT baseline for human decision modeling")
    parser.add_argument("--train", action="store_true", help="Launch the actual SFT training loop")
    parser.add_argument("--model-name", type=str, default=None, help="Override the default model name")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit the number of training rows for a quick run")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="models/sft")
    args = parser.parse_args()

    train_sft(
        model_name=args.model_name,
        train=args.train,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        output_dir=args.output_dir,
    )
