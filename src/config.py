"""Core configuration for the human decision modeling project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
SFT_MODEL_DIR = MODELS_DIR / "sft"
RL_MODEL_DIR = MODELS_DIR / "rl"
RESULTS_DIR = PROJECT_ROOT / "results"

SEED = 42

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEVELOPMENT_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
USE_DEVELOPMENT_MODEL = True

DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

# Training defaults
MAX_LENGTH = 1024
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
EPOCHS = 1
RL_STEPS = 1
SFT_MAX_SAMPLES = 256
SFT_DEV_MAX_SAMPLES = 32

# LoRA defaults
LORA_R = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Development mode overrides
DEV_BATCH_SIZE = 1
DEV_GRAD_ACCUM_STEPS = 2
DEV_EPOCHS = 1
DEV_RL_STEPS = 1
DEV_MAX_SAMPLES = 32

# Dataset expectations
EXPECTED_FIELDS = [
    "option_a",
    "option_b",
    "human_choice",
    "actual_choice_probabilities",
]
