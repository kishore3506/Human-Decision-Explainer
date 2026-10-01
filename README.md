# Human Decision Explainer

## Overview

This project builds a research prototype for modeling human risky-choice decisions with a large language model. The system combines:

- raw behavioral decision data from Choices13K
- prompt construction for risky-choice reasoning
- LoRA-based supervised fine-tuning
- behavioral reward modeling grounded in probability error
- a GRPO-based reinforcement learning scaffold
- evaluation against human choice distributions
- comparison plots and a Streamlit dashboard

The design is intended to study whether an LLM can approximate human choice distributions while also producing explanations that align with the observed decision behavior.

## Current status

The project is implemented as a working research pipeline with the following verified phases:

1. Project scaffold and environment setup
2. Dataset loading and validation
3. Data preprocessing and train/test split
4. Inference and probability parsing
5. LoRA SFT baseline scaffolding and dry-run validation
6. Evaluation metrics for probability comparison
7. RL/GRPO scaffold with behavioral reward logic
8. GRPO trainer integration using TRL
9. Model comparison summary generation
10. Plot generation for MSE/MAE comparisons
11. Streamlit demo application
12. Documentation and project wrap-up

## Dataset requirements

Place the raw dataset under:

- data/raw/

Expected files include the Choices13K CSV and associated problem metadata. The pipeline will validate the dataset structure before preprocessing.

## Environment setup

Use Python 3.11 with the project requirements installed:

```powershell
cd "c:\Users\kish3\Desktop\AML project\human-decision-rl"
py -3.11 -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
```

The standard requirements are for CPU-compatible inference and deployment. For optional 4-bit/8-bit CUDA quantization, install `requirements-gpu.txt` after the standard requirements in a supported GPU environment.

## Quick start

### 1) Preprocess the data

```powershell
python src/preprocess.py
```

This creates the processed train/test CSVs in `data/processed/`.

### 2) Run the SFT baseline in dry-run mode

```powershell
python src/train_sft.py
```

To launch a real training run:

```powershell
python src/train_sft.py --train --max-samples 256
```

### 3) Run the GRPO/RL scaffold

```powershell
python src/train_rl.py
```

To launch the real GRPO path when implemented for a specific run:

```powershell
python src/train_rl.py --train --steps 1 --max-samples 128
```

### 4) Evaluate model outputs

```powershell
python src/evaluate.py
```

This reads prediction files in `results/predictions/` and writes summary metrics into `results/metrics/`.

### 5) Compare model outputs

```powershell
python -c "from evaluation.compare_models import compare_models; compare_models()"
```

### 6) Generate comparison plots

```powershell
python -c "from evaluation.plots import generate_plots; generate_plots()"
```

### 7) Launch the Streamlit app

```powershell
streamlit run app/app.py
```

## Project structure

```text
human-decision-rl/
├── app/
│   └── app.py
├── data/
│   ├── processed/
│   └── raw/
├── evaluation/
│   ├── compare_models.py
│   ├── metrics.py
│   └── plots.py
├── models/
│   ├── rl/
│   └── sft/
├── results/
│   ├── metrics/
│   ├── plots/
│   └── predictions/
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── model_utils.py
│   ├── preprocess.py
│   ├── reward.py
│   ├── train_rl.py
│   ├── train_sft.py
│   └── __init__.py
├── tests/
│   ├── test_compare_models.py
│   ├── test_evaluation_metrics.py
│   ├── test_plots.py
│   ├── test_rl_dry_run.py
│   └── test_streamlit_app.py
├── .gitignore
├── README.md
├── requirements.txt
└──
```

## Core implementation notes

- The reward is intentionally transparent: it is based on negative MSE between human target probabilities and predicted probabilities.
- The project uses LoRA adaptation rather than full fine-tuning to keep the prototype efficient.
- The evaluation workflow compares the probability distributions for Option A and Option B against the empirical human-choice distribution.
- All generated summaries and plots are saved under `results/` so they can be used for reporting or a later demo pipeline.

## Limitations

This is a prototype for research and education. It is not a fully validated behavioral-science model and should be treated as a reproducible experimentation scaffold rather than a final production system.

## Citation

Please cite the underlying dataset and the relevant behavioral decision-making literature when using this project in a publication or extended analysis.
