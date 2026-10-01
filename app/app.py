"""Streamlit demo for the human decision explanatory model."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.inference import generate_response, parse_prediction_text


def build_custom_prompt(scenario: str, option_a: str, option_b: str) -> str:
    """Create a user-facing risky-choice prompt matching the project's inference format."""
    scenario_text = (scenario or "A human decision-maker must choose between two uncertain options.").strip()
    option_a_text = (option_a or "Option A: a safe, certain outcome.").strip()
    option_b_text = (option_b or "Option B: a risky outcome with uncertainty.").strip()

    return (
        "You are modeling human decision-making.\n\n"
        "Consider this risky-choice problem:\n\n"
        f"Scenario:\n{scenario_text}\n\n"
        f"Option A:\n{option_a_text}\n\n"
        f"Option B:\n{option_b_text}\n\n"
        "Predict the distribution of human choices.\n\n"
        "Provide:\n"
        "1. A brief explanation.\n"
        "2. Probability for Option A.\n"
        "3. Probability for Option B.\n"
    )


def analyze_custom_prompt(scenario: str, option_a: str, option_b: str):
    """Generate a prediction and explanation for a user-supplied risky-choice prompt."""
    if not scenario.strip():
        raise ValueError("Please enter a scenario before analyzing the decision.")
    if not option_a.strip():
        raise ValueError("Please describe Option A before analyzing the decision.")
    if not option_b.strip():
        raise ValueError("Please describe Option B before analyzing the decision.")

    prompt = build_custom_prompt(scenario, option_a, option_b)
    try:
        response_text = generate_response(prompt, max_new_tokens=256, temperature=0.7, do_sample=True)
        prob_a, prob_b = parse_prediction_text(response_text)
        return {
            "prompt": prompt,
            "response_text": response_text,
            "predicted_prob_a": prob_a,
            "predicted_prob_b": prob_b,
            "warning": "",
        }
    except Exception as exc:  # pragma: no cover - surfaced in UI diagnostics
        fallback_a, fallback_b = 50.0, 50.0
        return {
            "prompt": prompt,
            "response_text": (
                "The model could not complete generation in this environment. "
                "Fallback estimate: Option A = 50%, Option B = 50%."
            ),
            "predicted_prob_a": fallback_a,
            "predicted_prob_b": fallback_b,
            "warning": f"Model generation failed: {exc}",
        }


def build_app_data(project_root: Path | str | None = None):
    """Build a minimal dashboard dataset from the project's processed and result files."""
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parent.parent
    data_path = root / "data" / "processed" / "test.csv"
    metrics_path = root / "results" / "metrics" / "model_comparison.csv"
    plots_dir = root / "results" / "plots"

    dataset = pd.read_csv(data_path) if data_path.exists() else pd.DataFrame()
    comparison_summary = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()

    metrics_files = {
        "comparison": metrics_path,
        "mse_plot": plots_dir / "mse_comparison.png",
        "mae_plot": plots_dir / "mae_comparison.png",
    }

    return {
        "dataset_samples": dataset.head(10).to_dict(orient="records"),
        "comparison_summary": comparison_summary,
        "metrics_files": metrics_files,
    }


def render_app(project_root: Path | str | None = None):
    """Render the Streamlit dashboard for the project."""
    st.set_page_config(page_title="Human Decision Explainer", page_icon="🧠", layout="wide")
    st.title("Human Decision Explainer")
    st.caption("Reinforcement Learning-Based LLM for Human Decision Modeling")

    data = build_app_data(project_root=project_root)
    dataset = data["dataset_samples"]
    comparison_summary = data["comparison_summary"]

    st.subheader("Project overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset rows", len(dataset))
    col2.metric("Models compared", len(comparison_summary))
    col3.metric("Results folder", "ready")

    st.subheader("Model comparison")
    if comparison_summary.empty:
        st.info("No model comparison summary is available yet. Generate predictions and metrics first.")
    else:
        st.dataframe(comparison_summary, use_container_width=True)

    st.subheader("Performance plots")
    metrics_files = data["metrics_files"]
    plot_cols = st.columns(2)
    for idx, key in enumerate(["mse_plot", "mae_plot"]):
        path = metrics_files.get(key)
        with plot_cols[idx]:
            if path and path.exists():
                st.image(str(path), use_container_width=True)
            else:
                st.info(f"Plot not available yet: {key}")

    st.subheader("Sample risky-choice instance")
    if dataset:
        sample = st.selectbox("Select sample", options=range(len(dataset)), format_func=lambda i: f"Sample {i + 1}")
        row = dataset[sample]
        st.markdown(f"**Scenario:** {row.get('scenario', 'N/A')}")
        st.markdown(f"**Option A:** {row.get('option_a', 'N/A')}")
        st.markdown(f"**Option B:** {row.get('option_b', 'N/A')}")
        st.markdown(f"**Human distribution:** A={row.get('actual_prob_a', 'N/A')}, B={row.get('actual_prob_b', 'N/A')}")
    else:
        st.info("No processed test samples are available yet.")

    st.subheader("Try your own risky-choice prompt")
    with st.form("custom_prompt_form"):
        custom_scenario = st.text_area(
            "Scenario",
            value=("A person must choose between a guaranteed benefit and a risky high-reward gamble."),
            height=120,
        )
        custom_option_a = st.text_area(
            "Option A",
            value="Receive $500 with certainty.",
            height=100,
        )
        custom_option_b = st.text_area(
            "Option B",
            value="Receive $2,000 with 30% probability or $0 with 70% probability.",
            height=100,
        )
        submitted = st.form_submit_button("Analyze this prompt")

    if submitted:
        try:
            result = analyze_custom_prompt(custom_scenario, custom_option_a, custom_option_b)
            if result.get("warning"):
                st.warning(result["warning"])
            else:
                st.success("Model analysis complete.")
            st.markdown(f"**Predicted probability for Option A:** {result['predicted_prob_a']:.2f}%")
            st.markdown(f"**Predicted probability for Option B:** {result['predicted_prob_b']:.2f}%")
            st.markdown("**Reasoning / model output:**")
            st.code(result["response_text"], language="text")
        except ValueError as exc:
            st.error(str(exc))


if __name__ == "__main__":
    render_app()
