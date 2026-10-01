from pathlib import Path

import pytest

from app.app import build_app_data, build_custom_prompt, analyze_custom_prompt


def test_build_app_data_reads_project_files():
    app_data = build_app_data(project_root=Path(__file__).resolve().parent.parent)

    assert "dataset_samples" in app_data
    assert "comparison_summary" in app_data
    assert "metrics_files" in app_data


def test_build_custom_prompt_contains_expected_sections():
    prompt = build_custom_prompt(
        scenario="A person chooses between a sure gain and a risky payoff.",
        option_a="Receive $100 for sure.",
        option_b="Receive $300 with 50% probability or $0 with 50% probability.",
    )

    assert "Scenario" in prompt
    assert "Option A" in prompt
    assert "Option B" in prompt
    assert "Predict the distribution of human choices" in prompt


def test_analyze_custom_prompt_uses_fallback_when_model_generation_fails(monkeypatch):
    def _fail(*args, **kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr("app.app.generate_response", _fail)
    result = analyze_custom_prompt(
        "A person chooses between a guaranteed benefit and a risky high-reward gamble.",
        "Receive $500 with certainty.",
        "Receive $2000 with 30% probability or $0 with 70% probability.",
    )

    assert "predicted_prob_a" in result
    assert "predicted_prob_b" in result
    assert result["predicted_prob_a"] + result["predicted_prob_b"] == pytest.approx(100.0)
