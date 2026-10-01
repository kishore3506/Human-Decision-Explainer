from src.train_rl import train_rl


def test_train_rl_dry_run_prints_configuration(capsys):
    result = train_rl(train=False)
    captured = capsys.readouterr().out

    assert result is None
    assert "GRPO dry run only" in captured
    assert "Reward:" in captured
    assert "Model:" in captured
