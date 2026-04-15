"""Tests for the adversarial suite."""
from mechreward.hacking.adversarial import (
    STANDARD_SUITE,
    AdversarialSuite,
)


def test_standard_preset_loads():
    suite = AdversarialSuite.from_preset("standard")
    assert len(suite) == len(STANDARD_SUITE)
    assert all(hasattr(p, "name") for p in suite.prompts)


def test_unknown_preset_raises():
    import pytest

    with pytest.raises(ValueError):
        AdversarialSuite.from_preset("nonsense")


def test_evaluate_reward_runs():
    suite = AdversarialSuite.from_preset("standard")

    def dummy_reward(prompts, completions, **_):
        # Reward the string length — a deliberately terrible reward
        return [float(len(c)) for c in completions]

    result = suite.evaluate_reward(dummy_reward, threshold=100.0)
    assert "hack_rate" in result
    assert "false_negative_rate" in result
    assert "per_prompt" in result
    assert len(result["per_prompt"]) == len(suite)
