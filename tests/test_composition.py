"""Tests for CompositeReward."""
from mechreward.reward.composition import CompositeReward


def fake_reward_a(prompts, completions, **_):
    return [1.0, 2.0, 3.0]


def fake_reward_b(prompts, completions, **_):
    return [0.1, 0.2, 0.3]


def test_composite_sum_mode():
    import pytest

    comp = CompositeReward(rewards=[fake_reward_a, fake_reward_b], weights=[1.0, 2.0])
    out = comp(prompts=["p1", "p2", "p3"], completions=["c1", "c2", "c3"])
    # [1+0.2, 2+0.4, 3+0.6] = [1.2, 2.4, 3.6]
    assert out == pytest.approx([1.2, 2.4, 3.6])


def test_composite_stratified_mode_runs():
    comp = CompositeReward(
        rewards=[fake_reward_a, fake_reward_b],
        weights=[1.0, 0.3],
        mode="stratified",
    )
    out = comp(prompts=["p1", "p2", "p3"], completions=["c1", "c2", "c3"])
    assert len(out) == 3


def test_composite_rejects_mismatched_weights():
    import pytest

    with pytest.raises(ValueError):
        CompositeReward(rewards=[fake_reward_a], weights=[1.0, 2.0])


def test_composite_rejects_empty():
    import pytest

    with pytest.raises(ValueError):
        CompositeReward(rewards=[])
