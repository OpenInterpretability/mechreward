"""Tests for weight schedulers."""
import pytest

from mechreward.reward.scheduler import (
    ConstantSchedule,
    CosineSchedule,
    LinearAnnealSchedule,
    StepSchedule,
)


def test_constant_schedule():
    s = ConstantSchedule(value=0.7)
    assert s(0, 100) == 0.7
    assert s(50, 100) == 0.7
    assert s(100, 100) == 0.7


def test_linear_anneal_warmup():
    s = LinearAnnealSchedule(start=0.0, end=1.0, warmup_steps=10)
    assert s(0, 100) == 0.0
    assert s(5, 100) == 0.0
    assert s(10, 100) == 0.0
    assert s(55, 100) == pytest.approx(0.5, abs=0.01)
    assert s(100, 100) == pytest.approx(1.0, abs=0.01)


def test_cosine_schedule_endpoints():
    s = CosineSchedule(start=1.0, end=0.0)
    assert s(0, 100) == pytest.approx(1.0, abs=0.01)
    assert s(100, 100) == pytest.approx(0.0, abs=0.01)


def test_step_schedule():
    s = StepSchedule(values=[0.0, 0.3, 1.0], boundaries=[100, 500])
    assert s(0, 1000) == 0.0
    assert s(99, 1000) == 0.0
    assert s(100, 1000) == 0.3
    assert s(499, 1000) == 0.3
    assert s(500, 1000) == 1.0
    assert s(999, 1000) == 1.0


def test_step_schedule_rejects_mismatched():
    with pytest.raises(ValueError):
        StepSchedule(values=[0.0, 1.0], boundaries=[100, 500])
