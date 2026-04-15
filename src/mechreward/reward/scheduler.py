"""Weight schedulers for annealing reward components over training.

A common anti-Goodhart pattern: start with heavy outcome-reward and slowly
increase mechreward's weight as training stabilizes, or vice versa.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass


class WeightSchedule(ABC):
    """Abstract base class for a weight schedule over training steps."""

    @abstractmethod
    def __call__(self, step: int, total_steps: int) -> float:
        ...


@dataclass
class ConstantSchedule(WeightSchedule):
    value: float = 1.0

    def __call__(self, step: int, total_steps: int) -> float:
        return self.value


@dataclass
class LinearAnnealSchedule(WeightSchedule):
    """Linearly anneal from ``start`` to ``end`` over the full training run."""

    start: float
    end: float
    warmup_steps: int = 0

    def __call__(self, step: int, total_steps: int) -> float:
        if step < self.warmup_steps:
            return self.start
        if total_steps <= self.warmup_steps:
            return self.end
        progress = (step - self.warmup_steps) / max(1, total_steps - self.warmup_steps)
        progress = max(0.0, min(1.0, progress))
        return self.start + progress * (self.end - self.start)


@dataclass
class CosineSchedule(WeightSchedule):
    """Cosine anneal from ``start`` to ``end``."""

    start: float
    end: float
    warmup_steps: int = 0

    def __call__(self, step: int, total_steps: int) -> float:
        if step < self.warmup_steps:
            return self.start
        progress = (step - self.warmup_steps) / max(1, total_steps - self.warmup_steps)
        progress = max(0.0, min(1.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.end + (self.start - self.end) * cosine


@dataclass
class StepSchedule(WeightSchedule):
    """Piecewise-constant schedule.

    Example: ``StepSchedule(values=[0.0, 0.3, 1.0], boundaries=[100, 500])``
    returns 0.0 for steps < 100, 0.3 for 100 ≤ step < 500, 1.0 for step ≥ 500.
    """

    values: list[float]
    boundaries: list[int]

    def __post_init__(self) -> None:
        if len(self.values) != len(self.boundaries) + 1:
            raise ValueError(
                "StepSchedule needs len(values) == len(boundaries) + 1"
            )

    def __call__(self, step: int, total_steps: int) -> float:
        for i, boundary in enumerate(self.boundaries):
            if step < boundary:
                return self.values[i]
        return self.values[-1]
