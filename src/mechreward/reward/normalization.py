"""Reward normalization strategies.

Raw SAE feature activations have arbitrary scale — some features always fire
between 0 and 0.5, others between 0 and 50. Before feeding them to a policy
gradient, we normalize. Options here are complementary to HERO-style
stratified normalization (which is done at the `composition` layer).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


class Normalization(ABC):
    """Abstract base class."""

    @abstractmethod
    def __call__(self, rewards: torch.Tensor) -> torch.Tensor:
        """Map raw rewards to normalized rewards."""
        ...


@dataclass
class ZScoreNormalization(Normalization):
    """Subtract mean, divide by std (per batch)."""

    eps: float = 1e-6
    clip: float | None = 5.0

    def __call__(self, rewards: torch.Tensor) -> torch.Tensor:
        mean = rewards.mean()
        std = rewards.std(unbiased=False).clamp_min(self.eps)
        out = (rewards - mean) / std
        if self.clip is not None:
            out = out.clamp(-self.clip, self.clip)
        return out


@dataclass
class RankNormalization(Normalization):
    """Replace rewards with their rank percentile [0, 1]."""

    def __call__(self, rewards: torch.Tensor) -> torch.Tensor:
        n = rewards.numel()
        if n <= 1:
            return torch.zeros_like(rewards)
        sorted_idx = rewards.argsort()
        ranks = torch.empty_like(sorted_idx, dtype=rewards.dtype)
        ranks[sorted_idx] = torch.arange(n, dtype=rewards.dtype, device=rewards.device)
        return ranks / (n - 1)


@dataclass
class SigmoidSquash(Normalization):
    """Squash rewards to [0, 1] via sigmoid. Useful when combining with outcome rewards."""

    temperature: float = 1.0

    def __call__(self, rewards: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(rewards / self.temperature)


@dataclass
class MinMaxNormalization(Normalization):
    """Rescale to [0, 1] based on batch min and max."""

    eps: float = 1e-6

    def __call__(self, rewards: torch.Tensor) -> torch.Tensor:
        lo = rewards.min()
        hi = rewards.max()
        return (rewards - lo) / (hi - lo + self.eps)


@dataclass
class NoopNormalization(Normalization):
    """Identity."""

    def __call__(self, rewards: torch.Tensor) -> torch.Tensor:
        return rewards


def build_normalization(name: str, **kwargs) -> Normalization:
    """Factory for named normalizations."""
    name = name.strip().lower()
    if name in ("zscore", "z-score", "z_score"):
        return ZScoreNormalization(**kwargs)
    if name in ("rank", "rank_percentile"):
        return RankNormalization()
    if name == "sigmoid":
        return SigmoidSquash(**kwargs)
    if name in ("minmax", "min_max"):
        return MinMaxNormalization(**kwargs)
    if name in ("noop", "none", "identity"):
        return NoopNormalization()
    raise ValueError(f"Unknown normalization: '{name}'")
