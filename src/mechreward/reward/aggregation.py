"""Token-level SAE activations → single scalar reward per trajectory.

The SAE produces a feature activation at every token. The RL trainer needs
one number per rollout. This module bridges the gap with explicit,
composable aggregation strategies.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


class Aggregation(ABC):
    """Abstract base class for token → trajectory aggregations."""

    @abstractmethod
    def __call__(
        self,
        activations: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reduce ``[B, T]`` or ``[B, T, F]`` feature activations to ``[B]`` or ``[B, F]``.

        Args:
            activations: Per-token feature activations.
            attention_mask: Optional mask ``[B, T]``; positions where the mask is 0
                are excluded from the aggregation.

        Returns:
            Aggregated activations with the time dimension collapsed.
        """
        ...


@dataclass
class MeanAggregation(Aggregation):
    """Arithmetic mean over all (unmasked) tokens in the trajectory."""

    def __call__(
        self,
        activations: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            return activations.mean(dim=1)
        mask = attention_mask.to(activations.dtype)
        while mask.dim() < activations.dim():
            mask = mask.unsqueeze(-1)
        total = (activations * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp_min(1.0)
        return total / count


@dataclass
class LastKAggregation(Aggregation):
    """Mean over the last K unmasked tokens. Useful for reward-on-final-answer."""

    k: int = 16

    def __call__(
        self,
        activations: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T = activations.shape[0], activations.shape[1]
        k = min(self.k, T)

        if attention_mask is None:
            return activations[:, -k:].mean(dim=1)

        # For each batch element, average the last k unmasked positions
        out = []
        for b in range(B):
            mask = attention_mask[b].bool()
            idx = mask.nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                out.append(activations[b].mean(dim=0))
                continue
            selected = idx[-k:]
            out.append(activations[b, selected].mean(dim=0))
        return torch.stack(out, dim=0)


@dataclass
class MaxAggregation(Aggregation):
    """Max over the trajectory. Rewards peak activation of the feature."""

    def __call__(
        self,
        activations: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            return activations.max(dim=1).values
        mask = attention_mask.to(activations.dtype)
        while mask.dim() < activations.dim():
            mask = mask.unsqueeze(-1)
        very_negative = torch.finfo(activations.dtype).min
        masked = torch.where(mask.bool(), activations, torch.full_like(activations, very_negative))
        return masked.max(dim=1).values


@dataclass
class WeightedMeanAggregation(Aggregation):
    """Weighted mean where later tokens count more (linear ramp)."""

    def __call__(
        self,
        activations: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        T = activations.shape[1]
        weights = torch.linspace(0.1, 1.0, T, device=activations.device, dtype=activations.dtype)
        weights = weights.view(1, T, *([1] * (activations.dim() - 2)))

        if attention_mask is not None:
            mask = attention_mask.to(activations.dtype)
            while mask.dim() < activations.dim():
                mask = mask.unsqueeze(-1)
            weights = weights * mask

        total = (activations * weights).sum(dim=1)
        norm = weights.sum(dim=1).clamp_min(1e-6)
        return total / norm


def build_aggregation(name: str, **kwargs) -> Aggregation:
    """Factory that turns a string identifier into an Aggregation.

    Supports: ``mean``, ``max``, ``last_k`` (k arg), ``mean_last_N_tokens``,
    ``weighted_mean``.

    Examples:
        >>> build_aggregation("mean_last_32_tokens")
        LastKAggregation(k=32)
        >>> build_aggregation("max")
        MaxAggregation()
    """
    name = name.strip().lower()
    if name == "mean":
        return MeanAggregation()
    if name == "max":
        return MaxAggregation()
    if name == "weighted_mean":
        return WeightedMeanAggregation()
    if name == "last_k":
        return LastKAggregation(k=int(kwargs.get("k", 16)))
    # Pattern: mean_last_N_tokens → LastKAggregation(k=N)
    import re

    m = re.match(r"mean_last_(\d+)_tokens?", name)
    if m:
        return LastKAggregation(k=int(m.group(1)))
    m = re.match(r"last_(\d+)", name)
    if m:
        return LastKAggregation(k=int(m.group(1)))
    raise ValueError(f"Unknown aggregation: '{name}'")
