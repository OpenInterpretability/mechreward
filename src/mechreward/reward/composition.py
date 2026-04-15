"""Combine multiple reward functions into one.

Supports two combination modes:
- ``sum``: literal weighted sum of component rewards. Cheap, transparent.
- ``stratified``: HERO-style stratification (arxiv:2510.07242) — sub-rewards
  are normalized *within groups* defined by a primary signal, preventing
  one dominant component from drowning the others.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import torch

RewardFn = Callable[..., list[float]]


class CompositeReward:
    """Combine multiple reward callables into one TRL-compatible reward function.

    Args:
        rewards: List of reward callables. Each must accept TRL-style kwargs
            and return ``list[float]`` of the same length as ``prompts``.
        weights: Scalar weight per reward (applied after normalization).
        mode: ``"sum"`` (default) or ``"stratified"``. In stratified mode,
            the first reward defines the group boundary (pos/neg); all
            other rewards are z-normalized within each group.
        name: Display name.

    Example:
        >>> composite = CompositeReward(
        ...     rewards=[outcome_reward, feature_reward],
        ...     weights=[1.0, 0.3],
        ...     mode="stratified",
        ... )
        >>> trainer = GRPOTrainer(..., reward_funcs=composite)
    """

    def __init__(
        self,
        rewards: list[RewardFn],
        weights: list[float] | None = None,
        mode: Literal["sum", "stratified"] = "sum",
        name: str = "composite_reward",
    ) -> None:
        if not rewards:
            raise ValueError("CompositeReward requires at least one reward function.")

        self.rewards = list(rewards)
        if weights is None:
            weights = [1.0] * len(rewards)
        if len(weights) != len(rewards):
            raise ValueError(
                f"Got {len(rewards)} rewards but {len(weights)} weights."
            )
        self.weights = [float(w) for w in weights]
        self.mode = mode
        self.name = name

    def __call__(
        self,
        prompts: list[str] | None = None,
        completions: list[str] | None = None,
        **kwargs: Any,
    ) -> list[float]:
        all_rewards: list[list[float]] = []
        for fn in self.rewards:
            rewards = fn(prompts=prompts, completions=completions, **kwargs)
            all_rewards.append(list(rewards))

        if self.mode == "sum":
            return self._combine_sum(all_rewards)
        if self.mode == "stratified":
            return self._combine_stratified(all_rewards)
        raise ValueError(f"Unknown composition mode: {self.mode}")

    def _combine_sum(self, all_rewards: list[list[float]]) -> list[float]:
        n = len(all_rewards[0])
        out = [0.0] * n
        for r_list, w in zip(all_rewards, self.weights, strict=False):
            for i, r in enumerate(r_list):
                out[i] += w * r
        return out

    def _combine_stratified(self, all_rewards: list[list[float]]) -> list[float]:
        """HERO-style: group by sign of the first reward, z-norm the rest within group.

        This prevents, e.g., a large positive outcome reward from drowning
        a small feature reward. Within each group (outcome-positive,
        outcome-negative) the feature rewards are z-normalized independently
        before summation.
        """
        if len(all_rewards) < 2:
            return self._combine_sum(all_rewards)

        primary = torch.tensor(all_rewards[0])
        pos_mask = primary > 0
        neg_mask = ~pos_mask

        combined = primary * self.weights[0]
        for idx in range(1, len(all_rewards)):
            sub = torch.tensor(all_rewards[idx])
            w = self.weights[idx]
            normalized = torch.zeros_like(sub)

            if pos_mask.any():
                pos_vals = sub[pos_mask]
                if pos_vals.numel() > 1:
                    normalized[pos_mask] = (pos_vals - pos_vals.mean()) / pos_vals.std(
                        unbiased=False
                    ).clamp_min(1e-6)
                else:
                    normalized[pos_mask] = 0.0

            if neg_mask.any():
                neg_vals = sub[neg_mask]
                if neg_vals.numel() > 1:
                    normalized[neg_mask] = (neg_vals - neg_vals.mean()) / neg_vals.std(
                        unbiased=False
                    ).clamp_min(1e-6)
                else:
                    normalized[neg_mask] = 0.0

            combined = combined + w * normalized

        return combined.tolist()

    def __repr__(self) -> str:
        names = [getattr(r, "name", type(r).__name__) for r in self.rewards]
        return (
            f"CompositeReward(mode={self.mode}, "
            f"rewards={names}, weights={self.weights})"
        )
