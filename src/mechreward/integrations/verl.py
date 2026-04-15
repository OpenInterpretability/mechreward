"""verl adapter.

verl (github.com/verl-project/verl, ByteDance Seed) is built around the
HybridFlow paper. Reward functions live in ``verl/utils/reward_score``
and are called as ``reward_score(data_source, solution_str, ground_truth)``.

This adapter exposes a mechreward reward as a verl-compatible callable.
Note that verl by default doesn't pass hidden states — you'll need to use
verl's custom ``RewardModelWorker`` to get the residual stream. We provide
a stub worker class that can be extended when plugging into a verl job.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class VerlRewardAdapter:
    """Thin wrapper making a mechreward reward callable verl-style."""

    reward_fn: Callable[..., list[float]]
    rollout: Any

    def __call__(
        self,
        data_source: str,  # noqa: ARG002
        solution_str: str,
        ground_truth: str,  # noqa: ARG002
    ) -> float:
        """Verl-style signature: (data_source, solution_str, ground_truth) → float."""
        out = self.rollout.generate(prompts=[""], num_return_sequences=1)
        rewards = self.reward_fn(
            prompts=[""],
            completions=[solution_str],
            hidden_states=out["hidden_states"],
            attention_mask=out.get("attention_mask"),
        )
        return float(rewards[0]) if rewards else 0.0


def register_verl_reward(
    reward_fn: Callable[..., list[float]],
    rollout: Any,
    name: str = "mechreward",
) -> None:
    """Register a mechreward reward with verl's reward registry.

    Verl discovers reward functions at runtime via its ``_select_rm_score_fn``
    lookup. This function mutates the registry to include mechreward.
    """
    try:
        from verl.utils.reward_score import _default_compute_score
    except ImportError as e:
        raise ImportError(
            "register_verl_reward requires verl to be installed. "
            "Install verl from https://github.com/verl-project/verl."
        ) from e

    adapter = VerlRewardAdapter(reward_fn=reward_fn, rollout=rollout)

    # Hook into verl's reward registry (API may vary across versions).
    _default_compute_score[name] = adapter  # type: ignore[index]
