"""FeatureReward: SAE-feature activations as a trajectory-level reward.

This is the core primitive that users call. Given a tokenized batch of
completions and the matching hidden states, it:

1. Projects the hidden states through a SAE.
2. Selects the target features.
3. Reduces per-token activations to per-trajectory rewards via an aggregation.
4. Normalizes across the batch.
5. Returns a ``torch.Tensor`` of shape ``[B]`` for the trainer to consume.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from mechreward.features.catalog import FeaturePack, load_pack
from mechreward.reward.aggregation import (
    Aggregation,
    build_aggregation,
)
from mechreward.reward.normalization import (
    Normalization,
    build_normalization,
)
from mechreward.sae.batched_encode import batched_encode_selective
from mechreward.sae.loader import SAEHandle


class FeatureReward:
    """Turn SAE feature activations into a RL reward signal.

    Args:
        features: Either a ``FeaturePack`` or a ``dict[int, float]`` mapping
            feature ids to contribution weights. Positive weights reward
            activation, negative weights penalize it.
        sae: The SAE handle whose features we're rewarding.
        aggregation: How to reduce ``[B, T, F]`` → ``[B, F]``. Accepts an
            Aggregation instance or a string name (e.g. ``"mean"``,
            ``"mean_last_32_tokens"``).
        normalization: Post-aggregation normalization. Defaults to no-op so
            the raw signal reaches the trainer (use ZScore if combining with
            outcome rewards).
        threshold: Only count activations above this value.
        name: Display name (for logs).

    Example:
        >>> sae = load_sae("gemma-scope-9b-pt-res-canonical", "layer_22/width_16k/canonical")
        >>> reward = FeatureReward(
        ...     features={2817: 1.0, 4219: -0.5},
        ...     sae=sae,
        ...     aggregation="mean_last_16_tokens",
        ... )
    """

    def __init__(
        self,
        features: FeaturePack | dict[int, float],
        sae: SAEHandle,
        aggregation: Aggregation | str = "mean",
        normalization: Normalization | str = "noop",
        threshold: float = 0.0,
        name: str = "feature_reward",
    ) -> None:
        if isinstance(features, FeaturePack):
            self.pack: FeaturePack | None = features
            weights_dict = features.feature_weights()
        else:
            self.pack = None
            weights_dict = features

        if not weights_dict:
            raise ValueError("FeatureReward requires at least one feature.")

        self._feature_ids = sorted(weights_dict.keys())
        device = sae.device
        self._feature_ids_t = torch.tensor(self._feature_ids, dtype=torch.long, device=device)
        self._weights_t = torch.tensor(
            [weights_dict[i] for i in self._feature_ids],
            dtype=torch.float32,
            device=device,
        )

        self.sae = sae
        self.threshold = float(threshold)
        self.name = name

        self._aggregation: Aggregation = (
            aggregation
            if isinstance(aggregation, Aggregation)
            else build_aggregation(aggregation)
        )
        self._normalization: Normalization = (
            normalization
            if isinstance(normalization, Normalization)
            else build_normalization(normalization)
        )

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def compute(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the reward from hidden states.

        Args:
            hidden_states: ``[B, T, d_model]`` activations at the SAE's target layer.
            attention_mask: Optional ``[B, T]`` mask.

        Returns:
            ``[B]`` reward scores, normalized per the configured normalization.
        """
        if hidden_states.dim() != 3:
            raise ValueError(
                f"Expected 3D hidden states [B, T, d_model], got shape {hidden_states.shape}"
            )

        # Encode only the target features (selective encode is 5-20× faster
        # than full encoding when we only want a handful of features).
        feature_acts = batched_encode_selective(
            self.sae,
            hidden_states,
            feature_ids=self._feature_ids,
        )  # [B, T, F]

        if self.threshold > 0:
            feature_acts = torch.where(
                feature_acts > self.threshold,
                feature_acts,
                torch.zeros_like(feature_acts),
            )

        # Weighted sum over features → [B, T]
        per_token_reward = (feature_acts * self._weights_t.to(feature_acts.device)).sum(dim=-1)

        # Aggregate over time → [B]
        per_traj_reward = self._aggregation(per_token_reward, attention_mask)

        # Normalize across batch
        return self._normalization(per_traj_reward)

    # ------------------------------------------------------------------
    # TRL-compatible interface
    # ------------------------------------------------------------------
    def __call__(
        self,
        prompts: list[str] | None = None,
        completions: list[str] | None = None,
        hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> list[float]:
        """TRL-style reward function.

        TRL's GRPOTrainer will pass prompts, completions, and (via our
        rollout hook) ``hidden_states`` and ``attention_mask``. We ignore
        the text and compute reward from activations alone.

        Returns a Python list so TRL can serialize it across workers.
        """
        if hidden_states is None:
            raise ValueError(
                "FeatureReward requires hidden_states in kwargs. "
                "Use `MechRewardGRPOTrainer` or the rollout hook from "
                "`mechreward.rollout` to provide them."
            )

        rewards = self.compute(hidden_states, attention_mask)
        return rewards.detach().cpu().tolist()

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_pack(
        cls,
        pack_name: str,
        sae: SAEHandle,
        aggregation: Aggregation | str = "mean",
        normalization: Normalization | str = "noop",
        **kwargs: Any,
    ) -> FeatureReward:
        """Construct a ``FeatureReward`` from a named catalog pack."""
        pack = load_pack(pack_name)
        return cls(
            features=pack,
            sae=sae,
            aggregation=aggregation,
            normalization=normalization,
            name=pack.name,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def feature_ids(self) -> list[int]:
        return list(self._feature_ids)

    def __repr__(self) -> str:
        return (
            f"FeatureReward(n_features={len(self._feature_ids)}, "
            f"aggregation={type(self._aggregation).__name__}, "
            f"normalization={type(self._normalization).__name__}, "
            f"sae={self.sae.release}/{self.sae.sae_id})"
        )


# ---------------------------------------------------------------------------
# Outcome-reward helper for convenience
# ---------------------------------------------------------------------------


class OutcomeReward:
    """A thin wrapper to make outcome verifiers TRL-compatible.

    Outcome verifiers are pure string-in/bool-out predicates. This class
    adapts them to the ``(prompts, completions) -> list[float]`` signature
    that TRL's ``reward_funcs`` argument expects, so you can stack outcome
    and feature rewards in a ``CompositeReward`` without conversion boilerplate.

    Args:
        verifier: Callable ``(prompt, completion) -> bool`` that returns True
            if the completion is correct.
        true_reward: Value to emit when the verifier returns True.
        false_reward: Value to emit when the verifier returns False.
        name: Display name.
    """

    def __init__(
        self,
        verifier: Callable[[str, str], bool],
        true_reward: float = 1.0,
        false_reward: float = 0.0,
        name: str = "outcome_reward",
    ) -> None:
        self.verifier = verifier
        self.true_reward = float(true_reward)
        self.false_reward = float(false_reward)
        self.name = name

    def __call__(
        self,
        prompts: list[str],
        completions: list[str],
        **kwargs: Any,
    ) -> list[float]:
        rewards = []
        for p, c in zip(prompts, completions, strict=False):
            try:
                ok = bool(self.verifier(p, c))
            except Exception:
                ok = False
            rewards.append(self.true_reward if ok else self.false_reward)
        return rewards
