"""Dual verification: SAE feature activation AND an independent probe must agree.

Given a feature that claims to represent "fact retrieval", we also train a
separate linear probe on labeled retrieval-vs-hallucination examples. At
training time, we check both signals. If the SAE feature fires but the
probe says "not retrieving", we downweight that trajectory.

This is the core anti-Goodhart mechanism of mechreward.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from mechreward.probes.linear_probe import LinearProbe
from mechreward.reward.feature_reward import FeatureReward


@dataclass
class DualVerifier:
    """Combine a FeatureReward with one or more independent probes.

    Args:
        feature_reward: The main FeatureReward to guard.
        probes: Dict mapping feature-bundle name → LinearProbe. Each probe
            should return a higher score when the target behavior is present.
        disagreement_threshold: Fraction of disagreement above which to
            downweight the reward. 0.3 means "if probes disagree with SAE on
            more than 30% of the batch, downweight".
        downweight_factor: Multiplier applied to the reward when triggered.
    """

    feature_reward: FeatureReward
    probes: dict[str, LinearProbe]
    disagreement_threshold: float = 0.3
    downweight_factor: float = 0.2
    disagreement_history: list[float] = field(default_factory=list)

    @torch.inference_mode()
    def compute(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, float]:
        """Compute the dual-verified reward.

        Returns:
            Tuple ``(rewards, disagreement_rate)``.
            - ``rewards``: ``[B]`` reward scores, downweighted if probes disagree.
            - ``disagreement_rate``: float in [0, 1], fraction of batch where
              probe and feature disagreed.
        """
        # Base reward from feature activation
        raw_reward = self.feature_reward.compute(hidden_states, attention_mask)

        if not self.probes:
            return raw_reward, 0.0

        # For each probe, compute its own reward on the same hidden states
        probe_scores = []
        for probe in self.probes.values():
            # Probes consume per-token features, so we pass last-token activation.
            last_token = hidden_states[:, -1, :]
            probe_scores.append(probe.predict(last_token))

        probe_scores_t = torch.stack(probe_scores, dim=0).mean(dim=0)  # [B]

        # Binarize both: positive reward = "SAE says yes", probe > 0 = "probe says yes"
        sae_says_yes = (raw_reward > raw_reward.median()).float()
        probe_says_yes = (probe_scores_t > probe_scores_t.median()).float()
        disagreement = (sae_says_yes != probe_says_yes).float().mean().item()
        self.disagreement_history.append(disagreement)

        # Downweight batch-wide if disagreement exceeds threshold
        if disagreement > self.disagreement_threshold:
            return raw_reward * self.downweight_factor, disagreement

        return raw_reward, disagreement

    def __call__(
        self,
        prompts: list[str] | None = None,
        completions: list[str] | None = None,
        hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> list[float]:
        if hidden_states is None:
            raise ValueError("DualVerifier requires hidden_states.")
        rewards, _ = self.compute(hidden_states, attention_mask)
        return rewards.detach().cpu().tolist()

    def recent_disagreement(self, window: int = 50) -> float:
        if not self.disagreement_history:
            return 0.0
        recent = self.disagreement_history[-window:]
        return sum(recent) / len(recent)
