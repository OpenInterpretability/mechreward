"""Hacking detector based on Wilhelm et al. arxiv:2603.04069.

The paper shows that reward-hacking behaviors leave distinguishable
signatures in SAE feature activation patterns. We use a similar approach
during training: maintain a bank of features known to correlate with
"surface compliance without real work" (hedging phrases, template filler,
repetition) and flag trajectories where those features spike alongside
the reward features.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from mechreward.sae.batched_encode import batched_encode_selective
from mechreward.sae.loader import SAEHandle


@dataclass
class HackingReport:
    """Per-trajectory diagnostics for hacking detection."""

    trajectory_idx: int
    flagged: bool
    reason: str
    hacking_score: float
    reward_features_active: float
    hacking_features_active: float


@dataclass
class HackingDetector:
    """Detects reward-hacking patterns via a separate bank of "hacking signature" features.

    Logic:
    - We track a set of SAE features known to correlate with hacking behaviors
      (hedging, template filler, repetition, empty reasoning).
    - For each rollout, we compute both reward-feature activation and
      hacking-feature activation.
    - A trajectory is flagged when hacking-features activate HIGH while
      reward-features also activate — suggesting the model is producing
      surface patterns that light up the reward without doing real work.

    Args:
        sae: SAE handle.
        hacking_feature_ids: Feature ids known to signal hacking behaviors.
        reward_feature_ids: The features the FeatureReward uses (for ratio).
        hacking_threshold: Minimum hacking-feature activation to flag.
        ratio_threshold: Flag if hacking/reward activation ratio exceeds this.
    """

    sae: SAEHandle
    hacking_feature_ids: list[int]
    reward_feature_ids: list[int]
    hacking_threshold: float = 0.5
    ratio_threshold: float = 0.8
    history: list[HackingReport] = field(default_factory=list)

    @torch.inference_mode()
    def check_batch(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> list[HackingReport]:
        """Return a hacking report for each trajectory in the batch.

        Args:
            hidden_states: ``[B, T, d_model]`` activations.
            attention_mask: Optional ``[B, T]`` mask.

        Returns:
            One ``HackingReport`` per trajectory.
        """
        hack_acts = batched_encode_selective(
            self.sae, hidden_states, self.hacking_feature_ids
        )  # [B, T, H]
        reward_acts = batched_encode_selective(
            self.sae, hidden_states, self.reward_feature_ids
        )  # [B, T, R]

        if attention_mask is not None:
            mask = attention_mask.to(hack_acts.dtype).unsqueeze(-1)
            hack_acts = hack_acts * mask
            reward_acts = reward_acts * mask
            denom = mask.sum(dim=1).clamp_min(1.0)
            hack_mean = hack_acts.sum(dim=1) / denom
            reward_mean = reward_acts.sum(dim=1) / denom
        else:
            hack_mean = hack_acts.mean(dim=1)
            reward_mean = reward_acts.mean(dim=1)

        # Per-trajectory scalars
        hack_score = hack_mean.mean(dim=-1)  # [B]
        reward_score = reward_mean.mean(dim=-1)  # [B]

        reports = []
        for i in range(hack_score.shape[0]):
            h = float(hack_score[i].item())
            r = float(reward_score[i].item()) + 1e-6
            ratio = h / r
            flagged = h > self.hacking_threshold and ratio > self.ratio_threshold
            reason = ""
            if flagged:
                reason = (
                    f"hacking_feature_mean={h:.3f} > {self.hacking_threshold}, "
                    f"ratio={ratio:.3f} > {self.ratio_threshold}"
                )
            report = HackingReport(
                trajectory_idx=i,
                flagged=flagged,
                reason=reason,
                hacking_score=h,
                reward_features_active=r,
                hacking_features_active=h,
            )
            reports.append(report)
            self.history.append(report)

        return reports

    def flagged_indices(self, reports: list[HackingReport]) -> list[int]:
        return [r.trajectory_idx for r in reports if r.flagged]

    def flag_rate(self, window: int = 100) -> float:
        """Rolling flag rate over the last ``window`` trajectories."""
        recent = self.history[-window:]
        if not recent:
            return 0.0
        return sum(r.flagged for r in recent) / len(recent)
