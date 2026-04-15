"""Regularization terms to keep RL stable in the presence of mech-interp rewards.

These are plain torch tensor operations that a trainer can add to its loss.
They are intentionally stateless and batch-wise so they plug into any RL loop.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_penalty(
    policy_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """KL(policy || ref) per-token. Standard RLHF trust region.

    Args:
        policy_logprobs: ``[B, T]`` log-probs under the current policy.
        ref_logprobs: ``[B, T]`` log-probs under the frozen reference.
        reduction: ``"mean"``, ``"sum"``, or ``"none"``.

    Returns:
        A scalar (mean/sum) or a tensor (none) with the KL estimate.
    """
    delta = policy_logprobs - ref_logprobs
    if reduction == "mean":
        return delta.mean()
    if reduction == "sum":
        return delta.sum()
    return delta


def feature_diversity_bonus(
    feature_activations: torch.Tensor,
    target_entropy: float = 2.0,
    weight: float = 0.1,
) -> torch.Tensor:
    """Encourage diverse feature activation patterns across a batch.

    Prevents the policy from collapsing onto a single "easy-to-activate"
    feature. Computed as a (negative) distance from the target entropy of
    the feature activation distribution.

    Args:
        feature_activations: ``[B, F]`` activations summed or averaged over time.
        target_entropy: Desired entropy of the per-batch feature distribution.
        weight: Scale factor.

    Returns:
        A scalar bonus (larger = more diverse, add to reward).
    """
    if feature_activations.dim() == 3:
        feature_activations = feature_activations.mean(dim=1)  # [B, F]

    # Normalize across features for each batch element
    probs = F.softmax(feature_activations, dim=-1)  # [B, F]
    entropy_per_sample = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1)  # [B]
    mean_entropy = entropy_per_sample.mean()

    # Distance from target, one-sided (penalize below-target, tolerate above)
    deficit = (target_entropy - mean_entropy).clamp_min(0.0)
    return -weight * deficit


def entropy_bonus(logits: torch.Tensor, weight: float = 0.01) -> torch.Tensor:
    """Token-level entropy bonus. Encourages exploration.

    Args:
        logits: ``[B, T, V]`` policy logits.
        weight: Scale factor.

    Returns:
        Scalar bonus (larger = more exploration, add to reward).
    """
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    return weight * entropy


def reward_variance_penalty(
    rewards: torch.Tensor,
    target_std: float = 0.5,
    weight: float = 0.1,
) -> torch.Tensor:
    """Penalize batches where reward variance collapses (a hacking indicator).

    When a reward is being hacked, the model produces the same "winning"
    pattern for every prompt and reward variance drops to ~0. This penalty
    adds a loss term that grows as variance drops below the target.

    Args:
        rewards: ``[B]`` rewards from this batch.
        target_std: Desired minimum standard deviation.
        weight: Scale factor.

    Returns:
        Scalar penalty (larger = more collapsed, subtract from reward).
    """
    std = rewards.std(unbiased=False)
    deficit = (target_std - std).clamp_min(0.0)
    return weight * deficit
