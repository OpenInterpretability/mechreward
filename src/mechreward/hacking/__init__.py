"""Anti-Goodhart tools: hacking detection, dual verification, red team."""

from mechreward.hacking.adversarial import AdversarialPrompt, AdversarialSuite
from mechreward.hacking.detector import HackingDetector, HackingReport
from mechreward.hacking.dual_verifier import DualVerifier
from mechreward.hacking.regularization import (
    entropy_bonus,
    feature_diversity_bonus,
    kl_penalty,
)

__all__ = [
    "HackingDetector",
    "HackingReport",
    "DualVerifier",
    "AdversarialSuite",
    "AdversarialPrompt",
    "kl_penalty",
    "feature_diversity_bonus",
    "entropy_bonus",
]
