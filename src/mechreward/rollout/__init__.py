"""Rollout backends that expose hidden states to downstream reward functions."""

from mechreward.rollout.hf_rollout import HFRollout, HiddenStateCapture

__all__ = ["HFRollout", "HiddenStateCapture"]
