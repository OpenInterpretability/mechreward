"""TRL GRPOTrainer wrapper with residual-stream capture built in.

The standard ``trl.GRPOTrainer`` passes prompts/completions to reward
functions, but not hidden states. Mechreward needs the residual stream at
a specific layer. This wrapper:

1. Registers a forward hook on the policy at the configured SAE layer.
2. Captures hidden states during the rollout's forward passes.
3. Injects ``hidden_states`` + ``attention_mask`` into the kwargs that TRL
   forwards to reward functions.

Usage is otherwise identical to ``GRPOTrainer``.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


def wrap_reward_for_trl(
    reward_fn: Callable[..., list[float]],
    hidden_state_provider: Callable[[list[str]], dict[str, torch.Tensor]],
) -> Callable[..., list[float]]:
    """Adapt a mechreward-aware reward function to TRL's ``reward_funcs`` signature.

    Args:
        reward_fn: A FeatureReward or CompositeReward instance that expects
            ``hidden_states`` in kwargs.
        hidden_state_provider: Callable that takes ``completions`` and
            returns a dict with ``hidden_states`` and ``attention_mask``.

    Returns:
        A wrapped function with TRL's signature: ``(prompts, completions, **kwargs) -> list[float]``.
    """

    def wrapped(
        prompts: list[str] | None = None,
        completions: list[str] | None = None,
        **kwargs: Any,
    ) -> list[float]:
        if completions is None:
            return [0.0]
        hs_dict = hidden_state_provider(completions)
        merged = {**kwargs, **hs_dict}
        return reward_fn(prompts=prompts, completions=completions, **merged)

    wrapped.__name__ = getattr(reward_fn, "name", "mech_reward_wrapped")
    return wrapped


class MechRewardGRPOTrainer:
    """Lightweight wrapper around ``trl.GRPOTrainer`` that injects hidden states.

    This class is a thin orchestration layer — the actual TRL trainer is
    instantiated inside, and ``train()`` delegates.

    Args:
        model: HF model id or loaded model.
        reward_funcs: List of reward functions. FeatureReward / CompositeReward
            instances will be wrapped to receive hidden states. Other callables
            are passed through unchanged.
        layer_idx: Which layer to hook (must match the SAE).
        args: ``trl.GRPOConfig``.
        train_dataset: Training dataset.
        **trl_kwargs: Forwarded to ``trl.GRPOTrainer``.
    """

    def __init__(
        self,
        model: Any,
        reward_funcs: list[Any],
        layer_idx: int,
        args: Any = None,
        train_dataset: Any = None,
        **trl_kwargs: Any,
    ) -> None:
        try:
            from trl import GRPOTrainer
        except ImportError as e:
            raise ImportError(
                "MechRewardGRPOTrainer requires the `trl` extra. "
                "Install with `pip install 'mechreward[trl]'`."
            ) from e

        self.layer_idx = layer_idx
        self._capture_ref: list[torch.Tensor] = []

        wrapped_rewards = [
            wrap_reward_for_trl(r, self._provide_hidden_states) if self._needs_hidden(r) else r
            for r in reward_funcs
        ]

        self.trainer = GRPOTrainer(
            model=model,
            reward_funcs=wrapped_rewards,
            args=args,
            train_dataset=train_dataset,
            **trl_kwargs,
        )
        self._install_capture()

    @staticmethod
    def _needs_hidden(reward_fn: Any) -> bool:
        # FeatureReward, DualVerifier, CompositeReward containing them.
        from mechreward.hacking.dual_verifier import DualVerifier
        from mechreward.reward.composition import CompositeReward
        from mechreward.reward.feature_reward import FeatureReward

        if isinstance(reward_fn, (FeatureReward, DualVerifier)):
            return True
        if isinstance(reward_fn, CompositeReward):
            return any(
                MechRewardGRPOTrainer._needs_hidden(r) for r in reward_fn.rewards
            )
        return False

    def _install_capture(self) -> None:
        """Attach a persistent forward hook to the trainer's policy."""
        model = self.trainer.model
        layers = None
        for attr in ("layers", "h", "blocks"):
            if hasattr(model, "model") and hasattr(model.model, attr):
                layers = getattr(model.model, attr)
                break
            if hasattr(model, attr):
                layers = getattr(model, attr)
                break
        if layers is None:
            raise RuntimeError("Could not locate transformer layers on the policy model.")

        def hook(module, input, output):  # noqa: ARG001
            h = output[0] if isinstance(output, tuple) else output
            self._capture_ref.append(h.detach())

        layers[self.layer_idx].register_forward_hook(hook)

    def _provide_hidden_states(self, completions: list[str]) -> dict[str, torch.Tensor]:
        if not self._capture_ref:
            return {}
        hidden = self._capture_ref[-1]
        self._capture_ref.clear()
        return {"hidden_states": hidden}

    def train(self, *args: Any, **kwargs: Any) -> Any:
        return self.trainer.train(*args, **kwargs)

    def save_model(self, *args: Any, **kwargs: Any) -> Any:
        return self.trainer.save_model(*args, **kwargs)
