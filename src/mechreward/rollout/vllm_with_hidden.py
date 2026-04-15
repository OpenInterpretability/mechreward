"""vLLM rollout with hidden-state capture.

vLLM doesn't expose residual-stream hidden states through its public API.
We have two options:

1. **Two-stage**: use vLLM for fast generation, then rerun the sequences
   through an HF forward pass to capture hidden states. This doubles
   per-rollout compute but needs zero vLLM modifications.
2. **Fork-based**: patch vLLM's model runner to emit hidden states as an
   extra output. Single forward but requires maintaining a fork.

This module implements option 1 as the robust default. Option 2 can be
added via a custom ``vllm_backend`` parameter once we have a fork to target.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class VLLMRollout:
    """Two-stage vLLM rollout that falls back to HF for hidden state capture.

    This is a thin wrapper: vLLM handles sampling (fast), HF handles the
    downstream forward pass that the hook captures.

    Args:
        vllm_engine: A ``vllm.LLM`` instance (or compatible).
        hf_model: The HF version of the same model, used for hidden state capture.
        hf_tokenizer: Matching tokenizer.
        layer_idx: SAE target layer.
        sampling_params: vLLM ``SamplingParams`` for the rollout.
    """

    vllm_engine: Any
    hf_model: Any
    hf_tokenizer: Any
    layer_idx: int
    sampling_params: Any

    def __post_init__(self) -> None:
        try:
            import vllm  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "VLLMRollout requires the optional `vllm` extra. "
                "Install with `pip install 'mechreward[vllm]'`."
            ) from e

    @torch.inference_mode()
    def generate(
        self,
        prompts: list[str],
        num_return_sequences: int = 1,
    ) -> dict[str, Any]:
        """Run vLLM sampling + HF hidden-state capture in sequence.

        Returns the same dict shape as ``HFRollout.generate``.
        """
        # Expand prompts for group sampling
        expanded = []
        for p in prompts:
            expanded.extend([p] * num_return_sequences)

        # 1. vLLM sampling (fast)
        outputs = self.vllm_engine.generate(expanded, self.sampling_params)
        completions: list[str] = []
        full_texts: list[str] = []
        for o, prompt in zip(outputs, expanded, strict=False):
            text = o.outputs[0].text if o.outputs else ""
            completions.append(text)
            full_texts.append(prompt + text)

        # 2. HF forward for hidden state capture
        from mechreward.rollout.hf_rollout import attach_capture

        device = next(self.hf_model.parameters()).device
        enc = self.hf_tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        with attach_capture(self.hf_model, self.layer_idx) as capture:
            _ = self.hf_model(**enc, return_dict=True)

        hidden = capture.last()
        if hidden is None:
            raise RuntimeError("Failed to capture hidden states after vLLM rollout.")

        return {
            "prompts": expanded,
            "completions": completions,
            "hidden_states": hidden,
            "attention_mask": enc["attention_mask"],
        }
