"""HuggingFace generate + hidden state capture.

vLLM is the fast path (see ``vllm_with_hidden.py``) but HF generate is the
reliable fallback. This module runs ``model.generate`` with a forward hook
on the target layer so we get (completion, hidden_states) pairs for the
reward computation.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class HiddenStateCapture:
    """A hook that captures hidden states at a specific layer."""

    layer_idx: int
    captured: list[torch.Tensor]

    def __call__(self, module, input, output):  # noqa: ARG002
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        self.captured.append(hidden.detach())

    def last(self) -> torch.Tensor | None:
        if not self.captured:
            return None
        return self.captured[-1]

    def clear(self) -> None:
        self.captured.clear()


@contextmanager
def attach_capture(model, layer_idx: int):
    """Context manager: attach a forward hook to ``model.layers[layer_idx]``."""
    # Handle HF transformer variants. Most decoder models expose `model.layers`.
    layers_attr = None
    for attr in ("layers", "h", "blocks"):
        if hasattr(model, "model") and hasattr(model.model, attr):
            layers_attr = getattr(model.model, attr)
            break
        if hasattr(model, attr):
            layers_attr = getattr(model, attr)
            break
    if layers_attr is None:
        raise RuntimeError(
            "Could not locate transformer layers on the model. "
            "Tried: model.model.layers, model.model.h, model.layers, model.h."
        )
    if layer_idx < 0 or layer_idx >= len(layers_attr):
        raise IndexError(
            f"layer_idx={layer_idx} out of range (model has {len(layers_attr)} layers)"
        )

    capture = HiddenStateCapture(layer_idx=layer_idx, captured=[])
    handle = layers_attr[layer_idx].register_forward_hook(capture)
    try:
        yield capture
    finally:
        handle.remove()


class HFRollout:
    """Generate completions with HuggingFace and return captured hidden states.

    Args:
        model: An HF ``AutoModelForCausalLM`` (or compatible).
        tokenizer: Matching tokenizer.
        layer_idx: Which transformer layer to capture. Must match the SAE.
        generation_kwargs: Defaults forwarded to ``model.generate``.
    """

    def __init__(
        self,
        model,
        tokenizer,
        layer_idx: int,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.generation_kwargs = generation_kwargs or {
            "do_sample": True,
            "temperature": 1.0,
            "max_new_tokens": 512,
        }

    @torch.inference_mode()
    def generate(
        self,
        prompts: list[str],
        num_return_sequences: int = 1,
        **override_kwargs: Any,
    ) -> dict[str, Any]:
        """Generate completions and return ``{prompts, completions, hidden_states, attention_mask}``.

        Args:
            prompts: List of prompt strings.
            num_return_sequences: Number of samples per prompt (for GRPO groups).
            **override_kwargs: Overrides for ``model.generate``.

        Returns:
            Dict with keys:
                - ``prompts``: list of input prompts (expanded by group size)
                - ``completions``: list of decoded completions (without prompt)
                - ``hidden_states``: ``[B, T, d_model]`` of the SAE target layer
                - ``attention_mask``: ``[B, T]``
        """
        device = next(self.model.parameters()).device
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        gen_kwargs = {**self.generation_kwargs, **override_kwargs}
        gen_kwargs["num_return_sequences"] = num_return_sequences
        gen_kwargs["return_dict_in_generate"] = True
        gen_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)

        with attach_capture(self.model, self.layer_idx) as capture:
            out = self.model.generate(**enc, **gen_kwargs)

        sequences = out.sequences  # [B*K, T_full]
        prompt_len = enc["input_ids"].shape[1]
        completion_ids = sequences[:, prompt_len:]
        completions = self.tokenizer.batch_decode(
            completion_ids, skip_special_tokens=True
        )

        # Re-run forward on the full generated sequences to get hidden states at every token
        # (capture during generate only gets prompt encoding in most HF versions).
        full_enc = {
            "input_ids": sequences,
            "attention_mask": (sequences != self.tokenizer.pad_token_id).long()
            if self.tokenizer.pad_token_id is not None
            else torch.ones_like(sequences),
        }
        capture.clear()
        with attach_capture(self.model, self.layer_idx) as capture2:
            _ = self.model(**full_enc, output_hidden_states=False, return_dict=True)

        hidden = capture2.last()
        if hidden is None:
            raise RuntimeError("Failed to capture hidden states")

        expanded_prompts = []
        for p in prompts:
            expanded_prompts.extend([p] * num_return_sequences)

        return {
            "prompts": expanded_prompts,
            "completions": completions,
            "hidden_states": hidden,
            "attention_mask": full_enc["attention_mask"],
            "sequences": sequences,
        }
