"""Memory-efficient batched SAE encoding.

Running a SAE encoder on a long sequence at every GRPO rollout step is
the core compute hot-path in mechreward. This module centralises the
tricks that keep it fast and stable.
"""
from __future__ import annotations

from collections.abc import Iterator

import torch

from mechreward.sae.loader import SAEHandle


@torch.inference_mode()
def batched_encode(
    sae: SAEHandle,
    activations: torch.Tensor,
    batch_size: int = 64,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Encode a large tensor of hidden states in chunks.

    Args:
        sae: The SAE handle to use for encoding.
        activations: Hidden states of shape ``[B, T, d_model]`` or ``[N, d_model]``.
        batch_size: Number of rows (B*T) to encode at once. Tune for VRAM.
        dtype: Override the compute dtype. If None, uses the SAE's native dtype.

    Returns:
        A tensor of shape ``[B, T, d_sae]`` or ``[N, d_sae]`` on the same
        device as the input.
    """
    original_shape = activations.shape
    flat = activations.reshape(-1, original_shape[-1])

    if dtype is not None:
        flat = flat.to(dtype)

    outputs = []
    for chunk in _chunked(flat, batch_size):
        chunk = chunk.to(sae.device)
        encoded = sae.encode(chunk)
        outputs.append(encoded)

    out = torch.cat(outputs, dim=0)

    new_shape = list(original_shape[:-1]) + [out.shape[-1]]
    return out.reshape(new_shape)


def _chunked(tensor: torch.Tensor, size: int) -> Iterator[torch.Tensor]:
    for i in range(0, tensor.shape[0], size):
        yield tensor[i : i + size]


@torch.inference_mode()
def batched_encode_selective(
    sae: SAEHandle,
    activations: torch.Tensor,
    feature_ids: list[int] | torch.Tensor,
    batch_size: int = 64,
) -> torch.Tensor:
    """Encode and return only the activations for a subset of features.

    This is 2-10× faster than full encoding when you only care about
    a small number of pre-selected features (typical for reward signals).

    Args:
        sae: The SAE handle.
        activations: ``[B, T, d_model]`` or ``[N, d_model]`` hidden states.
        feature_ids: List or tensor of feature indices to return.
        batch_size: Chunking size.

    Returns:
        Tensor of shape ``[B, T, len(feature_ids)]`` or ``[N, len(feature_ids)]``.
    """
    if not isinstance(feature_ids, torch.Tensor):
        feature_ids = torch.tensor(feature_ids, dtype=torch.long, device=sae.device)

    full = batched_encode(sae, activations, batch_size=batch_size)
    return full.index_select(-1, feature_ids.to(full.device))
