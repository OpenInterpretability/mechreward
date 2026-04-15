"""SAE loading wrapper with a clean, library-agnostic interface.

The actual SAE computation is delegated to `sae_lens` when available. This
module exposes a minimal `SAEHandle` protocol so the rest of the code can
depend on a stable surface even if the backend changes.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from mechreward.sae.cache import ensure_cached


class SAEBackend(Protocol):
    """Minimal protocol a SAE implementation must satisfy."""

    def encode(self, activations: torch.Tensor) -> torch.Tensor:
        """[..., d_model] → [..., d_sae]."""
        ...

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """[..., d_sae] → [..., d_model]."""
        ...


@dataclass
class SAEHandle:
    """Library-agnostic SAE wrapper.

    Attributes:
        backend: The underlying SAE object (sae_lens.SAE or compatible).
        release: Release identifier (e.g. "gemma-scope-9b-pt-res-canonical").
        sae_id: Specific SAE within a release (e.g. "layer_22/width_16k/canonical").
        hook_name: The transformer hook where this SAE was trained (e.g. "blocks.22.hook_resid_post").
        layer: Integer layer index if available, else -1.
        d_model: Dimensionality of the model hidden state this SAE consumes.
        d_sae: Dimensionality of the SAE feature space.
        model_name: The base model this SAE is compatible with.
    """

    backend: Any
    release: str
    sae_id: str
    hook_name: str
    layer: int
    d_model: int
    d_sae: int
    model_name: str

    def encode(self, activations: torch.Tensor) -> torch.Tensor:
        return self.backend.encode(activations)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return self.backend.decode(features)

    @property
    def device(self) -> torch.device:
        if hasattr(self.backend, "device"):
            return self.backend.device  # type: ignore[no-any-return]
        # Fallback: infer from a parameter
        for p in self.backend.parameters():
            return p.device
        return torch.device("cpu")

    def to(self, device: torch.device | str) -> SAEHandle:
        self.backend = self.backend.to(device)
        return self


def _parse_layer_from_sae_id(sae_id: str) -> int:
    """Extract layer number from Gemma Scope / Llama Scope style ids."""
    import re

    for token in sae_id.replace("/", " ").split():
        m = re.match(r"layer[_-]?(\d+)", token, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
        m = re.match(r"l(\d+)", token, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    return -1


def load_sae(
    release: str,
    sae_id: str,
    device: str | torch.device | None = None,
    model_name: str | None = None,
) -> SAEHandle:
    """Load a SAE from `sae_lens` and wrap it in a `SAEHandle`.

    Args:
        release: The `sae_lens` release name. Examples:
            - ``gemma-scope-9b-pt-res-canonical``
            - ``gemma-scope-2b-pt-res-canonical``
            - ``llama_scope_lxr_32x``
        sae_id: The specific SAE within the release. Examples:
            - ``layer_22/width_16k/canonical``
            - ``l20r_32x``
        device: Target device. Defaults to CUDA if available, else CPU.
        model_name: The HF model id this SAE is compatible with. If omitted,
            it is inferred from the release string where possible.

    Returns:
        A populated ``SAEHandle``.

    Raises:
        ImportError: If ``sae_lens`` is not installed.
        RuntimeError: If the release/sae_id cannot be resolved.
    """
    try:
        from sae_lens import SAE
    except ImportError as e:
        raise ImportError(
            "mechreward requires sae_lens to load SAEs. "
            "Install with `pip install 'mechreward[sae]'`."
        ) from e

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cache_path = ensure_cached(release, sae_id)
    # sae_lens handles its own caching; we just ensure the directory exists
    # for downstream metadata writes.
    _ = cache_path

    sae, cfg_dict, _sparsity = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
        device=str(device),
    )

    hook_name = cfg_dict.get("hook_name") or cfg_dict.get("hook_point") or ""
    layer = cfg_dict.get("hook_layer")
    if layer is None:
        layer = _parse_layer_from_sae_id(sae_id)
    d_model = int(cfg_dict.get("d_in", getattr(sae, "d_in", 0)))
    d_sae = int(cfg_dict.get("d_sae", getattr(sae, "d_sae", 0)))
    if not d_sae and hasattr(sae, "W_enc"):
        d_sae = int(sae.W_enc.shape[-1])
    if not d_model and hasattr(sae, "W_enc"):
        d_model = int(sae.W_enc.shape[0])

    if model_name is None:
        model_name = _infer_model_from_release(release)

    return SAEHandle(
        backend=sae,
        release=release,
        sae_id=sae_id,
        hook_name=hook_name,
        layer=int(layer),
        d_model=d_model,
        d_sae=d_sae,
        model_name=model_name,
    )


_RELEASE_TO_MODEL = {
    "gemma-scope-2b-pt-res-canonical": "google/gemma-2-2b",
    "gemma-scope-2b-pt-res": "google/gemma-2-2b",
    "gemma-scope-2b-pt-mlp-canonical": "google/gemma-2-2b",
    "gemma-scope-2b-pt-att-canonical": "google/gemma-2-2b",
    "gemma-scope-9b-pt-res-canonical": "google/gemma-2-9b",
    "gemma-scope-9b-pt-res": "google/gemma-2-9b",
    "gemma-scope-9b-pt-mlp-canonical": "google/gemma-2-9b",
    "gemma-scope-9b-pt-att-canonical": "google/gemma-2-9b",
    "gemma-scope-9b-it-res-canonical": "google/gemma-2-9b-it",
    "gemma-scope-27b-pt-res-canonical": "google/gemma-2-27b",
    "llama_scope_lxr_8x": "meta-llama/Meta-Llama-3.1-8B",
    "llama_scope_lxr_32x": "meta-llama/Meta-Llama-3.1-8B",
    "llama_scope_lxa_32x": "meta-llama/Meta-Llama-3.1-8B",
    "llama_scope_lxm_32x": "meta-llama/Meta-Llama-3.1-8B",
}


def _infer_model_from_release(release: str) -> str:
    """Map a release string to a HuggingFace model id."""
    if release in _RELEASE_TO_MODEL:
        return _RELEASE_TO_MODEL[release]
    warnings.warn(
        f"Could not infer base model for SAE release '{release}'. "
        "Pass `model_name` explicitly to silence this warning.",
        stacklevel=2,
    )
    return "unknown"
