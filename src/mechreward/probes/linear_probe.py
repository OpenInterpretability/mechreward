"""LinearProbe: a simple logistic head on hidden states.

Two use cases in mechreward:
1. **Cheap alternative to SAE features**: when you don't have a SAE for the
   target model, a linear probe trained on a few hundred labeled examples
   can often capture simple behaviors (confidence, refusal, genre).
2. **Dual-verification signal**: a probe trained on *fresh* labels can catch
   cases where a SAE feature is being gamed by the policy.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


@dataclass
class LinearProbe:
    """A scikit-learn-style logistic probe over hidden states.

    Attributes:
        weight: ``[d_model]`` weight vector.
        bias: scalar bias.
        d_model: dimensionality of the input hidden state.
        name: identifier.
        metadata: free-form metadata (training examples, threshold, etc.).
    """

    weight: torch.Tensor
    bias: torch.Tensor
    d_model: int
    name: str = "probe"
    metadata: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}

    @torch.inference_mode()
    def predict(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return raw logits (signed). Positive = class 1, negative = class 0.

        Args:
            hidden_states: ``[B, d_model]`` or ``[B, T, d_model]``.

        Returns:
            ``[B]`` or ``[B, T]`` logits.
        """
        w = self.weight.to(hidden_states.device)
        b = self.bias.to(hidden_states.device)
        if hidden_states.dim() == 2:
            return hidden_states @ w + b
        if hidden_states.dim() == 3:
            # Project each token
            return (hidden_states * w).sum(dim=-1) + b
        raise ValueError(
            f"Probe expects 2D or 3D hidden states, got shape {hidden_states.shape}"
        )

    def predict_proba(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.predict(hidden_states))

    def to(self, device: str | torch.device) -> LinearProbe:
        self.weight = self.weight.to(device)
        self.bias = self.bias.to(device)
        return self

    @classmethod
    def from_sklearn(cls, sk_model, name: str = "probe") -> LinearProbe:
        """Convert a trained sklearn LogisticRegression into a LinearProbe."""
        import numpy as np

        w = torch.tensor(sk_model.coef_.reshape(-1), dtype=torch.float32)
        b_value = float(np.asarray(sk_model.intercept_).reshape(-1)[0])
        b = torch.tensor(b_value, dtype=torch.float32)
        return cls(weight=w, bias=b, d_model=w.numel(), name=name)


def save_probe(probe: LinearProbe, path: str | Path) -> None:
    """Serialize a probe to disk as a pair of .pt + .json files."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"weight": probe.weight, "bias": probe.bias, "d_model": probe.d_model},
        out.with_suffix(".pt"),
    )
    with out.with_suffix(".json").open("w") as f:
        json.dump({"name": probe.name, "metadata": probe.metadata}, f, indent=2)


def load_probe(name_or_path: str | Path) -> LinearProbe:
    """Load a probe from a bundled catalog or an arbitrary path."""
    path = Path(str(name_or_path))
    if not path.suffix:
        # Try as a bundled catalog entry
        here = Path(__file__).resolve().parents[3]
        candidate = here / "catalogs" / f"{path}.pt"
        if candidate.exists():
            path = candidate
        else:
            raise FileNotFoundError(f"Could not find probe: {name_or_path}")

    state = torch.load(path, map_location="cpu", weights_only=True)
    weight = state["weight"]
    bias = state["bias"]
    d_model = int(state.get("d_model", weight.numel()))

    meta_path = path.with_suffix(".json")
    name = "probe"
    metadata: dict[str, Any] = {}
    if meta_path.exists():
        with meta_path.open() as f:
            meta = json.load(f)
            name = meta.get("name", "probe")
            metadata = meta.get("metadata", {})

    return LinearProbe(weight=weight, bias=bias, d_model=d_model, name=name, metadata=metadata)


class TorchLinearProbe(nn.Module):
    """A torch.nn.Module version for in-training probe updates.

    Use this when you want the probe to be differentiable and trained with
    the rest of the pipeline (e.g. as a concurrent critic).
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden_states).squeeze(-1)

    def to_static(self, name: str = "probe") -> LinearProbe:
        w = self.linear.weight.detach().cpu().reshape(-1)
        b = self.linear.bias.detach().cpu().reshape(-1)[0]
        return LinearProbe(weight=w, bias=b, d_model=w.numel(), name=name)
