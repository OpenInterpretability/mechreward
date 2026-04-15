"""Train linear probes from labeled examples.

The interface is sklearn-shaped (``fit(X, y)``), backed by torch for fast
GPU training on large hidden-state matrices.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from mechreward.probes.linear_probe import LinearProbe


@dataclass
class ProbeTrainingResult:
    probe: LinearProbe
    train_accuracy: float
    train_loss: float
    epochs: int


def train_linear_probe(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    name: str = "probe",
    epochs: int = 200,
    lr: float = 1e-2,
    weight_decay: float = 1e-3,
    device: str | torch.device | None = None,
    verbose: bool = False,
) -> ProbeTrainingResult:
    """Train a logistic probe on ``(hidden_states, labels)``.

    Args:
        hidden_states: ``[N, d_model]`` activations.
        labels: ``[N]`` with values in ``{0, 1}``.
        name: Identifier for the probe.
        epochs: Full-batch training epochs.
        lr: Learning rate for Adam.
        weight_decay: L2 regularization (acts as a proper probe sparsifier).
        device: Where to train.
        verbose: Print progress.

    Returns:
        A ``ProbeTrainingResult`` containing a ready-to-use LinearProbe.
    """
    if hidden_states.dim() != 2:
        raise ValueError(
            f"Expected 2D hidden_states [N, d_model], got {hidden_states.shape}"
        )
    if labels.dim() != 1 or labels.shape[0] != hidden_states.shape[0]:
        raise ValueError(
            f"Label shape {labels.shape} does not match hidden_states {hidden_states.shape}"
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    x = hidden_states.to(device=device, dtype=torch.float32)
    y = labels.to(device=device, dtype=torch.float32)

    d_model = x.shape[-1]
    model = nn.Linear(d_model, 1).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    last_loss = 0.0
    for epoch in range(epochs):
        optim.zero_grad()
        logits = model(x).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        optim.step()
        last_loss = float(loss.item())

        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            acc = ((logits > 0).float() == y).float().mean().item()
            print(f"  probe epoch {epoch + 1}/{epochs}: loss={last_loss:.4f} acc={acc:.3f}")

    with torch.inference_mode():
        logits = model(x).squeeze(-1)
        acc = ((logits > 0).float() == y).float().mean().item()

    weight = model.weight.detach().cpu().reshape(-1)
    bias = model.bias.detach().cpu().reshape(-1)[0]

    probe = LinearProbe(
        weight=weight,
        bias=bias,
        d_model=d_model,
        name=name,
        metadata={
            "train_accuracy": float(acc),
            "train_loss": float(last_loss),
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "n_samples": int(x.shape[0]),
        },
    )
    return ProbeTrainingResult(
        probe=probe,
        train_accuracy=float(acc),
        train_loss=float(last_loss),
        epochs=epochs,
    )
