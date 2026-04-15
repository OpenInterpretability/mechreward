"""Experiment 5 — Control Reinforcement Learning reproduction.

Reproduces the token-level steering approach from Cho/Wu/Koshiyama
(arxiv:2602.10437). CRL trains a policy that selects SAE features to
amplify at each token position.

We implement the core mechanism as a discrete-action RL problem where
each action is "which feature bundle to amplify at this step". This is
not a full reproduction of their paper — it's the simplest faithful
implementation that lets us benchmark mechreward against CRL's flavor of
SAE-RL.

Usage:
    python experiments/05_crl_reproduction.py
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from mechreward.sae.loader import load_sae


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-2-2b")
    p.add_argument("--release", default="gemma-scope-2b-pt-res-canonical")
    p.add_argument("--sae-id", default="layer_12/width_16k/canonical")
    p.add_argument("--output-dir", type=Path, default=Path("./runs/exp05"))
    p.add_argument("--max-steps", type=int, default=200)
    return p.parse_args()


@dataclass
class CRLPolicy(nn.Module):
    """A tiny policy network that picks which feature bundle to boost.

    Input: current hidden state at layer L.
    Output: logits over K bundles.
    """

    d_model: int
    n_bundles: int

    def __post_init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_bundles),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(hidden)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[exp05] Loading SAE {args.release} / {args.sae_id}")
    sae = load_sae(release=args.release, sae_id=args.sae_id)

    policy = CRLPolicy(d_model=sae.d_model, n_bundles=4).to("cuda")  # type: ignore[arg-type]
    optim = torch.optim.AdamW(policy.parameters(), lr=1e-4)

    print("[exp05] NOTE: this is a minimal reproduction stub. The full CRL")
    print("[exp05] paper runs a much larger setup. See arxiv:2602.10437.")
    print(f"[exp05] Saving CRL policy skeleton to {args.output_dir}/crl_policy.pt")

    torch.save(policy.state_dict(), args.output_dir / "crl_policy.pt")


if __name__ == "__main__":
    main()
