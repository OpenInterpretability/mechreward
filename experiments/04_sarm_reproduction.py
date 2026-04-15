"""Experiment 4 — SARM reproduction (offline RLHF reward model via SAE).

Reproduces the core idea of Liu et al. (arxiv:2508.08746, AAAI 26 Oral).
SARM trains a linear head on top of SAE feature activations using a
preference dataset, then uses that reward model in offline RLHF.

This experiment exists so we can do a head-to-head comparison against
mechreward in the paper. The full SARM pipeline is at
github.com/schrieffer-z/sarm — we reproduce the essential path here in
~150 lines so the comparison uses the same SAE and model stack.

Usage:
    python experiments/04_sarm_reproduction.py \
        --preference-dataset HuggingFaceH4/ultrafeedback_binarized
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from mechreward.sae.batched_encode import batched_encode
from mechreward.sae.loader import load_sae


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-2-9b")
    p.add_argument("--release", default="gemma-scope-9b-pt-res-canonical")
    p.add_argument("--sae-id", default="layer_22/width_16k/canonical")
    p.add_argument(
        "--preference-dataset",
        default="HuggingFaceH4/ultrafeedback_binarized",
        help="HF dataset with (prompt, chosen, rejected) triples.",
    )
    p.add_argument("--output-dir", type=Path, default=Path("./runs/exp04"))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--n-samples", type=int, default=5000)
    return p.parse_args()


class SARMReward(nn.Module):
    """A linear head over SAE features for reward modeling."""

    def __init__(self, d_sae: int) -> None:
        super().__init__()
        self.head = nn.Linear(d_sae, 1)

    def forward(self, feature_acts: torch.Tensor) -> torch.Tensor:
        # feature_acts: [B, T, d_sae] → mean over T → [B, d_sae] → [B]
        pooled = feature_acts.mean(dim=1)
        return self.head(pooled).squeeze(-1)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[exp04] Loading SAE {args.release} / {args.sae_id}")
    sae = load_sae(release=args.release, sae_id=args.sae_id)

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    ds = load_dataset(args.preference_dataset, split="train_prefs").select(
        range(args.n_samples)
    )

    def encode_text(texts: list[str]) -> torch.Tensor:
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
        with torch.inference_mode():
            out = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )
        hs = out.hidden_states[sae.layer]  # [B, T, d_model]
        return batched_encode(sae, hs.float())  # [B, T, d_sae]

    reward_model = SARMReward(sae.d_sae).to("cuda")
    optim = torch.optim.AdamW(reward_model.parameters(), lr=args.lr)

    print(f"[exp04] Training SARM head on {len(ds)} preference pairs")
    for epoch in range(args.epochs):
        total_loss = 0.0
        correct = 0
        count = 0
        for i in range(0, len(ds), args.batch_size):
            batch = ds[i : i + args.batch_size]
            chosen = [b for b in batch["chosen"]]  # type: ignore[index]
            rejected = [b for b in batch["rejected"]]  # type: ignore[index]

            chosen_feats = encode_text([c[-1]["content"] if isinstance(c, list) else c for c in chosen])
            rejected_feats = encode_text([c[-1]["content"] if isinstance(c, list) else c for c in rejected])

            chosen_reward = reward_model(chosen_feats)
            rejected_reward = reward_model(rejected_feats)

            # Bradley-Terry loss
            loss = -torch.nn.functional.logsigmoid(chosen_reward - rejected_reward).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += float(loss.item())
            correct += int((chosen_reward > rejected_reward).sum().item())
            count += chosen_reward.numel()

        print(
            f"[exp04] epoch {epoch + 1}/{args.epochs}: "
            f"loss={total_loss / max(1, count // args.batch_size):.4f} "
            f"acc={correct / max(1, count):.3f}"
        )

    torch.save(reward_model.state_dict(), args.output_dir / "sarm_head.pt")
    print(f"[exp04] Saved SARM head to {args.output_dir}/sarm_head.pt")


if __name__ == "__main__":
    main()
