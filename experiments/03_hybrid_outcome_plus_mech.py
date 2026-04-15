"""Experiment 3 — Hybrid outcome + mechreward (the commercially relevant one).

Uses HERO-style stratified normalization: mechreward is normalized within
groups defined by outcome correctness, preventing one signal from
drowning the other.

Usage:
    python experiments/03_hybrid_outcome_plus_mech.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

from mechreward.features.catalog import load_pack
from mechreward.reward.composition import CompositeReward
from mechreward.reward.feature_reward import FeatureReward, OutcomeReward
from mechreward.sae.loader import load_sae
from mechreward.verifiers import gsm8k_verifier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-2-9b")
    p.add_argument("--pack", default="gemma-2-9b/reasoning_pack")
    p.add_argument("--output-dir", type=Path, default=Path("./runs/exp03"))
    p.add_argument("--mech-weight", type=float, default=0.3)
    p.add_argument("--outcome-weight", type=float, default=1.0)
    p.add_argument("--mode", choices=["sum", "stratified"], default="stratified")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--num-generations", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pack = load_pack(args.pack)
    sae = load_sae(release=pack.release, sae_id=pack.sae_id)

    feature_reward = FeatureReward(
        features=pack,
        sae=sae,
        aggregation="mean_last_32_tokens",
        normalization="zscore",
        name="feature_mech",
    )
    outcome_reward = OutcomeReward(verifier=gsm8k_verifier, name="outcome_gsm8k")

    composite = CompositeReward(
        rewards=[outcome_reward, feature_reward],
        weights=[args.outcome_weight, args.mech_weight],
        mode=args.mode,
    )

    from datasets import load_dataset

    dataset = load_dataset("gsm8k", "main", split="train")
    prepared = dataset.map(lambda ex: {"prompt": ex["question"]})

    try:
        from trl import GRPOConfig

        from mechreward.integrations.trl_grpo import MechRewardGRPOTrainer
    except ImportError:
        print("Install with `pip install 'mechreward[trl,sae]'`")
        return

    config = GRPOConfig(
        output_dir=str(args.output_dir),
        num_generations=args.num_generations,
        max_steps=args.max_steps,
        logging_steps=10,
        bf16=True,
        gradient_checkpointing=True,
    )

    trainer = MechRewardGRPOTrainer(
        model=args.model,
        reward_funcs=[composite],
        layer_idx=sae.layer,
        args=config,
        train_dataset=prepared,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir / "final"))
    print(f"[exp03] Done. Checkpoint at {args.output_dir / 'final'}")


if __name__ == "__main__":
    main()
