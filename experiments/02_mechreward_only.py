"""Experiment 2 — Mechreward alone, no outcome verifier.

The tensest hypothesis: can feature activations alone drive reasoning
improvements? If this works, it's the biggest finding of the project.
If it doesn't, Exp 3 (hybrid) still has a path.

Usage:
    python experiments/02_mechreward_only.py --model google/gemma-2-9b
"""
from __future__ import annotations

import argparse
from pathlib import Path

from mechreward.features.catalog import load_pack
from mechreward.reward.feature_reward import FeatureReward
from mechreward.sae.loader import load_sae


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-2-9b")
    p.add_argument(
        "--pack",
        default="gemma-2-9b/reasoning_pack",
        help="Feature pack to use as reward",
    )
    p.add_argument("--output-dir", type=Path, default=Path("./runs/exp02"))
    p.add_argument("--aggregation", default="mean_last_32_tokens")
    p.add_argument("--normalization", default="zscore")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-6)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[exp02] Loading feature pack: {args.pack}")
    pack = load_pack(args.pack)

    print(f"[exp02] Loading SAE: {pack.release} / {pack.sae_id}")
    sae = load_sae(release=pack.release, sae_id=pack.sae_id)

    reward = FeatureReward(
        features=pack,
        sae=sae,
        aggregation=args.aggregation,
        normalization=args.normalization,
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
        per_device_train_batch_size=args.batch_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=100,
        bf16=True,
        gradient_checkpointing=True,
    )

    trainer = MechRewardGRPOTrainer(
        model=args.model,
        reward_funcs=[reward],
        layer_idx=sae.layer,
        args=config,
        train_dataset=prepared,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir / "final"))
    print(f"[exp02] Done. Checkpoint at {args.output_dir / 'final'}")


if __name__ == "__main__":
    main()
