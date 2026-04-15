"""Experiment 1 — Baseline: outcome reward only, no mechreward.

This is the control condition. GRPO on GSM8K using a pure sympy-style
numeric verifier. All subsequent experiments measure deltas against this
baseline.

Usage:
    python experiments/01_baseline_outcome_only.py \
        --model google/gemma-2-9b \
        --dataset gsm8k \
        --output-dir ./runs/exp01
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mechreward.reward.feature_reward import OutcomeReward
from mechreward.verifiers import gsm8k_verifier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-2-9b")
    p.add_argument("--dataset", default="gsm8k")
    p.add_argument("--output-dir", type=Path, default=Path("./runs/exp01"))
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-6)
    return p.parse_args()


def load_dataset(name: str):
    from datasets import load_dataset as hf_load

    if name == "gsm8k":
        return hf_load("gsm8k", "main", split="train")
    raise ValueError(f"Unknown dataset: {name}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[exp01] Loading model: {args.model}")
    print(f"[exp01] Loading dataset: {args.dataset}")

    dataset = load_dataset(args.dataset)

    # Convert to GRPO-style: prompts include the gold answer inside "#### N"
    # (keep it for the verifier's regex, hide from the model by instruction).
    def _prep(ex: dict) -> dict:
        return {
            "prompt": f"Solve this problem. Put your final answer after 'Answer:'.\n\n{ex['question']}",
            "gold": ex["answer"],
        }

    prepared = dataset.map(_prep)

    outcome = OutcomeReward(
        verifier=lambda prompt, completion: gsm8k_verifier(
            # Put the gold back into the prompt for the verifier only
            prompt + "\n####" + prepared[0]["gold"].split("####")[-1],
            completion,
        ),
        name="gsm8k_outcome",
    )

    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError:
        print("Install with `pip install 'mechreward[trl]'`")
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

    trainer = GRPOTrainer(
        model=args.model,
        args=config,
        train_dataset=prepared,
        reward_funcs=outcome,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir / "final"))

    with (args.output_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    print(f"[exp01] Done. Checkpoint at {args.output_dir / 'final'}")


if __name__ == "__main__":
    main()
