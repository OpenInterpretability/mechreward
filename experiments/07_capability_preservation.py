"""Experiment 7 — Capability preservation check.

Measures MMLU, HellaSwag, ARC-Challenge, TruthfulQA BEFORE and AFTER each
RL run. Catastrophic forgetting is the most reported failure mode of RL
fine-tuning; this script gives a clean pre/post delta per capability.

Usage:
    python experiments/07_capability_preservation.py \
        --base-model google/gemma-2-9b \
        --trained-model ./runs/exp03/final
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="google/gemma-2-9b")
    p.add_argument("--trained-model", required=True, type=Path)
    p.add_argument(
        "--tasks",
        default="mmlu,hellaswag,arc_challenge,truthfulqa_mc2",
        help="Comma-separated lm-eval task names",
    )
    p.add_argument("--output", type=Path, default=Path("./runs/exp07/delta.json"))
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args()


def run_lm_eval(model_path: str, tasks: list[str], batch_size: int) -> dict[str, float]:
    """Run lm-evaluation-harness and return task → score dict."""
    try:
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "results.json"
            cmd = [
                "lm_eval",
                "--model",
                "hf",
                "--model_args",
                f"pretrained={model_path},dtype=bfloat16",
                "--tasks",
                ",".join(tasks),
                "--batch_size",
                str(batch_size),
                "--output_path",
                str(out_path),
            ]
            subprocess.run(cmd, check=True)
            # lm-eval produces a nested JSON
            files = list(Path(tmp).rglob("*.json"))
            if not files:
                return {}
            with files[0].open() as f:
                data = json.load(f)

        scores = {}
        for task, result in data.get("results", {}).items():
            for metric_name, value in result.items():
                if isinstance(value, (int, float)):
                    scores[f"{task}_{metric_name}"] = float(value)
        return scores
    except Exception as e:
        print(f"[exp07] lm-eval run failed for {model_path}: {e}")
        return {}


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    print(f"[exp07] Running lm-eval on base model: {args.base_model}")
    base_scores = run_lm_eval(args.base_model, tasks, args.batch_size)
    print(f"[exp07] Running lm-eval on trained model: {args.trained_model}")
    trained_scores = run_lm_eval(str(args.trained_model), tasks, args.batch_size)

    delta = {}
    for key in set(base_scores) | set(trained_scores):
        b = base_scores.get(key, 0.0)
        t = trained_scores.get(key, 0.0)
        delta[key] = {"base": b, "trained": t, "delta": t - b}

    with args.output.open("w") as f:
        json.dump(delta, f, indent=2)

    print(f"\n[exp07] Summary:")
    for key, v in sorted(delta.items()):
        marker = "✓" if v["delta"] >= -0.02 else "✗"
        print(f"  {marker} {key:<40} base={v['base']:.3f} trained={v['trained']:.3f} Δ={v['delta']:+.3f}")


if __name__ == "__main__":
    main()
