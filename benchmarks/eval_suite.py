"""Unified evaluation harness: GSM8K, MATH, HumanEval, MBPP, MMLU.

Thin wrapper over ``lm-evaluation-harness`` that writes a summary JSON.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


DEFAULT_TASKS = [
    "gsm8k",
    "hendrycks_math",
    "humaneval",
    "mbpp",
    "mmlu",
    "hellaswag",
]


def run_tasks(model_path: str, tasks: list[str], batch_size: int = 8) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "results.json"
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
            str(out),
        ]
        subprocess.run(cmd, check=True)
        files = list(Path(tmp).rglob("*.json"))
        with files[0].open() as f:
            return json.load(f)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    p.add_argument("--output", type=Path, default=Path("./bench/eval_suite.json"))
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = run_tasks(args.model, args.tasks, args.batch_size)
    with args.output.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
