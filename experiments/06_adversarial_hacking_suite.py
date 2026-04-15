"""Experiment 6 — Adversarial hacking suite.

Runs the canned adversarial prompts through every reward function and
reports hit rates. The goal is to detect when a reward signal is easily
gamed BEFORE starting a full RL run.

Usage:
    python experiments/06_adversarial_hacking_suite.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mechreward.features.catalog import load_pack
from mechreward.hacking.adversarial import AdversarialSuite
from mechreward.reward.feature_reward import FeatureReward, OutcomeReward
from mechreward.sae.loader import load_sae
from mechreward.verifiers import gsm8k_verifier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pack", default="gemma-2-9b/reasoning_pack")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--output", type=Path, default=Path("./runs/exp06/results.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    suite = AdversarialSuite.from_preset("standard")
    print(f"[exp06] Running adversarial suite ({len(suite)} prompts) against all reward fns")

    results: dict[str, dict] = {}

    outcome = OutcomeReward(verifier=gsm8k_verifier, name="outcome_gsm8k")
    print("[exp06]   Testing outcome_gsm8k...")
    results["outcome_gsm8k"] = suite.evaluate_reward(outcome, threshold=args.threshold)

    try:
        pack = load_pack(args.pack)
        sae = load_sae(release=pack.release, sae_id=pack.sae_id)
        fr = FeatureReward(features=pack, sae=sae, aggregation="mean")
        # Feature reward needs hidden states — the suite does not provide them,
        # so we wrap it in a mode that handles graceful degradation for testing.
        print(f"[exp06]   Skipping full FeatureReward eval (needs hidden states)")
        results[pack.name] = {"skipped": "requires hidden_states"}
    except (ImportError, FileNotFoundError) as e:
        print(f"[exp06]   Could not load feature reward: {e}")

    with args.output.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"[exp06] Results written to {args.output}")

    for name, res in results.items():
        if "skipped" in res:
            continue
        print(f"\n  {name}:")
        print(f"    hack_rate:           {res['hack_rate']:.3f}")
        print(f"    false_negative_rate: {res['false_negative_rate']:.3f}")
        if res["hack_rate"] > 0.3:
            print(f"    WARNING: hack rate > 30%. This reward is gameable.")


if __name__ == "__main__":
    main()
