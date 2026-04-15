"""mechreward CLI: inspect SAEs, validate features, run red team."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_info(args: argparse.Namespace) -> int:
    from mechreward import __version__

    print(f"mechreward {__version__}")
    print("Submodules:")
    for name in ("sae", "features", "reward", "hacking", "probes", "rollout", "integrations"):
        print(f"  - mechreward.{name}")
    return 0


def _cmd_list_packs(args: argparse.Namespace) -> int:
    catalogs = Path(__file__).resolve().parents[2] / "catalogs"
    if not catalogs.exists():
        print(f"No catalogs directory found at {catalogs}")
        return 1
    for p in sorted(catalogs.rglob("*.json")):
        rel = p.relative_to(catalogs).with_suffix("")
        print(rel)
    return 0


def _cmd_inspect_pack(args: argparse.Namespace) -> int:
    from mechreward.features.catalog import load_pack

    pack = load_pack(args.name)
    print(f"Pack: {pack.name}")
    print(f"Version: {pack.version}")
    print(f"Model: {pack.model_name}")
    print(f"SAE: {pack.release} / {pack.sae_id}")
    print(f"Features ({len(pack.features)}):")
    for f in pack.features:
        valid = "✓" if f.validated else "?"
        print(f"  [{valid}] {f.feature_id:>6}  w={f.weight:+.2f}  {f.name}")
        if f.description:
            print(f"          {f.description}")
    return 0


def _cmd_adversarial(args: argparse.Namespace) -> int:
    from mechreward.hacking.adversarial import AdversarialSuite

    suite = AdversarialSuite.from_preset(args.preset)
    print(f"Adversarial suite '{args.preset}' ({len(suite)} prompts):")
    for p in suite.prompts:
        marker = "HACK" if p.expected_hack else "CLEAN"
        print(f"  [{marker}] {p.name:<25} target={p.target}")
        print(f"          {p.description}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mechreward", description="mechreward CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_info = sub.add_parser("info", help="Show version and module info")
    p_info.set_defaults(func=_cmd_info)

    p_list = sub.add_parser("list-packs", help="List bundled feature packs")
    p_list.set_defaults(func=_cmd_list_packs)

    p_inspect = sub.add_parser("inspect-pack", help="Inspect a feature pack")
    p_inspect.add_argument("name", help="Pack name, e.g. gemma-2-9b/reasoning_pack")
    p_inspect.set_defaults(func=_cmd_inspect_pack)

    p_adv = sub.add_parser("adversarial", help="List adversarial prompts in a preset")
    p_adv.add_argument("--preset", default="standard")
    p_adv.set_defaults(func=_cmd_adversarial)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
