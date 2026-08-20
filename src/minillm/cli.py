"""Command-line entry points for measurement and smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import human_count, profile_model, render_profile
from .config import MiniLLMConfig
from .energy import HardwareEnergyProfile, estimate_decode_energy


def _analyze(args: argparse.Namespace) -> int:
    config = MiniLLMConfig.load(args.config)
    profile = profile_model(
        config,
        context_length=args.context,
        weight_bits=args.weight_bits,
        kv_bits=args.kv_bits,
        core_repetitions=args.recurrences,
    )
    if args.json:
        print(json.dumps(profile.to_dict(), indent=2))
    else:
        print(f"Configuration: {args.config}")
        print(render_profile(profile))
    return 0


def _compare(args: argparse.Namespace) -> int:
    paths = [Path(item) for item in args.configs]
    print(
        f"{'config':32} {'stored':>10} {'active/tok':>12} {'Q4 weights':>12} "
        f"{'KV':>10} {'depth':>8}"
    )
    for path in paths:
        config = MiniLLMConfig.load(path)
        profile = profile_model(config, context_length=args.context)
        print(
            f"{path.stem[:32]:32} {human_count(profile.unique_parameters):>10} "
            f"{human_count(profile.active_parameter_applications_per_token):>12} "
            f"{(human_count(profile.weight_memory_bytes, binary=True) + 'B'):>12} "
            f"{(human_count(profile.kv_cache_bytes, binary=True) + 'B'):>10} "
            f"{profile.unique_depth}/{profile.effective_depth:>4}"
        )
    return 0


def _energy(args: argparse.Namespace) -> int:
    config = MiniLLMConfig.load(args.config)
    estimate = estimate_decode_energy(
        config,
        context_length=args.context,
        weight_bits=args.weight_bits,
        kv_bits=args.kv_bits,
        hardware=HardwareEnergyProfile(
            name=args.hardware,
            memory_pj_per_byte=args.memory_pj_per_byte,
            mac_pj=args.mac_pj,
        ),
        core_repetitions=args.recurrences,
    )
    if args.json:
        print(json.dumps(estimate.to_dict(), indent=2))
    else:
        print(f"Configuration: {args.config}")
        print(f"Hardware proxy: {estimate.hardware}")
        print(
            f"Active weight reads/token: {human_count(estimate.active_weight_bytes, binary=True)}B"
        )
        print(
            f"KV reads/token:            {human_count(estimate.kv_read_bytes, binary=True)}B"
        )
        print(
            "State read+write/token:    "
            f"{human_count(estimate.recurrent_state_read_write_bytes, binary=True)}B"
        )
        print(
            f"Weight transport:          {estimate.weight_transport_pj / 1e9:.4f} mJ/token"
        )
        print(
            f"KV transport:              {estimate.kv_transport_pj / 1e9:.4f} mJ/token"
        )
        print(
            "State transport:           "
            f"{estimate.recurrent_state_transport_pj / 1e9:.4f} mJ/token"
        )
        print(f"Arithmetic:                {estimate.compute_pj / 1e9:.4f} mJ/token")
        print(f"Total proxy:               {estimate.total_mj:.4f} mJ/token")
        print(
            "Fermi estimate only; calibrate against device power and wall-clock measurements."
        )
    return 0


def _smoke(args: argparse.Namespace) -> int:
    # Import torch-dependent training only for this subcommand; static analysis remains light.
    from .train import smoke_train

    config = MiniLLMConfig.load(args.config)
    result = smoke_train(config, steps=args.steps)
    print(f"Parameters: {result.parameter_count:,}")
    print("Losses: " + ", ".join(f"{loss:.4f}" for loss in result.losses))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minillm")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser(
        "analyze", help="estimate model size, cache, and decode compute"
    )
    analyze.add_argument("config")
    analyze.add_argument("--context", type=int, default=4096)
    analyze.add_argument("--weight-bits", type=int, default=4)
    analyze.add_argument("--kv-bits", type=int, default=8)
    analyze.add_argument("--recurrences", type=int)
    analyze.add_argument("--json", action="store_true")
    analyze.set_defaults(func=_analyze)

    compare = subparsers.add_parser(
        "compare", help="compare architecture configurations"
    )
    compare.add_argument("configs", nargs="+")
    compare.add_argument("--context", type=int, default=4096)
    compare.set_defaults(func=_compare)

    energy = subparsers.add_parser(
        "energy", help="estimate batch-1 decode energy from active bytes"
    )
    energy.add_argument("config")
    energy.add_argument("--context", type=int, default=4096)
    energy.add_argument("--weight-bits", type=int, default=4)
    energy.add_argument("--kv-bits", type=int, default=8)
    energy.add_argument("--recurrences", type=int)
    energy.add_argument("--hardware", default="phone-lpddr-proxy")
    energy.add_argument("--memory-pj-per-byte", type=float, default=60.0)
    energy.add_argument("--mac-pj", type=float, default=0.5)
    energy.add_argument("--json", action="store_true")
    energy.set_defaults(func=_energy)

    smoke = subparsers.add_parser(
        "smoke-train", help="run a tiny end-to-end optimizer test"
    )
    smoke.add_argument("config", nargs="?", default="configs/toy.json")
    smoke.add_argument("--steps", type=int, default=8)
    smoke.set_defaults(func=_smoke)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
