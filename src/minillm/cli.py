"""Command-line entry points for measurement and smoke tests."""

from __future__ import annotations

import argparse
import hashlib
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


def _generate(args: argparse.Namespace) -> int:
    # Keep static architecture analysis usable without importing torch/tokenizers.
    from .aira import load_compact_shelf
    from .generation import (
        SamplingConfig,
        TriggerConfig,
        generate_ids,
        generate_triggered_ids,
        load_model_checkpoint,
    )
    from .tokenization import load_tokenizer

    prompt = (
        Path(args.prompt_file).read_text(encoding="utf-8")
        if args.prompt_file is not None
        else args.prompt
    )
    loaded = load_model_checkpoint(
        args.checkpoint, config_path=args.config, device=args.device
    )
    tokenizer = load_tokenizer(args.tokenizer)
    if tokenizer.get_vocab_size() != loaded.config.vocab_size:
        raise ValueError("tokenizer vocabulary does not match model configuration")
    prompt_ids = tokenizer.encode(prompt).ids
    bos_id = tokenizer.token_to_id("<bos>")
    if bos_id is not None and not args.no_bos:
        prompt_ids.insert(0, bos_id)
    eos_id = tokenizer.token_to_id("<eos>")
    stop_tokens = {eos_id} if eos_id is not None and not args.no_eos_stop else set()
    sampling = SamplingConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        use_cache=not args.no_cache,
    )
    if args.aira_shelf is None:
        result = generate_ids(
            loaded.model,
            prompt_ids,
            sampling,
            stop_token_ids=stop_tokens,
            core_repetitions=args.recurrences,
        )
    else:
        with Path(args.tokenizer).open("rb") as handle:
            tokenizer_hash = hashlib.file_digest(handle, "sha256").hexdigest()
        result = generate_triggered_ids(
            loaded.model,
            load_compact_shelf(
                args.aira_shelf, expected_tokenizer_sha256=tokenizer_hash
            ),
            prompt_ids,
            sampling,
            TriggerConfig(
                minimum_support=args.aira_min_support,
                confidence_threshold=args.aira_confidence,
                confidence_z=args.aira_confidence_z,
                selection=args.aira_selection,
                maximum_shelf_burst=args.aira_max_burst,
                cumulative_risk_budget=args.aira_risk_budget,
                neural_anchor_interval=args.aira_anchor_interval,
            ),
            stop_token_ids=stop_tokens,
            core_repetitions=args.recurrences,
        )
    completion = tokenizer.decode(
        list(result.generated_token_ids), skip_special_tokens=True
    )
    if args.json:
        print(
            json.dumps(
                {
                    "prompt": prompt,
                    "completion": completion,
                    "prompt_token_ids": result.prompt_token_ids,
                    "generated_token_ids": result.generated_token_ids,
                    "stop_reason": result.stop_reason,
                    "used_cache": result.used_cache,
                    "checkpoint_step": loaded.step,
                    "aira": {
                        "shelf_tokens": result.shelf_tokens,
                        "neural_tokens": result.neural_tokens,
                        "neural_calls": result.neural_calls,
                        "neural_input_tokens": result.neural_input_tokens,
                        "control_rejections": result.control_rejections,
                        "routes": result.routes,
                    }
                    if args.aira_shelf is not None
                    else None,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(completion)
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

    generate = subparsers.add_parser(
        "generate", help="generate text from a trusted checkpoint"
    )
    generate.add_argument("checkpoint")
    generate.add_argument("--tokenizer", required=True)
    generate.add_argument("--config")
    prompt = generate.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt")
    prompt.add_argument("--prompt-file")
    generate.add_argument("--max-new-tokens", type=int, default=64)
    generate.add_argument("--temperature", type=float, default=0.0)
    generate.add_argument("--top-k", type=int, default=0)
    generate.add_argument("--top-p", type=float, default=1.0)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--recurrences", type=int)
    generate.add_argument("--device", default="cpu")
    generate.add_argument("--no-cache", action="store_true")
    generate.add_argument(
        "--aira-shelf", help="NPZ shelf archive for true neural-bypass generation"
    )
    generate.add_argument("--aira-min-support", type=int, default=5)
    generate.add_argument("--aira-confidence", type=float, default=0.95)
    generate.add_argument("--aira-confidence-z", type=float, default=1.96)
    generate.add_argument(
        "--aira-selection", choices=("longest", "confidence"), default="longest"
    )
    generate.add_argument("--aira-max-burst", type=int, default=4)
    generate.add_argument("--aira-risk-budget", type=float, default=0.10)
    generate.add_argument("--aira-anchor-interval", type=int, default=8)
    generate.add_argument("--no-bos", action="store_true")
    generate.add_argument("--no-eos-stop", action="store_true")
    generate.add_argument("--json", action="store_true")
    generate.set_defaults(func=_generate)

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
