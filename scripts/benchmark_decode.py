#!/usr/bin/env python3
"""Compare exact cached decode with full-prefix recomputation on reference PyTorch."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from minillm.config import MiniLLMConfig
from minillm.model import MiniLLM


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--recurrences", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output")
    args = parser.parse_args()

    config = MiniLLMConfig.load(args.config)
    total = args.prompt_tokens + args.decode_tokens
    if min(args.prompt_tokens, args.decode_tokens, args.repeats) < 1:
        raise ValueError("benchmark counts must be positive")
    if total > config.max_seq_len:
        raise ValueError("benchmark sequence exceeds model context")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    model = MiniLLM(config).to(device).eval()
    if not model.supports_cached_decode:
        raise ValueError(
            "configuration contains a component without exact cache support"
        )
    token_ids = torch.randint(
        0, config.vocab_size, (1, total), device=device, dtype=torch.long
    )

    cached_times: list[float] = []
    full_times: list[float] = []
    with torch.inference_mode():
        # Warm both paths before timing.
        _, cache = model.forward_cached(
            token_ids[:, : args.prompt_tokens],
            core_repetitions=args.recurrences,
        )
        model(token_ids[:, : args.prompt_tokens], core_repetitions=args.recurrences)
        for _ in range(args.repeats):
            synchronize(device)
            started = time.perf_counter()
            _, cache = model.forward_cached(
                token_ids[:, : args.prompt_tokens],
                core_repetitions=args.recurrences,
            )
            for position in range(args.prompt_tokens, total):
                _, cache = model.forward_cached(
                    token_ids[:, position : position + 1],
                    cache,
                    core_repetitions=args.recurrences,
                )
            synchronize(device)
            cached_times.append(time.perf_counter() - started)

            synchronize(device)
            started = time.perf_counter()
            for position in range(args.prompt_tokens, total):
                model(
                    token_ids[:, : position + 1],
                    core_repetitions=args.recurrences,
                )
            synchronize(device)
            full_times.append(time.perf_counter() - started)

    cached_seconds = sorted(cached_times)[len(cached_times) // 2]
    full_seconds = sorted(full_times)[len(full_times) // 2]
    report = {
        "config": args.config,
        "device": str(device),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "prompt_tokens": args.prompt_tokens,
        "decode_tokens": args.decode_tokens,
        "repeats": args.repeats,
        "cached_total_seconds_including_prefill": cached_seconds,
        "full_prefix_total_seconds": full_seconds,
        "generated_tokens_per_second_cached": args.decode_tokens / cached_seconds,
        "generated_tokens_per_second_full_prefix": args.decode_tokens / full_seconds,
        "speedup": full_seconds / cached_seconds,
        "note": "Reference PyTorch benchmark; includes cached prefill but full-prefix timing starts after prompt construction.",
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
