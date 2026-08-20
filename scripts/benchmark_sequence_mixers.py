#!/usr/bin/env python3
"""Measure small-scale time/state scaling of attention, convolution, and GDN2."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from pathlib import Path

import torch

from minillm.modules import (
    CausalSelfAttention,
    GatedShortConv,
    ReferenceGatedDeltaNet2,
)


def parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def scaling_exponent(lengths: list[int], seconds: list[float]) -> float:
    x = [math.log(value) for value in lengths]
    y = [math.log(value) for value in seconds]
    x_mean = statistics.mean(x)
    y_mean = statistics.mean(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    denominator = sum((a - x_mean) ** 2 for a in x)
    return numerator / denominator


def state_bytes(
    mixer: str,
    *,
    batch_size: int,
    sequence_length: int,
    d_model: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    conv_kernel: int,
    element_bytes: int = 4,
) -> int:
    if mixer == "attention":
        return batch_size * sequence_length * 2 * n_kv_heads * head_dim * element_bytes
    if mixer == "conv":
        return batch_size * (conv_kernel - 1) * d_model * element_bytes
    if mixer == "gdn2":
        return batch_size * n_heads * head_dim * head_dim * element_bytes
    raise ValueError(mixer)


def mixer_work_units(
    mixer: str,
    *,
    batch_size: int,
    sequence_length: int,
    d_model: int,
    n_heads: int,
    head_dim: int,
    conv_kernel: int,
) -> int:
    """Operator-only work proxy, excluding dense input/output projections."""

    if mixer == "attention":
        return 4 * batch_size * n_heads * sequence_length**2 * head_dim
    if mixer == "conv":
        return batch_size * sequence_length * d_model * conv_kernel
    if mixer == "gdn2":
        return 6 * batch_size * sequence_length * n_heads * head_dim**2
    raise ValueError(mixer)


def benchmark_module(
    module: torch.nn.Module,
    *,
    lengths: list[int],
    batch_size: int,
    d_model: int,
    warmup: int,
    repeats: int,
) -> list[dict[str, float | int]]:
    rows = []
    module.train()
    for length in lengths:
        sample = torch.randn(batch_size, length, d_model)
        for _ in range(warmup):
            module.zero_grad(set_to_none=True)
            module(sample).square().mean().backward()
        durations = []
        for _ in range(repeats):
            module.zero_grad(set_to_none=True)
            started = time.perf_counter()
            output = module(sample)
            output.square().mean().backward()
            durations.append(time.perf_counter() - started)
        median = statistics.median(durations)
        rows.append(
            {
                "sequence_length": length,
                "median_forward_backward_seconds": median,
                "tokens_per_second": batch_size * length / median,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lengths", nargs="+", type=int, default=[16, 32, 64, 128, 256]
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--conv-kernel", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", default="results/sequence_mixer_benchmark.json")
    args = parser.parse_args()
    if args.d_model != args.n_heads * args.head_dim:
        raise ValueError("d_model must equal n_heads * head_dim")
    if any(length < 2 for length in args.lengths):
        raise ValueError("all sequence lengths must be at least two")
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    modules = {
        "attention": CausalSelfAttention(
            args.d_model,
            args.n_heads,
            args.n_kv_heads,
            args.head_dim,
            rope_base=50_000.0,
            norm_eps=1e-6,
        ),
        "conv": GatedShortConv(args.d_model, args.conv_kernel),
        "gdn2-reference": ReferenceGatedDeltaNet2(
            args.d_model, args.n_heads, args.head_dim
        ),
    }
    results = []
    for name, module in modules.items():
        rows = benchmark_module(
            module,
            lengths=args.lengths,
            batch_size=args.batch_size,
            d_model=args.d_model,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        seconds = [float(row["median_forward_backward_seconds"]) for row in rows]
        state_name = "gdn2" if name == "gdn2-reference" else name
        for row in rows:
            length = int(row["sequence_length"])
            row["decode_state_bytes_fp32"] = state_bytes(
                state_name,
                batch_size=args.batch_size,
                sequence_length=length,
                d_model=args.d_model,
                n_heads=args.n_heads,
                n_kv_heads=args.n_kv_heads,
                head_dim=args.head_dim,
                conv_kernel=args.conv_kernel,
            )
            row["operator_work_units"] = mixer_work_units(
                state_name,
                batch_size=args.batch_size,
                sequence_length=length,
                d_model=args.d_model,
                n_heads=args.n_heads,
                head_dim=args.head_dim,
                conv_kernel=args.conv_kernel,
            )
        results.append(
            {
                "mixer": name,
                "parameters": parameter_count(module),
                "theoretical_sequence_complexity": (
                    "O(T^2 * H * D)"
                    if name == "attention"
                    else "O(T * D * K)"
                    if name == "conv"
                    else "O(T * H * D_head^2)"
                ),
                "empirical_time_scaling_exponent": scaling_exponent(
                    args.lengths, seconds
                ),
                "measurements": rows,
                "implementation_note": (
                    "sequential correctness reference; requires chunkwise fused kernel"
                    if name == "gdn2-reference"
                    else "PyTorch reference primitives"
                ),
            }
        )
    attention_parameters = parameter_count(modules["attention"])
    for result in results:
        result["parameter_ratio_vs_attention"] = (
            int(result["parameters"]) / attention_parameters
        )
    payload = {
        "schema_version": 1,
        "experiment": "sequence-mixer-scaling-v1",
        "warning": "CPU microbenchmark; relative algorithm/runtime diagnostic, not phone latency.",
        "batch_size": args.batch_size,
        "d_model": args.d_model,
        "lengths": args.lengths,
        "threads": args.threads,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
