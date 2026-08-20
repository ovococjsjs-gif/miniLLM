#!/usr/bin/env python3
"""Finite-inference gradient-alignment comparison: old PC versus PC-ALM."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from minillm.aira import (
    backprop_gradients,
    gradient_cosine,
    local_augmented_gradients,
    minimum_layer_cosine,
)


def make_weights(depth: int, width: int, seed: int) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    return tuple(
        (torch.randn(width, width) / width**0.5).requires_grad_() for _ in range(depth)
    )


def run_case(
    *,
    depth: int,
    width: int,
    inference_steps: int,
    seed: int,
    state_step_size: float,
    dual_rate: float,
) -> dict[str, float | int]:
    model = make_weights(depth, width, seed)
    inputs = torch.randn(1, width)
    targets = torch.randn(1, width)
    bp = backprop_gradients(model, inputs, targets, activation="tanh")
    started = time.perf_counter()
    pc = local_augmented_gradients(
        model,
        inputs,
        targets,
        inference_steps=inference_steps,
        state_step_size=state_step_size,
        dual_rate=0.0,
        activation="tanh",
    )
    pc_seconds = time.perf_counter() - started
    started = time.perf_counter()
    alm = local_augmented_gradients(
        model,
        inputs,
        targets,
        inference_steps=inference_steps,
        state_step_size=state_step_size,
        dual_rate=dual_rate,
        activation="tanh",
    )
    alm_seconds = time.perf_counter() - started
    return {
        "depth": depth,
        "width": width,
        "inference_steps": inference_steps,
        "seed": seed,
        "pc_global_cosine": gradient_cosine(pc.gradients, bp),
        "pc_minimum_layer_cosine": minimum_layer_cosine(pc.gradients, bp),
        "pc_residual_norm": pc.residual_norm,
        "pc_seconds": pc_seconds,
        "pc_alm_global_cosine": gradient_cosine(alm.gradients, bp),
        "pc_alm_minimum_layer_cosine": minimum_layer_cosine(alm.gradients, bp),
        "pc_alm_residual_norm": alm.residual_norm,
        "pc_alm_seconds": alm_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depths", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--step-factors", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43])
    parser.add_argument("--state-step-size", type=float, default=0.2)
    parser.add_argument("--dual-rate", type=float, default=0.5)
    parser.add_argument("--output", default="results/pc_alm_proxy.json")
    args = parser.parse_args()
    rows = [
        run_case(
            depth=depth,
            width=args.width,
            inference_steps=depth * factor,
            seed=seed,
            state_step_size=args.state_step_size,
            dual_rate=args.dual_rate,
        )
        for depth in args.depths
        for factor in args.step_factors
        for seed in args.seeds
    ]
    summary = []
    for depth in args.depths:
        for factor in args.step_factors:
            selected = [
                row
                for row in rows
                if row["depth"] == depth and row["inference_steps"] == depth * factor
            ]
            summary.append(
                {
                    "depth": depth,
                    "step_factor": factor,
                    "inference_steps": depth * factor,
                    "pc_global_cosine": statistics.mean(
                        float(row["pc_global_cosine"]) for row in selected
                    ),
                    "pc_alm_global_cosine": statistics.mean(
                        float(row["pc_alm_global_cosine"]) for row in selected
                    ),
                    "pc_alm_minimum_layer_cosine": statistics.mean(
                        float(row["pc_alm_minimum_layer_cosine"]) for row in selected
                    ),
                    "pc_alm_over_pc_time": statistics.mean(
                        float(row["pc_alm_seconds"]) / float(row["pc_seconds"])
                        for row in selected
                    ),
                }
            )
    payload = {
        "schema_version": 1,
        "experiment": "aira-v2-pc-alm-gradient-alignment-v1",
        "warning": "Tiny tanh MLP reference using autograd for activity gradients; not a local runtime or language-quality result.",
        "paper": "https://arxiv.org/html/2605.31022v1",
        "configuration": {
            "width": args.width,
            "depths": args.depths,
            "step_factors": args.step_factors,
            "state_step_size": args.state_step_size,
            "dual_rate": args.dual_rate,
            "seeds": args.seeds,
        },
        "platform": platform.platform(),
        "torch": torch.__version__,
        "summary": summary,
        "runs": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
