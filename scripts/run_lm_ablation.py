#!/usr/bin/env python3
"""Matched real-text proxy comparison of attention-only and conv/GQA models."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from minillm.config import MiniLLMConfig
from minillm.training import TrainConfig, train_proxy

VARIANTS = {
    "conv_gqa": "configs/proxy_3m.json",
    "attention_only": "configs/proxy_attention_3m.json",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", default="data/tokens-4096")
    parser.add_argument(
        "--steps", type=int, default=300, choices=range(1, 301), metavar="[1-300]"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 314, 2718])
    parser.add_argument("--runs", default="runs/lm-ablation")
    parser.add_argument("--output", default="results/lm_ablation.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    rows = []
    for name, config_path in VARIANTS.items():
        model = MiniLLMConfig.load(config_path)
        for seed in args.seeds:
            training = TrainConfig(
                steps=args.steps,
                batch_size=4,
                sequence_length=128,
                learning_rate=5e-4,
                warmup_steps=30,
                eval_interval=50,
                eval_batches=20,
                checkpoint_interval=args.steps,
                seed=seed,
            )
            summary = train_proxy(
                model,
                training,
                train_tokens=Path(args.tokens) / "train.bin",
                validation_tokens=Path(args.tokens) / "validation.bin",
                output_directory=Path(args.runs) / f"{name}-seed{seed}",
            )
            rows.append(
                {"variant": name, "seed": seed, "config": config_path, **summary}
            )
    aggregate = []
    for name in VARIANTS:
        selected = [row for row in rows if row["variant"] == name]
        losses = [row["best_validation_main_loss"] for row in selected]
        times = [row["wall_seconds"] for row in selected]
        aggregate.append(
            {
                "variant": name,
                "runs": len(selected),
                "parameters": selected[0]["parameters"],
                "mean_best_validation_main_loss": statistics.mean(losses),
                "std_best_validation_main_loss": statistics.stdev(losses),
                "mean_wall_seconds": statistics.mean(times),
            }
        )
    payload = {
        "warning": "1.7M-parameter, 153.6K-token proxy; a ranking signal, not a target-model claim.",
        "steps": args.steps,
        "tokens_per_run": args.steps * 4 * 128,
        "seeds": args.seeds,
        "aggregate": aggregate,
        "runs": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
