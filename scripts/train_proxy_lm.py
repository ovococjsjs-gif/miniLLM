#!/usr/bin/env python3
"""Train a tiny language-model proxy in the constrained local environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minillm.config import MiniLLMConfig
from minillm.training import TrainConfig, train_proxy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="configs/proxy_3m.json")
    parser.add_argument("--tokens", default="data/tokens-4096")
    parser.add_argument("--output", default="runs/proxy-3m")
    parser.add_argument(
        "--steps", type=int, default=200, choices=range(1, 301), metavar="[1-300]"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    model = MiniLLMConfig.load(args.model)
    training = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        learning_rate=5e-4,
        warmup_steps=min(20, args.steps // 10),
        eval_interval=max(1, args.steps // 10),
        eval_batches=10,
        checkpoint_interval=args.steps,
        seed=args.seed,
    )
    summary = train_proxy(
        model,
        training,
        train_tokens=Path(args.tokens) / "train.bin",
        validation_tokens=Path(args.tokens) / "validation.bin",
        output_directory=args.output,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
