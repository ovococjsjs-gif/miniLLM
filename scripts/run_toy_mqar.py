#!/usr/bin/env python3
"""Run a small iso-step MQAR architecture screen and save machine-readable results."""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import replace
from pathlib import Path

import torch

from minillm.analysis import profile_model
from minillm.benchmarks.associative_recall import evaluate_recall, generate_recall_batch
from minillm.config import EngramConfig, MiniLLMConfig
from minillm.model import MiniLLM


def variants() -> dict[str, MiniLLMConfig]:
    base = MiniLLMConfig(
        vocab_size=68,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        ffn_hidden=128,
        max_seq_len=32,
        mtp_depth=0,
        dropout=0.0,
        prelude_layers=(),
        coda_layers=(),
        core_repetitions=1,
        max_core_repetitions=1,
        recurrent_input_injection=False,
    )
    return {
        "attention_4": replace(base, core_layers=("attention",) * 4),
        "conv3_attention1": replace(
            base, core_layers=("conv", "conv", "conv", "attention")
        ),
        "gdn2_3_attention1": replace(
            base, core_layers=("gdn2", "gdn2", "gdn2", "attention")
        ),
        "recursive_conv_attention": replace(
            base,
            prelude_layers=("attention",),
            core_layers=("conv", "attention"),
            core_repetitions=3,
            max_core_repetitions=3,
            recurrent_input_injection=True,
        ),
        "recursive_plus_engram": replace(
            base,
            prelude_layers=("attention",),
            core_layers=("conv", "attention"),
            core_repetitions=3,
            max_core_repetitions=3,
            recurrent_input_injection=True,
            engram=EngramConfig(
                enabled=True,
                ngram_orders=(2,),
                num_hash_heads=2,
                table_size=257,
                embedding_dim=8,
                conv_kernel=3,
            ),
        ),
    }


def train_variant(
    name: str,
    config: MiniLLMConfig,
    *,
    steps: int,
    batch_size: int,
    seed: int,
) -> dict[str, object]:
    torch.manual_seed(seed)
    train_gen = torch.Generator().manual_seed(seed + 1)
    eval_gen = torch.Generator().manual_seed(seed + 10_000)
    model = MiniLLM(config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-3, betas=(0.9, 0.95), weight_decay=0.01
    )
    losses: list[float] = []
    started = time.perf_counter()
    model.train()
    for step in range(steps):
        batch = generate_recall_batch(batch_size, generator=train_gen)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch.input_ids, labels=batch.labels)
        assert output.loss is not None
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in {0, steps // 4, steps // 2, 3 * steps // 4, steps - 1}:
            losses.append(float(output.loss.detach()))
    elapsed = time.perf_counter() - started
    accuracies = []
    for _ in range(8):
        test = generate_recall_batch(128, generator=eval_gen)
        accuracies.append(evaluate_recall(model, test))
    profile = profile_model(config, context_length=16, weight_bits=16, kv_bits=16)
    return {
        "name": name,
        "seed": seed,
        "stored_parameters": profile.unique_parameters,
        "active_parameter_applications_per_token": profile.active_parameter_applications_per_token,
        "effective_depth": profile.effective_depth,
        "train_steps": steps,
        "examples_seen": steps * batch_size,
        "wall_seconds": elapsed,
        "loss_samples": losses,
        "held_out_token_accuracy": sum(accuracies) / len(accuracies),
        "held_out_accuracy_std": torch.tensor(accuracies).std().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps", type=int, default=300, choices=range(1, 301), metavar="[1-300]"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=[123, 456, 789])
    parser.add_argument("--output", default="results/toy_mqar.json")
    args = parser.parse_args()
    results = [
        train_variant(
            name,
            config.validate(),
            steps=args.steps,
            batch_size=args.batch_size,
            seed=seed,
        )
        for seed in args.seeds
        for name, config in variants().items()
    ]
    summary = []
    for name in variants():
        rows = [row for row in results if row["name"] == name]
        accuracies = torch.tensor([row["held_out_token_accuracy"] for row in rows])
        times = torch.tensor([row["wall_seconds"] for row in rows])
        summary.append(
            {
                "name": name,
                "runs": len(rows),
                "mean_held_out_token_accuracy": float(accuracies.mean()),
                "between_seed_accuracy_std": float(accuracies.std())
                if len(rows) > 1
                else 0.0,
                "mean_wall_seconds": float(times.mean()),
            }
        )
    payload = {
        "warning": "Tiny CPU proxy screen; use for debugging/ranking hypotheses, not capability claims.",
        "task": "generated multi-query associative recall (4 bindings, 2 queries)",
        "chance_token_accuracy": 1 / 32,
        "seeds": args.seeds,
        "torch": torch.__version__,
        "python_platform": platform.platform(),
        "summary": summary,
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
