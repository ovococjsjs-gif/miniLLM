#!/usr/bin/env python3
"""Test whether extra shared-depth passes solve more pointer-composition hops."""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from minillm.adaptive_depth import masked_depth_consistency_kl
from minillm.benchmarks.pointer_chase import generate_pointer_chase_batch
from minillm.config import MiniLLMConfig
from minillm.model import MiniLLM


@dataclass(frozen=True)
class Variant:
    name: str
    step_conditioning: bool
    sandwich_distillation: bool


def config(step_conditioning: bool) -> MiniLLMConfig:
    return MiniLLMConfig(
        vocab_size=16,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        ffn_hidden=128,
        max_seq_len=32,
        dropout=0.0,
        prelude_layers=("attention",),
        core_layers=("conv", "attention"),
        coda_layers=(),
        core_repetitions=4,
        max_core_repetitions=4,
        recurrent_input_injection=True,
        recurrent_step_conditioning=step_conditioning,
        mtp_depth=0,
    ).validate()


def accuracy(
    model: MiniLLM,
    *,
    task_hops: int,
    recurrences: int,
    generator: torch.Generator,
    batches: int = 8,
) -> float:
    values = []
    model.eval()
    with torch.no_grad():
        for _ in range(batches):
            batch = generate_pointer_chase_batch(
                128, fixed_hops=task_hops, generator=generator
            )
            logits = model(batch.input_ids, core_repetitions=recurrences).logits
            predictions = logits[
                torch.arange(batch.input_ids.shape[0]), batch.prediction_positions
            ].argmax(dim=-1)
            values.append(float((predictions == batch.answers).float().mean()))
    return statistics.mean(values)


def train(
    variant: Variant, *, steps: int, batch_size: int, seed: int
) -> dict[str, object]:
    torch.manual_seed(seed)
    random.seed(seed)
    generator = torch.Generator().manual_seed(seed + 1)
    model = MiniLLM(config(variant.step_conditioning))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-3, betas=(0.9, 0.95), weight_decay=0.01
    )
    started = time.perf_counter()
    losses = []
    for step in range(steps):
        model.train()
        progress = step / max(1, steps)
        training_max_hops = (
            1
            if progress < 0.5
            else 2
            if progress < 0.7
            else 3
            if progress < 0.85
            else 4
        )
        batch = generate_pointer_chase_batch(
            batch_size,
            max_hops=training_max_hops,
            generator=generator,
        )
        optimizer.zero_grad(set_to_none=True)
        if variant.sandwich_distillation:
            shallow_depth = 1 + step % 3
            deep = model(batch.input_ids, labels=batch.labels, core_repetitions=4)
            shallow = model(
                batch.input_ids,
                labels=batch.labels,
                core_repetitions=shallow_depth,
            )
            assert deep.loss is not None and shallow.loss is not None
            consistency = masked_depth_consistency_kl(
                shallow.logits, deep.logits, batch.labels
            )
            loss = 0.5 * (deep.loss + shallow.loss) + 0.2 * consistency
        else:
            depth = random.randint(1, 4)
            output = model(batch.input_ids, labels=batch.labels, core_repetitions=depth)
            assert output.loss is not None
            loss = output.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in {0, steps // 4, steps // 2, 3 * steps // 4, steps - 1}:
            losses.append(float(loss.detach()))
    training_seconds = time.perf_counter() - started

    eval_generator = torch.Generator().manual_seed(seed + 10_000)
    matrix = {
        str(task_hops): {
            str(depth): accuracy(
                model,
                task_hops=task_hops,
                recurrences=depth,
                generator=eval_generator,
            )
            for depth in (1, 2, 4)
        }
        for task_hops in (1, 2, 3, 4)
    }
    diagonal = statistics.mean(matrix[str(depth)][str(depth)] for depth in (1, 2, 4))
    deep = statistics.mean(matrix[str(hops)]["4"] for hops in (1, 2, 3, 4))
    fast = statistics.mean(matrix[str(hops)]["1"] for hops in (1, 2, 3, 4))
    return {
        "variant": variant.name,
        "seed": seed,
        "steps": steps,
        "training_seconds": training_seconds,
        "loss_samples": losses,
        "accuracy_by_task_hops_and_recurrences": matrix,
        "diagonal_accuracy": diagonal,
        "mean_fast_accuracy": fast,
        "mean_deep_accuracy": deep,
        "deep_minus_fast_accuracy": deep - fast,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps", type=int, default=300, choices=range(1, 301), metavar="[1-300]"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seeds", nargs="+", type=int, default=[123, 456, 789])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", default="results/pointer_chase_depth_proxy.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    variants = (
        Variant("random-unroll", False, False),
        Variant("step-conditioned", True, False),
        Variant("step-conditioned-sandwich", True, True),
    )
    rows = [
        train(variant, steps=args.steps, batch_size=args.batch_size, seed=seed)
        for seed in args.seeds
        for variant in variants
    ]
    summary = []
    for variant in variants:
        selected = [row for row in rows if row["variant"] == variant.name]
        summary.append(
            {
                "variant": variant.name,
                "runs": len(selected),
                "mean_training_seconds": statistics.mean(
                    float(row["training_seconds"]) for row in selected
                ),
                "mean_diagonal_accuracy": statistics.mean(
                    float(row["diagonal_accuracy"]) for row in selected
                ),
                "mean_fast_accuracy": statistics.mean(
                    float(row["mean_fast_accuracy"]) for row in selected
                ),
                "mean_deep_accuracy": statistics.mean(
                    float(row["mean_deep_accuracy"]) for row in selected
                ),
                "mean_deep_minus_fast_accuracy": statistics.mean(
                    float(row["deep_minus_fast_accuracy"]) for row in selected
                ),
            }
        )
    payload = {
        "schema_version": 1,
        "experiment": "adaptive-depth-pointer-chase-v2",
        "warning": "Generated CPU proxy; tests iterative composition, not language quality.",
        "task": "curriculum: follow a random single-cycle 8-node mapping for 1-4 requested hops",
        "chance_accuracy": 1 / 8,
        "steps": args.steps,
        "seeds": args.seeds,
        "threads": args.threads,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "summary": summary,
        "runs": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
