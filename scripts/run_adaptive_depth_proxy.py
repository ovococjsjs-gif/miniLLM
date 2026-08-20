#!/usr/bin/env python3
"""Small generated-task screen for variable recurrent depth and self-distillation."""

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
from minillm.analysis import profile_model
from minillm.benchmarks.associative_recall import generate_recall_batch
from minillm.config import MiniLLMConfig
from minillm.model import MiniLLM


@dataclass(frozen=True)
class Variant:
    name: str
    step_conditioning: bool
    sandwich_distillation: bool


def model_config(step_conditioning: bool) -> MiniLLMConfig:
    return MiniLLMConfig(
        vocab_size=68,
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


def recall_accuracy(
    model: MiniLLM,
    *,
    recurrences: int,
    generator: torch.Generator,
    batches: int = 4,
    batch_size: int = 128,
) -> float:
    values = []
    model.eval()
    with torch.no_grad():
        for _ in range(batches):
            batch = generate_recall_batch(batch_size, generator=generator)
            logits = model(batch.input_ids, core_repetitions=recurrences).logits
            batch_indices = torch.arange(batch_size)[:, None]
            predictions = logits[
                batch_indices, batch.prediction_positions[None, :]
            ].argmax(dim=-1)
            values.append(float((predictions == batch.answers).float().mean()))
    return statistics.mean(values)


def forward_tokens_per_second(
    model: MiniLLM,
    *,
    recurrences: int,
    input_ids: torch.Tensor,
    repeats: int = 10,
) -> float:
    model.eval()
    durations = []
    with torch.no_grad():
        model(input_ids, core_repetitions=recurrences)
        for _ in range(repeats):
            started = time.perf_counter()
            model(input_ids, core_repetitions=recurrences)
            durations.append(time.perf_counter() - started)
    median = statistics.median(durations)
    return input_ids.numel() / median


def train_variant(
    variant: Variant,
    *,
    steps: int,
    batch_size: int,
    seed: int,
) -> dict[str, object]:
    torch.manual_seed(seed)
    random.seed(seed)
    generator = torch.Generator().manual_seed(seed + 1)
    eval_generator = torch.Generator().manual_seed(seed + 10_000)
    config = model_config(variant.step_conditioning)
    model = MiniLLM(config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-3, betas=(0.9, 0.95), weight_decay=0.01
    )
    losses = []
    consistency_values = []
    recurrence_histogram = {str(depth): 0 for depth in range(1, 5)}
    started = time.perf_counter()
    model.train()
    for step in range(steps):
        batch = generate_recall_batch(batch_size, generator=generator)
        optimizer.zero_grad(set_to_none=True)
        if variant.sandwich_distillation:
            student_depth = 1 + step % 3
            deep = model(batch.input_ids, labels=batch.labels, core_repetitions=4)
            shallow = model(
                batch.input_ids,
                labels=batch.labels,
                core_repetitions=student_depth,
            )
            assert deep.loss is not None and shallow.loss is not None
            consistency = masked_depth_consistency_kl(
                shallow.logits, deep.logits, batch.labels, temperature=2.0
            )
            loss = 0.5 * (deep.loss + shallow.loss) + 0.2 * consistency
            recurrence_histogram["4"] += 1
            recurrence_histogram[str(student_depth)] += 1
            consistency_values.append(float(consistency.detach()))
        else:
            depth = random.randint(1, 4)
            output = model(batch.input_ids, labels=batch.labels, core_repetitions=depth)
            assert output.loss is not None
            loss = output.loss
            recurrence_histogram[str(depth)] += 1
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in {0, steps // 4, steps // 2, 3 * steps // 4, steps - 1}:
            losses.append(float(loss.detach()))
    training_seconds = time.perf_counter() - started

    accuracy_by_depth = {
        str(depth): recall_accuracy(
            model,
            recurrences=depth,
            generator=eval_generator,
        )
        for depth in (1, 2, 4)
    }
    speed_input = generate_recall_batch(128, generator=eval_generator).input_ids
    speed_by_depth = {
        str(depth): forward_tokens_per_second(
            model,
            recurrences=depth,
            input_ids=speed_input,
        )
        for depth in (1, 2, 4)
    }
    profile = profile_model(config, context_length=16, weight_bits=16, kv_bits=16)
    return {
        "variant": variant.name,
        "seed": seed,
        "stored_parameters": profile.unique_parameters,
        "step_conditioning_parameters": 4 * config.d_model
        if variant.step_conditioning
        else 0,
        "steps": steps,
        "examples_seen": steps * batch_size,
        "training_seconds": training_seconds,
        "loss_samples": losses,
        "mean_depth_consistency_kl": statistics.mean(consistency_values)
        if consistency_values
        else None,
        "recurrence_histogram": recurrence_histogram,
        "held_out_accuracy_by_depth": accuracy_by_depth,
        "forward_tokens_per_second_by_depth": speed_by_depth,
        "deep_minus_fast_accuracy": accuracy_by_depth["4"] - accuracy_by_depth["1"],
        "deep_to_fast_speed_ratio": speed_by_depth["4"] / speed_by_depth["1"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps", type=int, default=200, choices=range(1, 301), metavar="[1-300]"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seeds", nargs="+", type=int, default=[123, 456, 789])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", default="results/adaptive_depth_proxy.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    variants = (
        Variant("random-unroll", False, False),
        Variant("step-conditioned", True, False),
        Variant("step-conditioned-sandwich", True, True),
    )
    rows = [
        train_variant(
            variant,
            steps=args.steps,
            batch_size=args.batch_size,
            seed=seed,
        )
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
                "mean_accuracy_by_depth": {
                    depth: statistics.mean(
                        float(row["held_out_accuracy_by_depth"][depth])
                        for row in selected
                    )
                    for depth in ("1", "2", "4")
                },
                "mean_forward_tokens_per_second_by_depth": {
                    depth: statistics.mean(
                        float(row["forward_tokens_per_second_by_depth"][depth])
                        for row in selected
                    )
                    for depth in ("1", "2", "4")
                },
            }
        )
    payload = {
        "schema_version": 1,
        "experiment": "adaptive-recurrent-depth-mqar-v1",
        "warning": "Generated CPU proxy; validates algorithmic behavior and cost, not language-model quality.",
        "task": "generated multi-query associative recall",
        "chance_token_accuracy": 1 / 32,
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
