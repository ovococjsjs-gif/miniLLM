#!/usr/bin/env python3
"""Matched 300-step architecture and recovery-curriculum event-core controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from run_aira_byte_event_proxy import (
    autonomous_generation,
    calibrate_generated_contexts,
    cascade_evaluation,
    encoded_context,
    neural_evaluation,
    prepare_examples,
)
from torch.nn import functional as F

from minillm.aira import (
    AttentionByteEventLM,
    ByteBPEBridge,
    ByteEventLM,
    ConvByteEventLM,
    build_compact_shelf,
)

EventCore = ByteEventLM | ConvByteEventLM | AttentionByteEventLM


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def prepare_noisy_examples(
    bridge: ByteBPEBridge,
    stream: bytes,
    positions: np.ndarray,
    *,
    corruption_probability: float,
    raw_context_bytes: int,
    token_context: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    contexts = []
    for position in positions:
        prefix = bytearray(
            stream[max(0, int(position) - raw_context_bytes) : int(position)]
        )
        mask = rng.random(len(prefix)) < corruption_probability
        replacements = rng.integers(0, 256, size=len(prefix), dtype=np.uint8)
        for index in np.flatnonzero(mask):
            prefix[int(index)] = int(replacements[index])
        contexts.append(
            encoded_context(
                bridge,
                prefix,
                raw_context_bytes=raw_context_bytes,
                token_context=token_context,
            )
        )
    targets = np.frombuffer(stream, dtype=np.uint8)[positions].astype(np.int64)
    return np.stack(contexts), targets


def prepare_contiguous_examples(
    bridge: ByteBPEBridge,
    stream: bytes,
    *,
    samples: int,
    span: int,
    raw_context_bytes: int,
    token_context: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    span_count = math.ceil(samples / span)
    starts = rng.choice(
        np.arange(raw_context_bytes, len(stream) - span),
        span_count,
        replace=False,
    )
    positions = np.concatenate([start + np.arange(span) for start in starts])[:samples]
    contexts = np.stack(
        [
            encoded_context(
                bridge,
                stream[max(0, int(position) - raw_context_bytes) : int(position)],
                raw_context_bytes=raw_context_bytes,
                token_context=token_context,
            )
            for position in positions
        ]
    )
    targets = np.frombuffer(stream, dtype=np.uint8)[positions].astype(np.int64)
    return contexts, targets


def make_model(architecture: str, vocab_size: int, context_size: int) -> EventCore:
    if architecture == "gated-mlp":
        return ByteEventLM(vocab_size, context_size, d_model=48)
    if architecture == "conv":
        return ConvByteEventLM(vocab_size, context_size, d_model=52, blocks=4)
    if architecture == "attention":
        return AttentionByteEventLM(
            vocab_size, context_size, d_model=52, heads=4, layers=2
        )
    raise ValueError(architecture)


def optimizer_steps(
    model: EventCore,
    optimizer: torch.optim.Optimizer,
    contexts: np.ndarray,
    targets: np.ndarray,
    *,
    steps: int,
    batch_size: int,
    rng: np.random.Generator,
    samples: list[float],
) -> None:
    model.train()
    for _ in range(steps):
        indices = rng.integers(0, len(contexts), size=batch_size)
        token_contexts = torch.from_numpy(contexts[indices])
        byte_targets = torch.from_numpy(targets[indices])
        loss = F.cross_entropy(model(token_contexts), byte_targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        samples.append(float(loss.detach()))


def prepare_recovery_examples(
    model: EventCore,
    bridge: ByteBPEBridge,
    stream: bytes,
    *,
    samples: int,
    rollout: int,
    raw_context_bytes: int,
    token_context: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    positions = rng.choice(
        np.arange(raw_context_bytes, len(stream) - rollout),
        min(samples, len(stream) - raw_context_bytes - rollout),
        replace=False,
    )
    contexts = []
    targets = []
    model.eval()
    with torch.inference_mode():
        for position in positions:
            prefix = bytearray(
                stream[int(position) - raw_context_bytes : int(position)]
            )
            for _ in range(rollout):
                encoded = encoded_context(
                    bridge,
                    prefix,
                    raw_context_bytes=raw_context_bytes,
                    token_context=token_context,
                )
                emitted = int(model(torch.from_numpy(encoded[None, :])).argmax(-1))
                prefix.append(emitted)
            contexts.append(
                encoded_context(
                    bridge,
                    prefix,
                    raw_context_bytes=raw_context_bytes,
                    token_context=token_context,
                )
            )
            targets.append(stream[int(position) + rollout])
    return np.stack(contexts), np.asarray(targets, dtype=np.int64)


def train_variant(
    architecture: str,
    curriculum: str,
    bridge: ByteBPEBridge,
    neural_stream: bytes,
    random_contexts: np.ndarray,
    random_targets: np.ndarray,
    contiguous_contexts: np.ndarray,
    contiguous_targets: np.ndarray,
    noisy_contexts: np.ndarray,
    noisy_targets: np.ndarray,
    *,
    context_size: int,
    steps: int,
    batch_size: int,
    recovery_examples: int,
    raw_context_bytes: int,
    seed: int,
) -> tuple[EventCore, dict[str, object]]:
    torch.manual_seed(seed)
    model = make_model(architecture, bridge.vocab_size, context_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    rng = np.random.default_rng(seed)
    losses: list[float] = []
    started = time.perf_counter()
    recovery_seconds = 0.0
    if curriculum == "recovery":
        first_steps = steps // 2
        optimizer_steps(
            model,
            optimizer,
            random_contexts,
            random_targets,
            steps=first_steps,
            batch_size=batch_size,
            rng=rng,
            samples=losses,
        )
        recovery_started = time.perf_counter()
        recovery_contexts, recovery_targets = prepare_recovery_examples(
            model,
            bridge,
            neural_stream,
            samples=recovery_examples,
            rollout=4,
            raw_context_bytes=raw_context_bytes,
            token_context=context_size,
            seed=seed + 10_000,
        )
        recovery_seconds = time.perf_counter() - recovery_started
        teacher_indices = rng.choice(
            len(random_contexts), len(recovery_contexts), replace=False
        )
        mixed_contexts = np.concatenate(
            (random_contexts[teacher_indices], recovery_contexts)
        )
        mixed_targets = np.concatenate(
            (random_targets[teacher_indices], recovery_targets)
        )
        optimizer_steps(
            model,
            optimizer,
            mixed_contexts,
            mixed_targets,
            steps=steps - first_steps,
            batch_size=batch_size,
            rng=rng,
            samples=losses,
        )
    else:
        if curriculum == "contiguous":
            contexts, targets = contiguous_contexts, contiguous_targets
        elif curriculum == "noise":
            contexts, targets = noisy_contexts, noisy_targets
        else:
            contexts, targets = random_contexts, random_targets
        optimizer_steps(
            model,
            optimizer,
            contexts,
            targets,
            steps=steps,
            batch_size=batch_size,
            rng=rng,
            samples=losses,
        )
    model.eval()
    return model, {
        "architecture": architecture,
        "curriculum": curriculum,
        "seed": seed,
        "steps": steps,
        "training_seconds": time.perf_counter() - started,
        "recovery_preparation_seconds": recovery_seconds,
        "loss_samples": [losses[0], losses[len(losses) // 2], losses[-1]],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "parameter_bytes": model.parameter_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-tokens", required=True)
    parser.add_argument("--validation-tokens", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--steps", type=int, default=300, choices=range(1, 301))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 314, 2718])
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=(
            "gated-mlp:random",
            "conv:random",
            "attention:random",
            "attention:contiguous",
            "attention:noise",
            "attention:recovery",
        ),
        default=None,
    )
    parser.add_argument("--shelf-train-tokens", type=int, default=600_000)
    parser.add_argument("--neural-train-tokens", type=int, default=800_000)
    parser.add_argument("--training-examples", type=int, default=40_000)
    parser.add_argument("--validation-examples", type=int, default=10_000)
    parser.add_argument("--recovery-examples", type=int, default=4_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--raw-context-bytes", type=int, default=64)
    parser.add_argument("--token-context", type=int, default=16)
    parser.add_argument("--generation-sequences", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", default="results/aira_event_core_ablation.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    train_path = Path(args.train_tokens)
    validation_path = Path(args.validation_tokens)
    tokenizer_path = Path(args.tokenizer)
    train_tokens = np.memmap(train_path, dtype=np.uint32, mode="r")
    validation_tokens = np.memmap(validation_path, dtype=np.uint32, mode="r")
    bridge = ByteBPEBridge.from_tokenizer_json(tokenizer_path)
    shelf_stream = bridge.tokens_to_bytes(train_tokens[: args.shelf_train_tokens])
    neural_stream = bridge.tokens_to_bytes(
        train_tokens[
            args.shelf_train_tokens : args.shelf_train_tokens + args.neural_train_tokens
        ]
    )
    validation_stream = bridge.tokens_to_bytes(validation_tokens)
    shelf_units = np.frombuffer(shelf_stream, dtype=np.uint8).astype(np.uint32)
    levels = [build_compact_shelf(shelf_units, order) for order in (4, 8, 16)]
    preparation_started = time.perf_counter()
    random_contexts, random_targets, random_positions = prepare_examples(
        bridge,
        neural_stream,
        samples=args.training_examples,
        raw_context_bytes=args.raw_context_bytes,
        token_context=args.token_context,
        seed=123,
    )
    noisy_contexts, noisy_targets = prepare_noisy_examples(
        bridge,
        neural_stream,
        random_positions,
        corruption_probability=0.10,
        raw_context_bytes=args.raw_context_bytes,
        token_context=args.token_context,
        seed=231,
    )
    contiguous_contexts, contiguous_targets = prepare_contiguous_examples(
        bridge,
        neural_stream,
        samples=args.training_examples,
        span=128,
        raw_context_bytes=args.raw_context_bytes,
        token_context=args.token_context,
        seed=321,
    )
    validation_contexts, validation_targets, validation_positions = prepare_examples(
        bridge,
        validation_stream,
        samples=args.validation_examples,
        raw_context_bytes=args.raw_context_bytes,
        token_context=args.token_context,
        seed=456,
    )
    example_preparation_seconds = time.perf_counter() - preparation_started
    rng = np.random.default_rng(789)
    population = np.arange(
        args.raw_context_bytes, len(validation_stream) - args.horizon
    )
    sequence_count = min(args.generation_sequences, len(population) // 2)
    generated_starts = rng.choice(population, sequence_count * 2, replace=False)
    calibration_starts = np.sort(generated_starts[:sequence_count])
    test_starts = np.sort(generated_starts[sequence_count:])
    all_variants = (
        ("gated-mlp", "random"),
        ("conv", "random"),
        ("attention", "random"),
        ("attention", "contiguous"),
        ("attention", "noise"),
        ("attention", "recovery"),
    )
    variants = (
        tuple(tuple(value.split(":")) for value in args.variants)
        if args.variants is not None
        else all_variants
    )
    rows = []
    for architecture, curriculum in variants:
        for seed in args.seeds:
            model, training = train_variant(
                architecture,
                curriculum,
                bridge,
                neural_stream,
                random_contexts,
                random_targets,
                contiguous_contexts,
                contiguous_targets,
                noisy_contexts,
                noisy_targets,
                context_size=args.token_context,
                steps=args.steps,
                batch_size=args.batch_size,
                recovery_examples=args.recovery_examples,
                raw_context_bytes=args.raw_context_bytes,
                seed=seed,
            )
            neural = neural_evaluation(model, validation_contexts, validation_targets)
            cascade = cascade_evaluation(
                model,
                bridge,
                levels,
                validation_stream,
                validation_contexts,
                validation_targets,
                validation_positions,
            )
            neural_generation = autonomous_generation(
                model,
                bridge,
                levels,
                validation_stream,
                test_starts,
                horizon=args.horizon,
                raw_context_bytes=args.raw_context_bytes,
                token_context=args.token_context,
                use_shelf=False,
                oracle_fallback=False,
            )
            generated_calibration = calibrate_generated_contexts(
                model,
                bridge,
                levels,
                validation_stream,
                calibration_starts,
                horizon=args.horizon,
                raw_context_bytes=args.raw_context_bytes,
                token_context=args.token_context,
            )
            threshold = generated_calibration["fitted"]["threshold"]
            calibrated_generation = autonomous_generation(
                model,
                bridge,
                levels,
                validation_stream,
                test_starts,
                horizon=args.horizon,
                raw_context_bytes=args.raw_context_bytes,
                token_context=args.token_context,
                use_shelf=threshold is not None,
                oracle_fallback=False,
                confidence_threshold=float(threshold or 0.95),
            )
            rows.append(
                {
                    **training,
                    "neural_validation": neural,
                    "cascade_validation": cascade,
                    "neural_generation": neural_generation,
                    "generated_context_calibration": generated_calibration,
                    "calibrated_generation": calibrated_generation,
                }
            )
    summary = []
    for architecture, curriculum in variants:
        selected = [
            row
            for row in rows
            if row["architecture"] == architecture and row["curriculum"] == curriculum
        ]
        summary.append(
            {
                "architecture": architecture,
                "curriculum": curriculum,
                "runs": len(selected),
                "mean_parameters": statistics.mean(
                    row["parameters"] for row in selected
                ),
                "mean_training_seconds": statistics.mean(
                    row["training_seconds"] for row in selected
                ),
                "mean_neural_validation_perplexity": statistics.mean(
                    row["neural_validation"]["perplexity"] for row in selected
                ),
                "mean_cascade_validation_perplexity": statistics.mean(
                    row["cascade_validation"]["perplexity"] for row in selected
                ),
                "mean_cascade_validation_accuracy": statistics.mean(
                    row["cascade_validation"]["accuracy"] for row in selected
                ),
                "mean_neural_generation_accuracy": statistics.mean(
                    row["neural_generation"]["byte_accuracy"] for row in selected
                ),
                "safe_generated_thresholds": sum(
                    row["generated_context_calibration"]["fitted"]["threshold"]
                    is not None
                    for row in selected
                ),
                "mean_calibrated_generation_accuracy": statistics.mean(
                    row["calibrated_generation"]["byte_accuracy"] for row in selected
                ),
                "mean_calibrated_neural_fraction": statistics.mean(
                    row["calibrated_generation"]["neural_fraction"] for row in selected
                ),
            }
        )
    payload = {
        "schema_version": 1,
        "experiment": "aira-v2-event-core-architecture-curriculum-ablation-v1",
        "warning": "All arms are 300-step tiny proxies. Parameter counts are matched within 1.4%; generated-context calibration is a hard gate, not tuned on test sequences.",
        "source": {
            "train_tokens": str(train_path),
            "train_sha256": sha256(train_path),
            "validation_tokens": str(validation_path),
            "validation_sha256": sha256(validation_path),
            "tokenizer": str(tokenizer_path),
            "tokenizer_sha256": sha256(tokenizer_path),
        },
        "environment": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "threads": args.threads,
        },
        "protocol": {
            "steps": args.steps,
            "seeds": args.seeds,
            "variants": [list(variant) for variant in variants],
            "shelf_train_tokens": args.shelf_train_tokens,
            "shelf_train_bytes": len(shelf_stream),
            "neural_train_tokens": args.neural_train_tokens,
            "training_examples": len(random_contexts),
            "contiguous_span": 128,
            "noise_corruption_probability": 0.10,
            "recovery_examples": args.recovery_examples,
            "recovery_rollout": 4,
            "recovery_second_phase_fraction": 0.5,
            "validation_examples": len(validation_contexts),
            "raw_context_bytes": args.raw_context_bytes,
            "dynamic_bpe_context_tokens": args.token_context,
            "batch_size": args.batch_size,
            "generated_calibration_sequences": len(calibration_starts),
            "generated_test_sequences": len(test_starts),
            "generation_horizon": args.horizon,
            "strict_utf8_generation": True,
        },
        "packed_shelf_bytes": sum(level.packed_bytes for level in levels),
        "example_preparation_seconds": example_preparation_seconds,
        "summary": summary,
        "runs": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
