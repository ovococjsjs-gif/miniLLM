#!/usr/bin/env python3
"""End-to-end raw-byte shelf with a bounded dynamic-BPE neural event core."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from minillm.aira import (
    ByteBPEBridge,
    ByteEventLM,
    EpisodicFactStore,
    build_compact_shelf,
    calibrate_reliability_threshold,
    predict_shelf_next,
    utf8_allowed_next_bytes,
)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def encoded_context(
    bridge: ByteBPEBridge,
    raw_prefix: bytes | bytearray,
    *,
    raw_context_bytes: int,
    token_context: int,
    pad_token: int = 0,
) -> np.ndarray:
    token_ids = bridge.encode_bytes(raw_prefix[-raw_context_bytes:])
    result = np.full(token_context, pad_token, dtype=np.int64)
    selected = token_ids[-token_context:]
    result[-len(selected) :] = selected
    return result


def prepare_examples(
    bridge: ByteBPEBridge,
    stream: bytes,
    *,
    samples: int,
    raw_context_bytes: int,
    token_context: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    positions = np.sort(
        rng.choice(
            np.arange(raw_context_bytes, len(stream)),
            min(samples, len(stream) - raw_context_bytes),
            replace=False,
        )
    )
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
    return contexts, targets, positions


def train_model(
    contexts: np.ndarray,
    targets: np.ndarray,
    *,
    vocab_size: int,
    token_context: int,
    d_model: int,
    steps: int,
    batch_size: int,
    seed: int,
) -> tuple[ByteEventLM, dict[str, object]]:
    torch.manual_seed(seed)
    model = ByteEventLM(vocab_size, token_context, d_model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    rng = np.random.default_rng(seed)
    loss_samples = []
    started = time.perf_counter()
    model.train()
    for step in range(steps):
        indices = rng.integers(0, len(contexts), size=batch_size)
        token_contexts = torch.from_numpy(contexts[indices])
        byte_targets = torch.from_numpy(targets[indices])
        logits = model(token_contexts)
        loss = F.cross_entropy(logits, byte_targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in {0, steps // 2, steps - 1}:
            loss_samples.append(float(loss.detach()))
    return model.eval(), {
        "seed": seed,
        "steps": steps,
        "training_seconds": time.perf_counter() - started,
        "loss_samples": loss_samples,
    }


def neural_evaluation(
    model: ByteEventLM,
    contexts: np.ndarray,
    targets: np.ndarray,
    *,
    batch_size: int = 1024,
) -> dict[str, float | int]:
    loss_sum = correct = 0.0
    started = time.perf_counter()
    with torch.inference_mode():
        for begin in range(0, len(contexts), batch_size):
            token_contexts = torch.from_numpy(contexts[begin : begin + batch_size])
            byte_targets = torch.from_numpy(targets[begin : begin + batch_size])
            logits = model(token_contexts)
            loss_sum += float(F.cross_entropy(logits, byte_targets, reduction="sum"))
            correct += float(torch.sum(logits.argmax(dim=-1) == byte_targets))
    seconds = time.perf_counter() - started
    nll = loss_sum / len(contexts)
    return {
        "bytes": len(contexts),
        "nll": nll,
        "perplexity": math.exp(nll),
        "accuracy": correct / len(contexts),
        "neural_calls": len(contexts),
        "seconds": seconds,
        "bytes_per_second": len(contexts) / seconds,
    }


def cascade_evaluation(
    model: ByteEventLM,
    bridge: ByteBPEBridge,
    levels,
    validation_stream: bytes,
    contexts: np.ndarray,
    targets: np.ndarray,
    positions: np.ndarray,
    *,
    batch_size: int = 1024,
) -> dict[str, float | int]:
    shelf_mask = np.zeros(len(positions), dtype=bool)
    shelf_predictions = np.zeros(len(positions), dtype=np.int64)
    shelf_probabilities = np.zeros(len(positions), dtype=np.float64)
    route_started = time.perf_counter()
    for index, position in enumerate(positions):
        candidate = predict_shelf_next(
            levels,
            np.frombuffer(
                validation_stream[max(0, int(position) - 16) : int(position)],
                dtype=np.uint8,
            ).astype(np.uint32),
            minimum_support=5,
            confidence_threshold=0.95,
            confidence_z=1.96,
        )
        if candidate is not None:
            shelf_mask[index] = True
            shelf_predictions[index] = candidate.token
            shelf_probabilities[index] = candidate.empirical_confidence
    route_seconds = time.perf_counter() - route_started
    fallback = np.flatnonzero(~shelf_mask)
    neural_losses = np.zeros(len(positions), dtype=np.float64)
    neural_predictions = np.zeros(len(positions), dtype=np.int64)
    model_started = time.perf_counter()
    with torch.inference_mode():
        for begin in range(0, len(fallback), batch_size):
            selected = fallback[begin : begin + batch_size]
            token_contexts = torch.from_numpy(contexts[selected])
            byte_targets = torch.from_numpy(targets[selected])
            logits = model(token_contexts)
            neural_losses[selected] = F.cross_entropy(
                logits, byte_targets, reduction="none"
            ).numpy()
            neural_predictions[selected] = logits.argmax(dim=-1).numpy()
    model_seconds = time.perf_counter() - model_started
    shelf_target_probability = np.where(
        shelf_predictions == targets,
        shelf_probabilities,
        (1 - shelf_probabilities) / 255,
    )
    losses = np.where(
        shelf_mask,
        -np.log(np.clip(shelf_target_probability, 1e-12, 1)),
        neural_losses,
    )
    predictions = np.where(shelf_mask, shelf_predictions, neural_predictions)
    nll = float(losses.mean())
    accepted = int(shelf_mask.sum())
    correct_shelf = int(np.sum(shelf_mask & (shelf_predictions == targets)))
    return {
        "bytes": len(positions),
        "nll": nll,
        "perplexity": math.exp(nll),
        "accuracy": float(np.mean(predictions == targets)),
        "shelf_bytes": accepted,
        "shelf_fraction": accepted / len(positions),
        "shelf_precision": correct_shelf / accepted if accepted else 0.0,
        "neural_calls": len(fallback),
        "neural_call_fraction": len(fallback) / len(positions),
        "route_seconds": route_seconds,
        "model_seconds": model_seconds,
        "total_seconds": route_seconds + model_seconds,
        "parameter_bytes_read_proxy": len(fallback) * model.parameter_bytes,
        "parameter_bytes_per_evaluated_byte": len(fallback)
        * model.parameter_bytes
        / len(positions),
    }


def repeats_cycle(values: bytearray, candidate: int) -> bool:
    sequence = values + bytes([candidate])
    for period in range(1, min(8, len(sequence) // 3) + 1):
        suffix = sequence[-period:]
        if all(
            sequence[-repeat * period : -(repeat - 1) * period] == suffix
            for repeat in range(2, 4)
        ):
            return True
    return False


def calibrate_generated_contexts(
    model: ByteEventLM,
    bridge: ByteBPEBridge,
    levels,
    validation_stream: bytes,
    starts: np.ndarray,
    *,
    horizon: int,
    raw_context_bytes: int,
    token_context: int,
) -> dict[str, object]:
    scores: list[float] = []
    correctness: list[bool] = []
    tiny_threshold = np.nextafter(0.0, 1.0)
    for start in starts:
        prefix = bytearray(
            validation_stream[int(start) - raw_context_bytes : int(start)]
        )
        reference = validation_stream[int(start) : int(start) + horizon]
        for offset in range(horizon):
            candidate = predict_shelf_next(
                levels,
                np.frombuffer(prefix[-16:], dtype=np.uint8).astype(np.uint32),
                minimum_support=5,
                confidence_threshold=tiny_threshold,
                confidence_z=1.96,
            )
            allowed = utf8_allowed_next_bytes(prefix)
            if candidate is not None and allowed[candidate.token]:
                scores.append(candidate.lower_confidence)
                correctness.append(candidate.token == reference[offset])
            encoded = encoded_context(
                bridge,
                prefix,
                raw_context_bytes=raw_context_bytes,
                token_context=token_context,
            )
            context = torch.from_numpy(encoded[None, :])
            with torch.inference_mode():
                logits = model(context)[0]
                logits[~torch.from_numpy(allowed)] = -torch.inf
                emitted = int(logits.argmax())
            prefix.append(emitted)
    fitted = calibrate_reliability_threshold(
        np.asarray(scores),
        np.asarray(correctness),
        target_precision=0.95,
        confidence_z=1.96,
        minimum_accepted=20,
    )
    return {
        "candidate_examples": len(scores),
        "candidate_precision": float(np.mean(correctness)) if scores else None,
        "fitted": asdict(fitted),
    }


def autonomous_generation(
    model: ByteEventLM | None,
    bridge: ByteBPEBridge,
    levels,
    validation_stream: bytes,
    starts: np.ndarray,
    *,
    horizon: int,
    raw_context_bytes: int,
    token_context: int,
    use_shelf: bool,
    oracle_fallback: bool,
    confidence_threshold: float = 0.95,
) -> dict[str, float | int]:
    correct = common_prefix_sum = shelf_bytes = shelf_correct = neural_bytes = (
        rejected
    ) = 0
    model_seconds = 0.0
    started = time.perf_counter()
    for start in starts:
        prefix = bytearray(
            validation_stream[int(start) - raw_context_bytes : int(start)]
        )
        reference = validation_stream[int(start) : int(start) + horizon]
        output = bytearray()
        burst = 0
        cumulative_risk = 0.0
        since_neural = 0
        for offset in range(horizon):
            candidate = None
            if use_shelf:
                candidate = predict_shelf_next(
                    levels,
                    np.frombuffer(prefix[-16:], dtype=np.uint8).astype(np.uint32),
                    minimum_support=5,
                    confidence_threshold=confidence_threshold,
                    confidence_z=1.96,
                )
                if candidate is not None:
                    byte_risk = 1 - candidate.lower_confidence
                    blocked = (
                        burst >= 4
                        or cumulative_risk + byte_risk > 0.10
                        or since_neural >= 8
                        or repeats_cycle(prefix, candidate.token)
                        or not utf8_allowed_next_bytes(prefix)[candidate.token]
                    )
                    if blocked:
                        candidate = None
                        rejected += 1
            if candidate is not None:
                emitted = candidate.token
                shelf_bytes += 1
                shelf_correct += emitted == reference[offset]
                burst += 1
                cumulative_risk += 1 - candidate.lower_confidence
            else:
                if oracle_fallback:
                    emitted = reference[offset]
                else:
                    assert model is not None
                    encoded = encoded_context(
                        bridge,
                        prefix,
                        raw_context_bytes=raw_context_bytes,
                        token_context=token_context,
                    )
                    context = torch.from_numpy(encoded[None, :])
                    model_started = time.perf_counter()
                    with torch.inference_mode():
                        logits = model(context)[0]
                        allowed = torch.from_numpy(utf8_allowed_next_bytes(prefix))
                        logits[~allowed] = -torch.inf
                        emitted = int(logits.argmax())
                    model_seconds += time.perf_counter() - model_started
                neural_bytes += 1
                burst = 0
                cumulative_risk = 0.0
                since_neural = 0
            output.append(emitted)
            prefix.append(emitted)
            since_neural += 1
        correct += sum(
            actual == expected for actual, expected in zip(output, reference)
        )
        common = 0
        for actual, expected in zip(output, reference):
            if actual != expected:
                break
            common += 1
        common_prefix_sum += common
    total = len(starts) * horizon
    parameter_bytes = model.parameter_bytes if model is not None else 0
    return {
        "sequences": len(starts),
        "bytes": total,
        "byte_accuracy": correct / total,
        "mean_common_prefix_bytes": common_prefix_sum / len(starts),
        "shelf_bytes": shelf_bytes,
        "shelf_fraction": shelf_bytes / total,
        "shelf_precision": shelf_correct / shelf_bytes if shelf_bytes else 0.0,
        "neural_or_oracle_bytes": neural_bytes,
        "neural_fraction": neural_bytes / total,
        "control_rejections": rejected,
        "wall_seconds": time.perf_counter() - started,
        "model_seconds": model_seconds,
        "parameter_bytes_read_proxy": neural_bytes * parameter_bytes,
        "parameter_bytes_per_generated_byte": neural_bytes * parameter_bytes / total,
    }


def episodic_trace() -> dict[str, object]:
    store = EpisodicFactStore(capacity=8, dimension=128)
    slot = store.remember(
        "user.home-city",
        "Pskov",
        provenance={"source": "user-turn", "editable": True},
    )
    hit = store.recall("USER home city")
    unknown = store.recall("user.work-city")
    deleted = store.delete(slot)
    return {
        "accepted": hit.accepted,
        "value": hit.payload,
        "provenance": hit.provenance,
        "unknown_accepted": unknown.accepted,
        "deleted": deleted,
        "accepted_after_delete": store.recall("user.home.city").accepted,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-tokens", required=True)
    parser.add_argument("--validation-tokens", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tag", default="broad-8m-token-window")
    parser.add_argument("--steps", type=int, default=300, choices=range(1, 301))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 314, 2718])
    parser.add_argument("--shelf-train-tokens", type=int, default=600_000)
    parser.add_argument("--neural-train-tokens", type=int, default=8_000_000)
    parser.add_argument("--training-examples", type=int, default=40_000)
    parser.add_argument("--validation-examples", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--raw-context-bytes", type=int, default=64)
    parser.add_argument("--token-context", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--generation-sequences", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", default="results/aira_byte_event_proxy.json")
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
    shelf_started = time.perf_counter()
    levels = [build_compact_shelf(shelf_units, order) for order in (4, 8, 16)]
    shelf_build_seconds = time.perf_counter() - shelf_started
    preparation_started = time.perf_counter()
    train_contexts, train_targets, _ = prepare_examples(
        bridge,
        neural_stream,
        samples=args.training_examples,
        raw_context_bytes=args.raw_context_bytes,
        token_context=args.token_context,
        seed=123,
    )
    validation_contexts, validation_targets, validation_positions = prepare_examples(
        bridge,
        validation_stream,
        samples=args.validation_examples,
        raw_context_bytes=args.raw_context_bytes,
        token_context=args.token_context,
        seed=456,
    )
    preparation_seconds = time.perf_counter() - preparation_started
    rng = np.random.default_rng(789)
    population = np.arange(
        args.raw_context_bytes, len(validation_stream) - args.horizon
    )
    sequence_count = min(args.generation_sequences, len(population) // 2)
    selected_starts = rng.choice(population, sequence_count * 2, replace=False)
    calibration_starts = np.sort(selected_starts[:sequence_count])
    starts = np.sort(selected_starts[sequence_count:])
    runs = []
    for seed in args.seeds:
        model, training = train_model(
            train_contexts,
            train_targets,
            vocab_size=bridge.vocab_size,
            token_context=args.token_context,
            d_model=args.d_model,
            steps=args.steps,
            batch_size=args.batch_size,
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
            starts,
            horizon=args.horizon,
            raw_context_bytes=args.raw_context_bytes,
            token_context=args.token_context,
            use_shelf=False,
            oracle_fallback=False,
        )
        cascade_generation = autonomous_generation(
            model,
            bridge,
            levels,
            validation_stream,
            starts,
            horizon=args.horizon,
            raw_context_bytes=args.raw_context_bytes,
            token_context=args.token_context,
            use_shelf=True,
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
        calibrated_threshold = generated_calibration["fitted"]["threshold"]
        calibrated_generation = autonomous_generation(
            model,
            bridge,
            levels,
            validation_stream,
            starts,
            horizon=args.horizon,
            raw_context_bytes=args.raw_context_bytes,
            token_context=args.token_context,
            use_shelf=calibrated_threshold is not None,
            oracle_fallback=False,
            confidence_threshold=float(calibrated_threshold or 0.95),
        )
        runs.append(
            {
                **training,
                "model_parameters": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "model_parameter_bytes": model.parameter_bytes,
                "neural_validation": neural,
                "cascade_validation": cascade,
                "neural_generation": neural_generation,
                "cascade_generation": cascade_generation,
                "generated_context_calibration": generated_calibration,
                "calibrated_cascade_generation": calibrated_generation,
            }
        )
    oracle_generation = autonomous_generation(
        None,
        bridge,
        levels,
        validation_stream,
        starts,
        horizon=args.horizon,
        raw_context_bytes=args.raw_context_bytes,
        token_context=args.token_context,
        use_shelf=True,
        oracle_fallback=True,
    )
    summary = {
        "runs": len(runs),
        "mean_neural_validation_perplexity": statistics.mean(
            run["neural_validation"]["perplexity"] for run in runs
        ),
        "mean_cascade_validation_perplexity": statistics.mean(
            run["cascade_validation"]["perplexity"] for run in runs
        ),
        "mean_neural_validation_accuracy": statistics.mean(
            run["neural_validation"]["accuracy"] for run in runs
        ),
        "mean_cascade_validation_accuracy": statistics.mean(
            run["cascade_validation"]["accuracy"] for run in runs
        ),
        "mean_cascade_neural_call_fraction": statistics.mean(
            run["cascade_validation"]["neural_call_fraction"] for run in runs
        ),
        "mean_validation_parameter_byte_reduction": statistics.mean(
            1
            - run["cascade_validation"]["parameter_bytes_per_evaluated_byte"]
            / run["model_parameter_bytes"]
            for run in runs
        ),
        "mean_neural_generation_accuracy": statistics.mean(
            run["neural_generation"]["byte_accuracy"] for run in runs
        ),
        "mean_cascade_generation_accuracy": statistics.mean(
            run["cascade_generation"]["byte_accuracy"] for run in runs
        ),
        "mean_cascade_generation_neural_fraction": statistics.mean(
            run["cascade_generation"]["neural_fraction"] for run in runs
        ),
        "mean_cascade_generation_shelf_precision": statistics.mean(
            run["cascade_generation"]["shelf_precision"] for run in runs
        ),
        "generated_calibrations_with_safe_threshold": sum(
            run["generated_context_calibration"]["fitted"]["threshold"] is not None
            for run in runs
        ),
        "mean_calibrated_generation_accuracy": statistics.mean(
            run["calibrated_cascade_generation"]["byte_accuracy"] for run in runs
        ),
        "mean_calibrated_generation_neural_fraction": statistics.mean(
            run["calibrated_cascade_generation"]["neural_fraction"] for run in runs
        ),
        "mean_neural_generation_wall_seconds": statistics.mean(
            run["neural_generation"]["wall_seconds"] for run in runs
        ),
        "mean_cascade_generation_wall_seconds": statistics.mean(
            run["cascade_generation"]["wall_seconds"] for run in runs
        ),
    }
    payload = {
        "schema_version": 1,
        "experiment": "aira-v2-dynamic-bpe-byte-event-core-v1",
        "tag": args.tag,
        "warning": "300-step bounded event core. Python BPE merging/shelf lookup is unfused, and FP32 parameter-byte counts are a batch-1 traffic proxy, not device energy.",
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
            "shelf_train_tokens": args.shelf_train_tokens,
            "shelf_train_bytes": len(shelf_stream),
            "neural_train_tokens": args.neural_train_tokens,
            "neural_train_bytes": len(neural_stream),
            "training_examples": len(train_contexts),
            "validation_examples": len(validation_contexts),
            "raw_context_bytes": args.raw_context_bytes,
            "dynamic_bpe_context_tokens": args.token_context,
            "d_model": args.d_model,
            "batch_size": args.batch_size,
            "generation_calibration_sequences": len(calibration_starts),
            "generation_test_sequences": len(starts),
            "generation_horizon_bytes": args.horizon,
            "trigger": "orders 4/8/16, support>=5, per-byte Wilson95>=0.95; autonomous strict UTF-8, max burst 4, cumulative risk 0.10, neural anchor 8, cycle guard",
        },
        "shelf": {
            "packed_bytes": sum(level.packed_bytes for level in levels),
            "build_seconds": shelf_build_seconds,
        },
        "example_preparation_seconds": preparation_seconds,
        "episodic_memory_trace": episodic_trace(),
        "oracle_fallback_generation": oracle_generation,
        "summary": summary,
        "runs": runs,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
