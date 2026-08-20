#!/usr/bin/env python3
"""Train and test a byte-shelf -> BPE event-neural AIra vertical slice."""

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
from torch.nn import functional as F

from minillm.aira import (
    ByteBPEBridge,
    EpisodicFactStore,
    EventContextLM,
    build_compact_shelf,
)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def contexts_at(
    tokens: np.ndarray, positions: np.ndarray, context_size: int
) -> torch.Tensor:
    offsets = np.arange(context_size, 0, -1)
    return torch.from_numpy(tokens[positions[:, None] - offsets].astype(np.int64))


def train_model(
    tokens: np.ndarray,
    *,
    vocab_size: int,
    context_size: int,
    d_model: int,
    first_position: int,
    last_position: int,
    steps: int,
    batch_size: int,
    seed: int,
) -> tuple[EventContextLM, dict[str, object]]:
    torch.manual_seed(seed)
    model = EventContextLM(vocab_size, context_size, d_model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    rng = np.random.default_rng(seed)
    samples = []
    started = time.perf_counter()
    model.train()
    for step in range(steps):
        positions = rng.integers(first_position, last_position, size=batch_size)
        token_contexts = contexts_at(tokens, positions, context_size)
        targets = torch.from_numpy(tokens[positions].astype(np.int64))
        logits = model(token_contexts)
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in {0, steps // 2, steps - 1}:
            samples.append(float(loss.detach()))
    return model.eval(), {
        "seed": seed,
        "steps": steps,
        "training_seconds": time.perf_counter() - started,
        "loss_samples": samples,
    }


def evaluate_neural(
    model: EventContextLM,
    tokens: np.ndarray,
    *,
    positions: np.ndarray,
    batch_size: int = 512,
) -> dict[str, float]:
    loss_sum = correct = 0.0
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(positions), batch_size):
            selected = positions[start : start + batch_size]
            contexts = contexts_at(tokens, selected, model.context_size)
            targets = torch.from_numpy(tokens[selected].astype(np.int64))
            logits = model(contexts)
            loss_sum += float(F.cross_entropy(logits, targets, reduction="sum"))
            correct += float(torch.sum(logits.argmax(dim=-1) == targets))
    seconds = time.perf_counter() - started
    nll = loss_sum / len(positions)
    return {
        "tokens": len(positions),
        "nll": nll,
        "perplexity": math.exp(nll),
        "accuracy": correct / len(positions),
        "seconds": seconds,
        "tokens_per_second": len(positions) / seconds,
    }


def repeats_cycle(tokens: list[int], candidate: int) -> bool:
    sequence = tokens + [candidate]
    for period in range(1, min(8, len(sequence) // 3) + 1):
        suffix = sequence[-period:]
        if all(
            sequence[-repeat * period : -(repeat - 1) * period] == suffix
            for repeat in range(2, 4)
        ):
            return True
    return False


def route_allowed(
    candidate,
    token_prefix: list[int],
    *,
    burst: int,
    risk: float,
    since_neural: int,
) -> bool:
    return not (
        burst >= 4
        or risk + candidate.cumulative_risk > 0.25
        or since_neural >= 8
        or repeats_cycle(token_prefix, candidate.token)
    )


def shelf_boundary_diagnostic(
    bridge: ByteBPEBridge,
    levels,
    validation: np.ndarray,
    positions: np.ndarray,
) -> dict[str, float | int]:
    accepted = exact = byte_prefix = predicted_bytes = 0
    selected = set(map(int, positions))
    prefix = bytearray()
    for position in range(int(positions[-1]) + 1):
        if position in selected:
            candidate = bridge.draft_token(
                levels,
                prefix,
                minimum_support=5,
                confidence_threshold=0.95,
                confidence_z=1.96,
            )
            if candidate is not None:
                accepted += 1
                exact += candidate.token == int(validation[position])
                future = bridge.tokens_to_bytes(validation[position : position + 16])
                byte_prefix += future.startswith(candidate.token_bytes)
                predicted_bytes += len(candidate.token_bytes)
        piece = bridge.token_bytes[int(validation[position])]
        if piece is None:
            prefix.clear()
        else:
            prefix.extend(piece)
    return {
        "boundaries": len(positions),
        "accepted": accepted,
        "coverage": accepted / len(positions),
        "canonical_token_precision": exact / accepted if accepted else 0.0,
        "byte_prefix_precision": byte_prefix / accepted if accepted else 0.0,
        "mean_candidate_bytes": predicted_bytes / accepted if accepted else 0.0,
    }


def generate_sequences(
    model: EventContextLM | None,
    bridge: ByteBPEBridge,
    levels,
    validation: np.ndarray,
    starts: np.ndarray,
    *,
    horizon_bytes: int,
    use_shelf: bool,
    oracle_fallback: bool,
) -> dict[str, float | int]:
    total_correct = common_prefix_sum = completed = 0
    shelf_events = neural_events = control_rejections = stalls = 0
    emitted_bytes = 0
    model_seconds = 0.0
    special_ids = [
        index for index, piece in enumerate(bridge.token_bytes) if piece is None
    ]
    started = time.perf_counter()
    for start in starts:
        token_prefix = (
            validation[start - model.context_size : start].astype(int).tolist()
            if model is not None
            else validation[start - 16 : start].astype(int).tolist()
        )
        byte_context = bytearray(bridge.suffix_bytes_after_special(token_prefix))
        reference = bridge.tokens_to_bytes(validation[start : start + 256])
        if len(reference) < horizon_bytes:
            raise ValueError("validation continuation is too short")
        output = bytearray()
        burst = 0
        risk = 0.0
        since_neural = 0
        events = 0
        while len(output) < horizon_bytes and events < horizon_bytes * 4:
            candidate = None
            if use_shelf:
                candidate = bridge.draft_token(
                    levels,
                    byte_context,
                    minimum_support=5,
                    confidence_threshold=0.95,
                    confidence_z=1.96,
                )
                if candidate is not None and not route_allowed(
                    candidate,
                    token_prefix,
                    burst=burst,
                    risk=risk,
                    since_neural=since_neural,
                ):
                    candidate = None
                    control_rejections += 1
            if candidate is not None:
                token = candidate.token
                piece = candidate.token_bytes
                shelf_events += 1
                burst += 1
                risk += candidate.cumulative_risk
            else:
                if oracle_fallback:
                    next_byte = reference[min(len(output), horizon_bytes - 1)]
                    token = bridge.exact_tokens[bytes([next_byte])]
                    piece = bytes([next_byte])
                else:
                    assert model is not None
                    context = torch.tensor(
                        [token_prefix[-model.context_size :]], dtype=torch.long
                    )
                    model_started = time.perf_counter()
                    with torch.inference_mode():
                        logits = model(context)[0]
                        logits[special_ids] = -torch.inf
                        token = int(logits.argmax())
                    model_seconds += time.perf_counter() - model_started
                    piece = bridge.token_bytes[token] or b""
                neural_events += 1
                burst = 0
                risk = 0.0
                since_neural = 0
            if not piece:
                stalls += 1
                break
            token_prefix.append(token)
            byte_context.extend(piece)
            output.extend(piece)
            since_neural += 1
            events += 1
        evaluated = bytes(output[:horizon_bytes])
        emitted_bytes += len(evaluated)
        total_correct += sum(a == b for a, b in zip(evaluated, reference))
        common = 0
        for actual, expected in zip(evaluated, reference):
            if actual != expected:
                break
            common += 1
        common_prefix_sum += common
        completed += len(evaluated) == horizon_bytes
    wall_seconds = time.perf_counter() - started
    total_target_bytes = len(starts) * horizon_bytes
    parameter_bytes = model.parameter_bytes if model is not None else 0
    return {
        "sequences": len(starts),
        "target_bytes": total_target_bytes,
        "emitted_bytes": emitted_bytes,
        "byte_accuracy": total_correct / total_target_bytes,
        "mean_common_prefix_bytes": common_prefix_sum / len(starts),
        "complete_horizon_rate": completed / len(starts),
        "shelf_events": shelf_events,
        "neural_or_oracle_events": neural_events,
        "shelf_event_fraction": shelf_events / max(1, shelf_events + neural_events),
        "control_rejections": control_rejections,
        "stalls": stalls,
        "wall_seconds": wall_seconds,
        "model_seconds": model_seconds,
        "model_parameter_bytes_read_proxy": neural_events * parameter_bytes,
        "model_parameter_bytes_per_target_byte": neural_events
        * parameter_bytes
        / total_target_bytes,
    }


def memory_trace() -> dict[str, object]:
    store = EpisodicFactStore(capacity=8, dimension=128)
    slot = store.remember(
        "user.favorite-color",
        "green",
        provenance={"source": "turn-7", "editable": True},
    )
    known = store.recall("USER favorite color")
    unknown = store.recall("user.favorite-food")
    store.remember(
        "user.favorite-color",
        "blue",
        provenance={"source": "turn-9", "conflict": True},
    )
    conflict = store.recall("user favorite color")
    deleted = store.delete(slot)
    return {
        "known_accepted": known.accepted,
        "known_value": known.payload,
        "known_provenance": known.provenance,
        "unknown_accepted": unknown.accepted,
        "conflict_accepted": conflict.accepted,
        "conflict_margin": conflict.margin,
        "delete_succeeded": deleted,
        "scan_operations_after_trace": store.memory.scan_operations(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-tokens", required=True)
    parser.add_argument("--validation-tokens", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--steps", type=int, default=300, choices=range(1, 301))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 314, 2718])
    parser.add_argument("--shelf-train-tokens", type=int, default=200_000)
    parser.add_argument("--neural-train-tokens", type=int, default=800_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--context-size", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--evaluation-tokens", type=int, default=20_000)
    parser.add_argument("--generation-sequences", type=int, default=100)
    parser.add_argument("--horizon-bytes", type=int, default=64)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", default="results/aira_token_bridge_proxy.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    train_path = Path(args.train_tokens)
    validation_path = Path(args.validation_tokens)
    tokenizer_path = Path(args.tokenizer)
    train = np.memmap(train_path, dtype=np.uint32, mode="r")
    validation = np.memmap(validation_path, dtype=np.uint32, mode="r")
    bridge = ByteBPEBridge.from_tokenizer_json(tokenizer_path)
    shelf_source = bridge.tokens_to_bytes(train[: args.shelf_train_tokens])
    shelf_units = np.frombuffer(shelf_source, dtype=np.uint8).astype(np.uint32)
    shelf_started = time.perf_counter()
    levels = [build_compact_shelf(shelf_units, order) for order in (4, 8, 16)]
    shelf_build_seconds = time.perf_counter() - shelf_started
    first_train = args.shelf_train_tokens + args.context_size
    last_train = min(len(train), args.shelf_train_tokens + args.neural_train_tokens)
    evaluation_positions = np.arange(
        args.context_size,
        min(len(validation), args.context_size + args.evaluation_tokens),
    )
    diagnostic_positions = evaluation_positions[::4]
    rng = np.random.default_rng(1234)
    generation_population = np.arange(args.context_size, len(validation) - 512)
    starts = np.sort(
        rng.choice(
            generation_population,
            min(args.generation_sequences, len(generation_population)),
            replace=False,
        )
    )
    shelf_diagnostic = shelf_boundary_diagnostic(
        bridge, levels, validation, diagnostic_positions
    )
    runs = []
    for seed in args.seeds:
        model, training = train_model(
            train,
            vocab_size=bridge.vocab_size,
            context_size=args.context_size,
            d_model=args.d_model,
            first_position=first_train,
            last_position=last_train,
            steps=args.steps,
            batch_size=args.batch_size,
            seed=seed,
        )
        neural_evaluation = evaluate_neural(
            model, validation, positions=evaluation_positions
        )
        neural_generation = generate_sequences(
            model,
            bridge,
            levels,
            validation,
            starts,
            horizon_bytes=args.horizon_bytes,
            use_shelf=False,
            oracle_fallback=False,
        )
        cascade_generation = generate_sequences(
            model,
            bridge,
            levels,
            validation,
            starts,
            horizon_bytes=args.horizon_bytes,
            use_shelf=True,
            oracle_fallback=False,
        )
        runs.append(
            {
                **training,
                "model_parameters": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "model_parameter_bytes": model.parameter_bytes,
                "neural_validation": neural_evaluation,
                "neural_generation": neural_generation,
                "cascade_generation": cascade_generation,
            }
        )
    oracle_generation = generate_sequences(
        None,
        bridge,
        levels,
        validation,
        starts,
        horizon_bytes=args.horizon_bytes,
        use_shelf=True,
        oracle_fallback=True,
    )
    summary = {
        "runs": len(runs),
        "mean_neural_validation_perplexity": statistics.mean(
            run["neural_validation"]["perplexity"] for run in runs
        ),
        "mean_neural_only_byte_accuracy": statistics.mean(
            run["neural_generation"]["byte_accuracy"] for run in runs
        ),
        "mean_cascade_byte_accuracy": statistics.mean(
            run["cascade_generation"]["byte_accuracy"] for run in runs
        ),
        "mean_neural_only_model_events": statistics.mean(
            run["neural_generation"]["neural_or_oracle_events"] for run in runs
        ),
        "mean_cascade_model_events": statistics.mean(
            run["cascade_generation"]["neural_or_oracle_events"] for run in runs
        ),
        "mean_cascade_shelf_event_fraction": statistics.mean(
            run["cascade_generation"]["shelf_event_fraction"] for run in runs
        ),
        "mean_neural_only_parameter_bytes_per_target_byte": statistics.mean(
            run["neural_generation"]["model_parameter_bytes_per_target_byte"]
            for run in runs
        ),
        "mean_cascade_parameter_bytes_per_target_byte": statistics.mean(
            run["cascade_generation"]["model_parameter_bytes_per_target_byte"]
            for run in runs
        ),
        "mean_neural_only_wall_seconds": statistics.mean(
            run["neural_generation"]["wall_seconds"] for run in runs
        ),
        "mean_cascade_wall_seconds": statistics.mean(
            run["cascade_generation"]["wall_seconds"] for run in runs
        ),
    }
    payload = {
        "schema_version": 1,
        "experiment": "aira-v2-token-boundary-byte-bpe-negative-control-v1",
        "warning": "Negative control: querying a byte shelf only at canonical BPE-token boundaries loses most raw-byte coverage. The 300-step neural proxy is not product LM quality.",
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
            "shelf_train_bytes": len(shelf_units),
            "neural_train_token_window": [first_train, last_train],
            "context_size": args.context_size,
            "d_model": args.d_model,
            "batch_size": args.batch_size,
            "evaluation_tokens": len(evaluation_positions),
            "generation_sequences": len(starts),
            "horizon_bytes": args.horizon_bytes,
            "trigger": "orders 4/8/16, support>=5, per-byte Wilson95>=0.95, max token burst 4, cumulative byte risk<=0.25, neural anchor 8, cycle guard",
        },
        "shelf": {
            "packed_bytes": sum(level.packed_bytes for level in levels),
            "build_seconds": shelf_build_seconds,
            "boundary_diagnostic": shelf_diagnostic,
        },
        "episodic_memory_trace": memory_trace(),
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
