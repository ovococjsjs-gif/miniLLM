#!/usr/bin/env python3
"""Compare full, hard-filtered, and soft-residual neural training behind a shelf."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from minillm.aira import (
    build_compact_shelf,
    lookup_compact_level,
    residual_training_weights,
)


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class TinyContextLM(nn.Module):
    def __init__(self, vocab_size: int = 64, context: int = 16) -> None:
        super().__init__()
        self.context = context
        self.embedding = nn.Embedding(vocab_size, 16)
        self.hidden = nn.Linear(context * 16, 96)
        self.output = nn.Linear(96, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids).flatten(1)
        return self.output(torch.tanh(self.hidden(embedded)))


@dataclass(frozen=True)
class RoutedStream:
    tokens: np.ndarray
    positions: np.ndarray
    shelf_gate: np.ndarray
    target_probabilities: np.ndarray
    shelf_predictions: np.ndarray


def char_mapping(text: str) -> dict[str, int]:
    counts = Counter(text)
    selected = sorted(
        symbol
        for symbol, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[
            :60
        ]
    )
    return {symbol: index + 4 for index, symbol in enumerate(selected)}


def encode(text: str, mapping: dict[str, int]) -> np.ndarray:
    return np.fromiter((mapping.get(symbol, 3) for symbol in text), dtype=np.uint32)


def route_stream(
    tokens: np.ndarray,
    shelf,
    *,
    context: int,
    minimum_support: int = 5,
    confidence_threshold: float = 0.95,
    vocab_size: int = 64,
) -> RoutedStream:
    lookup = lookup_compact_level(shelf, tokens)
    positions = np.arange(max(context, shelf.order), len(tokens))
    offset = positions - shelf.order
    found = lookup["found"][offset]
    totals = lookup["totals"][offset]
    top_counts = lookup["top_counts"][offset]
    predictions = lookup["top_tokens"][offset]
    empirical = np.divide(
        top_counts,
        totals,
        out=np.zeros_like(top_counts, dtype=np.float64),
        where=totals > 0,
    )
    gate = found & (totals >= minimum_support) & (empirical >= confidence_threshold)
    targets = tokens[positions]
    tail = np.divide(
        totals - top_counts,
        totals * (vocab_size - 1),
        out=np.full_like(empirical, 1 / vocab_size),
        where=totals > 0,
    )
    target_probabilities = np.where(
        found,
        np.where(targets == predictions, empirical, tail),
        1 / vocab_size,
    )
    return RoutedStream(tokens, positions, gate, target_probabilities, predictions)


def contexts(stream: RoutedStream, selected: np.ndarray, context: int) -> torch.Tensor:
    return torch.from_numpy(
        np.stack(
            [stream.tokens[position - context : position] for position in selected]
        ).astype(np.int64)
    )


def evaluate(
    model: TinyContextLM, stream: RoutedStream, *, batch_size: int = 1024
) -> dict[str, float]:
    model.eval()
    neural_loss = neural_correct = hybrid_loss = hybrid_correct = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(stream.positions), batch_size):
            positions = stream.positions[start : start + batch_size]
            token_contexts = contexts(stream, positions, model.context)
            targets = torch.from_numpy(stream.tokens[positions].astype(np.int64))
            logits = model(token_contexts)
            losses = F.cross_entropy(logits, targets, reduction="none").numpy()
            predictions = logits.argmax(dim=-1).numpy()
            gate = stream.shelf_gate[start : start + batch_size]
            shelf_probability = stream.target_probabilities[start : start + batch_size]
            shelf_predictions = stream.shelf_predictions[start : start + batch_size]
            neural_loss += float(losses.sum())
            neural_correct += float(np.sum(predictions == targets.numpy()))
            hybrid_loss += float(
                np.where(
                    gate, -np.log(np.clip(shelf_probability, 1e-12, 1)), losses
                ).sum()
            )
            hybrid_correct += float(
                np.sum(
                    np.where(gate, shelf_predictions, predictions) == targets.numpy()
                )
            )
            count += len(positions)
    return {
        "neural_nll": neural_loss / count,
        "neural_perplexity": math.exp(neural_loss / count),
        "neural_accuracy": neural_correct / count,
        "hybrid_nll": hybrid_loss / count,
        "hybrid_perplexity": math.exp(hybrid_loss / count),
        "hybrid_accuracy": hybrid_correct / count,
        "shelf_duty": float(stream.shelf_gate.mean()),
        "shelf_accuracy": float(
            np.mean(
                stream.shelf_predictions[stream.shelf_gate]
                == stream.tokens[stream.positions][stream.shelf_gate]
            )
        )
        if stream.shelf_gate.any()
        else 0.0,
    }


def train_variant(
    name: str,
    initial_state: dict[str, torch.Tensor],
    train_stream: RoutedStream,
    validation_stream: RoutedStream,
    *,
    steps: int,
    batch_size: int,
    seed: int,
) -> dict[str, object]:
    model = TinyContextLM()
    model.load_state_dict(copy.deepcopy(initial_state))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    rng = np.random.default_rng(seed)
    weight_sum = token_count = 0.0
    losses = []
    started = time.perf_counter()
    model.train()
    for step in range(steps):
        indices = rng.integers(0, len(train_stream.positions), size=batch_size)
        positions = train_stream.positions[indices]
        token_contexts = contexts(train_stream, positions, model.context)
        targets = torch.from_numpy(train_stream.tokens[positions].astype(np.int64))
        logits = model(token_contexts)
        per_token = F.cross_entropy(logits, targets, reduction="none")
        if name == "full":
            weights = torch.ones_like(per_token)
        elif name == "hard-filter":
            weights = torch.from_numpy(
                (~train_stream.shelf_gate[indices]).astype(np.float32)
            )
        elif name == "soft-residual":
            probabilities = torch.from_numpy(
                np.where(
                    train_stream.shelf_gate[indices],
                    train_stream.target_probabilities[indices],
                    0.0,
                ).astype(np.float32)
            )
            weights = residual_training_weights(probabilities, floor=0.15)
        else:
            raise ValueError(name)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.sum(per_token * weights) / weights.sum().clamp_min(1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        weight_sum += float(weights.sum())
        token_count += len(weights)
        if step in {0, steps // 2, steps - 1}:
            losses.append(float(loss.detach()))
    return {
        "variant": name,
        "seed": seed,
        "steps": steps,
        "training_seconds": time.perf_counter() - started,
        "effective_gradient_weight": weight_sum / token_count,
        "loss_samples": losses,
        "validation": evaluate(model, validation_stream),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument(
        "--steps", type=int, default=300, choices=range(1, 301), metavar="[1-300]"
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 314, 2718])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", default="results/aira_residual_proxy.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    train_path = Path(args.train)
    validation_path = Path(args.validation)
    text = train_path.read_text(encoding="utf-8", errors="replace")
    validation_text = validation_path.read_text(encoding="utf-8", errors="replace")[
        :300_000
    ]
    if len(text) < 1_500_000:
        raise ValueError("residual proxy requires at least 1.5M training characters")
    mapping = char_mapping(text[:1_000_000])
    shelf_tokens = encode(text[:1_000_000], mapping)
    neural_train_tokens = encode(text[1_000_000:1_500_000], mapping)
    validation_tokens = encode(validation_text, mapping)
    shelf = build_compact_shelf(shelf_tokens, 8)
    train_stream = route_stream(neural_train_tokens, shelf, context=16)
    validation_stream = route_stream(validation_tokens, shelf, context=16)

    torch.manual_seed(1234)
    initial_state = TinyContextLM().state_dict()
    variants = ("full", "hard-filter", "soft-residual")
    rows = [
        train_variant(
            variant,
            initial_state,
            train_stream,
            validation_stream,
            steps=args.steps,
            batch_size=args.batch_size,
            seed=seed,
        )
        for seed in args.seeds
        for variant in variants
    ]
    summary = []
    for variant in variants:
        selected = [row for row in rows if row["variant"] == variant]
        summary.append(
            {
                "variant": variant,
                "runs": len(selected),
                "effective_gradient_weight": statistics.mean(
                    float(row["effective_gradient_weight"]) for row in selected
                ),
                "mean_training_seconds": statistics.mean(
                    float(row["training_seconds"]) for row in selected
                ),
                "mean_neural_perplexity": statistics.mean(
                    float(row["validation"]["neural_perplexity"]) for row in selected
                ),
                "mean_hybrid_perplexity": statistics.mean(
                    float(row["validation"]["hybrid_perplexity"]) for row in selected
                ),
                "mean_hybrid_accuracy": statistics.mean(
                    float(row["validation"]["hybrid_accuracy"]) for row in selected
                ),
            }
        )
    payload = {
        "schema_version": 1,
        "experiment": "aira-v2-soft-residual-v1",
        "warning": "300-step character-MLP proxy; compares starvation behavior, not final LM quality.",
        "source": {
            "train": str(train_path),
            "train_sha256": file_sha256(train_path),
            "validation": str(validation_path),
            "validation_sha256": file_sha256(validation_path),
        },
        "protocol": {
            "shelf_build_characters": 1_000_000,
            "neural_train_characters": 500_000,
            "validation_characters": len(validation_text),
            "shelf_order": 8,
            "shelf_gate": "support>=5 and empirical top probability>=0.95",
            "soft_floor": 0.15,
            "soft_scope": "surprise weighting only where the shelf gate accepts; fallback positions retain weight 1",
            "proper_tail_approximation": "remaining count mass uniform over non-top symbols",
        },
        "platform": platform.platform(),
        "torch": torch.__version__,
        "steps": args.steps,
        "seeds": args.seeds,
        "summary": summary,
        "runs": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
