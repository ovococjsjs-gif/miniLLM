#!/usr/bin/env python3
"""Evaluate a compact 64-bit n-gram draft shelf on packed token streams.

This script does no training. It builds exact continuation counts after a 64-bit context
hash, then reports coverage and next-token accuracy under support/confidence gates.
Hash collisions are possible but negligible at this proxy scale and are disclosed in the
output.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ShelfStatistics:
    order: int
    contexts: int
    contexts_with_minimum_support: int
    maximum_support: int


@dataclass(frozen=True)
class GateResult:
    confidence_z: float
    confidence_threshold: float
    minimum_support: int
    covered_tokens: int
    evaluated_tokens: int
    coverage: float
    accuracy: float | None


def context_hashes(tokens: np.ndarray, order: int) -> np.ndarray:
    """FNV-like hashes for every context ending before its continuation token."""

    if order < 1 or len(tokens) <= order:
        return np.empty(0, dtype=np.uint64)
    length = len(tokens) - order
    hashes = np.full(length, np.uint64(1469598103934665603), dtype=np.uint64)
    prime = np.uint64(1099511628211)
    for offset in range(order):
        values = tokens[offset : offset + length].astype(np.uint64, copy=False)
        hashes ^= values + np.uint64(1)
        hashes *= prime
    return hashes


def wilson_lower_bound(
    successes: np.ndarray, total: np.ndarray, z: float
) -> np.ndarray:
    probability = successes / total
    if z == 0:
        return probability
    denominator = 1 + z**2 / total
    centre = probability + z**2 / (2 * total)
    margin = z * np.sqrt(
        probability * (1 - probability) / total + z**2 / (4 * total**2)
    )
    return (centre - margin) / denominator


def build_shelf(
    tokens: np.ndarray, order: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hashes = context_hashes(tokens, order)
    continuations = tokens[order:]
    pairs = np.empty(len(hashes), dtype=[("context", "<u8"), ("token", "<u4")])
    pairs["context"] = hashes
    pairs["token"] = continuations
    unique_pairs, pair_counts = np.unique(pairs, return_counts=True)
    del pairs, hashes

    contexts = unique_pairs["context"]
    starts = np.r_[0, np.flatnonzero(contexts[1:] != contexts[:-1]) + 1]
    totals = np.add.reduceat(pair_counts, starts)
    maximums = np.maximum.reduceat(pair_counts, starts)
    positions = np.arange(len(pair_counts))
    first_maximum = np.minimum.reduceat(
        np.where(
            pair_counts
            == np.repeat(maximums, np.diff(np.r_[starts, len(pair_counts)])),
            positions,
            len(pair_counts),
        ),
        starts,
    )
    return (
        contexts[starts],
        unique_pairs["token"][first_maximum],
        totals.astype(np.int64, copy=False),
        maximums.astype(np.int64, copy=False),
    )


def evaluate_gate(
    shelves: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    validation: np.ndarray,
    *,
    minimum_support: int,
    confidence_threshold: float,
    confidence_z: float,
) -> GateResult:
    best_confidence = np.full(len(validation), -1.0, dtype=np.float64)
    best_order = np.zeros(len(validation), dtype=np.int16)
    predictions = np.zeros(len(validation), dtype=np.uint32)
    for order, (hashes, tokens, totals, successes) in shelves.items():
        lower = wilson_lower_bound(successes, totals, confidence_z)
        accepted = (totals >= minimum_support) & (lower >= confidence_threshold)
        accepted_hashes = hashes[accepted]
        if not len(accepted_hashes):
            continue
        query_hashes = context_hashes(validation, order)
        indices = np.searchsorted(accepted_hashes, query_hashes)
        safe_indices = np.minimum(indices, len(accepted_hashes) - 1)
        found = (indices < len(accepted_hashes)) & (
            accepted_hashes[safe_indices] == query_hashes
        )
        query_positions = np.arange(order, len(validation))
        candidate_confidence = lower[accepted][safe_indices]
        candidate_tokens = tokens[accepted][safe_indices]
        improve = found & (
            (candidate_confidence > best_confidence[query_positions])
            | (
                (candidate_confidence == best_confidence[query_positions])
                & (order > best_order[query_positions])
            )
        )
        selected = query_positions[improve]
        best_confidence[selected] = candidate_confidence[improve]
        best_order[selected] = order
        predictions[selected] = candidate_tokens[improve]

    covered = best_confidence >= 0
    covered_tokens = int(covered.sum())
    accuracy = (
        float(np.mean(predictions[covered] == validation[covered]))
        if covered_tokens
        else None
    )
    return GateResult(
        confidence_z=confidence_z,
        confidence_threshold=confidence_threshold,
        minimum_support=minimum_support,
        covered_tokens=covered_tokens,
        evaluated_tokens=len(validation),
        coverage=covered_tokens / len(validation),
        accuracy=accuracy,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/tokens-4096/train.bin")
    parser.add_argument("--validation", default="data/tokens-4096/validation.bin")
    parser.add_argument("--orders", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--minimum-supports", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.9, 0.95])
    parser.add_argument("--confidence-z", nargs="+", type=float, default=[0.0, 1.96])
    parser.add_argument(
        "--contract", default="configs/experiments/ngram_draft_proxy.json"
    )
    parser.add_argument("--output", default="results/ngram_draft_proxy.json")
    args = parser.parse_args()

    orders = tuple(sorted(set(args.orders)))
    if not orders or orders[0] < 1:
        raise ValueError("orders must be positive")
    train = np.memmap(args.train, mode="r", dtype=np.uint32)
    validation = np.memmap(args.validation, mode="r", dtype=np.uint32)
    started = time.perf_counter()
    shelves: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    statistics: list[ShelfStatistics] = []
    for order in orders:
        shelf = build_shelf(train, order)
        shelves[order] = shelf
        statistics.append(
            ShelfStatistics(
                order=order,
                contexts=len(shelf[0]),
                contexts_with_minimum_support=int(
                    np.sum(shelf[2] >= min(args.minimum_supports))
                ),
                maximum_support=int(shelf[2].max(initial=0)),
            )
        )

    results = [
        evaluate_gate(
            shelves,
            validation,
            minimum_support=support,
            confidence_threshold=threshold,
            confidence_z=z,
        )
        for z in args.confidence_z
        for threshold in args.thresholds
        for support in args.minimum_supports
    ]
    contract_path = Path(args.contract)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    expected_data = contract["data"]
    if (
        expected_data["train"] != args.train
        or expected_data["validation"] != args.validation
        or tuple(expected_data["orders"]) != orders
    ):
        raise ValueError("command does not match preregistered data and orders")
    primary = contract["primary_gate"]
    primary_result = next(
        result
        for result in results
        if result.confidence_z == primary["confidence_z"]
        and result.confidence_threshold == primary["confidence_threshold"]
        and result.minimum_support == primary["minimum_support"]
    )
    primary_passed = (
        primary_result.coverage >= primary["pass"]["coverage_min"]
        and primary_result.accuracy is not None
        and primary_result.accuracy >= primary["pass"]["accuracy_min"]
    )
    report = {
        "method": "64-bit FNV-like hashed contexts; exact continuation counts per hash",
        "train_tokens": len(train),
        "validation_tokens": len(validation),
        "orders": orders,
        "statistics": [asdict(item) for item in statistics],
        "gates": [asdict(item) for item in results],
        "preregistered_contract": {
            "path": str(contract_path),
            "id": contract["id"],
            "status": contract["status"],
        },
        "primary_verdict": {
            "passed": primary_passed,
            "coverage": primary_result.coverage,
            "accuracy": primary_result.accuracy,
            "required_coverage": primary["pass"]["coverage_min"],
            "required_accuracy": primary["pass"]["accuracy_min"],
        },
        "wall_seconds": time.perf_counter() - started,
        "notes": [
            "This measures draft predictability, not end-to-end speculative speedup.",
            "A candidate must still be verified by the neural model to preserve exact greedy output.",
            "Confidence z=0 is empirical frequency; z=1.96 is the Wilson 95% lower bound.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
