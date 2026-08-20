"""Compact exact-count trigger shelves with conservative held-out routing.

Unlike speculative drafting, an AIra trigger is allowed to terminate computation early.
That makes calibration, static held-out evaluation, and explicit error budgets mandatory.
This module intentionally separates a static shelf from optional online adaptation.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CompactShelfLevel:
    order: int
    context_hashes: np.ndarray
    top_tokens: np.ndarray
    totals: np.ndarray
    top_counts: np.ndarray

    @property
    def contexts(self) -> int:
        return len(self.context_hashes)

    @property
    def packed_bytes(self) -> int:
        return sum(
            array.nbytes
            for array in (
                self.context_hashes,
                self.top_tokens,
                self.totals,
                self.top_counts,
            )
        )


@dataclass(frozen=True)
class ShelfPrediction:
    token: int
    support: int
    empirical_confidence: float
    lower_confidence: float
    order: int


@dataclass(frozen=True)
class ShelfRoutes:
    predictions: np.ndarray
    supports: np.ndarray
    empirical_confidences: np.ndarray
    lower_confidences: np.ndarray
    orders: np.ndarray

    @property
    def covered(self) -> np.ndarray:
        return self.orders > 0


@dataclass(frozen=True)
class ShelfEvaluation:
    selection: str
    minimum_support: int
    confidence_threshold: float
    confidence_z: float
    evaluated_tokens: int
    covered_tokens: int
    correct_tokens: int
    coverage: float
    accuracy: float | None
    mean_support: float | None
    mean_empirical_confidence: float | None
    mean_correct_burst: float
    p50_correct_burst: float
    p90_correct_burst: float
    maximum_correct_burst: int
    coverage_by_order: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def context_hashes(tokens: np.ndarray, order: int) -> np.ndarray:
    """FNV-like uint64 hashes for contexts immediately preceding each continuation."""

    if tokens.ndim != 1:
        raise ValueError("tokens must be one-dimensional")
    if order < 1:
        raise ValueError("order must be positive")
    if len(tokens) <= order:
        return np.empty(0, dtype=np.uint64)
    length = len(tokens) - order
    hashes = np.full(length, np.uint64(1469598103934665603), dtype=np.uint64)
    prime = np.uint64(1099511628211)
    for offset in range(order):
        hashes ^= tokens[offset : offset + length].astype(
            np.uint64, copy=False
        ) + np.uint64(1)
        hashes *= prime
    return hashes


def build_compact_shelf(tokens: np.ndarray, order: int) -> CompactShelfLevel:
    """Build exact continuation counts and retain top-1 plus its true support."""

    hashes = context_hashes(tokens, order)
    continuations = tokens[order:].astype(np.uint32, copy=False)
    pairs = np.empty(len(hashes), dtype=[("context", "<u8"), ("token", "<u4")])
    pairs["context"] = hashes
    pairs["token"] = continuations
    unique_pairs, pair_counts = np.unique(pairs, return_counts=True)
    del pairs, hashes

    contexts = unique_pairs["context"]
    starts = np.r_[0, np.flatnonzero(contexts[1:] != contexts[:-1]) + 1]
    totals = np.add.reduceat(pair_counts, starts)
    maximums = np.maximum.reduceat(pair_counts, starts)
    repeated_maximums = np.repeat(maximums, np.diff(np.r_[starts, len(pair_counts)]))
    positions = np.arange(len(pair_counts))
    first_maximum = np.minimum.reduceat(
        np.where(pair_counts == repeated_maximums, positions, len(pair_counts)), starts
    )
    return CompactShelfLevel(
        order=order,
        context_hashes=contexts[starts],
        top_tokens=unique_pairs["token"][first_maximum],
        totals=totals.astype(np.int64, copy=False),
        top_counts=maximums.astype(np.int64, copy=False),
    )


def save_compact_shelf(
    path: str | Path,
    levels: list[CompactShelfLevel],
    *,
    tokenizer_sha256: str | None = None,
    representation: str | None = None,
) -> None:
    """Atomically persist a multilevel shelf without Python pickle objects."""

    if not levels or len({level.order for level in levels}) != len(levels):
        raise ValueError("shelf levels must be non-empty with unique orders")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray([1], dtype=np.int64),
        "orders": np.asarray([level.order for level in levels], dtype=np.int64),
    }
    if tokenizer_sha256 is not None:
        if len(tokenizer_sha256) != 64:
            raise ValueError("tokenizer SHA-256 must contain 64 hexadecimal characters")
        try:
            int(tokenizer_sha256, 16)
        except ValueError as error:
            raise ValueError("tokenizer SHA-256 must be hexadecimal") from error
        payload["tokenizer_sha256"] = np.asarray([tokenizer_sha256.lower()])
    if representation is not None:
        if representation not in {"token-ids", "utf8-byte"}:
            raise ValueError("unsupported shelf representation")
        payload["representation"] = np.asarray([representation])
    for index, level in enumerate(levels):
        payload[f"context_hashes_{index}"] = level.context_hashes
        payload[f"top_tokens_{index}"] = level.top_tokens
        payload[f"totals_{index}"] = level.totals
        payload[f"top_counts_{index}"] = level.top_counts
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, output)


def load_compact_shelf(
    path: str | Path,
    *,
    expected_tokenizer_sha256: str | None = None,
    expected_representation: str | None = None,
) -> list[CompactShelfLevel]:
    """Load a shelf, optionally requiring an exact tokenizer identity match."""

    with np.load(Path(path), allow_pickle=False) as payload:
        if payload["schema_version"].tolist() != [1]:
            raise ValueError("unsupported shelf schema")
        if expected_tokenizer_sha256 is not None:
            if "tokenizer_sha256" not in payload:
                raise ValueError("shelf archive does not identify its tokenizer")
            stored_hash = str(payload["tokenizer_sha256"][0])
            if stored_hash != expected_tokenizer_sha256.lower():
                raise ValueError("shelf and generation tokenizers differ")
        if expected_representation is not None:
            if "representation" not in payload:
                raise ValueError("shelf archive does not identify its representation")
            stored_representation = str(payload["representation"][0])
            if stored_representation != expected_representation:
                raise ValueError("shelf archive has the wrong representation")
        orders = payload["orders"].astype(np.int64).tolist()
        levels = [
            CompactShelfLevel(
                order=int(order),
                context_hashes=payload[f"context_hashes_{index}"].astype(
                    np.uint64, copy=True
                ),
                top_tokens=payload[f"top_tokens_{index}"].astype(np.uint32, copy=True),
                totals=payload[f"totals_{index}"].astype(np.int64, copy=True),
                top_counts=payload[f"top_counts_{index}"].astype(np.int64, copy=True),
            )
            for index, order in enumerate(orders)
        ]
    if not levels or len(set(orders)) != len(orders):
        raise ValueError("shelf archive has invalid orders")
    for level in levels:
        lengths = {
            len(level.context_hashes),
            len(level.top_tokens),
            len(level.totals),
            len(level.top_counts),
        }
        invalid_counts = np.any(level.totals < 1) or np.any(
            (level.top_counts < 1) | (level.top_counts > level.totals)
        )
        if (
            level.order < 1
            or len(lengths) != 1
            or invalid_counts
            or np.any(level.context_hashes[1:] <= level.context_hashes[:-1])
        ):
            raise ValueError("shelf archive contains an invalid level")
    return levels


def wilson_lower_bound(
    successes: np.ndarray, totals: np.ndarray, z: float
) -> np.ndarray:
    probability = successes / totals
    if z == 0:
        return probability
    denominator = 1 + z**2 / totals
    centre = probability + z**2 / (2 * totals)
    margin = z * np.sqrt(
        probability * (1 - probability) / totals + z**2 / (4 * totals**2)
    )
    return (centre - margin) / denominator


def lookup_compact_level(
    level: CompactShelfLevel, tokens: np.ndarray
) -> dict[str, np.ndarray]:
    """Look up one frozen shelf level at every continuation position."""

    queries = context_hashes(tokens, level.order)
    if not level.contexts:
        zeros = np.zeros(len(queries), dtype=np.int64)
        return {
            "positions": np.arange(level.order, len(tokens)),
            "found": np.zeros(len(queries), dtype=bool),
            "top_tokens": zeros.astype(np.uint32),
            "totals": zeros,
            "top_counts": zeros,
        }
    indices = np.searchsorted(level.context_hashes, queries)
    safe = np.minimum(indices, len(level.context_hashes) - 1)
    found = (indices < len(level.context_hashes)) & (
        level.context_hashes[safe] == queries
    )
    return {
        "positions": np.arange(level.order, len(tokens)),
        "found": found,
        "top_tokens": level.top_tokens[safe],
        "totals": level.totals[safe],
        "top_counts": level.top_counts[safe],
    }


def _single_context_hash(tokens: np.ndarray) -> np.uint64:
    value = 1469598103934665603
    for token in tokens:
        value ^= int(token) + 1
        value = (value * 1099511628211) & ((1 << 64) - 1)
    return np.uint64(value)


def _scalar_wilson_lower(successes: int, total: int, z: float) -> float:
    probability = successes / total
    if z == 0:
        return probability
    denominator = 1 + z**2 / total
    centre = probability + z**2 / (2 * total)
    margin = z * math.sqrt(
        probability * (1 - probability) / total + z**2 / (4 * total**2)
    )
    return (centre - margin) / denominator


def predict_shelf_next(
    levels: list[CompactShelfLevel],
    prefix: np.ndarray,
    *,
    minimum_support: int,
    confidence_threshold: float,
    confidence_z: float = 1.96,
    selection: str = "longest",
) -> ShelfPrediction | None:
    """Predict one continuation without scanning or allocating for earlier positions."""

    if selection not in {"longest", "confidence"}:
        raise ValueError("selection must be longest or confidence")
    if minimum_support < 1 or not 0 < confidence_threshold <= 1:
        raise ValueError("invalid trigger gate")
    if not levels:
        raise ValueError("at least one shelf level is required")
    best: ShelfPrediction | None = None
    best_score = -math.inf
    for level in sorted(levels, key=lambda item: item.order):
        if len(prefix) < level.order or not level.contexts:
            continue
        query = _single_context_hash(prefix[-level.order :])
        index = int(np.searchsorted(level.context_hashes, query))
        if index >= level.contexts or level.context_hashes[index] != query:
            continue
        support = int(level.totals[index])
        top_count = int(level.top_counts[index])
        empirical = top_count / support
        lower = _scalar_wilson_lower(top_count, support, confidence_z)
        if support < minimum_support or lower < confidence_threshold:
            continue
        candidate = ShelfPrediction(
            token=int(level.top_tokens[index]),
            support=support,
            empirical_confidence=empirical,
            lower_confidence=lower,
            order=level.order,
        )
        score = float(level.order) if selection == "longest" else lower
        if score > best_score or (
            score == best_score and (best is None or level.order > best.order)
        ):
            best = candidate
            best_score = score
    return best


def _burst_statistics(mask: np.ndarray) -> tuple[float, float, float, int]:
    padded = np.r_[False, mask, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    lengths = changes[1::2] - changes[::2]
    if not len(lengths):
        return 0.0, 0.0, 0.0, 0
    return (
        float(lengths.mean()),
        float(np.quantile(lengths, 0.5)),
        float(np.quantile(lengths, 0.9)),
        int(lengths.max()),
    )


def route_hierarchical_shelf(
    levels: list[CompactShelfLevel],
    tokens: np.ndarray,
    *,
    minimum_support: int,
    confidence_threshold: float,
    confidence_z: float = 1.96,
    selection: str = "longest",
) -> ShelfRoutes:
    """Route tokens through the longest or statistically strongest shelf match."""

    if selection not in {"longest", "confidence"}:
        raise ValueError("selection must be longest or confidence")
    if minimum_support < 1 or not 0 < confidence_threshold <= 1:
        raise ValueError("invalid trigger gate")
    if not levels:
        raise ValueError("at least one shelf level is required")
    if len({level.order for level in levels}) != len(levels):
        raise ValueError("shelf orders must be unique")

    best_score = np.full(len(tokens), -math.inf, dtype=np.float64)
    best_order = np.zeros(len(tokens), dtype=np.int16)
    predictions = np.zeros(len(tokens), dtype=np.uint32)
    supports = np.zeros(len(tokens), dtype=np.int64)
    confidences = np.zeros(len(tokens), dtype=np.float64)
    lower_confidences = np.zeros(len(tokens), dtype=np.float64)
    for level in sorted(levels, key=lambda item: item.order):
        empirical = level.top_counts / level.totals
        lower = wilson_lower_bound(level.top_counts, level.totals, confidence_z)
        accepted = (level.totals >= minimum_support) & (lower >= confidence_threshold)
        hashes = level.context_hashes[accepted]
        if not len(hashes):
            continue
        queries = context_hashes(tokens, level.order)
        indices = np.searchsorted(hashes, queries)
        safe = np.minimum(indices, len(hashes) - 1)
        found = (indices < len(hashes)) & (hashes[safe] == queries)
        positions = np.arange(level.order, len(tokens))
        candidate_score = (
            np.full(len(queries), level.order, dtype=np.float64)
            if selection == "longest"
            else lower[accepted][safe]
        )
        improve = found & (
            (candidate_score > best_score[positions])
            | (
                (candidate_score == best_score[positions])
                & (level.order > best_order[positions])
            )
        )
        selected = positions[improve]
        best_score[selected] = candidate_score[improve]
        best_order[selected] = level.order
        predictions[selected] = level.top_tokens[accepted][safe][improve]
        supports[selected] = level.totals[accepted][safe][improve]
        confidences[selected] = empirical[accepted][safe][improve]
        lower_confidences[selected] = lower[accepted][safe][improve]
    return ShelfRoutes(
        predictions=predictions,
        supports=supports,
        empirical_confidences=confidences,
        lower_confidences=lower_confidences,
        orders=best_order,
    )


def evaluate_hierarchical_shelf(
    levels: list[CompactShelfLevel],
    validation: np.ndarray,
    *,
    minimum_support: int,
    confidence_threshold: float,
    confidence_z: float = 1.96,
    selection: str = "longest",
) -> ShelfEvaluation:
    """Evaluate a frozen shelf without adapting on validation.

    ``longest`` implements AIra's most-specific reliable backoff. ``confidence`` selects
    the strongest statistical lower bound and uses order only as a tie-breaker.
    """

    routes = route_hierarchical_shelf(
        levels,
        validation,
        minimum_support=minimum_support,
        confidence_threshold=confidence_threshold,
        confidence_z=confidence_z,
        selection=selection,
    )
    covered = routes.covered
    correct = covered & (routes.predictions == validation)
    covered_tokens = int(covered.sum())
    correct_tokens = int(correct.sum())
    burst_mean, burst_p50, burst_p90, burst_maximum = _burst_statistics(correct)
    return ShelfEvaluation(
        selection=selection,
        minimum_support=minimum_support,
        confidence_threshold=confidence_threshold,
        confidence_z=confidence_z,
        evaluated_tokens=len(validation),
        covered_tokens=covered_tokens,
        correct_tokens=correct_tokens,
        coverage=covered_tokens / len(validation),
        accuracy=correct_tokens / covered_tokens if covered_tokens else None,
        mean_support=float(routes.supports[covered].mean()) if covered_tokens else None,
        mean_empirical_confidence=float(routes.empirical_confidences[covered].mean())
        if covered_tokens
        else None,
        mean_correct_burst=burst_mean,
        p50_correct_burst=burst_p50,
        p90_correct_burst=burst_p90,
        maximum_correct_burst=burst_maximum,
        coverage_by_order={
            str(level.order): int(np.sum(routes.orders == level.order))
            for level in levels
        },
    )
