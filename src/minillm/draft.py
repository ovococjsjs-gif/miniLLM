"""Conservative n-gram draft shelf for optional speculative generation.

The shelf is an auxiliary predictor, not a replacement for the neural policy. By default
its candidates must be verified by the main model, so enabling it cannot change greedy
outputs.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class DraftCandidate:
    token_id: int
    confidence_lower_bound: float
    empirical_probability: float
    support: int
    order: int


@dataclass(frozen=True)
class NgramShelfConfig:
    orders: tuple[int, ...] = (4, 8, 16)
    minimum_support: int = 4
    confidence_threshold: float = 0.95
    confidence_z: float = 1.96

    def validate(self) -> NgramShelfConfig:
        if not self.orders or any(order < 1 for order in self.orders):
            raise ValueError("orders must contain positive integers")
        if tuple(sorted(set(self.orders))) != self.orders:
            raise ValueError("orders must be sorted and unique")
        if self.minimum_support < 1:
            raise ValueError("minimum_support must be positive")
        if not 0 < self.confidence_threshold <= 1 or self.confidence_z < 0:
            raise ValueError("invalid confidence settings")
        return self


def wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float:
    if not 0 <= successes <= total or total < 1:
        raise ValueError("invalid Bernoulli counts")
    probability = successes / total
    denominator = 1 + z**2 / total
    centre = probability + z**2 / (2 * total)
    margin = z * math.sqrt(
        probability * (1 - probability) / total + z**2 / (4 * total**2)
    )
    return (centre - margin) / denominator


class NgramDraftShelf:
    """Online exact-context counts with conservative confidence gating."""

    def __init__(self, config: NgramShelfConfig | None = None) -> None:
        self.config = (config or NgramShelfConfig()).validate()
        self.tables: dict[int, dict[tuple[int, ...], Counter[int]]] = {
            order: defaultdict(Counter) for order in self.config.orders
        }
        self.updates = 0

    def update(self, prefix: Sequence[int], next_token: int) -> None:
        for order in self.config.orders:
            if len(prefix) >= order:
                self.tables[order][tuple(prefix[-order:])][int(next_token)] += 1
        self.updates += 1

    def ingest(self, token_ids: Sequence[int]) -> None:
        maximum_order = self.config.orders[-1]
        for position in range(1, len(token_ids)):
            start = max(0, position - maximum_order)
            self.update(token_ids[start:position], int(token_ids[position]))

    def ingest_many(self, sequences: Iterable[Sequence[int]]) -> None:
        for sequence in sequences:
            self.ingest(sequence)

    def query(self, prefix: Sequence[int]) -> DraftCandidate | None:
        candidates: list[DraftCandidate] = []
        for order in reversed(self.config.orders):
            if len(prefix) < order:
                continue
            counts = self.tables[order].get(tuple(prefix[-order:]))
            if not counts:
                continue
            token, successes = min(counts.items(), key=lambda item: (-item[1], item[0]))
            total = sum(counts.values())
            if total < self.config.minimum_support:
                continue
            lower = wilson_lower_bound(successes, total, self.config.confidence_z)
            candidates.append(
                DraftCandidate(token, lower, successes / total, total, order)
            )
        if not candidates:
            return None
        # Confidence first, then longer context and support. This avoids blindly
        # trusting a unique long context over a well-supported shorter one.
        best = max(
            candidates,
            key=lambda item: (item.confidence_lower_bound, item.order, item.support),
        )
        return (
            best
            if best.confidence_lower_bound >= self.config.confidence_threshold
            else None
        )

    def number_of_contexts(self) -> int:
        return sum(len(table) for table in self.tables.values())


def verify_greedy_candidate(logits: Sequence[float], candidate: DraftCandidate) -> bool:
    """Lossless greedy verification: accept only if the main model agrees."""

    if len(logits) == 0 or not 0 <= candidate.token_id < len(logits):
        return False
    neural_argmax = max(range(len(logits)), key=lambda index: logits[index])
    return neural_argmax == candidate.token_id
