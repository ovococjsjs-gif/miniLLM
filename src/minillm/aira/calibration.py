"""Held-out precision calibration for autonomous shelf bypass decisions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReliabilityThreshold:
    threshold: float | None
    accepted_examples: int
    correct_examples: int
    empirical_precision: float | None
    precision_lower_bound: float | None
    target_precision: float
    confidence_z: float

    def accept(self, scores: np.ndarray) -> np.ndarray:
        """Apply the fitted threshold, rejecting everything when calibration failed."""

        if self.threshold is None:
            return np.zeros_like(scores, dtype=bool)
        return scores >= self.threshold


def scalar_wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float:
    """Wilson lower confidence bound for a Bernoulli precision estimate."""

    if not 0 <= successes <= total or total < 1:
        raise ValueError("successes and total must describe a non-empty sample")
    probability = successes / total
    if z == 0:
        return probability
    denominator = 1 + z**2 / total
    centre = probability + z**2 / (2 * total)
    margin = z * math.sqrt(
        probability * (1 - probability) / total + z**2 / (4 * total**2)
    )
    return (centre - margin) / denominator


def calibrate_reliability_threshold(
    scores: np.ndarray,
    correct: np.ndarray,
    *,
    target_precision: float = 0.95,
    confidence_z: float = 1.96,
    minimum_accepted: int = 100,
) -> ReliabilityThreshold:
    """Choose maximum-coverage score threshold with a precision lower bound target.

    The caller must use a calibration split independent of both shelf construction and
    final evaluation. Candidate thresholds are grouped at score ties, so the reported
    sample statistics exactly match subsequent ``>= threshold`` routing.
    """

    scores = np.asarray(scores, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    if scores.ndim != 1 or correct.ndim != 1 or scores.shape != correct.shape:
        raise ValueError("scores and correctness must be equal one-dimensional arrays")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite")
    if not 0 < target_precision <= 1 or confidence_z < 0 or minimum_accepted < 1:
        raise ValueError("invalid calibration configuration")
    if not len(scores):
        return ReliabilityThreshold(
            None, 0, 0, None, None, target_precision, confidence_z
        )

    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    cumulative_correct = np.cumsum(correct[order], dtype=np.int64)
    group_ends = np.r_[
        np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), len(scores) - 1
    ]
    best: tuple[float, int, int, float] | None = None
    for end in group_ends:
        accepted = int(end + 1)
        if accepted < minimum_accepted:
            continue
        successes = int(cumulative_correct[end])
        lower = scalar_wilson_lower_bound(successes, accepted, confidence_z)
        if lower >= target_precision:
            best = (float(sorted_scores[end]), accepted, successes, lower)
    if best is None:
        return ReliabilityThreshold(
            None, 0, 0, None, None, target_precision, confidence_z
        )
    threshold, accepted, successes, lower = best
    return ReliabilityThreshold(
        threshold=threshold,
        accepted_examples=accepted,
        correct_examples=successes,
        empirical_precision=successes / accepted,
        precision_lower_bound=lower,
        target_precision=target_precision,
        confidence_z=confidence_z,
    )
