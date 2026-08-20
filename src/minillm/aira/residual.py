"""Soft residual-learning weights that avoid starving the neural fallback."""

from __future__ import annotations

import torch


def residual_training_weights(
    target_probabilities: torch.Tensor,
    *,
    floor: float = 0.1,
    exponent: float = 1.0,
) -> torch.Tensor:
    """Upweight surprising shelf targets while retaining an all-token anchor.

    AIra H-33's hard filtering proved that zeroing easy-token gradients can leave the
    validator unusable. This smooth rule always retains ``floor`` of the ordinary LM loss.
    """

    if not 0 < floor <= 1:
        raise ValueError("floor must be in (0, 1]")
    if exponent <= 0:
        raise ValueError("exponent must be positive")
    if torch.any((target_probabilities < 0) | (target_probabilities > 1)):
        raise ValueError("target probabilities must be in [0, 1]")
    difficulty = (1 - target_probabilities.float()).pow(exponent)
    return floor + (1 - floor) * difficulty
