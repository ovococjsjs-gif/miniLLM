"""Soft residual-learning weights that avoid starving the neural fallback."""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True)
class ResidualPositionSample:
    selected: torch.Tensor
    inclusion_probabilities: torch.Tensor

    @property
    def effective_fraction(self) -> float:
        return float(self.selected.float().mean())


def sample_residual_positions(
    predictable: torch.Tensor,
    *,
    control_probability: float = 0.2,
    generator: torch.Generator | None = None,
) -> ResidualPositionSample:
    """Keep every event and a random control stream of predictable positions.

    Returned inclusion probabilities support Horvitz-Thompson weighting, so the sampled
    full-loss estimate stays unbiased rather than silently becoming hard filtering.
    """

    if predictable.dtype != torch.bool:
        raise TypeError("predictable mask must be boolean")
    if not 0 < control_probability <= 1:
        raise ValueError("control_probability must be in (0, 1]")
    random_values = torch.rand(
        predictable.shape,
        generator=generator,
        device=predictable.device,
    )
    controls = random_values < control_probability
    selected = ~predictable | controls
    probabilities = torch.where(
        predictable,
        torch.full_like(random_values, control_probability),
        torch.ones_like(random_values),
    )
    return ResidualPositionSample(selected, probabilities)


def importance_sampled_residual_loss(
    per_position_loss: torch.Tensor,
    sample: ResidualPositionSample,
) -> torch.Tensor:
    """Unbiased estimate of the dense mean loss from sampled residual positions."""

    if per_position_loss.shape != sample.selected.shape:
        raise ValueError("loss and residual sample shapes differ")
    if torch.any(sample.inclusion_probabilities <= 0):
        raise ValueError("inclusion probabilities must be positive")
    weighted = torch.where(
        sample.selected,
        per_position_loss / sample.inclusion_probabilities,
        torch.zeros_like(per_position_loss),
    )
    return weighted.sum() / per_position_loss.numel()
