from __future__ import annotations

import torch

from minillm.aira.residual import (
    importance_sampled_residual_loss,
    sample_residual_positions,
)


def test_residual_sampler_keeps_every_surprise_position() -> None:
    predictable = torch.tensor([True, False, True, False, True])
    generator = torch.Generator().manual_seed(4)
    sample = sample_residual_positions(
        predictable, control_probability=0.25, generator=generator
    )

    assert sample.selected[~predictable].all()
    assert torch.all(sample.inclusion_probabilities[~predictable] == 1)
    assert torch.all(sample.inclusion_probabilities[predictable] == 0.25)


def test_importance_sampled_loss_is_unbiased_in_expectation() -> None:
    predictable = torch.tensor([True, True, True, False, False])
    losses = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0])
    generator = torch.Generator().manual_seed(9)
    estimates = []
    for _ in range(4000):
        sample = sample_residual_positions(
            predictable, control_probability=0.2, generator=generator
        )
        estimates.append(importance_sampled_residual_loss(losses, sample))

    estimate = torch.stack(estimates).mean()
    torch.testing.assert_close(estimate, losses.mean(), rtol=0.03, atol=0.03)


def test_full_control_probability_recovers_dense_loss_exactly() -> None:
    losses = torch.tensor([0.2, 0.7, 1.3])
    sample = sample_residual_positions(
        torch.ones(3, dtype=torch.bool), control_probability=1.0
    )
    actual = importance_sampled_residual_loss(losses, sample)

    torch.testing.assert_close(actual, losses.mean())
