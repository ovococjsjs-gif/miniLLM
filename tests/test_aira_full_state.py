from __future__ import annotations

import pytest
import torch

from minillm.aira.full_state import (
    ConvRowUpdater,
    GatedDeltaParameters,
    GatedDeltaParameterUpdater,
)


def test_conv_row_updater_starts_as_copy_and_learns_residual() -> None:
    model = ConvRowUpdater(
        event_dim=12,
        layers=2,
        row_width=32,
        hidden_dim=16,
        bottleneck_dim=4,
        identity_dim=3,
    )
    previous = torch.randn(5, 32)
    output = model(previous, torch.randn(5, 12), torch.tensor([0, 1, 0, 1, 0]))

    assert torch.equal(output.row, previous)
    assert torch.equal(output.delta, torch.zeros_like(previous))
    loss = (output.row - torch.randn_like(output.row)).square().mean()
    loss.backward()
    assert model.delta_heads[0][-1].weight.grad is not None


def test_gated_delta_application_matches_explicit_transition() -> None:
    torch.manual_seed(3)
    state = torch.randn(2, 3, 8, 8)
    key = torch.nn.functional.normalize(torch.randn(2, 3, 8), dim=-1)
    value = torch.randn(2, 3, 8)
    gate = -torch.rand(2, 3)
    beta = torch.rand(2, 3)
    parameters = GatedDeltaParameters(key, value, gate, beta)

    actual = GatedDeltaParameterUpdater.apply(state, parameters)
    expected = torch.empty_like(actual)
    for batch in range(2):
        for head in range(3):
            decayed = state[batch, head] * gate[batch, head].exp()
            state_key = decayed @ key[batch, head]
            innovation = (value[batch, head] - state_key) * beta[batch, head]
            expected[batch, head] = decayed + torch.outer(innovation, key[batch, head])

    assert torch.allclose(actual, expected, atol=1e-6)


def test_gated_delta_predictor_emits_constrained_parameters_and_gradients() -> None:
    model = GatedDeltaParameterUpdater(
        event_dim=12,
        layers=2,
        heads=3,
        state_width=8,
        hidden_dim=16,
        identity_dim=4,
    )
    parameters = model(torch.randn(5, 12), torch.tensor([0, 1, 0, 1, 0]))
    state = torch.randn(5, 3, 8, 8)
    updated = model.apply(state, parameters)
    updated.square().mean().backward()

    assert parameters.key.shape == parameters.value.shape == (5, 3, 8)
    assert parameters.gate.shape == parameters.beta.shape == (5, 3)
    assert updated.shape == state.shape
    assert torch.allclose(parameters.key.norm(dim=-1), torch.ones(5, 3), atol=1e-5)
    assert torch.all(parameters.gate <= 0)
    assert torch.all((parameters.beta > 0) & (parameters.beta < 1))
    assert model.parameter_heads[0].weight.grad is not None


def test_gated_delta_predictor_rejects_invalid_layer_ids() -> None:
    model = GatedDeltaParameterUpdater(
        event_dim=4,
        layers=1,
        heads=2,
        state_width=8,
        hidden_dim=16,
        identity_dim=2,
    )
    with pytest.raises(ValueError, match="outside"):
        model(torch.zeros(1, 4), torch.ones(1, dtype=torch.long))
