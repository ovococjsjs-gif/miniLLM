"""Reference finite-inference Predictive Coding and PC-ALM.

This implementation uses autograd to validate the mathematics. It is not the intended
local runtime: every activity gradient depends only on neighbouring layers analytically,
but writing that fused kernel comes after the reference matches BP.

Paper: Augmented Lagrangian Predictive Coding (2026), arXiv:2605.31022.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch.nn import functional as F

Activation = Literal["linear", "tanh", "relu"]


@dataclass(frozen=True)
class LocalGradientResult:
    gradients: tuple[torch.Tensor, ...]
    residual_norm: float
    inference_steps: int
    dual_rate: float


def _activate(value: torch.Tensor, kind: Activation) -> torch.Tensor:
    if kind == "linear":
        return value
    if kind == "tanh":
        return torch.tanh(value)
    if kind == "relu":
        return F.relu(value)
    raise ValueError(f"unknown activation: {kind}")


def _forward_states(
    weights: tuple[torch.Tensor, ...],
    inputs: torch.Tensor,
    activation: Activation,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    hidden = inputs
    states = []
    for weight in weights[:-1]:
        hidden = _activate(hidden @ weight.T, activation)
        states.append(hidden)
    return states, hidden @ weights[-1].T


def _augmented_lagrangian(
    weights: tuple[torch.Tensor, ...],
    inputs: torch.Tensor,
    targets: torch.Tensor,
    states: list[torch.Tensor],
    duals: list[torch.Tensor],
    *,
    rho: float,
    activation: Activation,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    previous = inputs
    residuals = []
    for weight, state in zip(weights[:-1], states):
        prediction = _activate(previous @ weight.T, activation)
        residuals.append(state - prediction)
        previous = state
    outputs = previous @ weights[-1].T
    objective = 0.5 * (outputs - targets).square().sum(dim=-1).mean()
    for residual, dual in zip(residuals, duals):
        objective = objective + (dual * residual).sum(dim=-1).mean()
        objective = objective + 0.5 * rho * residual.square().sum(dim=-1).mean()
    return objective, residuals


def local_augmented_gradients(
    weights: tuple[torch.Tensor, ...],
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    inference_steps: int,
    state_step_size: float,
    dual_rate: float,
    rho: float = 1.0,
    activation: Activation = "tanh",
) -> LocalGradientResult:
    """Return finite-inference PC (`dual_rate=0`) or PC-ALM gradients."""

    if len(weights) < 2:
        raise ValueError("PC-ALM requires at least one hidden and one output layer")
    if inference_steps < 1 or state_step_size <= 0 or dual_rate < 0 or rho <= 0:
        raise ValueError("invalid PC-ALM hyperparameters")
    with torch.no_grad():
        initial_states, _ = _forward_states(weights, inputs, activation)
    states = [state.detach().requires_grad_(True) for state in initial_states]
    duals = [torch.zeros_like(state) for state in states]

    for step in range(inference_steps):
        objective, _ = _augmented_lagrangian(
            weights,
            inputs,
            targets,
            states,
            duals,
            rho=rho,
            activation=activation,
        )
        state_gradients = torch.autograd.grad(objective, states)
        states = [
            (state - state_step_size * gradient).detach().requires_grad_(True)
            for state, gradient in zip(states, state_gradients)
        ]
        if step < inference_steps - 1 and dual_rate:
            with torch.no_grad():
                _, residuals = _augmented_lagrangian(
                    weights,
                    inputs,
                    targets,
                    states,
                    duals,
                    rho=rho,
                    activation=activation,
                )
                duals = [
                    dual + dual_rate * residual
                    for dual, residual in zip(duals, residuals)
                ]

    detached_states = [state.detach() for state in states]
    objective, residuals = _augmented_lagrangian(
        weights,
        inputs,
        targets,
        detached_states,
        [dual.detach() for dual in duals],
        rho=rho,
        activation=activation,
    )
    gradients = torch.autograd.grad(objective, weights)
    residual_norm = max(
        (float(residual.detach().abs().max()) for residual in residuals), default=0.0
    )
    return LocalGradientResult(
        gradients=tuple(gradient.detach() for gradient in gradients),
        residual_norm=residual_norm,
        inference_steps=inference_steps,
        dual_rate=dual_rate,
    )


def backprop_gradients(
    weights: tuple[torch.Tensor, ...],
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    activation: Activation = "tanh",
) -> tuple[torch.Tensor, ...]:
    _, outputs = _forward_states(weights, inputs, activation)
    loss = 0.5 * (outputs - targets).square().sum(dim=-1).mean()
    return tuple(gradient.detach() for gradient in torch.autograd.grad(loss, weights))


def gradient_cosine(
    first: tuple[torch.Tensor, ...], second: tuple[torch.Tensor, ...]
) -> float:
    if len(first) != len(second):
        raise ValueError("gradient collections differ")
    a = torch.cat([value.reshape(-1).float() for value in first])
    b = torch.cat([value.reshape(-1).float() for value in second])
    return float(F.cosine_similarity(a, b, dim=0))


def minimum_layer_cosine(
    first: tuple[torch.Tensor, ...], second: tuple[torch.Tensor, ...]
) -> float:
    if len(first) != len(second):
        raise ValueError("gradient collections differ")
    return min(
        float(F.cosine_similarity(a.reshape(-1).float(), b.reshape(-1).float(), dim=0))
        for a, b in zip(first, second)
    )
