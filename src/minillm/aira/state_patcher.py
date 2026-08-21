"""Cheap recurrent-state catch-up for event spans emitted without a full donor pass."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class StatePatchOutput:
    state: torch.Tensor
    confidence: torch.Tensor
    delta: torch.Tensor


@dataclass(frozen=True)
class StatePatchLoss:
    total: torch.Tensor
    state_mse: torch.Tensor
    state_cosine: torch.Tensor
    future_kl: torch.Tensor
    confidence_bce: torch.Tensor


class RecurrentStatePatcher(nn.Module):
    """Predict omitted per-layer recurrent states from an event and emitted bytes.

    Anchor layers can be held fixed with ``patch_mask=False``. This makes the
    module suitable for hybrid recurrent/attention donors where only skipped
    recurrent groups should be advanced by the cheap path.
    """

    def __init__(
        self,
        *,
        layers: int,
        state_dim: int,
        event_dim: int,
        hidden_dim: int,
        byte_dim: int = 24,
    ) -> None:
        super().__init__()
        if min(layers, state_dim, event_dim, hidden_dim, byte_dim) < 1:
            raise ValueError("state patcher dimensions must be positive")
        self.layers = layers
        self.state_dim = state_dim
        self.event_dim = event_dim
        self.byte_embedding = nn.Embedding(256, byte_dim)
        self.layer_embedding = nn.Embedding(layers, state_dim)
        input_dim = state_dim * 2 + event_dim + byte_dim
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.delta_head = nn.Linear(hidden_dim, state_dim)
        self.gate_head = nn.Linear(hidden_dim, state_dim)
        self.confidence_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        state: torch.Tensor,
        event_features: torch.Tensor,
        emitted_bytes: torch.Tensor,
        *,
        patch_mask: torch.Tensor | None = None,
        emitted_byte_mask: torch.Tensor | None = None,
    ) -> StatePatchOutput:
        if state.ndim != 3 or state.shape[1:] != (self.layers, self.state_dim):
            raise ValueError("state must have shape [batch, layers, state_dim]")
        batch = state.shape[0]
        if event_features.shape != (batch, self.event_dim):
            raise ValueError("event features have the wrong shape")
        if emitted_bytes.ndim != 2 or emitted_bytes.shape[0] != batch:
            raise ValueError("emitted bytes must have shape [batch, span]")
        if emitted_bytes.shape[1] < 1:
            raise ValueError("event spans must contain at least one byte")
        if emitted_bytes.dtype != torch.long:
            raise ValueError("emitted bytes must use torch.long")
        if torch.any((emitted_bytes < 0) | (emitted_bytes > 255)):
            raise ValueError("emitted byte values must lie in [0, 255]")
        if patch_mask is None:
            patch_mask = torch.ones(
                (batch, self.layers), dtype=torch.bool, device=state.device
            )
        if patch_mask.shape != (batch, self.layers) or patch_mask.dtype != torch.bool:
            raise ValueError("patch mask must be boolean [batch, layers]")

        embedded_bytes = self.byte_embedding(emitted_bytes)
        if emitted_byte_mask is None:
            byte_summary = embedded_bytes.mean(dim=1)
        else:
            if (
                emitted_byte_mask.shape != emitted_bytes.shape
                or emitted_byte_mask.dtype != torch.bool
            ):
                raise ValueError("emitted byte mask must be boolean [batch, span]")
            if torch.any(emitted_byte_mask.sum(dim=1) == 0):
                raise ValueError("every event span needs at least one unmasked byte")
            weights = emitted_byte_mask.unsqueeze(-1)
            byte_summary = (embedded_bytes * weights).sum(dim=1) / weights.sum(dim=1)
        layer_ids = torch.arange(self.layers, device=state.device)
        layer_context = self.layer_embedding(layer_ids).unsqueeze(0).expand(batch, -1, -1)
        event_context = event_features.unsqueeze(1).expand(-1, self.layers, -1)
        byte_context = byte_summary.unsqueeze(1).expand(-1, self.layers, -1)
        hidden = self.trunk(
            torch.cat((state, layer_context, event_context, byte_context), dim=-1)
        )
        delta = torch.tanh(self.delta_head(hidden))
        gate = torch.sigmoid(self.gate_head(hidden))
        proposed = state + gate * delta
        mask = patch_mask.unsqueeze(-1)
        patched = torch.where(mask, proposed, state)
        confidence = torch.sigmoid(self.confidence_head(hidden).squeeze(-1))
        confidence = torch.where(patch_mask, confidence, torch.ones_like(confidence))
        applied_delta = torch.where(mask, patched - state, torch.zeros_like(delta))
        return StatePatchOutput(patched, confidence, applied_delta)


def state_patch_loss(
    output: StatePatchOutput,
    target_state: torch.Tensor,
    patch_mask: torch.Tensor,
    *,
    student_future_logits: torch.Tensor | None = None,
    teacher_future_logits: torch.Tensor | None = None,
    cosine_weight: float = 0.2,
    future_kl_weight: float = 0.5,
    confidence_weight: float = 0.1,
    confidence_error_scale: float = 4.0,
) -> StatePatchLoss:
    """Supervise state reconstruction, future behavior, and honest confidence."""

    if target_state.shape != output.state.shape:
        raise ValueError("target state shape differs from patched state")
    if patch_mask.shape != output.confidence.shape or patch_mask.dtype != torch.bool:
        raise ValueError("patch mask must match patch confidence")
    if not torch.any(patch_mask):
        raise ValueError("state patch loss needs at least one patched layer")
    mask = patch_mask.unsqueeze(-1).expand_as(output.state)
    squared_error = (output.state - target_state).square()
    state_mse = squared_error[mask].mean()

    predicted = output.state[patch_mask]
    target = target_state[patch_mask]
    state_cosine = (1 - F.cosine_similarity(predicted, target, dim=-1)).mean()

    if (student_future_logits is None) != (teacher_future_logits is None):
        raise ValueError("student and teacher future logits must be provided together")
    if student_future_logits is None:
        future_kl = output.state.new_zeros(())
    else:
        if student_future_logits.shape != teacher_future_logits.shape:
            raise ValueError("future logit tensors must have equal shape")
        future_kl = F.kl_div(
            F.log_softmax(student_future_logits, dim=-1),
            F.softmax(teacher_future_logits.detach(), dim=-1),
            reduction="batchmean",
        )

    per_layer_error = squared_error.mean(dim=-1).detach()
    confidence_target = torch.exp(-confidence_error_scale * per_layer_error).clamp(0, 1)
    confidence_bce = F.binary_cross_entropy(
        output.confidence[patch_mask], confidence_target[patch_mask]
    )
    total = (
        state_mse
        + cosine_weight * state_cosine
        + future_kl_weight * future_kl
        + confidence_weight * confidence_bce
    )
    return StatePatchLoss(
        total=total,
        state_mse=state_mse,
        state_cosine=state_cosine,
        future_kl=future_kl,
        confidence_bce=confidence_bce,
    )
