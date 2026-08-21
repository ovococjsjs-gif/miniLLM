"""Exact-shape learned updates for Qwen Gated DeltaNet recurrent states."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ConvRowUpdate:
    row: torch.Tensor
    delta: torch.Tensor
    confidence: torch.Tensor


class ConvRowUpdater(nn.Module):
    """Predict the newest 6144-wide Qwen convolution-cache row.

    The two older rows advance by an exact shift. This module learns only the
    unknown newest row as a residual over the previous newest row.
    """

    def __init__(
        self,
        *,
        event_dim: int,
        layers: int,
        row_width: int = 6144,
        hidden_dim: int = 256,
        bottleneck_dim: int = 64,
        identity_dim: int = 16,
    ) -> None:
        super().__init__()
        if (
            min(
                event_dim,
                layers,
                row_width,
                hidden_dim,
                bottleneck_dim,
                identity_dim,
            )
            < 1
        ):
            raise ValueError("convolution updater dimensions must be positive")
        self.event_dim = event_dim
        self.layers = layers
        self.row_width = row_width
        self.layer_embedding = nn.Embedding(layers, identity_dim)
        self.trunk = nn.Sequential(
            nn.LayerNorm(event_dim + identity_dim),
            nn.Linear(event_dim + identity_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.delta_heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_dim, bottleneck_dim),
                nn.SiLU(),
                nn.Linear(bottleneck_dim, row_width),
            )
            for _ in range(layers)
        )
        self.confidence_head = nn.Linear(hidden_dim, 1)
        for head in self.delta_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(
        self,
        previous_newest: torch.Tensor,
        event_features: torch.Tensor,
        layer_ids: torch.Tensor,
    ) -> ConvRowUpdate:
        if previous_newest.ndim != 2 or previous_newest.shape[1] != self.row_width:
            raise ValueError("previous convolution row has the wrong shape")
        batch = previous_newest.shape[0]
        if event_features.shape != (batch, self.event_dim):
            raise ValueError("event features have the wrong shape")
        if layer_ids.shape != (batch,) or layer_ids.dtype != torch.long:
            raise ValueError("layer ids must use shape [batch] and torch.long")
        if torch.any((layer_ids < 0) | (layer_ids >= self.layers)):
            raise ValueError("layer id is outside the configured updater")
        hidden = self.trunk(
            torch.cat((event_features, self.layer_embedding(layer_ids)), dim=-1)
        )
        all_deltas = torch.stack(
            tuple(head(hidden) for head in self.delta_heads), dim=1
        )
        delta = all_deltas[torch.arange(batch, device=event_features.device), layer_ids]
        confidence = torch.sigmoid(self.confidence_head(hidden).squeeze(-1))
        return ConvRowUpdate(previous_newest + delta, delta, confidence)


@dataclass(frozen=True)
class GatedDeltaParameters:
    key: torch.Tensor
    value: torch.Tensor
    gate: torch.Tensor
    beta: torch.Tensor


class GatedDeltaParameterUpdater(nn.Module):
    """Predict Qwen's stable one-token Gated DeltaNet parameters.

    The donor transition is a decay plus one outer product per value head. The
    network predicts key/value/gate/beta from a computed attention-anchor event;
    :meth:`apply` then emits the exact ``[heads, 128, 128]`` state shape accepted
    by Qwen's public recurrent-state serialization.
    """

    def __init__(
        self,
        *,
        event_dim: int,
        layers: int,
        heads: int = 16,
        state_width: int = 128,
        hidden_dim: int = 256,
        identity_dim: int = 16,
    ) -> None:
        super().__init__()
        if min(event_dim, layers, heads, state_width, hidden_dim, identity_dim) < 1:
            raise ValueError("gated-delta updater dimensions must be positive")
        self.event_dim = event_dim
        self.layers = layers
        self.heads = heads
        self.state_width = state_width
        self.layer_embedding = nn.Embedding(layers, identity_dim)
        self.trunk = nn.Sequential(
            nn.LayerNorm(event_dim + identity_dim),
            nn.Linear(event_dim + identity_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        output_dim = 2 * heads * state_width + 2 * heads
        self.parameter_heads = nn.ModuleList(
            nn.Linear(hidden_dim, output_dim) for _ in range(layers)
        )

    def forward(
        self, event_features: torch.Tensor, layer_ids: torch.Tensor
    ) -> GatedDeltaParameters:
        if event_features.ndim != 2 or event_features.shape[1] != self.event_dim:
            raise ValueError("event features have the wrong shape")
        batch = event_features.shape[0]
        if layer_ids.shape != (batch,) or layer_ids.dtype != torch.long:
            raise ValueError("layer ids must use shape [batch] and torch.long")
        if torch.any((layer_ids < 0) | (layer_ids >= self.layers)):
            raise ValueError("layer id is outside the configured updater")
        hidden = self.trunk(
            torch.cat((event_features, self.layer_embedding(layer_ids)), dim=-1)
        )
        all_parameters = torch.stack(
            tuple(head(hidden) for head in self.parameter_heads), dim=1
        )
        selected = all_parameters[
            torch.arange(batch, device=event_features.device), layer_ids
        ]
        vector_values = 2 * self.heads * self.state_width
        vectors = selected[:, :vector_values].view(
            batch, 2, self.heads, self.state_width
        )
        scalars = selected[:, vector_values:].view(batch, 2, self.heads)
        key = F.normalize(vectors[:, 0], dim=-1, eps=1e-6)
        value = vectors[:, 1]
        gate = -F.softplus(scalars[:, 0])
        beta = torch.sigmoid(scalars[:, 1])
        return GatedDeltaParameters(key, value, gate, beta)

    @staticmethod
    def apply(state: torch.Tensor, parameters: GatedDeltaParameters) -> torch.Tensor:
        if state.ndim != 4 or state.shape[1:] != (
            parameters.key.shape[1],
            parameters.key.shape[2],
            parameters.key.shape[2],
        ):
            raise ValueError("state and gated-delta parameter shapes disagree")
        decayed = state * parameters.gate.exp().unsqueeze(-1).unsqueeze(-1)
        state_key = torch.einsum("bhij,bhj->bhi", decayed, parameters.key)
        innovation = (parameters.value - state_key) * parameters.beta.unsqueeze(-1)
        return decayed + torch.einsum("bhi,bhj->bhij", innovation, parameters.key)


@dataclass(frozen=True)
class CacheUpdate:
    recurrent_state: torch.Tensor
    convolution_state: torch.Tensor
    recurrent_delta: torch.Tensor
    convolution_delta: torch.Tensor
    convolution_confidence: torch.Tensor


class AIraQwenCacheUpdater(nn.Module):
    """Normalized joint interface for all learned Qwen cache updates."""

    def __init__(
        self,
        *,
        event_mean: torch.Tensor,
        event_scale: torch.Tensor,
        layers: int,
        heads: int = 16,
        state_width: int = 128,
        conv_width: int = 6144,
        state_hidden_dim: int = 256,
        conv_hidden_dim: int = 256,
        conv_bottleneck_dim: int = 64,
        identity_dim: int = 16,
        state_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        if event_mean.ndim != 1 or event_scale.shape != event_mean.shape:
            raise ValueError("event normalization tensors must be equal-width vectors")
        if not 0 < state_alpha <= 1:
            raise ValueError("state alpha must lie in (0, 1]")
        self.register_buffer("event_mean", event_mean.detach().float().clone())
        self.register_buffer("event_scale", event_scale.detach().float().clone())
        self.register_buffer("state_alpha", torch.tensor(float(state_alpha)))
        event_dim = event_mean.shape[0]
        self.state_updater = GatedDeltaParameterUpdater(
            event_dim=event_dim,
            layers=layers,
            heads=heads,
            state_width=state_width,
            hidden_dim=state_hidden_dim,
            identity_dim=identity_dim,
        )
        self.conv_updater = ConvRowUpdater(
            event_dim=event_dim,
            layers=layers,
            row_width=conv_width,
            hidden_dim=conv_hidden_dim,
            bottleneck_dim=conv_bottleneck_dim,
            identity_dim=identity_dim,
        )

    def normalize_event(self, event_features: torch.Tensor) -> torch.Tensor:
        if event_features.shape[-1] != self.event_mean.shape[0]:
            raise ValueError("event feature width differs from updater normalization")
        return (event_features - self.event_mean) * self.event_scale

    def forward(
        self,
        recurrent_state: torch.Tensor,
        convolution_state: torch.Tensor,
        event_features: torch.Tensor,
        layer_ids: torch.Tensor,
    ) -> CacheUpdate:
        if convolution_state.ndim != 3 or convolution_state.shape[1] != 3:
            raise ValueError("convolution state must have shape [batch, 3, width]")
        normalized = self.normalize_event(event_features)
        parameters = self.state_updater(normalized, layer_ids)
        proposed_state = self.state_updater.apply(recurrent_state, parameters)
        recurrent_delta = self.state_alpha * (proposed_state - recurrent_state)
        updated_state = recurrent_state + recurrent_delta
        conv = self.conv_updater(convolution_state[:, 2], normalized, layer_ids)
        updated_conv = torch.stack(
            (convolution_state[:, 1], convolution_state[:, 2], conv.row), dim=1
        )
        return CacheUpdate(
            updated_state,
            updated_conv,
            recurrent_delta,
            conv.delta,
            conv.confidence,
        )
