"""Exact-shape learned updates for Qwen Gated DeltaNet recurrent states."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


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
