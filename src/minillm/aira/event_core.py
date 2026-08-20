"""Bounded-context neural fallbacks that run only on routed events."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class _BoundedContextEncoder(nn.Module):
    def __init__(self, vocab_size: int, context_size: int, d_model: int) -> None:
        super().__init__()
        if vocab_size < 2 or context_size < 1 or d_model < 4:
            raise ValueError("invalid event-context model dimensions")
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Parameter(torch.empty(context_size, d_model))
        self.context_projection = nn.Linear(context_size * d_model, d_model)
        self.gate_projection = nn.Linear(context_size * d_model, d_model)
        self.normalization = nn.LayerNorm(d_model)

    def reset_encoder_parameters(self) -> None:
        standard_deviation = self.d_model**-0.5
        nn.init.normal_(self.embedding.weight, std=standard_deviation)
        nn.init.normal_(self.position_embedding, std=standard_deviation)
        nn.init.xavier_uniform_(self.context_projection.weight)
        nn.init.zeros_(self.context_projection.bias)
        nn.init.xavier_uniform_(self.gate_projection.weight)
        nn.init.zeros_(self.gate_projection.bias)
        nn.init.ones_(self.normalization.weight)
        nn.init.zeros_(self.normalization.bias)

    def encode_context(self, context_ids: torch.Tensor) -> torch.Tensor:
        if context_ids.ndim != 2 or context_ids.shape[1] != self.context_size:
            raise ValueError(
                f"context_ids must have shape [batch, {self.context_size}]"
            )
        if context_ids.dtype != torch.long:
            raise TypeError("context_ids must use torch.long token IDs")
        embedded = self.embedding(context_ids) + self.position_embedding
        flattened = embedded.flatten(1)
        candidate = torch.tanh(self.context_projection(flattened))
        gate = torch.sigmoid(self.gate_projection(flattened))
        return self.normalization(candidate * gate)

    @property
    def parameter_bytes(self) -> int:
        return sum(
            parameter.numel() * parameter.element_size()
            for parameter in self.parameters()
        )


class EventContextLM(_BoundedContextEncoder):
    """Small tied-embedding BPE LM evaluated independently at each novelty event.

    Unlike a cached autoregressive Transformer, this model has no hidden state that must
    be advanced through shelf tokens. A fallback reads the last ``context_size`` token IDs
    directly, so skipped positions remain genuinely free of neural layer execution. The
    bounded context is an intentional AIra substrate primitive, not a quality claim.
    """

    def __init__(
        self, vocab_size: int, context_size: int = 16, d_model: int = 48
    ) -> None:
        super().__init__(vocab_size, context_size, d_model)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.reset_encoder_parameters()
        nn.init.zeros_(self.output_bias)

    def forward(self, context_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.encode_context(context_ids)
        return F.linear(hidden, self.embedding.weight, self.output_bias)


class ByteEventLM(_BoundedContextEncoder):
    """Gated flattened BPE-patch baseline with a next-raw-byte output."""

    def __init__(
        self, vocab_size: int, context_size: int = 16, d_model: int = 48
    ) -> None:
        super().__init__(vocab_size, context_size, d_model)
        self.byte_head = nn.Linear(d_model, 256)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.reset_encoder_parameters()
        nn.init.xavier_uniform_(self.byte_head.weight)
        nn.init.zeros_(self.byte_head.bias)

    def forward(self, context_ids: torch.Tensor) -> torch.Tensor:
        return self.byte_head(self.encode_context(context_ids))


class _GatedConvBlock(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(d_model)
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size=3, padding=1, groups=d_model
        )
        self.expand = nn.Linear(d_model, 2 * d_model)
        self.project = nn.Linear(d_model, d_model)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.normalization(values)
        hidden = self.depthwise(hidden.transpose(1, 2)).transpose(1, 2)
        content, gate = self.expand(hidden).chunk(2, dim=-1)
        return values + self.project(torch.tanh(content) * torch.sigmoid(gate))


class ConvByteEventLM(nn.Module):
    """Bounded local-convolution patch core with learned global pooling."""

    def __init__(
        self,
        vocab_size: int,
        context_size: int = 16,
        d_model: int = 48,
        blocks: int = 4,
    ) -> None:
        super().__init__()
        if vocab_size < 2 or context_size < 1 or d_model < 4 or blocks < 1:
            raise ValueError("invalid convolutional event-core dimensions")
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Parameter(torch.empty(context_size, d_model))
        self.blocks = nn.ModuleList(_GatedConvBlock(d_model) for _ in range(blocks))
        self.pool_score = nn.Linear(d_model, 1)
        self.final_normalization = nn.LayerNorm(d_model)
        self.byte_head = nn.Linear(d_model, 256)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        standard_deviation = self.d_model**-0.5
        nn.init.normal_(self.embedding.weight, std=standard_deviation)
        nn.init.normal_(self.position_embedding, std=standard_deviation)
        nn.init.xavier_uniform_(self.pool_score.weight)
        nn.init.zeros_(self.pool_score.bias)
        nn.init.xavier_uniform_(self.byte_head.weight)
        nn.init.zeros_(self.byte_head.bias)

    def forward(self, context_ids: torch.Tensor) -> torch.Tensor:
        if context_ids.ndim != 2 or context_ids.shape[1] != self.context_size:
            raise ValueError(
                f"context_ids must have shape [batch, {self.context_size}]"
            )
        hidden = self.embedding(context_ids) + self.position_embedding
        for block in self.blocks:
            hidden = block(hidden)
        scores = torch.softmax(self.pool_score(hidden), dim=1)
        pooled = torch.sum(scores * hidden, dim=1)
        return self.byte_head(self.final_normalization(pooled))

    @property
    def parameter_bytes(self) -> int:
        return sum(
            parameter.numel() * parameter.element_size()
            for parameter in self.parameters()
        )


class AttentionByteEventLM(nn.Module):
    """Two-layer bounded self-attention patch core with next-byte output."""

    def __init__(
        self,
        vocab_size: int,
        context_size: int = 16,
        d_model: int = 48,
        heads: int = 4,
        layers: int = 2,
    ) -> None:
        super().__init__()
        if (
            vocab_size < 2
            or context_size < 1
            or d_model < 4
            or heads < 1
            or d_model % heads
            or layers < 1
        ):
            raise ValueError("invalid attention event-core dimensions")
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Parameter(torch.empty(context_size, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model,
            heads,
            dim_feedforward=2 * d_model,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.final_normalization = nn.LayerNorm(d_model)
        self.byte_head = nn.Linear(d_model, 256)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        standard_deviation = self.d_model**-0.5
        nn.init.normal_(self.embedding.weight, std=standard_deviation)
        nn.init.normal_(self.position_embedding, std=standard_deviation)
        nn.init.xavier_uniform_(self.byte_head.weight)
        nn.init.zeros_(self.byte_head.bias)

    def forward(self, context_ids: torch.Tensor) -> torch.Tensor:
        if context_ids.ndim != 2 or context_ids.shape[1] != self.context_size:
            raise ValueError(
                f"context_ids must have shape [batch, {self.context_size}]"
            )
        hidden = self.embedding(context_ids) + self.position_embedding
        hidden = self.encoder(hidden)
        return self.byte_head(self.final_normalization(hidden[:, -1]))

    @property
    def parameter_bytes(self) -> int:
        return sum(
            parameter.numel() * parameter.element_size()
            for parameter in self.parameters()
        )
