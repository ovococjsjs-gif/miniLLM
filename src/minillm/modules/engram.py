"""Hashed conditional memory inspired by DeepSeek Engram."""

from __future__ import annotations

from itertools import product

import torch
from torch import nn
from torch.nn import functional as F

from minillm.config import EngramConfig

from .norm import RMSNorm, rms_normalize


class HashedNgramMemory(nn.Module):
    """Retrieve suffix n-gram embeddings in O(1) and context-gate them.

    Important distinction: this is *learned static model memory*, not the user's
    mutable episodic memory. Persistent personalization belongs in a separately
    auditable retrieval store.
    """

    def __init__(
        self, d_model: int, config: EngramConfig, norm_eps: float = 1e-6
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.keys = tuple(product(config.ngram_orders, range(config.num_hash_heads)))
        self.tables = nn.ModuleDict(
            {
                self._name(order, head): nn.Embedding(
                    config.table_size, config.embedding_dim
                )
                for order, head in self.keys
            }
        )
        self.key_proj = nn.Linear(config.retrieved_dim, d_model, bias=False)
        self.value_proj = nn.Linear(config.retrieved_dim, d_model, bias=False)
        self.memory_norm = RMSNorm(d_model, norm_eps)
        self.conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=config.conv_kernel,
            groups=d_model,
            bias=False,
        )
        self.norm_eps = norm_eps

    @staticmethod
    def _name(order: int, head: int) -> str:
        return f"n{order}_h{head}"

    def hash_indices(
        self, token_ids: torch.Tensor, order: int, head: int
    ) -> torch.Tensor:
        """Deterministic suffix hash; invalid prefixes map to zero and are masked later."""

        batch, time = token_ids.shape
        modulus = self.config.table_size
        hashed = torch.full(
            (batch, time),
            17 + 97 * head + order,
            device=token_ids.device,
            dtype=torch.long,
        )
        for offset in range(order):
            shifted = torch.zeros_like(token_ids)
            if offset == 0:
                shifted.copy_(token_ids)
            else:
                shifted[:, offset:] = token_ids[:, :-offset]
            hashed = torch.remainder(
                hashed * 1_000_003 + shifted + 31 * (offset + 1), modulus
            )
        return hashed

    def forward(self, hidden: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.shape != hidden.shape[:2]:
            raise ValueError("token IDs and hidden states must share [batch, time]")
        time = token_ids.shape[1]
        retrieved: list[torch.Tensor] = []
        for order, head in self.keys:
            indices = self.hash_indices(token_ids, order, head)
            value = self.tables[self._name(order, head)](indices)
            valid = (torch.arange(time, device=token_ids.device) >= order - 1)[
                None, :, None
            ]
            retrieved.append(value * valid.to(value.dtype))
        memory = torch.cat(retrieved, dim=-1)
        key = self.key_proj(memory)
        value = self.value_proj(memory)
        similarity = (
            rms_normalize(hidden.float(), self.norm_eps)
            * rms_normalize(key.float(), self.norm_eps)
        ).sum(dim=-1, keepdim=True) / (hidden.shape[-1] ** 0.5)
        gated = torch.sigmoid(similarity).to(value.dtype) * value

        conv_input = self.memory_norm(gated).transpose(1, 2)
        conv_input = F.pad(conv_input, (self.config.conv_kernel - 1, 0))
        refined = F.silu(self.conv(conv_input).transpose(1, 2))
        return gated + refined
