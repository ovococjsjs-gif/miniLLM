"""Grouped-query causal attention reference implementation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .norm import RMSNorm
from .rope import apply_rope


@dataclass(frozen=True)
class AttentionCache:
    """Unexpanded GQA keys and values with shape ``[batch, kv_heads, time, dim]``."""

    key: torch.Tensor
    value: torch.Tensor

    @property
    def length(self) -> int:
        return int(self.key.shape[-2])


class CausalSelfAttention(nn.Module):
    """Bias-free GQA with Q/K normalization and optional local window.

    This implementation prioritizes semantic clarity and correct autograd. Production
    deployment should map the same weights to an optimized runtime (llama.cpp,
    ExecuTorch, MLX, or a fused PyTorch attention kernel).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        *,
        rope_base: float,
        norm_eps: float,
        dropout: float = 0.0,
        sliding_window: int | None = None,
    ) -> None:
        super().__init__()
        if n_heads % n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.rope_base = rope_base
        self.dropout = dropout
        self.sliding_window = sliding_window
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.q_norm = RMSNorm(head_dim, norm_eps)
        self.k_norm = RMSNorm(head_dim, norm_eps)

    def _validate_cache(self, cache: AttentionCache, batch: int) -> None:
        expected_prefix = (batch, self.n_kv_heads)
        if (
            cache.key.shape[:2] != expected_prefix
            or cache.value.shape[:2] != expected_prefix
        ):
            raise ValueError("attention cache has incompatible batch or KV heads")
        if cache.key.shape != cache.value.shape or cache.key.shape[-1] != self.head_dim:
            raise ValueError("attention cache has incompatible key/value shapes")

    def forward_cached(
        self, x: torch.Tensor, cache: AttentionCache | None = None
    ) -> tuple[torch.Tensor, AttentionCache]:
        """Mix a new suffix and return the complete unexpanded KV cache."""

        batch, time, _ = x.shape
        if time < 1:
            raise ValueError("attention input must contain at least one token")
        if cache is not None:
            self._validate_cache(cache, batch)
        offset = cache.length if cache is not None else 0
        q = (
            self.q_proj(x)
            .view(batch, time, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        new_key = (
            self.k_proj(x)
            .view(batch, time, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        new_value = (
            self.v_proj(x)
            .view(batch, time, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        q = self.q_norm(q)
        new_key = self.k_norm(new_key)
        q, new_key = apply_rope(q, new_key, base=self.rope_base, offset=offset)

        if cache is None:
            key, value = new_key, new_value
        else:
            key = torch.cat((cache.key, new_key), dim=-2)
            value = torch.cat((cache.value, new_value), dim=-2)
        new_cache = AttentionCache(key=key, value=value)

        groups = self.n_heads // self.n_kv_heads
        attended_key = key.repeat_interleave(groups, dim=1) if groups > 1 else key
        attended_value = value.repeat_interleave(groups, dim=1) if groups > 1 else value
        dropout_p = self.dropout if self.training else 0.0

        if cache is None and self.sliding_window is None:
            out = F.scaled_dot_product_attention(
                q,
                attended_key,
                attended_value,
                dropout_p=dropout_p,
                is_causal=True,
            )
        elif time == 1 and self.sliding_window is None:
            # Every cached key is in the current token's past, so no mask is required.
            out = F.scaled_dot_product_attention(
                q,
                attended_key,
                attended_value,
                dropout_p=dropout_p,
                is_causal=False,
            )
        else:
            query_positions = torch.arange(offset, offset + time, device=x.device)[
                :, None
            ]
            key_positions = torch.arange(key.shape[-2], device=x.device)[None, :]
            allowed = key_positions <= query_positions
            if self.sliding_window is not None:
                allowed &= key_positions > query_positions - self.sliding_window
            out = F.scaled_dot_product_attention(
                q,
                attended_key,
                attended_value,
                attn_mask=allowed,
                dropout_p=dropout_p,
                is_causal=False,
            )

        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(batch, time, self.n_heads * self.head_dim)
        )
        return self.o_proj(out), new_cache

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.forward_cached(x)
        return output
