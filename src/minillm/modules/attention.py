"""Grouped-query causal attention reference implementation."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .norm import RMSNorm
from .rope import apply_rope


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, _ = x.shape
        q = (
            self.q_proj(x)
            .view(batch, time, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch, time, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch, time, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rope(q, k, base=self.rope_base)

        groups = self.n_heads // self.n_kv_heads
        if groups > 1:
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)

        dropout_p = self.dropout if self.training else 0.0
        if self.sliding_window is None:
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout_p, is_causal=True
            )
        else:
            row = torch.arange(time, device=x.device)[:, None]
            col = torch.arange(time, device=x.device)[None, :]
            allowed = (col <= row) & (col > row - self.sliding_window)
            mask = torch.zeros((time, time), device=x.device, dtype=q.dtype)
            mask.masked_fill_(~allowed, float("-inf"))
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=dropout_p
            )

        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(batch, time, self.n_heads * self.head_dim)
        )
        return self.o_proj(out)
