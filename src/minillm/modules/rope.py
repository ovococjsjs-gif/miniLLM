"""Rotary position embeddings."""

from __future__ import annotations

import torch


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    base: float,
    offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to ``[batch, heads, time, head_dim]`` queries and keys."""

    dim = q.shape[-1]
    if dim % 2:
        raise ValueError("RoPE head dimension must be even")
    positions = torch.arange(
        offset, offset + q.shape[-2], device=q.device, dtype=torch.float32
    )
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=q.device).float() / dim))
    angles = torch.outer(positions, inv_freq)
    cos = torch.repeat_interleave(angles.cos(), 2, dim=-1)[None, None, :, :]
    sin = torch.repeat_interleave(angles.sin(), 2, dim=-1)[None, None, :, :]
    return (
        q * cos.to(q.dtype) + _rotate_half(q) * sin.to(q.dtype),
        k * cos.to(k.dtype) + _rotate_half(k) * sin.to(k.dtype),
    )
