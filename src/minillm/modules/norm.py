"""Normalization primitives."""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """RMS normalization with a learned per-channel scale."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        normalized = x.float() * torch.rsqrt(
            x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return (normalized * self.weight.float()).to(dtype)


def rms_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Parameter-free RMS normalization used in scalar similarity gates."""

    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
