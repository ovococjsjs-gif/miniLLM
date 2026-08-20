"""Cheap local token mixing for edge-oriented hybrid models."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .norm import RMSNorm


class GatedShortConv(nn.Module):
    """Input-dependent, causal, depthwise short convolution.

    The block is deliberately composed of mobile-friendly primitives: two dense
    projections, one depthwise convolution, and a sigmoid gate. It is not claimed
    to reproduce Liquid's proprietary LIV block; it tests the broader finding that
    cheap gated convolutions plus a few global-attention layers are a strong edge
    baseline.
    """

    def __init__(
        self, d_model: int, kernel_size: int = 3, norm_eps: float = 1e-6
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            groups=d_model,
            bias=False,
        )
        self.out_norm = RMSNorm(d_model, norm_eps)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        value = value.transpose(1, 2)
        value = F.pad(value, (self.kernel_size - 1, 0))
        value = self.conv(value).transpose(1, 2)
        mixed = F.silu(value) * torch.sigmoid(gate)
        return self.out_proj(self.out_norm(mixed))
