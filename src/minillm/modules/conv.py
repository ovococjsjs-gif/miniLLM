"""Cheap local token mixing for edge-oriented hybrid models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .norm import RMSNorm


@dataclass(frozen=True)
class ConvCache:
    """Previous projected values, limited to ``kernel_size - 1`` tokens."""

    values: torch.Tensor

    @property
    def length(self) -> int:
        return int(self.values.shape[1])


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
        self.d_model = d_model
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

    def forward_cached(
        self, x: torch.Tensor, cache: ConvCache | None = None
    ) -> tuple[torch.Tensor, ConvCache]:
        """Mix a new suffix while retaining only the required left context."""

        if x.shape[1] < 1:
            raise ValueError("convolution input must contain at least one token")
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        if cache is None:
            history = value[:, :0]
        else:
            if (
                cache.values.shape[0] != x.shape[0]
                or cache.values.shape[2] != self.d_model
                or cache.length > self.kernel_size - 1
            ):
                raise ValueError("convolution cache has an incompatible shape")
            history = cache.values
        combined = torch.cat((history, value), dim=1)
        missing_left = max(0, self.kernel_size - 1 - history.shape[1])
        conv_input = combined.transpose(1, 2)
        if missing_left:
            conv_input = F.pad(conv_input, (missing_left, 0))
        convolved = self.conv(conv_input).transpose(1, 2)
        if convolved.shape[1] != x.shape[1]:
            raise RuntimeError("cached convolution produced an invalid suffix length")
        mixed = F.silu(convolved) * torch.sigmoid(gate)
        output = self.out_proj(self.out_norm(mixed))
        retained = self.kernel_size - 1
        next_values = combined[:, -retained:] if retained else combined[:, :0]
        return output, ConvCache(values=next_values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.forward_cached(x)
        return output
