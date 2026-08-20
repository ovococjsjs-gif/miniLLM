"""Readable recurrent Gated DeltaNet-2 implementation.

The recurrent equation follows Hatamizadeh, Choi & Kautz (2026). This module is
intentionally sequential and serves as a correctness/ablation reference. A useful
production model requires the paper's chunkwise WY algorithm and fused device kernels.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .norm import RMSNorm


class ReferenceGatedDeltaNet2(nn.Module):
    """Multi-head Gated Delta Rule-2 with fixed-size recurrent memory."""

    def __init__(
        self, d_model: int, n_heads: int, head_dim: int, norm_eps: float = 1e-6
    ) -> None:
        super().__init__()
        if d_model != n_heads * head_dim:
            raise ValueError(
                "d_model must equal n_heads * head_dim for the reference GDN2"
            )
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.decay_proj = nn.Linear(d_model, d_model, bias=False)
        self.erase_proj = nn.Linear(d_model, d_model, bias=False)
        self.write_proj = nn.Linear(d_model, d_model, bias=False)
        self.output_gate = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.head_norm = RMSNorm(head_dim, norm_eps)
        self.log_decay_rate = nn.Parameter(torch.zeros(n_heads, head_dim))
        self.decay_bias = nn.Parameter(torch.zeros(n_heads, head_dim))

    def _heads(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(*tensor.shape[:2], self.n_heads, self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        *,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch, time, _ = x.shape
        q = F.normalize(F.silu(self._heads(self.q_proj(x))), dim=-1)
        k = F.normalize(F.silu(self._heads(self.k_proj(x))), dim=-1)
        v = F.silu(self._heads(self.v_proj(x)))
        erase = torch.sigmoid(self._heads(self.erase_proj(x)))
        write = torch.sigmoid(self._heads(self.write_proj(x)))
        decay_logits = self._heads(self.decay_proj(x)).float()
        log_alpha = -torch.exp(self.log_decay_rate.float())[None, None] * F.softplus(
            decay_logits + self.decay_bias.float()[None, None]
        )
        alpha = torch.exp(log_alpha).to(x.dtype)

        if initial_state is None:
            state = x.new_zeros(batch, self.n_heads, self.head_dim, self.head_dim)
        else:
            expected = (batch, self.n_heads, self.head_dim, self.head_dim)
            if initial_state.shape != expected:
                raise ValueError(f"initial state must have shape {expected}")
            state = initial_state

        outputs: list[torch.Tensor] = []
        for index in range(time):
            # D_t S_{t-1}: channel-wise decay acts on the key axis.
            decayed = alpha[:, index, :, :, None] * state
            erase_direction = erase[:, index] * k[:, index]
            old_read = torch.einsum("bhkv,bhk->bhv", decayed, erase_direction)
            target = write[:, index] * v[:, index]
            correction = target - old_read
            state = decayed + k[:, index, :, :, None] * correction[:, :, None, :]
            output = torch.einsum("bhkv,bhk->bhv", state, q[:, index])
            outputs.append(output)

        mixed = torch.stack(outputs, dim=1)
        mixed = self.head_norm(mixed).reshape(batch, time, self.d_model)
        mixed = mixed * torch.sigmoid(self.output_gate(x))
        result = self.out_proj(mixed)
        return (result, state) if return_state else result
