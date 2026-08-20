"""Dense and sparse channel mixers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from minillm.config import MoEConfig


@dataclass
class RouterStats:
    counts: torch.Tensor
    mean_probabilities: torch.Tensor
    z_loss: torch.Tensor
    balance_loss: torch.Tensor


class DenseSwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden, bias=False)
        self.up_proj = nn.Linear(d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        out = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return out, x.new_zeros(()), None


class _ExpertSwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden, bias=False)
        self.up_proj = nn.Linear(d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SparseMoE(nn.Module):
    """Correct, sparse-compute MoE reference with an optional shared expert.

    Only selected token/expert pairs are evaluated. The Python dispatch loop is for
    research correctness, not final mobile speed; deployment needs grouped GEMM or
    a dedicated sparse operator.
    """

    def __init__(self, d_model: int, config: MoEConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.router = nn.Linear(d_model, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            _ExpertSwiGLU(d_model, config.expert_hidden)
            for _ in range(config.num_experts)
        )
        self.shared = (
            _ExpertSwiGLU(d_model, config.shared_expert_hidden)
            if config.shared_expert_hidden
            else None
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, RouterStats]:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        logits = self.router(flat.float())
        probabilities = torch.sigmoid(logits)
        top_values, top_indices = probabilities.topk(self.config.top_k, dim=-1)
        top_weights = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        output = torch.zeros_like(flat)
        counts = torch.zeros(self.config.num_experts, device=x.device, dtype=torch.long)
        for expert_id, expert in enumerate(self.experts):
            token_idx, slot_idx = torch.where(top_indices == expert_id)
            counts[expert_id] = token_idx.numel()
            if token_idx.numel() == 0:
                continue
            expert_out = expert(flat.index_select(0, token_idx))
            weighted = expert_out * top_weights[token_idx, slot_idx, None].to(
                expert_out.dtype
            )
            output.index_add_(0, token_idx, weighted)

        if self.shared is not None:
            output = output + self.shared(flat)

        # A differentiable probability term times a stop-gradient hard load term.
        mean_prob = probabilities.mean(dim=0)
        load = counts.float() / max(1, flat.shape[0] * self.config.top_k)
        balance_loss = self.config.num_experts * torch.sum(mean_prob * load.detach())
        z_loss = torch.mean(torch.logsumexp(logits, dim=-1).square())
        aux = (
            self.config.load_balance_loss * balance_loss
            + self.config.router_z_loss * z_loss
        )
        stats = RouterStats(
            counts, mean_prob.detach(), z_loss.detach(), balance_loss.detach()
        )
        return output.view(shape), aux.to(x.dtype), stats
