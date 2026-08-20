"""A modular decoder-only model for controlled small-model experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .config import MiniLLMConfig
from .modules import (
    AttentionCache,
    CausalSelfAttention,
    ConvCache,
    DenseSwiGLU,
    GatedShortConv,
    HashedNgramMemory,
    ReferenceGatedDeltaNet2,
    RMSNorm,
    SparseMoE,
)
from .modules.ffn import RouterStats


@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    main_loss: torch.Tensor | None = None
    mtp_loss: torch.Tensor | None = None
    router_loss: torch.Tensor | None = None
    router_stats: tuple[RouterStats, ...] = ()


MixerCache = AttentionCache | ConvCache | torch.Tensor | None


@dataclass(frozen=True)
class ModelCache:
    """Per-effective-layer decode state plus the number of consumed tokens."""

    block_caches: tuple[MixerCache, ...]
    token_count: int
    core_repetitions: int


class MiniBlock(nn.Module):
    def __init__(self, config: MiniLLMConfig, mixer_type: str) -> None:
        super().__init__()
        self.mixer_type = mixer_type
        if mixer_type == "attention":
            self.mixer = CausalSelfAttention(
                config.d_model,
                config.n_heads,
                config.n_kv_heads,
                config.head_dim,
                rope_base=config.rope_base,
                norm_eps=config.norm_eps,
                dropout=config.dropout,
            )
        elif mixer_type == "conv":
            self.mixer = GatedShortConv(
                config.d_model, config.conv_kernel, config.norm_eps
            )
        elif mixer_type == "gdn2":
            self.mixer = ReferenceGatedDeltaNet2(
                config.d_model, config.n_heads, config.head_dim, config.norm_eps
            )
        else:  # Protected by config validation, kept defensive for direct construction.
            raise ValueError(f"unknown mixer: {mixer_type}")

        self.pre_mixer_norm = RMSNorm(config.d_model, config.norm_eps)
        self.post_mixer_norm = (
            RMSNorm(config.d_model, config.norm_eps)
            if config.sandwich_norm
            else nn.Identity()
        )
        self.pre_ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.post_ffn_norm = (
            RMSNorm(config.d_model, config.norm_eps)
            if config.sandwich_norm
            else nn.Identity()
        )
        self.ffn = (
            SparseMoE(config.d_model, config.moe)
            if config.moe.enabled
            else DenseSwiGLU(config.d_model, config.ffn_hidden)
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, RouterStats | None]:
        x = self.post_mixer_norm(x + self.dropout(self.mixer(self.pre_mixer_norm(x))))
        ffn_output, router_loss, stats = self.ffn(self.pre_ffn_norm(x))
        x = self.post_ffn_norm(x + self.dropout(ffn_output))
        return x, router_loss, stats

    def forward_cached(
        self, x: torch.Tensor, cache: MixerCache
    ) -> tuple[torch.Tensor, torch.Tensor, RouterStats | None, MixerCache]:
        normalized = self.pre_mixer_norm(x)
        if self.mixer_type == "attention":
            if cache is not None and not isinstance(cache, AttentionCache):
                raise TypeError("attention block received the wrong cache type")
            mixed, next_cache = self.mixer.forward_cached(normalized, cache)
        elif self.mixer_type == "conv":
            if cache is not None and not isinstance(cache, ConvCache):
                raise TypeError("convolution block received the wrong cache type")
            mixed, next_cache = self.mixer.forward_cached(normalized, cache)
        else:
            if cache is not None and not isinstance(cache, torch.Tensor):
                raise TypeError("GDN2 block received the wrong cache type")
            mixed, next_cache = self.mixer(
                normalized, initial_state=cache, return_state=True
            )
        x = self.post_mixer_norm(x + self.dropout(mixed))
        ffn_output, router_loss, stats = self.ffn(self.pre_ffn_norm(x))
        x = self.post_ffn_norm(x + self.dropout(ffn_output))
        return x, router_loss, stats, next_cache


class MTPModule(nn.Module):
    """One sequential future-token representation module with a shared LM head."""

    def __init__(self, d_model: int, norm_eps: float) -> None:
        super().__init__()
        self.proj = nn.Linear(2 * d_model, d_model, bias=False)
        self.norm = RMSNorm(d_model, norm_eps)

    def forward(
        self, hidden: torch.Tensor, future_embedding: torch.Tensor
    ) -> torch.Tensor:
        return self.norm(self.proj(torch.cat((hidden, future_embedding), dim=-1)))


class MiniLLM(nn.Module):
    """Hybrid model with optional conditional memory, MoE, and recurrent depth.

    The prelude and coda have unique weights. Core blocks are registered once and
    repeatedly applied. Passing a larger ``core_repetitions`` at inference spends
    more latent compute without loading more parameters.
    """

    def __init__(self, config: MiniLLMConfig) -> None:
        super().__init__()
        self.config = config.validate()
        self.gradient_checkpointing = False
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.prelude = nn.ModuleList(
            MiniBlock(config, kind) for kind in config.prelude_layers
        )
        self.core = nn.ModuleList(
            MiniBlock(config, kind) for kind in config.core_layers
        )
        self.coda = nn.ModuleList(
            MiniBlock(config, kind) for kind in config.coda_layers
        )
        self.engram = (
            HashedNgramMemory(config.d_model, config.engram, config.norm_eps)
            if config.engram.enabled
            else None
        )
        if config.recurrent_input_injection:
            self.initial_state = nn.Parameter(torch.zeros(config.d_model))
            self.recurrent_adapter = nn.Linear(2 * config.d_model, config.d_model)
        else:
            self.register_parameter("initial_state", None)
            self.recurrent_adapter = None
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight
        self.mtp_modules = nn.ModuleList(
            MTPModule(config.d_model, config.norm_eps) for _ in range(config.mtp_depth)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = self.config.d_model**-0.5
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=std, a=-3 * std, b=3 * std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=std, a=-3 * std, b=3 * std)
        if self.initial_state is not None:
            nn.init.normal_(self.initial_state, std=std)

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        """Trade extra forward compute for lower activation memory during training."""

        self.gradient_checkpointing = enabled

    def _run_blocks(
        self,
        blocks: nn.ModuleList,
        hidden: torch.Tensor,
        router_losses: list[torch.Tensor],
        router_stats: list[RouterStats],
    ) -> torch.Tensor:
        for block in blocks:
            if self.training and self.gradient_checkpointing and hidden.requires_grad:

                def checkpointed(
                    value: torch.Tensor, current_block: MiniBlock = block
                ) -> tuple[torch.Tensor, torch.Tensor]:
                    output, auxiliary, _ = current_block(value)
                    return output, auxiliary

                hidden, aux = checkpoint(checkpointed, hidden, use_reentrant=False)
                stats = None
            else:
                hidden, aux, stats = block(hidden)
            router_losses.append(aux)
            if stats is not None:
                router_stats.append(stats)
        return hidden

    @property
    def supports_cached_decode(self) -> bool:
        """Whether all enabled components have an exact reference cache path."""

        return self.engram is None

    def forward_cached(
        self,
        input_ids: torch.Tensor,
        cache: ModelCache | None = None,
        *,
        core_repetitions: int | None = None,
    ) -> tuple[ModelOutput, ModelCache]:
        """Evaluate a prompt or suffix and return state for exact incremental decode.

        The model must be in evaluation mode. Engram currently falls back to full-prefix
        generation because its suffix hash and refinement convolution need a dedicated
        cache implementation.
        """

        if self.training:
            raise RuntimeError("cached decode requires model.eval()")
        if not self.supports_cached_decode:
            raise NotImplementedError("cached decode is not implemented for Engram")
        if input_ids.ndim != 2 or input_ids.shape[1] < 1:
            raise ValueError("input_ids must have shape [batch, positive time]")
        repetitions = (
            self.config.core_repetitions
            if core_repetitions is None
            else core_repetitions
        )
        if not 1 <= repetitions <= self.config.max_core_repetitions:
            raise ValueError("core_repetitions outside configured bounds")
        effective_blocks = (
            len(self.prelude) + repetitions * len(self.core) + len(self.coda)
        )
        if cache is None:
            previous: tuple[MixerCache, ...] = (None,) * effective_blocks
            token_count = 0
        else:
            if cache.core_repetitions != repetitions:
                raise ValueError("cache was created with a different recurrence count")
            if len(cache.block_caches) != effective_blocks:
                raise ValueError("cache has an incompatible effective depth")
            previous = cache.block_caches
            token_count = cache.token_count
        if token_count + input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("cached sequence exceeds configured maximum")

        router_losses: list[torch.Tensor] = []
        router_stats: list[RouterStats] = []
        next_caches: list[MixerCache] = []

        def run(blocks: nn.ModuleList, hidden: torch.Tensor) -> torch.Tensor:
            for block in blocks:
                block_cache = previous[len(next_caches)]
                hidden, aux, stats, next_cache = block.forward_cached(
                    hidden, block_cache
                )
                router_losses.append(aux)
                if stats is not None:
                    router_stats.append(stats)
                next_caches.append(next_cache)
            return hidden

        hidden = self.embedding(input_ids) * (self.config.d_model**0.5)
        hidden = run(self.prelude, hidden)
        injected = hidden
        if self.recurrent_adapter is not None:
            state = self.initial_state[None, None, :].expand_as(hidden)
        else:
            state = hidden
        for _ in range(repetitions):
            if self.recurrent_adapter is not None:
                state = self.recurrent_adapter(torch.cat((state, injected), dim=-1))
            state = run(self.core, state)
        hidden = run(self.coda, state)
        logits = self.lm_head(self.final_norm(hidden))
        output = ModelOutput(logits=logits, router_stats=tuple(router_stats))
        next_model_cache = ModelCache(
            block_caches=tuple(next_caches),
            token_count=token_count + input_ids.shape[1],
            core_repetitions=repetitions,
        )
        return output, next_model_cache

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        *,
        core_repetitions: int | None = None,
    ) -> ModelOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, time]")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("sequence exceeds configured maximum")
        repetitions = (
            self.config.core_repetitions
            if core_repetitions is None
            else core_repetitions
        )
        if not 1 <= repetitions <= self.config.max_core_repetitions:
            raise ValueError("core_repetitions outside configured bounds")

        hidden = self.embedding(input_ids) * (self.config.d_model**0.5)
        router_losses: list[torch.Tensor] = []
        router_stats: list[RouterStats] = []
        hidden = self._run_blocks(self.prelude, hidden, router_losses, router_stats)
        if self.engram is not None:
            hidden = hidden + self.engram(hidden, input_ids)

        injected = hidden
        if self.recurrent_adapter is not None:
            state = self.initial_state[None, None, :].expand_as(hidden)
        else:
            state = hidden
        for _ in range(repetitions):
            if self.recurrent_adapter is not None:
                state = self.recurrent_adapter(torch.cat((state, injected), dim=-1))
            state = self._run_blocks(self.core, state, router_losses, router_stats)
        hidden = self._run_blocks(self.coda, state, router_losses, router_stats)
        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden)

        if labels is None:
            return ModelOutput(logits=logits, router_stats=tuple(router_stats))
        if labels.shape != input_ids.shape:
            raise ValueError("labels must have the same shape as input_ids")

        main_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, self.config.vocab_size),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        mtp_terms: list[torch.Tensor] = []
        current = hidden
        for depth, module in enumerate(self.mtp_modules, start=1):
            if input_ids.shape[1] <= depth + 1:
                break
            future_embedding = self.embedding(input_ids[:, depth:])
            current = module(current[:, :-1], future_embedding)
            future_logits = F.linear(current[:, :-1], self.lm_head.weight)
            targets = labels[:, depth + 1 :]
            mtp_terms.append(
                F.cross_entropy(
                    future_logits.reshape(-1, self.config.vocab_size),
                    targets.reshape(-1),
                    ignore_index=-100,
                )
            )
        mtp_loss = (
            torch.stack(mtp_terms).mean() if mtp_terms else main_loss.new_zeros(())
        )
        router_loss = (
            torch.stack(router_losses).sum()
            if router_losses
            else main_loss.new_zeros(())
        )
        loss = main_loss + self.config.mtp_loss_weight * mtp_loss + router_loss
        return ModelOutput(
            logits=logits,
            loss=loss,
            main_loss=main_loss,
            mtp_loss=mtp_loss,
            router_loss=router_loss,
            router_stats=tuple(router_stats),
        )
