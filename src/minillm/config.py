"""Typed, validated model configuration.

The configuration intentionally describes *unique* and *effective* depth separately.
A recurrent core is stored once and can be executed multiple times. This keeps model
memory independent from the test-time latent-compute budget.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_TOKEN_MIXERS = frozenset({"attention", "conv", "gdn2"})


@dataclass(frozen=True)
class MoEConfig:
    """Sparse feed-forward configuration.

    ``num_experts`` counts routed experts. ``top_k`` of them run for each token.
    An optional shared expert is always active and acts as a generalist path.
    """

    enabled: bool = False
    num_experts: int = 8
    top_k: int = 2
    expert_hidden: int = 256
    shared_expert_hidden: int = 0
    router_z_loss: float = 1e-4
    load_balance_loss: float = 1e-2

    def validate(self) -> None:
        if not self.enabled:
            return
        if self.num_experts < 1:
            raise ValueError("moe.num_experts must be positive")
        if not 1 <= self.top_k <= self.num_experts:
            raise ValueError("moe.top_k must be in [1, num_experts]")
        if self.expert_hidden < 1:
            raise ValueError("moe.expert_hidden must be positive")
        if self.shared_expert_hidden < 0:
            raise ValueError("moe.shared_expert_hidden cannot be negative")
        if self.router_z_loss < 0 or self.load_balance_loss < 0:
            raise ValueError("router loss weights cannot be negative")


@dataclass(frozen=True)
class EngramConfig:
    """Hashed suffix n-gram conditional memory.

    This is a compact, single-stream adaptation of DeepSeek Engram. One embedding
    table is allocated per (n-gram order, hash head). Retrieved rows are concatenated,
    context-gated, and refined with a causal depthwise convolution.
    """

    enabled: bool = False
    ngram_orders: tuple[int, ...] = (2, 3)
    num_hash_heads: int = 2
    table_size: int = 4093
    embedding_dim: int = 16
    conv_kernel: int = 4

    def validate(self) -> None:
        if not self.enabled:
            return
        if not self.ngram_orders or any(order < 2 for order in self.ngram_orders):
            raise ValueError("engram.ngram_orders must contain values >= 2")
        if tuple(sorted(set(self.ngram_orders))) != self.ngram_orders:
            raise ValueError("engram.ngram_orders must be sorted and unique")
        if self.num_hash_heads < 1 or self.table_size < 2 or self.embedding_dim < 1:
            raise ValueError("invalid Engram table shape")
        if self.conv_kernel < 1:
            raise ValueError("engram.conv_kernel must be positive")

    @property
    def retrieved_dim(self) -> int:
        return len(self.ngram_orders) * self.num_hash_heads * self.embedding_dim


@dataclass(frozen=True)
class MiniLLMConfig:
    """Architecture definition for the experimental decoder-only language model."""

    vocab_size: int = 4096
    d_model: int = 256
    n_heads: int = 4
    n_kv_heads: int = 2
    head_dim: int = 64
    ffn_hidden: int = 768
    max_seq_len: int = 4096
    rope_base: float = 50_000.0
    conv_kernel: int = 3
    norm_eps: float = 1e-6
    dropout: float = 0.0
    tie_embeddings: bool = True
    sandwich_norm: bool = False

    # The core's parameters are stored once and reused ``core_repetitions`` times.
    prelude_layers: tuple[str, ...] = ("attention",)
    core_layers: tuple[str, ...] = ("conv", "attention")
    coda_layers: tuple[str, ...] = ("attention",)
    core_repetitions: int = 2
    max_core_repetitions: int = 4
    recurrent_input_injection: bool = True

    # Auxiliary next-next-token modules used only in training or as draft heads.
    mtp_depth: int = 1
    mtp_loss_weight: float = 0.3

    moe: MoEConfig = field(default_factory=MoEConfig)
    engram: EngramConfig = field(default_factory=EngramConfig)

    def validate(self) -> MiniLLMConfig:
        positive = {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "ffn_hidden": self.ffn_hidden,
            "max_seq_len": self.max_seq_len,
            "conv_kernel": self.conv_kernel,
            "core_repetitions": self.core_repetitions,
            "max_core_repetitions": self.max_core_repetitions,
        }
        for name, value in positive.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.d_model != self.n_heads * self.head_dim:
            raise ValueError("d_model must equal n_heads * head_dim")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.core_repetitions > self.max_core_repetitions:
            raise ValueError("core_repetitions exceeds max_core_repetitions")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.mtp_depth < 0 or self.mtp_loss_weight < 0:
            raise ValueError("invalid MTP settings")
        layers = self.prelude_layers + self.core_layers + self.coda_layers
        if not layers:
            raise ValueError("at least one layer is required")
        unknown = set(layers) - _TOKEN_MIXERS
        if unknown:
            raise ValueError(f"unknown token mixer(s): {sorted(unknown)}")
        self.moe.validate()
        self.engram.validate()
        return self

    @property
    def unique_depth(self) -> int:
        return len(self.prelude_layers) + len(self.core_layers) + len(self.coda_layers)

    @property
    def effective_depth(self) -> int:
        return (
            len(self.prelude_layers)
            + self.core_repetitions * len(self.core_layers)
            + len(self.coda_layers)
        )

    def effective_layer_types(self, repetitions: int | None = None) -> tuple[str, ...]:
        repeats = self.core_repetitions if repetitions is None else repetitions
        if not 1 <= repeats <= self.max_core_repetitions:
            raise ValueError("requested recurrence count is outside configured bounds")
        return self.prelude_layers + self.core_layers * repeats + self.coda_layers

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> MiniLLMConfig:
        data = dict(raw)
        data["moe"] = MoEConfig(**data.get("moe", {}))
        engram = dict(data.get("engram", {}))
        if "ngram_orders" in engram:
            engram["ngram_orders"] = tuple(engram["ngram_orders"])
        data["engram"] = EngramConfig(**engram)
        for key in ("prelude_layers", "core_layers", "coda_layers"):
            if key in data:
                data[key] = tuple(data[key])
        return cls(**data).validate()

    @classmethod
    def load(cls, path: str | Path) -> MiniLLMConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def layer_type_counts(layer_types: Sequence[str]) -> dict[str, int]:
    return {kind: layer_types.count(kind) for kind in sorted(_TOKEN_MIXERS)}
