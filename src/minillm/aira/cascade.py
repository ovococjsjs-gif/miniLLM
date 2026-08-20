"""Executable byte-trigger -> dynamic-BPE neural event cascade."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch

from .bridge import ByteBPEBridge
from .calibration import ReliabilityThreshold
from .event_core import ByteEventLM
from .memory import EpisodicFactStore, MemoryHit
from .trigger import CompactShelfLevel, predict_shelf_next


@dataclass(frozen=True)
class ByteEventConfig:
    minimum_support: int = 5
    confidence_threshold: float = 0.95
    confidence_z: float = 1.96
    maximum_shelf_burst: int = 4
    cumulative_risk_budget: float = 0.10
    neural_anchor_interval: int = 8
    raw_context_bytes: int = 64
    cycle_repetitions: int = 3
    maximum_cycle_period: int = 8

    def validate(self) -> ByteEventConfig:
        if self.minimum_support < 1 or not 0 < self.confidence_threshold <= 1:
            raise ValueError("invalid byte shelf gate")
        if self.confidence_z < 0:
            raise ValueError("confidence_z cannot be negative")
        if self.maximum_shelf_burst < 1 or not 0 < self.cumulative_risk_budget <= 1:
            raise ValueError("invalid byte shelf controls")
        if self.neural_anchor_interval < 1 or self.raw_context_bytes < 1:
            raise ValueError("context and anchor intervals must be positive")
        if self.cycle_repetitions < 2 or self.maximum_cycle_period < 1:
            raise ValueError("invalid cycle guard")
        return self


@dataclass(frozen=True)
class ByteEventGenerationResult:
    prompt_bytes: bytes
    generated_bytes: bytes
    routes: tuple[str, ...]
    shelf_bytes: int
    neural_bytes: int
    control_rejections: int
    neural_parameter_bytes_proxy: int

    @property
    def shelf_fraction(self) -> float:
        total = self.shelf_bytes + self.neural_bytes
        return self.shelf_bytes / total if total else 0.0


def _cycle_repeats(prefix: bytearray, candidate: int, config: ByteEventConfig) -> bool:
    sequence = prefix + bytes([candidate])
    maximum = min(
        config.maximum_cycle_period, len(sequence) // config.cycle_repetitions
    )
    for period in range(1, maximum + 1):
        suffix = sequence[-period:]
        if all(
            sequence[-repeat * period : -(repeat - 1) * period] == suffix
            for repeat in range(2, config.cycle_repetitions + 1)
        ):
            return True
    return False


def _dynamic_context(
    bridge: ByteBPEBridge,
    prefix: bytearray,
    model: ByteEventLM,
    raw_context_bytes: int,
) -> torch.Tensor:
    token_ids = bridge.encode_bytes(prefix[-raw_context_bytes:])
    context = np.zeros(model.context_size, dtype=np.int64)
    selected = token_ids[-model.context_size :]
    context[-len(selected) :] = selected
    return torch.from_numpy(context[None, :])


def generate_byte_events(
    model: ByteEventLM,
    bridge: ByteBPEBridge,
    shelf_levels: list[CompactShelfLevel],
    prompt_bytes: bytes,
    *,
    max_new_bytes: int,
    config: ByteEventConfig | None = None,
    shelf_enabled: bool = True,
) -> ByteEventGenerationResult:
    """Greedily generate bytes while invoking the neural core only on fallback events."""

    config = (config or ByteEventConfig()).validate()
    if max_new_bytes < 1 or not prompt_bytes:
        raise ValueError("prompt and max_new_bytes must be non-empty")
    if not shelf_levels:
        raise ValueError("at least one shelf level is required")
    if bridge.vocab_size != model.vocab_size:
        raise ValueError("bridge and event model vocabularies differ")
    if any(
        level.contexts and int(level.top_tokens.max()) > 255 for level in shelf_levels
    ):
        raise ValueError("byte shelf predicts a value outside [0, 255]")

    device = next(model.parameters()).device
    prefix = bytearray(prompt_bytes)
    generated = bytearray()
    routes: list[str] = []
    shelf_bytes = neural_bytes = rejected = 0
    burst = 0
    risk = 0.0
    since_neural = 0
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for _ in range(max_new_bytes):
                candidate = None
                if shelf_enabled:
                    maximum_order = max(level.order for level in shelf_levels)
                    context = np.frombuffer(
                        prefix[-maximum_order:], dtype=np.uint8
                    ).astype(np.uint32)
                    candidate = predict_shelf_next(
                        shelf_levels,
                        context,
                        minimum_support=config.minimum_support,
                        confidence_threshold=config.confidence_threshold,
                        confidence_z=config.confidence_z,
                    )
                    if candidate is not None:
                        candidate_risk = 1 - candidate.lower_confidence
                        blocked = (
                            burst >= config.maximum_shelf_burst
                            or risk + candidate_risk > config.cumulative_risk_budget
                            or since_neural >= config.neural_anchor_interval
                            or _cycle_repeats(prefix, candidate.token, config)
                        )
                        if blocked:
                            candidate = None
                            rejected += 1
                if candidate is not None:
                    emitted = candidate.token
                    shelf_bytes += 1
                    burst += 1
                    risk += 1 - candidate.lower_confidence
                    routes.append("shelf")
                else:
                    dynamic = _dynamic_context(
                        bridge, prefix, model, config.raw_context_bytes
                    ).to(device)
                    emitted = int(model(dynamic).argmax(dim=-1))
                    neural_bytes += 1
                    burst = 0
                    risk = 0.0
                    since_neural = 0
                    routes.append("neural")
                generated.append(emitted)
                prefix.append(emitted)
                since_neural += 1
    finally:
        model.train(was_training)
    return ByteEventGenerationResult(
        prompt_bytes=prompt_bytes,
        generated_bytes=bytes(generated),
        routes=tuple(routes),
        shelf_bytes=shelf_bytes,
        neural_bytes=neural_bytes,
        control_rejections=rejected,
        neural_parameter_bytes_proxy=neural_bytes * model.parameter_bytes,
    )


@dataclass(frozen=True)
class CognitiveCascadeResult:
    route: str
    payload: Any
    memory_hit: MemoryHit | None
    generation: ByteEventGenerationResult | None


class AIraCascade:
    """Request-level explicit memory route followed by the byte event decoder.

    Memory is queried only when the caller supplies a canonical user-visible fact key.
    Accepted facts can be returned directly with provenance; unknown or conflicted keys
    fall through to generation rather than injecting an untrusted near match. Autonomous
    shelf routing is disabled unless a fitted reliability threshold is supplied.
    """

    def __init__(
        self,
        model: ByteEventLM,
        bridge: ByteBPEBridge,
        shelf_levels: list[CompactShelfLevel],
        facts: EpisodicFactStore,
        *,
        config: ByteEventConfig | None = None,
        shelf_reliability: ReliabilityThreshold | None = None,
    ) -> None:
        self.model = model
        self.bridge = bridge
        self.shelf_levels = shelf_levels
        self.facts = facts
        self.config = (config or ByteEventConfig()).validate()
        self.shelf_reliability = shelf_reliability

    def resolve(
        self,
        prompt_bytes: bytes,
        *,
        max_new_bytes: int,
        memory_key: str | None = None,
        shelf_enabled: bool = True,
    ) -> CognitiveCascadeResult:
        hit = self.facts.recall(memory_key) if memory_key is not None else None
        if hit is not None and hit.accepted:
            return CognitiveCascadeResult("memory", hit.payload, hit, None)
        calibrated_threshold = (
            self.shelf_reliability.threshold
            if self.shelf_reliability is not None
            else None
        )
        effective_shelf = shelf_enabled and calibrated_threshold is not None
        effective_config = (
            replace(
                self.config,
                confidence_threshold=max(
                    self.config.confidence_threshold, calibrated_threshold
                ),
            )
            if calibrated_threshold is not None
            else self.config
        )
        generation = generate_byte_events(
            self.model,
            self.bridge,
            self.shelf_levels,
            prompt_bytes,
            max_new_bytes=max_new_bytes,
            config=effective_config,
            shelf_enabled=effective_shelf,
        )
        return CognitiveCascadeResult(
            "shelf-neural",
            generation.generated_bytes,
            hit,
            generation,
        )
