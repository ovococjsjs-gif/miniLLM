"""Trusted-checkpoint loading and deterministic reference text generation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .aira import CompactShelfLevel, predict_shelf_next
from .config import MiniLLMConfig
from .model import MiniLLM, ModelCache


@dataclass(frozen=True)
class SamplingConfig:
    max_new_tokens: int = 64
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    seed: int = 42
    use_cache: bool = True

    def validate(self) -> SamplingConfig:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("temperature cannot be negative")
        if self.top_k < 0:
            raise ValueError("top_k cannot be negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        return self


@dataclass(frozen=True)
class GenerationResult:
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    stop_reason: str
    used_cache: bool

    @property
    def all_token_ids(self) -> tuple[int, ...]:
        return self.prompt_token_ids + self.generated_token_ids


@dataclass(frozen=True)
class TriggerConfig:
    minimum_support: int = 5
    confidence_threshold: float = 0.95
    confidence_z: float = 1.96
    selection: str = "longest"
    maximum_shelf_burst: int = 4
    cumulative_risk_budget: float = 0.10
    neural_anchor_interval: int = 8
    cycle_repetitions: int = 3
    maximum_cycle_period: int = 8

    def validate(self) -> TriggerConfig:
        if self.minimum_support < 1 or not 0 < self.confidence_threshold <= 1:
            raise ValueError("invalid shelf confidence gate")
        if self.confidence_z < 0 or self.selection not in {"longest", "confidence"}:
            raise ValueError("invalid shelf confidence method")
        if self.maximum_shelf_burst < 1 or not 0 < self.cumulative_risk_budget <= 1:
            raise ValueError("invalid shelf burst controls")
        if self.neural_anchor_interval < 1:
            raise ValueError("neural anchor interval must be positive")
        if self.cycle_repetitions < 2 or self.maximum_cycle_period < 1:
            raise ValueError("invalid cycle guard")
        return self


@dataclass(frozen=True)
class TriggeredGenerationResult:
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    routes: tuple[str, ...]
    stop_reason: str
    used_cache: bool
    shelf_tokens: int
    neural_tokens: int
    neural_calls: int
    neural_input_tokens: int
    control_rejections: int

    @property
    def all_token_ids(self) -> tuple[int, ...]:
        return self.prompt_token_ids + self.generated_token_ids

    @property
    def shelf_fraction(self) -> float:
        total = self.shelf_tokens + self.neural_tokens
        return self.shelf_tokens / total if total else 0.0


@dataclass(frozen=True)
class LoadedCheckpoint:
    model: MiniLLM
    config: MiniLLMConfig
    step: int | None
    format_version: int | None
    metadata: dict[str, Any]


def load_model_checkpoint(
    checkpoint_path: str | Path,
    *,
    config_path: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> LoadedCheckpoint:
    """Load a trusted torch checkpoint for inference.

    ``torch.load`` uses pickle and must never be called on an untrusted file. Training
    format-v2 checkpoints and compact dictionaries containing ``model`` and
    ``model_config`` are supported. A raw state dict additionally requires ``config_path``.
    """

    path = Path(checkpoint_path)
    payload: Any = torch.load(path, map_location=device, weights_only=False)
    embedded_config: MiniLLMConfig | None = None
    step: int | None = None
    format_version: int | None = None
    metadata: dict[str, Any] = {}
    if isinstance(payload, dict) and "model" in payload:
        state_dict = payload["model"]
        if "model_config" in payload:
            embedded_config = MiniLLMConfig.from_dict(payload["model_config"])
        step = int(payload["step"]) if payload.get("step") is not None else None
        format_version = (
            int(payload["format_version"])
            if payload.get("format_version") is not None
            else None
        )
        if isinstance(payload.get("metadata"), dict):
            metadata = dict(payload["metadata"])
    elif isinstance(payload, dict) and all(
        isinstance(value, torch.Tensor) for value in payload.values()
    ):
        state_dict = payload
    else:
        raise ValueError("checkpoint does not contain a recognized model state")

    file_config = MiniLLMConfig.load(config_path) if config_path is not None else None
    if embedded_config is None and file_config is None:
        raise ValueError("checkpoint has no model config; pass config_path")
    if (
        embedded_config is not None
        and file_config is not None
        and embedded_config.to_dict() != file_config.to_dict()
    ):
        raise ValueError("checkpoint and external model configurations differ")
    config = embedded_config or file_config
    assert config is not None
    model = MiniLLM(config).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return LoadedCheckpoint(model, config, step, format_version, metadata)


def save_inference_checkpoint(
    path: str | Path,
    model: MiniLLM,
    *,
    step: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Atomically save weights/config without optimizer state."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(
        {
            "format_version": 1,
            "step": step,
            "model_config": model.config.to_dict(),
            "model": model.state_dict(),
            "metadata": metadata or {},
        },
        temporary,
    )
    os.replace(temporary, output)


def _sample_token(
    logits: torch.Tensor,
    config: SamplingConfig,
    generator: torch.Generator,
) -> torch.Tensor:
    if config.temperature == 0:
        return logits.argmax(dim=-1)
    filtered = logits.float() / config.temperature
    if config.top_k:
        count = min(config.top_k, filtered.shape[-1])
        threshold = torch.topk(filtered, count, dim=-1).values[..., -1, None]
        filtered = filtered.masked_fill(filtered < threshold, float("-inf"))
    if config.top_p < 1:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
        remove = torch.cumsum(sorted_probabilities, dim=-1) > config.top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        probabilities = torch.softmax(sorted_logits, dim=-1)
        sampled = torch.multinomial(probabilities, 1, generator=generator)
        return sorted_indices.gather(-1, sampled).squeeze(-1)
    probabilities = torch.softmax(filtered, dim=-1)
    return torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)


def generate_ids(
    model: MiniLLM,
    prompt_token_ids: list[int] | tuple[int, ...],
    sampling: SamplingConfig | None = None,
    *,
    stop_token_ids: set[int] | frozenset[int] | None = None,
    core_repetitions: int | None = None,
) -> GenerationResult:
    """Generate one continuation using exact cached or full-prefix reference decode."""

    sampling = (sampling or SamplingConfig()).validate()
    stop_tokens = stop_token_ids or frozenset()
    prompt = tuple(int(token) for token in prompt_token_ids)
    if not prompt:
        raise ValueError("prompt must contain at least one token")
    if len(prompt) > model.config.max_seq_len:
        raise ValueError("prompt exceeds configured maximum sequence length")
    if any(not 0 <= token < model.config.vocab_size for token in prompt):
        raise ValueError("prompt contains a token outside the model vocabulary")
    if any(not 0 <= token < model.config.vocab_size for token in stop_tokens):
        raise ValueError("stop token is outside the model vocabulary")

    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(sampling.seed)
    was_training = model.training
    model.eval()
    generated: list[int] = []
    all_tokens = list(prompt)
    use_cache = sampling.use_cache and model.supports_cached_decode
    cache: ModelCache | None = None
    stop_reason = "max_new_tokens"

    try:
        with torch.inference_mode():
            prompt_tensor = torch.tensor([prompt], dtype=torch.long, device=device)
            if use_cache:
                output, cache = model.forward_cached(
                    prompt_tensor, core_repetitions=core_repetitions
                )
            else:
                output = model(prompt_tensor, core_repetitions=core_repetitions)
            for step in range(sampling.max_new_tokens):
                if len(all_tokens) >= model.config.max_seq_len:
                    stop_reason = "max_sequence_length"
                    break
                next_token_tensor = _sample_token(
                    output.logits[:, -1], sampling, generator
                )
                next_token = int(next_token_tensor.item())
                generated.append(next_token)
                all_tokens.append(next_token)
                if next_token in stop_tokens:
                    stop_reason = "stop_token"
                    break
                if step + 1 == sampling.max_new_tokens:
                    break
                if len(all_tokens) >= model.config.max_seq_len:
                    stop_reason = "max_sequence_length"
                    break
                suffix = next_token_tensor[:, None]
                if use_cache:
                    assert cache is not None
                    output, cache = model.forward_cached(
                        suffix,
                        cache,
                        core_repetitions=core_repetitions,
                    )
                else:
                    full = torch.tensor([all_tokens], dtype=torch.long, device=device)
                    output = model(full, core_repetitions=core_repetitions)
    finally:
        model.train(was_training)

    return GenerationResult(
        prompt_token_ids=prompt,
        generated_token_ids=tuple(generated),
        stop_reason=stop_reason,
        used_cache=use_cache,
    )


def _would_repeat_cycle(
    tokens: list[int],
    candidate: int,
    *,
    repetitions: int,
    maximum_period: int,
) -> bool:
    sequence = tokens + [candidate]
    for period in range(1, min(maximum_period, len(sequence) // repetitions) + 1):
        suffix = sequence[-period:]
        if all(
            sequence[-repeat * period : -(repeat - 1) * period] == suffix
            for repeat in range(2, repetitions + 1)
        ):
            return True
    return False


def generate_triggered_ids(
    model: MiniLLM,
    shelf_levels: list[CompactShelfLevel],
    prompt_token_ids: list[int] | tuple[int, ...],
    sampling: SamplingConfig | None = None,
    trigger: TriggerConfig | None = None,
    *,
    stop_token_ids: set[int] | frozenset[int] | None = None,
    core_repetitions: int | None = None,
) -> TriggeredGenerationResult:
    """Generate with true shelf bypasses and bounded periodic neural anchors.

    Shelf tokens do not invoke the neural model. When a neural fallback is eventually
    needed, cached decoding catches up on all shelf-emitted suffix tokens in one call.
    This is an inference reference, not an optimized mobile runtime.
    """

    sampling = (sampling or SamplingConfig()).validate()
    trigger = (trigger or TriggerConfig()).validate()
    stop_tokens = stop_token_ids or frozenset()
    prompt = tuple(int(token) for token in prompt_token_ids)
    if not shelf_levels:
        raise ValueError("at least one shelf level is required")
    if not prompt:
        raise ValueError("prompt must contain at least one token")
    if len(prompt) > model.config.max_seq_len:
        raise ValueError("prompt exceeds configured maximum sequence length")
    if any(not 0 <= token < model.config.vocab_size for token in prompt):
        raise ValueError("prompt contains a token outside the model vocabulary")
    if any(not 0 <= token < model.config.vocab_size for token in stop_tokens):
        raise ValueError("stop token is outside the model vocabulary")
    if any(
        level.contexts and int(level.top_tokens.max()) >= model.config.vocab_size
        for level in shelf_levels
    ):
        raise ValueError("shelf predicts a token outside the model vocabulary")

    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(sampling.seed)
    was_training = model.training
    model.eval()
    generated: list[int] = []
    routes: list[str] = []
    all_tokens = list(prompt)
    use_cache = sampling.use_cache and model.supports_cached_decode
    cache: ModelCache | None = None
    stop_reason = "max_new_tokens"
    shelf_tokens = neural_tokens = neural_calls = neural_input_tokens = 0
    control_rejections = 0
    current_burst = 0
    cumulative_risk = 0.0
    tokens_since_neural = 0

    try:
        with torch.inference_mode():
            for _ in range(sampling.max_new_tokens):
                if len(all_tokens) >= model.config.max_seq_len:
                    stop_reason = "max_sequence_length"
                    break
                candidate = predict_shelf_next(
                    shelf_levels,
                    np.asarray(all_tokens, dtype=np.uint32),
                    minimum_support=trigger.minimum_support,
                    confidence_threshold=trigger.confidence_threshold,
                    confidence_z=trigger.confidence_z,
                    selection=trigger.selection,
                )
                use_shelf = candidate is not None
                if candidate is not None:
                    risk = 1.0 - candidate.lower_confidence
                    blocked = (
                        current_burst >= trigger.maximum_shelf_burst
                        or cumulative_risk + risk > trigger.cumulative_risk_budget
                        or tokens_since_neural >= trigger.neural_anchor_interval
                        or _would_repeat_cycle(
                            all_tokens,
                            candidate.token,
                            repetitions=trigger.cycle_repetitions,
                            maximum_period=trigger.maximum_cycle_period,
                        )
                    )
                    if blocked:
                        use_shelf = False
                        control_rejections += 1
                if use_shelf:
                    assert candidate is not None
                    next_token = candidate.token
                    shelf_tokens += 1
                    current_burst += 1
                    cumulative_risk += 1.0 - candidate.lower_confidence
                    routes.append("shelf")
                else:
                    if use_cache:
                        if cache is None:
                            neural_input = all_tokens
                        else:
                            neural_input = all_tokens[cache.token_count :]
                        input_tensor = torch.tensor(
                            [neural_input], dtype=torch.long, device=device
                        )
                        output, cache = model.forward_cached(
                            input_tensor,
                            cache,
                            core_repetitions=core_repetitions,
                        )
                    else:
                        neural_input = all_tokens
                        input_tensor = torch.tensor(
                            [neural_input], dtype=torch.long, device=device
                        )
                        output = model(input_tensor, core_repetitions=core_repetitions)
                    next_token = int(
                        _sample_token(output.logits[:, -1], sampling, generator).item()
                    )
                    neural_tokens += 1
                    neural_calls += 1
                    neural_input_tokens += len(neural_input)
                    current_burst = 0
                    cumulative_risk = 0.0
                    tokens_since_neural = 0
                    routes.append("neural")
                generated.append(next_token)
                all_tokens.append(next_token)
                tokens_since_neural += 1
                if next_token in stop_tokens:
                    stop_reason = "stop_token"
                    break
    finally:
        model.train(was_training)

    return TriggeredGenerationResult(
        prompt_token_ids=prompt,
        generated_token_ids=tuple(generated),
        routes=tuple(routes),
        stop_reason=stop_reason,
        used_cache=use_cache,
        shelf_tokens=shelf_tokens,
        neural_tokens=neural_tokens,
        neural_calls=neural_calls,
        neural_input_tokens=neural_input_tokens,
        control_rejections=control_rejections,
    )
