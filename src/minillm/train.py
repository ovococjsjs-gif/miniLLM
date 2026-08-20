"""Small, deterministic training smoke test—not a pretend foundation-model trainer."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from .config import MiniLLMConfig
from .data import ByteTokenizer, random_language_batch
from .model import MiniLLM

_SMOKE_CORPUS = (
    """
A small model should know when it does not know. It should retrieve facts instead of
inventing them, use a calculator for exact arithmetic, and spend extra computation only
on difficult questions. Маленькая модель должна быть быстрой, управляемой и честной.
Memory is not one thing: working context, learned static patterns, personal episodic
facts, and tool results need different storage and different trust rules.
"""
    * 80
)


@dataclass(frozen=True)
class SmokeResult:
    losses: tuple[float, ...]
    parameter_count: int


def smoke_train(
    config: MiniLLMConfig,
    *,
    steps: int = 8,
    batch_size: int = 2,
    sequence_length: int = 48,
    learning_rate: float = 2e-3,
    seed: int = 7,
) -> SmokeResult:
    """Run a few optimizer steps to validate forward/backward and loss plumbing."""

    if config.vocab_size != ByteTokenizer.vocab_size:
        raise ValueError(
            "smoke_train expects the 260-entry ByteTokenizer configuration"
        )
    torch.manual_seed(seed)
    rng = random.Random(seed)
    tokenizer = ByteTokenizer()
    tokens = tokenizer.encode(_SMOKE_CORPUS)
    model = MiniLLM(config)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.95)
    )
    losses: list[float] = []
    for step in range(steps):
        batch = random_language_batch(
            tokens,
            batch_size=batch_size,
            sequence_length=sequence_length,
            rng=rng,
        )
        # Sample depth during training so the shared core cannot rely on one fixed unroll.
        recurrences = 1 + (step % config.max_core_repetitions)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch, labels=batch, core_repetitions=recurrences)
        assert output.loss is not None
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(output.loss.detach()))
    return SmokeResult(
        tuple(losses), sum(parameter.numel() for parameter in model.parameters())
    )
