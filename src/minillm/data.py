"""Minimal data primitives for architecture smoke tests.

Large-scale corpus preparation is deliberately out of this module. The byte tokenizer
exists to test the complete model without downloading data or trusting an opaque
external tokenizer; target training should use a measured multilingual tokenizer.
"""

from __future__ import annotations

import random

import torch


class ByteTokenizer:
    bos_id = 256
    eos_id = 257
    pad_id = 258
    unk_id = 259
    vocab_size = 260

    def encode(self, text: str, *, document_tokens: bool = True) -> list[int]:
        payload = list(text.encode("utf-8", errors="replace"))
        return ([self.bos_id] + payload + [self.eos_id]) if document_tokens else payload

    def decode(self, ids: list[int]) -> str:
        payload = bytes(index for index in ids if 0 <= index < 256)
        return payload.decode("utf-8", errors="replace")


def random_language_batch(
    tokens: list[int],
    *,
    batch_size: int,
    sequence_length: int,
    device: torch.device | str = "cpu",
    rng: random.Random | None = None,
) -> torch.Tensor:
    if len(tokens) < sequence_length + 1:
        raise ValueError("token stream is shorter than one training sequence")
    rng = rng or random
    starts = [
        rng.randrange(0, len(tokens) - sequence_length) for _ in range(batch_size)
    ]
    return torch.tensor(
        [tokens[start : start + sequence_length] for start in starts],
        dtype=torch.long,
        device=device,
    )
