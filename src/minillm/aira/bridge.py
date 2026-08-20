"""Deterministic bridge from raw UTF-8 byte triggers to ByteLevel-BPE tokens."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np

from .trigger import CompactShelfLevel, ShelfPrediction, predict_shelf_next


def _bytelevel_inverse_alphabet() -> dict[str, int]:
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    unicode_values = byte_values.copy()
    extra = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            unicode_values.append(256 + extra)
            extra += 1
    return {chr(symbol): byte for byte, symbol in zip(byte_values, unicode_values)}


@dataclass(frozen=True)
class BridgedShelfPrediction:
    token: int
    token_bytes: bytes
    byte_predictions: tuple[ShelfPrediction, ...]

    @property
    def lower_confidence(self) -> float:
        confidence = 1.0
        for prediction in self.byte_predictions:
            confidence *= prediction.lower_confidence
        return confidence

    @property
    def empirical_confidence(self) -> float:
        confidence = 1.0
        for prediction in self.byte_predictions:
            confidence *= prediction.empirical_confidence
        return confidence

    @property
    def cumulative_risk(self) -> float:
        return sum(
            1.0 - prediction.lower_confidence for prediction in self.byte_predictions
        )


class ByteBPEBridge:
    """Map shelf-predicted bytes to valid tokens from a ByteLevel BPE vocabulary.

    Drafting continues along a vocabulary-prefix trie and returns the longest complete
    token. Tentative bytes after that token are discarded. The check is deterministic
    tokenization plumbing, not neural verification.
    """

    def __init__(
        self,
        token_bytes: tuple[bytes | None, ...],
        *,
        merge_ranks: dict[tuple[bytes, bytes], int] | None = None,
    ) -> None:
        if not token_bytes:
            raise ValueError("token vocabulary cannot be empty")
        self.token_bytes = token_bytes
        self.exact_tokens: dict[bytes, int] = {}
        self.prefixes: set[bytes] = set()
        for token, piece in enumerate(token_bytes):
            if not piece:
                continue
            if piece in self.exact_tokens:
                raise ValueError("ByteLevel vocabulary contains duplicate byte pieces")
            self.exact_tokens[piece] = token
            for length in range(1, len(piece) + 1):
                self.prefixes.add(piece[:length])
        if not self.exact_tokens:
            raise ValueError("token vocabulary contains no byte pieces")
        self.merge_ranks = dict(merge_ranks or {})
        self.maximum_token_bytes = max(map(len, self.exact_tokens))

    @classmethod
    def from_tokenizer_json(cls, path: str | Path) -> ByteBPEBridge:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("model", {}).get("type") != "BPE":
            raise ValueError("bridge requires a BPE tokenizer")
        if payload.get("pre_tokenizer", {}).get("type") != "ByteLevel":
            raise ValueError("bridge requires a ByteLevel pre-tokenizer")
        vocabulary = payload["model"]["vocab"]
        size = max(int(index) for index in vocabulary.values()) + 1
        special_ids = {
            int(token["id"])
            for token in payload.get("added_tokens", [])
            if token.get("special", False)
        }
        inverse = _bytelevel_inverse_alphabet()

        def symbol_bytes(symbol: str) -> bytes:
            try:
                return bytes(inverse[character] for character in symbol)
            except KeyError as error:
                raise ValueError(
                    "BPE symbol is outside the ByteLevel alphabet"
                ) from error

        pieces: list[bytes | None] = [None] * size
        for symbol, raw_index in vocabulary.items():
            index = int(raw_index)
            if index not in special_ids:
                pieces[index] = symbol_bytes(symbol)
        merge_ranks = {
            (symbol_bytes(left), symbol_bytes(right)): rank
            for rank, (left, right) in enumerate(payload["model"].get("merges", []))
        }
        return cls(tuple(pieces), merge_ranks=merge_ranks)

    @property
    def vocab_size(self) -> int:
        return len(self.token_bytes)

    def encode_bytes(self, values: bytes | bytearray) -> list[int]:
        """Apply the tokenizer's learned BPE merges directly to arbitrary bytes.

        Operating below Unicode lets the residual core consume a bounded context even
        when an event occurs in the middle of a multi-byte UTF-8 character. Pretokenizer
        regex boundaries are intentionally omitted; this is a deterministic dynamic patch
        representation, not guaranteed to match canonical whole-text tokenization.
        """

        pieces = [bytes([value]) for value in values]
        while len(pieces) > 1:
            best_rank: int | None = None
            best_pair: tuple[bytes, bytes] | None = None
            for left, right in pairwise(pieces):
                rank = self.merge_ranks.get((left, right))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = (left, right)
            if best_pair is None:
                break
            merged: list[bytes] = []
            index = 0
            while index < len(pieces):
                if (
                    index + 1 < len(pieces)
                    and (pieces[index], pieces[index + 1]) == best_pair
                ):
                    merged.append(pieces[index] + pieces[index + 1])
                    index += 2
                else:
                    merged.append(pieces[index])
                    index += 1
            pieces = merged
        return [self.exact_tokens[piece] for piece in pieces]

    def tokens_to_bytes(
        self, token_ids: Iterable[int], *, strict_special_tokens: bool = False
    ) -> bytes:
        pieces = []
        for token in token_ids:
            if not 0 <= token < self.vocab_size:
                raise ValueError("token is outside bridge vocabulary")
            piece = self.token_bytes[token]
            if piece is None:
                if strict_special_tokens:
                    raise ValueError("special token has no raw-byte representation")
                continue
            pieces.append(piece)
        return b"".join(pieces)

    def suffix_bytes_after_special(self, token_ids: Iterable[int]) -> bytes:
        """Return only the byte suffix after the most recent special token."""

        suffix = bytearray()
        for token in token_ids:
            if not 0 <= token < self.vocab_size:
                raise ValueError("token is outside bridge vocabulary")
            piece = self.token_bytes[token]
            if piece is None:
                suffix.clear()
            else:
                suffix.extend(piece)
        return bytes(suffix)

    def draft_token(
        self,
        shelf_levels: list[CompactShelfLevel],
        prefix_bytes: bytes | bytearray,
        *,
        minimum_support: int = 5,
        confidence_threshold: float = 0.95,
        confidence_z: float = 1.96,
        selection: str = "longest",
        maximum_bytes: int | None = None,
    ) -> BridgedShelfPrediction | None:
        if not shelf_levels:
            raise ValueError("at least one shelf level is required")
        limit = min(maximum_bytes or self.maximum_token_bytes, self.maximum_token_bytes)
        if limit < 1:
            raise ValueError("maximum_bytes must be positive")
        draft = bytearray()
        predictions: list[ShelfPrediction] = []
        best_token: int | None = None
        best_length = 0
        maximum_order = max(level.order for level in shelf_levels)
        context = bytearray(prefix_bytes[-maximum_order:])
        for _ in range(limit):
            prediction = predict_shelf_next(
                shelf_levels,
                np.frombuffer(context, dtype=np.uint8).astype(np.uint32),
                minimum_support=minimum_support,
                confidence_threshold=confidence_threshold,
                confidence_z=confidence_z,
                selection=selection,
            )
            if prediction is None or not 0 <= prediction.token <= 255:
                break
            draft.append(prediction.token)
            predictions.append(prediction)
            candidate = bytes(draft)
            if candidate not in self.prefixes:
                break
            context.append(prediction.token)
            token = self.exact_tokens.get(candidate)
            if token is not None:
                best_token = token
                best_length = len(candidate)
        if best_token is None:
            return None
        return BridgedShelfPrediction(
            token=best_token,
            token_bytes=bytes(draft[:best_length]),
            byte_predictions=tuple(predictions[:best_length]),
        )
