"""Bounded one-shot associative memory with an explicit familiarity gate."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MemoryHit:
    accepted: bool
    payload: Any | None
    similarity: float
    margin: float
    slot: int | None
    provenance: dict[str, Any] | None


_KEY_SEPARATOR = re.compile(r"[^\w]+", re.UNICODE)


class StructuredKeyEncoder:
    """Stable bipolar codes for explicit, user-visible fact keys.

    This encoder deliberately canonicalizes identifiers rather than claiming semantic
    paraphrase understanding. For example, ``user.favorite-color`` and
    ``USER favorite color`` address the same key; unrelated natural-language questions do
    not. Semantic retrieval remains a separate learned-encoder experiment.
    """

    def __init__(self, dimension: int = 512, *, namespace: str = "aira-v2") -> None:
        if dimension < 8:
            raise ValueError("key-code dimension must be at least eight")
        self.dimension = dimension
        self.namespace = namespace

    @staticmethod
    def canonicalize(key: str) -> str:
        normalized = unicodedata.normalize("NFKC", key).casefold().strip()
        return _KEY_SEPARATOR.sub(".", normalized).strip(".")

    def encode(self, key: str) -> np.ndarray:
        canonical = self.canonicalize(key)
        if not canonical:
            raise ValueError("memory key cannot be empty")
        payload = f"{self.namespace}\0{canonical}".encode()
        digest = hashlib.shake_256(payload).digest((self.dimension + 7) // 8)
        bits = np.unpackbits(np.frombuffer(digest, dtype=np.uint8))[: self.dimension]
        return np.where(bits == 0, -1, 1).astype(np.int8)


class BoundedAssociativeMemory:
    """Reference bipolar-code memory with ring-bounded storage.

    Retrieval is an honest O(ND) scan. AIra's original energy comparison sometimes called
    DAM lookup effectively O(1), but its implementation scanned every stored code. This
    reference reports that cost explicitly; an ANN/LSH index is a separate optimization.
    """

    def __init__(
        self,
        capacity: int,
        dimension: int,
        *,
        similarity_threshold: float = 0.25,
        margin_threshold: float = 0.02,
    ) -> None:
        if capacity < 1 or dimension < 8:
            raise ValueError("capacity must be positive and dimension at least eight")
        if not -1 <= similarity_threshold <= 1 or margin_threshold < 0:
            raise ValueError("invalid familiarity thresholds")
        self.capacity = capacity
        self.dimension = dimension
        self.similarity_threshold = similarity_threshold
        self.margin_threshold = margin_threshold
        self.codes = np.zeros((capacity, dimension), dtype=np.int8)
        self.payloads: list[Any | None] = [None] * capacity
        self.provenance: list[dict[str, Any] | None] = [None] * capacity
        self.active = np.zeros(capacity, dtype=bool)
        self.size = 0
        self.next_slot = 0

    def _code(self, vector: np.ndarray) -> np.ndarray:
        array = np.asarray(vector)
        if array.shape != (self.dimension,):
            raise ValueError(f"memory code must have shape ({self.dimension},)")
        code = np.where(array >= 0, 1, -1).astype(np.int8)
        return code

    def write(
        self,
        vector: np.ndarray,
        payload: Any,
        *,
        provenance: dict[str, Any] | None = None,
    ) -> int:
        slot = self.next_slot
        if self.size < self.capacity:
            for offset in range(self.capacity):
                candidate = (self.next_slot + offset) % self.capacity
                if not self.active[candidate]:
                    slot = candidate
                    break
        replacing = bool(self.active[slot])
        self.codes[slot] = self._code(vector)
        self.payloads[slot] = payload
        self.provenance[slot] = dict(provenance or {})
        self.active[slot] = True
        self.next_slot = (slot + 1) % self.capacity
        if not replacing:
            self.size += 1
        return slot

    def delete(self, slot: int) -> bool:
        """Delete one fact by slot, returning whether an active fact was removed."""

        if not 0 <= slot < self.capacity:
            raise ValueError("slot is outside memory capacity")
        if not self.active[slot]:
            return False
        self.active[slot] = False
        self.codes[slot].fill(0)
        self.payloads[slot] = None
        self.provenance[slot] = None
        self.size -= 1
        return True

    def query(self, vector: np.ndarray) -> MemoryHit:
        if self.size == 0:
            return MemoryHit(False, None, float("-inf"), 0.0, None, None)
        query = self._code(vector).astype(np.float32)
        contiguous = self.size == self.capacity or (
            bool(np.all(self.active[: self.size]))
            and not bool(np.any(self.active[self.size :]))
        )
        active_slots = None if contiguous else np.flatnonzero(self.active)
        active_codes = (
            self.codes[: self.size] if contiguous else self.codes[active_slots]
        )
        scores = active_codes.astype(np.float32) @ query / self.dimension
        if self.size > 1:
            top_two = np.argpartition(scores, -2)[-2:]
            ranked = top_two[np.argsort(scores[top_two])]
            best_score_index = int(ranked[-1])
            best_slot = (
                best_score_index
                if active_slots is None
                else int(active_slots[best_score_index])
            )
            second = float(scores[ranked[-2]])
        else:
            best_score_index = 0
            best_slot = 0 if active_slots is None else int(active_slots[0])
            second = -1.0
        best = float(scores[best_score_index])
        margin = best - second
        accepted = best >= self.similarity_threshold and margin >= self.margin_threshold
        return MemoryHit(
            accepted=accepted,
            payload=self.payloads[best_slot] if accepted else None,
            similarity=best,
            margin=margin,
            slot=best_slot if accepted else None,
            provenance=self.provenance[best_slot] if accepted else None,
        )

    @property
    def code_storage_bytes(self) -> int:
        return self.codes.nbytes

    def scan_operations(self) -> int:
        return self.size * self.dimension


class EpisodicFactStore:
    """One-shot explicit fact API over bounded associative storage."""

    def __init__(
        self,
        capacity: int,
        *,
        dimension: int = 512,
        namespace: str = "aira-v2-facts",
    ) -> None:
        self.encoder = StructuredKeyEncoder(dimension, namespace=namespace)
        self.memory = BoundedAssociativeMemory(
            capacity,
            dimension,
            similarity_threshold=0.75,
            margin_threshold=0.02,
        )

    def remember(
        self,
        key: str,
        value: Any,
        *,
        provenance: dict[str, Any] | None = None,
    ) -> int:
        return self.memory.write(self.encoder.encode(key), value, provenance=provenance)

    def recall(self, key: str) -> MemoryHit:
        return self.memory.query(self.encoder.encode(key))

    def delete(self, slot: int) -> bool:
        return self.memory.delete(slot)
