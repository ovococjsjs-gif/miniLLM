"""Lossless event packing for shelf, source-copy, and literal byte spans."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import IntEnum

import numpy as np

from .bridge import ByteBPEBridge
from .trigger import CompactShelfLevel, predict_shelf_next


class EventKind(IntEnum):
    LITERAL = 0
    SHELF_COPY = 1
    SOURCE_COPY = 2
    MEMORY_REFERENCE = 3
    TOOL_RESULT = 4


@dataclass(frozen=True)
class PackedEvent:
    kind: EventKind
    payload: bytes
    source_id: str | None = None
    source_offset: int | None = None
    minimum_confidence: float | None = None

    @property
    def length(self) -> int:
        return len(self.payload)

    @property
    def checksum(self) -> str:
        return hashlib.blake2s(self.payload, digest_size=8).hexdigest()

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["kind"] = self.kind.name.lower()
        result["payload_hex"] = self.payload.hex()
        del result["payload"]
        result["length"] = self.length
        result["checksum"] = self.checksum
        return result


@dataclass(frozen=True)
class EventPackingMetrics:
    bytes: int
    events: int
    literal_events: int
    shelf_copy_events: int
    source_copy_events: int
    literal_bytes: int
    shelf_copy_bytes: int
    source_copy_bytes: int
    event_sequence_compression: float
    neural_output_events: int
    neural_invocation_reduction_upper_bound: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class PackedEventStream:
    prefix: bytes
    target: bytes
    events: tuple[PackedEvent, ...]
    metrics: EventPackingMetrics

    def expand(self) -> bytes:
        return b"".join(event.payload for event in self.events)

    def validate_lossless(self) -> None:
        if self.expand() != self.target:
            raise ValueError("event stream does not reconstruct its target")
        if any(event.length < 1 for event in self.events):
            raise ValueError("event stream contains an empty event")


class SourceCopyIndex:
    """Bounded exact-copy index over trusted prompt/retrieval byte sources."""

    def __init__(
        self,
        sources: Mapping[str, bytes],
        *,
        anchor_bytes: int = 4,
        maximum_candidates: int = 64,
    ) -> None:
        if anchor_bytes < 2 or maximum_candidates < 1:
            raise ValueError("invalid source-copy index configuration")
        self.sources = dict(sources)
        self.anchor_bytes = anchor_bytes
        self.maximum_candidates = maximum_candidates
        self.positions: dict[bytes, list[tuple[str, int]]] = {}
        for source_id, value in self.sources.items():
            for offset in range(len(value) - anchor_bytes + 1):
                anchor = value[offset : offset + anchor_bytes]
                bucket = self.positions.setdefault(anchor, [])
                if len(bucket) < maximum_candidates:
                    bucket.append((source_id, offset))

    def longest_match(
        self,
        target: bytes,
        position: int,
        *,
        maximum_length: int,
    ) -> tuple[str, int, bytes] | None:
        if position + self.anchor_bytes > len(target):
            return None
        anchor = target[position : position + self.anchor_bytes]
        best: tuple[str, int, bytes] | None = None
        for source_id, offset in self.positions.get(anchor, ()):
            source = self.sources[source_id]
            length_limit = min(
                maximum_length,
                len(target) - position,
                len(source) - offset,
            )
            length = self.anchor_bytes
            while (
                length < length_limit
                and target[position + length] == source[offset + length]
            ):
                length += 1
            payload = target[position : position + length]
            if best is None or len(payload) > len(best[2]):
                best = (source_id, offset, payload)
        return best


def _shelf_copy_candidate(
    levels: list[CompactShelfLevel],
    prefix: bytearray,
    target: bytes,
    position: int,
    *,
    minimum_support: int,
    confidence_threshold: float,
    confidence_z: float,
    maximum_length: int,
    cumulative_risk_budget: float,
) -> tuple[bytes, float] | None:
    copied = bytearray()
    minimum_confidence = 1.0
    risk = 0.0
    maximum_order = max(level.order for level in levels)
    for offset in range(min(maximum_length, len(target) - position)):
        context = np.frombuffer(prefix[-maximum_order:], dtype=np.uint8).astype(
            np.uint32
        )
        candidate = predict_shelf_next(
            levels,
            context,
            minimum_support=minimum_support,
            confidence_threshold=confidence_threshold,
            confidence_z=confidence_z,
        )
        if candidate is None or candidate.token != target[position + offset]:
            break
        candidate_risk = 1 - candidate.lower_confidence
        if risk + candidate_risk > cumulative_risk_budget:
            break
        copied.append(candidate.token)
        prefix.append(candidate.token)
        risk += candidate_risk
        minimum_confidence = min(minimum_confidence, candidate.lower_confidence)
    if not copied:
        return None
    del prefix[-len(copied) :]
    return bytes(copied), minimum_confidence


def pack_event_stream(
    target: bytes,
    *,
    prefix: bytes = b"",
    shelf_levels: list[CompactShelfLevel] | None = None,
    source_index: SourceCopyIndex | None = None,
    minimum_support: int = 5,
    confidence_threshold: float = 0.95,
    confidence_z: float = 1.96,
    minimum_copy_bytes: int = 2,
    minimum_shelf_copy_bytes: int | None = None,
    minimum_source_copy_bytes: int | None = None,
    maximum_copy_bytes: int = 32,
    maximum_literal_bytes: int = 8,
    cumulative_risk_budget: float = 0.10,
) -> PackedEventStream:
    """Create lossless oracle labels for an event model.

    Copy events are selected only when their deterministic expansion equals the target.
    This is an offline supervised upper bound, not autonomous proof: deployment still
    requires generated-context calibration of the action selector.
    """

    if not target:
        raise ValueError("target cannot be empty")
    shelf_minimum = minimum_shelf_copy_bytes or minimum_copy_bytes
    source_minimum = minimum_source_copy_bytes or minimum_copy_bytes
    if (
        minimum_copy_bytes < 1
        or shelf_minimum < 1
        or source_minimum < 1
        or maximum_copy_bytes < max(shelf_minimum, source_minimum)
    ):
        raise ValueError("invalid copy span lengths")
    if maximum_literal_bytes < 1 or not 0 < cumulative_risk_budget <= 1:
        raise ValueError("invalid literal/risk configuration")
    if shelf_levels is not None and not shelf_levels:
        raise ValueError("shelf_levels cannot be an empty list")

    generated = bytearray(prefix)
    events: list[PackedEvent] = []
    position = 0
    while position < len(target):
        shelf_match = (
            _shelf_copy_candidate(
                shelf_levels,
                generated,
                target,
                position,
                minimum_support=minimum_support,
                confidence_threshold=confidence_threshold,
                confidence_z=confidence_z,
                maximum_length=maximum_copy_bytes,
                cumulative_risk_budget=cumulative_risk_budget,
            )
            if shelf_levels is not None
            else None
        )
        source_match = (
            source_index.longest_match(
                target, position, maximum_length=maximum_copy_bytes
            )
            if source_index is not None
            else None
        )
        shelf_length = len(shelf_match[0]) if shelf_match is not None else 0
        source_length = len(source_match[2]) if source_match is not None else 0
        eligible_shelf = shelf_length >= shelf_minimum
        eligible_source = source_length >= source_minimum
        if eligible_shelf or eligible_source:
            if eligible_shelf and (
                not eligible_source or shelf_length >= source_length
            ):
                assert shelf_match is not None
                payload, confidence = shelf_match
                event = PackedEvent(
                    EventKind.SHELF_COPY,
                    payload,
                    minimum_confidence=confidence,
                )
            else:
                assert source_match is not None
                source_id, offset, payload = source_match
                event = PackedEvent(
                    EventKind.SOURCE_COPY,
                    payload,
                    source_id=source_id,
                    source_offset=offset,
                )
            events.append(event)
            generated.extend(event.payload)
            position += event.length
            continue

        literal_start = position
        position += 1
        while (
            position < len(target) and position - literal_start < maximum_literal_bytes
        ):
            prospective_shelf = (
                _shelf_copy_candidate(
                    shelf_levels,
                    generated + target[literal_start:position],
                    target,
                    position,
                    minimum_support=minimum_support,
                    confidence_threshold=confidence_threshold,
                    confidence_z=confidence_z,
                    maximum_length=maximum_copy_bytes,
                    cumulative_risk_budget=cumulative_risk_budget,
                )
                if shelf_levels is not None
                else None
            )
            prospective_source = (
                source_index.longest_match(
                    target, position, maximum_length=maximum_copy_bytes
                )
                if source_index is not None
                else None
            )
            if (
                prospective_shelf is not None
                and len(prospective_shelf[0]) >= shelf_minimum
            ) or (
                prospective_source is not None
                and len(prospective_source[2]) >= source_minimum
            ):
                break
            position += 1
        payload = target[literal_start:position]
        events.append(PackedEvent(EventKind.LITERAL, payload))
        generated.extend(payload)

    literal_events = sum(event.kind == EventKind.LITERAL for event in events)
    shelf_events = sum(event.kind == EventKind.SHELF_COPY for event in events)
    source_events = sum(event.kind == EventKind.SOURCE_COPY for event in events)
    literal_bytes = sum(
        event.length for event in events if event.kind == EventKind.LITERAL
    )
    shelf_bytes = sum(
        event.length for event in events if event.kind == EventKind.SHELF_COPY
    )
    source_bytes = sum(
        event.length for event in events if event.kind == EventKind.SOURCE_COPY
    )
    neural_events = literal_events + source_events
    metrics = EventPackingMetrics(
        bytes=len(target),
        events=len(events),
        literal_events=literal_events,
        shelf_copy_events=shelf_events,
        source_copy_events=source_events,
        literal_bytes=literal_bytes,
        shelf_copy_bytes=shelf_bytes,
        source_copy_bytes=source_bytes,
        event_sequence_compression=len(target) / len(events),
        neural_output_events=neural_events,
        neural_invocation_reduction_upper_bound=(
            len(target) / neural_events if neural_events else float("inf")
        ),
    )
    stream = PackedEventStream(prefix, target, tuple(events), metrics)
    stream.validate_lossless()
    return stream


@dataclass(frozen=True)
class EventTrainingBatch:
    contexts: np.ndarray
    byte_targets: np.ndarray
    target_lengths: np.ndarray
    route_targets: np.ndarray
    byte_supervised: np.ndarray
    full_event_lengths: np.ndarray


def build_event_training_batch(
    stream: PackedEventStream,
    bridge: ByteBPEBridge,
    *,
    raw_context_bytes: int = 64,
    token_context: int = 16,
    maximum_literal_bytes: int = 8,
    pad_token: int = 0,
) -> EventTrainingBatch:
    """Convert packed events into arrays consumed by ``MultiByteEventLM``."""

    if raw_context_bytes < 1 or token_context < 1 or maximum_literal_bytes < 1:
        raise ValueError("invalid event training context")
    contexts = []
    byte_targets = []
    lengths = []
    routes = []
    supervised = []
    full_lengths = []
    prefix = bytearray(stream.prefix)
    for event in stream.events:
        if event.kind == EventKind.LITERAL and event.length > maximum_literal_bytes:
            raise ValueError("literal event exceeds the configured multi-byte head")
        token_ids = bridge.encode_bytes(prefix[-raw_context_bytes:])
        context = np.full(token_context, pad_token, dtype=np.int64)
        selected = token_ids[-token_context:]
        if selected:
            context[-len(selected) :] = selected
        target = np.zeros(maximum_literal_bytes, dtype=np.int64)
        visible = event.payload[:maximum_literal_bytes]
        target[: len(visible)] = np.frombuffer(visible, dtype=np.uint8)
        contexts.append(context)
        byte_targets.append(target)
        lengths.append(max(1, len(visible)))
        routes.append(int(event.kind))
        supervised.append(event.kind == EventKind.LITERAL)
        full_lengths.append(event.length)
        prefix.extend(event.payload)
    return EventTrainingBatch(
        contexts=np.stack(contexts),
        byte_targets=np.stack(byte_targets),
        target_lengths=np.asarray(lengths, dtype=np.int64),
        route_targets=np.asarray(routes, dtype=np.int64),
        byte_supervised=np.asarray(supervised, dtype=bool),
        full_event_lengths=np.asarray(full_lengths, dtype=np.int64),
    )
