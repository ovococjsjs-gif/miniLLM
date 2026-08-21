from __future__ import annotations

import numpy as np

from minillm.aira.bridge import ByteBPEBridge
from minillm.aira.events import (
    EventKind,
    SourceCopyIndex,
    build_event_training_batch,
    pack_event_stream,
)
from minillm.aira.trigger import build_compact_shelf


def test_event_packer_losslessly_compresses_reliable_shelf_spans() -> None:
    pattern = b"hello world\n"
    train = np.frombuffer(pattern * 200, dtype=np.uint8).astype(np.uint32)
    levels = [build_compact_shelf(train, order) for order in (4, 8)]

    packed = pack_event_stream(
        pattern * 10,
        prefix=pattern,
        shelf_levels=levels,
        confidence_threshold=0.9,
        confidence_z=0,
        minimum_copy_bytes=2,
        maximum_copy_bytes=16,
    )

    assert packed.expand() == pattern * 10
    assert packed.metrics.shelf_copy_bytes > 0
    assert packed.metrics.event_sequence_compression > 2
    assert any(event.kind == EventKind.SHELF_COPY for event in packed.events)


def test_source_copy_index_emits_auditable_exact_span() -> None:
    source = b"The verified answer is Pskov."
    index = SourceCopyIndex({"document-7": source}, anchor_bytes=4)
    target = b"Pskov. Pskov."

    packed = pack_event_stream(
        target,
        source_index=index,
        minimum_copy_bytes=4,
        maximum_literal_bytes=4,
    )

    copies = [event for event in packed.events if event.kind == EventKind.SOURCE_COPY]
    assert packed.expand() == target
    assert copies
    assert copies[0].source_id == "document-7"
    assert (
        source[copies[0].source_offset : copies[0].source_offset + copies[0].length]
        == copies[0].payload
    )


def test_packed_stream_converts_to_multibyte_training_arrays() -> None:
    pieces: list[bytes | None] = [bytes([value]) for value in range(256)]
    bridge = ByteBPEBridge(tuple(pieces))
    index = SourceCopyIndex({"document": b"Pskov city"})
    packed = pack_event_stream(
        b"Answer: Pskov city",
        prefix=b"Question?",
        source_index=index,
        minimum_copy_bytes=5,
        maximum_literal_bytes=4,
    )

    batch = build_event_training_batch(
        packed, bridge, token_context=8, maximum_literal_bytes=4
    )

    assert batch.contexts.shape == (packed.metrics.events, 8)
    assert batch.byte_targets.shape == (packed.metrics.events, 4)
    assert batch.byte_supervised.sum() == packed.metrics.literal_events
    assert batch.route_targets.tolist() == [int(event.kind) for event in packed.events]
    assert batch.full_event_lengths.max() >= 5


def test_event_packer_keeps_unmatched_bytes_in_bounded_literal_patches() -> None:
    target = b"abcdefghijk"
    packed = pack_event_stream(target, maximum_literal_bytes=3)

    assert packed.expand() == target
    assert all(event.kind == EventKind.LITERAL for event in packed.events)
    assert all(1 <= event.length <= 3 for event in packed.events)
    assert packed.metrics.events == 4
