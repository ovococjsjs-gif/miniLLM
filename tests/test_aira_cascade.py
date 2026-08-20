from __future__ import annotations

import numpy as np
import torch

from minillm.aira import (
    AIraCascade,
    ByteBPEBridge,
    ByteEventConfig,
    ByteEventLM,
    EpisodicFactStore,
    build_compact_shelf,
    generate_byte_events,
    utf8_allowed_next_bytes,
)


def test_utf8_constraint_tracks_incomplete_multibyte_sequences() -> None:
    start = utf8_allowed_next_bytes(b"")
    assert start[ord("A")] and start[0xC2] and not start[0x80] and not start[0xC0]

    after_e0 = utf8_allowed_next_bytes(b"\xe0")
    assert after_e0[0xA0] and not after_e0[0x80]
    after_f4 = utf8_allowed_next_bytes(b"\xf4")
    assert after_f4[0x8F] and not after_f4[0x90]
    completed = utf8_allowed_next_bytes("€".encode())
    assert completed[ord("!")] and not completed[0x80]


def test_byte_event_cascade_skips_neural_core_on_reliable_bytes() -> None:
    pieces: list[bytes | None] = [None] * 300
    for value in range(256):
        pieces[value] = bytes([value])
    pieces[256] = b"ab"
    bridge = ByteBPEBridge(tuple(pieces), merge_ranks={(b"a", b"b"): 0})
    stream = np.frombuffer(b"ab" * 100, dtype=np.uint8).astype(np.uint32)
    shelf = build_compact_shelf(stream, order=2)
    torch.manual_seed(5)
    model = ByteEventLM(vocab_size=300, context_size=4, d_model=8)

    result = generate_byte_events(
        model,
        bridge,
        [shelf],
        b"ab",
        max_new_bytes=2,
        config=ByteEventConfig(
            confidence_threshold=0.9,
            confidence_z=0,
            cumulative_risk_budget=1.0,
        ),
    )

    assert result.generated_bytes == b"ab"
    assert result.routes == ("shelf", "shelf")
    assert result.shelf_bytes == 2
    assert result.neural_bytes == 0
    assert result.neural_parameter_bytes_proxy == 0


def test_request_cascade_returns_explicit_fact_or_falls_through() -> None:
    pieces: list[bytes | None] = [None] * 256
    for value in range(256):
        pieces[value] = bytes([value])
    bridge = ByteBPEBridge(tuple(pieces))
    shelf = build_compact_shelf(
        np.frombuffer(b"ab" * 20, dtype=np.uint8).astype(np.uint32), order=2
    )
    model = ByteEventLM(vocab_size=256, context_size=4, d_model=8)
    facts = EpisodicFactStore(capacity=4, dimension=64)
    facts.remember("user.home-city", "Pskov", provenance={"source": "turn"})
    cascade = AIraCascade(model, bridge, [shelf], facts)

    known = cascade.resolve(b"Where?", max_new_bytes=2, memory_key="USER home city")
    unknown = cascade.resolve(b"Where?", max_new_bytes=2, memory_key="user.work-city")

    assert known.route == "memory" and known.payload == "Pskov"
    assert known.memory_hit is not None and known.memory_hit.provenance == {
        "source": "turn"
    }
    assert known.generation is None
    assert unknown.route == "shelf-neural"
    assert unknown.memory_hit is not None and not unknown.memory_hit.accepted
    assert unknown.generation is not None
    assert unknown.generation.routes == ("neural", "neural")


def test_byte_event_cascade_can_force_neural_control() -> None:
    pieces: list[bytes | None] = [None] * 256
    for value in range(256):
        pieces[value] = bytes([value])
    bridge = ByteBPEBridge(tuple(pieces))
    shelf = build_compact_shelf(
        np.frombuffer(b"zz" * 20, dtype=np.uint8).astype(np.uint32), order=2
    )
    model = ByteEventLM(vocab_size=256, context_size=4, d_model=8)

    result = generate_byte_events(
        model,
        bridge,
        [shelf],
        b"ab",
        max_new_bytes=3,
        shelf_enabled=False,
    )

    assert result.neural_bytes == 3
    assert result.routes == ("neural",) * 3
    assert result.neural_parameter_bytes_proxy == 3 * model.parameter_bytes
