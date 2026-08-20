from __future__ import annotations

from pathlib import Path

import numpy as np

from minillm.aira import ByteBPEBridge, build_compact_shelf
from minillm.tokenization import load_tokenizer

ROOT = Path(__file__).parents[1]
TOKENIZER = ROOT / "artifacts/tokenizer-github-pilot-v1/tokenizer.json"


def test_byte_bpe_bridge_reconstructs_normalized_text_exactly() -> None:
    tokenizer = load_tokenizer(TOKENIZER)
    bridge = ByteBPEBridge.from_tokenizer_json(TOKENIZER)
    text = 'Hello, мир! JSON: {"value": Ⅳ}\n'
    normalized = tokenizer.normalizer.normalize_str(text)
    token_ids = tokenizer.encode(text).ids

    reconstructed = bridge.tokens_to_bytes(token_ids)

    assert reconstructed == normalized.encode("utf-8")
    dynamic_ids = bridge.encode_bytes(normalized.encode("utf-8"))
    assert bridge.tokens_to_bytes(dynamic_ids) == normalized.encode("utf-8")
    assert len(dynamic_ids) < len(normalized.encode("utf-8"))
    partial_utf8 = b"prefix \xd0"
    assert bridge.tokens_to_bytes(bridge.encode_bytes(partial_utf8)) == partial_utf8
    assert bridge.vocab_size == tokenizer.get_vocab_size()


def test_bridge_returns_longest_complete_bpe_piece_from_byte_shelf() -> None:
    bridge = ByteBPEBridge.from_tokenizer_json(TOKENIZER)
    stream = np.frombuffer(b" world" * 200, dtype=np.uint8).astype(np.uint32)
    shelf = build_compact_shelf(stream, order=4)

    prediction = bridge.draft_token(
        [shelf],
        b" world",
        minimum_support=5,
        confidence_threshold=0.9,
        confidence_z=0,
    )

    assert prediction is not None
    assert prediction.token_bytes == b" world"
    assert bridge.token_bytes[prediction.token] == b" world"
    assert len(prediction.byte_predictions) == len(prediction.token_bytes)
    assert prediction.lower_confidence == 1.0
