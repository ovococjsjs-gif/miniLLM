#!/usr/bin/env python3
"""Build hash-bound event-model shards from frozen shelf and raw UTF-8 text."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from minillm.aira import (
    ByteBPEBridge,
    EventDatasetWriter,
    SourceCopyIndex,
    build_compact_shelf,
    build_event_training_batch,
    pack_event_stream,
)
from minillm.tokenization import load_tokenizer


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def pack_split(
    writer: EventDatasetWriter,
    split: str,
    text: str,
    bridge: ByteBPEBridge,
    levels,
    *,
    chunk_bytes: int,
    prompt_bytes: int,
    prompt_copy: bool,
    token_context: int,
    raw_context_bytes: int,
    maximum_literal_bytes: int,
) -> dict[str, int | float]:
    values = text.encode("utf-8")
    chunks = events = target_bytes = shelf_bytes = source_bytes = 0
    for start in range(0, len(values) - prompt_bytes, chunk_bytes):
        chunk = values[start : start + chunk_bytes]
        if len(chunk) <= prompt_bytes:
            continue
        prefix = chunk[:prompt_bytes]
        target = chunk[prompt_bytes:]
        source_index = (
            SourceCopyIndex({f"{split}-prompt-{start}": prefix})
            if prompt_copy
            else None
        )
        packed = pack_event_stream(
            target,
            prefix=prefix,
            shelf_levels=levels,
            source_index=source_index,
            minimum_support=5,
            confidence_threshold=0.95,
            confidence_z=1.96,
            minimum_copy_bytes=2,
            minimum_shelf_copy_bytes=2,
            minimum_source_copy_bytes=8 if prompt_copy else 2,
            maximum_copy_bytes=32,
            maximum_literal_bytes=maximum_literal_bytes,
            cumulative_risk_budget=0.10,
        )
        batch = build_event_training_batch(
            packed,
            bridge,
            raw_context_bytes=raw_context_bytes,
            token_context=token_context,
            maximum_literal_bytes=maximum_literal_bytes,
        )
        writer.add(split, batch)
        chunks += 1
        events += packed.metrics.events
        target_bytes += packed.metrics.bytes
        shelf_bytes += packed.metrics.shelf_copy_bytes
        source_bytes += packed.metrics.source_copy_bytes
    return {
        "chunks": chunks,
        "events": events,
        "target_bytes": target_bytes,
        "event_sequence_compression": target_bytes / events,
        "shelf_copy_fraction": shelf_bytes / target_bytes,
        "source_copy_fraction": source_bytes / target_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shelf-text", required=True)
    parser.add_argument("--train-text", required=True)
    parser.add_argument("--validation-text", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shelf-characters", type=int, default=2_000_000)
    parser.add_argument("--max-train-characters", type=int, default=8_000_000)
    parser.add_argument("--max-validation-characters", type=int, default=300_000)
    parser.add_argument("--chunk-bytes", type=int, default=4096)
    parser.add_argument("--prompt-bytes", type=int, default=512)
    parser.add_argument("--raw-context-bytes", type=int, default=128)
    parser.add_argument("--token-context", type=int, default=16)
    parser.add_argument("--maximum-literal-bytes", type=int, default=8)
    parser.add_argument("--maximum-events-per-shard", type=int, default=100_000)
    parser.add_argument("--prompt-copy", action="store_true")
    args = parser.parse_args()
    shelf_path = Path(args.shelf_text)
    train_path = Path(args.train_text)
    validation_path = Path(args.validation_text)
    tokenizer_path = Path(args.tokenizer)
    tokenizer = load_tokenizer(tokenizer_path)
    bridge = ByteBPEBridge.from_tokenizer_json(tokenizer_path)

    normalize = tokenizer.normalizer.normalize_str
    shelf_text = normalize(
        shelf_path.read_text(encoding="utf-8", errors="replace")[
            : args.shelf_characters
        ]
    )
    train_text = normalize(
        train_path.read_text(encoding="utf-8", errors="replace")[
            : args.max_train_characters
        ]
    )
    validation_text = normalize(
        validation_path.read_text(encoding="utf-8", errors="replace")[
            : args.max_validation_characters
        ]
    )
    shelf_units = np.frombuffer(shelf_text.encode("utf-8"), dtype=np.uint8).astype(
        np.uint32
    )
    levels = [build_compact_shelf(shelf_units, order) for order in (4, 8, 16)]
    metadata = {
        "schema": "aira-v2-event-training-v1",
        "tokenizer": str(tokenizer_path),
        "tokenizer_sha256": sha256(tokenizer_path),
        "shelf_source": str(shelf_path),
        "shelf_source_sha256": sha256(shelf_path),
        "train_source": str(train_path),
        "train_source_sha256": sha256(train_path),
        "validation_source": str(validation_path),
        "validation_source_sha256": sha256(validation_path),
        "shelf_orders": [4, 8, 16],
        "packed_shelf_bytes": sum(level.packed_bytes for level in levels),
        "raw_context_bytes": args.raw_context_bytes,
        "token_context": args.token_context,
        "maximum_literal_bytes": args.maximum_literal_bytes,
        "prompt_copy": args.prompt_copy,
    }
    writer = EventDatasetWriter(
        args.output,
        maximum_events_per_shard=args.maximum_events_per_shard,
        metadata=metadata,
    )
    split_reports = {
        "train": pack_split(
            writer,
            "train",
            train_text,
            bridge,
            levels,
            chunk_bytes=args.chunk_bytes,
            prompt_bytes=args.prompt_bytes,
            prompt_copy=args.prompt_copy,
            token_context=args.token_context,
            raw_context_bytes=args.raw_context_bytes,
            maximum_literal_bytes=args.maximum_literal_bytes,
        ),
        "validation": pack_split(
            writer,
            "validation",
            validation_text,
            bridge,
            levels,
            chunk_bytes=args.chunk_bytes,
            prompt_bytes=args.prompt_bytes,
            prompt_copy=args.prompt_copy,
            token_context=args.token_context,
            raw_context_bytes=args.raw_context_bytes,
            maximum_literal_bytes=args.maximum_literal_bytes,
        ),
    }
    manifest = writer.close()
    report = {"manifest": manifest, "splits": split_reports}
    report_path = Path(args.output) / "packing-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
