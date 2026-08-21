#!/usr/bin/env python3
"""Measure offline event-sequence compression before expensive model training."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

from minillm.aira import (
    SourceCopyIndex,
    build_compact_shelf,
    pack_event_stream,
)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def aggregate(chunks: list[dict[str, int | float]]) -> dict[str, int | float]:
    keys = (
        "bytes",
        "events",
        "literal_events",
        "shelf_copy_events",
        "source_copy_events",
        "literal_bytes",
        "shelf_copy_bytes",
        "source_copy_bytes",
        "neural_output_events",
    )
    result = {key: sum(int(chunk[key]) for chunk in chunks) for key in keys}
    result["event_sequence_compression"] = result["bytes"] / result["events"]
    result["neural_invocation_reduction_upper_bound"] = (
        result["bytes"] / result["neural_output_events"]
        if result["neural_output_events"]
        else float("inf")
    )
    result["shelf_copy_fraction"] = result["shelf_copy_bytes"] / result["bytes"]
    result["source_copy_fraction"] = result["source_copy_bytes"] / result["bytes"]
    return result


def run_configuration(
    validation: bytes,
    levels,
    *,
    name: str,
    maximum_literal_bytes: int,
    use_shelf: bool,
    use_prompt_copy: bool,
    minimum_copy_bytes: int,
    chunk_bytes: int,
    prompt_bytes: int,
) -> dict[str, object]:
    rows = []
    started = time.perf_counter()
    for chunk_start in range(0, len(validation) - prompt_bytes, chunk_bytes):
        chunk = validation[chunk_start : chunk_start + chunk_bytes]
        if len(chunk) <= prompt_bytes:
            continue
        prefix = chunk[:prompt_bytes]
        target = chunk[prompt_bytes:]
        source_index = (
            SourceCopyIndex({"prompt": prefix}, anchor_bytes=4)
            if use_prompt_copy
            else None
        )
        packed = pack_event_stream(
            target,
            prefix=prefix,
            shelf_levels=levels if use_shelf else None,
            source_index=source_index,
            minimum_support=5,
            confidence_threshold=0.95,
            confidence_z=1.96,
            minimum_copy_bytes=2,
            minimum_shelf_copy_bytes=2,
            minimum_source_copy_bytes=minimum_copy_bytes,
            maximum_copy_bytes=32,
            maximum_literal_bytes=maximum_literal_bytes,
            cumulative_risk_budget=0.10,
        )
        rows.append(packed.metrics.to_dict())
    return {
        "name": name,
        "maximum_literal_bytes": maximum_literal_bytes,
        "shelf": use_shelf,
        "prompt_copy": use_prompt_copy,
        "minimum_copy_bytes": minimum_copy_bytes,
        "seconds": time.perf_counter() - started,
        "metrics": aggregate(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--max-train-characters", type=int, default=2_000_000)
    parser.add_argument("--max-validation-characters", type=int, default=100_000)
    parser.add_argument("--chunk-bytes", type=int, default=4096)
    parser.add_argument("--prompt-bytes", type=int, default=512)
    parser.add_argument("--output", default="results/aira_event_packing_proxy.json")
    args = parser.parse_args()
    train_path = Path(args.train)
    validation_path = Path(args.validation)
    train_text = train_path.read_text(encoding="utf-8", errors="replace")[
        : args.max_train_characters
    ]
    validation_text = validation_path.read_text(encoding="utf-8", errors="replace")[
        : args.max_validation_characters
    ]
    train = np.frombuffer(train_text.encode("utf-8"), dtype=np.uint8).astype(np.uint32)
    validation = validation_text.encode("utf-8")
    levels = [build_compact_shelf(train, order) for order in (4, 8, 16)]
    configurations = (
        ("literal-byte", 1, False, False, 2),
        ("literal-4byte-head", 4, False, False, 2),
        ("literal-8byte-head", 8, False, False, 2),
        ("shelf-plus-byte", 1, True, False, 2),
        ("shelf-plus-8byte-head", 8, True, False, 2),
        ("shelf-prompt-copy-8byte-min2", 8, True, True, 2),
        ("shelf-prompt-copy-8byte-min8", 8, True, True, 8),
        ("shelf-prompt-copy-8byte-min16", 8, True, True, 16),
    )
    results = [
        run_configuration(
            validation,
            levels,
            name=name,
            maximum_literal_bytes=literal_bytes,
            use_shelf=use_shelf,
            use_prompt_copy=use_prompt_copy,
            minimum_copy_bytes=minimum_copy_bytes,
            chunk_bytes=args.chunk_bytes,
            prompt_bytes=args.prompt_bytes,
        )
        for name, literal_bytes, use_shelf, use_prompt_copy, minimum_copy_bytes in configurations
    ]
    payload = {
        "schema_version": 1,
        "experiment": "aira-v2-offline-event-packing-upper-bound-v1",
        "warning": "Lossless teacher-forced event labels. Multi-byte and copy compression are upper bounds until an action model passes generated-context calibration.",
        "source": {
            "train": str(train_path),
            "train_sha256": sha256(train_path),
            "validation": str(validation_path),
            "validation_sha256": sha256(validation_path),
        },
        "environment": {
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "protocol": {
            "train_characters": len(train_text),
            "validation_characters": len(validation_text),
            "validation_bytes": len(validation),
            "chunk_bytes": args.chunk_bytes,
            "prompt_bytes_per_chunk": args.prompt_bytes,
            "shelf_orders": [4, 8, 16],
            "gate": "support>=5 and per-byte Wilson95>=0.95, cumulative span risk<=0.10",
            "minimum_copy_bytes_tested": [2, 8, 16],
            "maximum_copy_bytes": 32,
        },
        "packed_shelf_bytes": sum(level.packed_bytes for level in levels),
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
