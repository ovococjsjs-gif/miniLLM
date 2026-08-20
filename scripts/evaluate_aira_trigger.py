#!/usr/bin/env python3
"""Evaluate AIra-v2 trigger shelves on a frozen raw-text holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from minillm.aira import build_compact_shelf, evaluate_hierarchical_shelf


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def encode_char60(train_text: str, text: str) -> tuple[np.ndarray, dict[str, object]]:
    counts = Counter(train_text)
    selected = sorted(
        symbol
        for symbol, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[
            :60
        ]
    )
    mapping = {symbol: index + 4 for index, symbol in enumerate(selected)}
    encoded = np.fromiter((mapping.get(symbol, 3) for symbol in text), dtype=np.uint32)
    return encoded, {
        "vocab_size": 64,
        "symbols": selected,
        "unknown_rate": float(np.mean(encoded == 3)),
    }


def encode_bytes(text: str) -> tuple[np.ndarray, dict[str, object]]:
    encoded = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.uint32)
    return encoded, {"vocab_size": 256, "unknown_rate": 0.0}


def evaluate_representation(
    name: str,
    train: np.ndarray,
    validation: np.ndarray,
    *,
    orders: tuple[int, ...],
    metadata: dict[str, object],
) -> dict[str, object]:
    started = time.perf_counter()
    levels = [build_compact_shelf(train, order) for order in orders]
    build_seconds = time.perf_counter() - started
    gates = (
        ("empirical-n2-p90", 2, 0.90, 0.0),
        ("empirical-n5-p90", 5, 0.90, 0.0),
        ("empirical-n5-p95", 5, 0.95, 0.0),
        ("wilson95-n2-p90", 2, 0.90, 1.96),
        ("wilson95-n5-p95", 5, 0.95, 1.96),
    )
    evaluations = {}
    for gate, support, threshold, z in gates:
        result = evaluate_hierarchical_shelf(
            levels,
            validation,
            minimum_support=support,
            confidence_threshold=threshold,
            confidence_z=z,
            selection="longest",
        )
        evaluations[gate] = result.to_dict()
    return {
        "representation": name,
        "train_units": len(train),
        "validation_units": len(validation),
        "orders": orders,
        "metadata": metadata,
        "build_seconds": build_seconds,
        "packed_shelf_bytes": sum(level.packed_bytes for level in levels),
        "levels": [
            {
                "order": level.order,
                "contexts": level.contexts,
                "packed_bytes": level.packed_bytes,
                "contexts_n2": int(np.sum(level.totals >= 2)),
                "contexts_n5": int(np.sum(level.totals >= 5)),
            }
            for level in levels
        ],
        "evaluations": evaluations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--max-train-characters", type=int, default=2_000_000)
    parser.add_argument("--max-validation-characters", type=int, default=300_000)
    parser.add_argument("--output", default="results/aira_trigger_proxy.json")
    args = parser.parse_args()
    train_path = Path(args.train)
    validation_path = Path(args.validation)
    train_text = train_path.read_text(encoding="utf-8", errors="replace")[
        : args.max_train_characters
    ]
    validation_text = validation_path.read_text(encoding="utf-8", errors="replace")[
        : args.max_validation_characters
    ]
    if len(train_text) < 1000 or len(validation_text) < 1000:
        raise ValueError(
            "trigger evaluation needs at least 1000 train/validation characters"
        )

    train_chars, char_metadata = encode_char60(train_text, train_text)
    validation_chars, validation_char_metadata = encode_char60(
        train_text, validation_text
    )
    char_metadata["validation_unknown_rate"] = validation_char_metadata["unknown_rate"]
    train_bytes, byte_metadata = encode_bytes(train_text)
    validation_bytes, _ = encode_bytes(validation_text)

    payload = {
        "schema_version": 1,
        "experiment": "aira-v2-frozen-trigger-v1",
        "tag": args.tag,
        "source": {
            "train": str(train_path),
            "train_sha256": file_sha256(train_path),
            "validation": str(validation_path),
            "validation_sha256": file_sha256(validation_path),
        },
        "protocol": {
            "frozen_holdout": True,
            "online_validation_updates": False,
            "hash": "uint64 FNV-like; collision risk disclosed",
            "selection": "longest reliable context",
            "note": "Correct-burst lengths are teacher-forced upper bounds, not autonomous generation speedups.",
        },
        "representations": [
            evaluate_representation(
                "char60",
                train_chars,
                validation_chars,
                orders=(4, 6, 8),
                metadata=char_metadata,
            ),
            evaluate_representation(
                "utf8-byte",
                train_bytes,
                validation_bytes,
                orders=(4, 8, 16),
                metadata=byte_metadata,
            ),
        ],
    }
    output = Path(args.output)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("schema_version") != 1:
            raise ValueError("cannot merge an incompatible trigger report")
        reports = existing.setdefault("reports", [])
        reports[:] = [report for report in reports if report["tag"] != args.tag]
        reports.append(payload)
        final = existing
    else:
        final = {
            "schema_version": 1,
            "warning": "Frozen small-text proxy; no end-to-end generation or energy claim.",
            "reports": [payload],
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
