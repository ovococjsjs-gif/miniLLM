#!/usr/bin/env python3
"""Build a tokenizer-bound compact shelf archive for triggered generation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from minillm.aira import build_compact_shelf, save_compact_shelf
from minillm.tokenization import load_tokenizer


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--orders", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument(
        "--representation",
        choices=("utf8-byte", "token-ids"),
        default="utf8-byte",
    )
    parser.add_argument("--max-characters", type=int, default=2_000_000)
    args = parser.parse_args()
    if len(set(args.orders)) != len(args.orders) or any(
        order < 1 for order in args.orders
    ):
        raise ValueError("orders must be unique positive integers")
    text_path = Path(args.text)
    tokenizer_path = Path(args.tokenizer)
    output = Path(args.output)
    text = text_path.read_text(encoding="utf-8", errors="replace")[
        : args.max_characters
    ]
    tokenizer = load_tokenizer(tokenizer_path)
    normalized = tokenizer.normalizer.normalize_str(text)
    if args.representation == "utf8-byte":
        units = np.frombuffer(normalized.encode("utf-8"), dtype=np.uint8).astype(
            np.uint32
        )
    else:
        units = np.asarray(tokenizer.encode(normalized).ids, dtype=np.uint32)
    if len(units) <= max(args.orders):
        raise ValueError("training text encodes to too few shelf units")
    levels = [build_compact_shelf(units, order) for order in args.orders]
    tokenizer_hash = sha256(tokenizer_path)
    save_compact_shelf(
        output,
        levels,
        tokenizer_sha256=tokenizer_hash,
        representation=args.representation,
    )
    manifest = {
        "schema_version": 1,
        "artifact": "aira-v2-compact-shelf-v1",
        "archive": str(output),
        "archive_sha256": sha256(output),
        "source_text": str(text_path),
        "source_text_sha256": sha256(text_path),
        "used_characters": len(text),
        "tokenizer": str(tokenizer_path),
        "tokenizer_sha256": tokenizer_hash,
        "vocab_size": tokenizer.get_vocab_size(),
        "representation": args.representation,
        "shelf_units": len(units),
        "orders": args.orders,
        "levels": [
            {
                "order": level.order,
                "contexts": level.contexts,
                "packed_bytes": level.packed_bytes,
            }
            for level in levels
        ],
        "packed_bytes": sum(level.packed_bytes for level in levels),
        "archive_bytes": output.stat().st_size,
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
