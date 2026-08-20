#!/usr/bin/env python3
"""Freeze one measured tokenizer candidate with corpus-bound provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from minillm.tokenization import SPECIAL_TOKENS, load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report_path = Path(args.report)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    matches = [
        item
        for item in report["candidates"]
        if item["requested_vocab_size"] == args.vocab_size
    ]
    if len(matches) != 1:
        raise ValueError("requested tokenizer candidate is absent or ambiguous")
    candidate = matches[0]
    source = Path(candidate["path"])
    if not source.is_absolute():
        source = report_path.parent / source
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_digest != candidate["tokenizer_sha256"]:
        raise ValueError("candidate tokenizer hash does not match report")

    tokenizer = load_tokenizer(source)
    special_ids = {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS}
    if any(value is None for value in special_ids.values()):
        raise ValueError("candidate tokenizer is missing required special tokens")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output / "tokenizer.json"
    manifest_path = output / "manifest.json"
    if tokenizer_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite a frozen tokenizer")
    temporary = tokenizer_path.with_suffix(".json.tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, tokenizer_path)

    manifest = {
        "schema_version": 1,
        "tokenizer_sha256": source_digest,
        "actual_vocab_size": tokenizer.get_vocab_size(),
        "requested_vocab_size": args.vocab_size,
        "special_token_ids": special_ids,
        "corpus_manifest_sha256": report["corpus_manifest_sha256"],
        "corpus_policy": report["corpus_policy"],
        "training_sample": report["training_sample"],
        "candidate_report": candidate["report"],
        "normalization": "NFKC",
        "pre_tokenizer": "ByteLevel(add_prefix_space=False)",
        "decoder": "ByteLevel",
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
