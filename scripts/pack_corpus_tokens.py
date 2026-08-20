#!/usr/bin/env python3
"""Stream deterministic corpus shards into uint32 training-token files."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from minillm.corpus_streaming import read_sharded_documents
from minillm.tokenization import load_tokenizer
from minillm.training import pack_document_stream


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-manifest")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    corpus = Path(args.corpus)
    corpus_manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    tokenizer_path = Path(args.tokenizer)
    tokenizer_digest = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    if args.tokenizer_manifest:
        tokenizer_manifest = json.loads(
            Path(args.tokenizer_manifest).read_text(encoding="utf-8")
        )
        if tokenizer_manifest["tokenizer_sha256"] != tokenizer_digest:
            raise ValueError("frozen tokenizer hash mismatch")
        if (
            tokenizer_manifest["corpus_manifest_sha256"]
            != corpus_manifest["corpus_sha256"]
        ):
            raise ValueError("tokenizer was frozen against a different corpus")

    tokenizer = load_tokenizer(tokenizer_path)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError("output already contains a packed-token manifest")
    splits = {}
    for split in ("train", "validation", "test"):
        manifest = pack_document_stream(
            read_sharded_documents(corpus, split=split),
            tokenizer,
            tokenizer_path=tokenizer_path,
            output_path=output / f"{split}.bin",
        )
        splits[split] = asdict(manifest)

    report = {
        "schema_version": 1,
        "corpus_sha256": corpus_manifest["corpus_sha256"],
        "corpus_policy": corpus_manifest["policy"]["id"],
        "tokenizer_sha256": tokenizer_digest,
        "tokenizer_vocab_size": tokenizer.get_vocab_size(),
        "dtype": "uint32",
        "splits": splits,
    }
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output / "manifest.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
