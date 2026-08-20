#!/usr/bin/env python3
"""Tokenize JSONL corpus splits into compact uint32 streams."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minillm.corpus import read_jsonl
from minillm.tokenization import load_tokenizer
from minillm.training import pack_documents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/tokenizer-proxy")
    parser.add_argument("--tokenizer", default="runs/tokenizers/byte_bpe_4096.json")
    parser.add_argument("--output", default="data/tokens-4096")
    args = parser.parse_args()
    tokenizer = load_tokenizer(args.tokenizer)
    output = Path(args.output)
    manifests = {}
    for split in ("train", "validation", "test"):
        documents = list(read_jsonl(Path(args.corpus) / f"{split}.jsonl"))
        manifest = pack_documents(
            documents,
            tokenizer,
            tokenizer_path=args.tokenizer,
            output_path=output / f"{split}.bin",
        )
        manifests[split] = manifest.__dict__
    print(json.dumps(manifests, indent=2))


if __name__ == "__main__":
    main()
