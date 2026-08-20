#!/usr/bin/env python3
"""Train tokenizer candidates from deterministic samples of corpus shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from minillm.corpus import CorpusDocument
from minillm.corpus_streaming import read_sharded_documents
from minillm.tokenization import tokenizer_report, train_byte_bpe


def stable_bucket(identifier: str, modulus: int = 1_000_000) -> int:
    digest = hashlib.blake2s(identifier.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % modulus


def sampled_documents(
    corpus: str | Path,
    *,
    split: str,
    languages: frozenset[str],
    maximum_bytes_per_language: int,
    sample_parts_per_million: int,
) -> Iterator[CorpusDocument]:
    used: Counter[str] = Counter()
    for document in read_sharded_documents(corpus, split=split):
        if document.language not in languages:
            continue
        if stable_bucket(document.id) >= sample_parts_per_million:
            continue
        byte_count = len(document.text.encode("utf-8"))
        if used[document.language] + byte_count > maximum_bytes_per_language:
            continue
        used[document.language] += byte_count
        yield document


def sample_summary(documents: Iterator[CorpusDocument]) -> dict[str, object]:
    count = 0
    byte_counts: Counter[str] = Counter()
    digest = hashlib.sha256()
    for document in documents:
        count += 1
        byte_counts[document.language] += len(document.text.encode("utf-8"))
        digest.update(document.id.encode())
        digest.update(document.sha256.encode())
    return {
        "documents": count,
        "bytes_by_language": dict(sorted(byte_counts.items())),
        "sample_sha256": digest.hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus")
    parser.add_argument("--output", default="runs/tokenizer-candidates")
    parser.add_argument(
        "--vocab-sizes", nargs="+", type=int, default=[16000, 32000, 48000]
    )
    parser.add_argument("--languages", nargs="+", default=["en", "ru"])
    parser.add_argument("--train-mib-per-language", type=int, default=256)
    parser.add_argument("--eval-mib-per-language", type=int, default=16)
    parser.add_argument("--sample-parts-per-million", type=int, default=1_000_000)
    parser.add_argument("--d-model", type=int, default=1024)
    args = parser.parse_args()
    if not 1 <= args.sample_parts_per_million <= 1_000_000:
        raise ValueError("sample rate must be in [1, 1_000_000]")
    languages = frozenset(args.languages)
    train_bytes = args.train_mib_per_language * 1024 * 1024
    eval_bytes = args.eval_mib_per_language * 1024 * 1024
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    expected_outputs = [output / "report.json"] + [
        output / f"byte-bpe-{vocabulary}.json" for vocabulary in args.vocab_sizes
    ]
    if any(path.exists() for path in expected_outputs):
        raise FileExistsError("refusing to overwrite tokenizer candidates")

    def sample(split: str, maximum_bytes: int) -> Iterator[CorpusDocument]:
        return sampled_documents(
            args.corpus,
            split=split,
            languages=languages,
            maximum_bytes_per_language=maximum_bytes,
            sample_parts_per_million=args.sample_parts_per_million,
        )

    training_summary = sample_summary(sample("train", train_bytes))
    if training_summary["documents"] == 0:
        raise ValueError("deterministic training sample is empty")
    validation_documents = tuple(sample("validation", eval_bytes))
    if not validation_documents:
        raise ValueError("deterministic validation sample is empty")
    corpus_manifest = json.loads(
        (Path(args.corpus) / "manifest.json").read_text(encoding="utf-8")
    )

    candidates = []
    for vocabulary in args.vocab_sizes:
        tokenizer_path = output / f"byte-bpe-{vocabulary}.json"
        tokenizer = train_byte_bpe(
            (document.text for document in sample("train", train_bytes)),
            vocab_size=vocabulary,
            output_path=tokenizer_path,
        )
        digest = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
        candidates.append(
            {
                "requested_vocab_size": vocabulary,
                "tokenizer_sha256": digest,
                "path": tokenizer_path.name,
                "report": tokenizer_report(
                    tokenizer, validation_documents, d_model=args.d_model
                ),
            }
        )

    report = {
        "schema_version": 1,
        "corpus_manifest_sha256": corpus_manifest["corpus_sha256"],
        "corpus_policy": corpus_manifest["policy"]["id"],
        "languages": sorted(languages),
        "training_sample": training_summary,
        "validation_documents": len(validation_documents),
        "validation_bytes_by_language": {
            language: sum(
                len(document.text.encode("utf-8"))
                for document in validation_documents
                if document.language == language
            )
            for language in sorted(languages)
        },
        "candidates": candidates,
        "selection_rule": "Choose Pareto point using bytes/token, per-language tokens/word, embedding bytes, LM-head FLOPs, and downstream iso-byte runs.",
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
