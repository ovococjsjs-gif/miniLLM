#!/usr/bin/env python3
"""Stream selected Common Corpus records into auditable MiniLLM JSONL documents."""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterator
from pathlib import Path

from minillm.corpus import CorpusDocument, write_jsonl
from minillm.importers import canonical_common_corpus_language, common_corpus_documents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--acquisition-date", required=True)
    parser.add_argument("--languages", nargs="+", default=["en", "ru"])
    parser.add_argument("--maximum-documents", type=int)
    parser.add_argument("--maximum-characters", type=int, default=100_000)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    if args.maximum_characters < 1000:
        raise ValueError("maximum record chunk must be at least 1000 characters")

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "Common Corpus import requires the optional 'datasets' package"
        ) from error

    stream = load_dataset(
        "PleIAs/common_corpus",
        split="train",
        streaming=True,
        revision=args.revision,
    )
    allowed_languages = set(args.languages)

    def selected() -> Iterator[CorpusDocument]:
        accepted_records = 0
        for record in stream:
            language = canonical_common_corpus_language(record.get("language", ""))
            if language not in allowed_languages:
                continue
            yielded = False
            for document in common_corpus_documents(
                record,
                acquisition_date=args.acquisition_date,
                maximum_characters=args.maximum_characters,
            ):
                yielded = True
                yield document
            if yielded:
                accepted_records += 1
                if (
                    args.maximum_documents is not None
                    and accepted_records >= args.maximum_documents
                ):
                    break

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if output.exists() or temporary.exists():
        raise FileExistsError("refusing to overwrite an imported corpus snapshot")
    try:
        write_jsonl(temporary, selected())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
