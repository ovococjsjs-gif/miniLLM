#!/usr/bin/env python3
"""Convert exact GitHub corpus checkouts into MiniLLM document JSONL."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Iterator
from pathlib import Path

from minillm.corpus import CorpusDocument, write_jsonl
from minillm.github_importers import (
    import_oanc_checkout,
    import_rusdracor_checkout,
    import_ruslit_checkout,
)


def verify_checkout(path: Path, revision: str) -> None:
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()
    if actual != revision:
        raise ValueError(f"checkout {path} is at {actual}, expected {revision}")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=path, text=True
    ).strip()
    if dirty:
        raise ValueError(f"checkout {path} has uncommitted changes")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", default="configs/corpus/github_pilot_sources.json")
    parser.add_argument("--checkouts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--acquisition-date", required=True)
    parser.add_argument("--maximum-characters", type=int, default=100_000)
    args = parser.parse_args()
    if args.maximum_characters < 1000:
        raise ValueError("maximum chunk size must be at least 1000 characters")

    config = json.loads(Path(args.sources).read_text(encoding="utf-8"))
    checkout_root = Path(args.checkouts)

    def documents() -> Iterator[CorpusDocument]:
        for source in config["sources"]:
            checkout = checkout_root / source["checkout"]
            verify_checkout(checkout, source["revision"])
            common = {
                "checkout": checkout,
                "repository": source["repository"],
                "revision": source["revision"],
                "acquisition_date": args.acquisition_date,
                "sample_parts_per_million": source["sample_parts_per_million"],
                "maximum_characters": args.maximum_characters,
            }
            if source["importer"] == "oanc":
                yield from import_oanc_checkout(**common)
            elif source["importer"] == "ruslit":
                yield from import_ruslit_checkout(
                    **common,
                    allowed_authors=frozenset(source["allowed_authors"]),
                )
            elif source["importer"] == "rusdracor":
                yield from import_rusdracor_checkout(**common)
            else:
                raise ValueError(f"unknown importer {source['importer']}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if output.exists() or temporary.exists():
        raise FileExistsError("refusing to overwrite imported GitHub documents")
    try:
        write_jsonl(temporary, documents())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
