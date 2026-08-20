#!/usr/bin/env python3
"""Export document-level provenance required to audit or attribute corpus inputs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from collections import Counter
from pathlib import Path

from minillm.corpus_streaming import read_sharded_documents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    corpus = Path(args.corpus)
    corpus_manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    temporary = output.with_suffix(output.suffix + ".tmp")
    if output.exists() or temporary.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite attribution bundle")

    counts: Counter[str] = Counter()
    documents = 0
    raw = temporary.open("wb")
    compressed = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    text = io.TextIOWrapper(compressed, encoding="utf-8", newline="\n")
    try:
        for document in read_sharded_documents(corpus):
            record = {
                "id": document.id,
                "source": document.source,
                "license": document.license,
                "url": document.url,
                "title": document.metadata.get("title"),
                "creator": document.metadata.get("creator"),
                "content_sha256": document.sha256,
            }
            text.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            counts[document.license] += 1
            documents += 1
    finally:
        text.flush()
        text.close()
        if not raw.closed:
            raw.close()
    os.replace(temporary, output)

    digest = hashlib.sha256()
    with output.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    manifest = {
        "schema_version": 1,
        "corpus_sha256": corpus_manifest["corpus_sha256"],
        "corpus_policy": corpus_manifest["policy"]["id"],
        "documents": documents,
        "licenses": dict(sorted(counts.items())),
        "bundle": output.name,
        "bundle_sha256": digest.hexdigest(),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
