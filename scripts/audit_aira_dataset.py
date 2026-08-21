#!/usr/bin/env python3
"""Audit candidate JSONL datasets against the AIra base-data handoff contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from minillm.corpus import assess_quality

REQUIRED_FIELDS = (
    "document_id",
    "text",
    "language",
    "domain",
    "source",
    "license",
    "split_group",
)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def audit_file(path: Path, *, maximum_records: int | None) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    language_bytes: Counter[str] = Counter()
    domain_bytes: Counter[str] = Counter()
    licenses: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    content_hashes: set[str] = set()
    document_ids: set[str] = set()
    split_groups: set[str] = set()
    missing_fields: Counter[str] = Counter()
    examples = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if maximum_records is not None and counts["records"] >= maximum_records:
                break
            if not line.strip():
                continue
            counts["records"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                counts["invalid_json"] += 1
                continue
            for field in REQUIRED_FIELDS:
                if field not in record or record[field] is None or record[field] == "":
                    missing_fields[field] += 1
            if any(field not in record for field in REQUIRED_FIELDS):
                continue
            text = record["text"]
            if not isinstance(text, str):
                counts["non_string_text"] += 1
                continue
            encoded = text.encode("utf-8")
            digest = hashlib.sha256(encoded).hexdigest()
            if digest in content_hashes:
                counts["exact_duplicates"] += 1
            content_hashes.add(digest)
            document_id = str(record["document_id"])
            if document_id in document_ids:
                counts["duplicate_document_ids"] += 1
            document_ids.add(document_id)
            split_groups.add(str(record["split_group"]))
            language = str(record["language"])
            domain = str(record["domain"])
            license_name = str(record["license"])
            source = str(record["source"])
            language_bytes[language] += len(encoded)
            domain_bytes[domain] += len(encoded)
            licenses[license_name] += 1
            sources[source] += 1
            report = assess_quality(text)
            counts["utf8_bytes"] += len(encoded)
            counts["characters"] += report.characters
            counts["words"] += report.words
            counts["has_email"] += report.has_email
            counts["has_phone"] += report.has_phone
            counts["has_secret_pattern"] += report.has_secret_pattern
            counts["encoding_damage"] += report.replacement_character_ratio > 0.01
            counts["repeated_lines"] += report.repeated_line_ratio > 0.3
            counts["too_short"] += report.characters < 200
            if len(examples) < 5:
                examples.append(
                    {
                        "document_id": document_id,
                        "language": language,
                        "domain": domain,
                        "license": license_name,
                        "utf8_bytes": len(encoded),
                        "sha256": digest,
                    }
                )
    blocking = (
        counts["invalid_json"]
        + counts["non_string_text"]
        + counts["duplicate_document_ids"]
        + sum(missing_fields.values())
    )
    return {
        "path": str(path),
        "sha256": sha256(path),
        "contract_pass": counts["records"] > 0 and blocking == 0,
        "counts": dict(sorted(counts.items())),
        "missing_fields": dict(sorted(missing_fields.items())),
        "unique_content_hashes": len(content_hashes),
        "unique_document_ids": len(document_ids),
        "unique_split_groups": len(split_groups),
        "utf8_bytes_by_language": dict(sorted(language_bytes.items())),
        "utf8_bytes_by_domain": dict(sorted(domain_bytes.items())),
        "licenses": dict(sorted(licenses.items())),
        "sources": dict(sorted(sources.items())),
        "examples": examples,
        "note": "contract_pass checks schema/identity only; license acceptance, near-dedup, PII quarantine and eval decontamination remain policy gates",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--maximum-records", type=int)
    parser.add_argument("--output", default="results/aira_dataset_audit.json")
    args = parser.parse_args()
    reports = [
        audit_file(Path(path), maximum_records=args.maximum_records)
        for path in args.files
    ]
    payload = {
        "schema_version": 1,
        "contract": "docs/aira-training-readiness.md",
        "reports": reports,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
