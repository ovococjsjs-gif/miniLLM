#!/usr/bin/env python3
"""Build the deterministic, verifier-first AIra Mentor v1 SFT dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from minillm.aira.synthetic import generate_aira_mentor_records
from minillm.tokenization import load_tokenizer


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/aira-mentor-v1")
    parser.add_argument("--examples-per-category", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tokenizer", default="artifacts/tokenizer-github-pilot-v1/tokenizer.json"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty {output}")
    output.mkdir(parents=True, exist_ok=True)
    records = generate_aira_mentor_records(
        examples_per_category=args.examples_per_category,
        seed=args.seed,
    )
    tokenizer_path = Path(args.tokenizer)
    tokenizer = load_tokenizer(tokenizer_path) if tokenizer_path.exists() else None
    split_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    category_language_counts: Counter[str] = Counter()
    split_category_counts: Counter[str] = Counter()
    verifier_counts: Counter[str] = Counter()
    template_counts: Counter[str] = Counter()
    character_count = utf8_bytes = token_count = 0
    dedup_retry_records = 0
    maximum_dedup_attempt = 0
    corpus_digest = hashlib.sha256()
    files = {}
    for split in ("train", "validation", "test"):
        path = output / f"{split}.jsonl"
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                if record.split != split:
                    continue
                payload = record.to_dict()
                line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
                handle.write(line + "\n")
                split_counts[split] += 1
                category_counts[record.category] += 1
                language_counts[record.language] += 1
                category_language_counts[f"{record.category}:{record.language}"] += 1
                split_category_counts[f"{split}:{record.category}"] += 1
                verifier_counts[str(record.verification["kind"])] += 1
                template_counts[
                    f"{record.category}:{record.provenance['template']}"
                ] += 1
                attempt = int(record.provenance["dedup_attempt"])
                dedup_retry_records += attempt > 0
                maximum_dedup_attempt = max(maximum_dedup_attempt, attempt)
                text = "\n".join(message.content for message in record.messages)
                character_count += len(text)
                encoded = text.encode("utf-8")
                utf8_bytes += len(encoded)
                if tokenizer is not None:
                    token_count += len(tokenizer.encode(text).ids)
                corpus_digest.update(split.encode())
                corpus_digest.update(record.identifier.encode())
                corpus_digest.update(record.content_sha256.encode())
        os.replace(temporary, path)
        files[split] = {
            "path": path.name,
            "records": split_counts[split],
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    generator_path = Path("src/minillm/aira/synthetic.py")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "dataset": "aira-mentor-v1",
        "description": "Project-owned deterministic verifier-first RU/EN capability SFT seed; not base pretraining and not a proprietary-model imitation.",
        "license": "CC0-1.0",
        "seed": args.seed,
        "examples_per_category": args.examples_per_category,
        "records": len(records),
        "splits": dict(sorted(split_counts.items())),
        "categories": dict(sorted(category_counts.items())),
        "languages": dict(sorted(language_counts.items())),
        "category_languages": dict(sorted(category_language_counts.items())),
        "split_categories": dict(sorted(split_category_counts.items())),
        "verifiers": dict(sorted(verifier_counts.items())),
        "templates": dict(sorted(template_counts.items())),
        "template_count": len(template_counts),
        "characters": character_count,
        "utf8_bytes": utf8_bytes,
        "tokenizer": str(tokenizer_path) if tokenizer is not None else None,
        "tokenizer_sha256": sha256(tokenizer_path) if tokenizer is not None else None,
        "tokens": token_count if tokenizer is not None else None,
        "generator": str(generator_path),
        "generator_sha256": sha256(generator_path),
        "corpus_sha256": corpus_digest.hexdigest(),
        "files": files,
        "quality": {
            "exact_conversation_duplicates": 0,
            "dedup_retry_records": dedup_retry_records,
            "maximum_dedup_attempt": maximum_dedup_attempt,
            "all_records_verified": True,
            "hidden_chain_of_thought": False,
            "assistant_only_loss_recommended": True,
            "base_pretraining_replacement": False,
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    readme = f"""# AIra Mentor v1

Deterministic, project-owned, verifier-first RU/EN seed dataset for later capability SFT and AI Babysit bootstrapping.

- Records: {len(records):,}
- Train/validation/test: {split_counts["train"]:,}/{split_counts["validation"]:,}/{split_counts["test"]:,}
- Languages: RU {language_counts["ru"]:,}, EN {language_counts["en"]:,}
- Categories: {len(category_counts)} × {args.examples_per_category}
- Verifier-backed template families: {len(template_counts)}
- License: CC0-1.0
- Generator seed: {args.seed}
- Generator SHA-256: `{manifest["generator_sha256"]}`
- Corpus SHA-256: `{manifest["corpus_sha256"]}`

This is **not** base pretraining data, not real or imitated Claude/Opus output, and contains no hidden chain-of-thought. It teaches compact verified behavior in arithmetic, algebra, logic, Python, calculator JSON, explicit memory, grounded QA, prompt-injection resistance, uncertainty and critique/revision.

All assistant targets are generated from deterministic state and carry machine-readable verification metadata. Code targets are executed against generated tests during construction. Use assistant-only loss and keep the protected test split out of training and teacher generation.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    (output / "LICENSE").write_text(
        "AIra Mentor v1 is dedicated to the public domain under CC0 1.0.\n"
        "https://creativecommons.org/publicdomain/zero/1.0/\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
