#!/usr/bin/env python3
"""Rebuild and verify the pinned GitHub L1 token stream for local or Kaggle use."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results" / "github_pilot_data.json"
TOKENIZER = ROOT / "artifacts" / "tokenizer-github-pilot-v1" / "tokenizer.json"
TOKENIZER_MANIFEST = TOKENIZER.with_name("manifest.json")


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def expected_identities() -> dict[str, Any]:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    return {
        "corpus_sha256": report["corpus"]["corpus_sha256"],
        "tokenizer_sha256": report["selected_tokenizer"]["tokenizer_sha256"],
        "splits": report["packed_tokens"]["splits"],
    }


def verify_token_streams(tokens: Path) -> dict[str, Any]:
    """Fully hash packed streams and compare them with committed pilot identities."""

    expected = expected_identities()
    tokenizer_digest = file_sha256(TOKENIZER)
    if tokenizer_digest != expected["tokenizer_sha256"]:
        raise ValueError("repository tokenizer does not match the committed L1 report")
    packed_manifest = json.loads((tokens / "manifest.json").read_text(encoding="utf-8"))
    if packed_manifest["corpus_sha256"] != expected["corpus_sha256"]:
        raise ValueError("packed stream belongs to a different corpus")
    if packed_manifest["tokenizer_sha256"] != tokenizer_digest:
        raise ValueError("packed stream belongs to a different tokenizer")

    verified_splits: dict[str, Any] = {}
    for split, identity in expected["splits"].items():
        path = tokens / f"{split}.bin"
        sidecar = json.loads(
            path.with_suffix(path.suffix + ".json").read_text(encoding="utf-8")
        )
        digest = file_sha256(path)
        size = path.stat().st_size
        if sidecar["dtype"] != "uint32" or identity["dtype"] != "uint32":
            raise ValueError(f"unexpected dtype for {split}")
        if size != int(identity["tokens"]) * 4:
            raise ValueError(f"unexpected packed byte count for {split}")
        if int(sidecar["tokens"]) != int(identity["tokens"]):
            raise ValueError(f"unexpected token count for {split}")
        if digest != identity["sha256"] or digest != sidecar["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {split}")
        verified_splits[split] = {
            "path": str(path),
            "tokens": int(identity["tokens"]),
            "sha256": digest,
            "size_bytes": size,
        }
    return {
        "corpus_sha256": expected["corpus_sha256"],
        "tokenizer_sha256": tokenizer_digest,
        "splits": verified_splits,
    }


def run_script(script: str, *arguments: str) -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *arguments],
        cwd=ROOT,
        check=True,
    )


def require_absent_or_complete(path: Path, marker: Path, stage: str) -> bool:
    if marker.exists():
        return True
    if path.exists():
        raise RuntimeError(
            f"partial {stage} output exists at {path}; inspect it, then remove that "
            "directory before retrying"
        )
    return False


def prepare(root: Path, *, acquisition_date: str, cleanup: bool) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    checkouts = root / "checkouts"
    imported_directory = root / "imported"
    imported = imported_directory / "github-pilot.jsonl"
    corpus = root / "corpus"
    tokens = root / "tokens"

    if (tokens / "manifest.json").exists():
        verified = verify_token_streams(tokens)
    else:
        require_absent_or_complete(
            checkouts,
            checkouts / "fetch-manifest.json",
            "GitHub checkout",
        ) or run_script(
            "fetch_github_corpora.py",
            "--output",
            str(checkouts),
        )

        if not imported.exists():
            if imported_directory.exists() and any(imported_directory.iterdir()):
                raise RuntimeError(
                    f"partial import output exists at {imported_directory}"
                )
            run_script(
                "import_github_corpora.py",
                "--checkouts",
                str(checkouts),
                "--output",
                str(imported),
                "--acquisition-date",
                acquisition_date,
            )

        if not require_absent_or_complete(
            corpus,
            corpus / "manifest.json",
            "corpus build",
        ):
            run_script(
                "build_corpus_shards.py",
                str(imported),
                "--output",
                str(corpus),
                "--registry",
                "configs/corpus/source_registry.json",
                "--policy",
                "configs/corpus/policy_production.json",
                "--protected",
                "eval/bilingual_smoke.json",
                "--shard-mib",
                "16",
            )
        corpus_manifest = json.loads(
            (corpus / "manifest.json").read_text(encoding="utf-8")
        )
        expected_corpus = expected_identities()["corpus_sha256"]
        if corpus_manifest["corpus_sha256"] != expected_corpus:
            raise ValueError(
                "rebuilt corpus hash differs from the committed pilot identity"
            )

        if tokens.exists():
            raise RuntimeError(f"partial token output exists at {tokens}")
        run_script(
            "pack_corpus_tokens.py",
            str(corpus),
            "--tokenizer",
            str(TOKENIZER),
            "--tokenizer-manifest",
            str(TOKENIZER_MANIFEST),
            "--output",
            str(tokens),
        )
        verified = verify_token_streams(tokens)

    if cleanup:
        for intermediate in (checkouts, imported_directory, corpus):
            if intermediate.exists():
                shutil.rmtree(intermediate)

    ready = {
        "schema_version": 1,
        "status": "ready",
        "tokens": str(tokens),
        "acquisition_date": acquisition_date,
        "intermediates_removed": cleanup,
        **verified,
    }
    (root / "l1-data-ready.json").write_text(
        json.dumps(ready, indent=2) + "\n", encoding="utf-8"
    )
    return ready


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--acquisition-date", default="2026-08-20")
    parser.add_argument(
        "--cleanup-intermediates",
        action="store_true",
        help="remove cloned source repositories, imported text, and corpus shards after verification",
    )
    args = parser.parse_args()
    result = prepare(
        Path(args.root).resolve(),
        acquisition_date=args.acquisition_date,
        cleanup=args.cleanup_intermediates,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
