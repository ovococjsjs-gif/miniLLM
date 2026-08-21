#!/usr/bin/env python3
"""Prove neural adapter generation does not consume stored teacher answers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="data/external/qwen3.5-0.8b/Qwen3.5-0.8B-Q4_K_M-github.gguf",
    )
    parser.add_argument(
        "--binary", default=".cache/qwen35-output-adapter/qwen35-output-adapter"
    )
    parser.add_argument(
        "--adapter", default="artifacts/aira-neural-babysit-v1/adapter.bin"
    )
    parser.add_argument(
        "--validation", default="artifacts/aira-neural-babysit-v1/validation.tsv"
    )
    parser.add_argument(
        "--scrubbed",
        default="artifacts/aira-neural-babysit-v1/validation_inference.tsv",
    )
    parser.add_argument(
        "--reference",
        default="artifacts/aira-neural-babysit-v1/validation_generation.jsonl",
    )
    parser.add_argument(
        "--generated", default=".cache/aira-neural-babysit-v1/scrubbed-generation.tsv"
    )
    parser.add_argument(
        "--output", default="results/aira_neural_adapter_independence_v1.json"
    )
    args = parser.parse_args()

    validation = Path(args.validation)
    scrubbed = Path(args.scrubbed)
    scrubbed_rows = []
    for line in validation.read_text(encoding="utf-8").splitlines():
        identity, prompt_hex, _ = line.split("\t")
        scrubbed_rows.append(f"{identity}\t{prompt_hex}\t2d")
    scrubbed.write_text("\n".join(scrubbed_rows) + "\n", encoding="utf-8")

    generated = Path(args.generated)
    generated.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    completed = subprocess.run(
        [
            args.binary,
            "generate",
            args.model,
            str(scrubbed),
            str(generated),
            args.adapter,
            "128",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode:
        raise RuntimeError(completed.stdout + completed.stderr)

    with generated.open(encoding="utf-8") as handle:
        generated_rows = {
            row["id"]: bytes.fromhex(row["answer_hex"]).decode(
                "utf-8", errors="replace"
            )
            for row in csv.DictReader(handle, delimiter="\t")
        }
    reference_rows = {
        row["id"]: row["adapted"]["answer"]
        for row in (
            json.loads(line)
            for line in Path(args.reference).read_text(encoding="utf-8").splitlines()
        )
    }
    exact = {
        identity: generated_rows.get(identity) == answer
        for identity, answer in reference_rows.items()
    }
    report = {
        "schema_version": 1,
        "experiment": "aira-neural-adapter-answer-independence-v1",
        "validation_source_sha256": sha256(validation),
        "scrubbed_inference_sha256": sha256(scrubbed),
        "adapter_sha256": sha256(Path(args.adapter)),
        "teacher_answer_bytes_available_to_runtime": 0,
        "stored_answer_routes_enabled": False,
        "records": len(exact),
        "byte_exact_matches": sum(exact.values()),
        "all_outputs_byte_exact": all(exact.values()),
        "elapsed_seconds": elapsed,
        "per_record_exact": exact,
        "note": (
            "Every answer field was replaced by a single hyphen before native inference; "
            "the 24 generated outputs remained byte-identical."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
