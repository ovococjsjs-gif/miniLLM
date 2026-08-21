#!/usr/bin/env python3
"""Write the downloadable AIra One v0.1 code/config/evidence package manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PACKAGE_FILES = (
    "configs/aira-one/v01.json",
    "configs/donors/qwen35_08b.json",
    "docs/aira-one-v01.md",
    "src/minillm/aira/one.py",
    "src/minillm/aira/local_runtime.py",
    "scripts/run_aira_one.py",
    "scripts/serve_aira_one.py",
    "scripts/evaluate_aira_one.py",
    "scripts/smoke_aira_one.py",
    "scripts/build_aira_one_babysit_v01.py",
    "scripts/package_aira_one_v01.py",
    "artifacts/aira-one-babysit-v01/records.jsonl",
    "artifacts/aira-one-babysit-v01/records.jsonl.manifest.json",
    "results/aira_one_v01_evaluation.json",
    "results/aira_one_v01_fresh_evaluation.json",
    "results/aira_one_v01_prepatch_runtime_smoke.json",
    "results/aira_one_v01_runtime_smoke.json",
)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/aira-one-v01")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for item in PACKAGE_FILES:
        path = Path(item)
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append(
            {
                "path": item,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    donor = json.loads(Path("configs/donors/qwen35_08b.json").read_text())
    manifest = {
        "schema_version": 1,
        "name": "AIra One v0.1",
        "kind": "integrated local assistant code/config/evidence package",
        "files": files,
        "package_source_bytes": sum(item["bytes"] for item in files),
        "external_runtime_artifact": {
            "path": donor["github_mirror"]["local_path"],
            "bytes": donor["github_mirror"]["size_bytes"],
            "sha256": donor["github_mirror"]["sha256"],
            "included_in_git": False,
            "bootstrap": "python scripts/bootstrap_qwen35_donor.py --build-runtime --source github",
        },
        "entry_points": {
            "chat": "python scripts/run_aira_one.py",
            "api": "python scripts/serve_aira_one.py --host 0.0.0.0 --port 8000",
            "offline_exact": (
                "python scripts/run_aira_one.py --offline --prompt "
                "\"Вычисли: 12 * (7 + 3)\""
            ),
        },
        "experimental_recurrent_bypass_enabled": False,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# AIra One v0.1 package\n\n"
        "The first integrated local AIra assistant. The Git package contains source, "
        "configuration, Babysit patches and measured evidence; the verified 532.5 MB "
        "GGUF is restored separately by the pinned bootstrap process.\n\n"
        "Run `python scripts/run_aira_one.py` for chat or "
        "`python scripts/serve_aira_one.py` for the local API.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
