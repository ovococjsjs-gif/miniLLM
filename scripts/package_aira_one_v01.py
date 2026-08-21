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
    "configs/aira-one/broad_curriculum_v1.json",
    "configs/aira-one/broad_manual_review_v1.json",
    "scripts/run_aira_one_broad_babysit.py",
    "scripts/apply_aira_one_broad_review.py",
    "artifacts/aira-one-babysit-v01/records.jsonl",
    "artifacts/aira-one-babysit-v01/records.jsonl.manifest.json",
    "results/aira_one_v01_evaluation.json",
    "results/aira_one_v01_fresh_evaluation.json",
    "results/aira_one_v01_prepatch_runtime_smoke.json",
    "results/aira_one_v01_runtime_smoke.json",
    "artifacts/aira-one-broad-babysit-v1/README.md",
    "artifacts/aira-one-broad-babysit-v1/records.jsonl",
    "artifacts/aira-one-broad-babysit-v1/records.jsonl.manifest.json",
    "artifacts/aira-one-broad-babysit-v1/records_audited.jsonl",
    "artifacts/aira-one-broad-babysit-v1/records_audited.jsonl.manifest.json",
    "artifacts/aira-one-broad-babysit-v1/skills_pre_review.json",
    "artifacts/aira-one-broad-babysit-v1/skills.json",
    "artifacts/aira-one-broad-babysit-v1/report.json",
    "artifacts/aira-one-broad-babysit-v1/audited_report.json",
    "results/aira_one_broad_babysit_v1.json",
    "results/aira_one_broad_babysit_v1_audited.json",
    "configs/aira-one/neural_babysit_v1.json",
    "configs/aira-one/neural_adapter_controls_v1.json",
    "configs/aira-one/neural_babysit_manual_review_v1.json",
    "docs/aira-neural-babysit-v1.md",
    "native/qwen35_output_adapter.cpp",
    "scripts/build_qwen35_output_adapter.py",
    "scripts/run_aira_neural_babysit.py",
    "scripts/audit_aira_neural_babysit.py",
    "scripts/verify_aira_neural_adapter_independence.py",
    "results/qwen35_output_adapter_build.json",
    "results/aira_neural_babysit_v1.json",
    "results/aira_neural_babysit_v1_audited.json",
    "results/aira_neural_adapter_independence_v1.json",
    "artifacts/aira-neural-babysit-v1/README.md",
    "artifacts/aira-neural-babysit-v1/adapter.bin",
    "artifacts/aira-neural-babysit-v1/audited_report.json",
    "artifacts/aira-neural-babysit-v1/control_generation.jsonl",
    "artifacts/aira-neural-babysit-v1/controls.tsv",
    "artifacts/aira-neural-babysit-v1/controls_validation.tsv",
    "artifacts/aira-neural-babysit-v1/manifest.json",
    "artifacts/aira-neural-babysit-v1/model.pt",
    "artifacts/aira-neural-babysit-v1/train.tsv",
    "artifacts/aira-neural-babysit-v1/validation.tsv",
    "artifacts/aira-neural-babysit-v1/validation_generation.jsonl",
    "artifacts/aira-neural-babysit-v1/validation_inference.tsv",
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
            "broad_babysit": "python scripts/run_aira_one_broad_babysit.py",
            "broad_manual_review": "python scripts/apply_aira_one_broad_review.py",
            "neural_babysit": "python scripts/run_aira_neural_babysit.py",
            "neural_babysit_audit": "python scripts/audit_aira_neural_babysit.py",
            "offline_exact": (
                "python scripts/run_aira_one.py --offline --prompt "
                '"Вычисли: 12 * (7 + 3)"'
            ),
        },
        "experimental_recurrent_bypass_enabled": False,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# AIra One v0.1 package\n\n"
        "The first integrated local AIra assistant. The Git package contains source, "
        "configuration and measured evidence. Reviewed SkillShelf entries are packaged "
        "as deterministic cache, not learning. Neural Babysit v1 adds a real 264K-parameter "
        "logit residual experiment; its production gate remains closed. The verified "
        "532.5 MB GGUF is restored separately by the pinned bootstrap process.\n\n"
        "Run `python scripts/run_aira_one.py` for chat, "
        "`python scripts/serve_aira_one.py` for the local API, or "
        "`python scripts/run_aira_neural_babysit.py` for parameter learning.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
