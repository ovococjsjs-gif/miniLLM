#!/usr/bin/env python3
"""Build the first failure-clustered AIra teacher packet and executable curriculum."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from minillm.aira import read_babysit_dataset
from minillm.aira.foundry import (
    build_teacher_packet,
    compile_curriculum,
    mentor_skill_patches_v1,
    write_curriculum_dataset,
    write_skill_patches,
    write_teacher_packet,
)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--babysit",
        default="artifacts/aira-mentor-babysit-v1/records.jsonl",
    )
    parser.add_argument(
        "--output-dir", default="artifacts/aira-teacher-foundry-v1"
    )
    parser.add_argument("--generated-examples-per-category", type=int, default=100)
    parser.add_argument("--generated-seed", type=int, default=44)
    parser.add_argument("--max-representatives", type=int, default=3)
    args = parser.parse_args()

    source_path = Path(args.babysit)
    output = Path(args.output_dir)
    records = read_babysit_dataset(source_path)
    packet = build_teacher_packet(
        records,
        packet_id="aira-teacher-packet-v1",
        max_representatives=args.max_representatives,
    )
    patches = mentor_skill_patches_v1(packet.content_sha256)
    curriculum = compile_curriculum(
        patches,
        generated_examples_per_category=args.generated_examples_per_category,
        generated_seed=args.generated_seed,
        babysit_records=records,
    )

    packet_path = write_teacher_packet(output / "teacher-packet.json", packet)
    patches_path = write_skill_patches(output / "skill-patches.json", patches)
    curriculum_path = output / "curriculum.jsonl"
    manifest_path = write_curriculum_dataset(
        curriculum_path,
        curriculum,
        metadata={
            "source_babysit": str(source_path),
            "source_babysit_sha256": sha256(source_path),
            "teacher_packet_sha256": packet.content_sha256,
            "generated_seed": args.generated_seed,
            "generated_examples_per_category": args.generated_examples_per_category,
            "protected_aira_mentor_v1_splits_used": False,
            "public_weight_use_status": (
                "deterministic records are project-owned; agent-authored patch text "
                "requires Arena terms review before public weight release"
            ),
        },
    )
    patch_counts = Counter(record.patch_id for record in curriculum)
    report = {
        "schema_version": 1,
        "source_records": len(records),
        "failed_records": packet.failed_records,
        "passed_records": packet.passed_records,
        "failure_clusters": len(packet.clusters),
        "cluster_counts": {
            f"{cluster.category}:{cluster.failure_mode}": cluster.count
            for cluster in packet.clusters
        },
        "skill_patches": len(patches),
        "curriculum_records": len(curriculum),
        "curriculum_patch_counts": dict(sorted(patch_counts.items())),
        "teacher_packet": str(packet_path),
        "skill_patches_path": str(patches_path),
        "curriculum": str(curriculum_path),
        "manifest": str(manifest_path),
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
