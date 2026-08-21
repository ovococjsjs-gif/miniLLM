#!/usr/bin/env python3
"""Run the pinned real-state probe and atomically publish its compact audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from audit_qwen35_state_probe import audit


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/donors/qwen35_08b.json")
    parser.add_argument(
        "--model",
        default=(
            "data/external/qwen3.5-0.8b/"
            "Qwen3.5-0.8B-Q4_K_M-github.gguf"
        ),
    )
    parser.add_argument(
        "--binary", default=".cache/qwen35-state-probe/qwen35-state-probe"
    )
    parser.add_argument(
        "--build-report", default="results/qwen35_state_probe_build.json"
    )
    parser.add_argument("--raw-dir", default="data/qwen35-state-probe-smoke")
    parser.add_argument(
        "--prompt", default="User: Calculate 17 + 8 - 6. Assistant:"
    )
    parser.add_argument("--continuation-tokens", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--output", default="results/qwen35_08b_real_state_probe.json"
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = Path(args.model)
    if model.stat().st_size != config["github_mirror"]["size_bytes"]:
        raise ValueError("state probe model size differs from pinned donor")
    if sha256(model) != config["github_mirror"]["sha256"]:
        raise ValueError("state probe model hash differs from pinned donor")
    binary = Path(args.binary)
    build_report_path = Path(args.build_report)
    build_report = json.loads(build_report_path.read_text(encoding="utf-8"))
    if sha256(binary) != build_report["binary_sha256"]:
        raise ValueError("probe binary differs from its build report")

    raw_dir = Path(args.raw_dir)
    temporary = raw_dir.with_name(raw_dir.name + ".tmp")
    if raw_dir.exists() and not args.overwrite:
        raise FileExistsError(f"raw output exists; pass --overwrite: {raw_dir}")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(binary),
        str(model),
        str(temporary),
        args.prompt,
        str(args.continuation_tokens),
    ]
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    temporary.mkdir(parents=True, exist_ok=True)
    (temporary / "probe.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (temporary / "probe.stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(
            f"state probe failed with {completed.returncode}; logs are in {temporary}"
        )
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    temporary.replace(raw_dir)

    report = audit(
        raw_dir,
        model_path=model,
        config_path=config_path,
        build_report_path=build_report_path,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "raw_directory": str(raw_dir),
                "tensor_captures": report["captures"]["total_tensor_captures"],
                "raw_bytes": report["captures"]["total_raw_bytes"],
                "cache_continuity": report["cache_continuity"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
