#!/usr/bin/env python3
"""Build the public-API llama.cpp Qwen3.5 recurrent-state probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def concrete_library(directory: Path, stem: str) -> Path:
    candidates = [
        path
        for path in directory.glob(f"{stem}.so.*")
        if path.is_file() and not path.is_symlink()
    ]
    if not candidates:
        raise FileNotFoundError(f"cannot find concrete {stem} shared library in {directory}")
    return max(candidates, key=lambda path: len(path.name))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/donors/qwen35_08b.json")
    parser.add_argument("--llama-source", default=".cache/llama.cpp")
    parser.add_argument("--source", default="native/qwen35_state_probe.cpp")
    parser.add_argument(
        "--binary", default=".cache/qwen35-state-probe/qwen35-state-probe"
    )
    parser.add_argument("--output", default="results/qwen35_state_probe_build.json")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    llama_source = Path(args.llama_source).resolve()
    revision = subprocess.run(
        ["git", "-C", str(llama_source), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    expected_revision = config["runtime"]["revision"]
    if revision != expected_revision:
        raise ValueError(
            f"llama.cpp revision mismatch: expected {expected_revision}, got {revision}"
        )

    source = Path(args.source).resolve()
    binary = Path(args.binary).resolve()
    binary.parent.mkdir(parents=True, exist_ok=True)
    library_dir = llama_source / "build" / "bin"
    compiler = shutil.which("c++")
    if not compiler:
        raise FileNotFoundError("a C++17 compiler is required")
    libraries = [
        concrete_library(library_dir, stem)
        for stem in ("libllama", "libggml", "libggml-base", "libggml-cpu")
    ]
    command = [
        compiler,
        "-std=c++17",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        str(source),
        f"-I{llama_source / 'include'}",
        f"-I{llama_source / 'ggml' / 'include'}",
        *(str(path) for path in libraries),
        f"-Wl,-rpath,{library_dir}",
        "-pthread",
        "-o",
        str(binary),
    ]
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(completed.stdout + completed.stderr)

    report = {
        "schema_version": 1,
        "source": str(Path(args.source)),
        "source_sha256": sha256(source),
        "binary": str(Path(args.binary)),
        "binary_sha256": sha256(binary),
        "llama_cpp_revision": revision,
        "compiler": subprocess.run(
            [compiler, "--version"], check=True, text=True, capture_output=True
        ).stdout.splitlines()[0],
        "libraries": [path.name for path in libraries],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
