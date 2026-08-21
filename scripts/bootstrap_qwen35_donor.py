#!/usr/bin/env python3
"""Build, download, or verify the pinned Qwen3.5-0.8B donor runtime."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import shutil
import subprocess
import urllib.parse
from pathlib import Path
from typing import Any

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_LFS_OID = re.compile(r"^oid sha256:([0-9a-f]{64})$", flags=re.MULTILINE)
_LFS_SIZE = re.compile(r"^size (\d+)$", flags=re.MULTILINE)


def clean_error(value: str, *, limit: int) -> str:
    return _ANSI_ESCAPE.sub("", value)[-limit:]


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify(path: Path, expected_size: int, expected_hash: str) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        return {
            "status": "invalid-size",
            "path": str(path),
            "expected_size": expected_size,
            "actual_size": actual_size,
        }
    actual_hash = sha256(path)
    return {
        "status": "verified" if actual_hash == expected_hash else "invalid-hash",
        "path": str(path),
        "size_bytes": actual_size,
        "expected_sha256": expected_hash,
        "actual_sha256": actual_hash,
    }


def llama_cli_path() -> Path | None:
    candidates = (
        Path(".cache/llama.cpp/build/bin/llama-cli"),
        Path(".cache/llama-b9222/build/bin/llama-cli"),
        Path(shutil.which("llama-cli") or "/missing"),
    )
    return next((path for path in candidates if path.exists()), None)


def runtime_status(config: dict[str, Any]) -> dict[str, Any]:
    llama_cli = llama_cli_path()
    if llama_cli is None:
        return {
            "status": "missing",
            "expected_revision": config["runtime"]["revision"],
        }
    completed = subprocess.run(
        [str(llama_cli), "--version"],
        check=False,
        text=True,
        capture_output=True,
        timeout=30,
    )
    version_text = (completed.stdout + completed.stderr).strip()
    return {
        "status": (
            "verified"
            if config["runtime"]["revision"][:7] in version_text
            else "revision-unconfirmed"
        ),
        "path": str(llama_cli),
        "version": version_text,
    }


def _tool(name: str) -> str:
    local = Path(".venv/bin") / name
    command = str(local) if local.exists() else shutil.which(name)
    if not command:
        raise FileNotFoundError(
            f"{name} is unavailable; install the optional runtime dependencies"
        )
    return command


def build_runtime(config: dict[str, Any]) -> dict[str, Any]:
    source = Path(".cache/llama.cpp")
    revision = str(config["runtime"]["revision"])
    try:
        if not (source / ".git").exists():
            source.parent.mkdir(parents=True, exist_ok=True)
            completed = subprocess.run(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "--no-checkout",
                    "https://github.com/ggml-org/llama.cpp.git",
                    str(source),
                ],
                check=False,
                text=True,
                capture_output=True,
                timeout=300,
            )
            if completed.returncode:
                raise RuntimeError(completed.stderr)
        completed = subprocess.run(
            ["git", "-C", str(source), "fetch", "--depth", "1", "origin", revision],
            check=False,
            text=True,
            capture_output=True,
            timeout=300,
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr)
        subprocess.run(
            ["git", "-C", str(source), "checkout", "--detach", "FETCH_HEAD"],
            check=True,
            text=True,
            capture_output=True,
            timeout=60,
        )
        cmake = _tool("cmake")
        ninja = _tool("ninja")
        build = source / "build"
        configure = subprocess.run(
            [
                cmake,
                "-S",
                str(source),
                "-B",
                str(build),
                "-G",
                "Ninja",
                f"-DCMAKE_MAKE_PROGRAM={Path(ninja).resolve()}",
                "-DGGML_NATIVE=OFF",
                "-DGGML_OPENMP=OFF",
                "-DLLAMA_CURL=OFF",
                "-DLLAMA_BUILD_TESTS=OFF",
                "-DLLAMA_BUILD_EXAMPLES=ON",
                "-DLLAMA_BUILD_SERVER=ON",
                "-DCMAKE_BUILD_TYPE=Release",
            ],
            check=False,
            text=True,
            capture_output=True,
            timeout=300,
        )
        if configure.returncode:
            raise RuntimeError(configure.stdout + configure.stderr)
        compile_run = subprocess.run(
            [
                cmake,
                "--build",
                str(build),
                "--target",
                "llama-cli",
                "llama-server",
                "-j",
                "2",
            ],
            check=False,
            text=True,
            capture_output=True,
            timeout=1800,
        )
        if compile_run.returncode:
            raise RuntimeError(compile_run.stdout + compile_run.stderr)
        return {"status": "completed", "revision": revision}
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        return {
            "status": "failed",
            "type": type(error).__name__,
            "detail": clean_error(str(error), limit=1000),
        }


def download_huggingface(artifact: dict[str, Any], hf_command: str) -> dict[str, Any] | None:
    destination = Path(artifact["local_path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(
            [
                hf_command,
                "download",
                artifact["repository"],
                artifact["filename"],
                "--revision",
                artifact["revision"],
                "--local-dir",
                str(destination.parent),
            ],
            check=False,
            text=True,
            capture_output=True,
            timeout=1800,
        )
        if completed.returncode:
            return {
                "type": "hf-cli-failure",
                "returncode": completed.returncode,
                "detail": clean_error(completed.stderr, limit=1000),
            }
    except (OSError, subprocess.SubprocessError) as error:
        return {
            "type": type(error).__name__,
            "detail": clean_error(str(error), limit=500),
        }
    return None


def _github_lfs_pointer(mirror: dict[str, Any]) -> str:
    endpoint = (
        f"repos/{mirror['repository']}/contents/{mirror['path']}"
        f"?ref={mirror['revision']}"
    )
    completed = subprocess.run(
        ["gh", "api", endpoint, "--jq", ".content"],
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )
    return base64.b64decode(completed.stdout).decode("ascii")


def download_github_lfs(mirror: dict[str, Any]) -> dict[str, Any] | None:
    """Resolve a pinned public Git LFS pointer through GitHub, then fetch its object."""

    destination = Path(mirror["local_path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        pointer = _github_lfs_pointer(mirror)
        oid_match = _LFS_OID.search(pointer)
        size_match = _LFS_SIZE.search(pointer)
        if oid_match is None or size_match is None:
            raise ValueError("GitHub path is not a Git LFS pointer")
        oid = oid_match.group(1)
        size = int(size_match.group(1))
        if oid != mirror["git_lfs_oid_sha256"] or size != mirror["size_bytes"]:
            raise ValueError("Git LFS pointer differs from the pinned mirror metadata")

        auth = subprocess.run(
            ["gh", "auth", "token"],
            check=True,
            text=True,
            capture_output=True,
            timeout=30,
        )
        token = auth.stdout.strip()
        body = json.dumps(
            {
                "operation": "download",
                "transfers": ["basic"],
                "objects": [{"oid": oid, "size": size}],
            }
        )
        batch_endpoint = (
            f"https://github.com/{mirror['repository']}.git/info/lfs/objects/batch"
        )
        batch = subprocess.run(
            [
                "curl",
                "-fsS",
                "--http1.1",
                "-u",
                f"x-access-token:{token}",
                "-H",
                "Accept: application/vnd.git-lfs+json",
                "-H",
                "Content-Type: application/vnd.git-lfs+json",
                "--data-binary",
                "@-",
                batch_endpoint,
            ],
            input=body,
            check=True,
            text=True,
            capture_output=True,
            timeout=60,
        )
        response = json.loads(batch.stdout)
        href = response["objects"][0]["actions"]["download"]["href"]
        storage_host = urllib.parse.urlsplit(href).hostname
        partial = destination.with_suffix(destination.suffix + ".part")
        transfer = subprocess.run(
            [
                "curl",
                "-4",
                "-fL",
                "--http1.1",
                "--retry",
                "5",
                "--retry-delay",
                "2",
                "--continue-at",
                "-",
                "-o",
                str(partial),
                href,
            ],
            check=False,
            text=True,
            capture_output=True,
            timeout=1800,
        )
        if transfer.returncode:
            return {
                "type": "github-lfs-transfer-failure",
                "returncode": transfer.returncode,
                "storage_host": storage_host,
                "detail": clean_error(transfer.stderr, limit=1000),
            }
        partial.replace(destination)
    except (OSError, KeyError, ValueError, json.JSONDecodeError, subprocess.SubprocessError) as error:
        return {
            "type": type(error).__name__,
            "detail": clean_error(str(error), limit=500),
        }
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/donors/qwen35_08b.json")
    parser.add_argument("--build-runtime", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--source", choices=("huggingface", "github"), default="huggingface"
    )
    parser.add_argument("--hf-command", default=".venv/bin/hf")
    parser.add_argument("--output", default="results/qwen35_donor_bootstrap.json")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    primary = config["gguf"]
    artifact = primary if args.source == "huggingface" else config["github_mirror"]
    destination = Path(artifact["local_path"])
    report: dict[str, Any] = {
        "schema_version": 1,
        "config": str(config_path),
        "model_id": config["upstream"]["model_id"],
        "model_revision": config["upstream"]["revision"],
        "selected_source": args.source,
        "artifact_source": (
            {
                "repository": artifact["repository"],
                "revision": artifact["revision"],
                "filename": artifact["filename"],
            }
            if args.source == "huggingface"
            else {
                "repository": artifact["repository"],
                "revision": artifact["revision"],
                "path": artifact["path"],
                "declared_upstream": artifact["declared_upstream"],
                "review_status": artifact["review_status"],
            }
        ),
        "runtime_build_attempted": False,
        "download_attempted": False,
    }

    runtime = runtime_status(config)
    if runtime["status"] == "missing" and args.build_runtime:
        report["runtime_build_attempted"] = True
        report["runtime_build"] = build_runtime(config)
        runtime = runtime_status(config)
    report["runtime"] = runtime

    checked = verify(destination, artifact["size_bytes"], artifact["sha256"])
    if checked["status"] == "missing" and args.download:
        report["download_attempted"] = True
        if args.source == "huggingface":
            error = download_huggingface(artifact, args.hf_command)
        else:
            error = download_github_lfs(artifact)
        if error is not None:
            report["download_error"] = error
        checked = verify(destination, artifact["size_bytes"], artifact["sha256"])
    report["artifact"] = checked

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if checked["status"] != "verified":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
