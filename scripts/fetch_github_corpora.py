#!/usr/bin/env python3
"""Fetch corpus repositories at exact commits without following moving branches."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def run(*arguments: str, cwd: Path | None = None) -> None:
    subprocess.run(arguments, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", default="configs/corpus/github_pilot_sources.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.sources).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported GitHub source schema")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "fetch-manifest.json"
    if manifest_path.exists():
        raise FileExistsError("GitHub corpus checkout already has a manifest")

    fetched = []
    for source in config["sources"]:
        checkout = output / source["checkout"]
        if checkout.exists():
            raise FileExistsError(f"refusing to overwrite checkout {checkout}")
        checkout.mkdir()
        run("git", "init", "--quiet", cwd=checkout)
        run("git", "remote", "add", "origin", source["repository"], cwd=checkout)
        run(
            "git",
            "fetch",
            "--quiet",
            "--depth",
            "1",
            "origin",
            source["revision"],
            cwd=checkout,
        )
        run("git", "checkout", "--quiet", "--detach", "FETCH_HEAD", cwd=checkout)
        actual = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
        ).strip()
        if actual != source["revision"]:
            raise RuntimeError(f"revision mismatch for {source['source_id']}")
        fetched.append(
            {
                "source_id": source["source_id"],
                "repository": source["repository"],
                "revision": actual,
                "checkout": source["checkout"],
            }
        )

    manifest = {
        "schema_version": 1,
        "source_config": config["id"],
        "sources": fetched,
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
