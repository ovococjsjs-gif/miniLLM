"""Managed pinned llama.cpp process for AIra One."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Self


class LocalDonorRuntime:
    def __init__(
        self,
        *,
        model: str | Path,
        llama_server: str | Path = ".cache/llama.cpp/build/bin/llama-server",
        config: str | Path = "configs/donors/qwen35_08b.json",
        host: str = "127.0.0.1",
        port: int = 8081,
        log_path: str | Path = ".aira-one/llama-server.log",
    ) -> None:
        self.model = Path(model)
        self.llama_server = Path(llama_server)
        self.config = Path(config)
        self.host = host
        self.port = port
        self.log_path = Path(log_path)
        self.process: subprocess.Popen[bytes] | None = None
        self._log_handle = None

    @property
    def endpoint(self) -> str:
        return f"http://{self.host}:{self.port}"

    @staticmethod
    def _sha256(path: Path) -> str:
        with path.open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()

    def verify(self) -> None:
        config = json.loads(self.config.read_text(encoding="utf-8"))
        mirror = config["github_mirror"]
        if not self.model.is_file():
            raise FileNotFoundError(self.model)
        if self.model.stat().st_size != mirror["size_bytes"]:
            raise ValueError("AIra One donor model has the wrong size")
        if self._sha256(self.model) != mirror["sha256"]:
            raise ValueError("AIra One donor model has the wrong hash")
        if not self.llama_server.is_file():
            raise FileNotFoundError(self.llama_server)

    def healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.endpoint}/health", timeout=1) as response:
                return response.status == 200
        except (OSError, urllib.error.URLError):
            return False

    def start(self) -> None:
        self.verify()
        if self.healthy():
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("ab")
        command = [
            str(self.llama_server),
            "--model",
            str(self.model),
            "--alias",
            "aira-one-donor",
            "--ctx-size",
            "4096",
            "--threads",
            "2",
            "--threads-batch",
            "2",
            "--batch-size",
            "128",
            "--ubatch-size",
            "128",
            "--parallel",
            "1",
            "--cache-ram",
            "0",
            "--reasoning",
            "off",
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]
        self.process = subprocess.Popen(
            command,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
        )
        started = time.monotonic()
        while time.monotonic() - started < 120:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited with {self.process.returncode}; see {self.log_path}"
                )
            if self.healthy():
                return
            time.sleep(0.25)
        self.stop()
        raise RuntimeError(f"llama-server did not become healthy; see {self.log_path}")

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
