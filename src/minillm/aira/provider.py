"""Minimal OpenAI-compatible provider adapter for local donor runtimes."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProviderResponse:
    content: str
    reasoning_content: str
    finish_reason: str
    prompt_tokens: int | None
    completion_tokens: int | None
    raw_model: str


class ProviderError(RuntimeError):
    pass


class OpenAIChatProvider:
    """Call a local or remote OpenAI-compatible chat-completions endpoint.

    API keys are read from an environment variable at call time and are never
    serialized into configs, requests saved by this project, or result records.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float = 120,
        api_key_environment: str | None = None,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("provider base URL must be HTTP(S)")
        if not model:
            raise ValueError("provider model cannot be empty")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.api_key_environment = api_key_environment

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float = 0,
        max_tokens: int = 256,
        extra_body: Mapping[str, Any] | None = None,
    ) -> ProviderResponse:
        if not messages or any(
            message.get("role") not in {"system", "user", "assistant", "tool"}
            or not message.get("content")
            for message in messages
        ):
            raise ValueError("provider messages need valid roles and non-empty content")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        payload.update(extra_body or {})
        headers = {"Content-Type": "application/json"}
        if self.api_key_environment:
            key = os.environ.get(self.api_key_environment)
            if not key:
                raise ProviderError(
                    f"provider credential environment {self.api_key_environment} is unset"
                )
            headers["Authorization"] = f"Bearer {key}"
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                decoded = json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read(2048).decode("utf-8", errors="replace")
            raise ProviderError(
                f"provider returned HTTP {error.code}: {detail}"
            ) from error
        except (TimeoutError, urllib.error.URLError) as error:
            raise ProviderError(f"provider request failed: {error}") from error
        try:
            choice = decoded["choices"][0]
            message = choice["message"]
            usage = decoded.get("usage", {})
            return ProviderResponse(
                content=str(message.get("content") or ""),
                reasoning_content=str(message.get("reasoning_content") or ""),
                finish_reason=str(choice.get("finish_reason") or ""),
                prompt_tokens=(
                    int(usage["prompt_tokens"])
                    if usage.get("prompt_tokens") is not None
                    else None
                ),
                completion_tokens=(
                    int(usage["completion_tokens"])
                    if usage.get("completion_tokens") is not None
                    else None
                ),
                raw_model=str(decoded.get("model") or self.model),
            )
        except (IndexError, KeyError, TypeError, ValueError) as error:
            raise ProviderError("provider response does not match chat schema") from error
