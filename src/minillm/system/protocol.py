"""Strict JSON action protocol between a language policy and deterministic runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias


@dataclass(frozen=True)
class ToolCall:
    tool: str
    arguments: dict[str, Any]
    type: Literal["tool_call"] = "tool_call"


@dataclass(frozen=True)
class MemoryProposal:
    subject: str
    predicate: str
    object: str
    confidence: float
    reason: str
    privacy_class: Literal["public", "private", "sensitive"] = "private"

    def __post_init__(self) -> None:
        if not all((self.subject.strip(), self.predicate.strip(), self.object.strip())):
            raise ValueError("memory proposal fields cannot be empty")
        if not 0 <= self.confidence <= 1:
            raise ValueError("memory proposal confidence must be in [0, 1]")


@dataclass(frozen=True)
class FinalAnswer:
    content: str
    confidence: float
    citations: tuple[str, ...] = ()
    memory_proposals: tuple[MemoryProposal, ...] = ()
    type: Literal["final"] = "final"

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("final answer cannot be empty")
        if not 0 <= self.confidence <= 1:
            raise ValueError("answer confidence must be in [0, 1]")


PolicyAction: TypeAlias = ToolCall | FinalAnswer


def parse_policy_action(payload: str | dict[str, Any]) -> PolicyAction:
    """Parse exactly one policy action and reject unknown protocol fields."""

    raw = json.loads(payload) if isinstance(payload, str) else dict(payload)
    action_type = raw.get("type")
    if action_type == "tool_call":
        allowed = {"type", "tool", "arguments"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown tool-call fields: {sorted(unknown)}")
        tool = raw.get("tool")
        arguments = raw.get("arguments")
        if not isinstance(tool, str) or not tool.strip():
            raise ValueError("tool must be a non-empty string")
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must be an object")
        return ToolCall(tool=tool, arguments=arguments)
    if action_type == "final":
        allowed = {"type", "content", "confidence", "citations", "memory_proposals"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown final-answer fields: {sorted(unknown)}")
        proposals = tuple(
            MemoryProposal(**item) for item in raw.get("memory_proposals", [])
        )
        citations = raw.get("citations", [])
        if not isinstance(citations, list) or not all(
            isinstance(item, str) for item in citations
        ):
            raise ValueError("citations must be a string array")
        return FinalAnswer(
            content=str(raw.get("content", "")),
            confidence=float(raw.get("confidence", 0.0)),
            citations=tuple(citations),
            memory_proposals=proposals,
        )
    raise ValueError("policy action type must be 'tool_call' or 'final'")


def action_to_json(action: PolicyAction) -> str:
    if isinstance(action, ToolCall):
        raw: dict[str, Any] = {
            "type": action.type,
            "tool": action.tool,
            "arguments": action.arguments,
        }
    else:
        raw = {
            "type": action.type,
            "content": action.content,
            "confidence": action.confidence,
            "citations": list(action.citations),
            "memory_proposals": [
                proposal.__dict__ for proposal in action.memory_proposals
            ],
        }
    return json.dumps(raw, ensure_ascii=False, separators=(",", ":"))
