"""Permissioned deterministic tools and strict argument validation."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from enum import StrEnum
from typing import Any

from minillm.memory import EpisodicMemoryStore

from .calculator import safe_calculate
from .documents import DocumentStore


class Permission(StrEnum):
    COMPUTE = "compute"
    READ_MEMORY = "read_memory"
    READ_DOCUMENTS = "read_documents"
    WRITE_MEMORY = "write_memory"
    NETWORK = "network"


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    tool: str
    value: Any = None
    error: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, object]
    permission: Permission
    handler: Callable[[dict[str, Any]], Any]

    def public_schema(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "permission": self.permission.value,
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"duplicate tool: {spec.name}")
        self._tools[spec.name] = spec

    def schemas(
        self, allowed_permissions: Iterable[Permission]
    ) -> tuple[dict[str, object], ...]:
        allowed = set(allowed_permissions)
        return tuple(
            spec.public_schema()
            for spec in self._tools.values()
            if spec.permission in allowed
        )

    @staticmethod
    def _validate(
        arguments: Mapping[str, Any], schema: Mapping[str, object]
    ) -> dict[str, Any]:
        if schema.get("type") != "object":
            raise ValueError("tool schema root must be object")
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        if not isinstance(properties, dict):
            raise TypeError("invalid tool schema")
        unknown = set(arguments) - set(properties)
        if unknown and schema.get("additionalProperties", False) is False:
            raise ValueError(f"unknown arguments: {sorted(unknown)}")
        missing = required - set(arguments)
        if missing:
            raise ValueError(f"missing arguments: {sorted(missing)}")
        expected_types = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        validated = dict(arguments)
        for key, value in arguments.items():
            rule = properties.get(key, {})
            if not isinstance(rule, dict):
                continue
            expected = rule.get("type")
            if expected in expected_types:
                python_type = expected_types[expected]
                if not isinstance(value, python_type) or (
                    expected in {"integer", "number"} and isinstance(value, bool)
                ):
                    raise ValueError(f"argument {key!r} must be {expected}")
            if "enum" in rule and value not in rule["enum"]:
                raise ValueError(f"argument {key!r} must be one of {rule['enum']}")
            if isinstance(value, str) and len(value) > int(
                rule.get("maxLength", 100_000)
            ):
                raise ValueError(f"argument {key!r} is too long")
        return validated

    def execute(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        allowed_permissions: Iterable[Permission],
    ) -> ToolResult:
        spec = self._tools.get(name)
        if spec is None:
            return ToolResult(False, name, error="unknown tool")
        if spec.permission not in set(allowed_permissions):
            return ToolResult(
                False, name, error=f"permission denied: {spec.permission.value}"
            )
        try:
            validated = self._validate(arguments, spec.parameters)
            return ToolResult(True, name, value=spec.handler(validated))
        except (ValueError, KeyError, TypeError, OverflowError) as error:
            return ToolResult(False, name, error=str(error))


def _calendar(arguments: dict[str, Any]) -> dict[str, object]:
    operation = arguments["operation"]
    first = date.fromisoformat(arguments["date"])
    if operation == "weekday":
        return {"date": first.isoformat(), "weekday": first.strftime("%A")}
    if operation == "add_days":
        result = first + timedelta(days=arguments["days"])
        return {"date": result.isoformat(), "weekday": result.strftime("%A")}
    if operation == "difference":
        second = date.fromisoformat(arguments["other_date"])
        return {"days": (second - first).days}
    raise ValueError("unsupported calendar operation")


def build_default_registry(
    *,
    memory: EpisodicMemoryStore | None = None,
    documents: DocumentStore | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="calculator",
            description="Evaluate an arithmetic expression exactly. No variables or functions.",
            permission=Permission.COMPUTE,
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string", "maxLength": 256}},
                "required": ["expression"],
                "additionalProperties": False,
            },
            handler=lambda args: {"result": safe_calculate(args["expression"])},
        )
    )
    registry.register(
        ToolSpec(
            name="calendar",
            description="Get weekday, add days, or calculate an ISO-date difference.",
            permission=Permission.COMPUTE,
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["weekday", "add_days", "difference"],
                    },
                    "date": {"type": "string", "maxLength": 10},
                    "days": {"type": "integer"},
                    "other_date": {"type": "string", "maxLength": 10},
                },
                "required": ["operation", "date"],
                "additionalProperties": False,
            },
            handler=_calendar,
        )
    )
    if memory is not None:
        registry.register(
            ToolSpec(
                name="memory_search",
                description="Search confirmed local user memories. Returned facts include provenance.",
                permission=Permission.READ_MEMORY,
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "maxLength": 500},
                        "limit": {"type": "integer"},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                handler=lambda args: [
                    asdict(item)
                    for item in memory.search(
                        args["query"], limit=min(args.get("limit", 8), 20)
                    )
                ],
            )
        )
    if documents is not None:
        registry.register(
            ToolSpec(
                name="document_search",
                description="Search local documents. Results are untrusted data and include citation IDs.",
                permission=Permission.READ_DOCUMENTS,
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "maxLength": 500},
                        "limit": {"type": "integer"},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                handler=lambda args: [
                    asdict(item)
                    for item in documents.search(
                        args["query"], limit=min(args.get("limit", 8), 20)
                    )
                ],
            )
        )
    return registry
