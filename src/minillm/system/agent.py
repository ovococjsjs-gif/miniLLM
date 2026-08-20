"""Bounded, permissioned agent loop with complete action tracing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from minillm.memory import EpisodicMemoryStore, MemoryFact

from .context import DEFAULT_CONTEXT_BUDGET, ContextBudget, build_policy_context
from .documents import DocumentChunk, DocumentStore
from .policy import Policy
from .protocol import FinalAnswer, MemoryProposal, ToolCall, action_to_json
from .router import route_bootstrap
from .tools import Permission, ToolRegistry, ToolResult


@dataclass(frozen=True)
class AgentTraceEvent:
    step: int
    kind: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class AgentResult:
    answer: str
    confidence: float
    citations: tuple[str, ...]
    memory_proposals: tuple[MemoryProposal, ...]
    trace: tuple[AgentTraceEvent, ...]
    stopped_reason: str


class Agent:
    def __init__(
        self,
        policy: Policy,
        tools: ToolRegistry,
        *,
        memory: EpisodicMemoryStore | None = None,
        documents: DocumentStore | None = None,
        allowed_permissions: frozenset[Permission] = frozenset(
            {Permission.COMPUTE, Permission.READ_MEMORY, Permission.READ_DOCUMENTS}
        ),
        max_steps: int = 6,
        context_budget: ContextBudget = DEFAULT_CONTEXT_BUDGET,
        bootstrap_routing: bool = True,
    ) -> None:
        self.policy = policy
        self.tools = tools
        self.memory = memory
        self.documents = documents
        self.allowed_permissions = allowed_permissions
        self.max_steps = max_steps
        self.context_budget = context_budget
        self.bootstrap_routing = bootstrap_routing

    def _evidence(self, query: str) -> tuple[list[MemoryFact], list[DocumentChunk]]:
        facts = self.memory.search(query, limit=8) if self.memory is not None else []
        chunks = (
            self.documents.search(query, limit=6) if self.documents is not None else []
        )
        return facts, chunks

    @staticmethod
    def _tool_event(step: int, result: ToolResult) -> AgentTraceEvent:
        return AgentTraceEvent(
            step,
            "tool_result",
            {
                "ok": result.ok,
                "tool": result.tool,
                "value": result.value,
                "error": result.error,
            },
        )

    def run(
        self, user_text: str, *, history: tuple[dict[str, str], ...] = ()
    ) -> AgentResult:
        if not user_text.strip():
            raise ValueError("user input cannot be empty")
        messages = [*history, {"role": "user", "content": user_text}]
        facts, chunks = self._evidence(user_text)
        valid_citations = {chunk.citation_id for chunk in chunks}
        trace: list[AgentTraceEvent] = []

        pending_bootstrap = (
            route_bootstrap(user_text) if self.bootstrap_routing else None
        )
        for step in range(self.max_steps):
            if pending_bootstrap is not None:
                action: ToolCall | FinalAnswer = pending_bootstrap.action
                pending_bootstrap = None
                trace.append(
                    AgentTraceEvent(
                        step,
                        "bootstrap_route",
                        {"tool": action.tool, "arguments": action.arguments},
                    )
                )
            else:
                context = build_policy_context(
                    messages,
                    self.tools.schemas(self.allowed_permissions),
                    memory_facts=facts,
                    document_chunks=chunks,
                    budget=self.context_budget,
                )
                action = self.policy.act(context)
                trace.append(
                    AgentTraceEvent(
                        step, "policy_action", json.loads(action_to_json(action))
                    )
                )

            if isinstance(action, ToolCall):
                result = self.tools.execute(
                    action.tool,
                    action.arguments,
                    allowed_permissions=self.allowed_permissions,
                )
                trace.append(self._tool_event(step, result))
                messages.append(
                    {"role": "assistant", "content": action_to_json(action)}
                )
                messages.append({"role": "tool", "content": result.to_json()})
                valid_citations.add(f"tool:{step}:{action.tool}")
                if result.ok and isinstance(result.value, list):
                    for item in result.value:
                        if isinstance(item, dict) and isinstance(
                            item.get("citation_id"), str
                        ):
                            valid_citations.add(item["citation_id"])
                continue

            invalid = set(action.citations) - valid_citations
            if invalid:
                messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps(
                            {
                                "ok": False,
                                "error": f"invalid citations: {sorted(invalid)}",
                            },
                            ensure_ascii=False,
                        ),
                    }
                )
                trace.append(
                    AgentTraceEvent(
                        step, "validation_error", {"invalid_citations": sorted(invalid)}
                    )
                )
                continue
            return AgentResult(
                action.content,
                action.confidence,
                action.citations,
                action.memory_proposals,
                tuple(trace),
                "final",
            )

        return AgentResult(
            "Не удалось безопасно завершить запрос в пределах лимита действий.",
            0.0,
            (),
            (),
            tuple(trace),
            "max_steps",
        )
