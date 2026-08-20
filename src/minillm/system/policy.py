"""Policy abstraction: the agent runtime does not depend on a model provider."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol

from .protocol import PolicyAction, parse_policy_action


@dataclass(frozen=True)
class PolicyContext:
    system: str
    messages: tuple[dict[str, str], ...]
    tool_schemas: tuple[dict[str, object], ...]


class Policy(Protocol):
    def act(self, context: PolicyContext) -> PolicyAction: ...


class ScriptedPolicy:
    """Deterministic policy for tests, trajectory replay, and protocol debugging."""

    def __init__(
        self, actions: Iterable[PolicyAction | str | dict[str, object]]
    ) -> None:
        self.actions = deque(actions)
        self.contexts: list[PolicyContext] = []

    def act(self, context: PolicyContext) -> PolicyAction:
        self.contexts.append(context)
        if not self.actions:
            raise RuntimeError("scripted policy has no action left")
        action = self.actions.popleft()
        return (
            action
            if not isinstance(action, (str, dict))
            else parse_policy_action(action)
        )


class CallablePolicy:
    """Adapter for local generation functions and OpenAI-compatible clients.

    The callable receives ``PolicyContext`` and must return one JSON object. Keeping
    provider integration outside the agent makes local/offline testing deterministic.
    """

    def __init__(
        self, generate: Callable[[PolicyContext], str | Mapping[str, object]]
    ) -> None:
        self.generate = generate

    def act(self, context: PolicyContext) -> PolicyAction:
        return parse_policy_action(self.generate(context))
