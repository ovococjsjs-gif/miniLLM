"""High-precision bootstrap routing before a learned router is available."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .protocol import ToolCall

_CALCULATOR = re.compile(
    r"^\s*(?:calc(?:ulate)?\s*:\s*)?([\d\s+\-*/%().]+)\s*$", re.IGNORECASE
)
_WEEKDAY = re.compile(
    r"^\s*(?:weekday|день\s+недели)\s*[: ]\s*(\d{4}-\d{2}-\d{2})\s*$", re.IGNORECASE
)


@dataclass(frozen=True)
class BootstrapDecision:
    action: ToolCall
    confidence: float


def route_bootstrap(text: str) -> BootstrapDecision | None:
    """Route only syntactically unambiguous requests; all others go to the policy."""

    weekday = _WEEKDAY.fullmatch(text)
    if weekday:
        return BootstrapDecision(
            ToolCall("calendar", {"operation": "weekday", "date": weekday.group(1)}),
            1.0,
        )
    arithmetic = _CALCULATOR.fullmatch(text)
    if arithmetic and any(character in arithmetic.group(1) for character in "+-*/%"):
        return BootstrapDecision(
            ToolCall("calculator", {"expression": arithmetic.group(1)}), 0.999
        )
    return None
