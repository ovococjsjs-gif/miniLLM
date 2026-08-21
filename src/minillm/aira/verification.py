"""Strict generated-answer verifiers shared by training and provider baselines."""

from __future__ import annotations

import ast
import builtins
import json
import re
from collections.abc import Mapping
from typing import Any

_CODE_BLOCK = re.compile(r"```python\s*\n(.*?)\n```", flags=re.DOTALL)
_SAFE_NODES = (
    ast.Module,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Return,
    ast.Assign,
    ast.For,
    ast.If,
    ast.Break,
    ast.Expr,
    ast.ListComp,
    ast.GeneratorExp,
    ast.comprehension,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Constant,
    ast.Compare,
    ast.BoolOp,
    ast.BinOp,
    ast.UnaryOp,
    ast.Attribute,
    ast.List,
    ast.Tuple,
    ast.Set,
    ast.Dict,
    ast.Subscript,
    ast.Slice,
    ast.keyword,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Mod,
    ast.And,
    ast.Or,
    ast.Not,
    ast.USub,
    ast.UAdd,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
)
_SAFE_CALLS = {"sum", "min", "max", "len", "set", "list"}
_SAFE_METHODS = {"add", "append"}
_SAFE_BUILTINS = {name: getattr(builtins, name) for name in _SAFE_CALLS}


def _record_field(record: Any, name: str) -> Any:
    if isinstance(record, Mapping):
        return record[name]
    return getattr(record, name)


def _message_content(record: Any, role: str) -> str:
    for message in _record_field(record, "messages"):
        message_role = message["role"] if isinstance(message, Mapping) else message.role
        if message_role == role:
            return str(
                message["content"] if isinstance(message, Mapping) else message.content
            )
    raise ValueError(f"record has no {role} message")


def _safe_python_function(code: str, expected_name: str):
    """Compile the tiny curriculum subset after rejecting unsafe Python syntax."""

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if (
        len(tree.body) != 1
        or len(functions) != 1
        or functions[0].name != expected_name
        or functions[0].decorator_list
        or len(functions[0].args.args) != 1
    ):
        return None
    for node in ast.walk(tree):
        if not isinstance(node, _SAFE_NODES):
            return None
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in _SAFE_CALLS:
                    return None
            elif isinstance(node.func, ast.Attribute):
                if (
                    not isinstance(node.func.value, ast.Name)
                    or node.func.attr not in _SAFE_METHODS
                ):
                    return None
            else:
                return None
        if isinstance(node, ast.Attribute) and not (
            isinstance(node.ctx, ast.Load) and node.attr in _SAFE_METHODS
        ):
            return None
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and abs(float(node.value)) > 1_000_000
        ):
            return None
    namespace: dict[str, Any] = {}
    try:
        exec(  # noqa: S102 - the AST and builtins are deliberately restricted above.
            compile(tree, "<verified-generated-code>", "exec"),
            {"__builtins__": _SAFE_BUILTINS},
            namespace,
        )
    except (MemoryError, RuntimeError, TypeError, ValueError):
        return None
    candidate = namespace.get(expected_name)
    return candidate if callable(candidate) else None


def _verify_python(verification: Mapping[str, Any], generated: str) -> bool:
    match = _CODE_BLOCK.search(generated)
    if match is None:
        return False
    function_name = str(verification["function"])
    candidate = _safe_python_function(match.group(1), function_name)
    if candidate is None:
        return False
    for test in verification.get("tests", ()):
        original = list(test["input"])
        argument = list(original)
        try:
            actual = candidate(argument)
        except (ArithmeticError, IndexError, KeyError, TypeError, ValueError):
            return False
        if actual != test["expected"] or argument != original:
            return False
    return True


def verify_synthetic_generation(record: Any, generated: str) -> bool:
    """Verify a generated answer against an AIra Mentor record.

    This evaluates behavior produced in the actual generated context. Python answers
    are AST-restricted and run only against the deterministic tests stored in the
    record; importing, I/O, unbounded loops, and arbitrary calls are rejected.
    """

    verification = _record_field(record, "verification")
    category = str(_record_field(record, "category"))
    expected = verification.get("expected")
    kind = verification["kind"]
    if kind == "json_equal":
        try:
            return json.loads(generated) == expected
        except (json.JSONDecodeError, TypeError):
            return False
    if kind == "python_tests":
        return _verify_python(verification, generated)
    if category == "arithmetic":
        escaped = re.escape(str(expected))
        return (
            re.search(
                rf"(?:Answer:|Ответ:|=)\s*{escaped}(?:\s+(?:рублей|units))?\.?\s*$",
                generated,
            )
            is not None
        )
    if category == "algebra":
        escaped = re.escape(str(expected))
        return re.search(rf"[xyz]\s*=\s*{escaped}(?:\D|$)", generated) is not None
    if category in {"logic", "memory_control", "uncertainty"}:
        return generated == _message_content(record, "assistant")
    if category in {"grounded_qa", "prompt_injection"}:
        return str(expected) in generated and f"[{verification['citation']}]" in generated
    if category == "critique_revision":
        return generated.rstrip(".").endswith(str(expected))
    return False
