"""Safe exact arithmetic for the local tool runtime."""

from __future__ import annotations

import ast
import operator
from decimal import Decimal, InvalidOperation, localcontext

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _number(value: float) -> Decimal:
    return Decimal(str(value))


def safe_calculate(expression: str, *, precision: int = 40) -> str:
    if len(expression) > 256:
        raise ValueError("expression is too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ValueError("invalid arithmetic expression") from error
    operations = 0

    def visit(node: ast.AST) -> Decimal:
        nonlocal operations
        operations += 1
        if operations > 128:
            raise ValueError("expression is too complex")
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if isinstance(node.value, bool):
                raise TypeError("booleans are not numbers")
            return _number(node.value)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](visit(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow):
                if right != right.to_integral_value() or abs(right) > 1000:
                    raise ValueError("power must be an integer between -1000 and 1000")
                if abs(left) > Decimal("1e100"):
                    raise ValueError("power base is too large")
                return left ** int(right)
            return _BINARY[type(node.op)](left, right)
        raise ValueError(f"unsupported expression element: {type(node).__name__}")

    try:
        with localcontext() as context:
            context.prec = precision
            value = visit(tree)
    except (InvalidOperation, ZeroDivisionError, OverflowError) as error:
        raise ValueError(str(error)) from error
    if not value.is_finite():
        raise ValueError("result is not finite")
    normalized = value.normalize()
    result = format(normalized, "f")
    return "0" if result in {"-0", ""} else result
