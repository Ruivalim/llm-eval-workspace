"""Deterministic fake tools for the tool_call eval type. The model gets these
as OpenAI function schemas; calls are answered locally and logged in the trace."""

from __future__ import annotations

import json
from typing import Any

TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "get_weather": {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    "calculate_math": {
        "type": "function",
        "function": {
            "name": "calculate_math",
            "description": "Evaluate an arithmetic expression exactly.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    "fetch_webpage_content": {
        "type": "function",
        "function": {
            "name": "fetch_webpage_content",
            "description": "Fetch the plain-text content of a URL.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
    "control_light": {
        "type": "function",
        "function": {
            "name": "control_light",
            "description": "Turn a room light on or off.",
            "parameters": {
                "type": "object",
                "properties": {
                    "room": {"type": "string"},
                    "action": {"type": "string", "enum": ["on", "off"]},
                },
                "required": ["room", "action"],
            },
        },
    },
}


def _safe_eval(expr: str) -> str:
    import ast
    import operator as op

    ops = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Pow: op.pow,
        ast.USub: op.neg,
        ast.Mod: op.mod,
    }

    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp):
            return ops[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp):
            return ops[type(node.op)](ev(node.operand))
        raise ValueError("unsupported expression")

    expr = expr.replace("×", "*").replace("x", "*").replace("÷", "/")
    val = ev(ast.parse(expr, mode="eval").body)
    return str(int(val)) if float(val).is_integer() else str(val)


def execute(name: str, arguments: dict[str, Any]) -> str:
    """Return the tool result as a JSON string, as the model expects."""
    if name == "get_weather":
        return json.dumps({"city": arguments.get("city"), "temp_c": 18, "condition": "cloudy"})
    if name == "calculate_math":
        try:
            return json.dumps({"result": _safe_eval(str(arguments.get("expression", "")))})
        except Exception as exc:  # noqa: BLE001 - the model must see the failure text
            return json.dumps({"error": str(exc)})
    if name == "fetch_webpage_content":
        return json.dumps(
            {
                "url": arguments.get("url"),
                "content": (
                    "A large language model (LLM) is a type of machine learning model designed "
                    "for natural language processing tasks. LLMs are trained on vast text corpora "
                    "and use the transformer architecture. They can perform text generation, "
                    "summarization, translation and question answering."
                ),
            }
        )
    if name == "control_light":
        room = str(arguments.get("room", ""))
        if room == "garage":
            return json.dumps({"error": "device offline: garage light is unreachable"})
        if room not in ("living_room", "bedroom", "kitchen"):
            return json.dumps({"error": f"unknown room {room!r}"})
        return json.dumps({"room": room, "state": arguments.get("action")})
    return json.dumps({"error": f"unknown tool {name}"})


def schemas_for(names: list[str]) -> list[dict[str, Any]]:
    return [TOOL_SCHEMAS[n] for n in names if n in TOOL_SCHEMAS]
