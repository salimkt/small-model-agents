"""Validated structured decisions, with repair-on-violation.

The model is never asked "what should happen next?". It is asked one bounded
question and must answer against a schema. If it violates the schema, the
violation is quoted back to it — that repair loop is what makes a 12B model
usable here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


class SchemaViolation(ValueError):
    pass


@dataclass
class Decision:
    """One narrow decision. Either call a tool, or finish."""

    action: str                      # "call_tool" | "finish"
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    answer: str | None = None
    confidence: float | None = None

    @staticmethod
    def json_schema(tool_names: list[str]) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["action"],
            "properties": {
                "action": {"enum": ["call_tool", "finish"]},
                "tool": {"enum": tool_names},
                "arguments": {"type": "object"},
                "answer": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
        }


def parse_decision(raw: str, tool_names: list[str]) -> Decision:
    """Parse and validate. Raises SchemaViolation with a message intended to be
    shown to the model, not to a human."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[-1] if "\n" in text else text
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise SchemaViolation(
            "Response was not JSON. Reply with a single JSON object and nothing else."
        )
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise SchemaViolation(f"Invalid JSON: {exc.msg}. Reply with valid JSON only.") from exc

    action = data.get("action")
    if action not in ("call_tool", "finish"):
        raise SchemaViolation('"action" must be exactly "call_tool" or "finish".')

    if action == "call_tool":
        tool = data.get("tool")
        if tool not in tool_names:
            raise SchemaViolation(
                f'"{tool}" is not an available tool. Choose one of: {tool_names}'
            )
        args = data.get("arguments")
        if not isinstance(args, dict):
            raise SchemaViolation('"arguments" must be a JSON object.')
        return Decision("call_tool", tool=tool, arguments=args,
                        confidence=data.get("confidence"))

    answer = data.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise SchemaViolation('When action is "finish", "answer" must be a non-empty string.')
    return Decision("finish", answer=answer, confidence=data.get("confidence"))
