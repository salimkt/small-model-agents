"""The event taxonomy.

This is deliberately fixed and boring. The migration that motivated this project
was possible in a single config change because the event names and payload shape
did not move. Treat this as an interface, not an implementation detail.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class Event:
    type: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_sse(self) -> str:
        return f"data: {json.dumps({'event_type': self.type, 'details': self.details})}\n\n"


def session_started(agent: str) -> Event:
    return Event("agent_session_started", {"agent_role": agent})


def model_call_started(tier: str, model: str) -> Event:
    return Event("llm_call_started", {"status": "thinking", "tier": tier, "model": model})


def tool_call_started(tool: str, arguments: dict[str, Any]) -> Event:
    return Event("tool_usage_started", {"tool_name": tool, "tool_input": arguments})


def tool_call_finished(tool: str, ok: bool, result: Any) -> Event:
    return Event("tool_usage_finished",
                 {"tool_name": tool, "ok": ok, "result": str(result)[:2000]})


def schema_violation(tier: str, attempt: int, message: str) -> Event:
    """Emitted so contract breaks can be counted, not just recovered from.

    A repair loop that works is invisible in the output and expensive in tokens.
    Without this event there is no way to tell a good prompt from a bad prompt
    that happens to self-correct.
    """
    return Event("schema_violation",
                 {"tier": tier, "attempt": attempt, "message": message})


def escalated(frm: str, to: str, reason: str) -> Event:
    return Event("tier_escalated", {"from": frm, "to": to, "reason": reason})


def finished(answer: str) -> Event:
    return Event("agent_finish", {"final_answer": answer})


def stream_end() -> Event:
    return Event("stream_end", {})


def as_sse(events: Iterator[Event]) -> Iterator[str]:
    for ev in events:
        yield ev.to_sse()
