"""The loop: decide -> validate -> execute -> observe.

Control flow lives here, in Python, where it can be tested. The model is called
only to answer bounded questions.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol

from . import events as ev
from .router import Router, Tier
from .schema import Decision, SchemaViolation, parse_decision
from .tools import Registry

SYSTEM = """You are the decision step of an agent loop.

Answer with a SINGLE JSON object and nothing else. No prose, no code fences.

To use a tool:
  {{"action": "call_tool", "tool": "<name>", "arguments": {{...}}, "confidence": 0.0-1.0}}

To answer the user:
  {{"action": "finish", "answer": "<your answer>", "confidence": 0.0-1.0}}

Available tools:
{tools}
"""


class ChatClient(Protocol):
    def complete(self, messages: list[dict[str, str]], model: str,
                 max_tokens: int, temperature: float) -> str: ...


class OpenAICompatibleClient:
    """Works against vLLM, Ollama, llama.cpp — anything serving /v1/chat/completions.

    Falls back to a deterministic stub when no server is configured, so the loop
    and the escalation policy are testable without a GPU.
    """

    def __init__(self, base_url: str | None = None, api_key: str | None = None) -> None:
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "").rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or "not-needed"

    @property
    def live(self) -> bool:
        return bool(self.base_url)

    def complete(self, messages, model, max_tokens, temperature):
        if not self.live:
            return json.dumps({"action": "finish",
                               "answer": "[stub: set OPENAI_BASE_URL for a live model]",
                               "confidence": 0.0})
        import urllib.request

        body = json.dumps({
            "model": model, "messages": messages,
            "max_tokens": max_tokens, "temperature": temperature,
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read())
        return payload["choices"][0]["message"]["content"]


@dataclass
class Agent:
    registry: Registry
    router: Router = field(default_factory=Router)
    client: ChatClient = field(default_factory=OpenAICompatibleClient)
    name: str = "agent"
    max_steps: int = 8

    def run(self, task: str) -> Iterator[ev.Event]:
        yield ev.session_started(self.name)

        names = sorted(self.registry.tools)
        catalogue = "\n".join(
            f"- {t.name}: {t.description}"
            + ("" if t.available else f"  [unavailable: needs {', '.join(t.missing)}]")
            for t in self.registry.tools.values()
        )
        messages = [
            {"role": "system", "content": SYSTEM.format(tools=catalogue)},
            {"role": "user", "content": task},
        ]

        tier = self.router.tier_for_step("act")
        schema_failures = 0
        tool_errors = 0

        for _ in range(self.max_steps):
            spec = self.router.spec_for(tier)
            yield ev.model_call_started(tier.name, spec.name)
            raw = self.client.complete(messages, spec.name, spec.max_tokens, spec.temperature)

            try:
                decision = parse_decision(raw, names)
                schema_failures = 0
            except SchemaViolation as exc:
                schema_failures += 1
                yield ev.schema_violation(tier.name, schema_failures, str(exc))
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user",
                                 "content": f"That response was rejected: {exc} Try again."})
                if schema_failures >= self.router.policy.max_schema_failures:
                    nxt = self.router.escalate(tier, f"{schema_failures} schema violations")
                    if nxt != tier:
                        yield ev.escalated(tier.name, nxt.name, "repeated schema violations")
                        tier, schema_failures = nxt, 0
                continue

            if decision.action == "finish":
                yield ev.finished(decision.answer or "")
                yield ev.stream_end()
                return

            yield ev.tool_call_started(decision.tool or "?", decision.arguments or {})
            ok, result = self.registry.call(decision.tool or "", decision.arguments or {})
            yield ev.tool_call_finished(decision.tool or "?", ok, result)

            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user",
                             "content": f"Tool result ({'ok' if ok else 'error'}): {result}"})

            if not ok:
                tool_errors += 1
                if tool_errors >= self.router.policy.max_repeated_tool_errors:
                    nxt = self.router.escalate(tier, f"{tool_errors} tool errors")
                    if nxt != tier:
                        yield ev.escalated(tier.name, nxt.name, "repeated tool errors")
                        tier, tool_errors = nxt, 0
            else:
                tool_errors = 0

        yield ev.finished("Step budget exhausted without a final answer.")
        yield ev.stream_end()
