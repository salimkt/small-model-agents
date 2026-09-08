"""A three-tool agent you can actually run.

  export OPENAI_BASE_URL=http://localhost:8000/v1
  python -m examples.research_agent "What is 17% of 2340, and what day is it?"
"""
from __future__ import annotations

import datetime as _dt
import sys

from small_model_agents import Agent, Registry

registry = Registry()


@registry.register(
    "calculate",
    "Evaluate a arithmetic expression. Use for any arithmetic.",
    {"type": "object", "required": ["expression"],
     "properties": {"expression": {"type": "string"}}},
)
def calculate(expression: str) -> str:
    allowed = set("0123456789+-*/(). %")
    if not set(expression) <= allowed:
        return "rejected: only arithmetic characters are allowed"
    return str(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307 - charset-gated


@registry.register(
    "today",
    "Return today's date in ISO format.",
    {"type": "object", "properties": {}},
)
def today() -> str:
    return _dt.date.today().isoformat()


@registry.register(
    "fetch_url",
    "Fetch the text of a URL. Requires the httpx package.",
    {"type": "object", "required": ["url"], "properties": {"url": {"type": "string"}}},
    requires=("httpx",),
)
def fetch_url(url: str) -> str:
    import httpx
    return httpx.get(url, timeout=20, follow_redirects=True).text[:4000]


def build_registry() -> Registry:
    return registry


def main() -> None:
    task = " ".join(sys.argv[1:]) or "What is 17% of 2340?"
    for event in Agent(registry=registry, name="researcher").run(task):
        print(f"[{event.type}] {event.details}")
    print()


if __name__ == "__main__":
    main()
