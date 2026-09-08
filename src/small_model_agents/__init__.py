"""Reliable multi-agent tool calling on a small model."""
from .agent import Agent, OpenAICompatibleClient
from .router import EscalationPolicy, Router, Tier
from .tools import Registry, Tool, ToolUnavailable

__all__ = ["Agent", "OpenAICompatibleClient", "Router", "Tier",
           "EscalationPolicy", "Registry", "Tool", "ToolUnavailable",
           "EvalTask", "SuiteReport", "SimulatedModel",
           "run_suite", "run_task", "compare"]

_LAZY = {"EvalTask", "SuiteReport", "SimulatedModel",
         "run_suite", "run_task", "compare"}


def __getattr__(name: str):
    """The eval harness is imported on demand.

    Importing it eagerly would put it in sys.modules before
    `python -m small_model_agents.evaluation` executes it, which makes runpy
    warn. Not a real bug, but a warning on the first command anyone runs is a
    bad first impression.
    """
    if name in _LAZY:
        from . import evaluation

        return getattr(evaluation, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
