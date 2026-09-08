"""The escalation policy, tested without a model server."""
from __future__ import annotations

import json

from small_model_agents import Agent, Registry, Router, Tier
from small_model_agents.router import EscalationPolicy
from small_model_agents.schema import SchemaViolation, parse_decision
from small_model_agents.tools import Registry as R


class AlwaysInvalid:
    """A model that never emits valid JSON — forces the schema-repair path."""

    def complete(self, messages, model, max_tokens, temperature):
        return "Sure! I'll use the calculate tool for that."


class FinishesImmediately:
    def complete(self, messages, model, max_tokens, temperature):
        return json.dumps({"action": "finish", "answer": "42", "confidence": 0.9})


def _registry() -> R:
    reg = R()

    @reg.register("noop", "does nothing", {"type": "object", "properties": {}})
    def noop() -> str:
        return "ok"

    return reg


def test_schema_violation_escalates_after_threshold():
    router = Router(policy=EscalationPolicy(max_schema_failures=2))
    agent = Agent(registry=_registry(), router=router,
                  client=AlwaysInvalid(), max_steps=6)
    kinds = [e.type for e in agent.run("anything")]
    assert "tier_escalated" in kinds, kinds
    assert router.stats.escalations, "escalation should be recorded with a reason"


def test_clean_finish_never_escalates():
    router = Router()
    agent = Agent(registry=_registry(), router=router, client=FinishesImmediately())
    kinds = [e.type for e in agent.run("anything")]
    assert "agent_finish" in kinds
    assert "tier_escalated" not in kinds
    assert router.stats.escalation_rate == 0.0


def test_router_holds_at_ceiling():
    router = Router(policy=EscalationPolicy(ceiling=Tier.SMALL))
    assert router.escalate(Tier.SMALL, "reason") == Tier.SMALL


def test_planning_starts_higher():
    router = Router()
    assert router.tier_for_step("plan") == Tier.LARGE
    assert router.tier_for_step("act") == Tier.SMALL
    assert router.tier_for_step("classify") == Tier.FAST


def test_unavailable_tool_is_listed_but_fails_clearly():
    reg = R()

    @reg.register("needs_missing", "needs a missing package",
                  {"type": "object", "properties": {}},
                  requires=("definitely_not_installed_xyz",))
    def needs_missing() -> str:
        return "unreachable"

    assert any("needs_missing" in json.dumps(t) for t in reg.catalogue())
    ok, msg = reg.call("needs_missing", {})
    assert ok is False and "definitely_not_installed_xyz" in msg


def test_schema_violation_message_is_actionable():
    try:
        parse_decision("not json at all", ["noop"])
    except SchemaViolation as exc:
        assert "JSON" in str(exc)
    else:
        raise AssertionError("should have raised")
