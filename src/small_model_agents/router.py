"""Tiered model routing.

The point of this module: most turns in an agent loop are easy, and a small
model handles them. A few turns are genuinely hard. Paying for the large model
on every turn is the default in most frameworks and it is what makes agents
expensive.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class Tier(IntEnum):
    """Ordered so that escalation is `Tier(tier + 1)`."""

    FAST = 0      # classification, routing, yes/no checks
    SMALL = 1     # tool selection and argument filling — the common case
    LARGE = 2     # multi-step planning, ambiguous instructions
    THINKING = 3  # last resort; slow and expensive, used on evidence only


@dataclass(frozen=True)
class ModelSpec:
    name: str
    max_tokens: int = 1024
    temperature: float = 0.0


DEFAULT_TIERS: dict[Tier, ModelSpec] = {
    Tier.FAST: ModelSpec("qwen2.5:1.5b", max_tokens=256),
    Tier.SMALL: ModelSpec("gemma3:12b", max_tokens=1024),
    Tier.LARGE: ModelSpec("gemma3:27b", max_tokens=2048),
    Tier.THINKING: ModelSpec("qwen2.5:32b", max_tokens=4096, temperature=0.3),
}


@dataclass
class EscalationPolicy:
    """When to move up a tier.

    Every trigger is something you can count. That matters more than it sounds:
    if escalation is a judgement call buried in a prompt, you cannot tell a
    weak prompt from a hard task.
    """

    max_schema_failures: int = 2
    max_repeated_tool_errors: int = 2
    escalate_on_planning: bool = True
    ceiling: Tier = Tier.THINKING

    def next_tier(self, current: Tier, *, reason: str) -> tuple[Tier, str]:
        if current >= self.ceiling:
            return current, f"at ceiling ({current.name}), holding: {reason}"
        return Tier(current + 1), reason


@dataclass
class RunStats:
    """Escalations are the interesting metric. Count them by reason."""

    calls_by_tier: dict[str, int] = field(default_factory=dict)
    escalations: list[str] = field(default_factory=list)

    def record_call(self, tier: Tier) -> None:
        self.calls_by_tier[tier.name] = self.calls_by_tier.get(tier.name, 0) + 1

    def record_escalation(self, frm: Tier, to: Tier, reason: str) -> None:
        self.escalations.append(f"{frm.name}->{to.name}: {reason}")

    @property
    def escalation_rate(self) -> float:
        total = sum(self.calls_by_tier.values())
        return len(self.escalations) / total if total else 0.0


class Router:
    def __init__(
        self,
        tiers: dict[Tier, ModelSpec] | None = None,
        policy: EscalationPolicy | None = None,
    ) -> None:
        self.tiers = tiers or dict(DEFAULT_TIERS)
        self.policy = policy or EscalationPolicy()
        self.stats = RunStats()

    def spec_for(self, tier: Tier) -> ModelSpec:
        self.stats.record_call(tier)
        return self.tiers[tier]

    def escalate(self, current: Tier, reason: str) -> Tier:
        nxt, note = self.policy.next_tier(current, reason=reason)
        if nxt != current:
            self.stats.record_escalation(current, nxt, note)
        return nxt

    def tier_for_step(self, kind: str) -> Tier:
        """Start each kind of step at the cheapest tier that plausibly handles it."""
        if kind == "plan" and self.policy.escalate_on_planning:
            return Tier.LARGE
        if kind in ("classify", "is_complete"):
            return Tier.FAST
        return Tier.SMALL
