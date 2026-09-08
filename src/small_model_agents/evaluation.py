"""An evaluation harness with a committed baseline and a regression gate.

Why this module exists: "the 12B is as reliable as the 32B" is a claim, and a
claim you cannot re-run is an anecdote. This is the smallest thing that turns it
into a measurement — a fixed task set, metrics that are counted rather than
judged, a baseline checked into the repository, and a non-zero exit code when a
change makes any of them worse.

The metric set is chosen so that a prompt tweak cannot quietly trade
reliability for cost, or cost for reliability:

    pass_rate               did the run do the thing            higher better
    escalation_rate         how often the cheap tier failed     lower better
    schema_violation_rate   how often the contract was broken   lower better
    mean_model_calls        cost proxy, per task                lower better
    tool_precision          tool calls the task actually wanted higher better

Optimising any one of these alone shows up as a regression in another, which is
the entire point of gating on all five.

Against a real server:

    OPENAI_BASE_URL=http://localhost:8000/v1 python -m small_model_agents.evaluation

Offline, against a simulated model whose failure rate is tier-dependent — this
is what CI runs, and it is deterministic:

    python -m small_model_agents.evaluation --simulate
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .agent import Agent, ChatClient, OpenAICompatibleClient
from .router import DEFAULT_TIERS, Router, Tier
from .tools import Registry

# ---------------------------------------------------------------- task + result


@dataclass(frozen=True)
class EvalTask:
    """One task, with expectations that are checkable without a human.

    Deliberately not an LLM-as-judge. A judge model would be another
    unversioned dependency in the measurement, and the failure modes this suite
    exists to catch — malformed JSON, wrong tool, runaway loops — are all
    mechanically detectable.
    """

    id: str
    prompt: str
    expect_tools: tuple[str, ...] = ()      # every one of these must be called
    forbid_tools: bool = False              # no tool may be called at all
    answer_contains: tuple[str, ...] = ()   # case-insensitive, all required
    answer_contains_any: tuple[str, ...] = ()
    max_steps: int = 8
    tags: tuple[str, ...] = ()


@dataclass
class TaskResult:
    id: str
    passed: bool
    reasons: list[str] = field(default_factory=list)
    steps: int = 0
    model_calls: int = 0
    schema_violations: int = 0
    escalations: list[str] = field(default_factory=list)
    tools_called: list[str] = field(default_factory=list)
    tool_errors: int = 0
    max_tier: str = Tier.SMALL.name
    answer: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# -------------------------------------------------------------------- the suite


def default_registry() -> Registry:
    """Tools for the suite. One of them is deliberately unavailable.

    `query_warehouse` declares a driver that is never installed here. That is
    not a gap in the suite — degrading cleanly on a tool you cannot run is one
    of the behaviours being measured, and it is the common case once a registry
    has seventy of them. Nothing in the suite touches the network or the clock;
    an eval that can flake is not a gate.
    """
    reg = Registry()

    @reg.register("calculate", "Evaluate an arithmetic expression.",
                  {"type": "object", "required": ["expression"],
                   "properties": {"expression": {"type": "string"}}})
    def calculate(expression: str) -> str:
        allowed = set("0123456789+-*/(). ")
        if not set(expression) <= allowed:
            raise ValueError(f"unsupported characters in {expression!r}")
        return str(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307 - charset-gated

    @reg.register("today", "Return today's date as YYYY-MM-DD.",
                  {"type": "object", "properties": {}})
    def today() -> str:
        # Fixed so the suite is reproducible. A clock in an eval is a flake.
        return "2026-01-01"

    @reg.register("word_count", "Count the words in a piece of text.",
                  {"type": "object", "required": ["text"],
                   "properties": {"text": {"type": "string"}}})
    def word_count(text: str) -> str:
        return str(len(text.split()))

    @reg.register("lookup_customer", "Look up a customer by their ID.",
                  {"type": "object", "required": ["customer_id"],
                   "properties": {"customer_id": {"type": "string"}}})
    def lookup_customer(customer_id: str) -> str:
        book = {"C-1042": "Acme Corp, plan=enterprise, seats=240"}
        if customer_id not in book:
            raise KeyError(f"no customer {customer_id}")
        return book[customer_id]

    @reg.register("query_warehouse", "Run a read-only SQL query against the warehouse.",
                  {"type": "object", "required": ["sql"],
                   "properties": {"sql": {"type": "string"}}},
                  requires=("snowflake.connector",))
    def query_warehouse(sql: str) -> str:  # pragma: no cover - driver never present in CI
        import snowflake.connector

        raise NotImplementedError("wire up a real connection to use this")

    return reg


DEFAULT_TASKS: tuple[EvalTask, ...] = (
    EvalTask("arith-single", "What is 18 * 24?",
             expect_tools=("calculate",), answer_contains=("432",), tags=("tool",)),
    EvalTask("arith-nested", "Compute (120 + 45) / 3.",
             expect_tools=("calculate",), answer_contains=("55",), tags=("tool",)),
    EvalTask("date", "What is today's date?",
             expect_tools=("today",), answer_contains=("2026-01-01",), tags=("tool",)),
    EvalTask("word-count", 'How many words are in "the quick brown fox jumps"?',
             expect_tools=("word_count",), answer_contains=("5",), tags=("tool",)),
    EvalTask("lookup-hit", "Look up customer C-1042 and tell me their plan.",
             expect_tools=("lookup_customer",), answer_contains=("enterprise",),
             tags=("tool",)),
    EvalTask("lookup-miss", "Look up customer C-9999.",
             expect_tools=("lookup_customer",),
             answer_contains_any=("not", "no customer", "could not", "unable"),
             tags=("tool", "error-path")),
    EvalTask("multi-step", "What is 12 * 12, and what is today's date?",
             expect_tools=("calculate", "today"),
             answer_contains=("144", "2026-01-01"), tags=("multi-step",)),
    EvalTask("no-tool", "Say hello.",
             forbid_tools=True, tags=("restraint",)),
    EvalTask("unavailable-tool", "Pull last quarter's revenue from the warehouse.",
             expect_tools=("query_warehouse",),
             answer_contains_any=("snowflake", "unavailable", "install", "cannot"),
             tags=("degradation",)),
    EvalTask("impossible", "Send an email to bob@acme.test saying hi.",
             forbid_tools=True,
             answer_contains_any=("cannot", "can't", "no tool", "unable", "not able"),
             tags=("restraint", "degradation")),
    EvalTask("bad-argument", "Calculate the square root of sixteen.",
             answer_contains_any=("4", "cannot", "unable", "unsupported"),
             tags=("error-path",)),
    EvalTask("ambiguous", "Do the usual thing.",
             forbid_tools=True,
             answer_contains_any=("clarif", "which", "what", "more detail", "unclear"),
             tags=("restraint",)),
)


# ------------------------------------------------------------- simulated model


_TIER_FAILURE_WEIGHT = {
    Tier.FAST: 1.6,
    Tier.SMALL: 1.0,
    Tier.LARGE: 0.25,
    Tier.THINKING: 0.05,
}

_MODEL_TO_TIER = {spec.name: tier for tier, spec in DEFAULT_TIERS.items()}


class SimulatedModel:
    """A stand-in whose failure rate depends on the tier, seeded and reproducible.

    This exists so the harness itself is testable, and so CI has a signal at all
    without a GPU. It is not a claim about any real model. The tier weighting is
    the assumption under test in production: cheap tiers break the output
    contract more often, and escalating is what fixes it.
    """

    def __init__(self, reliability: float = 0.75, seed: int = 7) -> None:
        self.reliability = reliability
        self.seed = seed

    # deterministic in [0, 1) from the full call context, so a re-run of the
    # suite draws exactly the same "luck" as the run that produced the baseline
    def _draw(self, *parts: object) -> float:
        key = "|".join(str(p) for p in (self.seed, *parts)).encode()
        return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64

    def complete(self, messages, model, max_tokens, temperature):
        task = next((m["content"] for m in messages if m["role"] == "user"), "")
        observations = [m["content"] for m in messages
                        if m["role"] == "user" and m["content"].startswith("Tool result")]
        step = len([m for m in messages if m["role"] == "assistant"])

        tier = _MODEL_TO_TIER.get(model, Tier.SMALL)
        p_fail = (1.0 - self.reliability) * _TIER_FAILURE_WEIGHT[tier]
        if self._draw(task, step, model) < p_fail:
            return "Sure — let me use a tool for that. First I will call calculate()."

        plan = self._plan(task)
        if len(observations) < len(plan):
            tool, args = plan[len(observations)]
            return json.dumps({"action": "call_tool", "tool": tool,
                               "arguments": args, "confidence": 0.8})
        return json.dumps({"action": "finish", "answer": self._answer(task, observations),
                           "confidence": 0.8})

    @staticmethod
    def _plan(task: str) -> list[tuple[str, dict[str, Any]]]:
        t = task.lower()
        plan: list[tuple[str, dict[str, Any]]] = []
        if "18 * 24" in t:
            plan.append(("calculate", {"expression": "18 * 24"}))
        if "(120 + 45)" in t:
            plan.append(("calculate", {"expression": "(120 + 45) / 3"}))
        if "12 * 12" in t:
            plan.append(("calculate", {"expression": "12 * 12"}))
        if "square root" in t:
            plan.append(("calculate", {"expression": "sqrt(16)"}))
        if "words" in t:
            plan.append(("word_count", {"text": "the quick brown fox jumps"}))
        if "c-1042" in t:
            plan.append(("lookup_customer", {"customer_id": "C-1042"}))
        if "c-9999" in t:
            plan.append(("lookup_customer", {"customer_id": "C-9999"}))
        if "warehouse" in t:
            plan.append(("query_warehouse", {"sql": "select sum(revenue) from q4"}))
        if "date" in t:
            plan.append(("today", {}))
        return plan

    @staticmethod
    def _answer(task: str, observations: Sequence[str]) -> str:
        t = task.lower()
        if not observations:
            if "hello" in t:
                return "Hello."
            if "email" in t:
                return "I cannot do that — I have no tool for sending email."
            if "usual" in t:
                return "Could you clarify which task you mean?"
            return "I do not have enough to act on."
        errors = [o for o in observations if o.startswith("Tool result (error")]
        if errors and len(errors) == len(observations):
            return f"That did not work: {errors[-1].split(':', 1)[-1].strip()}"
        values = [o.split(":", 1)[-1].strip() for o in observations
                  if not o.startswith("Tool result (error")]
        return "Result: " + "; ".join(values)


# --------------------------------------------------------------------- grading


def _grade(task: EvalTask, res: TaskResult) -> None:
    answer = res.answer.lower()

    for tool in task.expect_tools:
        if tool not in res.tools_called:
            res.reasons.append(f"expected a call to {tool}")

    if task.forbid_tools and res.tools_called:
        res.reasons.append(f"called tools when none were needed: {res.tools_called}")

    for needle in task.answer_contains:
        if needle.lower() not in answer:
            res.reasons.append(f"answer missing {needle!r}")

    if task.answer_contains_any and not any(
        n.lower() in answer for n in task.answer_contains_any
    ):
        res.reasons.append(f"answer matched none of {list(task.answer_contains_any)}")

    if "Step budget exhausted" in res.answer:
        res.reasons.append("ran out of steps")

    res.passed = not res.reasons


def run_task(
    task: EvalTask,
    *,
    client: ChatClient,
    registry_factory: Callable[[], Registry] = default_registry,
) -> TaskResult:
    router = Router()
    agent = Agent(registry=registry_factory(), router=router,
                  client=client, max_steps=task.max_steps)
    res = TaskResult(id=task.id, passed=False)
    max_tier = Tier.FAST

    for event in agent.run(task.prompt):
        if event.type == "llm_call_started":
            res.model_calls += 1
            max_tier = max(max_tier, Tier[event.details["tier"]])
        elif event.type == "schema_violation":
            res.schema_violations += 1
        elif event.type == "tool_usage_started":
            res.steps += 1
            res.tools_called.append(event.details["tool_name"])
        elif event.type == "tool_usage_finished":
            if not event.details["ok"]:
                res.tool_errors += 1
        elif event.type == "tier_escalated":
            res.escalations.append(
                f"{event.details['from']}->{event.details['to']}: {event.details['reason']}"
            )
        elif event.type == "agent_finish":
            res.answer = event.details["final_answer"]

    res.max_tier = max_tier.name
    _grade(task, res)
    return res


# --------------------------------------------------------------------- reporting

# Which way is better. The gate needs this; a raw diff cannot tell an
# improvement from a regression.
HIGHER_IS_BETTER = {
    "pass_rate": True,
    "tool_precision": True,
    "escalation_rate": False,
    "schema_violation_rate": False,
    "mean_model_calls": False,
}

# How much movement is noise. Tight on correctness, loose on cost.
DEFAULT_TOLERANCE = {
    "pass_rate": 0.0,
    "tool_precision": 0.02,
    "escalation_rate": 0.05,
    "schema_violation_rate": 0.05,
    "mean_model_calls": 0.5,
}


@dataclass
class SuiteReport:
    results: list[TaskResult]
    metrics: dict[str, float]
    label: str = "unlabelled"

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "metrics": self.metrics,
            "results": [r.to_dict() for r in self.results],
        }

    @property
    def failures(self) -> list[TaskResult]:
        return [r for r in self.results if not r.passed]


def _metrics(tasks: Sequence[EvalTask], results: Sequence[TaskResult]) -> dict[str, float]:
    n = len(results) or 1
    by_id = {t.id: t for t in tasks}

    model_calls = sum(r.model_calls for r in results) or 1
    wanted = expected = 0
    for r in results:
        task = by_id[r.id]
        if not task.expect_tools and not r.tools_called:
            wanted += 1
            expected += 1
            continue
        for tool in r.tools_called:
            wanted += 1
            expected += int(tool in task.expect_tools)

    return {
        "n": float(len(results)),
        "pass_rate": round(sum(r.passed for r in results) / n, 4),
        "escalation_rate": round(
            sum(len(r.escalations) for r in results) / model_calls, 4),
        "schema_violation_rate": round(
            sum(r.schema_violations for r in results) / model_calls, 4),
        "mean_model_calls": round(sum(r.model_calls for r in results) / n, 4),
        "tool_precision": round(expected / wanted, 4) if wanted else 1.0,
    }


def run_suite(
    tasks: Sequence[EvalTask] = DEFAULT_TASKS,
    *,
    client: ChatClient | None = None,
    registry_factory: Callable[[], Registry] = default_registry,
    label: str = "unlabelled",
) -> SuiteReport:
    client = client or OpenAICompatibleClient()
    results = [run_task(t, client=client, registry_factory=registry_factory)
               for t in tasks]
    return SuiteReport(results=results, metrics=_metrics(tasks, results), label=label)


@dataclass(frozen=True)
class Regression:
    metric: str
    baseline: float
    current: float
    tolerance: float

    def __str__(self) -> str:
        arrow = "up" if self.current > self.baseline else "down"
        return (f"{self.metric}: {self.baseline} -> {self.current} "
                f"({arrow}, tolerance {self.tolerance})")


def compare(
    baseline: dict[str, float],
    current: dict[str, float],
    tolerance: dict[str, float] | None = None,
) -> list[Regression]:
    """Metrics that moved the wrong way by more than their tolerance."""
    tol = {**DEFAULT_TOLERANCE, **(tolerance or {})}
    out: list[Regression] = []
    for metric, higher_better in HIGHER_IS_BETTER.items():
        if metric not in baseline or metric not in current:
            continue
        b, c, t = baseline[metric], current[metric], tol.get(metric, 0.0)
        worse_by = (b - c) if higher_better else (c - b)
        if worse_by > t:
            out.append(Regression(metric, b, c, t))
    return out


def format_report(report: SuiteReport, regressions: Iterable[Regression] = ()) -> str:
    lines = [f"suite: {report.label}", ""]
    width = max(len(r.id) for r in report.results)
    for r in report.results:
        mark = "pass" if r.passed else "FAIL"
        lines.append(
            f"  {mark}  {r.id:<{width}}  "
            f"calls={r.model_calls} steps={r.steps} "
            f"viol={r.schema_violations} esc={len(r.escalations)} tier={r.max_tier}"
        )
        for reason in r.reasons:
            lines.append(f"        - {reason}")
    lines += ["", "metrics:"]
    for k, v in report.metrics.items():
        lines.append(f"  {k:<24} {v}")
    regressions = list(regressions)
    if regressions:
        lines += ["", "REGRESSIONS:"] + [f"  {r}" for r in regressions]
    return "\n".join(lines)


# -------------------------------------------------------------------------- CLI

BASELINE_PATH = Path("evals/baseline.json")


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="small_model_agents.evaluation",
                                description="Run the eval suite and gate on regressions.")
    p.add_argument("--simulate", action="store_true",
                   help="use the seeded simulated model instead of a server")
    p.add_argument("--reliability", type=float, default=0.75,
                   help="simulated model reliability (with --simulate)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    p.add_argument("--update-baseline", action="store_true",
                   help="overwrite the baseline with this run")
    p.add_argument("--json", type=Path, help="also write the full report here")
    p.add_argument("--no-gate", action="store_true",
                   help="report regressions but still exit 0")
    args = p.parse_args(argv)

    if args.simulate:
        client: ChatClient = SimulatedModel(args.reliability, args.seed)
        label = f"simulated(reliability={args.reliability}, seed={args.seed})"
    else:
        client = OpenAICompatibleClient()
        if not client.live:
            print("no OPENAI_BASE_URL set; use --simulate for an offline run")
            return 2
        label = f"live({client.base_url})"

    report = run_suite(client=client, label=label)

    regressions: list[Regression] = []
    if args.update_baseline:
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(
            json.dumps({"label": label, "metrics": report.metrics}, indent=2) + "\n")
        print(f"baseline written to {args.baseline}")
    elif args.baseline.exists():
        stored = json.loads(args.baseline.read_text())
        regressions = compare(stored["metrics"], report.metrics)

    print(format_report(report, regressions))
    if args.json:
        args.json.write_text(json.dumps(report.to_dict(), indent=2) + "\n")

    if regressions and not args.no_gate:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
