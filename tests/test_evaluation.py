"""The harness itself, and the gate.

The important test here is not that the suite passes. It is that the gate
notices when the model gets worse — an eval that cannot fail is decoration.
"""
from __future__ import annotations

import json
from pathlib import Path

from small_model_agents import Registry
from small_model_agents.evaluation import (
    DEFAULT_TASKS,
    SimulatedModel,
    compare,
    default_registry,
    run_suite,
    run_task,
)

BASELINE = Path(__file__).resolve().parent.parent / "evals" / "baseline.json"


def _baseline_client() -> SimulatedModel:
    return SimulatedModel(reliability=0.75, seed=7)


def test_suite_matches_committed_baseline():
    stored = json.loads(BASELINE.read_text())["metrics"]
    metrics = run_suite(client=_baseline_client()).metrics
    assert metrics == stored, (
        "the committed baseline no longer reproduces; regenerate it deliberately "
        "with --update-baseline, do not edit it by hand"
    )


def test_run_is_deterministic():
    a = run_suite(client=_baseline_client()).metrics
    b = run_suite(client=_baseline_client()).metrics
    assert a == b


def test_gate_catches_a_worse_model():
    """A less reliable model must trip the gate on at least one metric."""
    baseline = run_suite(client=_baseline_client()).metrics
    degraded = run_suite(client=SimulatedModel(reliability=0.35, seed=7)).metrics
    regressions = compare(baseline, degraded)
    assert regressions, f"gate missed a degraded model: {degraded}"
    assert {r.metric for r in regressions} & {
        "pass_rate", "schema_violation_rate", "escalation_rate"
    }


def test_gate_stays_quiet_on_a_better_model():
    baseline = run_suite(client=_baseline_client()).metrics
    better = run_suite(client=SimulatedModel(reliability=0.99, seed=7)).metrics
    assert compare(baseline, better) == []


def test_escalation_recovers_rather_than_just_failing():
    """Escalated tasks should still pass — that is the whole claim."""
    report = run_suite(client=_baseline_client())
    escalated = [r for r in report.results if r.escalations]
    assert escalated, "expected the baseline run to exercise the escalation path"
    assert all(r.passed for r in escalated), [r.id for r in escalated if not r.passed]


def test_schema_violations_are_counted_not_just_recovered():
    report = run_suite(client=_baseline_client())
    assert sum(r.schema_violations for r in report.results) > 0
    assert report.metrics["schema_violation_rate"] > 0


def test_unavailable_tool_degrades_instead_of_looping():
    task = next(t for t in DEFAULT_TASKS if t.id == "unavailable-tool")
    res = run_task(task, client=_baseline_client())
    assert res.passed, res.reasons
    assert res.tool_errors >= 1
    assert "Step budget exhausted" not in res.answer


def test_dotted_requirement_does_not_crash_the_catalogue():
    """Regression: find_spec raises, not returns None, for a missing parent package."""
    reg = Registry()

    @reg.register("warehouse", "needs a submodule of an absent package",
                  {"type": "object", "properties": {}},
                  requires=("definitely_absent_pkg.submodule",))
    def warehouse() -> str:
        return "unreachable"

    assert reg.catalogue()  # would previously raise ModuleNotFoundError
    ok, msg = reg.call("warehouse", {})
    assert ok is False and "definitely_absent_pkg.submodule" in msg


def test_registry_reports_the_unavailable_tool():
    reg = default_registry()
    assert reg.tools["query_warehouse"].available is False
    assert reg.tools["calculate"].available is True
