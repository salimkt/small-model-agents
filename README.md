# small-model-agents

**Reliable multi-agent tool calling on a 12B model, by giving the model less to decide.**

I spent a chunk of the last two years replacing a framework-based multi-agent stack
with a deterministic orchestrator. The headline result was that a **12B model did the
work the previous stack needed a 32B model for** — enough GPU saving to run the whole
platform on a single node instead of a cluster.

This repository is the idea, extracted and rebuilt from scratch as something you can
run. It is not the system I built at work; it is the smallest honest expression of what
made that system work.

---

## The problem

Most agent frameworks hand the model a large, open-ended decision on every turn:
here are twelve tools, here is the conversation, here is a scratchpad — now emit
free-form reasoning and, somewhere in it, a well-formed tool call.

Large models absorb that. Small models do not. They emit prose where you wanted JSON,
invent tool names, forget the schema three turns in, and loop. The usual response is to
reach for a bigger model, which is why so many agent deployments quietly require 30B+
and a GPU budget to match.

The alternative is to move the hard parts out of the model.

## The approach

**1. Make every model call a narrow, schema-validated decision.**

The model never decides *what happens next* in a general sense. It answers one bounded
question — "which of these tools, with which arguments?" or "is this answer complete?"
— against a schema, and the orchestrator validates before anything executes. An invalid
response is a retry with the violation quoted back, not a crash and not a silent
misfire. Control flow lives in Python, where it is testable.

**2. Route by difficulty, not by default.**

Four tiers — `FAST`, `SMALL`, `LARGE`, `THINKING`. Most turns are tool selection and
argument filling, which the small tier handles. Planning a multi-step task is genuinely
harder, so that specific call escalates. You pay for the big model on the 5% of calls
that need it rather than the 100% that do not.

**3. Escalate on evidence, not on vibes.**

A turn escalates when something measurable says it should: schema validation failed
twice, the tool returned an error the model then ignored, confidence in a structured
self-check came back low. Escalation is a state transition with a reason attached, so
you can count them and see where your prompts are weak.

**4. Let the tool layer degrade instead of dying.**

A registry where every tool declares its dependencies. If a driver is missing, the tool
still appears in the catalogue and returns a clear "unavailable, install X" when called.
The alternative — an import error at startup because one optional integration is absent
— makes a large tool surface impossible to operate.

**5. Never break the wire format.**

The orchestrator emits a fixed server-sent-event taxonomy: session start, model call
start, tool call start, tool result, finish. Because the events did not change, the
system I built at work cut over from the old framework with a single config change and
no client rewrite. This is unglamorous and it is the reason the migration was possible
at all.

## What is in here

```
src/small_model_agents/
  router.py      tiered model router with escalation policy
  schema.py      validated structured decisions, with repair-on-violation
  tools.py       tool registry with dependency declaration and graceful degradation
  agent.py       the ReAct-style loop: decide -> validate -> execute -> observe
  events.py      the SSE event taxonomy
  evaluation.py  the eval suite, metrics, and the regression gate
  mcp_server.py  exposes the registry over Model Context Protocol
examples/
  research_agent.py   a three-tool agent you can run against a local model
evals/
  baseline.json       committed metrics; CI fails if a change is worse
tests/
  test_escalation.py  the escalation policy, without a model server
  test_evaluation.py  the harness, and proof that the gate can fail
```

It runs against anything OpenAI-compatible — vLLM, Ollama, llama.cpp. There is no
dependency on a hosted provider.

## Running it

```bash
uv sync
export OPENAI_BASE_URL=http://localhost:8000/v1   # your vLLM or Ollama
python -m examples.research_agent "What changed in the CPI print last month?"
```

The tests and the eval suite both run offline; the model client falls back to a stub
when no server is present, which is how the escalation policy is tested without a GPU.

## As an MCP server

```bash
python -m small_model_agents.mcp_server
```

The same registry, exposed over Model Context Protocol, so the tools can be used by any
MCP-aware client rather than only by this loop.

## Measuring it

A claim about reliability that you cannot re-run is an anecdote. `evaluation.py` is the
smallest thing that turns the claim into a measurement.

```bash
# offline, deterministic, no GPU — this is what CI runs
python -m small_model_agents.evaluation --simulate

# against a real server
OPENAI_BASE_URL=http://localhost:8000/v1 python -m small_model_agents.evaluation
```

Twelve tasks, graded mechanically — not by a judge model, which would put another
unversioned dependency inside the measurement. Five metrics, and the gate checks all
five together so that a prompt tweak cannot quietly trade one for another:

| metric | | why it is here |
|---|---|---|
| `pass_rate` | higher better | did the run do the thing |
| `escalation_rate` | lower better | how often the cheap tier was not enough |
| `schema_violation_rate` | lower better | how often the model broke the output contract |
| `mean_model_calls` | lower better | the cost proxy |
| `tool_precision` | higher better | tool calls the task actually wanted |

`schema_violation_rate` is the one people leave out, and it is the one that matters most
when you are pushing work down onto a smaller model. A repair loop that works is
invisible in the output and expensive in tokens — without counting violations there is
no way to tell a good prompt from a bad prompt that happens to self-correct. That is
why the event taxonomy carries a `schema_violation` event.

`evals/baseline.json` is checked in. A run is compared against it and the process exits
non-zero if any metric moved the wrong way by more than its tolerance — tight on
correctness, looser on cost. Regenerating the baseline is a deliberate, reviewable
commit rather than something a test can do to itself.

The suite is deterministic offline. `SimulatedModel` draws its failures from a seeded
hash of the call context, weighted by tier, so escalation has something real to fix and
two runs produce byte-identical metrics. It is a stand-in for testing the harness, not a
claim about any model. The test that matters is
`test_gate_catches_a_worse_model` — an eval that cannot fail is decoration.

Nothing in the suite touches the network or the clock. An eval that can flake is not a
gate.

## What I would not claim

This is a reference implementation, not a framework. It has no distributed execution, no
persistence beyond a run, and no multi-tenant isolation — all three of which a
production deployment needs and all three of which are the boring, necessary work that
a repository like this cannot honestly demonstrate.

The numbers in the opening paragraph come from a production system at my employer and
are not reproducible from this code. What is reproducible here is the design: narrow
validated decisions, tiered routing, evidence-based escalation, and a tool layer that
degrades.

---

**Muhammed Salim K T** — AI/ML engineer, Kochi, India.
[mohdsalimkt@gmail.com](mailto:mohdsalimkt@gmail.com) ·
[LinkedIn](https://linkedin.com/in/muhammed-salim-k-t)
