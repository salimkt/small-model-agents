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
  mcp_server.py  exposes the registry over Model Context Protocol
examples/
  research_agent.py   a three-tool agent you can run against a local model
tests/
  test_escalation.py  the escalation policy, without a model server
```

It runs against anything OpenAI-compatible — vLLM, Ollama, llama.cpp. There is no
dependency on a hosted provider.

## Running it

```bash
uv sync
export OPENAI_BASE_URL=http://localhost:8000/v1   # your vLLM or Ollama
python -m examples.research_agent "What changed in the CPI print last month?"
```

The tests run offline; the model client falls back to a stub when no server is present,
which is how the escalation policy is tested without a GPU.

## As an MCP server

```bash
python -m small_model_agents.mcp_server
```

The same registry, exposed over Model Context Protocol, so the tools can be used by any
MCP-aware client rather than only by this loop.

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
