"""Expose the tool registry over Model Context Protocol.

The same registry the loop uses, made available to any MCP-aware client. This is
stdio JSON-RPC with no third-party dependency, so it runs anywhere Python does.
"""
from __future__ import annotations

import json
import sys
from typing import Any

from .tools import Registry

PROTOCOL_VERSION = "2024-11-05"


def _result(req_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _error(req_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def handle(registry: Registry, msg: dict[str, Any]) -> dict[str, Any] | None:
    method, req_id = msg.get("method"), msg.get("id")

    if method == "initialize":
        return _result(req_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "small-model-agents", "version": "0.1.0"},
        })

    if method in ("notifications/initialized", "initialized"):
        return None  # notification: no reply

    if method == "tools/list":
        return _result(req_id, {"tools": [
            {
                "name": t.name,
                "description": t.description + (
                    "" if t.available else f" [unavailable: needs {', '.join(t.missing)}]"
                ),
                "inputSchema": t.parameters,
            }
            for t in registry.tools.values()
        ]})

    if method == "tools/call":
        params = msg.get("params") or {}
        ok, result = registry.call(params.get("name", ""), params.get("arguments") or {})
        return _result(req_id, {
            "content": [{"type": "text", "text": str(result)}],
            "isError": not ok,
        })

    if method == "ping":
        return _result(req_id, {})

    return _error(req_id, -32601, f"method not found: {method}")


def serve(registry: Registry, stdin=sys.stdin, stdout=sys.stdout) -> None:
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        reply = handle(registry, msg)
        if reply is not None:
            stdout.write(json.dumps(reply) + "\n")
            stdout.flush()


def default_registry() -> Registry:
    from examples.research_agent import build_registry
    return build_registry()


if __name__ == "__main__":
    serve(default_registry())
