"""Tool registry with declared dependencies and graceful degradation.

A tool whose optional driver is missing must still appear in the catalogue and
must fail with a useful message when called. The alternative — an ImportError at
startup because one integration of seventy is unavailable — makes a large tool
surface impossible to operate.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Any, Callable


def _absent(module: str) -> bool:
    """True if `module` cannot be imported.

    `find_spec` does not simply return None for a dotted name whose parent
    package is missing — it raises ModuleNotFoundError from the parent import.
    A registry that crashes on `requires=("snowflake.connector",)` defeats the
    whole purpose of declaring dependencies instead of importing them.
    """
    try:
        return importlib.util.find_spec(module) is None
    except (ImportError, AttributeError, ValueError):
        return True


class ToolUnavailable(RuntimeError):
    pass


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]           # JSON Schema for the arguments
    fn: Callable[..., Any]
    requires: tuple[str, ...] = ()       # importable module names

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(m for m in self.requires if _absent(m))

    @property
    def available(self) -> bool:
        return not self.missing

    def __call__(self, **kwargs: Any) -> Any:
        if self.missing:
            raise ToolUnavailable(
                f"{self.name} needs {', '.join(self.missing)} "
                f"(pip install {' '.join(self.missing)})"
            )
        return self.fn(**kwargs)

    def to_openai_schema(self) -> dict[str, Any]:
        note = "" if self.available else f" [UNAVAILABLE: needs {', '.join(self.missing)}]"
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description + note,
                "parameters": self.parameters,
            },
        }


@dataclass
class Registry:
    tools: dict[str, Tool] = field(default_factory=dict)

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        requires: tuple[str, ...] = (),
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.tools[name] = Tool(name, description, parameters, fn, requires)
            return fn

        return deco

    def catalogue(self, *, include_unavailable: bool = True) -> list[dict[str, Any]]:
        """The catalogue the model sees.

        Unavailable tools are included by default and marked. Hiding them makes
        the model's view of the world change depending on what happens to be
        installed, which is a debugging nightmare.
        """
        return [
            t.to_openai_schema()
            for t in self.tools.values()
            if include_unavailable or t.available
        ]

    def call(self, name: str, arguments: dict[str, Any]) -> tuple[bool, Any]:
        """Returns (ok, result). Errors are values, not exceptions — the loop
        feeds them back to the model as observations."""
        tool = self.tools.get(name)
        if tool is None:
            return False, f"no such tool: {name}. Available: {sorted(self.tools)}"
        try:
            return True, tool(**arguments)
        except ToolUnavailable as exc:
            return False, str(exc)
        except TypeError as exc:
            return False, f"bad arguments for {name}: {exc}"
        except Exception as exc:  # noqa: BLE001 - deliberately broad
            return False, f"{name} failed: {type(exc).__name__}: {exc}"
