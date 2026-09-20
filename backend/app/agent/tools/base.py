"""Tool registry and execution contract.

The model never runs anything. It *requests* a tool by name with arguments, and this layer
decides whether that request is allowed:

```
model output  →  name + raw arguments
                 ↓  schema validation (Pydantic)
                 ↓  safety checks (paths, allowlists)
                 ↓  execution with a timeout
                 ↓  ToolResult, recorded as an event
```

Every step is server-side. A malformed or malicious tool call fails validation before any
code runs, and the failure is returned to the model as an ordinary error so it can correct
itself rather than crashing the run.

Each tool also declares a JSON schema, which is what gets sent to the model as its list of
available functions. One definition drives both validation and advertisement, so they cannot
drift apart.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from app.agent.safety import SafetyError

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 30


@dataclass
class ToolContext:
    """Everything a tool is allowed to know about the run it belongs to."""

    workspace: Path
    run_id: str
    snapshot_id: str | None = None

    #: Incremented by the executor. The orchestrator stops the run when the cap is hit, so
    #: a confused agent cannot loop forever at our expense.
    tool_calls_used: int = 0
    max_tool_calls: int = 60

    #: Populated by the write tools. The review diff is built from what was actually written
    #: rather than from what the model said it did.
    changes: list[Any] = field(default_factory=list)


@dataclass
class ToolResult:
    ok: bool
    output: str
    data: dict[str, Any] = field(default_factory=dict)
    error_code: str | None = None
    duration_ms: int = 0

    @classmethod
    def success(cls, output: str, **data: Any) -> "ToolResult":
        return cls(ok=True, output=output, data=data)

    @classmethod
    def failure(cls, code: str, message: str) -> "ToolResult":
        return cls(ok=False, output=message, error_code=code)


@dataclass
class Tool:
    """One capability the agent may request."""

    name: str
    description: str
    arguments: type[BaseModel]

    #: The second parameter is `Any` rather than `BaseModel` on purpose. Each handler accepts its
    #: own args model, and a callable taking a *subclass* is not a subtype of one taking the base
    #: class — function parameters are contravariant. The alternative is making `Tool` generic in
    #: its argument type, which spreads a type variable through the registry and the executor for
    #: no safety gain: `execute` validates `raw_arguments` against `self.arguments` immediately
    #: before calling, so the handler cannot receive the wrong model at runtime.
    handler: Callable[[ToolContext, Any], Awaitable[ToolResult]]

    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS

    #: Marks tools that change the repository. Used to keep read-only phases read-only.
    mutating: bool = False

    def json_schema(self) -> dict[str, Any]:
        """The function declaration advertised to the model."""
        schema = self.arguments.model_json_schema()
        schema.pop("title", None)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": schema,
        }


class ToolRegistry:
    """The set of tools available, and the only way to invoke one."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")

        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self, *, include_mutating: bool = True) -> list[dict[str, Any]]:
        """Function declarations for the model.

        Excluding mutating tools is how a read-only phase such as planning is enforced: the
        model is never told those tools exist, and the executor refuses them anyway.
        """
        return [
            tool.json_schema()
            for tool in sorted(self._tools.values(), key=lambda item: item.name)
            if include_mutating or not tool.mutating
        ]

    async def execute(
        self,
        context: ToolContext,
        name: str,
        raw_arguments: dict[str, Any],
        *,
        allow_mutating: bool = True,
    ) -> ToolResult:
        """Validates and runs one tool call.

        Errors are returned as failed results rather than raised, because the caller is a
        language model that should be given a chance to correct itself. Only a genuine
        programming fault propagates.
        """
        started = time.perf_counter()

        def finish(result: ToolResult) -> ToolResult:
            result.duration_ms = int((time.perf_counter() - started) * 1000)
            logger.info(
                "tool %s -> %s in %sms",
                name,
                "ok" if result.ok else f"error:{result.error_code}",
                result.duration_ms,
            )
            return result

        if context.tool_calls_used >= context.max_tool_calls:
            return finish(
                ToolResult.failure(
                    "TOOL_BUDGET_EXCEEDED",
                    f"Tool call budget of {context.max_tool_calls} exhausted",
                )
            )

        tool = self._tools.get(name)

        if tool is None:
            # A hallucinated tool name is a normal occurrence, not an exception.
            return finish(
                ToolResult.failure(
                    "UNKNOWN_TOOL",
                    f"No such tool: {name}. Available: {', '.join(self.names())}",
                )
            )

        if tool.mutating and not allow_mutating:
            return finish(
                ToolResult.failure(
                    "TOOL_NOT_PERMITTED",
                    f"{name} modifies the repository and is not permitted in this phase",
                )
            )

        try:
            arguments = tool.arguments.model_validate(raw_arguments)
        except ValidationError as exc:
            # Returned verbatim so the model can see exactly which field was wrong.
            return finish(
                ToolResult.failure(
                    "INVALID_ARGUMENTS",
                    f"Arguments rejected for {name}: {exc.errors(include_url=False)}",
                )
            )

        context.tool_calls_used += 1

        try:
            result = await asyncio.wait_for(
                tool.handler(context, arguments), timeout=tool.timeout_seconds
            )
        except TimeoutError:
            return finish(
                ToolResult.failure(
                    "TOOL_TIMEOUT", f"{name} exceeded {tool.timeout_seconds}s and was stopped"
                )
            )
        except SafetyError as exc:
            # A boundary violation. Reported, never retried, and visible in the event log.
            logger.warning("safety refusal in %s: %s", name, exc)
            return finish(ToolResult.failure("SAFETY_REFUSED", str(exc)))
        except Exception as exc:
            logger.exception("tool %s raised", name)
            return finish(
                ToolResult.failure("TOOL_ERROR", f"{type(exc).__name__}: {exc}")
            )

        return finish(result)
