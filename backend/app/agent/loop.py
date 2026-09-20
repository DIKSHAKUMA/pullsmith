"""The tool-calling loop.

This is what makes the system an *agent* rather than a single prompt: the model can request
information, see the result, and decide what to do next.

```
model → "read app/profile.py"     → we execute → result returned
model → "search for validate"     → we execute → result returned
model → final structured answer   → loop ends
```

Three limits are non-negotiable, and each exists because of a specific failure mode:

* **max_steps** - a model can loop forever asking for the same file.
* **tool call budget** - carried on the context and shared across the whole run.
* **token budget** - a long loop grows the conversation, and every step re-sends it, so cost
  grows quadratically if unchecked.

Every tool call and result is reported through a callback so it lands on the run timeline.
The user sees which files the agent looked at, not the model's reasoning.
"""

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from app.agent.tools.base import ToolContext, ToolRegistry, ToolResult
from app.llm.base import (
    LLMProvider,
    Message,
    Role,
    Usage,
    generate_structured,
    schema_instruction,
)

logger = logging.getLogger(__name__)

#: Callback signature for reporting activity to the timeline.
ObserverType = Callable[[str, dict[str, Any]], Awaitable[None]]


@dataclass
class LoopLimits:
    max_steps: int = 12
    max_tokens: int = 200_000


@dataclass
class LoopOutcome:
    """What the loop gathered, and what it cost."""

    transcript: list[Message]
    usage: Usage = field(default_factory=Usage)
    steps_used: int = 0
    tool_calls: list[tuple[str, ToolResult]] = field(default_factory=list)
    stopped_because: str = "model_finished"

    def observations(self) -> str:
        """Tool results, as text to hand to a follow-up prompt."""
        return "\n\n".join(
            f"[{name}]\n{result.output}" for name, result in self.tool_calls if result.ok
        )


async def run_tool_loop(
    *,
    provider: LLMProvider,
    registry: ToolRegistry,
    context: ToolContext,
    messages: list[Message],
    limits: LoopLimits | None = None,
    allow_mutating: bool = False,
    observer: ObserverType | None = None,
) -> LoopOutcome:
    """Lets the model gather information with tools until it stops asking.

    ``allow_mutating`` defaults to False so an exploration phase cannot accidentally edit the
    repository. Write access is opt-in per phase, not the default.
    """
    limits = limits or LoopLimits()
    conversation = list(messages)
    outcome = LoopOutcome(transcript=conversation)

    async def report(kind: str, payload: dict[str, Any]) -> None:
        if observer is not None:
            await observer(kind, payload)

    for step in range(1, limits.max_steps + 1):
        outcome.steps_used = step

        response = await provider.generate(
            conversation, tools=registry.schemas(include_mutating=allow_mutating)
        )
        outcome.usage.add(response.usage)

        if outcome.usage.total_tokens > limits.max_tokens:
            outcome.stopped_because = "token_budget_exceeded"
            logger.warning("tool loop stopped: token budget exceeded")
            await report("warning", {"message": "Token budget reached while exploring"})
            return outcome

        if not response.wants_tools:
            # The model is done gathering and has produced an answer.
            conversation.append(Message(role=Role.assistant, content=response.text))
            outcome.stopped_because = "model_finished"
            return outcome

        conversation.append(
            Message(
                role=Role.assistant,
                content=response.text or "(requesting tools)",
            )
        )

        for call in response.tool_calls:
            result = await registry.execute(
                context, call.name, call.arguments, allow_mutating=allow_mutating
            )
            outcome.tool_calls.append((call.name, result))

            # Operational summary only: which tool, on what, did it work. Never the model's
            # reasoning, which is not shown to users.
            await report(
                "tool_call",
                {
                    "tool": call.name,
                    "arguments": call.arguments,
                    "ok": result.ok,
                    "error_code": result.error_code,
                    "duration_ms": result.duration_ms,
                },
            )

            conversation.append(
                Message(
                    role=Role.tool,
                    tool_call_id=call.id,
                    tool_name=call.name,
                    content=result.output,
                )
            )

            if result.error_code == "TOOL_BUDGET_EXCEEDED":
                outcome.stopped_because = "tool_budget_exceeded"
                logger.warning("tool loop stopped: tool budget exceeded")
                return outcome

    outcome.stopped_because = "max_steps_reached"
    logger.warning("tool loop stopped after %s steps", limits.max_steps)
    await report("warning", {"message": f"Stopped exploring after {limits.max_steps} steps"})
    return outcome


async def conclude_with_schema[SchemaT: BaseModel](
    *,
    provider: LLMProvider,
    outcome: LoopOutcome,
    schema: type[SchemaT],
    instruction: str,
) -> tuple[SchemaT, Usage]:
    """Turns what the loop gathered into a validated object.

    Deliberately a *separate* call from the exploration loop. Asking a model to both decide
    which tool to call next and emit strict JSON in one turn makes both jobs worse. Splitting
    them means the schema request is a clean, focused prompt with the observations already in
    hand.
    """
    messages = [
        *outcome.transcript,
        Message(
            role=Role.user,
            content=f"{instruction}\n\n{schema_instruction(schema)}",
        ),
    ]

    result, usage = await generate_structured(provider, messages, schema)
    outcome.usage.add(usage)
    return result, usage
