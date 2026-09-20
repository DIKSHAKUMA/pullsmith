"""LLM provider contract.

The rest of the application never imports a vendor SDK. It depends on this interface, which
exposes exactly three things the agent needs:

* **text generation** - free-form output, used rarely
* **structured output** - output validated against a Pydantic schema, used for anything the
  code then acts on
* **tool calling** - the model requests a function; our code decides whether to run it

Structured output is the important one. A plan, an issue analysis or a failure diagnosis all
get parsed by code, so free-form prose is unusable. Asking for JSON and validating it turns
"the model said something" into "the model said something my code can rely on".
"""

import json
import logging
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)


class Role(StrEnum):
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


@dataclass
class Message:
    role: Role
    content: str

    #: Set when the assistant asked for a tool, or when we return a tool's output.
    tool_call_id: str | None = None
    tool_name: str | None = None


@dataclass
class ToolCall:
    """A tool the model asked for. Not yet executed, and possibly never."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Usage:
    """Token and cost accounting.

    Tracked per call and summed per run so a run can be stopped when it becomes expensive.
    An agent with a repair loop can otherwise burn money quietly.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.calls += other.calls


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = "stop"

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LLMError(RuntimeError):
    pass


class StructuredOutputError(LLMError):
    """The model's output could not be parsed or validated against the schema."""


class QuotaExceededError(LLMError):
    """The provider refused on quota, after retries.

    Its own type because the caller's response is different: a daily quota cannot be waited out
    inside a run, so the run should stop and say so rather than look like a bug in the agent.
    """


class LLMProvider(Protocol):
    @property
    def model(self) -> str: ...

    async def generate(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        #: None means "use the provider's configured budget". An explicit value overrides it.
        max_output_tokens: int | None = None,
    ) -> LLMResponse: ...


def extract_json(text: str) -> str:
    """Pulls a JSON object out of model output.

    Even when asked for JSON only, models wrap it in ```json fences or add a sentence of
    explanation. Stripping that here is cheaper and more reliable than another API round
    trip to ask for a correction.
    """
    stripped = text.strip()

    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1).strip()

    # Fall back to the outermost braces if prose surrounds the object.
    if not stripped.startswith("{"):
        first = stripped.find("{")
        last = stripped.rfind("}")

        if first != -1 and last > first:
            stripped = stripped[first : last + 1]

    return stripped


async def generate_structured[SchemaT: BaseModel](
    provider: LLMProvider,
    messages: list[Message],
    schema: type[SchemaT],
    *,
    temperature: float = 0.1,
    max_output_tokens: int | None = None,
    max_repair_attempts: int = 2,
) -> tuple[SchemaT, Usage]:
    """Generates output validated against ``schema``.

    On failure the validation error is fed back to the model as a message, which is far more
    effective than retrying the same prompt: the model is told precisely which field was
    wrong. Repairs are bounded, and exhausting them raises rather than returning a
    half-valid object, because the caller is about to act on this.

    Temperature defaults low: for structured extraction, variety is not a feature.
    """
    conversation = list(messages)
    total = Usage()
    last_error = ""

    for attempt in range(1, max_repair_attempts + 2):
        response = await provider.generate(
            conversation, temperature=temperature, max_output_tokens=max_output_tokens
        )
        total.add(response.usage)

        if not response.text.strip():
            if response.tool_calls and attempt <= max_repair_attempts:
                # This is the common case at the end of a tool-calling phase: the transcript
                # is full of function calls, so the model reaches for another tool instead of
                # answering — even though no tools were offered on this request. Telling it
                # plainly that the gathering phase is over recovers it in one turn, which is
                # much cheaper than abandoning the run.
                requested = ", ".join(call.name for call in response.tool_calls)
                last_error = f"model requested tools ({requested}) instead of returning JSON"

                logger.warning("structured output: %s, pushing back", last_error)

                conversation = [
                    *conversation,
                    Message(role=Role.assistant, content=f"(requested {requested})"),
                    Message(
                        role=Role.user,
                        content=(
                            "No tools are available in this step, and no further investigation "
                            "is possible. Answer using only what you have already gathered. "
                            "Reply with a single JSON object and nothing else."
                        ),
                    ),
                ]
                continue

            # Distinguished from malformed JSON on purpose. A reasoning model that spends its
            # whole output budget thinking returns nothing, and reporting that as
            # "Invalid JSON: EOF while parsing" sends you looking in the wrong place.
            raise StructuredOutputError(
                f"Model returned no text for {schema.__name__} after {attempt} attempt(s) "
                f"(finish_reason={response.finish_reason}, "
                f"{response.usage.output_tokens} output tokens charged"
                + (
                    f", requested tools: {', '.join(c.name for c in response.tool_calls)}"
                    if response.tool_calls
                    else ""
                )
                + "). If finish_reason is MAX_TOKENS the output budget was consumed before any "
                "answer was produced; raise LLM_MAX_OUTPUT_TOKENS."
            )

        payload = extract_json(response.text)

        try:
            return schema.model_validate_json(payload), total
        except (ValidationError, ValueError) as exc:
            last_error = str(exc)[:600]
            logger.warning(
                "structured output rejected on attempt %s/%s: %s",
                attempt,
                max_repair_attempts + 1,
                last_error[:200],
            )

            if attempt > max_repair_attempts:
                break

            conversation = [
                *conversation,
                Message(role=Role.assistant, content=response.text[:2000]),
                Message(
                    role=Role.user,
                    content=(
                        "That response was not valid for the required schema. "
                        f"Errors:\n{last_error}\n\n"
                        "Reply with corrected JSON only, no prose and no code fences."
                    ),
                ),
            ]

    raise StructuredOutputError(
        f"Could not obtain valid {schema.__name__} after "
        f"{max_repair_attempts + 1} attempts: {last_error}"
    )


def schema_instruction(schema: type[BaseModel]) -> str:
    """The instruction appended to a prompt describing the required JSON shape."""
    return (
        "Reply with a single JSON object and nothing else. No prose, no code fences.\n"
        "It must satisfy this JSON Schema:\n"
        f"{json.dumps(schema.model_json_schema(), indent=2)}"
    )
