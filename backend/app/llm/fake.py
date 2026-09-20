"""A scripted LLM provider for tests.

Testing an agent against a real model is slow, costs money, and is not reproducible: the
same prompt can yield different output, so a failing test tells you nothing about whether
your code or the model changed.

This provider returns a pre-written list of responses in order. That makes it possible to
test the parts that actually contain our logic:

* the tool-calling loop, including a model that asks for a tool that does not exist
* structured-output repair, by scripting an invalid response followed by a valid one
* step limits, by scripting a model that never stops asking for tools
* usage accounting

It records every request, so a test can assert *what the agent asked*, not only what it did
with the answer.
"""

from dataclasses import dataclass, field
from typing import Any

from app.llm.base import LLMError, LLMResponse, Message, ToolCall, Usage


@dataclass
class ScriptedTurn:
    """One reply. Either text, or tool calls, or both."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 100
    output_tokens: int = 50
    finish_reason: str = "stop"

    #: Raised instead of returning, for testing failure handling.
    error: Exception | None = None


@dataclass
class RecordedRequest:
    messages: list[Message]
    tools: list[dict[str, Any]] | None
    temperature: float


class FakeLLMProvider:
    def __init__(self, turns: list[ScriptedTurn], *, model: str = "fake-llm-v1") -> None:
        self._turns = list(turns)
        self._model = model
        self.requests: list[RecordedRequest] = []

    @property
    def model(self) -> str:
        return self._model

    @property
    def turns_remaining(self) -> int:
        return len(self._turns)

    def last_prompt_text(self) -> str:
        """All text sent in the most recent request, for asserting prompt construction."""
        if not self.requests:
            return ""

        return "\n".join(message.content for message in self.requests[-1].messages)

    async def generate(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        self.requests.append(
            RecordedRequest(messages=list(messages), tools=tools, temperature=temperature)
        )

        if not self._turns:
            # Running out means the test scripted fewer replies than the code needed, which
            # is a test bug worth failing loudly on rather than returning empty text.
            raise LLMError("FakeLLMProvider script exhausted")

        turn = self._turns.pop(0)

        if turn.error is not None:
            raise turn.error

        return LLMResponse(
            text=turn.text,
            tool_calls=list(turn.tool_calls),
            usage=Usage(
                input_tokens=turn.input_tokens,
                output_tokens=turn.output_tokens,
                calls=1,
            ),
            finish_reason=turn.finish_reason,
        )


def text_turn(text: str) -> ScriptedTurn:
    return ScriptedTurn(text=text)


def tool_turn(name: str, arguments: dict[str, Any], *, call_id: str = "call-1") -> ScriptedTurn:
    return ScriptedTurn(tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)])
