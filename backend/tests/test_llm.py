"""LLM abstraction tests.

Tested against the fake provider and a mock HTTP transport, never the live API: tests must be
free, offline and reproducible. What is verified is our own translation, validation and retry
logic, not the model's intelligence.
"""

import json

import httpx
import pytest
from pydantic import BaseModel, Field

from app.llm.base import (
    LLMError,
    Message,
    Role,
    StructuredOutputError,
    Usage,
    extract_json,
    generate_structured,
    schema_instruction,
)
from app.llm.fake import FakeLLMProvider, ScriptedTurn, text_turn, tool_turn
from app.llm.gemini import GeminiProvider

#: The provider takes no default model on purpose: a hardcoded one went stale and broke a live
#: run when Google retired it. Tests name one explicitly like production does.
TEST_MODEL = "gemini-test-flash"


class Plan(BaseModel):
    summary: str = Field(max_length=100)
    files: list[str]
    confidence: str


class TestExtractJson:
    def test_plain_json_passes_through(self) -> None:
        assert extract_json('{"a": 1}') == '{"a": 1}'

    def test_strips_code_fences(self) -> None:
        """Models add fences even when told not to."""
        assert extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_strips_unlabelled_fences(self) -> None:
        assert extract_json('```\n{"a": 1}\n```') == '{"a": 1}'

    def test_strips_surrounding_prose(self) -> None:
        text = 'Here is the plan:\n{"a": 1}\nLet me know if you need changes.'

        assert extract_json(text) == '{"a": 1}'

    def test_leaves_unparseable_text_alone(self) -> None:
        """No braces to find, so the caller gets a clear validation error instead."""
        assert extract_json("no json here") == "no json here"


class TestStructuredOutput:
    async def test_valid_response_is_parsed(self) -> None:
        provider = FakeLLMProvider(
            [text_turn('{"summary":"fix validation","files":["a.py"],"confidence":"high"}')]
        )

        result, usage = await generate_structured(
            provider, [Message(role=Role.user, content="plan it")], Plan
        )

        assert result.summary == "fix validation"
        assert usage.calls == 1

    async def test_fenced_response_is_parsed(self) -> None:
        provider = FakeLLMProvider(
            [text_turn('```json\n{"summary":"s","files":[],"confidence":"low"}\n```')]
        )

        result, _ = await generate_structured(
            provider, [Message(role=Role.user, content="plan")], Plan
        )

        assert result.confidence == "low"

    async def test_invalid_response_is_repaired(self) -> None:
        """The validation error is fed back, which works far better than retrying blindly:
        the model is told exactly which field was wrong."""
        provider = FakeLLMProvider(
            [
                text_turn('{"summary":"missing other fields"}'),
                text_turn('{"summary":"second try","files":["a.py"],"confidence":"medium"}'),
            ]
        )

        result, usage = await generate_structured(
            provider, [Message(role=Role.user, content="plan")], Plan
        )

        assert result.summary == "second try"
        assert usage.calls == 2

    async def test_repair_prompt_contains_the_validation_error(self) -> None:
        provider = FakeLLMProvider(
            [
                text_turn("{}"),
                text_turn('{"summary":"ok","files":[],"confidence":"low"}'),
            ]
        )

        await generate_structured(provider, [Message(role=Role.user, content="plan")], Plan)

        repair_prompt = provider.requests[1].messages[-1].content

        assert "not valid" in repair_prompt.lower()
        assert "files" in repair_prompt

    async def test_exhausted_repairs_raise(self) -> None:
        """Better to fail loudly than hand the caller a half-valid object it will act on."""
        provider = FakeLLMProvider([text_turn("not json") for _ in range(5)])

        with pytest.raises(StructuredOutputError, match="Could not obtain valid Plan"):
            await generate_structured(
                provider, [Message(role=Role.user, content="plan")], Plan
            )

    async def test_a_tool_request_instead_of_json_is_pushed_back(self) -> None:
        """At the end of a tool phase the model often reaches for one more tool.

        The transcript is full of function calls, so it continues the pattern even though no
        tools were offered on this request. This happened on the first live run and killed it.
        Telling the model the gathering phase is over recovers it in one turn.
        """
        provider = FakeLLMProvider(
            [
                tool_turn("read_file", {"path": "calc.py"}),
                text_turn('{"summary":"s","files":["calc.py"],"confidence":"high"}'),
            ]
        )

        plan, _usage = await generate_structured(
            provider, [Message(role=Role.user, content="plan")], Plan
        )

        assert plan.summary == "s"

        pushback = provider.requests[1].messages[-1].content
        assert "No tools are available" in pushback
        assert "already gathered" in pushback

    async def test_endless_tool_requests_eventually_fail_with_the_tool_names(self) -> None:
        provider = FakeLLMProvider([tool_turn("read_file", {"path": "a.py"}) for _ in range(5)])

        with pytest.raises(StructuredOutputError) as error:
            await generate_structured(
                provider, [Message(role=Role.user, content="plan")], Plan
            )

        assert "requested tools: read_file" in str(error.value)

    async def test_empty_response_is_reported_as_empty_not_as_bad_json(self) -> None:
        """A reasoning model that spends its whole output budget thinking returns no text.

        This actually happened on the first live run. Reporting it as
        "Invalid JSON: EOF while parsing" sent the investigation in completely the wrong
        direction, and retrying is pointless because the same prompt burns the same budget.
        """
        provider = FakeLLMProvider(
            [ScriptedTurn(text="", output_tokens=4096, finish_reason="MAX_TOKENS")]
        )

        with pytest.raises(StructuredOutputError) as error:
            await generate_structured(
                provider, [Message(role=Role.user, content="plan")], Plan
            )

        message = str(error.value)
        assert "no text" in message
        assert "MAX_TOKENS" in message
        assert "LLM_MAX_OUTPUT_TOKENS" in message

        # One attempt only. Repairing an empty response wastes three calls for nothing.
        assert len(provider.requests) == 1

    async def test_structured_calls_use_low_temperature(self) -> None:
        """For extraction, variety is not a feature."""
        provider = FakeLLMProvider(
            [text_turn('{"summary":"s","files":[],"confidence":"low"}')]
        )

        await generate_structured(provider, [Message(role=Role.user, content="p")], Plan)

        assert provider.requests[0].temperature <= 0.2

    def test_schema_instruction_includes_the_json_schema(self) -> None:
        instruction = schema_instruction(Plan)

        assert "JSON Schema" in instruction
        assert "summary" in instruction
        assert "no code fences" in instruction.lower()


class TestUsage:
    def test_totals_accumulate(self) -> None:
        total = Usage()
        total.add(Usage(input_tokens=100, output_tokens=50, calls=1))
        total.add(Usage(input_tokens=200, output_tokens=80, calls=1))

        assert total.input_tokens == 300
        assert total.total_tokens == 430
        assert total.calls == 2


class TestFakeProvider:
    async def test_returns_scripted_turns_in_order(self) -> None:
        provider = FakeLLMProvider([text_turn("first"), text_turn("second")])

        first = await provider.generate([Message(role=Role.user, content="x")])
        second = await provider.generate([Message(role=Role.user, content="y")])

        assert first.text == "first"
        assert second.text == "second"

    async def test_exhausted_script_raises(self) -> None:
        """A test that scripted too few replies is a test bug, so fail loudly."""
        provider = FakeLLMProvider([])

        with pytest.raises(LLMError, match="script exhausted"):
            await provider.generate([Message(role=Role.user, content="x")])

    async def test_records_requests_for_assertions(self) -> None:
        provider = FakeLLMProvider([text_turn("ok")])

        await provider.generate(
            [Message(role=Role.system, content="be careful"),
             Message(role=Role.user, content="do it")]
        )

        assert "be careful" in provider.last_prompt_text()

    async def test_tool_turn_requests_a_tool(self) -> None:
        provider = FakeLLMProvider([tool_turn("read_file", {"path": "a.py"})])

        response = await provider.generate([Message(role=Role.user, content="x")])

        assert response.wants_tools
        assert response.tool_calls[0].name == "read_file"

    async def test_scripted_error_is_raised(self) -> None:
        provider = FakeLLMProvider([ScriptedTurn(error=LLMError("provider down"))])

        with pytest.raises(LLMError, match="provider down"):
            await provider.generate([Message(role=Role.user, content="x")])


def gemini_transport(
    *,
    text: str = "ok",
    function_call: dict | None = None,
    capture: list[httpx.Request] | None = None,
    statuses: list[int] | None = None,
    payload: dict | None = None,
) -> httpx.MockTransport:
    remaining = list(statuses or [])

    def handler(request: httpx.Request) -> httpx.Response:
        if capture is not None:
            capture.append(request)

        if remaining:
            status = remaining.pop(0)
            if status >= 400:
                return httpx.Response(status, json={"error": {"message": "transient"}})

        if payload is not None:
            return httpx.Response(200, json=payload)

        parts: list[dict] = [{"text": text}]

        if function_call is not None:
            parts.append({"functionCall": function_call})

        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": parts}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 120, "candidatesTokenCount": 45},
            },
        )

    return httpx.MockTransport(handler)


class TestGeminiProvider:
    async def test_requires_an_api_key(self) -> None:
        with pytest.raises(LLMError, match="API key"):
            GeminiProvider(api_key="", model=TEST_MODEL)

    async def test_parses_text_and_usage(self) -> None:
        provider = GeminiProvider(
            api_key="k", model=TEST_MODEL, transport=gemini_transport(text="hello")
        )

        response = await provider.generate([Message(role=Role.user, content="hi")])

        assert response.text == "hello"
        assert response.usage.input_tokens == 120
        assert response.usage.output_tokens == 45
        assert response.usage.calls == 1

    async def test_thinking_tokens_are_counted_as_output(self) -> None:
        """Reasoning tokens are billed, so they belong in the run's cost accounting.

        Leaving them out understates the cost of a run by more than half on a thinking model.
        """
        provider = GeminiProvider(
            api_key="k",
            model=TEST_MODEL,
            transport=gemini_transport(
                payload={
                    "candidates": [
                        {"content": {"parts": [{"text": "answer"}]}, "finishReason": "STOP"}
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 600,
                        "candidatesTokenCount": 300,
                        "thoughtsTokenCount": 950,
                    },
                }
            ),
        )

        response = await provider.generate([Message(role=Role.user, content="hi")])

        assert response.usage.output_tokens == 1250

    async def test_configured_output_budget_is_sent(self) -> None:
        captured: list[httpx.Request] = []
        provider = GeminiProvider(
            api_key="k",
            model=TEST_MODEL,
            max_output_tokens=12_345,
            transport=gemini_transport(capture=captured),
        )

        await provider.generate([Message(role=Role.user, content="hi")])

        body = json.loads(captured[0].content)
        assert body["generationConfig"]["maxOutputTokens"] == 12_345

    async def test_parses_a_function_call(self) -> None:
        provider = GeminiProvider(
            api_key="k",
            model=TEST_MODEL,
            transport=gemini_transport(
                text="", function_call={"name": "read_file", "args": {"path": "app/main.py"}}
            ),
        )

        response = await provider.generate([Message(role=Role.user, content="find the bug")])

        assert response.wants_tools
        assert response.tool_calls[0].name == "read_file"
        assert response.tool_calls[0].arguments == {"path": "app/main.py"}

    async def test_system_message_becomes_system_instruction(self) -> None:
        """Gemini has no system role: it takes a separate top-level field. Getting this wrong
        silently demotes the system prompt to an ordinary user message."""
        captured: list[httpx.Request] = []
        provider = GeminiProvider(
            api_key="k", model=TEST_MODEL, transport=gemini_transport(capture=captured)
        )

        await provider.generate(
            [
                Message(role=Role.system, content="you are careful"),
                Message(role=Role.user, content="hello"),
            ]
        )

        body = captured[0].read().decode()

        assert "systemInstruction" in body
        assert "you are careful" in body

    async def test_assistant_role_is_renamed_to_model(self) -> None:
        captured: list[httpx.Request] = []
        provider = GeminiProvider(
            api_key="k", model=TEST_MODEL, transport=gemini_transport(capture=captured)
        )

        await provider.generate(
            [
                Message(role=Role.user, content="hi"),
                Message(role=Role.assistant, content="hello back"),
            ]
        )

        # Parsed rather than substring-matched: asserting on raw JSON text couples the
        # test to the serialiser's whitespace, which is not behaviour we care about.
        body = json.loads(captured[0].read().decode())

        assert [item["role"] for item in body["contents"]] == ["user", "model"]

    async def test_tool_result_is_sent_as_function_response(self) -> None:
        captured: list[httpx.Request] = []
        provider = GeminiProvider(
            api_key="k", model=TEST_MODEL, transport=gemini_transport(capture=captured)
        )

        await provider.generate(
            [
                Message(role=Role.user, content="find it"),
                Message(role=Role.tool, tool_name="read_file", content="file contents"),
            ]
        )

        assert "functionResponse" in captured[0].read().decode()

    async def test_tools_are_advertised_as_function_declarations(self) -> None:
        captured: list[httpx.Request] = []
        provider = GeminiProvider(
            api_key="k", model=TEST_MODEL, transport=gemini_transport(capture=captured)
        )

        await provider.generate(
            [Message(role=Role.user, content="x")],
            tools=[{"name": "read_file", "description": "d", "parameters": {}}],
        )

        assert "functionDeclarations" in captured[0].read().decode()

    async def test_transient_failure_is_retried(self, monkeypatch) -> None:  # noqa: ANN001
        async def no_sleep(_seconds: float) -> None:
            return None

        monkeypatch.setattr("app.llm.gemini.asyncio.sleep", no_sleep)

        provider = GeminiProvider(
            api_key="k",
            model=TEST_MODEL,
            max_attempts=3,
            transport=gemini_transport(text="recovered", statuses=[503, 200]),
        )

        response = await provider.generate([Message(role=Role.user, content="x")])

        assert response.text == "recovered"

    async def test_permanent_failure_is_not_retried(self) -> None:
        captured: list[httpx.Request] = []
        provider = GeminiProvider(
            api_key="k",
            model=TEST_MODEL,
            max_attempts=3,
            transport=gemini_transport(capture=captured, statuses=[400]),
        )

        with pytest.raises(LLMError, match="400"):
            await provider.generate([Message(role=Role.user, content="x")])

        assert len(captured) == 1

    async def test_empty_candidates_raises_with_the_reason(self) -> None:
        """Usually a safety block. Surfacing the feedback beats an opaque IndexError."""
        provider = GeminiProvider(
            api_key="k",
            model=TEST_MODEL,
            transport=gemini_transport(
                payload={"promptFeedback": {"blockReason": "SAFETY"}}
            ),
        )

        with pytest.raises(LLMError, match="no candidates"):
            await provider.generate([Message(role=Role.user, content="x")])


