"""Gemini provider.

Translates our provider-neutral types into Gemini's request shape and back. Two details in
that translation are easy to get wrong:

* Gemini has no ``system`` role. A system instruction goes in a separate top-level
  ``systemInstruction`` field, not in the message list.
* Gemini calls messages "contents", the assistant role "model", and tool results
  ``functionResponse`` parts.

Keeping those quirks inside this file is the entire point of the abstraction.
"""

import asyncio
import logging
from typing import Any

import httpx

from app.llm.base import (
    LLMError,
    LLMResponse,
    Message,
    QuotaExceededError,
    Role,
    ToolCall,
    Usage,
)

logger = logging.getLogger(__name__)

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class GeminiProvider:
    def __init__(
        self,
        *,
        api_key: str,
        # No default on purpose. Hardcoding one cost a failed end-to-end run when
        # gemini-2.5-flash was retired for new API keys: the model name is a moving target and
        # belongs in configuration, not in a constructor default nobody revisits.
        model: str,
        #: Thinking tokens are charged against this, so it must cover reasoning *and* the
        #: answer. Too small and the model returns nothing at all.
        max_output_tokens: int = 16_384,
        timeout: float = 120.0,
        max_attempts: int = 3,
        requests_per_minute: int = 0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise LLMError("Gemini API key is required")

        self._api_key = api_key
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._timeout = timeout
        self._max_attempts = max_attempts
        self._transport = transport

        #: Pacing beats retrying: a rejected request still counts against quota.
        self._min_interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self._last_request_at = 0.0

    @property
    def model(self) -> str:
        return self._model

    def _to_contents(self, messages: list[Message]) -> tuple[list[dict], dict | None]:
        """Converts our messages into Gemini's ``contents`` plus a system instruction."""
        contents: list[dict] = []
        system_parts: list[str] = []

        for message in messages:
            if message.role is Role.system:
                system_parts.append(message.content)
                continue

            if message.role is Role.tool:
                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": message.tool_name or "tool",
                                    "response": {"result": message.content},
                                }
                            }
                        ],
                    }
                )
                continue

            contents.append(
                {
                    "role": "model" if message.role is Role.assistant else "user",
                    "parts": [{"text": message.content}],
                }
            )

        system_instruction = (
            {"parts": [{"text": "\n\n".join(system_parts)}]} if system_parts else None
        )
        return contents, system_instruction

    async def _pace(self) -> None:
        if self._min_interval <= 0.0:
            return

        loop = asyncio.get_running_loop()
        wait = self._min_interval - (loop.time() - self._last_request_at)

        if wait > 0:
            await asyncio.sleep(wait)

        self._last_request_at = loop.time()

    async def generate(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        contents, system_instruction = self._to_contents(messages)

        body: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens or self._max_output_tokens,
            },
        }

        if system_instruction:
            body["systemInstruction"] = system_instruction

        if tools:
            body["tools"] = [{"functionDeclarations": tools}]

        payload = await self._post(f"/models/{self._model}:generateContent", body)
        return self._parse(payload)

    async def _post(self, path: str, body: dict) -> dict:
        delay = 2.0

        async with httpx.AsyncClient(
            timeout=self._timeout, transport=self._transport
        ) as client:
            for attempt in range(1, self._max_attempts + 1):
                await self._pace()

                try:
                    response = await client.post(
                        f"{GEMINI_BASE}{path}",
                        headers={
                            "x-goog-api-key": self._api_key,
                            "Content-Type": "application/json",
                        },
                        json=body,
                    )
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    if attempt == self._max_attempts:
                        raise LLMError(f"transport failure: {type(exc).__name__}") from exc

                    await asyncio.sleep(delay)
                    delay *= 2
                    continue

                if response.status_code in RETRYABLE_STATUS and attempt < self._max_attempts:
                    wait = float(response.headers.get("Retry-After", delay))
                    logger.warning(
                        "llm request returned %s, retrying in %.1fs (attempt %s/%s)",
                        response.status_code,
                        wait,
                        attempt,
                        self._max_attempts,
                    )
                    await asyncio.sleep(wait)
                    delay *= 2
                    continue

                if response.status_code >= 400:
                    # Truncated: an error body can echo prompt content.
                    if response.status_code == 429:
                        # Retries are already exhausted here. A per-minute limit would have
                        # cleared by now, so this is a quota ceiling: reported as itself so the
                        # run says "out of quota" instead of "something went wrong".
                        raise QuotaExceededError(
                            f"provider quota exceeded: {response.text[:300]}"
                        )

                    raise LLMError(
                        f"llm request failed with {response.status_code}: "
                        f"{response.text[:300]}"
                    )

                return response.json()

        raise LLMError("retry loop exhausted")

    def _parse(self, payload: dict) -> LLMResponse:
        candidates = payload.get("candidates") or []

        if not candidates:
            # Usually a safety block; the reason is worth surfacing rather than hiding.
            feedback = payload.get("promptFeedback", {})
            raise LLMError(f"model returned no candidates: {feedback}")

        candidate = candidates[0]
        parts = candidate.get("content", {}).get("parts") or []

        texts: list[str] = []
        tool_calls: list[ToolCall] = []

        for index, part in enumerate(parts):
            if "text" in part:
                texts.append(part["text"])

            if "functionCall" in part:
                call = part["functionCall"]
                tool_calls.append(
                    ToolCall(
                        # Gemini does not supply call ids, so one is synthesised for
                        # correlating the response back to the request.
                        id=f"call-{index}",
                        name=call.get("name", ""),
                        arguments=call.get("args") or {},
                    )
                )

        metadata = payload.get("usageMetadata", {})
        finish_reason = candidate.get("finishReason", "stop")

        # Gemini 3 models reason before answering, and those thinking tokens are billed against
        # maxOutputTokens. A budget that is ample for the answer alone can therefore be spent
        # entirely on reasoning, returning finishReason=MAX_TOKENS with no text at all. Counting
        # them separately is what makes that diagnosable instead of looking like an empty reply.
        thinking_tokens = metadata.get("thoughtsTokenCount", 0)
        text = "".join(texts)

        if not text and not tool_calls:
            logger.warning(
                "model returned no content: finish_reason=%s, thinking_tokens=%s, "
                "output_tokens=%s. If finish_reason is MAX_TOKENS, raise LLM_MAX_OUTPUT_TOKENS.",
                finish_reason,
                thinking_tokens,
                metadata.get("candidatesTokenCount", 0),
            )

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            usage=Usage(
                input_tokens=metadata.get("promptTokenCount", 0),
                # Thinking tokens are charged, so they belong in the run's cost accounting.
                output_tokens=metadata.get("candidatesTokenCount", 0) + thinking_tokens,
                calls=1,
            ),
            finish_reason=finish_reason,
        )
