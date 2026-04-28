"""LLM clients for Groq (extraction / IVR side) and Anthropic (rep side).

Two backends with intentionally different shapes:

- `GroqLLMClient`: Llama 4 Scout / Qwen3 via Groq's chat completions. One-shot
  `complete_free_form` / `complete_structured(system, user, schema)`. Used by the
  retiring M3/M4 graph; M5'/D1 will get a separate tool-calling Groq client.
- `AnthropicRepClient`: Claude Haiku 4.5 via Anthropic's `messages.parse()` —
  multi-turn `complete_structured(system, history, schema)` for rep-mode
  conversation. Persona prompt marked for prompt caching.

`@observe` makes the raw SDK calls visible to Langfuse — they bypass
`langchain_core`, so the LangGraph callback handler alone can't see them.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, cast

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam
from groq import Groq
from langfuse import observe
from pydantic import BaseModel

from agent.logging_config import log
from agent.observability import enrich_current_generation

EXTRACTION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
FREE_FORM_MODEL = "qwen/qwen3-32b"
REP_MODEL = "claude-haiku-4-5"


class GroqLLMClient:
    """Synchronous Groq SDK wrapped in `asyncio.to_thread`. Schema inlined in prompt."""

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set")
        self._client = Groq(api_key=key)

    @observe(as_type="generation", name="groq.complete_free_form")
    async def complete_free_form(self, system: str, user: str) -> str:
        def _call() -> tuple[str, dict[str, Any] | None]:
            r = self._client.chat.completions.create(
                model=FREE_FORM_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=300,
            )
            return (r.choices[0].message.content or "", _usage(r))

        text, usage = await asyncio.to_thread(_call)
        enrich_current_generation(model=FREE_FORM_MODEL, usage=usage)
        return text

    @observe(as_type="generation", name="groq.complete_structured")
    async def complete_structured[T: BaseModel](self, system: str, user: str, schema: type[T]) -> T:
        schema_json: dict[str, Any] = schema.model_json_schema()

        def _call() -> tuple[str, dict[str, Any] | None]:
            r = self._client.chat.completions.create(
                model=EXTRACTION_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"{system}\nRespond with ONLY a single JSON object that matches "
                            f"this JSON schema exactly (use the exact field names):\n"
                            f"{json.dumps(schema_json)}"
                        ),
                    },
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                max_tokens=300,
            )
            return (r.choices[0].message.content or "{}", _usage(r))

        raw, usage = await asyncio.to_thread(_call)
        log.debug("groq_structured_raw", raw=raw)
        enrich_current_generation(model=EXTRACTION_MODEL, usage=usage)
        return schema.model_validate_json(raw)


def _usage(response: Any) -> dict[str, Any] | None:
    u = getattr(response, "usage", None)
    if u is None:
        return None
    return {
        "input": getattr(u, "prompt_tokens", None),
        "output": getattr(u, "completion_tokens", None),
        "total": getattr(u, "total_tokens", None),
    }


class AnthropicRepClient:
    """Claude Haiku 4.5 client for the rep-mode LLM.

    Wraps `messages.parse()` so the response is a validated Pydantic instance —
    no manual JSON parse, no client-side schema check (the SDK handles both).

    The persona system prompt is marked with `cache_control: ephemeral` to
    enable prompt caching across rep turns within a call. Caveat: Haiku 4.5's
    minimum cacheable prefix is 4096 tokens — a typical persona prompt won't
    actually cache. The marker is harmless on short prompts (no error, just
    `cache_creation_input_tokens: 0`) and ready for a longer persona later.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: AsyncAnthropic | None = None,
    ) -> None:
        # `client` is the test seam: tests pass a pre-built mock SDK instance
        # so they don't have to mutate `self._client` post-construction. In
        # production both args are omitted; the API key resolves from env.
        if client is not None:
            self._client = client
            return
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self._client = AsyncAnthropic(api_key=key)

    @observe(as_type="generation", name="anthropic.complete_structured")
    async def complete_structured[T: BaseModel](
        self,
        system: str,
        history: list[dict[str, Any]],
        schema: type[T],
        max_tokens: int = 1024,
    ) -> T:
        response = await self._client.messages.parse(
            model=REP_MODEL,
            max_tokens=max_tokens,
            system=[
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            # Cast at the SDK boundary so callers (and the FakeAnthropicRepClient)
            # don't have to import anthropic-specific types into the seam.
            messages=cast(list[MessageParam], history),
            output_format=schema,
        )
        enrich_current_generation(model=REP_MODEL, usage=_anthropic_usage(response))
        parsed = response.parsed_output
        if parsed is None:
            # Refusal or schema mismatch — surface explicitly so callers can fall back.
            # Include the response id for Langfuse / dashboard correlation.
            stop = getattr(response, "stop_reason", "unknown")
            request_id = getattr(response, "id", "unknown")
            raise RuntimeError(
                f"Anthropic messages.parse() returned no parsed_output "
                f"(stop_reason={stop}, response_id={request_id})"
            )
        return parsed


def _anthropic_usage(response: Any) -> dict[str, Any] | None:
    u = getattr(response, "usage", None)
    if u is None:
        return None
    return {
        "input": getattr(u, "input_tokens", None),
        "output": getattr(u, "output_tokens", None),
        "cache_creation": getattr(u, "cache_creation_input_tokens", None),
        "cache_read": getattr(u, "cache_read_input_tokens", None),
    }
