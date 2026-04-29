"""Anthropic rep-mode LLM client.

`AnthropicRepClient.complete_structured(system, history, schema)` wraps
`messages.parse()` so the response is a validated Pydantic instance. The
persona system prompt is marked with `cache_control: ephemeral` to enable
prompt caching across rep turns within a call.

`@observe` produces one Langfuse generation span per round-trip.
"""

from __future__ import annotations

import os
from typing import Any, cast

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam
from pydantic import BaseModel

from agent.observability import enrich_current_generation, observe

REP_MODEL = "claude-haiku-4-5"


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
