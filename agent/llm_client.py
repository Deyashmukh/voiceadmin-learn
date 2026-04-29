"""LLM clients: Anthropic for rep-mode structured output, Groq for IVR
tool-calling. Both expose `@observe`-decorated entry points so each LLM
round-trip surfaces as a Langfuse generation span.
"""

from __future__ import annotations

import json
import os
from typing import Any, cast

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam
from groq import AsyncGroq
from pydantic import BaseModel, ValidationError

from agent.logging_config import log
from agent.observability import enrich_current_generation, observe
from agent.schemas import IVRTurnResponse, ToolCall, ToolName, Turn

REP_MODEL = "claude-haiku-4-5"
# Llama 4 Scout via Groq — fast first-token latency (~80ms p50), tool
# calling supported, $0.11/$0.34 per 1M in/out at the time of writing.
IVR_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


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


class GroqToolCallingClient:
    """Tool-calling LLM client for IVR mode (Llama 4 Scout via Groq).

    Implements the `IVRLLMClient` Protocol. Maps the project's flat
    `Turn` history into Groq's chat-completion message shape, including
    `assistant.tool_calls` + `tool.tool_call_id` round-tripping.

    Hallucinated tool names or malformed JSON arguments are dropped at the
    boundary — the dispatcher in `agent/tools.py` would reject them anyway,
    but filtering here keeps the per-turn `IVRTurnResponse.tool_calls` list
    clean for the watchdog's no-progress accounting.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: AsyncGroq | None = None,
        model: str = IVR_MODEL,
    ) -> None:
        # `client` is the test seam — tests pass a pre-built mock SDK instance.
        if client is not None:
            self._client = client
            self._model = model
            return
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set")
        self._client = AsyncGroq(api_key=key)
        self._model = model

    @observe(as_type="generation", name="groq.complete_with_tools")
    async def complete_with_tools(
        self,
        system: str,
        history: list[Turn],
        tools: list[dict[str, Any]],
        temperature: float = 0.1,
    ) -> IVRTurnResponse:
        messages = _history_to_groq_messages(system, history)
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=cast(Any, messages),
            tools=cast(Any, tools),
            tool_choice="auto",
            temperature=temperature,
            max_tokens=512,
        )
        choice = response.choices[0]
        msg = choice.message
        parsed_calls: list[ToolCall] = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                # Without this log the no-progress watchdog would see "0 tool
                # calls" with no signal that the LLM tried but emitted garbage.
                log.warning(
                    "ivr_tool_call_dropped",
                    reason="malformed_json",
                    name=tc.function.name,
                    raw_args=tc.function.arguments,
                )
                continue
            try:
                parsed_calls.append(
                    ToolCall(name=cast(ToolName, tc.function.name), args=args, id=tc.id)
                )
            except ValidationError:
                log.warning(
                    "ivr_tool_call_dropped",
                    reason="hallucinated_name",
                    name=tc.function.name,
                )
                continue
        enrich_current_generation(model=self._model, usage=_groq_usage(response))
        return IVRTurnResponse(tool_calls=parsed_calls, text=msg.content or "")


def _history_to_groq_messages(system: str, history: list[Turn]) -> list[dict[str, object]]:
    """Convert internal flat `Turn` history to Groq chat-completion messages.

    Pairs consecutive `tool_call` + `tool_result` Turns into an
    `assistant.tool_calls` + `tool.tool_call_id` message pair. The pairing
    relies on the call-loop's invariant that `_ivr_turn` always appends a
    `tool_result` immediately after the `tool_call` it dispatches.
    """
    messages: list[dict[str, object]] = [{"role": "system", "content": system}]
    i = 0
    while i < len(history):
        turn = history[i]
        if turn.role == "user" and turn.content:
            messages.append({"role": "user", "content": turn.content})
            i += 1
        elif turn.role == "assistant" and turn.content:
            messages.append({"role": "assistant", "content": turn.content})
            i += 1
        elif turn.role == "tool_call" and turn.tool_call is not None:
            tc = turn.tool_call
            # Synthesized ids are local to this conversion (not persistent
            # across turns). They only appear when the SDK omits one — Groq
            # always provides ids in practice; this is a defensive fallback.
            tc_id = tc.id or f"call_{i}_{tc.name}"
            messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": tc_id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.args),
                            },
                        }
                    ],
                }
            )
            if i + 1 < len(history) and history[i + 1].role == "tool_result":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": history[i + 1].content,
                    }
                )
                i += 2
            else:
                i += 1
        else:
            # Skip unpaired tool_result Turns and empty-content user/assistant Turns.
            i += 1
    return messages


def _groq_usage(response: Any) -> dict[str, Any] | None:
    u = getattr(response, "usage", None)
    if u is None:
        return None
    return {
        "input": getattr(u, "prompt_tokens", None),
        "output": getattr(u, "completion_tokens", None),
        "total": getattr(u, "total_tokens", None),
    }
