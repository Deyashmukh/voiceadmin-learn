"""Groq-backed `LLMClient`. Structured output lands via `response_format=json_object`.

Kept in `agent/` (not `scripts/`) so both `main.py` and the M3 REPL drive-through
(`scripts/m3_repl_check.py`) import it instead of duplicating.

Both methods are decorated with `@observe(as_type="generation")` so Langfuse
captures them as LLM generations in the trace. The raw `groq` SDK doesn't go
through `langchain_core`, so the LangGraph callback handler can't see these
calls — without `@observe` they'd be invisible in the Langfuse UI. The decorator
is a no-op when Langfuse env vars aren't set, so this stays safe offline.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from groq import Groq
from langfuse import get_client, observe
from pydantic import BaseModel

from agent.logging_config import log

EXTRACTION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
FREE_FORM_MODEL = "qwen/qwen3-32b"


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
        _enrich_generation(model=FREE_FORM_MODEL, usage=usage)
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
        _enrich_generation(model=EXTRACTION_MODEL, usage=usage)
        return schema.model_validate_json(raw)


def _usage(response: Any) -> dict[str, Any] | None:
    """Pull token counts off a Groq chat completion response, if present."""
    u = getattr(response, "usage", None)
    if u is None:
        return None
    return {
        "input": getattr(u, "prompt_tokens", None),
        "output": getattr(u, "completion_tokens", None),
        "total": getattr(u, "total_tokens", None),
    }


def _enrich_generation(*, model: str, usage: dict[str, Any] | None) -> None:
    """Attach model name and token usage to the active Langfuse generation span.

    No-op when Langfuse is disabled — `get_client()` returns a stub client whose
    `update_current_generation` swallows the call.
    """
    try:
        get_client().update_current_generation(model=model, usage_details=usage)
    except Exception as exc:  # noqa: BLE001
        log.debug("langfuse_enrich_failed", error=str(exc))
