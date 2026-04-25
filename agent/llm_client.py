"""Groq-backed `LLMClient`. Structured output lands via `response_format=json_object`.

Kept in `agent/` (not `scripts/`) so both `main.py` and the M3 REPL drive-through
(`scripts/m3_repl_check.py`) import it instead of duplicating.

`@observe` makes the raw Groq calls visible to Langfuse — they bypass
`langchain_core`, so the LangGraph callback handler alone can't see them.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from groq import Groq
from langfuse import observe
from pydantic import BaseModel

from agent.logging_config import log
from agent.observability import enrich_current_generation

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
