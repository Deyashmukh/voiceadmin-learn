"""Groq-backed `LLMClient`. Structured output lands via `response_format=json_object`.

Kept in `agent/` (not `scripts/`) so both `main.py` and the M3 REPL drive-through
(`scripts/m3_repl_check.py`) import it instead of duplicating.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from groq import Groq
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

    async def complete_free_form(self, system: str, user: str) -> str:
        def _call() -> str:
            r = self._client.chat.completions.create(
                model=FREE_FORM_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=300,
            )
            return r.choices[0].message.content or ""

        return await asyncio.to_thread(_call)

    async def complete_structured[T: BaseModel](self, system: str, user: str, schema: type[T]) -> T:
        schema_json: dict[str, Any] = schema.model_json_schema()

        def _call() -> T:
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
            raw = r.choices[0].message.content or "{}"
            log.debug("groq_structured_raw", raw=raw)
            return schema.model_validate_json(raw)

        return await asyncio.to_thread(_call)
