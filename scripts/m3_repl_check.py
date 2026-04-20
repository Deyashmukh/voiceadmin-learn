"""M3 exit bar: REPL drive-through with a real Groq-backed LLMClient.

Builds the LangGraph state machine with a real `meta-llama/llama-4-scout`
Groq client (structured JSON) and a stub classifier, then drives it turn-by-turn
through the happy path: auth -> patient_id -> extract_benefits -> handoff -> done.

Run: `uv run python scripts/m3_repl_check.py`
Requires: .env with GROQ_API_KEY.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel

from agent.graph import build_graph
from agent.graph_runner import CallContext, GraphRunner
from agent.schemas import Benefits, ClassifierResult, PatientInfo

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=True)

EXTRACTION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

REP_UTTERANCE = (
    "Yes, the member is active. Deductible remaining is $250. Copay is $30. "
    "Coinsurance is 20 percent. No out of network coverage."
)


class GroqLLMClient:
    """Minimal real LLMClient for the REPL check. M4 owns the production one."""

    def __init__(self, api_key: str) -> None:
        self._client = Groq(api_key=api_key)

    async def complete_free_form(self, system: str, user: str) -> str:
        def _call() -> str:
            r = self._client.chat.completions.create(
                model=EXTRACTION_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=200,
            )
            return (r.choices[0].message.content or "").strip()

        return await asyncio.to_thread(_call)

    async def complete_structured[T: BaseModel](self, system: str, user: str, schema: type[T]) -> T:
        # Spell out the exact field names; otherwise Llama happily invents
        # synonyms (e.g. "is_active" instead of "active") and pydantic rejects.
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
            payload = r.choices[0].message.content or "{}"
            return schema.model_validate_json(payload)

        return await asyncio.to_thread(_call)


class StubClassifier:
    def classify(self, transcript: str) -> ClassifierResult:
        return ClassifierResult(outcome="speak", confidence=1.0)


async def main() -> None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY not set in environment")

    patient = PatientInfo(
        member_id="M123456", first_name="Alice", last_name="Example", dob="1980-05-12"
    )
    graph = build_graph(GroqLLMClient(api_key), StubClassifier())
    runner = GraphRunner(graph, CallContext(call_sid="REPL", patient=patient))
    await runner.start()

    async def next_response(label: str) -> str:
        response = await asyncio.wait_for(runner.out_queue.get(), timeout=15.0)
        node = runner.state.get("current_node")
        print(f"[{label}] node={node} response={response!r}")
        return response

    try:
        # Turn 1 — auth: emits the member readout, advances to patient_id.
        runner.submit_transcript("ready")
        await next_response("after auth")

        # Turn 2 — patient_id: silent (speak-outcome classifier), advances to
        # extract_benefits. No response is expected on out_queue; assert state moved.
        runner.submit_transcript("please speak to a rep")
        for _ in range(30):
            if runner.state.get("current_node") == "extract_benefits":
                break
            await asyncio.sleep(0.05)
        assert runner.state.get("current_node") == "extract_benefits", (
            "expected patient_id to advance to extract_benefits silently"
        )
        assert runner.out_queue.empty(), "silent patient_id turn must not emit a response"
        print("[after patient_id] node=extract_benefits (no response, as designed)")

        # Turn 3 — extract_benefits: real Groq call, returns structured Benefits.
        runner.submit_transcript(REP_UTTERANCE)
        await next_response("after extract_benefits")

        # Turn 4 — handoff: any transcript triggers goodbye + node=done.
        runner.submit_transcript("thanks")
        await next_response("after handoff")
    finally:
        await runner.stop()

    extracted = runner.state.get("extracted")
    print("---")
    print(f"final_node = {runner.state.get('current_node')}")
    print(f"turn_count = {runner.state.get('turn_count')}")
    print(f"fallback_reason = {runner.state.get('fallback_reason')}")
    if extracted is None:
        print("extracted = None  (fallback fired)")
        raise SystemExit(1)
    print(f"extracted  = {json.dumps(extracted.model_dump(), indent=2)}")

    assert runner.state.get("current_node") == "done", "expected terminal node=done"
    assert runner.state.get("fallback_reason") is None, "unexpected fallback fired"
    assert isinstance(extracted, Benefits), "extracted must be Benefits"
    assert extracted.active is True, "rep said the member is active"
    print("OK — M3 REPL drive-through completed end-to-end with real Groq.")


if __name__ == "__main__":
    asyncio.run(main())
