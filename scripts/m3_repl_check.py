"""M3 exit bar: REPL drive-through with a real Groq-backed LLMClient.

Builds the LangGraph state machine with the production `GroqLLMClient` and the
rule-based classifier, then drives it turn-by-turn through the happy path:
auth -> patient_id -> extract_benefits -> handoff -> done.

Run: `uv run python scripts/m3_repl_check.py`
Requires: .env with GROQ_API_KEY.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from agent.classifier import RuleBasedClassifier
from agent.graph import build_graph
from agent.graph_runner import CallContext, GraphRunner
from agent.llm_client import GroqLLMClient
from agent.schemas import Benefits, PatientInfo

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=True)

REP_UTTERANCE = (
    "Yes, the member is active. Deductible remaining is $250. Copay is $30. "
    "Coinsurance is 20 percent. No out of network coverage."
)


async def main() -> None:
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY not set in environment")

    patient = PatientInfo(
        member_id="M123456", first_name="Alice", last_name="Example", dob="1980-05-12"
    )
    graph = build_graph(GroqLLMClient(), RuleBasedClassifier())
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
