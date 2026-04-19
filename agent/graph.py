"""LangGraph state machine for the eligibility-verification call.

One `ainvoke` = one turn. A dispatcher node reads `current_node` and routes
to the handler for that node. Each handler returns a partial state with the
next `current_node`, then goes to END. The next turn re-enters dispatcher.

Handlers are built as closures over injected dependencies so they stay
unit-testable with fakes (no pipecat / twilio imports here).
"""

from __future__ import annotations

import json

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import ValidationError

from agent.logging_config import log
from agent.schemas import Benefits, CallState, IVRClassifier, LLMClient, PatientInfo

NODES = ("auth", "patient_id", "extract_benefits", "handoff", "fallback", "done")


def initial_state(patient: PatientInfo) -> CallState:
    return {
        "current_node": "auth",
        "patient": patient,
        "extracted": None,
        "turn_count": 0,
        "transcript": "",
        "response_text": None,
        "fallback_reason": None,
    }


def _route_from_dispatcher(state: CallState) -> str:
    node = state.get("current_node", "fallback")
    if node not in NODES:
        log.warning("unknown_node_routed_to_fallback", node=node)
        return "fallback"
    return node


def _dispatcher(state: CallState) -> dict:
    """Pure pass-through. Conditional edges do the routing."""
    return {}


def build_graph(llm: LLMClient, classifier: IVRClassifier) -> CompiledStateGraph:
    """Compile the graph with dependencies captured in handler closures."""

    async def auth_handler(state: CallState) -> dict:
        patient = state.get("patient")
        if patient is None:
            return {
                "current_node": "fallback",
                "fallback_reason": "auth_missing_patient",
                "response_text": "Sorry, I don't have caller info.",
            }
        transcript = state.get("transcript", "").lower()
        if "ready" in transcript or "go ahead" in transcript or state.get("turn_count", 0) == 0:
            return {
                "current_node": "patient_id",
                "response_text": (
                    f"Calling for member {patient.member_id}, "
                    f"{patient.first_name} {patient.last_name}, DOB {patient.dob}."
                ),
            }
        return {
            "current_node": "auth",
            "response_text": "I'm calling to verify eligibility. Are you ready?",
        }

    async def patient_id_handler(state: CallState) -> dict:
        # Simple gate: once we've said the patient info and got any acknowledgement,
        # proceed to extraction. Real IVR handling comes at M5.
        transcript = state.get("transcript", "")
        result = classifier.classify(transcript)
        if result.outcome == "speak":
            return {"current_node": "extract_benefits"}
        if result.outcome == "dtmf":
            return {
                "current_node": "extract_benefits",
                "response_text": f"DTMF {result.dtmf} sent.",
            }
        return {"current_node": "extract_benefits"}

    async def extract_benefits_handler(state: CallState) -> dict:
        transcript = state.get("transcript", "")
        if not transcript:
            return {
                "current_node": "fallback",
                "fallback_reason": "extract_no_transcript",
                "response_text": "Sorry, could you repeat that?",
            }
        try:
            benefits = await llm.complete_structured(
                system=(
                    "Extract benefits from the rep utterance into the Benefits schema. "
                    "Only populate fields you can infer with high confidence."
                ),
                user=transcript,
                schema=Benefits,
            )
        except (ValidationError, json.JSONDecodeError, ValueError) as exc:
            log.warning("extract_malformed", error=str(exc))
            return {
                "current_node": "fallback",
                "fallback_reason": "extract_malformed_output",
                "response_text": "Sorry, could you repeat that?",
            }
        return {
            "current_node": "handoff",
            "extracted": benefits,
            "response_text": "Got it, thank you.",
        }

    async def handoff_handler(state: CallState) -> dict:
        # Once we've got Benefits, the call is effectively done.
        if state.get("extracted") is None:
            return {
                "current_node": "fallback",
                "fallback_reason": "handoff_no_benefits",
                "response_text": "Sorry, could you repeat that?",
            }
        return {
            "current_node": "done",
            "response_text": "Thanks, goodbye.",
        }

    async def fallback_handler(state: CallState) -> dict:
        # Idempotent: if we landed here without a reason, record one.
        reason = state.get("fallback_reason") or "unspecified"
        log.info("fallback_reached", reason=reason)
        return {
            "current_node": "done",
            "response_text": state.get("response_text") or "Sorry, could you repeat that?",
            "fallback_reason": reason,
        }

    async def done_handler(state: CallState) -> dict:
        return {"current_node": "done"}

    builder = StateGraph(CallState)
    builder.add_node("dispatcher", _dispatcher)
    builder.add_node("auth", auth_handler)
    builder.add_node("patient_id", patient_id_handler)
    builder.add_node("extract_benefits", extract_benefits_handler)
    builder.add_node("handoff", handoff_handler)
    builder.add_node("fallback", fallback_handler)
    builder.add_node("done", done_handler)

    builder.set_entry_point("dispatcher")
    builder.add_conditional_edges(
        "dispatcher",
        _route_from_dispatcher,
        {n: n for n in NODES},
    )
    for n in NODES:
        builder.add_edge(n, END)

    return builder.compile()
