"""LangGraph state machine for the eligibility-verification call.

One `ainvoke` = one turn. Conditional edges from START route directly to the
handler for `state["current_node"]`. Each handler returns a partial state with
the next `current_node`, then goes to END. The next turn re-enters from START.

Handlers are built as closures over injected dependencies so they stay
unit-testable with fakes (no pipecat / twilio imports here). Handlers must be
idempotent and cancellation-safe; errors return `_to_fallback`, never partial state.
"""

from __future__ import annotations

import json

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import ValidationError

from agent.logging_config import log
from agent.schemas import (
    FALLBACK_RESPONSE,
    Benefits,
    CallState,
    IVRClassifier,
    LLMClient,
    PatientInfo,
)

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


def _to_fallback(reason: str, response: str = FALLBACK_RESPONSE) -> CallState:
    return {
        "current_node": "fallback",
        "fallback_reason": reason,
        "response_text": response,
    }


def _route_from_start(state: CallState) -> str:
    """Route the turn to the handler matching `state['current_node']`."""
    node = state.get("current_node", "fallback")
    if node not in NODES:
        log.warning("unknown_node_routed_to_fallback", node=node)
        return "fallback"
    return node


def build_graph(llm: LLMClient, classifier: IVRClassifier) -> CompiledStateGraph:
    """Compile the graph with dependencies captured in handler closures."""

    async def auth_handler(state: CallState) -> CallState:
        patient = state.get("patient")
        if patient is None:
            return _to_fallback("auth_missing_patient", "Sorry, I don't have caller info.")
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

    async def patient_id_handler(state: CallState) -> CallState:
        # Real IVR keyword handling lands at M5. For now, any classifier outcome
        # advances to extraction; DTMF additionally emits the tone string.
        result = classifier.classify(state.get("transcript", ""))
        out: CallState = {"current_node": "extract_benefits"}
        if result.outcome == "dtmf":
            out["response_text"] = f"DTMF {result.dtmf} sent."
        return out

    async def extract_benefits_handler(state: CallState) -> CallState:
        transcript = state.get("transcript", "")
        if not transcript:
            return _to_fallback("extract_no_transcript")
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
            return _to_fallback("extract_malformed_output")
        return {
            "current_node": "handoff",
            "extracted": benefits,
            "response_text": "Got it, thank you.",
        }

    async def handoff_handler(state: CallState) -> CallState:
        if state.get("extracted") is None:
            return _to_fallback("handoff_no_benefits")
        return {
            "current_node": "done",
            "response_text": "Thanks, goodbye.",
        }

    async def fallback_handler(state: CallState) -> CallState:
        reason = state.get("fallback_reason") or "unspecified"
        log.info("fallback_reached", reason=reason)
        return {
            "current_node": "done",
            "response_text": state.get("response_text") or FALLBACK_RESPONSE,
            "fallback_reason": reason,
        }

    async def done_handler(state: CallState) -> CallState:
        # Unreachable in the normal runner loop (consumer exits on current_node=='done'),
        # but kept so forcing current_node='done' into ainvoke doesn't raise.
        return {"current_node": "done"}

    builder = StateGraph(CallState)
    builder.add_node("auth", auth_handler)
    builder.add_node("patient_id", patient_id_handler)
    builder.add_node("extract_benefits", extract_benefits_handler)
    builder.add_node("handoff", handoff_handler)
    builder.add_node("fallback", fallback_handler)
    builder.add_node("done", done_handler)

    builder.add_conditional_edges(
        START,
        _route_from_start,
        {n: n for n in NODES},
    )
    for n in NODES:
        builder.add_edge(n, END)

    # No checkpointer: per-call state lives in GraphRunner; revisit if resume-after-crash matters.
    return builder.compile()
