"""Graph-level tests: call `graph.ainvoke(state)` directly. No runner, no network."""

from __future__ import annotations

import pytest

from agent.graph import build_graph, initial_state
from agent.schemas import Benefits, ClassifierResult, PatientInfo

from .conftest import FakeClassifier, FakeLLMClient


async def _invoke(graph, state):
    return await graph.ainvoke(state, config={"recursion_limit": 10})


async def test_auth_happy_path(
    patient: PatientInfo, fake_llm: FakeLLMClient, fake_classifier: FakeClassifier
) -> None:
    graph = build_graph(fake_llm, fake_classifier)
    state = initial_state(patient)
    result = await _invoke(graph, state)
    assert result["current_node"] == "patient_id"
    assert "Alice" in result["response_text"]


async def test_extract_benefits_happy_path(
    patient: PatientInfo,
    benefits: Benefits,
    fake_llm: FakeLLMClient,
    fake_classifier: FakeClassifier,
) -> None:
    fake_llm.structured_responses = [benefits]
    graph = build_graph(fake_llm, fake_classifier)
    state = {**initial_state(patient), "current_node": "extract_benefits"}
    state["transcript"] = "deductible 250, copay 30, active yes"
    result = await _invoke(graph, state)
    assert result["current_node"] == "handoff"
    assert result["extracted"] == benefits


async def test_extract_malformed_output_routes_to_fallback(
    patient: PatientInfo, fake_llm: FakeLLMClient, fake_classifier: FakeClassifier
) -> None:
    # ValueError is in the handler's except clause; simulate a malformed LLM output.
    fake_llm.structured_exception = ValueError("no valid JSON in completion")
    graph = build_graph(fake_llm, fake_classifier)
    state = {**initial_state(patient), "current_node": "extract_benefits"}
    state["transcript"] = "???"
    result = await _invoke(graph, state)
    assert result["current_node"] == "fallback"
    assert result["fallback_reason"] == "extract_malformed_output"


async def test_handler_exception_propagates_out_of_ainvoke(
    patient: PatientInfo, fake_llm: FakeLLMClient, fake_classifier: FakeClassifier
) -> None:
    # An exception NOT in the handler's except clause should bubble up so the
    # runner can route to fallback (that's the runner's responsibility, not the graph's).
    fake_llm.structured_exception = RuntimeError("boom")
    graph = build_graph(fake_llm, fake_classifier)
    state = {**initial_state(patient), "current_node": "extract_benefits"}
    state["transcript"] = "anything"
    with pytest.raises(RuntimeError):
        await _invoke(graph, state)


async def test_unknown_current_node_routes_to_fallback(
    patient: PatientInfo, fake_llm: FakeLLMClient, fake_classifier: FakeClassifier
) -> None:
    graph = build_graph(fake_llm, fake_classifier)
    state = {**initial_state(patient), "current_node": "does_not_exist"}
    result = await _invoke(graph, state)
    assert result["current_node"] == "done"
    assert result["fallback_reason"] == "unspecified"


async def test_fallback_handler_preserves_reason(
    patient: PatientInfo, fake_llm: FakeLLMClient, fake_classifier: FakeClassifier
) -> None:
    graph = build_graph(fake_llm, fake_classifier)
    state = {
        **initial_state(patient),
        "current_node": "fallback",
        "fallback_reason": "extract_no_transcript",
    }
    result = await _invoke(graph, state)
    assert result["current_node"] == "done"
    assert result["fallback_reason"] == "extract_no_transcript"


async def test_handoff_without_benefits_routes_to_fallback(
    patient: PatientInfo, fake_llm: FakeLLMClient, fake_classifier: FakeClassifier
) -> None:
    graph = build_graph(fake_llm, fake_classifier)
    state = {**initial_state(patient), "current_node": "handoff", "extracted": None}
    result = await _invoke(graph, state)
    assert result["current_node"] == "fallback"
    assert result["fallback_reason"] == "handoff_no_benefits"


async def test_patient_id_speak_goes_to_extract(
    patient: PatientInfo, fake_llm: FakeLLMClient, fake_classifier: FakeClassifier
) -> None:
    fake_classifier.results = [ClassifierResult(outcome="speak", confidence=1.0)]
    graph = build_graph(fake_llm, fake_classifier)
    state = {**initial_state(patient), "current_node": "patient_id"}
    state["transcript"] = "please provide patient info"
    result = await _invoke(graph, state)
    assert result["current_node"] == "extract_benefits"


@pytest.mark.parametrize("bad_node", ["banana", "AUTH", "fall_back"])
async def test_various_unknown_nodes_all_route_to_fallback(
    patient: PatientInfo,
    fake_llm: FakeLLMClient,
    fake_classifier: FakeClassifier,
    bad_node: str,
) -> None:
    graph = build_graph(fake_llm, fake_classifier)
    state = {**initial_state(patient), "current_node": bad_node}
    result = await _invoke(graph, state)
    assert result["current_node"] == "done"
