"""GraphRunner tests: bounded queues, cancellation, stale-drain, fallback routing."""

from __future__ import annotations

import asyncio

import pytest
import structlog

from agent.graph import build_graph
from agent.graph_runner import QUEUE_MAX, CallContext, GraphRunner
from agent.schemas import FALLBACK_RESPONSE, PatientInfo

from .conftest import FakeClassifier, FakeLLMClient


def _ctx(patient: PatientInfo) -> CallContext:
    return CallContext(call_sid="CA-test", patient=patient)


async def test_runner_processes_transcript(
    patient: PatientInfo, fake_llm: FakeLLMClient, fake_classifier: FakeClassifier
) -> None:
    graph = build_graph(fake_llm, fake_classifier)
    runner = GraphRunner(graph, _ctx(patient))
    await runner.start()
    try:
        runner.submit_transcript("ready")
        out = await asyncio.wait_for(runner.out_queue.get(), timeout=1.0)
        assert "Alice" in out
        # Turn advanced state to patient_id.
        assert runner.state.get("current_node") == "patient_id"
        assert runner.state.get("turn_count") == 1
    finally:
        await runner.stop()


async def test_mark_interrupted_cancels_slow_turn(
    patient: PatientInfo, fake_classifier: FakeClassifier
) -> None:
    slow_llm = FakeLLMClient(slow_mode_seconds=2.0)
    graph = build_graph(slow_llm, fake_classifier)
    runner = GraphRunner(graph, _ctx(patient))
    # Start already in extract_benefits so the first turn goes through the LLM.
    runner.state = {**runner.state, "current_node": "extract_benefits"}
    await runner.start()
    try:
        runner.submit_transcript("rep utterance")
        # Give the consumer time to pick up and start the turn.
        await asyncio.sleep(0.1)
        assert runner.state.get("current_node") == "extract_benefits"
        runner.mark_interrupted()
        # The turn should be cancelled quickly, well under the 2s slow sleep.
        await asyncio.sleep(0.2)
        # State should NOT have advanced — cancelled turn does not commit state.
        assert runner.state.get("current_node") == "extract_benefits"
        assert runner.state.get("turn_count") == 0
        # out_queue should be empty — no partial response emitted.
        assert runner.out_queue.empty()
    finally:
        await runner.stop()


async def test_stale_transcripts_are_dropped_on_new_turn(
    patient: PatientInfo, fake_llm: FakeLLMClient, fake_classifier: FakeClassifier
) -> None:
    graph = build_graph(fake_llm, fake_classifier)
    runner = GraphRunner(graph, _ctx(patient))
    # Queue multiple transcripts before starting; consume should collapse to the
    # last one per turn via the drain loop in _consume.
    for t in ("stale 1", "stale 2", "stale 3", "ready"):
        runner.submit_transcript(t)
    await runner.start()
    try:
        out = await asyncio.wait_for(runner.out_queue.get(), timeout=1.0)
        assert "Alice" in out
        # Only one turn fired; state advanced exactly once.
        assert runner.state.get("turn_count") == 1
    finally:
        await runner.stop()


async def test_submit_transcript_drops_oldest_on_full(
    patient: PatientInfo, fake_llm: FakeLLMClient, fake_classifier: FakeClassifier
) -> None:
    graph = build_graph(fake_llm, fake_classifier)
    runner = GraphRunner(graph, _ctx(patient))
    # Fill the queue beyond capacity; drop-oldest keeps the newest QUEUE_MAX items.
    for i in range(QUEUE_MAX + 5):
        runner.submit_transcript(f"item-{i}")
    assert runner.in_queue.qsize() == QUEUE_MAX
    # The oldest items should have been evicted; the queue should contain the tail.
    remaining = []
    while not runner.in_queue.empty():
        remaining.append(runner.in_queue.get_nowait())
    assert remaining[-1] == f"item-{QUEUE_MAX + 4}"
    assert "item-0" not in remaining


async def test_handler_exception_routes_to_fallback(
    patient: PatientInfo, fake_classifier: FakeClassifier
) -> None:
    fake_llm = FakeLLMClient(structured_exception=RuntimeError("explode"))
    graph = build_graph(fake_llm, fake_classifier)
    runner = GraphRunner(graph, _ctx(patient))
    runner.state = {**runner.state, "current_node": "extract_benefits"}
    await runner.start()
    try:
        runner.submit_transcript("anything")
        out = await asyncio.wait_for(runner.out_queue.get(), timeout=1.0)
        assert out == FALLBACK_RESPONSE
        assert runner.state.get("current_node") == "done"
        assert runner.state.get("fallback_reason") == "node_exception"
    finally:
        await runner.stop()


async def test_stop_cancels_in_flight_turn(
    patient: PatientInfo, fake_classifier: FakeClassifier
) -> None:
    slow_llm = FakeLLMClient(slow_mode_seconds=5.0)
    graph = build_graph(slow_llm, fake_classifier)
    runner = GraphRunner(graph, _ctx(patient))
    runner.state = {**runner.state, "current_node": "extract_benefits"}
    await runner.start()
    runner.submit_transcript("go")
    await asyncio.sleep(0.05)
    # stop() should complete quickly despite the 5s slow LLM.
    await asyncio.wait_for(runner.stop(), timeout=1.0)


async def test_unknown_node_in_state_routes_through_fallback(
    patient: PatientInfo, fake_llm: FakeLLMClient, fake_classifier: FakeClassifier
) -> None:
    graph = build_graph(fake_llm, fake_classifier)
    runner = GraphRunner(graph, _ctx(patient))
    runner.state = {**runner.state, "current_node": "banana"}
    await runner.start()
    try:
        runner.submit_transcript("anything")
        out = await asyncio.wait_for(runner.out_queue.get(), timeout=1.0)
        assert out == FALLBACK_RESPONSE
        assert runner.state.get("current_node") == "done"
    finally:
        await runner.stop()


@pytest.mark.parametrize("slow_seconds", [1.0, 3.0])
async def test_cancel_latency_is_low(
    patient: PatientInfo, fake_classifier: FakeClassifier, slow_seconds: float
) -> None:
    slow_llm = FakeLLMClient(slow_mode_seconds=slow_seconds)
    graph = build_graph(slow_llm, fake_classifier)
    runner = GraphRunner(graph, _ctx(patient))
    runner.state = {**runner.state, "current_node": "extract_benefits"}
    await runner.start()
    try:
        runner.submit_transcript("rep utterance")
        await asyncio.sleep(0.05)
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        runner.mark_interrupted()
        # Wait until the consumer observes cancellation.
        while runner._current_turn and not runner._current_turn.done():
            await asyncio.sleep(0.01)
        elapsed = loop.time() - t0
        assert elapsed < 0.2, f"cancel took {elapsed:.3f}s"
    finally:
        await runner.stop()


async def test_interrupt_between_turns_does_not_stick(
    patient: PatientInfo, fake_llm: FakeLLMClient, fake_classifier: FakeClassifier
) -> None:
    """mark_interrupted() with no in-flight turn must be a no-op on the flag.

    If Pipecat fires VADActiveFrame while the consumer is parked on in_queue.get()
    (between turns), we must not stash a stale 'interrupt' that would conflate the
    next turn's CancelledError semantics.
    """
    graph = build_graph(fake_llm, fake_classifier)
    runner = GraphRunner(graph, _ctx(patient))
    await runner.start()
    try:
        # No turn yet. This used to stick _interrupt_requested=True forever.
        runner.mark_interrupted()
        assert runner._interrupt_requested is False
        # Next real turn should complete normally.
        runner.submit_transcript("ready")
        out = await asyncio.wait_for(runner.out_queue.get(), timeout=1.0)
        assert "Alice" in out
        assert runner.state.get("turn_count") == 1
        assert runner._interrupt_requested is False
    finally:
        await runner.stop()


async def test_stop_during_interrupt_completes_cleanly(
    patient: PatientInfo, fake_classifier: FakeClassifier
) -> None:
    """stop() called while an interrupt is mid-flight must shut down cleanly."""
    slow_llm = FakeLLMClient(slow_mode_seconds=5.0)
    graph = build_graph(slow_llm, fake_classifier)
    runner = GraphRunner(graph, _ctx(patient))
    runner.state = {**runner.state, "current_node": "extract_benefits"}
    await runner.start()
    runner.submit_transcript("rep utterance")
    await asyncio.sleep(0.05)
    runner.mark_interrupted()
    # stop() must not hang despite the 5s slow LLM and the in-flight interrupt.
    await asyncio.wait_for(runner.stop(), timeout=1.0)
    assert runner._consumer is not None and runner._consumer.done()


async def test_state_is_committed_before_response_published(
    patient: PatientInfo, fake_llm: FakeLLMClient, fake_classifier: FakeClassifier
) -> None:
    """Out-queue readers must never observe a response before state advances.

    Locks in the ordering fix (NOTES #2): state update precedes out_queue.put.
    """
    graph = build_graph(fake_llm, fake_classifier)
    runner = GraphRunner(graph, _ctx(patient))
    observed_state_at_put: list[dict] = []
    original_put = runner.out_queue.put

    async def recording_put(item: str) -> None:
        observed_state_at_put.append({**runner.state})
        await original_put(item)

    runner.out_queue.put = recording_put  # type: ignore[method-assign]
    await runner.start()
    try:
        runner.submit_transcript("ready")
        _ = await asyncio.wait_for(runner.out_queue.get(), timeout=1.0)
        assert len(observed_state_at_put) == 1
        # At the instant of put, state must already reflect the completed turn.
        assert observed_state_at_put[0].get("turn_count") == 1
        assert observed_state_at_put[0].get("current_node") == "patient_id"
    finally:
        await runner.stop()


async def test_ctor_overrides_queue_max_and_recursion_limit(
    patient: PatientInfo, fake_llm: FakeLLMClient, fake_classifier: FakeClassifier
) -> None:
    graph = build_graph(fake_llm, fake_classifier)
    runner = GraphRunner(graph, _ctx(patient), queue_max=2, recursion_limit=3)
    assert runner.recursion_limit == 3
    for i in range(5):
        runner.submit_transcript(f"item-{i}")
    assert runner.in_queue.qsize() == 2


async def test_contextvars_propagate_into_run_turn(
    patient: PatientInfo, fake_classifier: FakeClassifier
) -> None:
    """call_sid and turn_index must be bound in the _run_turn task context."""
    fake_llm = FakeLLMClient(structured_exception=RuntimeError("observe"))
    graph = build_graph(fake_llm, fake_classifier)
    runner = GraphRunner(graph, _ctx(patient))
    runner.state = {**runner.state, "current_node": "extract_benefits"}
    observed: dict[str, object] = {}
    real_run = runner._run_turn

    async def wrapper(transcript: str):
        observed.update(structlog.contextvars.get_contextvars())
        return await real_run(transcript)

    runner._run_turn = wrapper  # type: ignore[method-assign]
    await runner.start()
    try:
        runner.submit_transcript("anything")
        await asyncio.wait_for(runner.out_queue.get(), timeout=1.0)
        assert observed.get("call_sid") == "CA-test"
        assert observed.get("turn_index") == 0
    finally:
        await runner.stop()
