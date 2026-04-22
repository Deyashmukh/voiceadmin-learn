"""M4 critical verification tests (a–d).

Drive `StateMachineProcessor` with hand-crafted Pipecat frames. Zero network,
zero real Pipecat transport — just the processor, a capturing sink, and the
graph runner underneath.

Coverage:
  (a) Slow-LLM barge-in: a UserStartedSpeakingFrame cancels the in-flight turn.
  (b) Rapid transcripts: two transcripts 100ms apart → only the newest lands.
  (c) Handler exception: fake LLM raises → state goes to `done`/fallback.
  (d) Contextvar propagation: every log from the runner has `call_sid` bound.
"""

from __future__ import annotations

import asyncio

import pytest
import structlog
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    StartFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from agent.graph import build_graph
from agent.graph_runner import CallContext, GraphRunner
from agent.processors.state_processor import StateMachineProcessor
from agent.schemas import FALLBACK_RESPONSE, PatientInfo

from .conftest import FakeClassifier, FakeLLMClient


def _finalized(text: str) -> TranscriptionFrame:
    return TranscriptionFrame(text=text, user_id="u", timestamp="t", finalized=True)


def _interim(text: str) -> TranscriptionFrame:
    return TranscriptionFrame(text=text, user_id="u", timestamp="t", finalized=False)


class _Sink(FrameProcessor):
    """Minimal sink: record every frame that arrives on its input."""

    def __init__(self) -> None:
        super().__init__(enable_direct_mode=True)
        self.frames: list[Frame] = []

    async def queue_frame(
        self,
        frame: Frame,
        direction: FrameDirection = FrameDirection.DOWNSTREAM,
        callback: object = None,
    ) -> None:
        self.frames.append(frame)


def _ctx(patient: PatientInfo) -> CallContext:
    return CallContext(call_sid="CA-m4", patient=patient)


async def _make_processor(
    patient: PatientInfo, fake_llm: FakeLLMClient, fake_classifier: FakeClassifier
) -> tuple[StateMachineProcessor, GraphRunner, _Sink]:
    graph = build_graph(fake_llm, fake_classifier)
    runner = GraphRunner(graph, _ctx(patient))
    proc = StateMachineProcessor(runner, enable_direct_mode=True)
    sink = _Sink()
    proc.link(sink)
    # StartFrame must flow through first so push_frame is allowed.
    start = StartFrame(
        audio_in_sample_rate=16000,
        audio_out_sample_rate=16000,
    )
    await proc.process_frame(start, FrameDirection.DOWNSTREAM)
    return proc, runner, sink


async def _shutdown(proc: StateMachineProcessor) -> None:
    await proc.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)


def _text_frames(sink: _Sink) -> list[TextFrame]:
    # TranscriptionFrame extends TextFrame; the pump emits bare TextFrames only.
    return [f for f in sink.frames if type(f) is TextFrame]


async def _wait_until(pred, timeout: float = 1.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if pred():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("predicate did not become true in time")


# (a) Slow-LLM barge-in ------------------------------------------------------


async def test_slow_llm_barge_in_cancels_turn(
    patient: PatientInfo, fake_classifier: FakeClassifier
) -> None:
    """A slow LLM mid-turn must be cancelled by UserStartedSpeakingFrame."""
    slow_llm = FakeLLMClient(slow_mode_seconds=2.0)
    proc, runner, sink = await _make_processor(patient, slow_llm, fake_classifier)
    # Start in extract_benefits so the first turn goes through the LLM.
    runner.state = {**runner.state, "current_node": "extract_benefits"}
    try:
        await proc.process_frame(_finalized("rep utterance"), FrameDirection.DOWNSTREAM)
        # Let the runner pick up the turn.
        await asyncio.sleep(0.05)
        turn = runner._current_turn
        assert turn is not None and not turn.done()
        # VAD fires: user started speaking mid-TTS.
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        await proc.process_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await _wait_until(lambda: turn.done(), timeout=0.5)
        elapsed = loop.time() - t0
        assert elapsed < 0.2, f"cancel took {elapsed:.3f}s"
        assert turn.cancelled()
        # No response emitted from the cancelled turn.
        assert not _text_frames(sink)
        assert runner.state.get("turn_count") == 0
    finally:
        await _shutdown(proc)


# (b) Rapid transcripts ------------------------------------------------------


async def test_rapid_transcripts_only_newest_completes(
    patient: PatientInfo, fake_classifier: FakeClassifier
) -> None:
    """Two finalized transcripts 100ms apart → only the newest drives a turn."""
    # Slow enough that the first enters the queue before consumer takes it.
    slow_llm = FakeLLMClient(slow_mode_seconds=0.05)
    proc, runner, _sink = await _make_processor(patient, slow_llm, fake_classifier)
    try:
        # Queue both before the consumer has a chance to take either.
        await proc.process_frame(_finalized("stale"), FrameDirection.DOWNSTREAM)
        await proc.process_frame(_finalized("newest"), FrameDirection.DOWNSTREAM)
        # Wait for the turn to complete.
        await _wait_until(lambda: runner.state.get("turn_count") == 1, timeout=1.0)
        # Only one turn's worth of progress; stale was drained by _consume.
        assert runner.state.get("turn_count") == 1
        assert runner.state.get("current_node") == "patient_id"
        assert runner.in_queue.empty()
    finally:
        await _shutdown(proc)


# (c) Handler exception ------------------------------------------------------


async def test_handler_exception_routes_to_fallback(
    patient: PatientInfo, fake_classifier: FakeClassifier
) -> None:
    """A raising LLM must land the call in `done` with FALLBACK_RESPONSE."""
    fake_llm = FakeLLMClient(structured_exception=RuntimeError("explode"))
    proc, runner, sink = await _make_processor(patient, fake_llm, fake_classifier)
    runner.state = {**runner.state, "current_node": "extract_benefits"}
    try:
        await proc.process_frame(_finalized("anything"), FrameDirection.DOWNSTREAM)
        await _wait_until(
            lambda: any(
                isinstance(f, TextFrame) and f.text == FALLBACK_RESPONSE for f in sink.frames
            ),
            timeout=1.0,
        )
        assert runner.state.get("current_node") == "done"
        assert runner.state.get("fallback_reason") == "node_exception"
    finally:
        await _shutdown(proc)


# (d) Contextvar propagation -------------------------------------------------


async def test_every_log_from_runner_has_call_sid(
    patient: PatientInfo, fake_classifier: FakeClassifier
) -> None:
    """Every log line emitted from the runner must have `call_sid` bound.

    `structlog.testing.capture_logs()` bypasses the processor chain, so it
    wouldn't see `merge_contextvars` (the whole point of the test). Instead
    install a capture processor that sits after merge_contextvars and inspect
    the fully-merged event dict.
    """
    captured: list[dict[str, object]] = []

    def _capture(_logger, _name, event_dict):  # noqa: ANN001
        captured.append(dict(event_dict))
        return event_dict

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            _capture,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(0),
        cache_logger_on_first_use=False,
    )
    try:
        fake_llm = FakeLLMClient(structured_exception=RuntimeError("observe-me"))
        proc, runner, _sink = await _make_processor(patient, fake_llm, fake_classifier)
        runner.state = {**runner.state, "current_node": "extract_benefits"}
        try:
            await proc.process_frame(_finalized("anything"), FrameDirection.DOWNSTREAM)
            await _wait_until(lambda: runner.state.get("current_node") == "done", timeout=1.0)
        finally:
            await _shutdown(proc)

        runner_logs = [e for e in captured if e.get("event") in {"node_error", "turn_cancelled"}]
        assert runner_logs, f"expected a runner log; got {captured!r}"
        assert any(
            e.get("event") == "node_error" and "observe-me" in str(e.get("error", ""))
            for e in runner_logs
        ), f"no node_error log captured: {runner_logs!r}"
        for entry in runner_logs:
            assert entry.get("call_sid") == "CA-m4", f"log line missing call_sid: {entry!r}"
    finally:
        # Restore the project's default logging config for other tests.
        from agent.logging_config import configure_logging

        configure_logging()


# Plumbing smoke tests -------------------------------------------------------


async def test_non_finalized_transcripts_are_ignored(
    patient: PatientInfo, fake_llm: FakeLLMClient, fake_classifier: FakeClassifier
) -> None:
    proc, runner, _sink = await _make_processor(patient, fake_llm, fake_classifier)
    try:
        await proc.process_frame(_interim("interim"), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0.1)
        assert runner.state.get("turn_count") == 0
        assert runner.in_queue.empty()
    finally:
        await _shutdown(proc)


async def test_response_emitted_as_text_frame_downstream(
    patient: PatientInfo, fake_llm: FakeLLMClient, fake_classifier: FakeClassifier
) -> None:
    proc, runner, sink = await _make_processor(patient, fake_llm, fake_classifier)
    try:
        await proc.process_frame(_finalized("ready"), FrameDirection.DOWNSTREAM)
        await _wait_until(lambda: _text_frames(sink), timeout=1.0)
        assert "Alice" in _text_frames(sink)[0].text
    finally:
        await _shutdown(proc)


async def test_end_frame_stops_runner(
    patient: PatientInfo, fake_llm: FakeLLMClient, fake_classifier: FakeClassifier
) -> None:
    proc, runner, _sink = await _make_processor(patient, fake_llm, fake_classifier)
    await proc.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)
    assert runner._consumer is not None and runner._consumer.done()


async def test_cancel_frame_stops_runner(
    patient: PatientInfo, fake_llm: FakeLLMClient, fake_classifier: FakeClassifier
) -> None:
    proc, runner, _sink = await _make_processor(patient, fake_llm, fake_classifier)
    await proc.process_frame(CancelFrame(), FrameDirection.DOWNSTREAM)
    assert runner._consumer is not None and runner._consumer.done()


async def test_frames_forwarded_downstream(
    patient: PatientInfo, fake_llm: FakeLLMClient, fake_classifier: FakeClassifier
) -> None:
    """All frames (Transcription, VAD, StartFrame, EndFrame) pass through."""
    proc, _runner, sink = await _make_processor(patient, fake_llm, fake_classifier)
    try:
        await proc.process_frame(_finalized("hi"), FrameDirection.DOWNSTREAM)
        await proc.process_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    finally:
        await _shutdown(proc)
    types = [type(f).__name__ for f in sink.frames]
    assert "StartFrame" in types
    assert "TranscriptionFrame" in types
    assert "UserStartedSpeakingFrame" in types
    assert "EndFrame" in types


# Silence the unused-import warning if pytest-asyncio's autouse changes.
pytestmark = pytest.mark.asyncio
