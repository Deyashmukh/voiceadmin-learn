"""Unit tests for `StateMachineProcessor`. Zero network."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

import pytest
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    StartFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from agent import tools
from agent.call_session import CallSessionRunner
from agent.processors.state_processor import StateMachineProcessor
from agent.schemas import CallSession, IVRTurnResponse, ToolCall

from .conftest import FakeActuator, FakeAnthropicRepClient, FakeIVRLLMClient


class _SinkProcessor(FrameProcessor):
    # Direct mode skips Pipecat's task setup so tests don't need a TaskManager.

    def __init__(self) -> None:
        super().__init__(enable_direct_mode=True)
        self.received: list[Frame] = []

    async def queue_frame(
        self,
        frame: Frame,
        direction: FrameDirection = FrameDirection.DOWNSTREAM,
        callback: object = None,
    ) -> None:
        self.received.append(frame)


async def _make_processor(
    make_session: Callable[..., CallSession],
    ivr_responses: list[IVRTurnResponse] | None = None,
) -> tuple[StateMachineProcessor, CallSessionRunner, _SinkProcessor]:
    session = make_session()
    ivr_llm = FakeIVRLLMClient(responses=list(ivr_responses or []))
    runner = CallSessionRunner(
        session=session,
        ivr_llm=ivr_llm,
        rep_llm=FakeAnthropicRepClient(),
        tool_dispatcher=tools.dispatch,
        ivr_system_prompt="x",
        rep_system_prompt="x",
        tools=[],
        actuator=FakeActuator(),
    )
    proc = StateMachineProcessor(runner, enable_direct_mode=True)
    sink = _SinkProcessor()
    proc.link(sink)
    return proc, runner, sink


async def _wait_until(predicate, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"timed out waiting on {predicate.__name__}")


def _finalized(text: str) -> TranscriptionFrame:
    """TranscriptionFrame in the shape Pipecat emits at end-of-utterance."""
    frame = TranscriptionFrame(text=text, user_id="user", timestamp="t")
    frame.finalized = True
    return frame


# --- Lifecycle ----------------------------------------------------------------


async def test_start_frame_starts_runner_and_pump(make_session):
    proc, runner, _ = await _make_processor(make_session)
    try:
        await proc.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
        assert runner._consumer is not None and not runner._consumer.done()
        assert proc._pump_task is not None and not proc._pump_task.done()
    finally:
        await proc.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)


async def test_end_frame_stops_runner_and_pump(make_session):
    proc, runner, _ = await _make_processor(make_session)
    await proc.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
    await proc.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)
    assert runner._consumer is None or runner._consumer.done()
    assert proc._pump_task is None


async def test_cancel_frame_stops_runner(make_session):
    proc, runner, _ = await _make_processor(make_session)
    await proc.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
    await proc.process_frame(CancelFrame(), FrameDirection.DOWNSTREAM)
    assert runner._consumer is None or runner._consumer.done()


# --- Frame routing ------------------------------------------------------------


async def test_finalized_transcription_enqueues_to_runner(make_session):
    ivr_llm_responses = [
        IVRTurnResponse(tool_calls=[ToolCall(name="complete_call", args={"reason": "user_hangup"})])
    ]
    proc, runner, _ = await _make_processor(make_session, ivr_responses=ivr_llm_responses)
    try:
        await proc.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
        await proc.process_frame(_finalized("hello"), FrameDirection.DOWNSTREAM)
        # The consumer pulls the transcript and runs a turn; queued response
        # ends the call. If routing or enqueue is broken the call never
        # terminates and the wait times out.
        await _wait_until(lambda: runner.session.done)
    finally:
        await proc.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)


async def test_non_finalized_transcription_is_ignored(make_session):
    proc, runner, _ = await _make_processor(make_session)
    try:
        await proc.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
        interim = TranscriptionFrame(text="hel", user_id="user", timestamp="t")
        interim.finalized = False
        await proc.process_frame(interim, FrameDirection.DOWNSTREAM)
        assert runner.in_queue.empty()
    finally:
        await proc.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)


async def test_empty_finalized_transcription_is_ignored(make_session):
    proc, runner, _ = await _make_processor(make_session)
    try:
        await proc.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
        await proc.process_frame(_finalized("   "), FrameDirection.DOWNSTREAM)
        assert runner.in_queue.empty()
    finally:
        await proc.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)


@pytest.mark.parametrize("interrupt_frame", [UserStartedSpeakingFrame(), InterruptionFrame()])
async def test_interrupt_frames_mark_interrupted(make_session, interrupt_frame):
    proc, runner, _ = await _make_processor(make_session)
    try:
        await proc.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
        await runner.out_queue.put("stale")
        await proc.process_frame(interrupt_frame, FrameDirection.DOWNSTREAM)
        assert runner.out_queue.empty()
    finally:
        await proc.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)


# --- Output pump -------------------------------------------------------------


async def test_pump_pushes_out_queue_text_downstream_as_textframe(make_session):
    proc, runner, sink = await _make_processor(make_session)
    try:
        await proc.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
        await runner.out_queue.put("hello there")
        await _wait_until(
            lambda: any(isinstance(f, TextFrame) and f.text == "hello there" for f in sink.received)
        )
    finally:
        await proc.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)
