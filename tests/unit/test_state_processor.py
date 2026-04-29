"""Unit tests for `StateMachineProcessor`. Zero network."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    StartFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from agent import tools
from agent.call_session import CallSessionRunner
from agent.processors.state_processor import StateMachineProcessor
from agent.schemas import CallSession, IVRTurnResponse, ToolCall

from .conftest import (
    FakeActuator,
    FakeAnthropicRepClient,
    FakeIVRLLMClient,
    MakeSession,
    wait_until,
)


class _SinkProcessor(FrameProcessor):
    # Direct mode skips Pipecat's task setup so tests don't need a TaskManager.

    def __init__(self) -> None:
        super().__init__(enable_direct_mode=True)  # pyright: ignore[reportUnknownMemberType] (Pipecat stub gap)
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
    vad_stopped_grace_s: float = 0.0,
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
    # Default 0s grace so transcripts flush as soon as VAD says stopped (or
    # immediately for late-arriving transcripts), without making tests pay
    # the production wait. The dedicated grace test overrides to a small
    # positive value.
    proc = StateMachineProcessor(
        runner,
        enable_direct_mode=True,
        vad_stopped_grace_s=vad_stopped_grace_s,
    )
    sink = _SinkProcessor()
    proc.link(sink)
    return proc, runner, sink


def _transcription(text: str) -> TranscriptionFrame:
    """TranscriptionFrame in the shape Pipecat emits at end-of-utterance.

    Pipecat's STT services (Deepgram, etc.) push `TranscriptionFrame` only for
    is_final results — interim results use `InterimTranscriptionFrame`. The
    `finalized` field is unrelated metadata for forced-finalize cases and is
    not set by streaming STT, so we don't set it here either.
    """
    return TranscriptionFrame(text=text, user_id="user", timestamp="t")


# --- Lifecycle ----------------------------------------------------------------


async def test_start_frame_starts_runner_and_pump(make_session: MakeSession):
    proc, runner, _ = await _make_processor(make_session)
    try:
        await proc.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
        assert runner._consumer is not None and not runner._consumer.done()  # pyright: ignore[reportPrivateUsage]
        assert proc._pump_task is not None and not proc._pump_task.done()  # pyright: ignore[reportPrivateUsage]
    finally:
        await proc.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)


async def test_end_frame_stops_runner_and_pump(make_session: MakeSession):
    proc, runner, _ = await _make_processor(make_session)
    await proc.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
    await proc.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)
    assert runner._consumer is None or runner._consumer.done()  # pyright: ignore[reportPrivateUsage]
    assert proc._pump_task is None  # pyright: ignore[reportPrivateUsage]


async def test_cancel_frame_stops_runner(make_session: MakeSession):
    proc, runner, _ = await _make_processor(make_session)
    await proc.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
    await proc.process_frame(CancelFrame(), FrameDirection.DOWNSTREAM)
    assert runner._consumer is None or runner._consumer.done()  # pyright: ignore[reportPrivateUsage]


# --- Frame routing ------------------------------------------------------------


async def test_transcription_enqueues_to_runner(make_session: MakeSession):
    ivr_llm_responses = [
        IVRTurnResponse(tool_calls=[ToolCall(name="complete_call", args={"reason": "user_hangup"})])
    ]
    proc, runner, _ = await _make_processor(make_session, ivr_responses=ivr_llm_responses)
    try:
        await proc.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
        await proc.process_frame(_transcription("hello"), FrameDirection.DOWNSTREAM)
        # The consumer pulls the transcript and runs a turn; queued response
        # ends the call. If routing or enqueue is broken the call never
        # terminates and the wait times out.
        await wait_until(lambda: runner.session.done)
    finally:
        await proc.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)


async def test_interim_transcription_is_not_enqueued(make_session: MakeSession):
    proc, runner, _ = await _make_processor(make_session)
    try:
        await proc.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
        interim = InterimTranscriptionFrame(text="hel", user_id="user", timestamp="t")
        await proc.process_frame(interim, FrameDirection.DOWNSTREAM)
        assert runner.in_queue.empty()
    finally:
        await proc.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)


async def test_vad_driven_flush_aggregates_menu_into_one_submission(
    make_session: MakeSession,
):
    """Multi-sentence menu: three transcripts arrive while VAD reports
    speech is still active (caller still talking through the menu). When
    VAD signals stopped, the processor flushes the buffer as a single
    joined submission after the grace period.

    No StartFrame — we want to inspect what `submit_transcript` receives
    without the consumer immediately draining `in_queue`.
    """
    proc, runner, _ = await _make_processor(make_session, vad_stopped_grace_s=0.05)
    # Caller is mid-menu — speech_active set, no flush should fire on
    # transcript arrival.
    await proc.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await proc.process_frame(_transcription("Press 1 for benefits."), FrameDirection.DOWNSTREAM)
    await proc.process_frame(_transcription("Press 2 for claims."), FrameDirection.DOWNSTREAM)
    await proc.process_frame(_transcription("Press 9 for a rep."), FrameDirection.DOWNSTREAM)
    assert runner.in_queue.empty()
    # Caller stops speaking — flush fires after the grace period.
    await proc.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await wait_until(lambda: not runner.in_queue.empty(), timeout=5.0)
    combined = runner.in_queue.get_nowait()
    assert combined == "Press 1 for benefits. Press 2 for claims. Press 9 for a rep."
    assert runner.in_queue.empty()


async def test_vad_restart_during_grace_cancels_pending_flush(make_session: MakeSession):
    """Inter-option pause: VAD reports stopped, then re-started before the
    grace period elapses. The pending flush must cancel — fragmenting a
    menu into two turns is exactly the bug VAD-driven flushing prevents.
    """
    proc, runner, _ = await _make_processor(make_session, vad_stopped_grace_s=0.2)
    await proc.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await proc.process_frame(_transcription("Press 1 for benefits."), FrameDirection.DOWNSTREAM)
    await proc.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    # Resume speaking before the 200ms grace expires.
    await asyncio.sleep(0.05)
    await proc.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await proc.process_frame(_transcription("Press 9 for a rep."), FrameDirection.DOWNSTREAM)
    await proc.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    # Now wait for the actual final flush.
    await wait_until(lambda: not runner.in_queue.empty(), timeout=5.0)
    combined = runner.in_queue.get_nowait()
    assert combined == "Press 1 for benefits. Press 9 for a rep."
    assert runner.in_queue.empty()


async def test_empty_transcription_is_ignored(make_session: MakeSession):
    proc, runner, _ = await _make_processor(make_session)
    try:
        await proc.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
        await proc.process_frame(_transcription("   "), FrameDirection.DOWNSTREAM)
        assert runner.in_queue.empty()
    finally:
        await proc.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)


async def test_interruption_frame_marks_interrupted_unconditionally(
    make_session: MakeSession,
):
    """Explicit `InterruptionFrame` always interrupts — no bot-speaking gate."""
    proc, runner, _ = await _make_processor(make_session)
    try:
        await proc.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
        await runner.out_queue.put("stale")
        await proc.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
        assert runner.out_queue.empty()
    finally:
        await proc.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)


@pytest.mark.parametrize("vad_frame", [UserStartedSpeakingFrame(), VADUserStartedSpeakingFrame()])
async def test_vad_speech_start_only_interrupts_while_bot_speaking(
    make_session: MakeSession, vad_frame: Frame
):
    """VAD-driven speech-start frames only mark an interrupt while the agent
    is actually playing TTS audio (bracketed by Bot{Started,Stopped}SpeakingFrame).
    Otherwise — e.g. during a multi-sentence IVR menu where the caller
    naturally pauses between options — the buffer must NOT be drained.
    """
    proc, runner, _ = await _make_processor(make_session)
    try:
        await proc.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
        # Bot NOT speaking yet → VAD speech-start is a no-op.
        await runner.out_queue.put("stale-1")
        await proc.process_frame(vad_frame, FrameDirection.DOWNSTREAM)
        assert not runner.out_queue.empty()
        # Bot starts speaking → VAD speech-start now interrupts.
        await proc.process_frame(BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await proc.process_frame(vad_frame, FrameDirection.DOWNSTREAM)
        assert runner.out_queue.empty()
        # Bot stops speaking → VAD speech-start is a no-op again.
        await proc.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await runner.out_queue.put("stale-2")
        await proc.process_frame(vad_frame, FrameDirection.DOWNSTREAM)
        assert not runner.out_queue.empty()
    finally:
        await proc.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)


async def test_vad_barge_in_pushes_interruption_frame_downstream(
    make_session: MakeSession,
):
    """When VAD-gated barge-in fires, an `InterruptionFrame` must be pushed
    downstream so Pipecat's TTS service stops mid-synthesis and the output
    transport drops any buffered audio. Without this, the agent keeps
    talking over the user for several beats after the interrupt is logged.
    """
    proc, runner, sink = await _make_processor(make_session)
    try:
        await proc.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
        await proc.process_frame(BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await proc.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        # Two InterruptionFrames reach the sink: the one we synthesized
        # for the downstream TTS-clear, plus the original VAD frame as it
        # passes through (push_frame at the end of process_frame). We
        # assert at least one InterruptionFrame is present.
        assert any(isinstance(f, InterruptionFrame) for f in sink.received), (
            f"expected InterruptionFrame in {[type(f).__name__ for f in sink.received]}"
        )
        # Also re-assert the runner side: queues drained.
        await runner.out_queue.put("would-be-stale")
        await proc.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        assert runner.out_queue.empty()
    finally:
        await proc.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)


# --- Output pump -------------------------------------------------------------


async def test_pump_pushes_out_queue_text_downstream_as_tts_speak_frame(
    make_session: MakeSession,
):
    proc, runner, sink = await _make_processor(make_session)
    try:
        await proc.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
        await runner.out_queue.put("hello there")
        await wait_until(
            lambda: any(
                isinstance(f, TTSSpeakFrame) and f.text == "hello there" for f in sink.received
            )
        )
    finally:
        await proc.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)
