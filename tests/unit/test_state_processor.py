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


async def test_late_transcript_after_grace_schedules_fresh_flush(make_session: MakeSession):
    """Deepgram can lag VAD by hundreds of ms. If a TranscriptionFrame
    arrives AFTER the VAD-stopped grace already expired (and the flush
    ran on an empty buffer), the late transcript must trigger a fresh
    flush — not sit in the buffer forever waiting for the next
    VAD-stopped that may never come.

    Locks the late-transcript fallback: removing the
    `if not self._speech_active and self._flush_task is None: schedule`
    branch in process_frame would silently drop late transcripts.
    """
    proc, runner, _ = await _make_processor(make_session, vad_stopped_grace_s=0.05)
    await proc.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await proc.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    # Wait for the grace to expire on an empty buffer. The flush task
    # runs, finds the buffer empty, returns early, and clears
    # `_flush_task` in its `finally`.
    await wait_until(
        lambda: proc._flush_task is None,  # pyright: ignore[reportPrivateUsage]
        timeout=2.0,
    )
    assert runner.in_queue.empty()
    # Now Deepgram catches up with a late final.
    await proc.process_frame(_transcription("late text"), FrameDirection.DOWNSTREAM)
    await wait_until(lambda: not runner.in_queue.empty(), timeout=2.0)
    assert runner.in_queue.get_nowait() == "late text"


async def test_flush_preserves_buffer_when_submit_raises(make_session: MakeSession):
    """If `submit_transcript` raises (closed queue, future refactor), the
    buffer must NOT be cleared — submit-then-clear ordering means a raise
    leaves the data intact for the next flush. Pre-fix, the buffer was
    cleared first and the data went on the floor.

    Asserts the next successful flush still sees the original transcript.
    """
    proc, runner, _ = await _make_processor(make_session, vad_stopped_grace_s=0.05)
    # First flush: monkey-patch submit_transcript to raise.
    original_submit = runner.submit_transcript

    def submit_then_fail(text: str) -> None:
        raise RuntimeError("simulated queue failure")

    runner.submit_transcript = submit_then_fail  # pyright: ignore[reportAttributeAccessIssue]

    await proc.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await proc.process_frame(_transcription("Press 1 for benefits."), FrameDirection.DOWNSTREAM)
    await proc.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    # Wait for the flush task to actually run (and raise into the asyncio
    # task — that's the bug we're proving doesn't lose data).
    await wait_until(
        lambda: proc._flush_task is None,  # pyright: ignore[reportPrivateUsage]
        timeout=2.0,
    )
    # Buffer must still hold the unsubmitted text.
    assert proc._transcript_buffer == ["Press 1 for benefits."]  # pyright: ignore[reportPrivateUsage]

    # Restore submit_transcript so the next flush succeeds. Add a fresh
    # transcript and trigger another stop — the original text should be
    # part of the eventual submission.
    runner.submit_transcript = original_submit  # pyright: ignore[reportAttributeAccessIssue]
    await proc.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await proc.process_frame(_transcription("Press 9 for a rep."), FrameDirection.DOWNSTREAM)
    await proc.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await wait_until(lambda: not runner.in_queue.empty(), timeout=5.0)
    combined = runner.in_queue.get_nowait()
    assert combined == "Press 1 for benefits. Press 9 for a rep."


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
    """Explicit `InterruptionFrame` always interrupts — no bot-speaking gate.

    Asserts EXACTLY ONE `InterruptionFrame` lands at the sink (the original,
    re-emitted by the trailing `push_frame` in `process_frame`). The
    explicit branch must NOT also synthesize a second one — `_handle_barge_in`
    is called with `downstream_interruption=False` precisely to avoid that
    double-push. A future flip to `True` here would put two
    `InterruptionFrame`s downstream and confuse the TTS service.
    """
    proc, runner, sink = await _make_processor(make_session)
    try:
        await proc.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
        await runner.out_queue.put("stale")
        baseline_interrupts = sum(1 for f in sink.received if isinstance(f, InterruptionFrame))
        await proc.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
        new_interrupts = (
            sum(1 for f in sink.received if isinstance(f, InterruptionFrame)) - baseline_interrupts
        )
        assert new_interrupts == 1, (
            f"expected exactly one InterruptionFrame at sink (the re-emitted "
            f"original), got {new_interrupts}; sink contents: "
            f"{[type(f).__name__ for f in sink.received]}"
        )
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

    Asserts EXACTLY ONE synthesized `InterruptionFrame` lands at the sink
    per barge-in — not just `>=1` — so a future refactor that double-
    pushes (or drops the synthesis) is caught immediately. The trailing
    `await self.push_frame(frame, direction)` in `process_frame` re-emits
    the original `VADUserStartedSpeakingFrame`, which is a different type
    and doesn't satisfy the assertion.
    """
    proc, runner, sink = await _make_processor(make_session)
    try:
        await proc.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
        await proc.process_frame(BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        baseline_interrupts = sum(1 for f in sink.received if isinstance(f, InterruptionFrame))
        await proc.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        new_interrupts = (
            sum(1 for f in sink.received if isinstance(f, InterruptionFrame)) - baseline_interrupts
        )
        assert new_interrupts == 1, (
            f"expected exactly one new InterruptionFrame after barge-in, "
            f"got {new_interrupts}; sink contents: "
            f"{[type(f).__name__ for f in sink.received]}"
        )
        # Re-assert the runner side: queues drained on barge-in. Use a
        # fresh BotStartedSpeakingFrame because the prior barge-in left
        # `_bot_speaking` False (no Stopped frame fired).
        await proc.process_frame(BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
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


async def test_pump_gives_up_after_consecutive_failures_and_marks_session(
    make_session: MakeSession,
):
    """If the downstream pipeline is gone (transport torn down), each
    `push_frame` from `_pump_output` raises. After `_PUMP_FAILURE_GIVE_UP`
    consecutive failures the pump must stop trying and set
    `completion_reason='pipeline_torn_down'` so the call session terminates
    instead of consuming `out_queue` forever and discarding every reply.
    """
    proc, runner, _sink = await _make_processor(make_session)
    # Capture the original BEFORE the try so the finally always sees it.
    original_push = proc.push_frame

    async def always_fail(frame: Frame, direction: FrameDirection) -> None:
        if isinstance(frame, TTSSpeakFrame):
            raise RuntimeError("downstream gone")
        await original_push(frame, direction)

    try:
        await proc.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
        # Replace push_frame so every TTSSpeakFrame pump-out raises. After
        # `_PUMP_FAILURE_GIVE_UP` failures the pump must give up and mark
        # the session.
        proc.push_frame = always_fail  # pyright: ignore[reportAttributeAccessIssue]
        for i in range(StateMachineProcessor._PUMP_FAILURE_GIVE_UP):  # pyright: ignore[reportPrivateUsage]
            await runner.out_queue.put(f"reply-{i}")
        await wait_until(
            lambda: runner.session.completion_reason == "pipeline_torn_down",
            timeout=2.0,
        )
    finally:
        # Restore so EndFrame-on-the-way-out can flow through.
        proc.push_frame = original_push  # pyright: ignore[reportAttributeAccessIssue]
        await proc.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)
