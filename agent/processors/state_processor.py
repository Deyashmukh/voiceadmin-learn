"""Pipecat adapter for `CallSessionRunner`.

Kept deliberately thin. The CLAUDE.md non-negotiable: the call loop must not
run inside `process_frame`. This processor only touches the runner via
bounded queues and `Task.cancel()`, so audio-loop latency is unaffected by
LLM work.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    StartFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from agent.call_session import CallSessionRunner
from agent.logging_config import log

_VAD_STOPPED_GRACE_S = 0.7
"""Cooldown after VAD signals end-of-speech before flushing buffered
transcripts. 0.7s is longer than typical inter-option pauses in recorded
payer IVRs (300-500ms) — so a multi-sentence menu doesn't fragment — but
short enough that conversation in rep mode feels live."""


class StateMachineProcessor(FrameProcessor):
    """Bridge Pipecat frames to the call session runner.

    Transcript flushing is VAD-driven, not time-driven: we buffer
    `TranscriptionFrame`s as they arrive and flush them as a single
    submission `_VAD_STOPPED_GRACE_S` after VAD signals end-of-speech.
    A new VAD-started before the grace period elapses cancels the flush
    (the speaker was just pausing between options). This adapts to the
    actual cadence of the caller — short menus flush quickly, long
    multi-sentence menus aggregate naturally.

    Other responsibilities:
    - `UserStartedSpeakingFrame` / `InterruptionFrame` (only while the bot
      is mid-TTS) cancel the in-flight turn AND propagate
      `InterruptionFrame` downstream so Pipecat's TTS and output transport
      stop synthesizing / streaming any tail audio.
    - Responses from `runner.out_queue` are pushed downstream as
      `TTSSpeakFrame`s.
    - The runner's lifecycle is pinned to `StartFrame` / `EndFrame` /
      `CancelFrame`.
    """

    def __init__(
        self,
        runner: CallSessionRunner,
        *,
        vad_stopped_grace_s: float = _VAD_STOPPED_GRACE_S,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)  # pyright: ignore[reportUnknownMemberType] (Pipecat stub gap)
        self._runner = runner
        self._pump_task: asyncio.Task[None] | None = None
        self._transcript_buffer: list[str] = []
        self._flush_task: asyncio.Task[None] | None = None
        self._vad_stopped_grace_s = vad_stopped_grace_s
        # Tracks whether the agent's TTS is currently being played to the
        # caller. Barge-in (VAD-driven interrupt) is only meaningful while
        # the agent is talking — VAD fires on every breath pause during a
        # multi-sentence IVR menu, and treating each as an interrupt drains
        # the transcript buffer and kills the turn.
        self._bot_speaking: bool = False
        # Tracks whether the caller (IVR/rep) is currently speaking, per
        # VAD. Used to decide when to flush buffered transcripts.
        self._speech_active: bool = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self._start()
        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self._stop()
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
        elif isinstance(frame, InterruptionFrame):
            # Explicit InterruptionFrame is unconditional — caller has
            # decided this is a real interrupt.
            await self._handle_barge_in(downstream_interruption=False)
        elif isinstance(frame, (UserStartedSpeakingFrame, VADUserStartedSpeakingFrame)):
            self._speech_active = True
            # New speech started — cancel any pending flush; this could be
            # a brief pause between menu options, OR (if the bot is
            # speaking) a real barge-in that needs to drain the buffer.
            self._cancel_flush()
            if self._bot_speaking:
                await self._handle_barge_in(downstream_interruption=True)
        elif isinstance(frame, (UserStoppedSpeakingFrame, VADUserStoppedSpeakingFrame)):
            self._speech_active = False
            # Speaker just stopped — schedule a flush after a short grace
            # period in case it's just an inter-option pause.
            self._schedule_flush()
        elif isinstance(frame, TranscriptionFrame) and frame.text.strip():
            self._transcript_buffer.append(frame.text.strip())
            # Late transcript (Deepgram lags VAD): if speech has already
            # ended, ensure a flush is pending. Otherwise we'll wait for
            # the upcoming VAD-stopped event.
            if not self._speech_active and self._flush_task is None:
                self._schedule_flush()

        await self.push_frame(frame, direction)

    async def _handle_barge_in(self, *, downstream_interruption: bool) -> None:
        """Cancel the in-flight runner turn and (optionally) synthesize an
        InterruptionFrame downstream so Pipecat's TTS service stops
        mid-synthesis and the output transport clears any buffered audio.

        Without the downstream propagation, mark_interrupted only stops
        the runner from generating *new* turns — bytes already queued in
        the WSS output buffer keep playing and the user keeps hearing
        the agent for a beat after they barged in.

        Note on the `downstream_interruption` flag: BOTH branches fall
        through to the trailing `await self.push_frame(frame, direction)`
        in `process_frame`, so the original incoming frame ALWAYS reaches
        downstream. The flag only controls whether we *also* synthesize a
        new InterruptionFrame:
        - explicit-InterruptionFrame branch: no synthesis (the re-emit
          already covers it; synthesizing would push two).
        - VAD-driven branch: synthesize, because the original is a
          VADUserStartedSpeakingFrame which downstream TTS/transport
          don't react to as an interrupt signal.
        """
        self._cancel_flush()
        self._transcript_buffer.clear()
        self._runner.mark_interrupted()
        if downstream_interruption:
            try:
                await self.push_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Runner is already cancelled but TTS may still be playing
                # buffered audio. The user will hear the agent finish
                # talking after their barge-in — a UX defect. Log loudly
                # so a stuck-tail-audio symptom is greppable.
                log.error("barge_in_downstream_push_failed", error=str(exc))

    def _schedule_flush(self) -> None:
        """Arm a delayed flush of the transcript buffer. Called when VAD
        signals the speaker has stopped — the grace period catches an
        inter-option pause vs. a true end-of-utterance.
        """
        self._cancel_flush()
        self._flush_task = asyncio.create_task(
            self._flush_after_quiet(self._vad_stopped_grace_s),
            name="state-processor-flush",
        )

    def _cancel_flush(self) -> None:
        if self._flush_task is not None and not self._flush_task.done():
            self._flush_task.cancel()
        self._flush_task = None

    async def _flush_after_quiet(self, delay: float) -> None:
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return
        try:
            if not self._transcript_buffer:
                return
            combined = " ".join(self._transcript_buffer)
            # Submit BEFORE clearing — `submit_transcript` is sync today and
            # can't raise, but a future raise (closed queue, refactor) would
            # otherwise drop the buffered text on the floor. Submit-then-clear
            # means a raise leaves the buffer intact for the next flush.
            self._runner.submit_transcript(combined)
            self._transcript_buffer.clear()
        finally:
            # Clear the reference so a late-arriving transcript (Deepgram
            # lagged past the VAD-stopped grace) can schedule a fresh flush.
            self._flush_task = None

    async def _start(self) -> None:
        if self._pump_task is not None:
            return
        await self._runner.start()
        self._pump_task = asyncio.create_task(self._pump_output(), name="state-processor-pump")
        log.info("state_processor_started")

    async def _stop(self) -> None:
        self._cancel_flush()
        self._transcript_buffer.clear()
        if self._pump_task is not None:
            self._pump_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._pump_task
            self._pump_task = None
        await self._runner.stop()
        log.info("state_processor_stopped")

    _PUMP_FAILURE_GIVE_UP = 3
    """Consecutive `push_frame` failures before we stop retrying — a single
    failure can be a transient frame-routing race; three in a row means
    the transport is torn down and further attempts are a waste of CPU
    and log volume."""

    async def _pump_output(self) -> None:
        consecutive_failures = 0
        given_up = False
        while True:
            response = await self._runner.out_queue.get()
            if given_up:
                # Once we've given up retrying, KEEP draining `out_queue` —
                # the runner's actuator does a blocking `out_queue.put()`
                # (intentional TTS backpressure), so a stalled pump would
                # deadlock the runner once 8 SpeakIntents pile up. Drop the
                # response on the floor; the user can't hear it anyway.
                # Actual call termination still relies on transport
                # disconnect → `runner.stop()`; `pipeline_torn_down` is
                # only a label on the eventual JSONL deliverable.
                continue
            try:
                # `TTSSpeakFrame` (vs raw `TextFrame`) is the explicit
                # "synthesize this now" signal: it bypasses the TTS service's
                # sentence-aggregation buffer and fires `run_tts` directly.
                # Pipecat also pauses frame processing during the synthesis to
                # prevent overlapping audio — that's our concurrency guard for
                # back-to-back agent utterances.
                await self.push_frame(TTSSpeakFrame(text=response), FrameDirection.DOWNSTREAM)
                consecutive_failures = 0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # A discarded TTS response is a user-visible defect (the
                # agent's reply never reaches the caller), not a warning —
                # log at error level.
                consecutive_failures += 1
                log.error(
                    "pump_push_failed",
                    error_class=type(exc).__name__,
                    error=str(exc),
                    consecutive_failures=consecutive_failures,
                )
                if consecutive_failures >= self._PUMP_FAILURE_GIVE_UP:
                    if self._runner.session.completion_reason is None:
                        self._runner.session.completion_reason = "pipeline_torn_down"
                    log.error(
                        "pump_giving_up",
                        consecutive_failures=consecutive_failures,
                    )
                    given_up = True
