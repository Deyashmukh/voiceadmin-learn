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
    VADUserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from agent.call_session import CallSessionRunner
from agent.logging_config import log

_TRANSCRIPT_DEBOUNCE_S = 2.5
"""Quiet window after the last transcript before flushing the buffered text as
a single turn. IVR menus arrive as 3-5 separate Deepgram finals (one per
sentence with natural pauses between them); without aggregation each becomes
its own LLM turn and the agent reacts to fragments. 2.5s spans the typical
inter-sentence pause in a recorded payer IVR menu, while still feeling
responsive once the menu finishes."""


class StateMachineProcessor(FrameProcessor):
    """Bridge Pipecat frames to the call session runner.

    - Finalized `TranscriptionFrame`s are buffered and flushed as a single
      submission once the speaker has been quiet for `_TRANSCRIPT_DEBOUNCE_S`.
    - `UserStartedSpeakingFrame` / `InterruptionFrame` cancel the in-flight turn.
    - Responses from `runner.out_queue` are pushed downstream as `TTSSpeakFrame`s.
    - The runner's lifecycle is pinned to `StartFrame` / `EndFrame` / `CancelFrame`.
    """

    def __init__(
        self,
        runner: CallSessionRunner,
        *,
        transcript_debounce_s: float = _TRANSCRIPT_DEBOUNCE_S,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)  # pyright: ignore[reportUnknownMemberType] (Pipecat stub gap)
        self._runner = runner
        self._pump_task: asyncio.Task[None] | None = None
        self._transcript_buffer: list[str] = []
        self._flush_task: asyncio.Task[None] | None = None
        self._transcript_debounce_s = transcript_debounce_s
        # Tracks whether the agent's TTS is currently being played to the
        # caller. Barge-in (VAD-driven interrupt) is only meaningful while
        # the agent is talking — VAD fires on every breath pause during a
        # multi-sentence IVR menu, and treating each as an interrupt drains
        # the transcript buffer and kills the turn.
        self._bot_speaking: bool = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self._start()
        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self._stop()
        elif isinstance(frame, TranscriptionFrame):
            if frame.text.strip():
                self._buffer_transcript(frame.text.strip())
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
        elif isinstance(frame, InterruptionFrame):
            # Explicit InterruptionFrame is unconditional — caller has
            # decided this is a real interrupt.
            self._cancel_flush()
            self._transcript_buffer.clear()
            self._runner.mark_interrupted()
        elif (
            isinstance(frame, (UserStartedSpeakingFrame, VADUserStartedSpeakingFrame))
            and self._bot_speaking
        ):
            # VAD-driven barge-in: only act if the agent is currently
            # speaking. Otherwise this is just the caller talking to the
            # IVR with normal inter-sentence pauses and we should NOT drop
            # the buffered transcripts.
            self._cancel_flush()
            self._transcript_buffer.clear()
            self._runner.mark_interrupted()

        await self.push_frame(frame, direction)

    def _buffer_transcript(self, text: str) -> None:
        """Append a finalized transcript and (re)arm the debounce timer."""
        self._transcript_buffer.append(text)
        self._cancel_flush()
        self._flush_task = asyncio.create_task(
            self._flush_after_quiet(self._transcript_debounce_s),
            name="state-processor-debounce",
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
        if not self._transcript_buffer:
            return
        combined = " ".join(self._transcript_buffer)
        self._transcript_buffer.clear()
        self._runner.submit_transcript(combined)

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

    async def _pump_output(self) -> None:
        while True:
            response = await self._runner.out_queue.get()
            try:
                # `TTSSpeakFrame` (vs raw `TextFrame`) is the explicit
                # "synthesize this now" signal: it bypasses the TTS service's
                # sentence-aggregation buffer and fires `run_tts` directly.
                # Pipecat also pauses frame processing during the synthesis to
                # prevent overlapping audio — that's our concurrency guard for
                # back-to-back agent utterances.
                await self.push_frame(TTSSpeakFrame(text=response), FrameDirection.DOWNSTREAM)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # If downstream tears down before _stop() runs, push_frame can
                # raise. Log and keep consuming so the runner isn't back-pressured
                # on a full out_queue forever.
                log.warning("pump_push_failed", error=str(exc))
