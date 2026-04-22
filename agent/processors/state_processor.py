"""Pipecat adapter for `GraphRunner`.

Kept deliberately thin. The rule from `CLAUDE.md` is absolute: LangGraph must
not run inside `process_frame`. This processor only touches the runner via
bounded queues and `Task.cancel()`, so audio-loop latency is unaffected by
LLM work.
"""

from __future__ import annotations

import asyncio
from typing import Any

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

from agent.graph_runner import GraphRunner
from agent.logging_config import log


class StateMachineProcessor(FrameProcessor):
    """Bridge Pipecat frames to the graph runner.

    - Finalized `TranscriptionFrame`s are enqueued on `runner.in_queue`.
    - `UserStartedSpeakingFrame` / `InterruptionFrame` cancel the in-flight turn.
    - Responses from `runner.out_queue` are pushed downstream as `TextFrame`s.
    - The runner's lifecycle is pinned to `StartFrame` / `EndFrame` / `CancelFrame`.
    """

    def __init__(self, runner: GraphRunner, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._runner = runner
        self._pump_task: asyncio.Task[None] | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self._start()
        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self._stop()
        elif isinstance(frame, TranscriptionFrame):
            if frame.finalized and frame.text.strip():
                self._runner.submit_transcript(frame.text)
        elif isinstance(frame, (UserStartedSpeakingFrame, InterruptionFrame)):
            self._runner.mark_interrupted()

        await self.push_frame(frame, direction)

    async def _start(self) -> None:
        if self._pump_task is not None:
            return
        await self._runner.start()
        self._pump_task = asyncio.create_task(self._pump_output(), name="state-processor-pump")
        log.info("state_processor_started")

    async def _stop(self) -> None:
        if self._pump_task is not None:
            self._pump_task.cancel()
            try:
                await self._pump_task
            except asyncio.CancelledError:
                pass
            self._pump_task = None
        await self._runner.stop()
        log.info("state_processor_stopped")

    async def _pump_output(self) -> None:
        while True:
            response = await self._runner.out_queue.get()
            await self.push_frame(TextFrame(text=response), FrameDirection.DOWNSTREAM)
