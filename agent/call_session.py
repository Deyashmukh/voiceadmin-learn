"""Per-turn loop for the two-mode call session.

Lives alongside the Pipecat pipeline (not inside a FrameProcessor) so LLM
latency never blocks the audio loop. `mark_interrupted()` cancels the in-
flight turn task — actual `asyncio.Task.cancel()`, not a flag check — so the
LLM `await` raises `CancelledError` and the handler aborts cleanly.

Currently routes all turns to `_ivr_turn`; mode-aware dispatch and rep mode
are added in later milestones.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

import structlog

from agent.actuator import Actuator, CallActuator
from agent.logging_config import log
from agent.schemas import (
    CallSession,
    IVRTurnResponse,
    ToolCall,
    ToolResult,
    Turn,
)

QUEUE_MAX = 8
# After 2 IVR turns where no tool advanced call state, terminate. Distinct
# from `llm_aborted_ivr` (LLM-deliberate) — this is the "the LLM is spinning
# on validator rejections or producing no tool calls at all" case.
IVR_NO_PROGRESS_LIMIT = 2


class IVRLLMClient(Protocol):
    """Tool-calling LLM for IVR mode. Real Groq implementation lands as a
    follow-up; tests inject `FakeIVRLLMClient`."""

    async def complete_with_tools(
        self,
        system: str,
        history: list[Turn],
        # TODO(D1.5): replace `dict[str, Any]` with the Groq tool-schema TypedDict
        # once the real client lands.
        tools: list[dict[str, Any]],
        temperature: float = 0.1,
    ) -> IVRTurnResponse: ...


# Type alias for the dispatcher callable so callers can inject `agent.tools.dispatch`
# directly or pass a wrapper for testing.
ToolDispatcher = Callable[[ToolCall, CallSession], Awaitable[ToolResult]]


class CallSessionRunner:
    """One per call. Spawned on Pipecat transport-connect, stopped on
    transport-disconnect. Never process-global."""

    def __init__(
        self,
        session: CallSession,
        ivr_llm: IVRLLMClient,
        tool_dispatcher: ToolDispatcher,
        ivr_system_prompt: str,
        tools: list[dict[str, Any]],
        *,
        actuator: Actuator | None = None,
        twilio_client: Any = None,
        in_queue_size: int = QUEUE_MAX,
        out_queue_size: int = QUEUE_MAX,
    ) -> None:
        self.session = session
        self.ivr_llm = ivr_llm
        self.dispatch = tool_dispatcher
        self.ivr_system_prompt = ivr_system_prompt
        self.tools = tools
        self.in_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=in_queue_size)
        self.out_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=out_queue_size)
        # Default actuator wires SpeakIntent → our own out_queue. Tests pass an
        # explicit Fake when they want to assert on intents directly. Production
        # passes a `twilio_client` so DTMFIntent dispatches to a live call.
        self.actuator: Actuator = actuator or CallActuator(
            session=session, out_queue=self.out_queue, twilio_client=twilio_client
        )
        self._consumer: asyncio.Task[None] | None = None
        self._current_turn: asyncio.Task[None] | None = None
        self._interrupt_requested: bool = False

    # --- Public API (called from Pipecat side) ------------------------------

    def submit_transcript(self, text: str) -> None:
        """Non-blocking enqueue with drop-oldest on full. Stale transcripts
        are worthless — keep the newest."""
        try:
            self.in_queue.put_nowait(text)
        except asyncio.QueueFull:
            try:
                _ = self.in_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self.in_queue.put_nowait(text)

    def mark_interrupted(self) -> None:
        """Cancel the in-flight turn AND drain both queues. A turn that
        finished moments before the user started speaking would otherwise
        still be spoken after the barge-in (out_queue), and old queued
        transcripts (in_queue) would feed the next turn instead of waiting
        for a fresh user utterance — both defeat the barge-in."""
        for queue in (self.out_queue, self.in_queue):
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
        if self._current_turn and not self._current_turn.done():
            self._interrupt_requested = True
            self._current_turn.cancel()

    async def start(self) -> None:
        self._consumer = asyncio.create_task(self._consume(), name="call-session-consume")
        self._consumer.add_done_callback(self._on_consumer_done)

    async def stop(self) -> None:
        if self._current_turn and not self._current_turn.done():
            self._current_turn.cancel()
        if self._consumer and not self._consumer.done():
            self._consumer.cancel()
        for task in (self._current_turn, self._consumer):
            if task is None:
                continue
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as exc:  # noqa: BLE001
                # A task that crashed during shutdown is still a real bug.
                log.warning("task_error_during_stop", task=task.get_name(), error=str(exc))

    # --- Internal loop ------------------------------------------------------

    async def _consume(self) -> None:
        structlog.contextvars.bind_contextvars(call_sid=self.session.call_sid, turn_index=0)
        while True:
            # Each transcript = one turn. Staleness only applies on barge-in,
            # which mark_interrupted handles by draining in_queue. If turns
            # back up because the LLM is slow, in_queue's drop-oldest-on-full
            # bounds growth.
            transcript = await self.in_queue.get()
            structlog.contextvars.bind_contextvars(turn_index=self.session.turn_count)
            self._current_turn = asyncio.create_task(
                self._run_turn(transcript), name=f"turn-{self.session.turn_count}"
            )
            try:
                await self._current_turn
            except asyncio.CancelledError:
                if self._interrupt_requested:
                    self._interrupt_requested = False
                    log.info("turn_cancelled", reason="interrupt")
                    continue
                raise
            # After a successful (non-cancelled) turn: if it terminated the
            # call, exit before re-entering `in_queue.get()` — otherwise the
            # consumer would block forever waiting for a transcript that
            # never comes.
            if self.session.done:
                log.info("call_session_complete", reason=self.session.completion_reason)
                return

    async def _run_turn(self, transcript: str) -> None:
        self.session.history.append(Turn(role="user", content=transcript))
        # If the turn is cancelled mid-flight (barge-in), turn_count is NOT
        # incremented and the watchdog counter is NOT touched — barge-in is
        # a re-do, not a "tried and failed" turn.
        await self._ivr_turn()
        self.session.turn_count += 1

    async def _ivr_turn(self) -> None:
        response = await self.ivr_llm.complete_with_tools(
            system=self.ivr_system_prompt,
            history=self.session.history,
            tools=self.tools,
            temperature=0.1,
        )
        advanced = False
        for call in response.tool_calls:
            # Pair each tool_call with a tool_result entry — even on cancellation
            # so the history stays well-formed (Groq's tool API rejects mismatched
            # tool_call / tool_result pairing on the next turn).
            tool_call_turn = Turn(role="tool_call", tool_call=call)
            self.session.history.append(tool_call_turn)
            try:
                result = await self.dispatch(call, self.session)
            except asyncio.CancelledError:
                self.session.history.append(
                    Turn(
                        role="tool_result",
                        content="cancelled before dispatch completed",
                    )
                )
                raise
            self.session.history.append(
                Turn(role="tool_result", tool_result=result, content=result.message)
            )
            if result.side_effect is not None:
                try:
                    await self.actuator.execute(result.side_effect)
                except asyncio.CancelledError:
                    raise
            if result.advanced_call_state:
                advanced = True
        if not advanced:
            self.session.ivr_no_progress_turns += 1
            if self.session.ivr_no_progress_turns >= IVR_NO_PROGRESS_LIMIT:
                self.session.completion_reason = "ivr_no_progress"
                log.info("ivr_no_progress_watchdog_tripped")
        else:
            self.session.ivr_no_progress_turns = 0

    def _on_consumer_done(self, task: asyncio.Task[None]) -> None:
        """If the consumer dies with an unhandled exception, the pipeline keeps
        awaiting `out_queue` forever. Set a completion reason so the
        pipeline-side observer terminates the call instead of hanging."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is None:
            return
        log.error("call_session_consumer_died", error=str(exc))
        if self.session.completion_reason is None:
            self.session.completion_reason = "consumer_died"
