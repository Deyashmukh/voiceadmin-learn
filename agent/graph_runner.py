"""Async owner of the LangGraph state machine for a single call.

Lives alongside the Pipecat pipeline (not inside a FrameProcessor) so LLM
latency never blocks the audio loop. `mark_interrupted()` uses real
`asyncio.Task.cancel()` — not a flag — so an in-flight LLM call actually
aborts and the handler's `await` raises `CancelledError`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import structlog
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph

from agent.graph import initial_state
from agent.logging_config import log
from agent.schemas import FALLBACK_RESPONSE, CallState, PatientInfo

QUEUE_MAX = 8
RECURSION_LIMIT = 10


@dataclass
class CallContext:
    call_sid: str
    patient: PatientInfo


class GraphRunner:
    """One per call. Consume transcripts, drive the graph, emit responses."""

    def __init__(
        self,
        graph: CompiledStateGraph,
        call_ctx: CallContext,
        *,
        queue_max: int = QUEUE_MAX,
        recursion_limit: int = RECURSION_LIMIT,
    ) -> None:
        self.graph = graph
        self.call_ctx = call_ctx
        self.recursion_limit = recursion_limit
        self.in_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=queue_max)
        self.out_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=queue_max)
        self.state: CallState = initial_state(call_ctx.patient)
        self._consumer: asyncio.Task[None] | None = None
        self._current_turn: asyncio.Task[CallState] | None = None
        self._interrupt_requested: bool = False

    async def start(self) -> None:
        self._consumer = asyncio.create_task(self._consume(), name="graph-runner-consume")

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
                # A task that crashed during shutdown is still a real bug — don't silently swallow.
                log.warning("task_error_during_stop", task=task.get_name(), error=str(exc))

    def submit_transcript(self, text: str) -> None:
        """Non-blocking enqueue with drop-oldest on full. Callable from Pipecat."""
        try:
            self.in_queue.put_nowait(text)
        except asyncio.QueueFull:
            try:
                _ = self.in_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self.in_queue.put_nowait(text)

    def mark_interrupted(self) -> None:
        """Cancel the in-flight turn. Must be called from the event-loop thread."""
        # Only set the flag when there's a real turn to cancel. Setting it
        # between turns would leave it stuck True and conflate a future
        # stop() cancellation with an interrupt.
        if self._current_turn and not self._current_turn.done():
            self._interrupt_requested = True
            self._current_turn.cancel()

    async def _consume(self) -> None:
        structlog.contextvars.bind_contextvars(call_sid=self.call_ctx.call_sid, turn_index=0)
        while self.state.get("current_node") != "done":
            transcript = await self.in_queue.get()
            # Drain stale transcripts that piled up during the last turn.
            while not self.in_queue.empty():
                try:
                    transcript = self.in_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

            structlog.contextvars.bind_contextvars(turn_index=self.state.get("turn_count", 0))
            self._current_turn = asyncio.create_task(self._run_turn(transcript))
            try:
                new_state = await self._current_turn
            except asyncio.CancelledError:
                # Distinguish interrupt (mark_interrupted) from outer cancel (stop).
                if self._interrupt_requested:
                    self._interrupt_requested = False
                    log.info("turn_cancelled", reason="interrupt")
                    continue
                raise
            # State MUST update before we publish the response, otherwise a caller
            # reading out_queue and then state sees stale state.
            self.state = new_state
            response = new_state.get("response_text")
            if response:
                await self.out_queue.put(response)

    def _hard_fallback(self, reason: str) -> CallState:
        return {
            **self.state,
            "current_node": "done",
            "fallback_reason": reason,
            "response_text": FALLBACK_RESPONSE,
            "turn_count": self.state.get("turn_count", 0) + 1,
        }

    async def _run_turn(self, transcript: str) -> CallState:
        # create_task inherits the consumer's contextvars, but re-bind defensively
        # so a future refactor that spawns _run_turn from a different context
        # doesn't silently lose the call_sid tag on node-level logs.
        structlog.contextvars.bind_contextvars(
            call_sid=self.call_ctx.call_sid,
            turn_index=self.state.get("turn_count", 0),
        )
        # Reset response_text so a turn whose handler emits no response doesn't
        # re-publish last turn's text. LangGraph's TypedDict reducer is
        # replace-per-key, not merge-per-key, so stale values stick otherwise.
        turn_state: CallState = {
            **self.state,
            "transcript": transcript,
            "response_text": None,
        }
        try:
            result = await self.graph.ainvoke(
                turn_state,
                config={"recursion_limit": self.recursion_limit},
            )
        except GraphRecursionError:
            log.warning("graph_recursion_limit")
            return self._hard_fallback("recursion_limit")
        except Exception as exc:  # noqa: BLE001
            # Scope this to the graph invocation only. If _hard_fallback
            # itself raises (a real bug), we want the exception to propagate
            # to _consume rather than silently emit another fallback.
            log.exception("node_error", error=str(exc))
            return self._hard_fallback("node_exception")

        new_state: CallState = result  # type: ignore[assignment]
        new_state["turn_count"] = self.state.get("turn_count", 0) + 1
        return new_state
