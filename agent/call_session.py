# pyright: strict
"""Per-turn loop for the two-mode call session.

Lives alongside the Pipecat pipeline (not inside a FrameProcessor) so LLM
latency never blocks the audio loop. `mark_interrupted()` cancels the in-
flight turn task — actual `asyncio.Task.cancel()`, not a flag check — so the
LLM `await` raises `CancelledError` and the handler aborts cleanly.

Two modes, dispatched per turn from `session.mode`:
- `ivr` → `_ivr_turn` runs an LLM-with-tools loop (Llama 4 Scout)
- `rep` → `_rep_turn` runs a structured-output LLM (Claude Haiku 4.5)

Mode flips one-way `ivr → rep` via the `transfer_to_rep` tool call.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Protocol

import structlog
from pydantic import BaseModel

from agent.actuator import Actuator, CallActuator
from agent.logging_config import log
from agent.observability import observe, trace_session
from agent.schemas import (
    CallSession,
    IVRTurnResponse,
    RepTurnOutput,
    SpeakIntent,
    ToolCall,
    ToolResult,
    Turn,
)

QUEUE_MAX = 8

_LOG_PREVIEW_CHARS = 300
"""Cap on free-form text fields in structured logs (transcripts, LLM
replies, tool-result messages). When Langfuse is recording, the full
text is also captured there; when it isn't, the truncated stdout log
is the only record. 300 chars is enough to read the gist of a typical
IVR menu or rep utterance without blowing up log lines."""

_DEFAULT_BENEFITS_LOG_PATH = "benefits.jsonl"
"""Default location for the per-call benefits JSONL. Overridable via the
`BENEFITS_LOG_PATH` env var (read on each call so tests can redirect to a
tmp file). Path is gitignored at the default location."""


async def _append_benefits_record(session: CallSession) -> None:
    """Write one JSONL entry for this completed call: call_sid,
    completion_reason, patient identifiers, extracted benefits, UTC
    timestamp. JSONL (vs. a single JSON array) because append is a single
    write — no read-modify-write race when multiple calls finish close
    together.

    These records are the call deliverable, so failures log at `error`,
    not `warning` — but never raise: best-effort logging must not affect
    call teardown. The actual file write is dispatched to a worker thread
    via `asyncio.to_thread` so a slow disk doesn't block the asyncio loop
    (which would back-pressure every other processor in the pipeline).
    """
    path = Path(os.environ.get("BENEFITS_LOG_PATH", _DEFAULT_BENEFITS_LOG_PATH))
    record = {
        "call_sid": session.call_sid,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "completion_reason": session.completion_reason,
        "patient": session.patient.model_dump(),
        "benefits": session.benefits.model_dump(),
        "turn_count": session.turn_count,
        "mode_at_completion": session.mode,
    }
    try:
        # Serialize FIRST so a non-JSON-serializable field (e.g. a future
        # model gains a `datetime`) surfaces as a log line, not a half-
        # written file with a torn JSONL row that breaks every reader.
        line = json.dumps(record) + "\n"
    except (TypeError, ValueError) as exc:
        log.error(
            "benefits_record_serialize_failed",
            call_sid=session.call_sid,
            error_class=type(exc).__name__,
            error=str(exc),
        )
        return
    try:
        await asyncio.to_thread(_write_jsonl_line, path, line)
    except OSError as exc:
        log.error(
            "benefits_record_write_failed",
            call_sid=session.call_sid,
            path=str(path),
            error_class=type(exc).__name__,
            error=str(exc),
        )


def _write_jsonl_line(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(line)


# After 2 IVR turns where no tool advanced call state, terminate. Distinct
# from `llm_aborted_ivr` (LLM-deliberate) — this is the "the LLM is spinning
# on validator rejections or producing no tool calls at all" case.
IVR_NO_PROGRESS_LIMIT = 2
# After 2 consecutive rep turns where the LLM emits phase="stuck", terminate.
REP_STUCK_LIMIT = 2

REP_LLM_TIMEOUT_S = 8.0
"""Wall-clock budget for a single rep LLM round-trip. Claude Haiku typically
returns in 1-2s; setting this at 8s leaves p99 headroom while still cutting
off anomalous stalls (live testing observed a 16s outlier on one turn,
which surfaced as 17s of silence + a flood of stacked replies when other
transcripts queued up behind it). On timeout the runner speaks a brief
filler so the user knows the agent is still alive, and skips merging
benefits for that turn — the user can volunteer the value again."""

# Context-neutral filler — the timeout fires without us knowing what the rep
# just said, so the line must work in any conversational position: greeting,
# rep asking a question, rep providing a value, rep saying "hold on"
# themselves. Avoid phrasing that implies a specific activity (e.g.,
# "let me check that" implies looking-up, which only fits the "rep just
# gave info" context).
_TIMEOUT_FILLER_REPLY = "Sorry, just one second."


class IVRLLMClient(Protocol):
    """Tool-calling LLM for IVR mode. Production: `GroqToolCallingClient`.
    Tests inject `FakeIVRLLMClient`."""

    async def complete_with_tools(
        self,
        system: str,
        history: list[Turn],
        tools: list[dict[str, Any]],
        temperature: float = 0.1,
    ) -> IVRTurnResponse: ...


class RepLLMClient(Protocol):
    """Structured-output LLM for rep mode. Production: `AnthropicRepClient`.
    Tests inject `FakeAnthropicRepClient`."""

    async def complete_structured[T: BaseModel](
        self,
        system: str,
        history: list[dict[str, Any]],
        schema: type[T],
        max_tokens: int = 1024,
    ) -> T: ...


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
        rep_llm: RepLLMClient,
        tool_dispatcher: ToolDispatcher,
        ivr_system_prompt: str,
        rep_system_prompt: str,
        tools: list[dict[str, Any]],
        *,
        actuator: Actuator | None = None,
        twilio_client: Any = None,
        in_queue_size: int = QUEUE_MAX,
        out_queue_size: int = QUEUE_MAX,
    ) -> None:
        self.session = session
        self.ivr_llm = ivr_llm
        self.rep_llm = rep_llm
        self.dispatch = tool_dispatcher
        self.ivr_system_prompt = ivr_system_prompt
        self.rep_system_prompt = rep_system_prompt
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
        # One-shot: flips True the first time `_record_completion_if_needed`
        # actually fires the JSONL write. Both the in-turn call site (in
        # `_run_turn` after the mode-specific handler returns) and the
        # `stop()`-time call site share this flag, so a terminal
        # `completion_reason` set outside any turn (consumer_died,
        # pipeline_torn_down) still produces exactly one JSONL line.
        self._benefits_recorded: bool = False

    # --- Public API (called from Pipecat side) ------------------------------

    def submit_transcript(self, text: str) -> None:
        """Non-blocking enqueue with drop-oldest on full. Stale transcripts
        are worthless when they're partial fragments — but the upstream
        VAD-driven flush hands us whole-menu aggregates, so an eviction
        now drops a complete IVR menu the LLM never gets to see. Log
        loudly when it happens so a "the agent silently skipped a menu"
        symptom is greppable.
        """
        log.info("transcript_submitted", mode=self.session.mode, text=text[:_LOG_PREVIEW_CHARS])
        try:
            self.in_queue.put_nowait(text)
        except asyncio.QueueFull:
            evicted: str | None = None
            with contextlib.suppress(asyncio.QueueEmpty):
                evicted = self.in_queue.get_nowait()
            log.warning(
                "transcript_dropped_queue_full",
                evicted_preview=evicted[:120] if evicted else None,
                new_preview=text[:120],
            )
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
            except Exception as exc:
                # A task that crashed during shutdown is still a real bug.
                log.warning("task_error_during_stop", task=task.get_name(), error=str(exc))
        # Catch the terminal `completion_reason` cases that don't run inside
        # any turn (`_on_consumer_done` → "consumer_died", state_processor's
        # pump-giving-up → "pipeline_torn_down"), and the rare in-turn case
        # where cancellation landed between completion_reason being set and
        # `_run_turn`'s own write call. The one-shot flag makes this idempotent
        # against the in-turn write that already happened on the happy path.
        await self._record_completion_if_needed()

    # --- Internal loop ------------------------------------------------------

    async def _consume(self) -> None:
        structlog.contextvars.bind_contextvars(call_sid=self.session.call_sid, turn_index=0)
        while True:
            # One transcript = one turn — UNLESS more transcripts queued up
            # behind it during the previous turn's LLM round-trip, in which
            # case `_coalesce_pending` joins them all into a single user turn.
            # A continuous user utterance fragmented by VAD into N flushes
            # would otherwise produce N rephrased agent replies (observed in
            # live test post-barge-in: three "good morning" replies for one
            # user "good morning"). Staleness on barge-in is still handled by
            # `mark_interrupted`'s synchronous queue drain.
            first = await self.in_queue.get()
            transcript = self._coalesce_pending(first)
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
            # Don't exit on `session.done`: the user may still speak after
            # the rep LLM emits phase="complete" (the prompt's rule #9 +
            # "after your close, ack briefly" guidance counts on us routing
            # post-close acks back through the rep LLM). Teardown is owned
            # by the pipeline observer — when transport disconnects, the
            # state processor calls `runner.stop()`, which cancels this
            # consumer. The JSONL write happens inside `_run_turn` the
            # moment `completion_reason` is first set; further turns past
            # the close are no-ops on the one-shot guard.

    def _coalesce_pending(self, first: str) -> str:
        """Join `first` with any other transcripts currently sitting in
        `in_queue`. The queue is bounded at QUEUE_MAX so the drained list is
        bounded by construction; the explicit cap is defensive against a
        future maxsize-changing refactor and surfaces a "we coalesced this
        many" log line that's grep-able under live-test triage.

        Coalescing happens BEFORE turn dispatch and WITHOUT mode awareness
        — transcripts queued during an IVR turn that ends in
        `transfer_to_rep` get processed as part of the first rep turn. That
        is the desired behavior in practice (a fragmented user utterance
        spanning the flip should still arrive as one user turn), and the
        rep LLM is robust to a heterogeneous user message.
        """
        joined: list[str] = [first]
        while len(joined) < QUEUE_MAX and not self.in_queue.empty():
            try:
                joined.append(self.in_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        joined_text = " ".join(joined)
        if len(joined) > 1:
            log.info(
                "transcript_coalesced",
                count=len(joined),
                preview=joined_text[:_LOG_PREVIEW_CHARS],
            )
        return joined_text

    async def _record_completion_if_needed(self) -> None:
        """One-shot: log + write the per-call benefits JSONL the first time
        `completion_reason` is observed set. Two callers — `_run_turn` for
        in-turn completions, `stop()` for terminal-outside-a-turn cases —
        share the `_benefits_recorded` flag so exactly one line per call.

        Flag flips in `finally`, which is the only ordering that's correct
        under cancellation:
        - Flag-before-await: barge-in cancels the `to_thread` await; the
          worker thread completes the file write but the flag was already
          set, so a future call short-circuits — record persisted, flag
          consistent.  But if cancel lands BEFORE the await dispatches
          to_thread (rare, during the `json.dumps` prelude), the flag is
          set with NO write — record lost, no retry. Strictly worse than
          finally.
        - Flag-after-await: same cancel-lands-during-to_thread case sees
          the worker thread complete the write but the flag NEVER set
          (cancel raises out of the await before reaching the assignment).
          `stop()` then retries and double-writes — two JSONL lines for
          one call.
        - Flag-in-finally: the assignment runs whether the await returned
          normally, raised, or was cancelled. The worker thread always
          completes its write (asyncio cancellation doesn't propagate to
          `to_thread` workers), so the file is correct in every case
          except the narrow cancel-before-dispatch window — where we lose
          the record but never double-write. That trade is the right one:
          `_append_benefits_record` is already best-effort (OSErrors
          logged, not raised), so a single missed record on shutdown
          cancel matches its existing contract."""
        if self._benefits_recorded:
            return
        if self.session.completion_reason is None:
            return
        log.info(
            "call_session_complete",
            reason=self.session.completion_reason,
            benefits=self.session.benefits.model_dump(),
        )
        try:
            await _append_benefits_record(self.session)
        finally:
            self._benefits_recorded = True

    @observe(name="call_turn")
    async def _run_turn(self, transcript: str) -> None:
        with trace_session(self.session.call_sid):
            self.session.history.append(Turn(role="user", content=transcript))
            # If the turn is cancelled mid-flight (barge-in), turn_count is NOT
            # incremented and the watchdog counter is NOT touched — barge-in is
            # a re-do, not a "tried and failed" turn.
            if self.session.mode == "rep":
                await self._rep_turn()
            else:
                await self._ivr_turn()
            self.session.turn_count += 1
            await self._record_completion_if_needed()

    @observe(name="ivr_turn")
    async def _ivr_turn(self) -> None:
        response = await self.ivr_llm.complete_with_tools(
            system=self.ivr_system_prompt,
            history=self.session.history,
            tools=self.tools,
            temperature=0.1,
        )
        log.info(
            "ivr_response_received",
            tool_call_count=len(response.tool_calls),
            tool_calls=[{"name": c.name, "args": c.args} for c in response.tool_calls],
            text=response.text[:_LOG_PREVIEW_CHARS] if response.text else "",
        )
        advanced = False
        for call in response.tool_calls:
            # Pair each tool_call with a tool_result entry — even on cancellation
            # so the history stays well-formed (tool-calling APIs reject mismatched
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
            log.info(
                "ivr_tool_dispatched",
                name=call.name,
                args=call.args,
                success=result.success,
                advanced=result.advanced_call_state,
                message=result.message[:_LOG_PREVIEW_CHARS] if result.message else "",
            )
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
        # `wait` is a deliberate non-action (acknowledging filler/hold/greeting)
        # — it must NOT trip the no-progress watchdog the way a stuck-on-
        # validator-rejection LLM would. The hold-budget watchdog inside
        # `_dispatch_wait` is the correct guard for "spent too long waiting".
        # A turn whose only tool calls were `wait` is exempt here.
        only_wait = bool(response.tool_calls) and all(c.name == "wait" for c in response.tool_calls)
        if not advanced and not only_wait:
            self.session.ivr_no_progress_turns += 1
            if self.session.ivr_no_progress_turns >= IVR_NO_PROGRESS_LIMIT:
                self.session.completion_reason = "ivr_no_progress"
                log.info("ivr_no_progress_watchdog_tripped")
        elif advanced:
            self.session.ivr_no_progress_turns = 0

    @observe(name="rep_turn")
    async def _rep_turn(self) -> None:
        # Slice history starting at the mode-flip point so the rep LLM doesn't
        # receive the IVR phase's user transcripts (which would arrive as
        # consecutive user messages — Anthropic 400s on that). When the test
        # harness sets `mode="rep"` directly, `rep_mode_index` is None and we
        # send the full (typically empty or test-shaped) history. `is None`
        # rather than `or 0` because index 0 is a legitimate value (flip at
        # an empty history) that `or` would silently collapse.
        start = 0 if self.session.rep_mode_index is None else self.session.rep_mode_index
        try:
            output = await asyncio.wait_for(
                self.rep_llm.complete_structured(
                    system=self.rep_system_prompt,
                    history=_history_to_anthropic_messages(self.session.history[start:]),
                    schema=RepTurnOutput,
                ),
                timeout=REP_LLM_TIMEOUT_S,
            )
        except TimeoutError:
            # Anomalous stall (Anthropic transient slowness, internal retry
            # backoff, network blip). Speak a brief filler so the user
            # doesn't sit in dead silence, append a placeholder assistant
            # turn so the next call has narrative continuity, and return
            # without touching benefits or stuck_turns. The user can
            # volunteer the same value again on the next turn.
            log.warning(
                "rep_llm_timeout",
                timeout_s=REP_LLM_TIMEOUT_S,
                history_len=len(self.session.history),
            )
            # Best-effort race guard: `mark_interrupted` runs synchronously
            # from the state_processor task and sets `_interrupt_requested`
            # before calling `_current_turn.cancel()`. Cancellation lands at
            # the next await — which is the `actuator.execute(...)` below.
            # Whether `await out_queue.put()` actually yields (and thus lets
            # CancelledError fire) depends on asyncio internals: a non-full
            # queue completes the put without yielding, so cancellation may
            # not fire here. Checking the flag explicitly is defense-in-
            # depth. We must also CONSUME the flag — the consumer's
            # CancelledError handler resets it on the cancel path, but we
            # may have skipped that path entirely by returning before any
            # await. Without resetting, a future turn's guard would mis-
            # fire on a stale flag.
            if self._interrupt_requested:
                self._interrupt_requested = False
                return
            await self.actuator.execute(SpeakIntent(text=_TIMEOUT_FILLER_REPLY))
            self.session.history.append(Turn(role="assistant", content=_TIMEOUT_FILLER_REPLY))
            return
        log.info(
            "rep_response_received",
            reply=output.reply[:_LOG_PREVIEW_CHARS] if output.reply else "",
            extracted=output.extracted.model_dump(exclude_none=True),
            phase=output.phase,
        )
        # Non-None merge into session.benefits — the LLM emits only the fields
        # learned from THIS rep utterance; previously-extracted fields stay.
        # Conflicts surface naturally on the next turn as the LLM's own
        # follow-up question.
        for field, value in output.extracted.model_dump(exclude_none=True).items():
            setattr(self.session.benefits, field, value)
        # Empty `reply` = stay silent (e.g., during a hold announcement).
        if output.reply:
            await self.actuator.execute(SpeakIntent(text=output.reply))
        self.session.history.append(
            Turn(role="assistant", content=output.reply, extracted=output.extracted)
        )
        if output.phase == "complete":
            self.session.completion_reason = "rep_complete"
            log.info(
                "rep_complete",
                reasoning=output.reasoning,
                benefits=self.session.benefits.model_dump(),
            )
        elif output.phase == "stuck":
            self.session.stuck_turns += 1
            if self.session.stuck_turns >= REP_STUCK_LIMIT:
                self.session.completion_reason = "rep_stuck"
                log.info("rep_stuck_watchdog_tripped")
        else:
            self.session.stuck_turns = 0

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


def _history_to_anthropic_messages(history: list[Turn]) -> list[dict[str, Any]]:
    """Project session history into the Anthropic messages shape — only
    `user` and `assistant` turns, no tool calls (the rep LLM doesn't run
    tools). Empty assistant content is skipped so we don't emit hold-music
    silence as visible turns."""
    messages: list[dict[str, Any]] = []
    for turn in history:
        if turn.role == "user" and turn.content:
            messages.append({"role": "user", "content": turn.content})
        elif turn.role == "assistant" and turn.content:
            messages.append({"role": "assistant", "content": turn.content})
    return messages
