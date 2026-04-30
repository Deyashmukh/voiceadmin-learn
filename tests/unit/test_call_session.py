# pyright: strict
"""Unit tests for CallSessionRunner — IVR turn loop (D1), rep turn handler
+ mode-aware dispatch (D2/D3). Zero network — both LLMs are faked; the
default CallActuator (with no twilio_client) routes SpeakIntent into
out_queue natively, so tests can read spoken text from there directly."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

import pytest
import structlog.contextvars
import structlog.testing

from agent import tools
from agent.actuator import Actuator, CallActuator
from agent.call_session import CallSessionRunner, ToolDispatcher
from agent.schemas import (
    Benefits,
    CallSession,
    DTMFIntent,
    IVRTurnResponse,
    RepTurnOutput,
    SideEffectIntent,
    SpeakIntent,
    ToolCall,
    ToolResult,
)

from .conftest import (
    FakeActuator,
    FakeAnthropicRepClient,
    FakeIVRLLMClient,
    MakeSession,
    wait_until,
)


def _build_runner(
    session: CallSession,
    *,
    ivr_llm: FakeIVRLLMClient | None = None,
    rep_llm: FakeAnthropicRepClient | None = None,
    actuator: Actuator | None = None,
    tool_dispatcher: ToolDispatcher | None = None,
) -> tuple[CallSessionRunner, FakeIVRLLMClient, FakeAnthropicRepClient]:
    """Construct a CallSessionRunner with both LLM fakes pre-installed.
    Default actuator is the runner's auto-built `CallActuator` (no
    twilio_client), which routes SpeakIntent → out_queue natively. Tests
    pass an explicit `FakeActuator` when they want to assert on intents
    directly, or an alternative `tool_dispatcher` to inject failures."""
    ivr = ivr_llm or FakeIVRLLMClient()
    rep = rep_llm or FakeAnthropicRepClient()
    runner = CallSessionRunner(
        session=session,
        ivr_llm=ivr,
        rep_llm=rep,
        tool_dispatcher=tool_dispatcher or tools.dispatch,
        ivr_system_prompt="ivr-system-prompt",
        rep_system_prompt="rep-persona-prompt",
        tools=[{"name": "send_dtmf"}],
        actuator=actuator,
    )
    return runner, ivr, rep


# --- submit_transcript: drop-oldest -----------------------------------------


async def test_submit_transcript_drops_oldest_when_full(make_session: MakeSession):
    runner, _, _ = _build_runner(make_session())
    with structlog.testing.capture_logs() as captured:
        for i in range(15):
            runner.submit_transcript(f"item-{i}")
    assert runner.in_queue.qsize() == 8
    drained: list[str] = []
    while not runner.in_queue.empty():
        drained.append(runner.in_queue.get_nowait())
    assert drained[-1] == "item-14"
    assert "item-0" not in drained
    # Each eviction must surface as a `transcript_dropped_queue_full` warning
    # (the "agent silently skipped a menu" greppability contract). 15 puts
    # against an 8-slot queue → 7 evictions.
    eviction_events = [
        e
        for e in captured
        if e.get("event") == "transcript_dropped_queue_full" and e.get("log_level") == "warning"
    ]
    assert len(eviction_events) == 7, (
        f"expected 7 warning-level transcript_dropped_queue_full events, got {len(eviction_events)}"
    )


# --- IVR turn happy path ----------------------------------------------------


async def test_call_completion_appends_jsonl_record(make_session: MakeSession):
    """One JSONL line per completed call — regardless of mode. Captures the
    call_sid, completion_reason, patient identifiers, and final benefits.
    `BENEFITS_LOG_PATH` is redirected to a tmp file by the autouse fixture
    in conftest, so this assertion reads back exactly what the runner just
    wrote."""
    runner, ivr_llm, _rep = _build_runner(make_session(), actuator=FakeActuator())
    ivr_llm.responses = [
        IVRTurnResponse(
            tool_calls=[ToolCall(name="complete_call", args={"reason": "benefits_extracted"})]
        )
    ]
    try:
        await runner.start()
        runner.submit_transcript("Thank you, goodbye.")
        # Wait for the consumer to finish (not just for `session.done` to
        # flip) — JSONL write happens AFTER `session.done` is set, via
        # `asyncio.to_thread`, and the consumer only exits once the write
        # returns. Cancelling mid-write would leave the file unwritten.
        await wait_until(
            lambda: runner._consumer is not None and runner._consumer.done(),  # pyright: ignore[reportPrivateUsage]
            description="consumer task done (post-write)",
        )
    finally:
        await runner.stop()
    log_path = Path(os.environ["BENEFITS_LOG_PATH"])
    lines = log_path.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["call_sid"] == runner.session.call_sid
    assert record["completion_reason"] == "benefits_extracted"
    assert record["patient"]["member_id"] == runner.session.patient.member_id
    assert "benefits" in record
    assert "completed_at" in record


async def test_jsonl_append_does_not_overwrite_prior_calls(make_session: MakeSession):
    """JSONL was chosen over a single-array file precisely so back-to-back
    completions don't race read-modify-write. Two calls must produce two
    distinct lines, both readable. Catches a regression where someone
    flips `"a"` to `"w"` on the file open and silently drops history."""

    async def _run_one_completed_call(sid: str) -> None:
        session = make_session()
        session.call_sid = sid
        runner, ivr_llm, _rep = _build_runner(session, actuator=FakeActuator())
        ivr_llm.responses = [
            IVRTurnResponse(
                tool_calls=[ToolCall(name="complete_call", args={"reason": "benefits_extracted"})]
            )
        ]
        try:
            await runner.start()
            runner.submit_transcript("Thank you, goodbye.")
            # Wait for the consumer to actually finish — see the equivalent
            # comment in test_call_completion_appends_jsonl_record.
            await wait_until(
                lambda: runner._consumer is not None and runner._consumer.done(),  # pyright: ignore[reportPrivateUsage]
                description="consumer task done (post-write)",
            )
        finally:
            await runner.stop()

    await _run_one_completed_call("CA-call-A")
    await _run_one_completed_call("CA-call-B")
    log_path = Path(os.environ["BENEFITS_LOG_PATH"])
    lines = log_path.read_text().splitlines()
    assert len(lines) == 2, f"expected 2 entries, got {len(lines)}"
    sids = {json.loads(line)["call_sid"] for line in lines}
    assert sids == {"CA-call-A", "CA-call-B"}, f"expected both sids, got {sids}"


async def test_jsonl_write_failure_does_not_abort_call(
    make_session: MakeSession, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Best-effort logging contract: a failed write must not propagate into
    call teardown, AND it must surface in the logs. Without the log assertion
    a refactor that swapped `except OSError:` to `except Exception: pass`
    would silently drop benefits records and pass this test.

    Uses `structlog.testing.capture_logs` to assert the
    `benefits_record_write_failed` event fires at error level — the contract
    is "best-effort, but observable", not "best-effort and silent".
    """
    monkeypatch.setenv("BENEFITS_LOG_PATH", str(tmp_path / "missing-dir" / "benefits.jsonl"))
    runner, ivr_llm, _rep = _build_runner(make_session(), actuator=FakeActuator())
    ivr_llm.responses = [
        IVRTurnResponse(
            tool_calls=[ToolCall(name="complete_call", args={"reason": "benefits_extracted"})]
        )
    ]
    with structlog.testing.capture_logs() as captured:
        try:
            await runner.start()
            runner.submit_transcript("Thank you, goodbye.")
            await wait_until(
                lambda: runner._consumer is not None and runner._consumer.done(),  # pyright: ignore[reportPrivateUsage]
                description="consumer task done (post-write)",
            )
        finally:
            await runner.stop()
    assert runner.session.completion_reason == "benefits_extracted"
    write_failed_events = [
        e
        for e in captured
        if e.get("event") == "benefits_record_write_failed" and e.get("log_level") == "error"
    ]
    assert len(write_failed_events) == 1, (
        f"expected exactly one error-level benefits_record_write_failed event, "
        f"got {len(write_failed_events)}; all events: {[e.get('event') for e in captured]}"
    )


async def test_jsonl_serialize_failure_logs_and_skips_write(
    make_session: MakeSession, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """If `json.dumps(record)` raises (e.g. a future model gains a
    non-serializable field like `datetime`), the serialize-first ordering
    in `_append_benefits_record` must surface the failure as
    `benefits_record_serialize_failed` at error level AND skip the file
    write entirely — no torn JSONL row, no half-written line that breaks
    every downstream reader.

    Locks the contract for the new serialize-then-write split. A refactor
    that moves serialization back inline with the write would silently
    re-introduce torn writes and fail this test.
    """
    log_path = tmp_path / "benefits.jsonl"
    monkeypatch.setenv("BENEFITS_LOG_PATH", str(log_path))

    def _raising_dumps(*_args: object, **_kwargs: object) -> NoReturn:
        raise TypeError("bad field")

    monkeypatch.setattr("agent.call_session.json.dumps", _raising_dumps)
    runner, ivr_llm, _rep = _build_runner(make_session(), actuator=FakeActuator())
    ivr_llm.responses = [
        IVRTurnResponse(
            tool_calls=[ToolCall(name="complete_call", args={"reason": "benefits_extracted"})]
        )
    ]
    with structlog.testing.capture_logs() as captured:
        try:
            await runner.start()
            runner.submit_transcript("Thank you, goodbye.")
            await wait_until(
                lambda: runner._consumer is not None and runner._consumer.done(),  # pyright: ignore[reportPrivateUsage]
                description="consumer task done (post-write)",
            )
        finally:
            await runner.stop()
    assert runner.session.completion_reason == "benefits_extracted"
    serialize_failed_events = [
        e
        for e in captured
        if e.get("event") == "benefits_record_serialize_failed" and e.get("log_level") == "error"
    ]
    assert len(serialize_failed_events) == 1, (
        f"expected exactly one error-level benefits_record_serialize_failed event, "
        f"got {len(serialize_failed_events)}"
    )
    # The file must NOT exist — serialize-first ordering means we never
    # opened the file at all on a serialize failure.
    assert not log_path.exists(), (
        "log file was created despite serialize failure; serialize-then-write ordering is broken"
    )


async def test_ivr_turn_dispatches_send_dtmf_intent(make_session: MakeSession):
    """Turn 1 sends DTMF, turn 2 completes the call. Use a FakeActuator so
    we can assert on the intents directly."""
    actuator = FakeActuator()
    runner, ivr_llm, _rep = _build_runner(make_session(), actuator=actuator)
    ivr_llm.responses = [
        IVRTurnResponse(tool_calls=[ToolCall(name="send_dtmf", args={"digits": "1"})]),
        IVRTurnResponse(
            tool_calls=[ToolCall(name="complete_call", args={"reason": "benefits_extracted"})]
        ),
    ]
    try:
        await runner.start()
        runner.submit_transcript("Press 1 for eligibility")
        runner.submit_transcript("Thank you, that completes the menu")
        await wait_until(lambda: runner.session.done)
        assert any(isinstance(i, DTMFIntent) and i.digits == "1" for i in actuator.executed)
        assert runner.session.completion_reason == "benefits_extracted"
        assert runner.session.done
    finally:
        await runner.stop()


async def test_ivr_turn_records_history(make_session: MakeSession):
    runner, ivr_llm, _rep = _build_runner(make_session(), actuator=FakeActuator())
    ivr_llm.responses = [
        IVRTurnResponse(tool_calls=[ToolCall(name="speak", args={"text": "one moment"})]),
        IVRTurnResponse(
            tool_calls=[ToolCall(name="complete_call", args={"reason": "user_hangup"})]
        ),
    ]
    try:
        await runner.start()
        runner.submit_transcript("hello")
        runner.submit_transcript("anything")
        await wait_until(lambda: runner.session.done)
        roles = [t.role for t in runner.session.history]
        assert roles == [
            "user",
            "tool_call",
            "tool_result",
            "user",
            "tool_call",
            "tool_result",
        ]
    finally:
        await runner.stop()


async def test_speak_intent_is_pushed_to_out_queue(make_session: MakeSession):
    """The default CallActuator routes SpeakIntent.text into out_queue —
    that's the production wire to the configured TTS service via
    state_processor's pump."""
    runner, ivr_llm, _rep = _build_runner(make_session())
    ivr_llm.responses = [
        IVRTurnResponse(tool_calls=[ToolCall(name="speak", args={"text": "hello there"})]),
    ]
    try:
        await runner.start()
        runner.submit_transcript("kickoff")
        spoken = await asyncio.wait_for(runner.out_queue.get(), timeout=1.0)
        assert spoken == "hello there"
    finally:
        await runner.stop()


# --- Validator rejection: re-pick path --------------------------------------


async def test_validator_rejection_does_not_advance_call_state(make_session: MakeSession):
    """Bad DTMF digit (not in recent menu options): dispatcher returns
    `advanced_call_state=False`, so the turn doesn't count as progress and
    the watchdog ticks. Two such turns trip the watchdog."""
    actuator = FakeActuator()
    runner, ivr_llm, _rep = _build_runner(
        make_session(recent_menu_options=["1", "2", "3"]), actuator=actuator
    )
    ivr_llm.responses = [
        IVRTurnResponse(tool_calls=[ToolCall(name="send_dtmf", args={"digits": "9"})]),
        IVRTurnResponse(tool_calls=[ToolCall(name="send_dtmf", args={"digits": "9"})]),
    ]
    try:
        await runner.start()
        runner.submit_transcript("first")
        runner.submit_transcript("second")
        await wait_until(lambda: runner.session.done)
        assert runner.session.completion_reason == "ivr_no_progress"
        # No DTMF intents reached the actuator (validator rejected both).
        assert not any(isinstance(i, DTMFIntent) for i in actuator.executed)
    finally:
        await runner.stop()


# --- Watchdog: zero-tool-call case (Groq timeout-style) --------------------


async def test_watchdog_trips_on_zero_tool_call_turns(make_session: MakeSession):
    """If the LLM produces no tool calls at all (timeout / hallucinated text
    response with no tools), the watchdog must still tick. Two such turns
    in a row → terminate."""
    runner, ivr_llm, _rep = _build_runner(make_session(), actuator=FakeActuator())
    ivr_llm.responses = [IVRTurnResponse(), IVRTurnResponse()]
    try:
        await runner.start()
        runner.submit_transcript("first")
        runner.submit_transcript("second")
        await wait_until(lambda: runner.session.done)
        assert runner.session.completion_reason == "ivr_no_progress"
    finally:
        await runner.stop()


async def test_watchdog_resets_on_an_advancing_turn(make_session: MakeSession):
    """One bad turn followed by a good turn must reset the counter."""
    runner, ivr_llm, _rep = _build_runner(make_session(), actuator=FakeActuator())
    ivr_llm.responses = [
        IVRTurnResponse(),  # 1 no-progress
        IVRTurnResponse(  # advances → counter resets
            tool_calls=[ToolCall(name="speak", args={"text": "yes"})]
        ),
        IVRTurnResponse(),  # 1 no-progress (counter just reset)
        IVRTurnResponse(),  # 2 no-progress → terminate
    ]
    try:
        await runner.start()
        for i in range(4):
            runner.submit_transcript(f"transcript-{i}")
        await wait_until(lambda: runner.session.done)
        assert runner.session.completion_reason == "ivr_no_progress"
    finally:
        await runner.stop()


async def test_mixed_wait_and_advancing_turn_resets_no_progress(make_session: MakeSession):
    """A turn with BOTH a `wait` and an advancing tool (e.g., a `send_dtmf`)
    must reset the no-progress counter via the `elif advanced` branch — the
    `only_wait` guard is False because not all tool calls are `wait`, but
    `advanced=True` so the counter resets. Locks the AND-of-two-conditions
    semantics so a refactor that flips `all()` to `any()` would fail this
    test.
    """
    runner, ivr_llm, _rep = _build_runner(make_session(), actuator=FakeActuator())
    ivr_llm.responses = [
        IVRTurnResponse(),  # 1 no-progress (counter=1)
        IVRTurnResponse(  # mixed: wait + send_dtmf — must reset counter
            tool_calls=[
                ToolCall(name="wait", args={}),
                ToolCall(name="send_dtmf", args={"digits": "1"}),
            ]
        ),
        IVRTurnResponse(  # 1 no-progress again (counter just reset, so 1 not 2)
            tool_calls=[]
        ),
        IVRTurnResponse(  # advancing
            tool_calls=[ToolCall(name="complete_call", args={"reason": "user_hangup"})]
        ),
    ]
    try:
        await runner.start()
        for i in range(4):
            runner.submit_transcript(f"transcript-{i}")
        await wait_until(lambda: runner.session.done)
        # Reached complete_call cleanly — no_progress never hit 2.
        assert runner.session.completion_reason == "user_hangup"
    finally:
        await runner.stop()


async def test_wait_only_turn_does_not_trip_no_progress_watchdog(
    make_session: MakeSession,
):
    """`wait` is a deliberate non-action (acknowledging IVR filler), not a
    stuck spin. A turn whose only tool was `wait` MUST be exempt from the
    no-progress watchdog — otherwise the IVR opening sequence ('Welcome
    to Aetna' + 'Please listen carefully') would terminate the call after
    just two filler turns. The hold-budget watchdog inside `_dispatch_wait`
    is the correct guard for "spent too long waiting".
    """
    runner, ivr_llm, _rep = _build_runner(make_session(), actuator=FakeActuator())
    # Three consecutive wait-only turns. Pre-fix, this would trip the
    # 2-turn no-progress watchdog. Post-fix, the call stays alive and we
    # finally complete via an explicit complete_call.
    ivr_llm.responses = [
        IVRTurnResponse(tool_calls=[ToolCall(name="wait", args={})]),
        IVRTurnResponse(tool_calls=[ToolCall(name="wait", args={})]),
        IVRTurnResponse(tool_calls=[ToolCall(name="wait", args={})]),
        IVRTurnResponse(
            tool_calls=[ToolCall(name="complete_call", args={"reason": "user_hangup"})]
        ),
    ]
    try:
        await runner.start()
        for i, text in enumerate(
            ["Welcome to Aetna.", "Please listen carefully.", "Calls may be recorded.", "goodbye"]
        ):
            runner.submit_transcript(text)
            if i < 3:
                # Give the consumer a chance to process each wait turn so we
                # observe the watchdog NOT tripping mid-sequence rather than
                # just at the end.
                await wait_until(
                    lambda i=i: runner.session.turn_count > i,
                    timeout=2.0,
                    description=f"wait-only turn {i} processed",
                )
        await wait_until(lambda: runner.session.done)
        # Exit reason must be the explicit close, NOT no-progress.
        assert runner.session.completion_reason == "user_hangup", (
            f"expected user_hangup, got {runner.session.completion_reason} — "
            "wait-only turns are tripping the no-progress watchdog"
        )
        assert runner.session.ivr_no_progress_turns == 0
    finally:
        await runner.stop()


# --- Interrupt: queue drains + task.cancel --------------------------------


async def test_mark_interrupted_drains_out_queue(make_session: MakeSession):
    """A turn that finished moments before the user started speaking would
    otherwise still get spoken after the barge-in. Drain first."""
    runner, _, _ = _build_runner(make_session())
    await runner.out_queue.put("stale-1")
    await runner.out_queue.put("stale-2")
    runner.mark_interrupted()
    assert runner.out_queue.empty()


async def test_interrupt_drains_in_queue(make_session: MakeSession):
    """Barge-in should not let stale queued transcripts feed the next turn."""
    runner, _, _ = _build_runner(make_session())
    runner.submit_transcript("stale-A")
    runner.submit_transcript("stale-B")
    assert runner.in_queue.qsize() == 2
    runner.mark_interrupted()
    assert runner.in_queue.empty()


async def test_mark_interrupted_cancels_in_flight_turn(make_session: MakeSession):
    """A slow LLM call (simulated via slow_mode_seconds) is cancelled cleanly
    by mark_interrupted — the turn task receives CancelledError, the consumer
    catches it, and the loop continues."""
    runner, ivr_llm, _rep = _build_runner(make_session(), actuator=FakeActuator())
    ivr_llm.slow_mode_seconds = 5.0
    ivr_llm.responses = [
        IVRTurnResponse(tool_calls=[ToolCall(name="speak", args={"text": "won't reach"})])
    ]
    try:
        await runner.start()
        runner.submit_transcript("kickoff")
        await wait_until(
            lambda: runner._current_turn is not None and not runner._current_turn.done()  # pyright: ignore[reportPrivateUsage]
        )
        runner.mark_interrupted()
        await wait_until(lambda: runner._current_turn is not None and runner._current_turn.done())  # pyright: ignore[reportPrivateUsage]
        # Turn was cancelled → no actuator call happened, no completion set.
        assert runner.session.completion_reason is None
    finally:
        await runner.stop()


async def test_mark_interrupted_does_not_complete_call(make_session: MakeSession):
    """Interrupt is a turn-level cancel, not a session-level abort. The runner
    keeps consuming after the interrupt clears."""
    runner, ivr_llm, _rep = _build_runner(make_session(), actuator=FakeActuator())
    ivr_llm.slow_mode_seconds = 5.0
    ivr_llm.responses = [
        IVRTurnResponse(tool_calls=[ToolCall(name="speak", args={"text": "won't reach"})]),
        IVRTurnResponse(
            tool_calls=[ToolCall(name="complete_call", args={"reason": "user_hangup"})]
        ),
    ]
    try:
        await runner.start()
        runner.submit_transcript("first")
        await wait_until(
            lambda: runner._current_turn is not None and not runner._current_turn.done()  # pyright: ignore[reportPrivateUsage]
        )
        runner.mark_interrupted()
        ivr_llm.slow_mode_seconds = 0.0
        runner.submit_transcript("second")
        await wait_until(lambda: runner.session.done)
        assert runner.session.completion_reason == "user_hangup"
    finally:
        await runner.stop()


# --- Cancellation mid tool-loop preserves history pairing -----------------


async def test_cancellation_mid_tool_dispatch_pairs_history(make_session: MakeSession):
    """If a turn is cancelled while awaiting tool dispatch, the history must
    still have a `tool_result` paired to the just-appended `tool_call` —
    otherwise Groq's tool API rejects the next turn."""

    async def _slow_dispatcher(call: ToolCall, session: CallSession) -> ToolResult:
        await asyncio.sleep(5.0)  # cancelled before dispatch finishes
        return await tools.dispatch(call, session)

    ivr_llm = FakeIVRLLMClient(
        responses=[IVRTurnResponse(tool_calls=[ToolCall(name="speak", args={"text": "x"})])]
    )
    runner, _, _ = _build_runner(
        make_session(),
        ivr_llm=ivr_llm,
        actuator=FakeActuator(),
        tool_dispatcher=_slow_dispatcher,
    )
    session = runner.session
    try:
        await runner.start()
        runner.submit_transcript("kickoff")
        await wait_until(
            lambda: runner._current_turn is not None and not runner._current_turn.done()  # pyright: ignore[reportPrivateUsage]
        )
        runner.mark_interrupted()
        await wait_until(lambda: runner._current_turn is not None and runner._current_turn.done())  # pyright: ignore[reportPrivateUsage]
        # History must end with a tool_result paired to the tool_call.
        roles = [t.role for t in session.history]
        # user → tool_call → tool_result(cancelled)
        assert roles == ["user", "tool_call", "tool_result"]
        assert "cancelled" in (session.history[-1].content or "")
    finally:
        await runner.stop()


# --- stop() shutdown -------------------------------------------------------


async def test_stop_cancels_in_flight_turn_quickly(make_session: MakeSession):
    """stop() must complete promptly even when a turn is mid-LLM-call."""
    runner, ivr_llm, _rep = _build_runner(make_session(), actuator=FakeActuator())
    ivr_llm.slow_mode_seconds = 5.0
    ivr_llm.responses = [
        IVRTurnResponse(tool_calls=[ToolCall(name="speak", args={"text": "won't reach"})])
    ]
    await runner.start()
    runner.submit_transcript("kickoff")
    await asyncio.sleep(0.05)
    await asyncio.wait_for(runner.stop(), timeout=1.0)


# --- Consumer exits cleanly when the call completes ------------------------


async def test_consumer_exits_after_complete_call_without_extra_transcript(
    make_session: MakeSession,
):
    """When the LLM emits `complete_call`, the consumer must exit the loop
    immediately rather than blocking on `in_queue.get()` waiting for a
    transcript that will never arrive."""
    runner, ivr_llm, _rep = _build_runner(make_session(), actuator=FakeActuator())
    ivr_llm.responses = [
        IVRTurnResponse(
            tool_calls=[ToolCall(name="complete_call", args={"reason": "benefits_extracted"})]
        ),
    ]
    await runner.start()
    runner.submit_transcript("Done with menu")
    # Wait for both: session.done AND consumer task itself completes.
    await wait_until(lambda: runner.session.done)
    await wait_until(
        lambda: runner._consumer is not None and runner._consumer.done(),  # pyright: ignore[reportPrivateUsage]
        description="consumer task done (post-write)",
    )
    assert runner.session.completion_reason == "benefits_extracted"
    # Consumer is done — stop() should be a no-op for the consumer task.
    await runner.stop()


# --- Consumer-died safety net ---------------------------------------------


async def test_consumer_death_sets_completion_reason(make_session: MakeSession):
    """If the consumer task crashes unhandled, the pipeline-side observer
    needs a signal to end the call instead of hanging on out_queue forever.
    `_on_consumer_done` sets `completion_reason = consumer_died`."""

    async def _exploding_dispatcher(call: ToolCall, session: CallSession) -> NoReturn:
        raise RuntimeError("boom")

    ivr_llm = FakeIVRLLMClient(
        responses=[IVRTurnResponse(tool_calls=[ToolCall(name="speak", args={"text": "x"})])]
    )
    runner, _, _ = _build_runner(
        make_session(),
        ivr_llm=ivr_llm,
        actuator=FakeActuator(),
        tool_dispatcher=_exploding_dispatcher,
    )
    try:
        await runner.start()
        runner.submit_transcript("kickoff")
        await wait_until(
            lambda: runner._consumer is not None and runner._consumer.done(),  # pyright: ignore[reportPrivateUsage]
            description="consumer task done (post-write)",
        )
        assert runner.session.completion_reason == "consumer_died"
    finally:
        await runner.stop()


# --- contextvar binding ----------------------------------------------------


async def test_call_sid_bound_in_contextvars_during_turn(make_session: MakeSession):
    """The runner binds `call_sid` and `turn_index` as structlog contextvars
    before kicking off each turn — so any log emission from inside the turn
    (handlers, actuator, dispatcher) carries the call's identity."""
    captured: list[dict[str, object]] = []

    @dataclass
    class _ContextvarCapturingActuator:
        async def execute(self, intent: SideEffectIntent) -> None:
            captured.append(dict(structlog.contextvars.get_contextvars()))

    runner, ivr_llm, _rep = _build_runner(make_session(), actuator=_ContextvarCapturingActuator())
    ivr_llm.responses = [
        IVRTurnResponse(tool_calls=[ToolCall(name="speak", args={"text": "x"})]),
        IVRTurnResponse(
            tool_calls=[ToolCall(name="complete_call", args={"reason": "user_hangup"})]
        ),
    ]
    try:
        await runner.start()
        runner.submit_transcript("first")
        runner.submit_transcript("second")
        await wait_until(lambda: runner.session.done)
        assert len(captured) == 2
        assert captured[0]["call_sid"] == runner.session.call_sid
        assert captured[1]["call_sid"] == runner.session.call_sid
        assert captured[0]["turn_index"] == 0
        assert captured[1]["turn_index"] == 1
    finally:
        await runner.stop()


# --- Each queued transcript drives its own turn ----------------------------


async def test_each_queued_transcript_drives_its_own_turn(make_session: MakeSession):
    """Without barge-in, multiple queued transcripts must each fire a
    distinct turn — distinct user utterances are not collapsed."""
    runner, ivr_llm, _rep = _build_runner(make_session(), actuator=FakeActuator())
    ivr_llm.slow_mode_seconds = 0.02
    ivr_llm.responses = [
        IVRTurnResponse(tool_calls=[ToolCall(name="speak", args={"text": "one"})]),
        IVRTurnResponse(tool_calls=[ToolCall(name="speak", args={"text": "two"})]),
        IVRTurnResponse(
            tool_calls=[ToolCall(name="complete_call", args={"reason": "user_hangup"})]
        ),
    ]
    try:
        await runner.start()
        runner.submit_transcript("first")
        runner.submit_transcript("second")
        runner.submit_transcript("third")
        await wait_until(lambda: runner.session.done)
        assert len(ivr_llm.calls) == 3
        user_contents = [t.content for t in runner.session.history if t.role == "user"]
        assert user_contents == ["first", "second", "third"]
    finally:
        await runner.stop()


# --- DTMFIntent currently routes through speech (TEMP — see actuator.py) ---


async def test_call_actuator_dtmf_speaks_digits_via_out_queue(make_session: MakeSession):
    """TEMP: until DTMF is rendered as audio tones over the Media Stream, the
    actuator stands in by enqueueing a spoken-digit string. This test pins that
    contract; when real DTMF lands it should be replaced with one asserting an
    audio frame goes out."""
    session = make_session()
    out_queue: asyncio.Queue[str] = asyncio.Queue()
    actuator = CallActuator(session=session, out_queue=out_queue, twilio_client=None)
    await actuator.execute(DTMFIntent(digits="123"))
    assert await out_queue.get() == "Pressing 123."


async def test_call_actuator_hangup_is_noop(make_session: MakeSession):
    """`HangupIntent` is a no-op at the actuator boundary — termination flows
    through `session.completion_reason`, not through actuator I/O. Lock the
    no-op so a future change that adds Twilio-side hangup doesn't accidentally
    fire on every call's HangupIntent."""
    from agent.schemas import HangupIntent

    session = make_session()
    out_queue: asyncio.Queue[str] = asyncio.Queue()
    actuator = CallActuator(session=session, out_queue=out_queue, twilio_client=None)
    await actuator.execute(HangupIntent())  # must not raise; must not enqueue
    assert out_queue.empty()


# --- Rep-mode turn handler (D2) -------------------------------------------


async def test_rep_turn_speaks_reply_and_merges_partial_benefits(make_session: MakeSession):
    """Happy path: rep LLM returns reply + a partial Benefits extraction.
    The reply is spoken via SpeakIntent; non-None Benefits fields are merged
    into session.benefits while None fields are preserved (no overwrite)."""
    actuator = FakeActuator()
    runner, _, rep_llm = _build_runner(make_session(mode="rep"), actuator=actuator)
    rep_llm.responses = [
        RepTurnOutput(
            reply="Got it. What's the deductible?",
            extracted=Benefits(active=True, copay=30.0),
            phase="extracting",
        ),
        RepTurnOutput(
            reply="Thanks, that's everything I needed. Have a great day.",
            extracted=Benefits(deductible_remaining=250.0),
            phase="complete",
        ),
    ]
    try:
        await runner.start()
        runner.submit_transcript("Hi, this is Sam.")
        runner.submit_transcript("The deductible remaining is 250.")
        await wait_until(lambda: runner.session.done)
        # Two SpeakIntents (one per turn).
        speaks = [i for i in actuator.executed if isinstance(i, SpeakIntent)]
        assert len(speaks) == 2
        assert "deductible" in speaks[0].text
        # Both partial extractions merged: active + copay from turn 1, then
        # deductible_remaining added from turn 2 (without erasing the prior).
        assert runner.session.benefits.active is True
        assert runner.session.benefits.copay == 30.0
        assert runner.session.benefits.deductible_remaining == 250.0
        # phase="complete" → rep_complete.
        assert runner.session.completion_reason == "rep_complete"
    finally:
        await runner.stop()


async def test_rep_turn_timeout_speaks_filler_and_continues(
    make_session: MakeSession, monkeypatch: pytest.MonkeyPatch
):
    """If the rep LLM stalls past `REP_LLM_TIMEOUT_S`, the runner must NOT
    sit in dead silence — it speaks a brief filler and continues.

    Live testing observed a 16s anomalous Anthropic stall that produced
    17s of silence + 3 stacked replies once the call returned. The timeout
    + filler reply caps the silent gap at ~8s and gives the user verbal
    feedback that the agent is alive.

    Patches `REP_LLM_TIMEOUT_S` to 0.05s so the test runs in well under a
    second. `slow_mode_seconds=10` on the fake means the underlying
    `asyncio.sleep` is cancelled when wait_for trips — no real 10s wait.
    """
    monkeypatch.setattr("agent.call_session.REP_LLM_TIMEOUT_S", 0.05)
    actuator = FakeActuator()
    rep_llm = FakeAnthropicRepClient(slow_mode_seconds=10.0, responses=[])
    runner = CallSessionRunner(
        session=make_session(mode="rep"),
        ivr_llm=FakeIVRLLMClient(),
        rep_llm=rep_llm,
        tool_dispatcher=tools.dispatch,
        ivr_system_prompt="ivr",
        rep_system_prompt="rep",
        tools=[],
        actuator=actuator,
    )
    try:
        await runner.start()
        runner.submit_transcript("Hi, are you there?")
        await wait_until(
            lambda: any(
                isinstance(i, SpeakIntent) and "moment" in i.text.lower() for i in actuator.executed
            ),
            timeout=2.0,
            description="filler reply spoken after rep timeout",
        )
        speaks = [i for i in actuator.executed if isinstance(i, SpeakIntent)]
        assert len(speaks) == 1
        assert "moment" in speaks[0].text.lower()
        # No benefits extracted from a timed-out turn.
        assert runner.session.benefits.model_dump(exclude_none=True) == {}
        # Stuck counter NOT incremented — timeout ≠ phase=stuck.
        assert runner.session.stuck_turns == 0
        # Call is still alive (no completion_reason set).
        assert runner.session.completion_reason is None
    finally:
        await runner.stop()


async def test_rep_turn_timeout_skips_filler_when_interrupt_pending(
    make_session: MakeSession, monkeypatch: pytest.MonkeyPatch
):
    """If a barge-in fires between TimeoutError and the filler dispatch,
    `mark_interrupted` sets `_interrupt_requested` and drains out_queue.
    The filler MUST NOT be spoken in that race window — pushing it into
    a freshly-drained queue would defeat the barge-in. Locks the guard
    added at agent/call_session.py for this race.
    """
    monkeypatch.setattr("agent.call_session.REP_LLM_TIMEOUT_S", 0.05)
    actuator = FakeActuator()
    rep_llm = FakeAnthropicRepClient(slow_mode_seconds=10.0, responses=[])
    runner = CallSessionRunner(
        session=make_session(mode="rep"),
        ivr_llm=FakeIVRLLMClient(),
        rep_llm=rep_llm,
        tool_dispatcher=tools.dispatch,
        ivr_system_prompt="ivr",
        rep_system_prompt="rep",
        tools=[],
        actuator=actuator,
    )
    try:
        await runner.start()
        runner.submit_transcript("Hi, are you there?")
        # Pre-set the interrupt flag to simulate a barge-in landing in the
        # race window. (In production, mark_interrupted() sets this after
        # cancelling the current turn — but the timeout handler runs even
        # when _current_turn isn't yet cancelled, so the pre-set models
        # the worst case where mark_interrupted ran but the cancel hasn't
        # yet hit a yield point.)
        await wait_until(
            lambda: runner._current_turn is not None,  # pyright: ignore[reportPrivateUsage]
            timeout=1.0,
            description="turn task created",
        )
        runner._interrupt_requested = True  # pyright: ignore[reportPrivateUsage]
        # Wait for the consumer to finish the timed-out turn (it will hit
        # TimeoutError, see _interrupt_requested, and return early without
        # appending a filler placeholder).
        await wait_until(
            lambda: runner.session.turn_count > 0,
            timeout=2.0,
            description="rep turn completed (turn_count incremented)",
        )
        # No filler was spoken — the race guard caught it.
        speaks = [i for i in actuator.executed if isinstance(i, SpeakIntent)]
        assert len(speaks) == 0, (
            f"expected NO filler when interrupt pending; got {[s.text for s in speaks]}"
        )
    finally:
        await runner.stop()


async def test_rep_turn_complete_phase_ends_call(make_session: MakeSession):
    runner, _, rep_llm = _build_runner(make_session(mode="rep"), actuator=FakeActuator())
    rep_llm.responses = [
        RepTurnOutput(
            reply="That's everything. Thanks!",
            extracted=Benefits(),
            phase="complete",
        )
    ]
    try:
        await runner.start()
        runner.submit_transcript("Anything else?")
        await wait_until(lambda: runner.session.done)
        assert runner.session.completion_reason == "rep_complete"
    finally:
        await runner.stop()


async def test_rep_turn_two_consecutive_stuck_phases_end_call(make_session: MakeSession):
    """Watchdog: phase='stuck' for 2 consecutive turns terminates with
    rep_stuck. Distinct from llm_aborted_rep (deliberate fail_with_reason
    on the IVR side)."""
    runner, _, rep_llm = _build_runner(make_session(mode="rep"), actuator=FakeActuator())
    rep_llm.responses = [
        RepTurnOutput(reply="Could you repeat?", extracted=Benefits(), phase="stuck"),
        RepTurnOutput(
            reply="Sorry, I'm having trouble. Goodbye.", extracted=Benefits(), phase="stuck"
        ),
    ]
    try:
        await runner.start()
        runner.submit_transcript("garbled-1")
        runner.submit_transcript("garbled-2")
        await wait_until(lambda: runner.session.done)
        assert runner.session.completion_reason == "rep_stuck"
        assert runner.session.stuck_turns == 2
    finally:
        await runner.stop()


async def test_rep_turn_stuck_counter_resets_on_extracting_turn(make_session: MakeSession):
    """A non-stuck turn between two stuck turns resets the counter — only
    consecutive stuck turns count toward the watchdog."""
    runner, _, rep_llm = _build_runner(make_session(mode="rep"), actuator=FakeActuator())
    rep_llm.responses = [
        RepTurnOutput(reply="?", extracted=Benefits(), phase="stuck"),  # 1
        RepTurnOutput(  # advances → counter resets
            reply="Got it",
            extracted=Benefits(active=True),
            phase="extracting",
        ),
        RepTurnOutput(reply="?", extracted=Benefits(), phase="stuck"),  # 1 (just reset)
        RepTurnOutput(reply="goodbye", extracted=Benefits(), phase="stuck"),  # 2 → end
    ]
    try:
        await runner.start()
        for i in range(4):
            runner.submit_transcript(f"transcript-{i}")
        await wait_until(lambda: runner.session.done)
        assert runner.session.completion_reason == "rep_stuck"
    finally:
        await runner.stop()


async def test_rep_turn_empty_reply_does_not_speak(make_session: MakeSession):
    """Hold-music / silent-listening case: reply='' → actuator gets no
    SpeakIntent, but the rep turn still records a Turn in history."""
    actuator = FakeActuator()
    runner, _, rep_llm = _build_runner(make_session(mode="rep"), actuator=actuator)
    rep_llm.responses = [
        RepTurnOutput(reply="", extracted=Benefits(), phase="extracting"),
        RepTurnOutput(
            reply="That's everything",
            extracted=Benefits(),
            phase="complete",
        ),
    ]
    try:
        await runner.start()
        runner.submit_transcript("[hold music transcript]")
        runner.submit_transcript("Final answer.")
        await wait_until(lambda: runner.session.done)
        speaks = [i for i in actuator.executed if isinstance(i, SpeakIntent)]
        assert len(speaks) == 1  # only the second turn produced one
        assert speaks[0].text == "That's everything"
    finally:
        await runner.stop()


async def test_rep_turn_records_assistant_history_with_extracted(make_session: MakeSession):
    """The runner appends the assistant turn with the partial Benefits
    extraction recorded — useful for trace replay / debugging."""
    runner, _, rep_llm = _build_runner(make_session(mode="rep"), actuator=FakeActuator())
    extracted = Benefits(active=True, copay=30.0)
    rep_llm.responses = [RepTurnOutput(reply="Got it.", extracted=extracted, phase="complete")]
    try:
        await runner.start()
        runner.submit_transcript("Coverage active, copay 30.")
        await wait_until(lambda: runner.session.done)
        assistant_turns = [t for t in runner.session.history if t.role == "assistant"]
        assert len(assistant_turns) == 1
        assert assistant_turns[0].content == "Got it."
        # The partial extraction is recorded on the turn for trace inspection.
        assert assistant_turns[0].extracted == extracted
    finally:
        await runner.stop()


async def test_rep_turn_filters_tool_history_when_calling_llm(make_session: MakeSession):
    """Rep mode shouldn't see the IVR phase's tool_call/tool_result entries —
    the rep LLM only handles user/assistant text. With `mode="rep"` set
    directly (no transfer_to_rep flip), `rep_mode_index` is None, so the
    helper sees the full history minus tool entries."""
    from agent.schemas import Turn

    session = make_session(mode="rep")
    session.history.extend(
        [
            Turn(role="user", content="hello there"),
            Turn(role="tool_call", tool_call=ToolCall(name="speak", args={"text": "x"})),
            Turn(role="tool_result", content="dispatched"),
            Turn(role="assistant", content="how can I help?"),
        ]
    )
    runner, _, rep_llm = _build_runner(session, actuator=FakeActuator())
    rep_llm.responses = [RepTurnOutput(reply="hi sam", extracted=Benefits(), phase="complete")]
    try:
        await runner.start()
        runner.submit_transcript("hello, this is sam")
        await wait_until(lambda: runner.session.done)
        sent_history = rep_llm.calls[0][1]
        roles = [m["role"] for m in sent_history]
        # tool_call / tool_result filtered out; user + assistant + new user kept.
        assert roles == ["user", "assistant", "user"]
    finally:
        await runner.stop()


async def test_rep_turn_skips_pre_flip_history_after_transfer(make_session: MakeSession):
    """**Regression for the consecutive-user-message Anthropic 400 bug.**
    After transfer_to_rep flips the mode, the rep LLM must NOT see the IVR
    phase's user transcripts — those would arrive as consecutive user
    messages (Anthropic rejects). `rep_mode_index` records the flip point
    so `_rep_turn` slices history from there."""
    actuator = FakeActuator()
    runner, ivr_llm, rep_llm = _build_runner(make_session(), actuator=actuator)
    ivr_llm.responses = [
        IVRTurnResponse(tool_calls=[ToolCall(name="send_dtmf", args={"digits": "1"})]),
        IVRTurnResponse(tool_calls=[ToolCall(name="transfer_to_rep", args={})]),
    ]
    rep_llm.responses = [
        RepTurnOutput(
            reply="Hi Sam.",
            extracted=Benefits(),
            phase="complete",
        )
    ]
    try:
        await runner.start()
        runner.submit_transcript("Press 1 for eligibility")
        runner.submit_transcript("Connecting you to a representative")
        await wait_until(lambda: runner.session.mode == "rep")
        runner.submit_transcript("Hi, this is Sam.")
        await wait_until(lambda: runner.session.done)
        # The rep LLM's first call must have received exactly one user
        # message — the rep's greeting — not the two prior IVR transcripts.
        sent_history = rep_llm.calls[0][1]
        assert len(sent_history) == 1
        assert sent_history[0]["role"] == "user"
        assert sent_history[0]["content"] == "Hi, this is Sam."
    finally:
        await runner.stop()


# --- Mode-aware routing + transfer_to_rep integration (D3) ----------------


async def test_default_mode_is_ivr(make_session: MakeSession):
    """A fresh session starts in IVR mode — `_run_turn` routes to
    `_ivr_turn` until something flips the mode."""
    runner, ivr_llm, rep_llm = _build_runner(make_session(), actuator=FakeActuator())
    ivr_llm.responses = [
        IVRTurnResponse(tool_calls=[ToolCall(name="complete_call", args={"reason": "user_hangup"})])
    ]
    try:
        await runner.start()
        runner.submit_transcript("kickoff")
        await wait_until(lambda: runner.session.done)
        # IVR LLM was called, rep LLM was not.
        assert len(ivr_llm.calls) == 1
        assert len(rep_llm.calls) == 0
    finally:
        await runner.stop()


async def test_transfer_to_rep_flips_mode_and_next_turn_routes_to_rep(make_session: MakeSession):
    """Integration: IVR LLM emits transfer_to_rep on turn 1 → mode flips →
    next transcript routes to _rep_turn (the rep LLM gets called)."""
    actuator = FakeActuator()
    runner, ivr_llm, rep_llm = _build_runner(make_session(), actuator=actuator)
    ivr_llm.responses = [
        IVRTurnResponse(tool_calls=[ToolCall(name="transfer_to_rep", args={})]),
    ]
    rep_llm.responses = [
        RepTurnOutput(
            reply="Hi Sam, calling about benefits.",
            extracted=Benefits(),
            phase="complete",
        ),
    ]
    try:
        await runner.start()
        runner.submit_transcript("Connecting you to a representative.")
        # Mode flips during turn 1's tool dispatch.
        await wait_until(lambda: runner.session.mode == "rep")
        runner.submit_transcript("Hi, this is Sam.")
        await wait_until(lambda: runner.session.done)
        assert runner.session.mode == "rep"
        assert len(ivr_llm.calls) == 1
        assert len(rep_llm.calls) == 1
        assert runner.session.completion_reason == "rep_complete"
        # The rep's reply was spoken.
        speaks = [i for i in actuator.executed if isinstance(i, SpeakIntent)]
        assert any("Hi Sam" in s.text for s in speaks)
    finally:
        await runner.stop()


async def test_rep_turn_cancellation_propagates_cleanly(make_session: MakeSession):
    """Symmetry with the IVR side: a slow rep LLM call cancelled by
    mark_interrupted raises CancelledError cleanly, the consumer catches it,
    no Benefits merge / history append / phase tick happens."""
    actuator = FakeActuator()
    runner, _, rep_llm = _build_runner(make_session(mode="rep"), actuator=actuator)
    rep_llm.slow_mode_seconds = 5.0
    rep_llm.responses = [
        RepTurnOutput(reply="won't reach", extracted=Benefits(active=True), phase="extracting"),
    ]
    try:
        await runner.start()
        runner.submit_transcript("Hi, this is Sam.")
        await wait_until(
            lambda: runner._current_turn is not None and not runner._current_turn.done()  # pyright: ignore[reportPrivateUsage]
        )
        runner.mark_interrupted()
        await wait_until(lambda: runner._current_turn is not None and runner._current_turn.done())  # pyright: ignore[reportPrivateUsage]
        # Cancelled before the LLM returned → no merge, no assistant Turn, no
        # phase advancement, no completion.
        assert runner.session.benefits.active is None
        assert not any(t.role == "assistant" for t in runner.session.history)
        assert runner.session.completion_reason is None
        assert not any(isinstance(i, SpeakIntent) for i in actuator.executed)
    finally:
        await runner.stop()
