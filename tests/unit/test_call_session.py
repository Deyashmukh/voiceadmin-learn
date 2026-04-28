"""Unit tests for M5'/D1: CallSessionRunner — IVR-only turn loop, watchdog,
interrupt + queue drain. Zero network — IVR LLM is faked; the default
CallActuator (with no twilio_client) routes SpeakIntent into out_queue
natively, so tests can read spoken text from there directly."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import pytest
import structlog.contextvars

from agent import tools
from agent.actuator import CallActuator
from agent.call_session import CallSessionRunner
from agent.schemas import (
    DTMFIntent,
    IVRTurnResponse,
    SideEffectIntent,
    ToolCall,
)

from .conftest import FakeActuator, FakeIVRLLMClient


def _build_runner(
    session,
    *,
    ivr_llm: FakeIVRLLMClient | None = None,
    actuator=None,
) -> tuple[CallSessionRunner, FakeIVRLLMClient]:
    """Construct a CallSessionRunner with the FakeIVRLLMClient pre-installed.
    Default actuator is the runner's auto-built `CallActuator` (no twilio_client),
    which routes SpeakIntent → out_queue natively. Tests pass an explicit
    `FakeActuator` when they want to assert on intents directly."""
    llm = ivr_llm or FakeIVRLLMClient()
    runner = CallSessionRunner(
        session=session,
        ivr_llm=llm,
        tool_dispatcher=tools.dispatch,
        ivr_system_prompt="ivr-system-prompt",
        tools=[{"name": "send_dtmf"}],
        actuator=actuator,
    )
    return runner, llm


async def _wait_until(predicate, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"timed out waiting on {predicate.__name__}")


# --- submit_transcript: drop-oldest -----------------------------------------


async def test_submit_transcript_drops_oldest_when_full(make_session):
    runner, _ = _build_runner(make_session())
    for i in range(15):
        runner.submit_transcript(f"item-{i}")
    assert runner.in_queue.qsize() == 8
    drained: list[str] = []
    while not runner.in_queue.empty():
        drained.append(runner.in_queue.get_nowait())
    assert drained[-1] == "item-14"
    assert "item-0" not in drained


# --- IVR turn happy path ----------------------------------------------------


async def test_ivr_turn_dispatches_send_dtmf_intent(make_session):
    """Turn 1 sends DTMF, turn 2 completes the call. Use a FakeActuator so
    we can assert on the intents directly."""
    actuator = FakeActuator()
    runner, ivr_llm = _build_runner(make_session(), actuator=actuator)
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
        await _wait_until(lambda: runner.session.done)
        assert any(isinstance(i, DTMFIntent) and i.digits == "1" for i in actuator.executed)
        assert runner.session.completion_reason == "benefits_extracted"
        assert runner.session.done
    finally:
        await runner.stop()


async def test_ivr_turn_records_history(make_session):
    runner, ivr_llm = _build_runner(make_session(), actuator=FakeActuator())
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
        await _wait_until(lambda: runner.session.done)
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


async def test_speak_intent_is_pushed_to_out_queue(make_session):
    """The default CallActuator routes SpeakIntent.text into out_queue —
    that's the production wire to Cartesia via state_processor."""
    runner, ivr_llm = _build_runner(make_session())
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


async def test_validator_rejection_does_not_advance_call_state(make_session):
    """Bad DTMF digit (not in recent menu options): dispatcher returns
    `advanced_call_state=False`, so the turn doesn't count as progress and
    the watchdog ticks. Two such turns trip the watchdog."""
    actuator = FakeActuator()
    runner, ivr_llm = _build_runner(
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
        await _wait_until(lambda: runner.session.done)
        assert runner.session.completion_reason == "ivr_no_progress"
        # No DTMF intents reached the actuator (validator rejected both).
        assert not any(isinstance(i, DTMFIntent) for i in actuator.executed)
    finally:
        await runner.stop()


# --- Watchdog: zero-tool-call case (Groq timeout-style) --------------------


async def test_watchdog_trips_on_zero_tool_call_turns(make_session):
    """If the LLM produces no tool calls at all (timeout / hallucinated text
    response with no tools), the watchdog must still tick. Two such turns
    in a row → terminate."""
    runner, ivr_llm = _build_runner(make_session(), actuator=FakeActuator())
    ivr_llm.responses = [IVRTurnResponse(), IVRTurnResponse()]
    try:
        await runner.start()
        runner.submit_transcript("first")
        runner.submit_transcript("second")
        await _wait_until(lambda: runner.session.done)
        assert runner.session.completion_reason == "ivr_no_progress"
    finally:
        await runner.stop()


async def test_watchdog_resets_on_an_advancing_turn(make_session):
    """One bad turn followed by a good turn must reset the counter."""
    runner, ivr_llm = _build_runner(make_session(), actuator=FakeActuator())
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
        await _wait_until(lambda: runner.session.done)
        assert runner.session.completion_reason == "ivr_no_progress"
    finally:
        await runner.stop()


# --- Interrupt: queue drains + task.cancel --------------------------------


async def test_mark_interrupted_drains_out_queue(make_session):
    """A turn that finished moments before the user started speaking would
    otherwise still get spoken after the barge-in. Drain first."""
    runner, _ = _build_runner(make_session())
    await runner.out_queue.put("stale-1")
    await runner.out_queue.put("stale-2")
    runner.mark_interrupted()
    assert runner.out_queue.empty()


async def test_interrupt_drains_in_queue(make_session):
    """Barge-in should not let stale queued transcripts feed the next turn."""
    runner, _ = _build_runner(make_session())
    runner.submit_transcript("stale-A")
    runner.submit_transcript("stale-B")
    assert runner.in_queue.qsize() == 2
    runner.mark_interrupted()
    assert runner.in_queue.empty()


async def test_mark_interrupted_cancels_in_flight_turn(make_session):
    """A slow LLM call (simulated via slow_mode_seconds) is cancelled cleanly
    by mark_interrupted — the turn task receives CancelledError, the consumer
    catches it, and the loop continues."""
    runner, ivr_llm = _build_runner(make_session(), actuator=FakeActuator())
    ivr_llm.slow_mode_seconds = 5.0
    ivr_llm.responses = [
        IVRTurnResponse(tool_calls=[ToolCall(name="speak", args={"text": "won't reach"})])
    ]
    try:
        await runner.start()
        runner.submit_transcript("kickoff")
        await _wait_until(
            lambda: runner._current_turn is not None and not runner._current_turn.done()
        )
        runner.mark_interrupted()
        await _wait_until(lambda: runner._current_turn is not None and runner._current_turn.done())
        # Turn was cancelled → no actuator call happened, no completion set.
        assert runner.session.completion_reason is None
    finally:
        await runner.stop()


async def test_mark_interrupted_does_not_complete_call(make_session):
    """Interrupt is a turn-level cancel, not a session-level abort. The runner
    keeps consuming after the interrupt clears."""
    runner, ivr_llm = _build_runner(make_session(), actuator=FakeActuator())
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
        await _wait_until(
            lambda: runner._current_turn is not None and not runner._current_turn.done()
        )
        runner.mark_interrupted()
        ivr_llm.slow_mode_seconds = 0.0
        runner.submit_transcript("second")
        await _wait_until(lambda: runner.session.done)
        assert runner.session.completion_reason == "user_hangup"
    finally:
        await runner.stop()


# --- Cancellation mid tool-loop preserves history pairing -----------------


async def test_cancellation_mid_tool_dispatch_pairs_history(make_session):
    """If a turn is cancelled while awaiting tool dispatch, the history must
    still have a `tool_result` paired to the just-appended `tool_call` —
    otherwise Groq's tool API rejects the next turn."""

    async def _slow_dispatcher(call, session):
        await asyncio.sleep(5.0)  # cancelled before dispatch finishes
        return await tools.dispatch(call, session)

    session = make_session()
    ivr_llm = FakeIVRLLMClient(
        responses=[IVRTurnResponse(tool_calls=[ToolCall(name="speak", args={"text": "x"})])]
    )
    runner = CallSessionRunner(
        session=session,
        ivr_llm=ivr_llm,
        tool_dispatcher=_slow_dispatcher,
        ivr_system_prompt="x",
        tools=[],
        actuator=FakeActuator(),
    )
    try:
        await runner.start()
        runner.submit_transcript("kickoff")
        await _wait_until(
            lambda: runner._current_turn is not None and not runner._current_turn.done()
        )
        runner.mark_interrupted()
        await _wait_until(lambda: runner._current_turn is not None and runner._current_turn.done())
        # History must end with a tool_result paired to the tool_call.
        roles = [t.role for t in session.history]
        # user → tool_call → tool_result(cancelled)
        assert roles == ["user", "tool_call", "tool_result"]
        assert "cancelled" in (session.history[-1].content or "")
    finally:
        await runner.stop()


# --- stop() shutdown -------------------------------------------------------


async def test_stop_cancels_in_flight_turn_quickly(make_session):
    """stop() must complete promptly even when a turn is mid-LLM-call."""
    runner, ivr_llm = _build_runner(make_session(), actuator=FakeActuator())
    ivr_llm.slow_mode_seconds = 5.0
    ivr_llm.responses = [
        IVRTurnResponse(tool_calls=[ToolCall(name="speak", args={"text": "won't reach"})])
    ]
    await runner.start()
    runner.submit_transcript("kickoff")
    await asyncio.sleep(0.05)
    await asyncio.wait_for(runner.stop(), timeout=1.0)


# --- Consumer exits cleanly when the call completes ------------------------


async def test_consumer_exits_after_complete_call_without_extra_transcript(make_session):
    """When the LLM emits `complete_call`, the consumer must exit the loop
    immediately rather than blocking on `in_queue.get()` waiting for a
    transcript that will never arrive."""
    runner, ivr_llm = _build_runner(make_session(), actuator=FakeActuator())
    ivr_llm.responses = [
        IVRTurnResponse(
            tool_calls=[ToolCall(name="complete_call", args={"reason": "benefits_extracted"})]
        ),
    ]
    await runner.start()
    runner.submit_transcript("Done with menu")
    # Wait for both: session.done AND consumer task itself completes.
    await _wait_until(lambda: runner.session.done)
    await _wait_until(lambda: runner._consumer is not None and runner._consumer.done())
    assert runner.session.completion_reason == "benefits_extracted"
    # Consumer is done — stop() should be a no-op for the consumer task.
    await runner.stop()


# --- Consumer-died safety net ---------------------------------------------


async def test_consumer_death_sets_completion_reason(make_session):
    """If the consumer task crashes unhandled, the pipeline-side observer
    needs a signal to end the call instead of hanging on out_queue forever.
    `_on_consumer_done` sets `completion_reason = consumer_died`."""

    async def _exploding_dispatcher(call, session):
        raise RuntimeError("boom")

    session = make_session()
    ivr_llm = FakeIVRLLMClient(
        responses=[IVRTurnResponse(tool_calls=[ToolCall(name="speak", args={"text": "x"})])]
    )
    runner = CallSessionRunner(
        session=session,
        ivr_llm=ivr_llm,
        tool_dispatcher=_exploding_dispatcher,
        ivr_system_prompt="x",
        tools=[],
        actuator=FakeActuator(),
    )
    try:
        await runner.start()
        runner.submit_transcript("kickoff")
        await _wait_until(lambda: runner._consumer is not None and runner._consumer.done())
        assert runner.session.completion_reason == "consumer_died"
    finally:
        await runner.stop()


# --- contextvar binding ----------------------------------------------------


async def test_call_sid_bound_in_contextvars_during_turn(make_session):
    """The runner binds `call_sid` and `turn_index` as structlog contextvars
    before kicking off each turn — so any log emission from inside the turn
    (handlers, actuator, dispatcher) carries the call's identity."""
    captured: list[dict[str, object]] = []

    @dataclass
    class _ContextvarCapturingActuator:
        async def execute(self, intent: SideEffectIntent) -> None:
            captured.append(dict(structlog.contextvars.get_contextvars()))

    runner, ivr_llm = _build_runner(make_session(), actuator=_ContextvarCapturingActuator())
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
        await _wait_until(lambda: runner.session.done)
        assert len(captured) == 2
        assert captured[0]["call_sid"] == runner.session.call_sid
        assert captured[1]["call_sid"] == runner.session.call_sid
        assert captured[0]["turn_index"] == 0
        assert captured[1]["turn_index"] == 1
    finally:
        await runner.stop()


# --- Each queued transcript drives its own turn ----------------------------


async def test_each_queued_transcript_drives_its_own_turn(make_session):
    """Without barge-in, multiple queued transcripts must each fire a
    distinct turn — distinct user utterances are not collapsed."""
    runner, ivr_llm = _build_runner(make_session(), actuator=FakeActuator())
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
        await _wait_until(lambda: runner.session.done)
        assert len(ivr_llm.calls) == 3
        user_contents = [t.content for t in runner.session.history if t.role == "user"]
        assert user_contents == ["first", "second", "third"]
    finally:
        await runner.stop()


# --- DTMFIntent without a twilio client raises (CallActuator path) --------


async def test_call_actuator_without_twilio_client_rejects_dtmf(make_session):
    """The default CallActuator raises if asked to dispatch DTMF without a
    Twilio client. Locks the precondition for live-call wiring."""
    session = make_session()
    out_queue: asyncio.Queue[str] = asyncio.Queue()
    actuator = CallActuator(session=session, out_queue=out_queue, twilio_client=None)
    with pytest.raises(RuntimeError, match="DTMFIntent emitted but actuator has no twilio_client"):
        await actuator.execute(DTMFIntent(digits="1"))
