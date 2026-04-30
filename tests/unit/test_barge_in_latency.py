# pyright: strict
"""Measure-only barge-in cancel-to-silence latency.

This test does NOT assert a budget. The historical `<150ms` figure was a
manual observation from earlier in the project, never enforced in code. A
later PR converts the measurement into a regression assertion once a budget
multiplier is set against this baseline.

What we measure: wall-clock from `runner.mark_interrupted()` returning to the
`_current_turn` task transitioning to `done()` (cancelled). This is the
runner-side cancel-to-silence interval — the point at which no further
`SpeakIntent`s can be produced and `out_queue` has been drained. It is NOT
end-to-end audio silence (which also includes Pipecat's frame interrupt
path); the runner-side delta is the bound the rest of the pipeline composes
against.

Why the slow-LLM rep path: `complete_structured` is the longest-blocking
await in a typical rep turn, so cancelling mid-call exercises the worst case
the runner controls. We park the fake LLM in `await asyncio.sleep(slow_mode)`
so the cancellation lands in the awaited coroutine — same shape as a real
Anthropic SDK call cancelled mid-stream.

Behavior assertions only:
  - the in-flight turn task ends up cancelled
  - `out_queue` is empty after `mark_interrupted()`

The latency numbers are recorded via `print(...)` so `pytest -s` (and the CI
runner with `-s`) shows them, and aggregated across runs at the end of the
test for a min/median/p95/max readout.

Resolution caveat: the wait loop polls at 1ms granularity (see `wait_until`
in `conftest.py`), so measured values are quantized to ~1ms. Values
clustered around 1ms mean "cancel returns within one event-loop tick," not
literally 1ms.
"""

from __future__ import annotations

import statistics
import time

from agent import tools
from agent.call_session import CallSessionRunner
from agent.schemas import Benefits, RepTurnOutput

from .conftest import (
    FakeActuator,
    FakeAnthropicRepClient,
    FakeIVRLLMClient,
    MakeSession,
    wait_until,
)

# Chosen to give a real p95 (statistics.quantiles(n=20)[18] requires at
# least ~20 samples to stop collapsing toward max under linear interpolation)
# while keeping wallclock under ~10s — each iteration spawns a fresh runner
# and runs through the cancel path, which is fast.
RUNS = 20

# How long the fake LLM stays parked inside its `await asyncio.sleep` before
# it would have returned. Must exceed `wait_until`'s default 1.0s timeout —
# if a regression breaks `mark_interrupted()` and the cancel never lands,
# `wait_until` will time out and AssertionError before the LLM completes,
# so the test fails closed instead of silently passing on a slow-but-not-
# -actually-cancelled measurement.
SLOW_LLM_SECONDS = 5.0


async def _measure_one_cancel_to_silence(make_session: MakeSession) -> float:
    """Run one barge-in cycle and return cancel-to-silence latency in seconds.

    Setup: rep-mode session, slow rep LLM. Submit a transcript so the
    consumer spawns a turn task that parks inside `complete_structured`.
    Once the task is in flight, timestamp, fire `mark_interrupted()`, wait
    for the task to reach `done()` (cancelled), timestamp again. The delta
    is the latency.

    The runner is built fresh per measurement: a single-shot fake LLM has
    only one queued response, and turn cancellation is one-way per task.
    """
    rep_llm = FakeAnthropicRepClient(
        slow_mode_seconds=SLOW_LLM_SECONDS,
        responses=[
            RepTurnOutput(reply="won't reach", extracted=Benefits(active=True), phase="extracting"),
        ],
    )
    runner = CallSessionRunner(
        session=make_session(mode="rep"),
        ivr_llm=FakeIVRLLMClient(),
        rep_llm=rep_llm,
        tool_dispatcher=tools.dispatch,
        ivr_system_prompt="ivr-system-prompt",
        rep_system_prompt="rep-persona-prompt",
        tools=[{"name": "send_dtmf"}],
        actuator=FakeActuator(),
    )
    try:
        await runner.start()
        runner.submit_transcript("Hi, this is Sam.")
        await wait_until(
            lambda: runner._current_turn is not None and not runner._current_turn.done(),  # pyright: ignore[reportPrivateUsage]
            description="rep turn task to enter in-flight LLM await",
        )
        # Pre-load `out_queue` so we can also assert the drain happened. Even
        # though the rep handler hasn't enqueued anything yet (it's blocked
        # on the LLM), real barge-ins can land just after a SpeakIntent has
        # been routed but before TTS picks it up — `mark_interrupted` must
        # drain it.
        await runner.out_queue.put("stale-pre-bargein")

        # Snapshot the in-flight task so the await target survives the
        # `mark_interrupted` call (which leaves `_current_turn` set to the
        # cancelled task, but a follow-on transcript could replace it).
        in_flight = runner._current_turn  # pyright: ignore[reportPrivateUsage]
        assert in_flight is not None

        t0 = time.monotonic()
        runner.mark_interrupted()
        # Wait deterministically for the task to reach the cancelled-done
        # state. Iteration-capped helper, not wall-clock polling — keeps the
        # test variance-immune under loaded CI schedulers.
        await wait_until(
            lambda: in_flight.done(),
            description="in-flight turn task to reach done() after cancel",
        )
        # `t1` lands one `wait_until` tick AFTER the task transitions, not
        # at the transition itself. With a 1ms poll interval, measurements
        # are quantized to ~1ms and a sub-millisecond cancel reads as ~1ms.
        # That floor is structural to this measurement approach.
        t1 = time.monotonic()

        assert in_flight.cancelled(), (
            "in-flight turn task must end cancelled after mark_interrupted"
        )
        assert runner.out_queue.empty(), "out_queue must be drained by mark_interrupted"

        return t1 - t0
    finally:
        await runner.stop()


async def test_barge_in_cancel_to_silence_latency(make_session: MakeSession) -> None:
    """Report cancel-to-silence latency over `RUNS` iterations. No budget
    assertion; only the cancellation behavior is gated.

    Output goes through `print` so `pytest -s` surfaces the numbers in CI
    logs and locally — the measurement is the deliverable, not just a green
    check.
    """
    samples_ms: list[float] = []
    for i in range(RUNS):
        latency_ms = (await _measure_one_cancel_to_silence(make_session)) * 1000
        samples_ms.append(latency_ms)
        print(f"[barge-in latency] run {i + 1}/{RUNS}: {latency_ms:.3f} ms")

    # 95th percentile via 20-quantile cut #18 — a real p95 with N=20 samples.
    p95_ms = statistics.quantiles(samples_ms, n=20)[18]
    summary = {
        "runs": RUNS,
        "min_ms": min(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "p95_ms": p95_ms,
        "max_ms": max(samples_ms),
    }
    print(f"[barge-in latency] summary: {summary}")
