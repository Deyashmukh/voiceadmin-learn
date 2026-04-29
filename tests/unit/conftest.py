"""Fakes + fixtures for offline unit tests. Zero network."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple, Protocol, TypedDict, Unpack

import pytest
from pydantic import BaseModel

from agent.schemas import (
    Benefits,
    CallMode,
    CallSession,
    CompletionReason,
    IVRTurnResponse,
    PatientInfo,
    SideEffectIntent,
    Turn,
)


@pytest.fixture(autouse=True)
def _redirect_benefits_log(  # pyright: ignore[reportUnusedFunction]
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Send the per-call benefits JSONL to a tmp file for the duration of
    each test, so unit tests don't pollute the repo root with `benefits.jsonl`
    every time they run a complete call. Pytest discovers this by name; the
    pyright ignore is for the `autouse` indirection."""
    monkeypatch.setenv("BENEFITS_LOG_PATH", str(tmp_path / "benefits.jsonl"))


class CallSessionOverrides(TypedDict, total=False):
    """Per-test mutable fields on `CallSession` that `make_session(**kwargs)`
    can override. Mirrors the dataclass's mutable fields so a typo like
    `make_session(mod="rep")` fails type-check instead of silently no-op."""

    mode: CallMode
    recent_menu_options: list[str]
    benefits: Benefits
    history: list[Turn]
    turn_count: int
    ivr_no_progress_turns: int
    stuck_turns: int
    completion_reason: CompletionReason | None
    completion_note: str | None
    rep_mode_index: int | None


class MakeSession(Protocol):
    """Factory signature for the `make_session` fixture. Keyword args are
    constrained to `CallSessionOverrides` so typos are caught at type-check."""

    def __call__(self, **overrides: Unpack[CallSessionOverrides]) -> CallSession: ...


class RepCall(NamedTuple):
    """Recorded `FakeAnthropicRepClient.complete_structured` invocation."""

    system: str
    history: list[dict[str, object]]


class IVRCall(NamedTuple):
    """Recorded `FakeIVRLLMClient.complete_with_tools` invocation."""

    system: str
    history: list[Turn]
    tools: list[dict[str, object]]


async def wait_until(predicate: Callable[[], bool], timeout: float = 1.0) -> None:
    """Yield to the event loop until `predicate()` returns truthy, or fail
    after a bounded number of iterations.

    Iteration-cap (not wall-clock deadline) so a slow scheduler doesn't trip
    spurious timeouts: predicates that depend on cancellation, queue puts, or
    other event-loop-driven state will see those changes within a few yields
    regardless of wall-clock pressure. `asyncio.sleep(0.001)` rather than
    `sleep(0)` because some fakes (`slow_mode_seconds`) advance via real-time
    timers — `sleep(0)` would starve those timers and never let the predicate
    flip. Caveat: under sustained event-loop blocking (e.g. a long synchronous
    call that never yields), iterations stall just as the old wall-clock
    version did. The change buys variance-immunity, not blocking-immunity.
    """
    # 1 ms per iteration → `timeout` seconds maps to `timeout * 1000` iterations.
    # Floor at 1 to keep `timeout=0` from short-circuiting the first check.
    max_iterations = max(1, int(timeout * 1000))
    for _ in range(max_iterations):
        if predicate():
            return
        await asyncio.sleep(0.001)
    raise AssertionError(f"timed out waiting on {predicate.__name__}")


@dataclass
class FakeAnthropicRepClient:
    """Stand-in for `agent.llm_client.AnthropicRepClient` in offline tests.

    Mirrors `complete_structured(system, history, schema)` shape; pops queued
    responses in order. Records each call so tests can assert on the rendered
    prompt / message history."""

    responses: list[BaseModel] = field(default_factory=list[BaseModel])
    exception: Exception | None = None
    slow_mode_seconds: float = 0.0
    calls: list[RepCall] = field(default_factory=list[RepCall])

    async def complete_structured[T: BaseModel](
        self,
        system: str,
        history: list[dict[str, object]],
        schema: type[T],
        max_tokens: int = 1024,
    ) -> T:
        self.calls.append(RepCall(system, history))
        if self.slow_mode_seconds:
            await asyncio.sleep(self.slow_mode_seconds)
        if self.exception is not None:
            raise self.exception
        if not self.responses:
            raise AssertionError("no responses queued for FakeAnthropicRepClient")
        response = self.responses.pop(0)
        assert isinstance(response, schema), (
            f"queued {type(response).__name__}, asked for {schema.__name__}"
        )
        return response


@dataclass
class FakeIVRLLMClient:
    """Fake tool-calling IVR LLM client.

    Queue `IVRTurnResponse` instances; each call pops one. Records every
    call so tests can assert on the rendered prompt + history shape."""

    responses: list[IVRTurnResponse] = field(default_factory=list[IVRTurnResponse])
    exception: Exception | None = None
    slow_mode_seconds: float = 0.0
    calls: list[IVRCall] = field(default_factory=list[IVRCall])

    async def complete_with_tools(
        self,
        system: str,
        history: list[Turn],
        tools: list[dict[str, object]],
        temperature: float = 0.1,
    ) -> IVRTurnResponse:
        # Snapshot the history at call time so the test can see exactly what
        # the LLM was sent — appending to history happens after this returns.
        self.calls.append(IVRCall(system, list(history), tools))
        # Pop the response BEFORE any sleep so a cancellation mid-call burns
        # the response (matching real-LLM semantics — a cancelled call
        # doesn't carry its commitment to the next attempt).
        if self.exception is not None:
            raise self.exception
        if not self.responses:
            raise AssertionError("no responses queued for FakeIVRLLMClient")
        response = self.responses.pop(0)
        if self.slow_mode_seconds:
            await asyncio.sleep(self.slow_mode_seconds)
        return response


@dataclass
class FakeActuator:
    """Records every executed `SideEffectIntent`. Tests assert the runner
    called the actuator with the right intents in the right order."""

    executed: list[SideEffectIntent] = field(default_factory=list[SideEffectIntent])
    exception: Exception | None = None

    async def execute(self, intent: SideEffectIntent) -> None:
        self.executed.append(intent)
        if self.exception is not None:
            raise self.exception


@pytest.fixture
def patient() -> PatientInfo:
    return PatientInfo(
        member_id="M123456",
        first_name="Alice",
        last_name="Example",
        dob="1980-05-12",
    )


@pytest.fixture
def benefits() -> Benefits:
    return Benefits(
        active=True,
        deductible_remaining=250.0,
        copay=30.0,
        coinsurance=0.2,
        out_of_network_coverage=False,
    )


@pytest.fixture
def make_session(patient: PatientInfo) -> MakeSession:
    """Factory for `CallSession`. Per-test overrides via kwargs:
    `make_session(mode="rep", recent_menu_options=["1", "2"])`."""

    def _make(**overrides: Unpack[CallSessionOverrides]) -> CallSession:
        s = CallSession(call_sid="CA-test", patient=patient)
        for key, value in overrides.items():
            setattr(s, key, value)
        return s

    return _make
