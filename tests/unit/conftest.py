"""Fakes + fixtures for offline unit tests. Zero network."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest
from pydantic import BaseModel

from agent.schemas import (
    Benefits,
    CallSession,
    IVRTurnResponse,
    PatientInfo,
    SideEffectIntent,
    Turn,
)


@dataclass
class FakeAnthropicRepClient:
    """Stand-in for `agent.llm_client.AnthropicRepClient` in offline tests.

    Mirrors `complete_structured(system, history, schema)` shape; pops queued
    responses in order. Records each call so tests can assert on the rendered
    prompt / message history."""

    responses: list[BaseModel] = field(default_factory=list)
    exception: Exception | None = None
    slow_mode_seconds: float = 0.0
    calls: list[tuple[str, list[dict[str, object]]]] = field(default_factory=list)

    async def complete_structured[T: BaseModel](
        self,
        system: str,
        history: list[dict[str, object]],
        schema: type[T],
        max_tokens: int = 1024,
    ) -> T:
        self.calls.append((system, history))
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

    responses: list[IVRTurnResponse] = field(default_factory=list)
    exception: Exception | None = None
    slow_mode_seconds: float = 0.0
    calls: list[tuple[str, list[Turn], list[dict[str, object]]]] = field(default_factory=list)

    async def complete_with_tools(
        self,
        system: str,
        history: list[Turn],
        tools: list[dict[str, object]],
        temperature: float = 0.1,
    ) -> IVRTurnResponse:
        # Snapshot the history at call time so the test can see exactly what
        # the LLM was sent — appending to history happens after this returns.
        self.calls.append((system, list(history), tools))
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

    executed: list[SideEffectIntent] = field(default_factory=list)
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
def make_session(patient: PatientInfo):
    """Factory for `CallSession`. Per-test overrides via kwargs:
    `make_session(mode="rep", recent_menu_options=["1", "2"])`."""

    def _make(**overrides) -> CallSession:
        s = CallSession(call_sid="CA-test", patient=patient)
        for key, value in overrides.items():
            setattr(s, key, value)
        return s

    return _make
