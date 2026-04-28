"""Fakes + fixtures for offline unit tests. Zero network."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest
from pydantic import BaseModel

from agent.schemas import Benefits, CallSession, ClassifierResult, PatientInfo


@dataclass
class FakeLLMClient:
    """Deterministic fake. Queue up responses per method; raise to simulate errors."""

    structured_responses: list[BaseModel] = field(default_factory=list)
    structured_exception: Exception | None = None
    slow_mode_seconds: float = 0.0
    calls: list[tuple[str, str, str]] = field(default_factory=list)

    async def complete_free_form(self, system: str, user: str) -> str:
        # No M3 node uses free-form completion; Protocol requires the method.
        self.calls.append(("free_form", system, user))
        if self.slow_mode_seconds:
            await asyncio.sleep(self.slow_mode_seconds)
        return ""

    async def complete_structured[T: BaseModel](self, system: str, user: str, schema: type[T]) -> T:
        self.calls.append(("structured", system, user))
        if self.slow_mode_seconds:
            await asyncio.sleep(self.slow_mode_seconds)
        if self.structured_exception is not None:
            raise self.structured_exception
        if not self.structured_responses:
            raise AssertionError("no structured_responses queued")
        response = self.structured_responses.pop(0)
        assert isinstance(response, schema), (
            f"queued {type(response).__name__}, asked for {schema.__name__}"
        )
        return response


@dataclass
class FakeClassifier:
    """Returns queued ClassifierResult per call; defaults to outcome='speak'."""

    results: list[ClassifierResult] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)

    def classify(self, transcript: str, patient: PatientInfo | None = None) -> ClassifierResult:
        self.calls.append(transcript)
        if not self.results:
            return ClassifierResult(outcome="speak", confidence=1.0)
        return self.results.pop(0)


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
def fake_llm() -> FakeLLMClient:
    return FakeLLMClient()


@pytest.fixture
def fake_classifier() -> FakeClassifier:
    return FakeClassifier()


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
