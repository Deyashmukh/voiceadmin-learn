"""Shared data models and Protocol interfaces for the voice agent."""

from __future__ import annotations

from typing import Literal, Protocol, TypedDict

from pydantic import BaseModel, Field


class PatientInfo(BaseModel):
    """Caller/subscriber info we authenticate with on the payer call."""

    member_id: str
    first_name: str
    last_name: str
    dob: str  # ISO date string; not a date() because TwiML-friendly


class Benefits(BaseModel):
    """Structured output of the Extract Benefits node — the goal of the call."""

    active: bool
    deductible_remaining: float | None = None
    copay: float | None = None
    coinsurance: float | None = None
    out_of_network_coverage: bool | None = None


class ClassifierResult(BaseModel):
    """Output of the rule-based IVR keyword classifier."""

    outcome: Literal["dtmf", "speak", "unknown"]
    dtmf: str | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)


class CallState(TypedDict, total=False):
    """Per-call state owned by GraphRunner; each ainvoke mutates a copy."""

    current_node: str
    patient: PatientInfo | None
    extracted: Benefits | None
    turn_count: int
    transcript: str
    response_text: str | None
    fallback_reason: str | None


class LLMClient(Protocol):
    """Injected LLM interface. Two methods keep reasoning and extraction separate."""

    async def complete_free_form(self, system: str, user: str) -> str: ...

    async def complete_structured[T: BaseModel](
        self, system: str, user: str, schema: type[T]
    ) -> T: ...


class IVRClassifier(Protocol):
    """Injected rule-based IVR keyword classifier."""

    def classify(self, transcript: str) -> ClassifierResult: ...
