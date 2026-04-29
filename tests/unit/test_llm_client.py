"""Unit tests for M5'/C: Anthropic rep-LLM client + the FakeAnthropicRepClient
seam. Zero network — the SDK call sites are mocked."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.llm_client import REP_MODEL, AnthropicRepClient
from agent.schemas import Benefits, RepTurnOutput

from .conftest import FakeAnthropicRepClient


@pytest.fixture
def rep_output() -> RepTurnOutput:
    return RepTurnOutput(
        reply="Got it. Could you also confirm the deductible remaining?",
        extracted=Benefits(active=True, copay=30.0),
        phase="extracting",
    )


def _patched_client(
    parse_returns: object, response_id: str = "msg_test"
) -> tuple[AnthropicRepClient, AsyncMock]:
    """Build an AnthropicRepClient whose AsyncAnthropic surface is mocked.

    Returns `(client, parse_mock)` so tests can assert on call kwargs without
    reaching into private attributes. `parse_returns` is the value the SDK's
    `parsed_output` field will carry."""
    fake_response = MagicMock()
    fake_response.parsed_output = parse_returns
    fake_response.usage = MagicMock(
        input_tokens=42,
        output_tokens=120,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    fake_response.stop_reason = "end_turn"
    fake_response.id = response_id

    parse_mock = AsyncMock(return_value=fake_response)
    fake_messages = MagicMock()
    fake_messages.parse = parse_mock
    fake_async = MagicMock()
    fake_async.messages = fake_messages

    client = AnthropicRepClient(client=fake_async)
    return client, parse_mock


# --- AnthropicRepClient construction ---------------------------------------


def test_anthropic_rep_client_requires_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY is not set"):
        AnthropicRepClient()


def test_anthropic_rep_client_accepts_explicit_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = AnthropicRepClient(api_key="sk-explicit")
    assert client._client is not None  # pyright: ignore[reportPrivateUsage]


def test_anthropic_rep_client_reads_env_var_when_no_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
    client = AnthropicRepClient()
    assert client._client is not None  # pyright: ignore[reportPrivateUsage]


# --- complete_structured: happy path + call shape --------------------------


async def test_complete_structured_returns_parsed_output(rep_output: RepTurnOutput):
    client, _ = _patched_client(rep_output)
    result = await client.complete_structured(
        system="You are Morgan...",
        history=[{"role": "user", "content": "Hello, this is Sam."}],
        schema=RepTurnOutput,
    )
    assert result is rep_output


async def test_complete_structured_calls_parse_with_haiku_model(rep_output: RepTurnOutput):
    client, parse_mock = _patched_client(rep_output)
    await client.complete_structured(
        system="x",
        history=[{"role": "user", "content": "y"}],
        schema=RepTurnOutput,
    )
    kwargs = parse_mock.call_args.kwargs
    assert kwargs["model"] == REP_MODEL == "claude-haiku-4-5"


async def test_complete_structured_marks_system_for_caching(rep_output: RepTurnOutput):
    """System prompt must carry `cache_control: ephemeral` so prompt-cache
    activates once the persona grows past Haiku's 4096-token minimum."""
    client, parse_mock = _patched_client(rep_output)
    await client.complete_structured(
        system="Persona text",
        history=[{"role": "user", "content": "x"}],
        schema=RepTurnOutput,
    )
    kwargs = parse_mock.call_args.kwargs
    system_blocks = kwargs["system"]
    assert isinstance(system_blocks, list)
    assert system_blocks[0]["type"] == "text"
    assert system_blocks[0]["text"] == "Persona text"
    assert system_blocks[0]["cache_control"] == {"type": "ephemeral"}


async def test_complete_structured_passes_history_through(rep_output: RepTurnOutput):
    history = [
        {"role": "user", "content": "Hi, this is Sam."},
        {"role": "assistant", "content": "Hi Sam, calling for member M123456..."},
        {"role": "user", "content": "Coverage is active. The copay is thirty dollars."},
    ]
    client, parse_mock = _patched_client(rep_output)
    await client.complete_structured(system="x", history=history, schema=RepTurnOutput)
    kwargs = parse_mock.call_args.kwargs
    assert kwargs["messages"] == history


async def test_complete_structured_binds_output_format_to_schema(rep_output: RepTurnOutput):
    client, parse_mock = _patched_client(rep_output)
    await client.complete_structured(
        system="x",
        history=[{"role": "user", "content": "y"}],
        schema=RepTurnOutput,
    )
    kwargs = parse_mock.call_args.kwargs
    assert kwargs["output_format"] is RepTurnOutput


# --- complete_structured: refusal / parse miss -----------------------------


async def test_complete_structured_raises_when_parsed_output_is_none():
    """Anthropic returns parsed_output=None on refusal or schema mismatch.
    Surface explicitly so callers can fall back rather than crash with
    AttributeError downstream. Error message must include the response id so
    the failure is correlatable in Langfuse / dashboards."""
    client, _ = _patched_client(parse_returns=None, response_id="msg_refused_xyz")
    with pytest.raises(RuntimeError, match=r"no parsed_output.*response_id=msg_refused_xyz"):
        await client.complete_structured(
            system="x", history=[{"role": "user", "content": "y"}], schema=RepTurnOutput
        )


async def test_complete_structured_extracts_usage_from_response(rep_output: RepTurnOutput):
    """Locks the `_anthropic_usage` getattr-chain — if Anthropic renames any of
    `input_tokens` / `output_tokens` / `cache_creation_input_tokens` /
    `cache_read_input_tokens`, the helper silently returns Nones and Langfuse
    traces lose the data. Assert the shape end-to-end via Langfuse's
    `enrich_current_generation` call."""
    from unittest.mock import patch

    with patch("agent.llm_client.enrich_current_generation") as enrich_mock:
        client, _ = _patched_client(rep_output)
        await client.complete_structured(
            system="x", history=[{"role": "user", "content": "y"}], schema=RepTurnOutput
        )
    enrich_mock.assert_called_once()
    usage = enrich_mock.call_args.kwargs["usage"]
    assert usage == {
        "input": 42,
        "output": 120,
        "cache_creation": 0,
        "cache_read": 0,
    }


async def test_complete_structured_propagates_cancellation(rep_output: RepTurnOutput):
    """A CancelledError raised on the in-flight SDK call must propagate, not
    leave the session in a half-mutated state. The IVR/rep turn handlers in
    M5'/D depend on this for barge-in cancellation to actually abort the LLM."""
    import asyncio

    async def _slow_parse(*_args: object, **_kwargs: object):
        await asyncio.sleep(10.0)
        return MagicMock()

    client, parse_mock = _patched_client(rep_output)
    parse_mock.side_effect = _slow_parse

    task = asyncio.create_task(
        client.complete_structured(
            system="x", history=[{"role": "user", "content": "y"}], schema=RepTurnOutput
        )
    )
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# --- FakeAnthropicRepClient ------------------------------------------------


async def test_fake_returns_queued_responses_in_order(rep_output: RepTurnOutput):
    second = RepTurnOutput(
        reply="That's everything I needed. Thanks, have a great day.",
        extracted=Benefits(),
        phase="complete",
    )
    fake = FakeAnthropicRepClient(responses=[rep_output, second])
    r1 = await fake.complete_structured(system="x", history=[], schema=RepTurnOutput)
    r2 = await fake.complete_structured(system="x", history=[], schema=RepTurnOutput)
    assert r1 is rep_output
    assert r2 is second
    assert len(fake.calls) == 2


async def test_fake_records_calls_for_assertion(rep_output: RepTurnOutput):
    fake = FakeAnthropicRepClient(responses=[rep_output])
    await fake.complete_structured(
        system="Persona text",
        history=[{"role": "user", "content": "Hi"}],
        schema=RepTurnOutput,
    )
    system_arg, history_arg = fake.calls[0]
    assert system_arg == "Persona text"
    assert history_arg == [{"role": "user", "content": "Hi"}]


async def test_fake_raises_assertion_when_queue_empty():
    fake = FakeAnthropicRepClient()
    with pytest.raises(AssertionError, match="no responses queued"):
        await fake.complete_structured(system="x", history=[], schema=RepTurnOutput)


async def test_fake_propagates_configured_exception():
    fake = FakeAnthropicRepClient(exception=RuntimeError("simulated rate limit"))
    with pytest.raises(RuntimeError, match="simulated rate limit"):
        await fake.complete_structured(system="x", history=[], schema=RepTurnOutput)


async def test_fake_rejects_schema_mismatch(rep_output: RepTurnOutput):
    """Catches a queued type that doesn't match the requested schema —
    surfaces test-side mistakes loudly instead of silently mismatching."""
    fake = FakeAnthropicRepClient(responses=[rep_output])
    with pytest.raises(AssertionError, match="asked for Benefits"):
        await fake.complete_structured(system="x", history=[], schema=Benefits)


async def test_fake_slow_mode_delays_response(rep_output: RepTurnOutput):
    """Used by M5'/D1 to simulate a slow LLM for barge-in cancellation tests."""
    import asyncio
    import time

    fake = FakeAnthropicRepClient(responses=[rep_output], slow_mode_seconds=0.05)
    t0 = time.perf_counter()
    await fake.complete_structured(system="x", history=[], schema=RepTurnOutput)
    elapsed = time.perf_counter() - t0
    assert elapsed >= 0.04  # allow a hair of timing slack

    # Cancellation: a CancelledError raised during slow_mode propagates cleanly.
    fake_slow = FakeAnthropicRepClient(responses=[rep_output], slow_mode_seconds=10.0)
    task = asyncio.create_task(
        fake_slow.complete_structured(system="x", history=[], schema=RepTurnOutput)
    )
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
