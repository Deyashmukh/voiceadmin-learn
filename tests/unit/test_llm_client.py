"""Unit tests for M5'/C: Anthropic rep-LLM client + the FakeAnthropicRepClient
seam. Zero network — the SDK call sites are mocked."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.errors import ConfigurationError, LLMRefusalError
from agent.llm_client import (
    IVR_MODEL,
    REP_MODEL,
    AnthropicRepClient,
    GroqToolCallingClient,
    _history_to_groq_messages,  # pyright: ignore[reportPrivateUsage]
)
from agent.schemas import Benefits, IVRTurnResponse, RepTurnOutput, ToolCall, Turn

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
    with pytest.raises(ConfigurationError) as exc_info:
        AnthropicRepClient()
    assert exc_info.value.setting == "ANTHROPIC_API_KEY"


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
    client, parse_mock = _patched_client(parse_returns=None, response_id="msg_refused_xyz")
    # Set a realistic refusal stop_reason so the assertion locks the
    # provider-value passthrough rather than the fixture default.
    parse_mock.return_value.stop_reason = "refusal"
    with pytest.raises(LLMRefusalError) as exc_info:
        await client.complete_structured(
            system="x", history=[{"role": "user", "content": "y"}], schema=RepTurnOutput
        )
    assert exc_info.value.response_id == "msg_refused_xyz"
    assert exc_info.value.stop_reason == "refusal"


async def test_complete_structured_falls_back_to_unknown_on_unrecognized_stop_reason():
    """If the SDK ever returns a stop_reason outside the known Literal set
    (new Anthropic value, garbled response), the client narrows to
    `unknown` so dashboards don't bucket on typos."""
    client, parse_mock = _patched_client(parse_returns=None, response_id="msg_x")
    parse_mock.return_value.stop_reason = "pause_turn"  # hypothetical future reason
    with pytest.raises(LLMRefusalError) as exc_info:
        await client.complete_structured(
            system="x", history=[{"role": "user", "content": "y"}], schema=RepTurnOutput
        )
    assert exc_info.value.stop_reason == "unknown"


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


# --- GroqToolCallingClient tests -------------------------------------------


def _patched_groq(
    message_content: str = "", tool_calls: list[dict[str, object]] | None = None
) -> tuple[GroqToolCallingClient, AsyncMock]:
    """Build a GroqToolCallingClient whose AsyncGroq surface returns a mock
    response shaped like Groq's chat-completion output."""
    fake_message = MagicMock()
    fake_message.content = message_content
    if tool_calls is not None:
        fake_tcs: list[MagicMock] = []
        for tc in tool_calls:
            m = MagicMock()
            m.id = tc["id"]
            m.function = MagicMock()
            m.function.name = tc["name"]
            m.function.arguments = tc["arguments"]
            fake_tcs.append(m)
        fake_message.tool_calls = fake_tcs
    else:
        fake_message.tool_calls = None
    fake_choice = MagicMock(message=fake_message)
    fake_response = MagicMock(choices=[fake_choice])
    fake_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    create_mock = AsyncMock(return_value=fake_response)
    fake_completions = MagicMock(create=create_mock)
    fake_chat = MagicMock(completions=fake_completions)
    fake_async = MagicMock(chat=fake_chat)
    client = GroqToolCallingClient(client=fake_async)
    return client, create_mock


def test_groq_client_requires_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ConfigurationError) as exc_info:
        GroqToolCallingClient()
    assert exc_info.value.setting == "GROQ_API_KEY"


def test_groq_client_accepts_explicit_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    client = GroqToolCallingClient(api_key="gsk-explicit")
    assert client._client is not None  # pyright: ignore[reportPrivateUsage]


async def test_groq_client_returns_parsed_tool_calls():
    """Happy path: Groq returns tool_calls; client maps them into IVRTurnResponse."""
    client, _ = _patched_groq(
        tool_calls=[
            {"id": "call_1", "name": "send_dtmf", "arguments": '{"digits": "1"}'},
        ],
    )
    result = await client.complete_with_tools(
        system="x", history=[Turn(role="user", content="hi")], tools=[], temperature=0.1
    )
    assert isinstance(result, IVRTurnResponse)
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "send_dtmf"
    assert result.tool_calls[0].args == {"digits": "1"}
    assert result.tool_calls[0].id == "call_1"


async def test_groq_client_drops_malformed_json_args():
    """Malformed JSON args from a hallucinating model are dropped silently —
    the dispatcher would reject them downstream anyway, but filtering here
    keeps `IVRTurnResponse.tool_calls` clean for the watchdog."""
    client, _ = _patched_groq(
        tool_calls=[
            {"id": "call_1", "name": "send_dtmf", "arguments": "not valid json{"},
            {"id": "call_2", "name": "speak", "arguments": '{"text": "ok"}'},
        ],
    )
    result = await client.complete_with_tools(system="x", history=[], tools=[])
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "speak"


async def test_groq_client_drops_hallucinated_tool_names():
    """Tool names not in the `ToolName` Literal are dropped at the boundary."""
    client, _ = _patched_groq(
        tool_calls=[
            {"id": "call_1", "name": "made_up_tool", "arguments": "{}"},
            {"id": "call_2", "name": "complete_call", "arguments": '{"reason": "user_hangup"}'},
        ],
    )
    result = await client.complete_with_tools(system="x", history=[], tools=[])
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "complete_call"


async def test_groq_client_returns_text_with_no_tool_calls():
    """If the LLM only emits text (no tools), `IVRTurnResponse.tool_calls`
    is empty and `text` carries the content. The watchdog uses empty
    tool_calls as the no-progress signal."""
    client, _ = _patched_groq(message_content="thinking...", tool_calls=None)
    result = await client.complete_with_tools(system="x", history=[], tools=[])
    assert result.tool_calls == []
    assert result.text == "thinking..."


async def test_groq_client_passes_call_args_to_sdk():
    """Verify model, tool_choice, temperature, max_tokens reach the SDK."""
    client, create_mock = _patched_groq()
    await client.complete_with_tools(system="sys", history=[], tools=[{"x": 1}], temperature=0.2)
    kwargs = create_mock.call_args.kwargs
    assert kwargs["model"] == IVR_MODEL
    assert kwargs["tool_choice"] == "required"
    assert kwargs["temperature"] == 0.2
    assert kwargs["tools"] == [{"x": 1}]
    assert kwargs["max_tokens"] == 512


# --- _history_to_groq_messages tests ---------------------------------------


def test_history_to_groq_messages_user_only():
    msgs = _history_to_groq_messages("sys", [Turn(role="user", content="hi")])
    assert msgs == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]


def test_history_to_groq_messages_pairs_tool_call_with_result():
    """Consecutive tool_call + tool_result Turns become assistant.tool_calls
    + tool message with matching tool_call_id."""
    history = [
        Turn(role="user", content="hi"),
        Turn(
            role="tool_call",
            tool_call=ToolCall(name="send_dtmf", args={"digits": "1"}, id="call_1"),
        ),
        Turn(role="tool_result", content="DTMF 1 dispatched."),
    ]
    msgs = _history_to_groq_messages("sys", history)
    assert msgs[0] == {"role": "system", "content": "sys"}
    assert msgs[1] == {"role": "user", "content": "hi"}
    assert msgs[2]["role"] == "assistant"
    assert msgs[2]["tool_calls"][0]["id"] == "call_1"  # type: ignore[index, call-overload]
    assert msgs[2]["tool_calls"][0]["function"]["name"] == "send_dtmf"  # type: ignore[index, call-overload]
    assert msgs[3] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "DTMF 1 dispatched.",
    }


def test_history_to_groq_messages_synthesizes_id_when_missing():
    history = [
        Turn(role="tool_call", tool_call=ToolCall(name="speak", args={"text": "ok"})),
        Turn(role="tool_result", content="Speaking."),
    ]
    msgs = _history_to_groq_messages("sys", history)
    assert msgs[1]["tool_calls"][0]["id"] == "call_0_speak"  # type: ignore[index, call-overload]
    assert msgs[2]["tool_call_id"] == "call_0_speak"


def test_history_to_groq_messages_skips_unpaired_tool_result():
    """A tool_result with no preceding tool_call (shouldn't happen, but
    defensive) is silently skipped — Groq would 400 on it."""
    msgs = _history_to_groq_messages(
        "sys",
        [
            Turn(role="user", content="hi"),
            Turn(role="tool_result", content="orphan"),
        ],
    )
    assert msgs == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]


def test_history_to_groq_messages_skips_empty_content():
    msgs = _history_to_groq_messages(
        "sys",
        [Turn(role="user", content=""), Turn(role="assistant", content="")],
    )
    assert msgs == [{"role": "system", "content": "sys"}]


def test_history_to_groq_messages_assistant_text():
    msgs = _history_to_groq_messages(
        "sys",
        [
            Turn(role="user", content="hi"),
            Turn(role="assistant", content="hello back"),
        ],
    )
    assert msgs[2] == {"role": "assistant", "content": "hello back"}
