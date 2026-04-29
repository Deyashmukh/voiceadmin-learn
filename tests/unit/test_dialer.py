"""Unit tests for the outbound dial allowlist.

These tests run offline with no network calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent.errors import ConfigurationError
from agent.telephony.dialer import (
    DestinationNotAllowedError,
    check_destination,
    dial,
)


def test_allowlist_rejects_unlisted_destination() -> None:
    allowlist = frozenset({"+15551112222"})
    with pytest.raises(DestinationNotAllowedError):
        check_destination("+19998887777", allowlist)


def test_allowlist_accepts_listed_destination() -> None:
    allowlist = frozenset({"+15551112222"})
    check_destination("+15551112222", allowlist)


def test_dial_blocks_unlisted_destination_before_twilio_call() -> None:
    client = MagicMock()
    allowlist = frozenset({"+15551112222"})

    with pytest.raises(DestinationNotAllowedError):
        dial(
            to="+19998887777",
            from_="+15550000000",
            twilio_client=client,
            url="https://example.test/twiml",
            allowlist=allowlist,
        )

    client.calls.create.assert_not_called()


def test_dial_calls_twilio_for_listed_destination() -> None:
    client = MagicMock()
    client.calls.create.return_value = MagicMock(sid="CA123")
    allowlist = frozenset({"+15551112222"})

    result = dial(
        to="+15551112222",
        from_="+15550000000",
        twilio_client=client,
        url="https://example.test/twiml",
        allowlist=allowlist,
    )

    assert result.call_sid == "CA123"
    assert result.to == "+15551112222"
    client.calls.create.assert_called_once_with(
        to="+15551112222", from_="+15550000000", url="https://example.test/twiml"
    )


def test_empty_allowlist_rejects_everything() -> None:
    with pytest.raises(DestinationNotAllowedError):
        check_destination("+15551112222", frozenset())


def test_dial_without_twilio_client_raises_configuration_error() -> None:
    """`dial()` requires `twilio_client`; absence is a wiring bug, not a
    runtime failure. Allowlist check passes first (so this exercises the
    None-client branch, not the fence)."""
    allowlist = frozenset({"+15551112222"})
    with pytest.raises(ConfigurationError) as exc_info:
        dial(
            to="+15551112222",
            from_="+15550000000",
            twilio_client=None,
            url="https://example.test/twiml",
            allowlist=allowlist,
        )
    assert exc_info.value.setting == "twilio_client"
