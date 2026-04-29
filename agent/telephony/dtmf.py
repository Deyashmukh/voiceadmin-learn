"""SUPERSEDED — kept as an artifact for the real-DTMF wiring.

This module's `<Play digits>`-via-TwiML-update approach is NOT the active
DTMF path. Live testing showed that posting a TwiML update to the active
call sid replaces the running TwiML and ends the Media Stream — i.e., it
hangs up the call. See `agent/actuator.py` for the current TEMP path
(speak the digits via `out_queue`) and the planned real fix (generate
DTMF dual-tone PCM and emit it as `OutputAudioRawFrame` over the WSS).

`send_digits` is no longer called from `agent/`; only a unit test exercises
it. The module stays in-tree as a reminder of the failed approach so a
future implementer doesn't re-derive it.
"""

from __future__ import annotations

import asyncio
from typing import Any

# digits is constrained to ^[0-9*#]+$ at the SendDTMFArgs schema; no XML-special
# chars can reach this string, so format-string interpolation is safe.
_DTMF_TWIML = '<Response><Play digits="{digits}"/></Response>'


# `Any` rather than `twilio.rest.Client` because the Twilio SDK's stubs are
# poor — chained `.calls(sid).update(...)` doesn't type-check usefully.
async def send_digits(twilio_client: Any, call_sid: str, digits: str) -> None:
    """Inject DTMF tones into a live Twilio call. Sync Twilio SDK call is run
    on a worker thread so the asyncio event loop isn't blocked."""
    twiml = _DTMF_TWIML.format(digits=digits)
    await asyncio.to_thread(twilio_client.calls(call_sid).update, twiml=twiml)
