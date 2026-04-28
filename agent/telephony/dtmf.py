"""Mid-stream DTMF injection over Twilio.

Once a call is connected (Media Streams open), `<Play digits>` injected via a
TwiML update on the live call sid sends DTMF tones to the remote side. The
allowlist check in `dialer.py` doesn't apply here — the call already exists,
so this isn't a new outbound dial.
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
