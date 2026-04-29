"""Place a test outbound call to a roleplay payer.

Run: `uv run python scripts/dial_test.py --url https://<ngrok>.ngrok-free.app`

Calls the first entry in ALLOWED_DESTINATIONS using TWILIO_AGENT_NUMBER as
the caller ID. Twilio will hit `<url>/twiml` to fetch the Media Streams
TwiML once the callee picks up.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from twilio.rest import Client as TwilioRestClient

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", required=True, help="Public https base URL (e.g. ngrok). /twiml is appended."
    )
    parser.add_argument(
        "--to",
        default=None,
        help="E.164 destination. Defaults to first ALLOWED_DESTINATIONS entry.",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))
    from agent.telephony.dialer import dial

    allowlist_raw = os.environ.get("ALLOWED_DESTINATIONS", "")
    to = args.to or next((p.strip() for p in allowlist_raw.split(",") if p.strip()), "")
    if not to:
        print("No --to and ALLOWED_DESTINATIONS empty.", file=sys.stderr)
        return 1

    client = TwilioRestClient(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])
    twiml_url = args.url.rstrip("/") + "/twiml"
    result = dial(
        to=to,
        from_=os.environ["TWILIO_AGENT_NUMBER"],
        twilio_client=client,
        url=twiml_url,
    )
    print(f"Placed call sid={result.call_sid} to={result.to} via={twiml_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
