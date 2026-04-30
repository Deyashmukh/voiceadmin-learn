"""One-liner smoke tests for every provider credential in .env.

Run: `uv run python scripts/smoke_test.py`. Makes a tiny live call per
provider and prints OK/FAIL with a short reason. No audio or phone calls
placed.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=True)

results: dict[str, str] = {}


def smoke_twilio() -> str:
    from twilio.rest import Client

    sid = os.environ["TWILIO_ACCOUNT_SID"]
    client = Client(sid, os.environ["TWILIO_AUTH_TOKEN"])
    acct = client.api.accounts(sid).fetch()
    return f"OK — status={acct.status}, type={acct.type}"


def smoke_groq_model(model_id: str) -> str:
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    r = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": "Say 'pong' and nothing else."}],
        max_tokens=10,
    )
    content = r.choices[0].message.content or ""
    return f"OK — {content.strip()[:40]!r}"


def smoke_deepgram() -> str:
    from deepgram import DeepgramClient

    dg = DeepgramClient(api_key=os.environ["DEEPGRAM_API_KEY"])
    projects = dg.manage.v1.projects.list()
    return f"OK — {len(projects.projects)} project(s)"


def smoke_elevenlabs() -> str:
    from elevenlabs.client import ElevenLabs

    el = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
    voices = el.voices.get_all()
    return f"OK — {len(voices.voices)} voice(s)"


def smoke_anthropic() -> str:
    from anthropic import Anthropic

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    r = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=10,
        messages=[{"role": "user", "content": "Say 'pong' and nothing else."}],
    )
    text_block = next((b for b in r.content if b.type == "text"), None)
    text = (text_block.text if text_block else "").strip()
    return f"OK — {text[:40]!r}"


for name, fn in [
    ("twilio", smoke_twilio),
    ("groq/qwen3-32b", lambda: smoke_groq_model("qwen/qwen3-32b")),
    ("groq/llama-4-scout", lambda: smoke_groq_model("meta-llama/llama-4-scout-17b-16e-instruct")),
    ("deepgram", smoke_deepgram),
    ("elevenlabs", smoke_elevenlabs),
    ("anthropic/haiku-4-5", smoke_anthropic),
]:
    try:
        results[name] = fn()
    except Exception as e:
        results[name] = f"FAIL — {type(e).__name__}: {e}"

width = max(len(k) for k in results)
for k, v in results.items():
    print(f"{k:<{width}}  {v}")
