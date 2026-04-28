# voiceadmin-learn

Hybrid voice agent (two-mode `CallSession` + Pipecat audio pipeline + Twilio
telephony) that automates healthcare eligibility verification calls. A
learning project, not a production system.

The plan is in `docs/plan.md`. Execution rules are in `CLAUDE.md`. Pre-pivot
LangGraph + regex-classifier architecture lives in PR #4 as a learning
artifact.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
# install deps into .venv/
make install

# copy env template and fill in keys
cp .env.example .env

# install pre-commit hook (runs ruff + pyright before every commit)
uv run pre-commit install
```

## Running

```bash
make agent         # run the voice agent
make test          # unit tests (offline, zero network)
make lint          # ruff + pyright
make format        # auto-fix ruff issues
```

## Layout

```
agent/                   # the voice agent
  main.py                # FastAPI + Pipecat pipeline + WSS entrypoint
  call_session.py        # per-turn loop; mode-aware IVR/rep dispatch
  tools.py               # IVR tool registry + dispatcher + arg validators
  actuator.py            # executes side-effect intents (DTMF, TTS, hangup)
  llm_client.py          # Anthropic rep-mode client (messages.parse)
  observability.py       # Langfuse @observe wiring + session tagging
  schemas.py             # CallSession, Turn, RepTurnOutput, tool-arg models
  processors/
    state_processor.py   # Pipecat FrameProcessor adapter to CallSession
  telephony/
    dialer.py            # outbound dial + ALLOWED_DESTINATIONS allowlist
    dtmf.py              # Twilio mid-stream <Play digits> wrapper
  prompts/
    rep_turn.v1.txt      # rep-mode persona prompt
  logging_config.py      # structlog setup + call_sid contextvar binding
tests/
  unit/                  # offline, zero network
```

## Safety

The dialer refuses to place a call unless the destination is in
`ALLOWED_DESTINATIONS` (comma-separated E.164 numbers, loaded from `.env`).
Never disable this check.
