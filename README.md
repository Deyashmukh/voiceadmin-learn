# voiceadmin-learn

Hybrid voice agent (LangGraph state machine + Pipecat audio pipeline + Twilio
telephony) that automates healthcare eligibility verification calls. A
learning project, not a production system.

The plan is in `docs/plan.md`. Execution rules are in `CLAUDE.md`.

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
make mock-payer    # run the mock payer webhook (FastAPI)
make test          # unit tests (offline, zero network)
make lint          # ruff + pyright
make format        # auto-fix ruff issues
```

## Layout

```
agent/                 # the voice agent
  main.py              # Pipecat pipeline entrypoint
  graph.py             # LangGraph StateGraph + handlers + Protocols
  graph_runner.py      # async task that owns the graph + in/out queues
  classifier.py        # rule-based IVR keyword classifier
  llm_client.py        # Groq-backed LLMClient implementation
  processors/
    state_processor.py # thin Pipecat FrameProcessor adapter
  telephony/
    dialer.py          # outbound dial + ALLOWED_DESTINATIONS allowlist
    dtmf.py            # Twilio sendDigits helper
  prompts/             # versioned prompts
  logging_config.py    # structlog setup
  config.py            # typed settings loaded from .env
  schemas.py           # PatientInfo, Benefits, CallState
mock_payer/            # FastAPI + TwiML webhook
tests/
  unit/                # offline, zero network
  integration/         # mock payer + local Pipecat, no real telephony
```

## Safety

The dialer refuses to place a call unless the destination is in
`ALLOWED_DESTINATIONS` (comma-separated E.164 numbers, loaded from `.env`).
Never disable this check.
