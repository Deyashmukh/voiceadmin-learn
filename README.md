# voiceadmin-learn

Hybrid voice agent (two-mode `CallSession` + Pipecat audio pipeline + Twilio
telephony) that automates healthcare eligibility verification calls. A
learning project, not a production system — but the code, tests, and
process target a senior production-readiness review.

The plan is in `docs/plan.md`. Execution rules and architectural non-
negotiables are in `CLAUDE.md`. Pre-pivot LangGraph + regex-classifier
architecture lives in PR #4 as a learning artifact.

## Architecture

```
        Twilio Media Stream (μ-law 8kHz, WSS)
                       │
                       ▼
       ┌───────────────────────────────────────────┐
       │ FastAPI + Pipecat pipeline                │
       │   transport.input()                       │
       │     → Silero VAD                          │
       │     → Deepgram STT                        │
       │     → StateMachineProcessor ◄────┐        │
       │     → ElevenLabs TTS             │        │
       │     → transport.output()         │ frames │
       └─────────────────────────────────────┬─────┘
                                             │ queues
                       ┌─────────────────────▼──────┐
                       │ CallSessionRunner          │
                       │   (one per call,           │
                       │    runs alongside pipeline)│
                       │                            │
                       │   ivr mode: tool-calling   │
                       │     LLM (Groq)             │
                       │   rep mode: structured-    │
                       │     output LLM (Anthropic) │
                       │                            │
                       │   one-way ivr → rep        │
                       │   via transfer_to_rep()    │
                       └────────────────────────────┘
```

The `CallSessionRunner` lives **alongside** the Pipecat pipeline, not
inside a `FrameProcessor.process_frame()` — that's a non-negotiable. The
processor and runner communicate via bounded queues
(`in_queue` drop-oldest, `out_queue` blocking-put for TTS backpressure).
Barge-in is real `asyncio.Task.cancel()`, not a flag check.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
# install deps into .venv/
make install

# copy env template and fill in keys
cp .env.example .env

# install pre-commit hook (runs ruff + pyright before every commit)
uv run pre-commit install

# verify provider credentials are good
uv run python scripts/smoke_test.py
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
agent/                       # the voice agent
  main.py                    # FastAPI + Pipecat pipeline + WSS entrypoint
  call_session.py            # per-turn loop; mode-aware IVR/rep dispatch;
                             #   bounded queues; benefits.jsonl on completion
  tools.py                   # IVR tool registry + dispatcher + arg validators
  actuator.py                # executes side-effect intents (DTMF, TTS, hangup)
  llm_client.py              # Anthropic rep client (messages.parse)
                             #   + Groq IVR client (tool-calling)
                             #   model pins live in this file
  observability.py           # Langfuse @observe wiring + session tagging
  errors.py                  # AgentError taxonomy (5 typed subclasses)
  schemas.py                 # CallSession, Turn, RepTurnOutput, tool-arg models
  processors/
    state_processor.py       # Pipecat FrameProcessor adapter to CallSession;
                             #   VAD-driven transcript flush + barge-in
  telephony/
    dialer.py                # outbound dial + ALLOWED_DESTINATIONS allowlist
    dtmf.py                  # SUPERSEDED <Play digits> path; kept as artifact
                             #   (real DTMF-as-audio TODO)
  prompts/
    rep_turn.v1.txt          # rep-mode persona prompt
  logging_config.py          # structlog setup + call_sid contextvar binding
scripts/                     # dev convenience tools (out of pyright scope)
  smoke_test.py              # one-call-per-provider credentials smoke test
  dial_test.py               # initiate an outbound call to your own cell
tests/
  unit/                      # offline, zero network; ruff + pyright strict
```

## Tooling baseline

- `uv` for dependency management; `uv.lock` is committed
- `ruff` for lint + format (rule set pinned in `pyproject.toml`)
- `pyright` strict per-file via `# pyright: strict` headers in every `.py`
  under `agent/` and `tests/` (new files MUST add the header)
- `pytest` + `pytest-asyncio` + `pytest-cov`; coverage floor enforced in
  `pyproject.toml` (`agent/` excl. `main.py`)
- `pre-commit` hook runs ruff + pyright before every commit (developer
  convenience; CI is the gate when M8'/B lands)
- `structlog` with `call_sid` + `turn_index` bound as contextvars

## Status

- ✅ Two-mode `CallSession` (IVR-with-tools → rep-with-structured-output)
- ✅ Pipecat audio pipeline with VAD-driven transcript flush + barge-in
- ✅ Tool dispatcher with per-tool arg validation + Hypothesis property tests
- ✅ Error taxonomy (`AgentError` + typed subclasses; no string-matching)
- ✅ Per-call `benefits.jsonl` deliverable (best-effort, async write)
- ✅ Langfuse `@observe` tracing across IVR + rep + tool dispatch
- ✅ End-to-end live-call flow verified on Twilio trial
- ✅ Cancel-to-silence latency measured at the runner-side (sub-2ms,
  quantized to the test harness's 1ms poll floor; full assertion deferred
  until a budget multiplier is picked — see
  `tests/unit/test_barge_in_latency.py`)

**Deferred / in flight:**

- M6': real DTMF tone generation as `OutputAudioRawFrame` (current path
  speaks digits aloud — TEMP, voice-roleplay testing only)
- M7': 5/5 manual phone-test runs documented end-to-end
- M8'/B: GitHub Actions CI (ruff + pyright + pytest gate)
- M8'/F (step ii): convert the latency measurement into a regression
  assertion against a chosen budget
- M8'/G, /I: `pip-audit` + secret-scanning gates (block on M8'/B)

## Process

Every commit follows the chain in order:

1. `superpowers:simplify` — review changed code for reuse / quality / efficiency
2. `superpowers:code-reviewer` (subagent) — independent review
3. `superpowers:verification-before-completion` — fresh test/lint/typecheck
4. Commit
5. Push + open PR
6. `/pr-review-toolkit:review-pr <N>` — self-review on the open PR

Independent milestones run in their own worktree on their own branch
(`.worktrees/<branch-name>`).

## Branch protection

`main` is protected via a GitHub ruleset (Standard tier):

- Force-push and deletion blocked
- Direct pushes blocked (PRs only)
- Linear history (squash or rebase merges only)
- Bypass enabled for repo admin so the owner can recover from misfires

Required status checks (`ruff`, `pyright`, `pytest`) will be added once
M8'/B lands.

## Safety

The dialer refuses to place a call unless the destination is in
`ALLOWED_DESTINATIONS` (comma-separated E.164 numbers, loaded from `.env`).
Never disable this check. The SUPERSEDED `agent/telephony/dtmf.py` path
must NOT run against a real payer — current DTMF dispatches are a TEMP
TTS stand-in (see `agent/actuator.py`).
