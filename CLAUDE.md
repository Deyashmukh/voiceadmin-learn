# VoiceAdmin Learn — Project Instructions

This is a learning project: a hybrid voice agent (two-mode `CallSession` + Pipecat audio pipeline + Twilio telephony) that automates healthcare eligibility verification calls. IVR navigation runs an LLM-with-tools loop; rep conversation runs a structured-output LLM (`RepTurnOutput`). The goal is to internalize how production voice agents are architected — including the development discipline (typing, tests, CI, error handling, observability) that production-grade work demands.

We're not deploying to real users. **But the code, tests, and process should pass a senior production-readiness review.** Dev-quality bar is production-grade; deployment-grade infrastructure (IaC, secrets management, alerting, multi-region, persistence) stays out of scope.

Full plan: `docs/plan.md`. Pre-pivot architecture (LangGraph state machine + regex IVR classifier) lives in PR #4 as a learning artifact.

## Execution rules

These rules apply to every milestone. Violating them is a defect — if you notice a rule was skipped, stop and correct it before moving on.

### 1. Git worktrees for isolated work

Independently executable milestones and sub-tasks run in their own git worktree on their own branch. This keeps streams of work from stepping on each other and makes parallelization safe.

- Use the `superpowers:using-git-worktrees` skill to create worktrees. Do not create them ad hoc.
- Name worktree branches after the milestone or sub-task (e.g., `arch-pivot-plan`, `arch-pivot-tools`, `arch-pivot-call-session`).
- Merge worktrees back to `main` only after simplify + verify have both passed for that branch.

### 2. Parallelize aggressively

Whenever two or more tasks have no shared state and no sequential dependency, dispatch them in parallel via subagents.

- Use the `superpowers:dispatching-parallel-agents` skill as the default pattern for independent work.
- Sequential execution is the exception, not the rule. If work looks sequential, check twice whether it actually is.
- When dispatching parallel agents, pair this with worktrees so each agent edits its own isolated copy of the repo.

### 3. Simplify before verifying

At the end of every milestone (or sub-task), run the `superpowers:simplify` skill over the changed code. Review for reuse, quality, and efficiency; fix any issues found.

- Simplify happens BEFORE verification, not after.
- Simplify is not optional and not "only if the code looks complex." Run it every time.
- Record any non-trivial simplifications in the commit message.

### 4. Verify before claiming done

After simplify, run the `superpowers:verification-before-completion` skill. No milestone is complete — and no commit is made — until verification has produced evidence.

- Evidence means test output, lint output, type-check output, and any quantitative checks the plan specifies (e.g., M2's cancel-to-silence < 150ms assertion, M4's slow-LLM barge-in test).
- "I think it works" and "the code looks right" are not verification.
- If verification fails, fix the underlying issue. Do not bypass, stub out, or comment out failing checks.

### 5. Commit only after simplify and verify both pass

One commit per completed milestone or logical sub-task.

- Commit messages describe the milestone completed and summarize the evidence gathered during verification.
- Never commit partial or broken work. If a milestone cannot be completed, document why in `NOTES.md` and leave the branch uncommitted.
- Never use `--no-verify`, `--no-gpg-sign`, or amend published commits unless the user explicitly asks for it.

## Tooling baseline (established in M1)

- **Dependency manager:** `uv` with pinned exact versions. `uv.lock` is committed.
- **Lint + format:** `ruff` (both).
- **Type check:** `pyright` in basic mode (not strict).
- **Tests:** `pytest` + `pytest-asyncio`. Unit tests run offline with zero network calls.
- **Pre-commit:** `pre-commit` hook runs `ruff` + `pyright` before every commit.
- **Logging:** `structlog` with `call_sid` and `turn_index` bound as contextvars from M2 onward.
- **Secrets:** `.env` is gitignored. `.env.example` is checked in with placeholders and comments marking public vs secret keys.
- **Outbound dial allowlist:** every `calls.create()` passes through a check against the `ALLOWED_DESTINATIONS` env var. Non-negotiable.

## Things to cut as YAGNI (deployment-only concerns)

These matter only when real users call the agent. We're not deploying.

- Docker for the agent itself (Langfuse uses Docker Compose, but the agent runs with `uv run`).
- OpenTelemetry / Prometheus / custom production metrics (Langfuse covers LLM traces).
- `tenacity` or any retry framework (hand-write the one or two retries that improve correctness).
- Multiple env profiles beyond a single `.env`.
- Separate fallback nodes for `CLARIFICATION`, `WAIT`, `RETRY` — one fallback node, split only if M7' forces it.
- Health checks, readiness probes, IaC, secrets managers, alerting, multi-region, persistence layer — all deployment infrastructure.

## Things we DO require for production-grade dev quality

These matter even without real users — the difference between a learning project that hides bugs and one that catches them. Tracked as M8' milestones in `docs/plan.md`.

- **Pyright strict mode**, per-file with explicit `# pyright: strict` and explicit `# type: ignore[X]` for the few real Pipecat snags. Basic mode hides bugs at boundaries.
- **GitHub Actions CI** running `ruff + pyright + pytest` on every PR. Pre-commit is bypassable; CI is the gate.
- **Test coverage thresholds** via `pytest-cov` (95% on `agent/call_session.py`, 90% on `agent/tools.py`, lower on entrypoints).
- **Hypothesis property-based tests** for the tool dispatcher's combinatorial validation matrix.
- **Error taxonomy** (`AgentError` base + specific subclasses). No string-matching on `RuntimeError` messages; callers catch by type.
- **Barge-in latency regression test** — `cancel-to-silence < 150ms` (M2's quantitative gate) must keep holding through future changes.
- **`pip-audit` in CI** — catches CVE'd transitive deps.
- **Determinism in async tests** — fake clocks for timing-sensitive tests; no flaky polling.

## Architectural non-negotiables

- **`CallSession` runs alongside the Pipecat pipeline, never inside a FrameProcessor's `process_frame()`.** Embedding the LLM call in the frame path blocks the audio loop and destroys barge-in latency. The `CallSession` task-alongside pattern is the only acceptable composition.
- **Interrupts are real `asyncio.Task.cancel()` calls, not flag checks.** Setting `session.interrupted = True` does not cancel an in-flight LLM call. Handlers must be cancellation-safe.
- **Bounded queues.** `in_queue` uses drop-oldest on full (stale transcripts are worthless). `out_queue` blocks on full (TTS backpressure is desired).
- **One `CallSession` per call**, spawned on Pipecat transport-connect, stopped on transport-disconnect. Never process-global.
- **Determinism is engineered at the tool-dispatch boundary, not assumed from the LLM.** Every IVR tool call has an arg validator; failed validation returns a tool-error message back into history so the LLM can re-pick. No exceptions.
- **Mode is one-way: `ivr → rep`.** Triggered by the IVR LLM's `transfer_to_rep()` tool call. No reverse path; if a rep puts the agent on hold and an IVR comes back, the agent stays in rep mode (acceptable for learning, flagged for production).

## Things to cut as YAGNI (post-pivot additions)

- LangGraph + LangChain (state-machine framework retired; the new shape is a turn loop with no graph).
- Regex-based IVR classifier (per-payer rules don't scale; LLM-with-tools handles all payers with one prompt).
- Mock payer (user roleplays IVR + rep when dialing own cell).
- Second Twilio number (trial tier handles own-cell dialing).
