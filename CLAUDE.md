# VoiceAdmin Learn — Project Instructions

This is a learning project: a hybrid voice agent (LangGraph state machine + Pipecat audio pipeline + Twilio telephony) that automates healthcare eligibility verification calls. The goal is to internalize how production voice agents are architected, not to ship to production.

Full plan: `docs/plan.md` (copied from `~/.claude/plans/transient-tinkering-pebble.md`).

## Execution rules

These rules apply to every milestone. Violating them is a defect — if you notice a rule was skipped, stop and correct it before moving on.

### 1. Git worktrees for isolated work

Independently executable milestones and sub-tasks run in their own git worktree on their own branch. This keeps streams of work from stepping on each other and makes parallelization safe.

- Use the `superpowers:using-git-worktrees` skill to create worktrees. Do not create them ad hoc.
- Name worktree branches after the milestone or sub-task (e.g., `m2-pipecat-loop`, `m3-graph`, `m5-mock-payer`).
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

## Things to cut as YAGNI

- GitHub Actions CI (local pre-commit is enough).
- Pyright strict mode (will tar-pit on Pipecat internals).
- Docker for the agent itself (Langfuse uses Docker Compose, but the agent runs with `uv run`).
- OpenTelemetry / Prometheus / custom metrics.
- `tenacity` or any retry framework (hand-write the one retry needed).
- Multiple env profiles beyond a single `.env`.
- Separate fallback nodes for `CLARIFICATION`, `WAIT`, `RETRY` — one fallback node, split only if M7 forces it.

## Architectural non-negotiables

- **LangGraph runs alongside the Pipecat pipeline, never inside a FrameProcessor's `process_frame()`.** Embedding `graph.ainvoke` in the frame path blocks the audio loop and destroys barge-in latency. The `GraphRunner` pattern in the plan is the only acceptable composition.
- **Interrupts are real `asyncio.Task.cancel()` calls, not flag checks.** Setting `state["interrupted"] = True` does not cancel an in-flight LLM call. Handlers must be cancellation-safe.
- **Bounded queues.** `in_queue` uses drop-oldest on full (stale transcripts are worthless). `out_queue` blocks on full (TTS backpressure is desired).
- **No checkpointer.** `MemorySaver` is cargo culting for this project. Re-add only if resume-after-crash becomes a real requirement.
- **One `GraphRunner` per call**, spawned on Pipecat transport-connect, stopped on transport-disconnect. Never process-global.
