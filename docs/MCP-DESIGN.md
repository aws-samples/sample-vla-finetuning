# vla-ft MCP — self-serve VLA fine-tuning for requesting CC sessions

**Status**: **v1 built & live-verified** (2026-06-20) — 7 tools in [`../mcp/`](../mcp/),
`test_mcp.py` 43/43, verified against a live RUNNING job and the real GR00T checkpoint gate.
See [`../mcp/README.md`](../mcp/README.md) for the as-built surface. This doc is the
*design rationale*; the sections below describe the design as decided, and the
"Open questions — RESOLVED" block at the end records how v1 shipped. Decided 2026-06-20
with the user.
**Goal (user's words)**: a research/task CC session must be able to fine-tune a VLA **without the
user acting as a manual relay** — "여기 데이터 있으니 저기 checkpoint 확인해봐라" 중간 개입 없이,
의뢰 세션이 직접 submit·조회·checkpoint 확인. So the MCP surface is defined **from the requesting
session's point of view**, not the platform's.

## Why this exists (the pain it removes)

Today a FT request is a hand-written order doc (`.temp/copy-and-paste/vla-ft-order-for-*.md`) that
the user shuttles between sessions, then a human reads CloudWatch to answer "is it actually learning?",
then hand-writes a reply with the S3 prefix. Three concrete failures this caused:
- **Relay friction**: every submit + status + handoff went through the user.
- **"RUNNING ≠ learning"**: diagnosing the RL 5.5h idle burn took 10+ SSM calls.
- **Bad export shipped silently**: the GR00T G1 ckpt had model `action_horizon=40` ↔ processor `50`
  internal mismatch that only surfaced at the *consumer's* rollout step 0. A checkpoint-validation
  gate at export would have caught it. **This is the motivating example for tool (4) below.**

## The requester's mental model (drives the tool set)

A requesting session knows four things and wants four answers:

| It has | It wants |
|---|---|
| a dataset (S3 LeRobot) + a model (pi05/groot/…) + intent (IL/RL) | "start the FT, pick the backend for me" |
| a job id | "is it really training? loss? step? or stalled?" |
| a finished job | "give me a checkpoint I can load — and is it internally consistent?" |
| a past run | "what did I train before, with what config?" |

## Proposed MCP tools (v1)

All thin wrappers over the **existing verified core** — they MUST NOT fork the decision logic or
launchers:
- `submit_finetune` → reuses `vla_ft_decide.decide()` + the unchanged launchers (or the Step
  Functions orchestrator once deployed). Inputs: `dataset_s3`, `model`, `intent` (il|rl),
  optional overrides (`steps`, `backend`, `lora`, `action_horizon`, `embodiment_tag`, `instance_type`).
  Returns: `job_id`, resolved plan (backend/instance/cost estimate), and the **output S3 prefix it will
  write to** — so the requester needs no follow-up question.
- `get_job_status` → **enriched, not raw Batch state**. Returns `{batch_status, elapsed_s, gpu_util,
  latest_ckpt_step, latest_loss, last_log_ts, liveness_ok}`. One call answers "RUNNING ≠ learning."
  (Designed in the 2026-06-18 session, never built.)
- `list_my_jobs` / `get_job` → history with the resolved config (model, dataset, steps, horizon, ft_mode)
  so a requester can see what it ran.
- `describe_checkpoint` → **the horizon-accident gate.** Given a finished job's output prefix, returns
  `{model_action_horizon, processor_delta_indices_len_per_embodiment, adapter_only?, base_model,
  consistency: OK|MISMATCH, files}`. MUST flag model↔processor horizon mismatch (and adapter-only
  vs merged) at **export-read time**, so a consumer never discovers it at rollout step 0.
- `register_checkpoint` / `list_checkpoints` → a lightweight registry (which ckpt came from which
  dataset+config, validated or flagged-bad). Backing store TBD (DynamoDB or S3 manifest).

## Architecture decision (Q1 — CC's call, leaning (b) per user)

The platform grew bottom-up (CLI → orchestrator → 3 engines). The user wants it **done right even if
that means re-laying the foundation**. Recommended target: **MCP is the first-class entry; the
deterministic orchestration underneath is CDK + Step Functions** (the user explicitly said "CDK가 Step
Functions로 모든 것을 orchestrate 해야 할 수도"). So:
- MCP server (stdio, Python — reuses `vla_ft_decide` by direct import) is the front door for sessions.
- Behind it, the **Step Functions orchestrator** (already coded in `lib/orchestrator/`, currently
  deploy-gated) becomes the deployed execution engine: classify → profile → decide → submit → notify,
  with checkpoint-validation as a post-train state.
- The existing Pattern A/B/C stacks + container engines are **reused as the orchestrator's targets**
  (not rewritten). "(b) 재설계"는 *진입점·orchestration의 재배치*이지 검증된 엔진/런처 폐기가 아님
  — that would re-trigger the M1 hand-assembly failures.

This is a **migration**, not a rewrite: keep every verified artifact, change what sits in front.

## Repo (Q3) = inside `pai-training-platform`

New `mcp/` dir in the submodule (NOT a new GitLab repo). Keeps the MCP server in one place with
`vla_ft_decide.py` (its single source of truth for backend/instance decisions) and the CDK that
provisions the registry. No cross-repo dependency management.

## Scope notes
- The legacy M1 shell engine was **fully absorbed** into `containers/vla-ft/`.
- Build order (Q4, user): **unblock first, design later.** GR00T ckpt fix → OpenArm full FT (user
  gate) → MCP foundation (this doc → tools → Step Functions deploy → registry). MCP is the durable
  payoff but must not delay the two blocked consumers.

## Open questions — RESOLVED (v1 built 2026-06-20, see `../mcp/`)
- **Registry backing store → single S3 JSON manifest** (`s3://<artifacts>/registry/checkpoints.json`),
  not DynamoDB. Tiny query volume + bucket already exists → DynamoDB power unused. Idempotent upsert-by-id for re-validation.
- **MCP transport → stdio**, per-session, like the other bdsa MCPs.
- **`submit_finetune` does NOT require Step Functions deployed.** v1 runs the orchestrator's
  own `plan()` + `submit()` **in-process**, injecting the wiring contract from live CFN
  outputs (not CDK cross-stack imports). This is the migration path: v2 deploys the same
  code behind Step Functions, env comes from CDK, zero logic change. Launchers byte-identical.
- v1 status: 7 tools built + gated (test_mcp 43/43) + **live-verified** against the running
  job (RUNNING≠learning) and the real GR00T ckpt (40/50 MISMATCH gate). See `../mcp/README.md`.
