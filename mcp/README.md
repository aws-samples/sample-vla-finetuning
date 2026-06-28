# vla-ft MCP — self-serve VLA fine-tuning for requesting CC sessions

A research/task session can **submit a fine-tune, check whether it is really learning,
read back a checkpoint (with an export-consistency gate), and keep a registry of what it
trained — without the user hand-relaying order docs, reading CloudWatch by hand, or
pasting S3 prefixes between sessions.** Design rationale: [`../docs/MCP-DESIGN.md`](../docs/MCP-DESIGN.md).

## What it is (and is NOT)

This is a **thin front door** over the already-verified platform core. It does **not**
fork the decision logic or the launchers:

- `submit_finetune` runs the Phase-4 orchestrator **in-process** —
  `orchestrator_plan.plan()` + `orchestrator_submit.submit()` from
  `../containers/vla-ft/` — which themselves mirror the verified `batch_launch.py` /
  `launch.py` byte-for-byte. Backend/instance/cost decisions come from `vla_ft_decide.py`,
  the single source of truth the CLI also uses.
- The pure cores here (`vla_status`, `vla_checkpoint`, `vla_registry`) only **parse and
  decide**; every AWS side effect is isolated in `vla_aws.py`.

## Tools

All three engines route through ONE front door — `submit_finetune` classifies the intent
and picks the engine + backend:

- **lerobot (IL)** — `dataset_s3` (+ `model`, default pi05) → Batch (A) or SageMaker (B).
- **GR00T N1.7 (IL)** — `dataset_s3` + `embodiment_tag` (e.g. UNITREE_G1) → Batch (GR00T A);
  UNITREE_G1 auto-sets `action_horizon=50` (avoids the 40/50 step-0 crash).
- **RL** — `task` (an Isaac Lab task id), `intent='rl'` → Batch (RL A).

| Tool | The requester's question | Returns |
|---|---|---|
| `submit_finetune` | "start the FT, pick the engine + backend for me" | resolved plan (engine/backend/instance/**cost**) + `job_id` + the **output S3 prefix** it writes to. Routes lerobot · GR00T · RL. `dry_run=True` by default (plan only, no launch, no creds). `image_uri` pins the container to an exact image (digest) — drift guard against a `:latest` rebuild changing the image under a run (honored on the SageMaker path; advisory on Batch). |
| `get_job_status` | "is it **really** training? loss? step? or stalled?" | enriched verdict: `batch_status, elapsed_s, learning, liveness_ok, latest_step, latest_loss, latest_epoch (GR00T), latest_reward (RL), output_s3, summary`. **One call answers "RUNNING ≠ learning."** |
| `list_my_jobs` | "what's on the queue?" | recent Batch jobs (newest first) for the `il` (lerobot), `gr00t`, or `rl` queue. |
| `get_job` | "what config + which image did I run?" | resolved model/dataset/steps/ft-mode/horizon/output from the job's env, plus `image_uri` and the digest it resolves to **now** (`image_digest`) — compare across two runs of the same tag to catch `:latest` drift. |
| `describe_checkpoint` | "give me a checkpoint I can load — is it internally consistent?" | `kind, consistency (OK\|MISMATCH\|UNKNOWN), adapter_only, base_model, model_action_horizon, processor_action_horizon, files, loadable_hint`. **The GR00T 40/50 horizon gate.** |
| `register_checkpoint` / `list_checkpoints` | "remember what I trained / which are validated" | S3-manifest registry, idempotent by job name. |

## The two payoffs, verified against live AWS (2026-06-20)

1. **"RUNNING ≠ learning" in one call.** Against the live OpenArm LoRA job
   `vla-ft-pi05-20260620-122229`, `get_job_status` returned
   `RUNNING and learning (step 800, loss 0.136), liveness_ok=True` — the verdict that
   previously took 10+ SSM calls. A booted-but-idle job (no step/loss line) returns
   `liveness_ok=False` with a probable-stall note.
2. **The export-consistency gate that the GR00T accident needed.** Against the real
   checkpoint `gr00t-n17-20260619-103523/output/`, `describe_checkpoint` read the actual
   S3 config bytes and returned `consistency=MISMATCH, model_action_horizon=40,
   processor_action_horizon=50` — the exact mismatch that crashed the consumer's rollout
   at step 0 — while the hand-fixed `...-h50fix` prefix returned `consistency=OK` (50/50).
   The OpenArm LoRA smoke ckpt correctly reports `adapter_only=True,
   base_model=lerobot/pi05_base` with the "base must be reachable to serve" caveat.

## Open questions — resolved (the v1 design decisions)

`MCP-DESIGN.md` left three; this is how v1 resolved them and why.

- **Registry backing store → a single S3 JSON manifest** (`s3://<artifacts>/registry/checkpoints.json`),
  NOT DynamoDB. The artifacts bucket is already there; the query volume is a handful of
  rows (read + filter in memory). DynamoDB's query power would be unused complexity, plus
  a new table + IAM grants. Idempotent upsert-by-id makes
  re-validation (e.g. `OK` after a hot-fix) a simple in-place update.
- **Transport → stdio**, per-session, like the other bdsa MCPs. No long-running service to
  operate; the server is spawned per CC session and reads creds from the ambient AWS
  profile.
- **Does `submit_finetune` need Step Functions deployed first? → No.** v1 calls the
  orchestrator's own `plan()` + `submit()` **in-process**, injecting the wiring contract
  (queue/jobdef/code/output ARNs + HF-token SSM) from live CloudFormation outputs instead
  of from CDK cross-stack imports. This is the **migration path, not a rewrite**: v2 simply
  deploys the same code behind Step Functions and the env comes from CDK — zero logic
  change, and v1 ships without gating on a deploy. The Pattern B (SageMaker) path returns
  the exact verified `launch.py` command as a handoff (it is not forked in-process), and
  Pattern C (HyperPod) returns recommend-only, exactly as the orchestrator already does.
- **MCP-only launch (no shell access)? → check `mcp_can_submit` in the dry_run plan.** It is
  `true` only when `submit_finetune(dry_run=False)` will actually launch the job (Pattern A /
  Batch). Pattern B is `runnable` but `mcp_can_submit:false` — it returns a `launch.py`
  handoff an operator must run, so a consumer restricted to MCP cannot complete it. To force
  an MCP-launchable path, pass `backend='batch'` (routes to Pattern A). The dry_run `note`
  and `plan_text` say this explicitly so the backend can be chosen *before* submitting.

## Layout

```
mcp/
  server.py          FastMCP server — the 7 tools (thin wiring only)
  vla_status.py      pure: progress regexes + the RUNNING≠learning verdict
  vla_checkpoint.py  pure: adapter/merged classify + GR00T horizon consistency gate
  vla_registry.py    pure: S3-manifest schema (upsert/query)
  vla_aws.py         boto3 I/O + in-process reuse of orchestrator_plan/submit
  test_mcp.py        65 self-contained asserts (no pytest), real log lines + config keys
  requirements.txt   mcp + boto3 (core imported from ../containers/vla-ft/)
```

## Run / register

```bash
# dedicated venv
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
# smoke the tools
python test_mcp.py            # 65/65, pure, offline
python server.py              # stdio server (Ctrl-C to stop)
```

Register it in your MCP client's `.mcp.json` (stdio) so any session can call it.
Set `REPO_ROOT` to the absolute path of your clone of this repository:

```jsonc
"vla-ft": {
  "command": "/bin/sh",
  "args": ["-c", "${REPO_ROOT}/mcp/.venv/bin/python ${REPO_ROOT}/mcp/server.py"],
  "env": { "AWS_PROFILE": "default", "AWS_DEFAULT_REGION": "us-west-2", "FASTMCP_LOG_LEVEL": "ERROR" }
}
```

## v2 (not built — the migration's next leg)

Deploy `lib/orchestrator/` (Step Functions + the plan/submit Lambdas) and flip
`submit_intent` to start an SFN execution instead of calling `submit()` in-process. The
checkpoint-consistency gate becomes a post-train state in the state machine (so a
MISMATCH export is flagged before the consumer ever sees it). The tool surface here does
not change — only what sits behind `submit_finetune`.
