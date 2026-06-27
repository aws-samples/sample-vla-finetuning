#!/usr/bin/env python3
"""
vla-ft MCP server — self-serve VLA fine-tuning for requesting CC sessions.

A research/task session can submit a fine-tune, ask "is it really learning?", read back a
checkpoint (with the export-consistency gate that the GR00T horizon accident would have
tripped), and keep a registry of what it trained — all WITHOUT the user hand-relaying order
docs, reading CloudWatch by hand, or pasting S3 prefixes between sessions. See MCP-DESIGN.md.

Design rules honored here:
  - **No fork of the verified core.** submit runs orchestrator_plan.plan() +
    orchestrator_submit.submit() in-process (the Phase-4 code, which mirrors the launchers);
    decisions come from vla_ft_decide. This MCP is a thin front door.
  - **stdio transport** (per-session, like the other bdsa MCPs) — registered in .mcp.json.
  - **Enriched status, not raw Batch state** — get_job_status answers "RUNNING != learning"
    in one call.

Tools (the requester's four questions + a registry):
  submit_finetune · get_job_status · list_my_jobs · get_job ·
  describe_checkpoint · register_checkpoint · list_checkpoints
"""

from __future__ import annotations

import dataclasses

from mcp.server.fastmcp import FastMCP

import vla_aws as aws
import vla_status as status
import vla_checkpoint as ckpt
import vla_registry as reg

mcp = FastMCP("vla-ft")


def _d(obj) -> dict:
    """dataclass → dict (for clean JSON tool output)."""
    return dataclasses.asdict(obj) if dataclasses.is_dataclass(obj) else obj


def _hp_from_job(sess, env: dict) -> dict | None:
    """Resolve a job's hyperparameters from its container env.

    Two wire formats coexist: legacy jobs carried each knob as an SM_HP_* override env
    var; current jobs carry only a VLA_FT_HP_S3 pointer to a hyperparameters.json (so the
    long LoRA regex stays out of Batch's 8192B override). Try the env first (zero I/O),
    then fetch the pointer JSON and undo SageMaker's one json.dumps layer (train.py's
    _coerce contract), so both shapes return the same plain {name: value} dict. Returns
    {} (never None) for jobs that carry neither — RL/GR00T — so callers can .get() safely."""
    hp = {k[len("SM_HP_"):].lower(): v for k, v in env.items() if k.startswith("SM_HP_")}
    if hp:
        return hp
    hp_s3 = env.get("VLA_FT_HP_S3")
    if not hp_s3:
        return {}
    raw = aws.get_json(sess, hp_s3) or {}
    out = {}
    for k, v in raw.items():
        if isinstance(v, str):
            try:
                v = __import__("json").loads(v)  # '"pi05"' -> 'pi05', '200' -> 200
            except ValueError:
                pass
        out[k.lower()] = v if isinstance(v, str) else (
            "true" if v is True else "false" if v is False else str(v))
    return out


# ── 1. submit_finetune ───────────────────────────────────────────────────────────────
@mcp.tool()
def submit_finetune(
    dataset_s3: str | None = None,
    model: str = "pi05",
    intent: str = "il",
    task: str | None = None,
    steps: int | None = None,
    backend: str | None = None,
    instance_type: str | None = None,
    lora: bool = False,
    lora_r: int | None = None,
    lora_alpha: int | None = None,
    lora_target_modules: str | None = None,
    full_vlm: bool = False,
    pretrained_path: str | None = None,
    action_horizon: int | None = None,
    embodiment_tag: str | None = None,
    base_model: str | None = None,
    save_steps: int | None = None,
    global_batch: int | None = None,
    learning_rate: float | None = None,
    liveness_deadline_s: int | None = None,
    num_envs: int | None = None,
    max_iterations: int | None = None,
    num_gpus: int = 1,
    num_nodes: int = 1,
    select_best: bool = False,
    val_episodes: int | None = None,
    early_stop_patience: int | None = None,
    budget_usd: float | None = None,
    spot: bool = True,
    timeout_hours: float | None = None,
    job_name: str | None = None,
    region: str | None = None,
    dry_run: bool = True,
) -> dict:
    """Start a VLA fine-tune and let the platform pick the backend + engine. Returns the
    resolved plan (engine/backend/instance/cost) + the job id + the EXACT output S3 prefix.

    Three engines, ONE front door — the platform classifies the intent and routes it:
      - **lerobot (IL)** — pass `dataset_s3` (+ `model`, default pi05). The 10 LeRobot
        policies; LoRA/expert-only/select-best supported. → Batch (A) or SageMaker (B).
      - **GR00T N1.7 (IL)** — pass `dataset_s3` + `embodiment_tag` (e.g. UNITREE_G1). The
        NVIDIA 3B Cosmos VLA; output is a full merged 3B HF checkpoint. → Batch (GR00T A).
      - **RL** — pass `task` (an Isaac Lab task id), `intent='rl'`. → Batch (RL A).

    The requester gives what it has and gets back everything it needs to track and later
    load the result, with no follow-up question to the user.

    RESUME: pass job_name to reuse a prior run's name. The job writes checkpoints to an
    EFS dir derived from job_name, so resubmitting with the SAME job_name resumes from the
    last checkpoint (after a timeout / Spot reclaim / host-termination) instead of starting
    from step 0. Keep every other arg identical to the original submit. Omit job_name → a
    fresh timestamped name (a brand-new run).

    TIMEOUT: pass timeout_hours to set THIS job's wall-clock ceiling (Batch SIGKILLs the
    attempt at the limit). It overrides the Job Definition's default per job with NO
    redeploy — size it to the run (e.g. ~19 h for a single-L40S 20000-step full-VLM
    fine-tune; a short probe needs ~2 h). Omit it → the deployed JobDefinition default.

    SPOT: spot=True (default) runs on the cheap Spot queue (reclaim restarts the attempt,
    resuming from the EFS checkpoint); spot=False routes to the On-Demand queue for a long
    sanctioned run that should not eat Spot-reclaim restarts. Both queues are deployed once
    (idle CEs scale to 0 vCPU, so the waiting queue costs nothing) — switching is per-job,
    no redeploy.

    MULTI-NODE (Pattern C): pass num_nodes>1 to request a multi-node FSDP2 fine-tune (the
    model sharded across nodes over EFA — for a replica too big for one GPU, or to go
    faster). This routes to Pattern C (HyperPod Slurm), which is a DEPLOY-GATED reference,
    NOT auto-submittable from here: submit returns status='recommend_only' with the plan +
    the operator steps (cdk deploy -c enableHyperPod=true [-c hyperPodFsx=true], then sbatch
    hyperpod_fsdp_launch.sh on the cluster head). num_nodes=1 (default) uses the runnable
    single-node Pattern A/B.

    SAFETY: dry_run defaults to True — it returns the resolved plan + cost estimate WITHOUT
    launching (and without needing creds for the plan). Call again with dry_run=False to
    actually submit (incurs GPU cost). For GR00T UNITREE_G1, the platform auto-sets
    action_horizon=50 (the full-body embodiment needs it — omitting it ships the 40/50
    mismatch); pass action_horizon explicitly to override for other embodiments.
    """
    event = {
        "intent": intent,
        "dataset": dataset_s3,
        "model": model,
        "task": task,
        "num_gpus": num_gpus,
        "num_nodes": num_nodes,
        "full_vlm": full_vlm,
        "lora": lora,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_target_modules": lora_target_modules,
        "pretrained_path": pretrained_path,
        "select_best": select_best,
        "val_episodes": val_episodes,
        "early_stop_patience": early_stop_patience,
        "budget_usd": budget_usd,
        "backend": backend,
        "instance_type": instance_type,
        "spot": spot,
    }
    if job_name is not None:
        event["job_name"] = job_name
    if timeout_hours is not None:
        event["timeout_hours"] = timeout_hours
    if steps is not None:
        event["steps"] = steps
    if max_iterations is not None:
        event["max_iterations"] = max_iterations
    if num_envs is not None:
        event["num_envs"] = num_envs
    if action_horizon is not None:
        event["action_horizon"] = action_horizon
    if embodiment_tag is not None:
        event["embodiment_tag"] = embodiment_tag
    # GR00T-specific knobs (ignored by the lerobot/RL planners — they read what they need).
    if base_model is not None:
        event["base_model"] = base_model
    if save_steps is not None:
        event["save_steps"] = save_steps
    if global_batch is not None:
        event["global_batch"] = global_batch
    if learning_rate is not None:
        event["learning_rate"] = learning_rate
    if liveness_deadline_s is not None:
        event["liveness_deadline_s"] = liveness_deadline_s
    # Drop Nones so orchestrator_plan's .get() defaults apply.
    event = {k: v for k, v in event.items() if v is not None}

    plan_out = aws.plan_intent(event)

    if dry_run:
        return {
            "submitted": False,
            "dry_run": True,
            "axis": plan_out["axis"],
            "pattern": plan_out["pattern"],
            "backend": plan_out["backend"],
            "runnable": plan_out["runnable"],
            "plan_text": plan_out["plan_text"],
            "decision": plan_out["decision"],
            "note": "dry_run=True — nothing launched. Re-call with dry_run=False to submit "
                    "(GPU cost). GR00T G1: pass action_horizon=50.",
        }

    result = aws.submit_intent(event, region=region)
    result["submitted"] = result.get("status") == "submitted"
    return result


# ── 2. get_job_status ────────────────────────────────────────────────────────────────
@mcp.tool()
def get_job_status(job_id: str, region: str | None = None, log_lines: int = 120) -> dict:
    """Enriched status: 'is it REALLY training?' in one call (RUNNING != learning).

    Combines Batch state + the recent CloudWatch log tail + checkpoint presence + GPU
    saturation into a verdict: {batch_status, elapsed_s, learning, liveness_ok, latest_step,
    latest_loss, latest_epoch (GR00T), latest_reward (RL), last_log_ts, output_s3,
    gpu_util_mean, gpu_mem_pct, gpu_temp_max, gpu_throttle, gpu_idle, summary, notes}.
    `liveness_ok=False` while RUNNING is the danger signal the platform kept diagnosing by
    hand (the 5.5h idle burn).

    GPU saturation (the DCGM-pattern signal, from train.py's [gpu-telemetry] log line)
    refines the verdict in BOTH directions when present: RUNNING + busy GPU + no step line
    yet = warming up (cold load), NOT a stall; RUNNING + idle GPU = idle-burn even if a
    stale step line lingers in the tail. gpu_* fields are None for jobs whose image predates
    GPU telemetry (the verdict then degrades to the log-line-only logic — back-compatible).
    The default log_lines=120 ensures a 30 s-cadence telemetry line is in the tail."""
    sess = aws.session(region)
    job = aws.describe_job(sess, job_id)
    if not job:
        return {"error": f"no Batch job found for id {job_id!r} in region "
                         f"{sess.region_name}."}
    facts = aws.job_facts(job)
    engine = status.classify_engine(facts["job_name"])
    log_tail, last_ts = aws.tail_log(sess, facts["log_stream"], limit=log_lines)
    progress = status.parse_progress(log_tail, engine)

    # Cheap checkpoint corroboration: does the EFS/S3 output prefix already have a ckpt?
    has_ckpt = False
    if facts.get("output_s3"):
        try:
            keys = aws.list_prefix(sess, facts["output_s3"], max_keys=50)
            has_ckpt = any("checkpoint" in k or "pretrained_model" in k
                           or k.endswith(".safetensors") for k in keys)
        except Exception:  # noqa: BLE001
            has_ckpt = False

    verdict = status.build_verdict(
        job_name=facts["job_name"], job_id=facts["job_id"],
        batch_status=facts["status"], elapsed_s=facts["elapsed_s"],
        progress=progress, last_log_ts=last_ts, output_s3=facts.get("output_s3"),
        liveness_deadline_s=facts.get("liveness_deadline_s"), has_checkpoint=has_ckpt,
    )
    out = _d(verdict)
    if facts.get("status_reason"):
        out["status_reason"] = facts["status_reason"]
    return out


# ── 3. list_my_jobs ──────────────────────────────────────────────────────────────────
@mcp.tool()
def list_my_jobs(intent: str = "il", region: str | None = None, max_results: int = 25) -> dict:
    """Recent jobs on the platform's Batch queue (newest first), with id/name/status/created.
    intent='il' uses the IL (lerobot) Pattern A queue; 'gr00t' the GR00T queue; 'rl' the RL
    queue. Pair with get_job_status for the enriched 'is it learning?' read on any one."""
    sess = aws.session(region)
    stack = {"rl": aws.STACK_RL_A, "gr00t": aws.STACK_GROOT_A,
             "groot": aws.STACK_GROOT_A}.get(intent, aws.STACK_IL_A)
    try:
        outs = aws.stack_outputs(sess, stack)
    except RuntimeError as e:
        return {"error": str(e)}
    # IL Pattern A owns a Spot + On-Demand queue pair; a job lands on whichever the
    # submit picked, so list across BOTH. RL/GR00T are single-queue. De-dupe in case an
    # older stack lacks the OD output (it falls back to the same Spot ARN).
    queues = [outs.get("JobQueueArn")]
    if intent not in ("rl", "gr00t", "groot") and outs.get("JobQueueArnOnDemand"):
        queues.append(outs.get("JobQueueArnOnDemand"))
    queues = [q for i, q in enumerate(queues) if q and q not in queues[:i]]
    if not queues:
        return {"error": f"{stack} has no JobQueueArn output — is it deployed?"}
    jobs = [j for q in queues for j in aws.list_jobs(sess, q, max_results=max_results)]
    jobs.sort(key=lambda j: j.get("createdAt", 0), reverse=True)
    jobs = jobs[:max_results]
    return {"queues": queues, "count": len(jobs), "jobs": [
        {"job_id": j.get("jobId"), "job_name": j.get("jobName"),
         "status": j.get("status"), "created_at": j.get("createdAt"),
         "status_reason": j.get("statusReason")}
        for j in jobs
    ]}


# ── 4. get_job ───────────────────────────────────────────────────────────────────────
@mcp.tool()
def get_job(job_id: str, region: str | None = None) -> dict:
    """The resolved config of one job (model, dataset, steps, ft-mode, horizon, output
    prefix) read from its Batch container env — 'what did I run, with what config?'."""
    sess = aws.session(region)
    job = aws.describe_job(sess, job_id)
    if not job:
        return {"error": f"no Batch job found for id {job_id!r}."}
    container = job.get("container", {}) or {}
    env = {e["name"]: e.get("value") for e in container.get("environment", []) or []}
    hp = _hp_from_job(sess, env)
    return {
        "job_id": job.get("jobId"),
        "job_name": job.get("jobName"),
        "status": job.get("status"),
        "engine": status.classify_engine(job.get("jobName", "")),
        "dataset_s3": env.get("VLA_FT_DATASET_S3") or env.get("GROOT_DATASET_S3"),
        "output_s3": (env.get("VLA_FT_OUTPUT_S3") or env.get("RL_OUTPUT_S3")
                      or env.get("GROOT_OUTPUT_S3")),
        "task": env.get("RL_TASK"),
        "embodiment_tag": env.get("GROOT_EMBODIMENT_TAG"),
        "hyperparameters": hp or None,
        "rl_env": {k: v for k, v in env.items() if k.startswith("RL_")} or None,
        "groot_env": {k: v for k, v in env.items()
                      if k.startswith("GROOT_") and k != "GROOT_OUTPUT_S3"} or None,
    }


# ── 5. describe_checkpoint ───────────────────────────────────────────────────────────
@mcp.tool()
def describe_checkpoint(output_s3: str, region: str | None = None,
                        embodiment_tag: str | None = None) -> dict:
    """The export-consistency gate. Given a finished job's output prefix, report kind,
    adapter-only vs merged, base model, and (GR00T) the model<->processor action_horizon
    CONSISTENCY — flagging the 40/50 mismatch that crashes a rollout at step 0, at
    export-read time instead of at the consumer's step 0.

    Returns {kind, consistency: OK|MISMATCH|UNKNOWN, adapter_only, base_model,
    model_action_horizon, processor_action_horizon, files, evidence, loadable_hint, notes}."""
    sess = aws.session(region)
    try:
        keys = aws.list_prefix(sess, output_s3, max_keys=2000)
    except Exception as e:  # noqa: BLE001
        return {"error": f"cannot list {output_s3!r}: {e}"}
    if not keys:
        return {"error": f"no objects under {output_s3!r} — wrong prefix or job not finished?"}

    kind = ckpt.detect_kind(keys)

    if kind == "gr00t":
        model_cfg = aws.get_json_at(sess, output_s3, "config.json")
        final_model = aws.get_json_at(sess, output_s3, "experiment_cfg/final_model_config.json")
        proc_cfg = aws.get_json_at(sess, output_s3, "processor/processor_config.json")
        final_proc = aws.get_json_at(sess, output_s3,
                                     "experiment_cfg/final_processor_config.json")
        report = ckpt.inspect_gr00t(
            output_s3, keys, model_config=model_cfg, processor_config=proc_cfg,
            final_model_config=final_model, final_processor_config=final_proc,
            embodiment_tag=embodiment_tag)
        return _d(report)

    if kind == "lerobot":
        # The adapter/policy configs live under pretrained_model/ (model.tar.gz is staged
        # extracted in EFS; on S3 the bootstrap uploads the dir contents).
        adapter_cfg = aws.get_json_at(sess, output_s3, "pretrained_model/adapter_config.json") \
            or aws.get_json_at(sess, output_s3, "adapter_config.json")
        train_cfg = aws.get_json_at(sess, output_s3, "pretrained_model/train_config.json") \
            or aws.get_json_at(sess, output_s3, "train_config.json")
        report = ckpt.inspect_lerobot(output_s3, keys, adapter_config=adapter_cfg,
                                      train_config=train_cfg)
        return _d(report)

    return {"prefix": output_s3, "kind": "unknown", "consistency": "UNKNOWN",
            "files": sorted({k.rsplit('/', 1)[-1] for k in keys})[:40],
            "notes": ["Could not classify this checkpoint as GR00T or lerobot from its files."]}


# ── 6. register_checkpoint ───────────────────────────────────────────────────────────
@mcp.tool()
def register_checkpoint(
    job_id: str,
    region: str | None = None,
    consistency: str | None = None,
    notes: list[str] | None = None,
) -> dict:
    """Record a finished checkpoint in the registry (S3 manifest under the artifacts bucket),
    enriched from the job's own config + a describe_checkpoint pass. Idempotent by job name:
    re-registering after a re-validation updates the entry in place.

    Pass `consistency` to override the auto-detected verdict (e.g. 'OK' after a hot-fix)."""
    import time
    sess = aws.session(region)
    job = aws.describe_job(sess, job_id)
    if not job:
        return {"error": f"no Batch job found for id {job_id!r}."}
    facts = aws.job_facts(job)
    container = job.get("container", {}) or {}
    env = {e["name"]: e.get("value") for e in container.get("environment", []) or []}
    hp = _hp_from_job(sess, env)
    engine = status.classify_engine(facts["job_name"])
    output_s3 = facts.get("output_s3") or ""

    # describe_checkpoint pass to capture consistency + adapter/horizon, best-effort.
    desc = describe_checkpoint(output_s3, region=region) if output_s3 else {}
    auto_consistency = desc.get("consistency", "UNKNOWN")

    # Resolve the artifacts bucket (registry lives beside the outputs).
    try:
        a_out = aws.stack_outputs(sess, aws.STACK_IL_A)
        artifacts = a_out.get("OutputS3Hint") or output_s3
    except RuntimeError:
        artifacts = output_s3
    # registry root = the artifacts bucket root (strip the gr00t-n17/vla-ft sub-prefix).
    bucket, _ = aws._split_s3(artifacts)  # noqa: SLF001 — internal helper, same package
    registry_uri = f"s3://{bucket}"

    manifest = aws.get_json(sess, registry_uri, reg.REGISTRY_KEY) or reg.empty_manifest()
    entry = reg.CheckpointEntry(
        id=facts["job_name"],
        output_s3=output_s3,
        engine=engine,
        model=(hp.get("policy") or env.get("RL_TASK")
               or (f"groot/{env.get('GROOT_EMBODIMENT_TAG')}" if engine == "gr00t" else None)
               or "pi05"),
        dataset_s3=env.get("VLA_FT_DATASET_S3") or env.get("GROOT_DATASET_S3"),
        intent="rl" if engine == "rl" else "il",  # GR00T is an IL engine
        ft_mode=("lora" if hp.get("lora") == "true"
                 else "expert_only" if hp.get("train_expert_only") == "true"
                 else None if engine in ("rl", "gr00t")  # GR00T = full merged (not a lerobot mode)
                 else "full_ft"),
        steps=aws._int_or_none(hp.get("steps") or env.get("GROOT_MAX_STEPS")),  # noqa: SLF001
        action_horizon=desc.get("model_action_horizon")
        or desc.get("processor_action_horizon"),
        adapter_only=desc.get("adapter_only"),
        base_model=desc.get("base_model"),
        consistency=consistency or auto_consistency,
        registered_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        notes=notes or [],
    )
    new_manifest = reg.upsert(manifest, entry)
    written = aws.put_json(sess, registry_uri, reg.REGISTRY_KEY, new_manifest)
    return {"registered": True, "manifest": written, "entry": dataclasses.asdict(entry)}


# ── 7. list_checkpoints ──────────────────────────────────────────────────────────────
@mcp.tool()
def list_checkpoints(
    engine: str | None = None,
    model: str | None = None,
    intent: str | None = None,
    consistency: str | None = None,
    region: str | None = None,
) -> dict:
    """List registered checkpoints (newest first), optionally filtered by engine/model/intent/
    consistency. Reads the S3 manifest — 'what have I trained, which are validated?'."""
    sess = aws.session(region)
    try:
        a_out = aws.stack_outputs(sess, aws.STACK_IL_A)
        bucket, _ = aws._split_s3(a_out.get("OutputS3Hint", ""))  # noqa: SLF001
    except RuntimeError as e:
        return {"error": str(e)}
    if not bucket:
        return {"error": "could not resolve the artifacts bucket from IL Pattern A outputs."}
    registry_uri = f"s3://{bucket}"
    manifest = aws.get_json(sess, registry_uri, reg.REGISTRY_KEY) or reg.empty_manifest()
    rows = reg.query(manifest, engine=engine, model=model, intent=intent,
                     consistency=consistency)
    return {"registry": f"{registry_uri}/{reg.REGISTRY_KEY}", "count": len(rows),
            "checkpoints": rows}


if __name__ == "__main__":
    mcp.run()
