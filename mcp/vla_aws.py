#!/usr/bin/env python3
"""
vla-ft MCP — the AWS I/O layer (boto3). All side effects live here.

The pure cores (vla_status / vla_checkpoint / vla_registry) decide; this module fetches
and persists. It also holds the one structural decision that makes v1 a *migration* and
not a rewrite:

  ★ submit reuses the Phase-4 orchestrator code IN-PROCESS.
  `orchestrator_plan.plan()` + `orchestrator_submit.submit()` already mirror the verified
  launchers byte-for-byte (built for the Step Functions Lambda, deploy-gated, never run
  live). orchestrator_submit reads its wiring (queue/jobdef/code/output ARNs + HF SSM) from
  ENVIRONMENT VARIABLES — at synth time CDK injects them as cross-stack imports. Here we
  inject the SAME contract from live CloudFormation outputs and call submit() directly.
  → v1 runs the orchestrator's own code; v2 just deploys it behind Step Functions and the
  env comes from CDK instead of from us. Zero logic change, no launcher fork.


Stack output keys (verified against vla_ft_cli.py's resolver):
  IL Pattern A : PaiTrainingPlatform-IL-PatternA       → JobQueueArn JobDefinitionArn CodeS3Hint OutputS3Hint
  RL Pattern A : PaiTrainingPlatform-RL-PatternA       → JobQueueArn JobDefinitionArn OutputS3Hint
  GR00T Ptn A  : PaiTrainingPlatform-IL-GrootPatternA  → JobQueueArn JobDefinitionArn OutputS3Hint
  Pattern B    : PaiTrainingPlatform-IL-PatternB       → ExecutionRoleArn ImageUriHint OutputS3Hint
"""

from __future__ import annotations

import io
import json
import os
import sys
import time

import boto3

# Make the verified core importable: orchestrator_plan / orchestrator_submit / vla_ft_decide
# all live in containers/vla-ft/ (one source of truth, no copy that can drift).
_HERE = os.path.dirname(os.path.abspath(__file__))
_CORE = os.path.normpath(os.path.join(_HERE, "..", "containers", "vla-ft"))
if _CORE not in sys.path:
    sys.path.insert(0, _CORE)

DEFAULT_REGION = "us-west-2"
DEFAULT_HF_TOKEN_SSM = "/pai/hf-token"
DEFAULT_HF_TOKEN_SSM_REGION = "us-east-1"

STACK_IL_A = "PaiTrainingPlatform-IL-PatternA"
STACK_IL_B = "PaiTrainingPlatform-IL-PatternB"
STACK_RL_A = "PaiTrainingPlatform-RL-PatternA"
STACK_GROOT_A = "PaiTrainingPlatform-IL-GrootPatternA"

LOG_GROUP = "/aws/batch/job"


def session(region: str | None = None) -> boto3.Session:
    return boto3.Session(region_name=region or DEFAULT_REGION)


# ── CloudFormation outputs ───────────────────────────────────────────────────────────
def stack_outputs(sess: boto3.Session, stack: str) -> dict:
    """{OutputKey: OutputValue} for a stack, or raise a clear error if it isn't deployed."""
    cfn = sess.client("cloudformation")
    try:
        resp = cfn.describe_stacks(StackName=stack)
    except Exception as e:  # noqa: BLE001 — surface the real reason to the requester
        raise RuntimeError(f"cannot read stack {stack!r}: {e}") from e
    outs = resp["Stacks"][0].get("Outputs", []) or []
    return {o["OutputKey"]: o["OutputValue"] for o in outs}


def resolve_wiring(sess: boto3.Session, *, hf_token_ssm: str | None,
                   hf_token_ssm_region: str | None) -> dict:
    """Read the deployed stacks' outputs and return the env contract orchestrator_submit
    expects. Missing stacks are tolerated (only the chosen pattern's vars are required at
    submit time) — we set what we can and let submit() raise if its own pattern is unwired."""
    env: dict[str, str] = {}
    region = sess.region_name or DEFAULT_REGION
    env["REGION"] = region
    if hf_token_ssm:
        env["HF_TOKEN_SSM"] = hf_token_ssm
    if hf_token_ssm_region:
        env["HF_TOKEN_SSM_REGION"] = hf_token_ssm_region

    def safe(stack: str) -> dict:
        try:
            return stack_outputs(sess, stack)
        except RuntimeError:
            return {}

    a = safe(STACK_IL_A)
    if a:
        env["IL_A_JOB_QUEUE"] = a.get("JobQueueArn", "")
        env["IL_A_JOB_DEFINITION"] = a.get("JobDefinitionArn", "")
        env["IL_A_CODE_S3"] = a.get("CodeS3Hint", "")
        env["IL_A_OUTPUT_S3"] = a.get("OutputS3Hint", "")
    r = safe(STACK_RL_A)
    if r:
        env["RL_A_JOB_QUEUE"] = r.get("JobQueueArn", "")
        env["RL_A_JOB_DEFINITION"] = r.get("JobDefinitionArn", "")
        env["RL_A_OUTPUT_S3"] = r.get("OutputS3Hint", "")
    g = safe(STACK_GROOT_A)
    if g:
        env["GROOT_A_JOB_QUEUE"] = g.get("JobQueueArn", "")
        env["GROOT_A_JOB_DEFINITION"] = g.get("JobDefinitionArn", "")
        env["GROOT_A_OUTPUT_S3"] = g.get("OutputS3Hint", "")
    b = safe(STACK_IL_B)
    if b:
        env["B_EXECUTION_ROLE"] = b.get("ExecutionRoleArn", "")
        env["B_IMAGE_URI"] = b.get("ImageUriHint", "")
        env["B_OUTPUT_S3"] = b.get("OutputS3Hint", "")
    return env


# ── submit (reuse orchestrator_plan + orchestrator_submit in-process) ────────────────
def plan_intent(event: dict) -> dict:
    """Pure plan via the orchestrator's own classify→profile→decide (no creds needed)."""
    import orchestrator_plan  # from containers/vla-ft/ (on sys.path)
    return orchestrator_plan.plan(event)

def submit_intent(event: dict, *, region: str | None = None,
                  hf_token_ssm: str | None = None,
                  hf_token_ssm_region: str | None = None) -> dict:
    """Plan + submit by running the orchestrator code in-process with env wiring from CFN.

    Returns the orchestrator_submit envelope: {status, backend, axis, job_name, job_id,
    output_s3} for Batch, or {status: 'handoff', ...} for Pattern B (SageMaker)."""
    import orchestrator_plan
    import orchestrator_submit

    sess = session(region)
    plan_out = orchestrator_plan.plan(event)
    if not plan_out.get("runnable"):
        # Pattern C (recommend-only) — return the plan, do not attempt to submit.
        return {"status": "recommend_only", **plan_out}

    wiring = resolve_wiring(
        sess,
        hf_token_ssm=hf_token_ssm or DEFAULT_HF_TOKEN_SSM,
        hf_token_ssm_region=hf_token_ssm_region or DEFAULT_HF_TOKEN_SSM_REGION,
    )
    # Inject the env contract orchestrator_submit reads, run it, then restore os.environ.
    saved = {k: os.environ.get(k) for k in wiring}
    try:
        os.environ.update(wiring)
        result = orchestrator_submit.submit(plan_out)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    # Thread the plan text through so the requester sees backend/instance/cost in one call.
    result["plan_text"] = plan_out.get("plan_text")
    result["plan"] = plan_out.get("decision")
    return result


# ── Batch describe / list ────────────────────────────────────────────────────────────
def describe_job(sess: boto3.Session, job_id: str) -> dict | None:
    """Raw Batch describe-jobs for one job id (None if not found)."""
    resp = sess.client("batch").describe_jobs(jobs=[job_id])
    jobs = resp.get("jobs", [])
    return jobs[0] if jobs else None


def job_facts(job: dict) -> dict:
    """Extract the status-enrichment inputs from a Batch job object.

    elapsed_s is measured from startedAt (RUNNING) or startedAt→stoppedAt (terminal).
    log_stream is container.logStreamName (the CloudWatch stream to tail)."""
    status = job.get("status", "")
    started = job.get("startedAt")        # epoch ms
    stopped = job.get("stoppedAt")        # epoch ms
    now_ms = int(time.time() * 1000)
    if started:
        end = stopped or now_ms
        elapsed_s = max(0, int((end - started) / 1000))
    else:
        elapsed_s = None
    container = job.get("container", {}) or {}
    env = {e["name"]: e.get("value") for e in container.get("environment", []) or []}
    # The three engines name their output prefix differently (VLA_FT_/RL_/GROOT_).
    out_base = (env.get("VLA_FT_OUTPUT_S3") or env.get("RL_OUTPUT_S3")
                or env.get("GROOT_OUTPUT_S3") or "")
    return {
        "job_name": job.get("jobName", ""),
        "job_id": job.get("jobId", ""),
        "status": status,
        "elapsed_s": elapsed_s,
        "log_stream": container.get("logStreamName"),
        "status_reason": job.get("statusReason"),
        "output_s3": out_base + ("/output/" if out_base else ""),
        "liveness_deadline_s": _int_or_none(env.get("RL_LIVENESS_DEADLINE_S")
                                            or env.get("VLA_FT_LIVENESS_DEADLINE_S")
                                            or env.get("GROOT_LIVENESS_DEADLINE_S")),
    }


def list_jobs(sess: boto3.Session, queue: str, *, statuses: list[str] | None = None,
              max_results: int = 50) -> list[dict]:
    """List Batch jobs in a queue across the given statuses (default: all lifecycle states)."""
    statuses = statuses or ["SUBMITTED", "PENDING", "RUNNABLE", "STARTING",
                            "RUNNING", "SUCCEEDED", "FAILED"]
    client = sess.client("batch")
    out: list[dict] = []
    for st in statuses:
        token = None
        while True:
            kw = {"jobQueue": queue, "jobStatus": st, "maxResults": 100}
            if token:
                kw["nextToken"] = token
            resp = client.list_jobs(**kw)
            out += resp.get("jobSummaryList", [])
            token = resp.get("nextToken")
            if not token or len(out) >= max_results:
                break
    out.sort(key=lambda j: j.get("createdAt", 0), reverse=True)
    return out[:max_results]


# ── CloudWatch logs tail ─────────────────────────────────────────────────────────────
def tail_log(sess: boto3.Session, log_stream: str, *, limit: int = 60,
             log_group: str = LOG_GROUP) -> tuple[str, int | None]:
    """Return (joined newest log messages, last_event_ts_ms). Empty string if no stream."""
    if not log_stream:
        return "", None
    logs = sess.client("logs")
    try:
        resp = logs.get_log_events(
            logGroupName=log_group, logStreamName=log_stream,
            limit=limit, startFromHead=False,
        )
    except Exception:  # noqa: BLE001 — stream may not exist yet (pre-RUNNING)
        return "", None
    events = resp.get("events", [])
    text = "\n".join(e.get("message", "") for e in events)
    last_ts = events[-1].get("timestamp") if events else None
    return text, last_ts


# ── S3 (checkpoint inspect + registry) ───────────────────────────────────────────────
def _split_s3(uri: str) -> tuple[str, str]:
    rest = uri[len("s3://"):] if uri.startswith("s3://") else uri
    bucket, _, key = rest.partition("/")
    return bucket, key


def list_prefix(sess: boto3.Session, prefix_uri: str, *, max_keys: int = 1000) -> list[str]:
    """List object keys (relative to the prefix) under an s3:// prefix."""
    bucket, key = _split_s3(prefix_uri)
    if key and not key.endswith("/"):
        key += "/"
    s3 = sess.client("s3")
    keys: list[str] = []
    token = None
    while True:
        kw = {"Bucket": bucket, "Prefix": key, "MaxKeys": 1000}
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        for o in resp.get("Contents", []):
            rel = o["Key"][len(key):] if o["Key"].startswith(key) else o["Key"]
            if rel:
                keys.append(rel)
        token = resp.get("NextContinuationToken")
        if not token or len(keys) >= max_keys:
            break
    return keys[:max_keys]


def get_json(sess: boto3.Session, bucket_or_uri: str, key: str | None = None) -> dict | None:
    """Fetch+parse a JSON object. Call get_json(sess, 's3://b/k') or get_json(sess,'b','k').
    Returns None if the object is missing (so optional configs degrade gracefully)."""
    if key is None:
        bucket, key = _split_s3(bucket_or_uri)
    else:
        bucket = bucket_or_uri
    try:
        body = sess.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception:  # noqa: BLE001 — missing/forbidden object → treat as absent
        return None
    try:
        return json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return None


def get_json_at(sess: boto3.Session, prefix_uri: str, rel_key: str) -> dict | None:
    """JSON at <prefix>/<rel_key> (prefix-relative convenience for checkpoint configs)."""
    bucket, base = _split_s3(prefix_uri)
    if base and not base.endswith("/"):
        base += "/"
    return get_json(sess, bucket, base + rel_key)


def put_json(sess: boto3.Session, prefix_uri: str, rel_key: str, obj: dict) -> str:
    """Write a JSON object under a prefix; return the full s3:// uri written."""
    bucket, base = _split_s3(prefix_uri)
    if base and not base.endswith("/"):
        base += "/"
    full = base + rel_key
    sess.client("s3").put_object(
        Bucket=bucket, Key=full,
        Body=io.BytesIO(json.dumps(obj, indent=2).encode("utf-8")).read(),
        ContentType="application/json",
    )
    return f"s3://{bucket}/{full}"


def _int_or_none(v) -> int | None:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None
