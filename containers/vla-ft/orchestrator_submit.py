#!/usr/bin/env python3
"""
Phase 4 orchestrator — the submit Lambda (the "submit" step).

Takes the plan from orchestrator_plan.py and dispatches the chosen backend. The hard
rule here is **faithfulness to the verified launchers** — the M1 lesson was that
hand-assembling what is already verified causes a cascade of failures. So this Lambda mirrors the launchers' submit
contract exactly, and refuses to fork the parts that can't be reproduced safely:

  - **Pattern A (Batch)** — IL (lerobot), RL, or GR00T N1.7 — is submitted DIRECTLY
    here via boto3 `submit_job`, reproducing batch_launch.py / rl_launch.py /
    gr00t_launch.py byte-for-byte on the container-override contract (SM_HP_* env + the
    VLA_FT_* / RL_* / GROOT_* contract, plus the HF-token read-and-inject). This is a
    thin, well-understood API call, so doing it in-Lambda is safe and is the
    orchestrator's native path.
  - **Pattern B (SageMaker)** is NOT submitted from boto3 here. The verified launcher
    is launch.py, which uses the SageMaker Python SDK `PyTorch` estimator (it tars
    `source_dir`, wires checkpoint_s3_uri + Managed-Spot resume, sets the
    sagemaker_program conventions). Re-implementing that in raw `create_training_job`
    would fork the verified path — exactly the trap to avoid. Instead this Lambda
    RESOLVES the full plan into the exact `launch.py` command and returns it as a
    **handoff** for an operator / CI to run. Honest and verified-lock-safe.
  - **Pattern C (HyperPod)** is code+synth only (not in bin/app.ts), so it is never
    reached here — the plan's `runnable` gate routes it to a recommendation.

Wiring (queue/jobdef/role/image/output ARNs, HF-token SSM name) is injected as
ENVIRONMENT VARIABLES at synth time from the orchestrator stack's cross-stack imports
— NOT resolved at runtime via describe_stacks. That keeps this Lambda deterministic,
fast, and free of CloudFormation read IAM. The HF token VALUE is read from SSM inside
this Lambda and injected as container env, so the secret never lands in the Step
Functions execution history (the SFN input only carries the SSM parameter NAME).

Determinism: given the same plan + the same env wiring, the submitted job spec is
identical (job names use a timestamp the caller may pin via `job_name`).
"""

from __future__ import annotations

import os
import time


# ── env wiring (injected at synth from the orchestrator stack's stack imports) ──────
def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def _split_s3(uri: str):
    rest = uri[len("s3://"):]
    bucket, _, key = rest.partition("/")
    return bucket, key


# ── HF token (read here, inject as container env — never into SFN history) ──────────
def _read_hf_token(session, ssm_name: str | None, ssm_region: str | None):
    """Read the gated-backbone HF token from SSM. Mirrors batch_launch.py / launch.py:
    the token is read on the submitting side and injected as job env, so it never
    lands in the JobDefinition, CloudWatch, or the Step Functions execution record."""
    if not ssm_name:
        return None
    ssm = session.client("ssm", region_name=ssm_region) if ssm_region else session.client("ssm")
    return ssm.get_parameter(Name=ssm_name, WithDecryption=True)["Parameter"]["Value"]


# ── IL Pattern A (AWS Batch) — direct submit, faithful to batch_launch.py ───────────
def _submit_il_batch(session, region: str, submit: dict, *, want_hf: bool) -> dict:
    import boto3  # noqa: F401  (runtime-provided; imported for clarity / local runs)

    queue = _env("IL_A_JOB_QUEUE")
    jobdef = _env("IL_A_JOB_DEFINITION")
    code_s3 = _env("IL_A_CODE_S3")
    output_s3 = _env("IL_A_OUTPUT_S3")
    missing = [k for k, v in {"IL_A_JOB_QUEUE": queue, "IL_A_JOB_DEFINITION": jobdef,
                              "IL_A_CODE_S3": code_s3, "IL_A_OUTPUT_S3": output_s3}.items() if not v]
    if missing:
        raise RuntimeError(f"orchestrator stack did not inject {missing} — is IL Pattern A wired?")

    policy = submit["policy"]
    job_name = submit.get("job_name") or f"vla-ft-{policy}-{time.strftime('%Y%m%d-%H%M%S')}"

    # 1. Upload the verified train.py to the code S3 location (the source_dir
    #    equivalent), exactly as batch_launch.py does. train.py ships in this Lambda's
    #    asset (the SAME file as the container's src/train.py — not a copy that can drift).
    here = os.path.dirname(os.path.abspath(__file__))
    train_py = os.path.join(here, "src", "train.py")
    cb, ck = _split_s3(code_s3)
    session.client("s3").upload_file(train_py, cb, ck)

    # 2. Hyperparameters -> a single S3 JSON file (NOT per-knob override env). Identical
    #    key set + ordering to batch_launch.py; the override carries only the pointer so
    #    the long LoRA regex never counts against Batch's 8192B ceiling.
    hp = {
        "policy": policy,
        "steps": str(submit["steps"]),
        "batch_size": str(submit["per_device_batch"]),
        "save_freq": str(submit.get("save_freq") or 2000),
        "dtype": "bfloat16",
        "gradient_checkpointing": "true",
        # Front door (MCP / vla_ft_cli) exposes 3 modes — expert_only / full_vlm / lora.
        # Vision-encoder freeze is a launcher-level power-user knob (launch.py /
        # batch_launch.py --freeze-vision-encoder), deliberately NOT surfaced on the
        # orchestrated path, so this pins the verified default. train.py honors it if set.
        "freeze_vision_encoder": "false",
        "train_expert_only": "true" if submit.get("expert_only") else "false",
        "job_name": job_name.replace("-", "_"),
    }
    if submit.get("pretrained_path"):
        hp["pretrained_path"] = submit["pretrained_path"]
    # LoRA (full-VLM-on-one-GPU; train.py enforces the expert_only mutual exclusion).
    if submit.get("lora"):
        hp["lora"] = "true"
        if submit.get("lora_r") is not None:
            hp["lora_r"] = str(submit["lora_r"])
        if submit.get("lora_alpha") is not None:
            hp["lora_alpha"] = str(submit["lora_alpha"])
        if submit.get("lora_target_modules") is not None:
            hp["lora_target_modules"] = submit["lora_target_modules"]
    if submit.get("val_episodes") is not None:
        hp["val_episodes"] = str(submit["val_episodes"])
    if submit.get("select_best"):
        hp["select_best"] = "true"
    if submit.get("early_stop_patience") is not None:
        hp["early_stop_patience"] = str(submit["early_stop_patience"])

    # Upload the hp JSON beside the code (keyed by job name; cb/ck = the code bucket/key
    # from step 1), double-encoding each value the way SageMaker's hyperparameters.json
    # does so train.py's _coerce undoes one layer.
    import json
    hp_key = f"{ck.rsplit('/', 1)[0]}/{job_name}/hyperparameters.json" if "/" in ck \
        else f"{job_name}/hyperparameters.json"
    hp_s3 = f"s3://{cb}/{hp_key}"
    session.client("s3").put_object(
        Bucket=cb, Key=hp_key,
        Body=json.dumps({k: json.dumps(v) for k, v in hp.items()}).encode("utf-8"),
        ContentType="application/json",
    )

    # Override env: bootstrap wiring + the single hp pointer (no SM_HP_*).
    env_overrides = [
        {"name": "VLA_FT_HP_S3", "value": hp_s3},
        {"name": "VLA_FT_DATASET_S3", "value": submit["dataset_s3"]},
        {"name": "VLA_FT_OUTPUT_S3", "value": f"{output_s3.rstrip('/')}/{job_name}"},
        {"name": "VLA_FT_CODE_S3", "value": code_s3},
        {"name": "VLA_FT_CHECKPOINT_DIR", "value": f"/mnt/efs/checkpoints/{job_name}"},
        {"name": "SM_NUM_GPUS", "value": str(submit["num_gpus"])},
    ]

    if want_hf:
        token = _read_hf_token(session, _env("HF_TOKEN_SSM"), _env("HF_TOKEN_SSM_REGION"))
        if token:
            env_overrides += [
                {"name": "HF_TOKEN", "value": token},
                {"name": "HUGGING_FACE_HUB_TOKEN", "value": token},
            ]

    container_overrides = {"environment": env_overrides}
    if submit["num_gpus"] and submit["num_gpus"] != 1:
        container_overrides["resourceRequirements"] = [
            {"type": "GPU", "value": str(submit["num_gpus"])}]

    resp = session.client("batch", region_name=region).submit_job(
        jobName=job_name, jobQueue=queue, jobDefinition=jobdef,
        containerOverrides=container_overrides,
    )
    return {"status": "submitted", "backend": "batch", "axis": "il",
            "job_name": job_name, "job_id": resp["jobId"],
            "output_s3": f"{output_s3.rstrip('/')}/{job_name}/output/"}


# ── RL Pattern A (AWS Batch) — direct submit, faithful to rl_launch.py ──────────────
def _submit_rl_batch(session, region: str, submit: dict) -> dict:
    queue = _env("RL_A_JOB_QUEUE")
    jobdef = _env("RL_A_JOB_DEFINITION")
    output_s3 = _env("RL_A_OUTPUT_S3")
    missing = [k for k, v in {"RL_A_JOB_QUEUE": queue, "RL_A_JOB_DEFINITION": jobdef,
                              "RL_A_OUTPUT_S3": output_s3}.items() if not v]
    if missing:
        raise RuntimeError(f"orchestrator stack did not inject {missing} — is RL Pattern A wired?")

    job_name = submit.get("job_name") or f"isaac-rl-{time.strftime('%Y%m%d-%H%M%S')}"

    # RL_* contract consumed by rl_train_bootstrap.py — identical to rl_launch.py.
    env = {
        "RL_TASK": submit["task"],
        "RL_EXPERIMENT_NAME": submit["experiment_name"],
        "RL_OUTPUT_S3": f"{output_s3.rstrip('/')}/{job_name}",
        "RL_CHECKPOINT_DIR": f"/mnt/efs/checkpoints/{job_name}",
        "RL_NUM_GPUS": str(submit["num_gpus"]),
        "RL_PLAY_ENVS": str(submit.get("play_envs") or 32),
    }
    if submit.get("num_envs") is not None:
        env["RL_NUM_ENVS"] = str(submit["num_envs"])
    if submit.get("max_iterations") is not None:
        env["RL_MAX_ITERATIONS"] = str(submit["max_iterations"])
    if submit.get("seed") is not None:
        env["RL_SEED"] = str(submit["seed"])
    if submit.get("skip_export"):
        env["RL_SKIP_EXPORT"] = "true"
    if submit.get("overrides"):
        env["RL_EXTRA_OVERRIDES"] = " ".join(submit["overrides"])

    container_overrides = {"environment": [{"name": k, "value": v} for k, v in env.items()]}
    if submit["num_gpus"] and submit["num_gpus"] != 1:
        container_overrides["resourceRequirements"] = [
            {"type": "GPU", "value": str(submit["num_gpus"])}]

    resp = session.client("batch", region_name=region).submit_job(
        jobName=job_name, jobQueue=queue, jobDefinition=jobdef,
        containerOverrides=container_overrides,
    )
    return {"status": "submitted", "backend": "batch", "axis": "rl",
            "job_name": job_name, "job_id": resp["jobId"],
            "output_s3": f"{output_s3.rstrip('/')}/{job_name}/output/"}


# ── GR00T Pattern A (AWS Batch) — direct submit, faithful to gr00t_launch.py ────────
def _submit_groot_batch(session, region: str, submit: dict, *, want_hf: bool) -> dict:
    queue = _env("GROOT_A_JOB_QUEUE")
    jobdef = _env("GROOT_A_JOB_DEFINITION")
    output_s3 = _env("GROOT_A_OUTPUT_S3")
    missing = [k for k, v in {"GROOT_A_JOB_QUEUE": queue, "GROOT_A_JOB_DEFINITION": jobdef,
                              "GROOT_A_OUTPUT_S3": output_s3}.items() if not v]
    if missing:
        raise RuntimeError(f"orchestrator stack did not inject {missing} — is GR00T Pattern A wired?")

    job_name = submit.get("job_name") or f"gr00t-n17-{time.strftime('%Y%m%d-%H%M%S')}"

    # GROOT_* contract consumed by gr00t_train_bootstrap.py — identical key set + values to
    # gr00t_launch.py (the verified launcher). The trainer (launch_finetune.py) is baked in
    # the image, so unlike IL Pattern A there is NO train.py upload step.
    env = {
        "GROOT_DATASET_S3": submit["dataset_s3"],
        "GROOT_OUTPUT_S3": f"{output_s3.rstrip('/')}/{job_name}",
        "GROOT_BASE_MODEL": submit.get("base_model") or "nvidia/GR00T-N1.7-3B",
        "GROOT_EMBODIMENT_TAG": submit.get("embodiment_tag") or "UNITREE_G1",
        "GROOT_MAX_STEPS": str(submit["max_steps"]),
        "GROOT_SAVE_STEPS": str(submit["save_steps"]),
        "GROOT_GLOBAL_BATCH": str(submit.get("global_batch") or 64),
        "GROOT_NUM_GPUS": str(submit["num_gpus"]),
    }
    if submit.get("learning_rate") is not None:
        env["GROOT_LEARNING_RATE"] = str(submit["learning_rate"])
    if submit.get("action_horizon") is not None:
        env["GROOT_ACTION_HORIZON"] = str(submit["action_horizon"])
    if submit.get("liveness_deadline_s") is not None:
        env["GROOT_LIVENESS_DEADLINE_S"] = str(submit["liveness_deadline_s"])
    if submit.get("extra"):
        env["GROOT_EXTRA_ARGS"] = " ".join(submit["extra"])

    # HF token: GR00T's VLM backbone (nvidia/Cosmos-Reason2-2B) is GATED and downloads at
    # runtime, so the token is REQUIRED (not optional like the GR00T-N1.7-3B repo itself).
    # Read here, inject as env — never into the JobDefinition / CloudWatch / SFN history.
    if want_hf:
        token = _read_hf_token(session, _env("HF_TOKEN_SSM"), _env("HF_TOKEN_SSM_REGION"))
        if token:
            env["HF_TOKEN"] = token
            env["HUGGING_FACE_HUB_TOKEN"] = token

    container_overrides = {"environment": [{"name": k, "value": v} for k, v in env.items()]}
    if submit["num_gpus"] and submit["num_gpus"] != 1:
        container_overrides["resourceRequirements"] = [
            {"type": "GPU", "value": str(submit["num_gpus"])}]

    resp = session.client("batch", region_name=region).submit_job(
        jobName=job_name, jobQueue=queue, jobDefinition=jobdef,
        containerOverrides=container_overrides,
    )
    return {"status": "submitted", "backend": "batch", "axis": "gr00t",
            "job_name": job_name, "job_id": resp["jobId"],
            "output_s3": f"{output_s3.rstrip('/')}/{job_name}/output/"}


# ── Pattern B (SageMaker) — recommend-only handoff (do NOT fork launch.py) ──────────
def _handoff_sagemaker(submit: dict, *, want_hf: bool) -> dict:
    """Resolve the plan into the exact verified launch.py command, and hand it off.

    We deliberately do NOT call create_training_job from this Lambda: the verified
    launcher uses the SageMaker SDK estimator (source_dir tar + checkpoint/Managed-Spot
    wiring) and reproducing it in raw boto3 would fork the verified path. Returning the
    command keeps the orchestrator's decision value while leaving the actual submit to
    the verified launcher (run by an operator or CI)."""
    role = _env("B_EXECUTION_ROLE")
    image = _env("B_IMAGE_URI")
    output_s3 = _env("B_OUTPUT_S3")
    instance = submit.get("sm_instance_type") or f"ml.{submit['instance_type']}"
    argv = [
        "python containers/vla-ft/launch.py",
        f"--policy {submit['policy']}",
        f"--dataset-s3 {submit['dataset_s3']}",
        f"--image-uri {image or '<PatternB ImageUriHint>'}",
        f"--role {role or '<PatternB ExecutionRoleArn>'}",
        f"--output-s3 {output_s3 or '<PatternB OutputS3Hint>'}",
        f"--instance-type {instance}",
        f"--num-gpus {submit['num_gpus']}",
        f"--steps {submit['steps']}",
        f"--batch-size {submit['per_device_batch']}",
    ]
    if submit.get("pretrained_path"):
        argv.append(f"--pretrained-path {submit['pretrained_path']}")
    if submit.get("expert_only"):
        argv.append("--train-expert-only true")
    if submit.get("lora"):
        argv.append("--lora true")
        if submit.get("lora_r") is not None:
            argv.append(f"--lora-r {submit['lora_r']}")
        if submit.get("lora_alpha") is not None:
            argv.append(f"--lora-alpha {submit['lora_alpha']}")
        if submit.get("lora_target_modules") is not None:
            argv.append(f"--lora-target-modules {submit['lora_target_modules']}")
    if not submit.get("spot", True):
        argv.append("--no-spot")
    if submit.get("select_best"):
        argv += ["--select-best true",
                 f"--val-episodes {submit.get('val_episodes') or 5}",
                 f"--save-freq {submit.get('save_freq') or 2000}"]
    if submit.get("early_stop_patience") is not None:
        argv.append(f"--early-stop-patience {submit['early_stop_patience']}")
    if want_hf:
        argv += [f"--hf-token-ssm {_env('HF_TOKEN_SSM') or '/pai/hf-token'}",
                 f"--hf-token-ssm-region {_env('HF_TOKEN_SSM_REGION') or 'us-east-1'}"]
    return {
        "status": "handoff", "backend": "sagemaker", "axis": "il",
        "reason": ("Pattern B (SageMaker) is launched by the verified launch.py "
                   "(SageMaker SDK estimator), not forked in-Lambda. Run this command:"),
        "handoff_command": " \\\n    ".join(argv),
    }


# ── dispatch ────────────────────────────────────────────────────────────────────────
def submit(plan_output: dict) -> dict:
    """Dispatch the plan to the chosen backend. The pure-ish core (boto3 is the only
    side effect, on the Batch paths). `plan_output` is orchestrator_plan's return."""
    import boto3

    s = plan_output["submit"]
    axis = plan_output["axis"]
    pattern = plan_output["pattern"]
    region = _env("REGION") or _env("AWS_REGION")
    session = boto3.Session(region_name=region) if region else boto3.Session()
    region = session.region_name

    # HF token wanted for gated backbones: pi-family (IL) and ALWAYS for GR00T (its
    # Cosmos-Reason2-2B VLM backbone is gated and pulled at runtime). Matches vla_ft_cli /
    # gr00t_launch.py.
    want_hf_il = axis == "il" and (s.get("policy") or "").startswith("pi") and bool(_env("HF_TOKEN_SSM"))
    want_hf_groot = axis == "gr00t" and bool(_env("HF_TOKEN_SSM"))

    if pattern == "A" and axis == "il":
        return _submit_il_batch(session, region, s, want_hf=want_hf_il)
    if pattern == "A" and axis == "rl":
        return _submit_rl_batch(session, region, s)
    if pattern == "A" and axis == "gr00t":
        return _submit_groot_batch(session, region, s, want_hf=want_hf_groot)
    if pattern == "B":
        return _handoff_sagemaker(s, want_hf=want_hf_il)
    # Pattern C is gated out by plan.runnable; reaching here is a programming error.
    raise RuntimeError(f"submit reached an un-runnable plan (pattern {pattern}, axis {axis}) "
                       f"— the Choice state should have routed Pattern C to a recommendation.")


def handler(event, context=None):  # noqa: ARG001 (Lambda signature)
    """AWS Lambda entry point. Step Functions passes the plan Lambda's output as `event`
    (the whole envelope, so `event['submit']` / `event['pattern']` are present)."""
    return submit(event)
