#!/usr/bin/env python3
"""
Submit a VLA fine-tune as an AWS Batch job (Pattern A).

The Batch counterpart of launch.py (Pattern B / SageMaker). Same verified container,
same train.py, same hyperparameter surface — only the orchestration differs:

  - SageMaker auto-downloads the dataset, reads hyperparameters.json, and uploads the
    model. Batch does none of that, so batch_bootstrap.py (the JobDefinition command)
    does it instead. This launcher feeds the bootstrap via container-override env.
  - train.py is shipped to SageMaker as `source_dir`; here we upload it to S3
    (VLA_FT_CODE_S3) so the bootstrap can fetch the exact, unchanged file. Re-run this
    launcher after editing train.py and the next job picks it up — no image rebuild.

Hyperparameters travel as a single S3 JSON file (VLA_FT_HP_S3), NOT as per-knob
SM_HP_<name> override env. The bootstrap stages that file at SageMaker's standard
hyperparameters.json path, so the SAME train.py code path runs under both backends —
and the per-knob hp (notably the long LoRA target regex) no longer counts against
Batch's 8192B container-override ceiling, which is what kept tipping it over.

Quickstart (π0.5 probe — the single-GPU tier Pattern A targets):

  python batch_launch.py \
    --policy pi05 --pretrained-path lerobot/pi05_base \
    --dataset-s3 s3://.../lerobot_dataset/ \
    --steps 2000 --batch-size 16 \
    --job-queue   <PatternA JobQueueArn output> \
    --job-definition <PatternA JobDefinitionArn output> \
    --code-s3     <PatternA CodeS3Hint output> \
    --output-s3   <PatternA OutputS3Hint output> \
    --hf-token-ssm /pai/hf-token --hf-token-ssm-region us-east-1

All --job-queue / --job-definition / --code-s3 / --output-s3 values come straight
from the PatternAStack CloudFormation outputs (no hardcoding).
"""

import argparse
import sys
import time


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Model / data (mirror launch.py)
    p.add_argument("--policy", required=True, help="LeRobot policy: pi05, pi0, act, ...")
    p.add_argument("--pretrained-path", default=None, help="Base ckpt (HF id or s3://).")
    p.add_argument("--dataset-s3", required=True, help="S3 URI of a LeRobot v3 dataset root.")

    # Training knobs (mirror launch.py / openarm-lift-pi05.yaml)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--save-freq", type=int, default=2000)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--gradient-checkpointing", default="true", choices=["true", "false"])
    p.add_argument("--freeze-vision-encoder", default="false", choices=["true", "false"])
    p.add_argument("--train-expert-only", default="false", choices=["true", "false"],
                   help="OOM fallback: freeze VLM, train action expert only (~24GB).")
    # LoRA / PEFT — full-VLM fine-tune on one GPU (lerobot-native, no fork). Mutually
    # exclusive with --train-expert-only. QLoRA (4-bit) is not available at this commit.
    p.add_argument("--lora", default="false", choices=["true", "false"],
                   help="LoRA fine-tune: freeze the base, train low-rank adapters only "
                        "(full VLM on one L40S, no OOM). Checkpoint is adapter-only.")
    p.add_argument("--lora-r", type=int, default=None, help="LoRA rank (lerobot default 16).")
    p.add_argument("--lora-alpha", type=int, default=None,
                   help="LoRA scaling alpha (lerobot default = r).")
    p.add_argument("--lora-target-modules", default=None,
                   help="OVERRIDE adapted modules (omit = policy default; pi0/pi05 built-in).")
    # Early-stop / overfit guard (opt-in; omit = verified path).
    p.add_argument("--val-episodes", type=int, default=None)
    p.add_argument("--select-best", default="false", choices=["true", "false"])
    p.add_argument("--early-stop-patience", type=int, default=None)
    p.add_argument("--early-stop-min-delta", type=float, default=None)
    p.add_argument("--extra", action="append", default=[], metavar="key=value",
                   help="Passthrough lerobot-train flag, repeatable.")

    # Batch plumbing (all from PatternAStack outputs)
    p.add_argument("--job-queue", required=True, help="PatternA JobQueueArn output.")
    p.add_argument("--job-definition", required=True, help="PatternA JobDefinitionArn output.")
    p.add_argument("--code-s3", required=True, help="PatternA CodeS3Hint output (train.py upload target).")
    p.add_argument("--output-s3", required=True, help="PatternA OutputS3Hint output (model lands here).")
    p.add_argument("--num-gpus", type=int, default=1,
                   help="GPUs to train across on the node. Pattern A is single-GPU tier; default 1.")
    p.add_argument("--region", default=None)
    p.add_argument("--job-name", default=None, help="Override the auto job name.")

    # HF token for gated backbones (PaliGemma for pi05) — injected as job env.
    p.add_argument("--hf-token-secret", default=None, help="Secrets Manager secret name.")
    p.add_argument("--hf-token-ssm", default=None, help="SSM SecureString parameter name.")
    p.add_argument("--hf-token-ssm-region", default=None, help="Region of --hf-token-ssm.")
    return p.parse_args()


def _split_s3(uri):
    rest = uri[len("s3://"):]
    bucket, _, key = rest.partition("/")
    return bucket, key


def main():
    args = parse_args()
    try:
        import boto3
    except ImportError:
        print("ERROR: pip install boto3", file=sys.stderr)
        sys.exit(1)

    sess = boto3.Session(region_name=args.region) if args.region else boto3.Session()
    region = sess.region_name
    job_name = args.job_name or f"vla-ft-{args.policy}-{time.strftime('%Y%m%d-%H%M%S')}"

    # 1. Upload the verified train.py to S3 (the source_dir equivalent). The bootstrap
    #    fetches it at job start, so editing train.py never needs an image rebuild.
    import os
    train_py = os.path.join(os.path.dirname(__file__), "src", "train.py")
    cb, ck = _split_s3(args.code_s3)
    sess.client("s3").upload_file(train_py, cb, ck)
    print(f"Uploaded {train_py} -> {args.code_s3}")

    # 2. Hyperparameters -> a single S3 JSON file (NOT per-knob override env). Plain
    #    string values, so train.py's file branch resolves them identically to the old
    #    SM_HP_* env path (same _coerce per value); the override then carries only the
    #    VLA_FT_HP_S3 pointer, independent of how large the hp (e.g. LoRA regex) get.
    hp = {
        "policy": args.policy,
        "steps": str(args.steps),
        "batch_size": str(args.batch_size),
        "save_freq": str(args.save_freq),
        "dtype": args.dtype,
        "gradient_checkpointing": args.gradient_checkpointing,
        "freeze_vision_encoder": args.freeze_vision_encoder,
        "train_expert_only": args.train_expert_only,
        "job_name": job_name.replace("-", "_"),
    }
    if args.pretrained_path:
        hp["pretrained_path"] = args.pretrained_path
    # LoRA (only emit when on -> default jobs byte-identical; values stringified for the
    # SM_HP_ env contract that train.py's env fallback reads).
    if args.lora == "true":
        hp["lora"] = "true"
        if args.lora_r is not None:
            hp["lora_r"] = str(args.lora_r)
        if args.lora_alpha is not None:
            hp["lora_alpha"] = str(args.lora_alpha)
        if args.lora_target_modules is not None:
            hp["lora_target_modules"] = args.lora_target_modules
    if args.val_episodes is not None:
        hp["val_episodes"] = str(args.val_episodes)
    if args.select_best == "true":
        hp["select_best"] = args.select_best
    if args.early_stop_patience is not None:
        hp["early_stop_patience"] = str(args.early_stop_patience)
    if args.early_stop_min_delta is not None:
        hp["early_stop_min_delta"] = str(args.early_stop_min_delta)
    for kv in args.extra:
        if "=" not in kv:
            print(f"ERROR: --extra must be key=value, got: {kv}", file=sys.stderr)
            sys.exit(1)
        k, v = kv.split("=", 1)
        hp[k] = v

    # Upload the hp JSON beside the code (the VLA_FT_CODE_S3 train.py prefix), keyed by
    # job name so concurrent/resumed jobs don't collide. SageMaker stores each value as
    # json.dumps(value); the launcher hp values are already plain strings, so dump them
    # the same way train.py's _coerce expects to undo one JSON layer.
    import json
    hp_key = f"{ck.rsplit('/', 1)[0]}/{job_name}/hyperparameters.json" if "/" in ck \
        else f"{job_name}/hyperparameters.json"
    hp_s3 = f"s3://{cb}/{hp_key}"
    sess.client("s3").put_object(
        Bucket=cb, Key=hp_key,
        Body=json.dumps({k: json.dumps(v) for k, v in hp.items()}).encode("utf-8"),
        ContentType="application/json",
    )
    print(f"Uploaded hyperparameters -> {hp_s3}")

    # Override env: bootstrap wiring + the single hp pointer. No SM_HP_* here, so the
    # merged override stays well under Batch's 8192B ceiling regardless of hp size.
    env_overrides = [
        {"name": "VLA_FT_HP_S3", "value": hp_s3},
        {"name": "VLA_FT_DATASET_S3", "value": args.dataset_s3},
        {"name": "VLA_FT_OUTPUT_S3", "value": f"{args.output_s3.rstrip('/')}/{job_name}"},
        {"name": "VLA_FT_CODE_S3", "value": args.code_s3},
        {"name": "VLA_FT_CHECKPOINT_DIR", "value": f"/mnt/efs/checkpoints/{job_name}"},
        {"name": "SM_NUM_GPUS", "value": str(args.num_gpus)},
    ]

    # 3. HF token (gated backbones) — read here, inject as env so it never lands in the
    #    JobDefinition or CloudWatch. Mirrors launch.py.
    token = None
    if args.hf_token_secret:
        token = sess.client("secretsmanager").get_secret_value(
            SecretId=args.hf_token_secret)["SecretString"]
    elif args.hf_token_ssm:
        token = sess.client("ssm", region_name=args.hf_token_ssm_region or region).get_parameter(
            Name=args.hf_token_ssm, WithDecryption=True)["Parameter"]["Value"]
    if token:
        env_overrides += [
            {"name": "HF_TOKEN", "value": token},
            {"name": "HUGGING_FACE_HUB_TOKEN", "value": token},
        ]

    resource_overrides = []
    if args.num_gpus and args.num_gpus != 1:
        resource_overrides.append({"type": "GPU", "value": str(args.num_gpus)})

    print("=" * 64)
    print(f"Submitting VLA-FT Batch job: {job_name}")
    print(f"  region    : {region}")
    print(f"  policy    : {args.policy}  base={args.pretrained_path}")
    print(f"  dataset   : {args.dataset_s3}")
    print(f"  output    : {args.output_s3.rstrip('/')}/{job_name}/output/")
    print(f"  queue     : {args.job_queue}")
    print("=" * 64)

    container_overrides = {"environment": env_overrides}
    if resource_overrides:
        container_overrides["resourceRequirements"] = resource_overrides

    resp = sess.client("batch", region_name=region).submit_job(
        jobName=job_name,
        jobQueue=args.job_queue,
        jobDefinition=args.job_definition,
        containerOverrides=container_overrides,
    )
    print(f"Submitted Batch job id: {resp['jobId']}")
    print(f"Track: aws batch describe-jobs --jobs {resp['jobId']} --region {region} "
          f"--query 'jobs[0].status'")


if __name__ == "__main__":
    main()
