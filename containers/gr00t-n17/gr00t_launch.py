#!/usr/bin/env python3
"""
Submit a GR00T N1.7 fine-tune as an AWS Batch job (IL GR00T Pattern A).

IL-axis counterpart of vla-ft's batch_launch.py — same shape (set container-override env
the bootstrap reads, then Batch SubmitJob), but the "intent" is a LeRobot dataset + the
GR00T embodiment tag rather than a lerobot policy. There is no train.py to upload: the
GR00T trainer (launch_finetune.py) is baked into the image, so this launcher only sets
the GROOT_* env contract that gr00t_train_bootstrap.py (the JobDefinition command) reads.

The trained checkpoint lands at:
  <output-s3>/<job>/output/   (checkpoint-<N>/ + experiment_cfg/ + processor — the layout
                               run_gr00t_server.py --model-path loads for sim rollout)

Quickstart (the G1 adapter fine-tune — the verified reference dataset):

  python gr00t_launch.py \
    --dataset-s3 s3://example-gr00t-g1-dataset/2026-06-17-0410/task-0/lerobot/ \
    --embodiment-tag UNITREE_G1 \
    --max-steps 2000 --save-steps 1000 \
    --job-queue      <GrootPatternA JobQueueArn output> \
    --job-definition <GrootPatternA JobDefinitionArn output> \
    --output-s3      <GrootPatternA OutputS3Hint output>

For a cheap smoke first (verify checkpoint format before the full run):
    --max-steps 60 --save-steps 50 --liveness-deadline 2400

All --job-queue / --job-definition / --output-s3 values come straight from the
GrootPatternAStack CloudFormation outputs (no hardcoding). nvidia/GR00T-N1.7-3B is an
ungated HF model, so --hf-token-ssm is optional (only needed if HF rate-limits anon pulls).
"""

import argparse
import sys
import time


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # GR00T intent (dataset + embodiment)
    p.add_argument("--dataset-s3", required=True,
                   help="S3 URI of a GR00T-flavored LeRobot v2.1 dataset root.")
    p.add_argument("--embodiment-tag", default="UNITREE_G1",
                   help="Registered GR00T embodiment tag (default UNITREE_G1, a posttrain "
                        "tag → fine-tune required; no --modality-config-path needed).")
    p.add_argument("--base-model", default="nvidia/GR00T-N1.7-3B",
                   help="Base checkpoint (HF id or local path). Ungated; ~6 GB.")
    p.add_argument("--modality-config-path", default=None,
                   help="Custom modality config .py (only for NEW_EMBODIMENT; omit for "
                        "registered UNITREE_G1).")

    # Training knobs (launch_finetune.py flags; defaults match the bootstrap defaults)
    p.add_argument("--max-steps", type=int, default=2000,
                   help="Total optimizer steps (upstream default 10000; we default 2000 "
                        "for the 'one video' goal).")
    p.add_argument("--save-steps", type=int, default=1000,
                   help="Checkpoint interval (first checkpoint-<N>/ at this step).")
    p.add_argument("--global-batch-size", type=int, default=64,
                   help="Total batch across GPUs, pre-grad-accumulation (upstream default 64).")
    p.add_argument("--learning-rate", type=float, default=None,
                   help="Override the 1e-4 default.")
    p.add_argument("--action-horizon", type=int, default=None,
                   help="Set config.model.action_horizon before the model is built (via the "
                        "baked g1_finetune.py wrapper). REQUIRED = 50 for UNITREE_G1 "
                        "full-body: its data config uses a 50-step action horizon but the "
                        "base GR00T-N1.7-3B / model config default to 40, and "
                        "launch_finetune.py exposes no flag to change it (an unset value "
                        "would hit 'Action sequence length 50 exceeds max_action_horizon "
                        "40'). Omit for embodiments whose horizon already matches the "
                        "40-wide base head.")
    p.add_argument("--extra", action="append", default=[], metavar="FLAG",
                   help="Verbatim launch_finetune flag(s), repeatable "
                        "(e.g. --extra --weight-decay --extra 1e-4, or --extra --tune-llm).")
    p.add_argument("--liveness-deadline", type=int, default=None, metavar="SECONDS",
                   help="Fail-fast guard: max seconds from train launch to the first sign "
                        "of real training (a loss/learning_rate log line, a 'Starting "
                        "training' line, or a checkpoint-<N>/ dir). If none appear, the "
                        "job is booted-but-idle and is killed so Batch fails in minutes, "
                        "not hours. Default 1800 (30 min) in the bootstrap; 0 disables.")

    # Batch plumbing (all from GrootPatternAStack outputs)
    p.add_argument("--job-queue", required=True, help="GrootPatternA JobQueueArn output.")
    p.add_argument("--job-definition", required=True, help="GrootPatternA JobDefinitionArn output.")
    p.add_argument("--output-s3", required=True, help="GrootPatternA OutputS3Hint output (ckpt lands here).")
    p.add_argument("--num-gpus", type=int, default=1,
                   help="GPUs to train across on the node (>1 → torchrun --nproc_per_node).")
    p.add_argument("--region", default=None)
    p.add_argument("--job-name", default=None, help="Override the auto job name.")

    # HF token (optional — GR00T-N1.7-3B is ungated; only if HF throttles anon pulls).
    p.add_argument("--hf-token-secret", default=None, help="Secrets Manager secret name.")
    p.add_argument("--hf-token-ssm", default=None, help="SSM SecureString parameter name.")
    p.add_argument("--hf-token-ssm-region", default=None, help="Region of --hf-token-ssm.")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        import boto3
    except ImportError:
        print("ERROR: pip install boto3", file=sys.stderr)
        sys.exit(1)

    sess = boto3.Session(region_name=args.region) if args.region else boto3.Session()
    region = sess.region_name
    job_name = args.job_name or f"gr00t-n17-{time.strftime('%Y%m%d-%H%M%S')}"

    # GROOT_* contract consumed by gr00t_train_bootstrap.py.
    env = {
        "GROOT_DATASET_S3": args.dataset_s3,
        "GROOT_OUTPUT_S3": f"{args.output_s3.rstrip('/')}/{job_name}",
        "GROOT_BASE_MODEL": args.base_model,
        "GROOT_EMBODIMENT_TAG": args.embodiment_tag,
        "GROOT_MAX_STEPS": str(args.max_steps),
        "GROOT_SAVE_STEPS": str(args.save_steps),
        "GROOT_GLOBAL_BATCH": str(args.global_batch_size),
        "GROOT_NUM_GPUS": str(args.num_gpus),
    }
    if args.learning_rate is not None:
        env["GROOT_LEARNING_RATE"] = str(args.learning_rate)
    if args.action_horizon is not None:
        env["GROOT_ACTION_HORIZON"] = str(args.action_horizon)
    if args.modality_config_path:
        env["GROOT_MODALITY_CONFIG"] = args.modality_config_path
    if args.liveness_deadline is not None:
        env["GROOT_LIVENESS_DEADLINE_S"] = str(args.liveness_deadline)
    if args.extra:
        env["GROOT_EXTRA_ARGS"] = " ".join(args.extra)

    # HF token (optional) — read here, inject as env so it never lands in the JobDefinition
    # or CloudWatch. Mirrors batch_launch.py.
    token = None
    if args.hf_token_secret:
        token = sess.client("secretsmanager").get_secret_value(
            SecretId=args.hf_token_secret)["SecretString"]
    elif args.hf_token_ssm:
        token = sess.client("ssm", region_name=args.hf_token_ssm_region or region).get_parameter(
            Name=args.hf_token_ssm, WithDecryption=True)["Parameter"]["Value"]
    if token:
        env["HF_TOKEN"] = token
        env["HUGGING_FACE_HUB_TOKEN"] = token

    env_overrides = [{"name": k, "value": v} for k, v in env.items()]

    resource_overrides = []
    if args.num_gpus and args.num_gpus != 1:
        resource_overrides.append({"type": "GPU", "value": str(args.num_gpus)})

    print("=" * 64)
    print(f"Submitting GR00T N1.7 Batch job: {job_name}")
    print(f"  region     : {region}")
    print(f"  dataset    : {args.dataset_s3}")
    print(f"  embodiment : {args.embodiment_tag}")
    print(f"  steps      : {args.max_steps}  save={args.save_steps}  global-batch={args.global_batch_size}")
    if args.action_horizon is not None:
        print(f"  action-horizon : {args.action_horizon} (config.model.action_horizon via g1_finetune.py)")
    print(f"  output     : {args.output_s3.rstrip('/')}/{job_name}/output/")
    print(f"  queue      : {args.job_queue}")
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
