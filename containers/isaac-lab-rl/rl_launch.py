#!/usr/bin/env python3
"""
Submit an Isaac Lab RL training run as an AWS Batch job (RL Pattern A).

The RL-axis counterpart of vla-ft's batch_launch.py. The "intent" of an RL job is the
**task id** (the env + reward cfg are registered to it) plus optional training knobs
and Hydra-style reward/task overrides — there is no dataset to upload (the simulator
generates experience on the GPU). So this launcher is thinner than the IL one: it sets
RL_* container-override env vars that rl_train_bootstrap.py (the JobDefinition command)
reads, then calls Batch SubmitJob.

The trained policy lands at:
  <output-s3>/<job>/output/<run>/   (checkpoints model_<N>.pt + exported/policy.onnx)

Quickstart (the verified reference task — Unitree H1 rough-terrain locomotion):

  python rl_launch.py \
    --task Isaac-Velocity-Rough-H1-v0 \
    --max-iterations 3000 --num-envs 4096 \
    --job-queue      <RlPatternA JobQueueArn output> \
    --job-definition <RlPatternA JobDefinitionArn output> \
    --output-s3      <RlPatternA OutputS3Hint output>

All --job-queue / --job-definition / --output-s3 values come straight from the
RlPatternAStack CloudFormation outputs (no hardcoding).
"""

import argparse
import sys
import time


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # RL intent
    p.add_argument("--task", default="Isaac-Velocity-Rough-H1-v0",
                   help="Isaac Lab task id (env + reward registered to it).")
    p.add_argument("--experiment-name", default="h1_rough",
                   help="rsl_rl experiment dir (logs/rsl_rl/<this>/). Match the task's RunnerCfg.")

    # Training knobs (rsl_rl/train.py flags; omit = task-registered defaults)
    p.add_argument("--num-envs", type=int, default=None, help="Parallel sim envs.")
    p.add_argument("--max-iterations", type=int, default=None, help="PPO training iterations.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--play-envs", type=int, default=32, help="Envs for the export/eval rollout.")
    p.add_argument("--skip-export", action="store_true",
                   help="Skip the play.py ONNX export step (checkpoints only).")
    p.add_argument("--liveness-deadline", type=int, default=None, metavar="SECONDS",
                   help="Fail-fast guard: max seconds from train launch to the first "
                        "sign of real training (a 'Learning iteration' log line or a "
                        "model_*.pt checkpoint). If neither appears, the job is "
                        "booted-but-idle and is killed so Batch fails in minutes, not "
                        "hours (the 5.5 h idle-run class). Default 1200 (20 min) in the "
                        "bootstrap; pass 0 to disable.")
    p.add_argument("--override", action="append", default=[], metavar="key=value",
                   help="Hydra-style reward/task/agent override, repeatable "
                        "(e.g. --override agent.max_iterations=500).")

    # Batch plumbing (all from RlPatternAStack outputs)
    p.add_argument("--job-queue", required=True, help="RlPatternA JobQueueArn output.")
    p.add_argument("--job-definition", required=True, help="RlPatternA JobDefinitionArn output.")
    p.add_argument("--output-s3", required=True, help="RlPatternA OutputS3Hint output (run dir lands here).")
    p.add_argument("--num-gpus", type=int, default=1,
                   help="GPUs to train across on the node (>1 → torchrun + --distributed).")
    p.add_argument("--region", default=None)
    p.add_argument("--job-name", default=None, help="Override the auto job name.")
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
    job_name = args.job_name or f"isaac-rl-{time.strftime('%Y%m%d-%H%M%S')}"

    # RL_* contract consumed by rl_train_bootstrap.py.
    env = {
        "RL_TASK": args.task,
        "RL_EXPERIMENT_NAME": args.experiment_name,
        "RL_OUTPUT_S3": f"{args.output_s3.rstrip('/')}/{job_name}",
        "RL_CHECKPOINT_DIR": f"/mnt/efs/checkpoints/{job_name}",
        "RL_NUM_GPUS": str(args.num_gpus),
        "RL_PLAY_ENVS": str(args.play_envs),
    }
    if args.num_envs is not None:
        env["RL_NUM_ENVS"] = str(args.num_envs)
    if args.max_iterations is not None:
        env["RL_MAX_ITERATIONS"] = str(args.max_iterations)
    if args.seed is not None:
        env["RL_SEED"] = str(args.seed)
    if args.skip_export:
        env["RL_SKIP_EXPORT"] = "true"
    if args.liveness_deadline is not None:
        env["RL_LIVENESS_DEADLINE_S"] = str(args.liveness_deadline)
    if args.override:
        env["RL_EXTRA_OVERRIDES"] = " ".join(args.override)

    env_overrides = [{"name": k, "value": v} for k, v in env.items()]

    resource_overrides = []
    if args.num_gpus and args.num_gpus != 1:
        resource_overrides.append({"type": "GPU", "value": str(args.num_gpus)})

    print("=" * 64)
    print(f"Submitting Isaac Lab RL Batch job: {job_name}")
    print(f"  region    : {region}")
    print(f"  task      : {args.task}")
    print(f"  iters     : {args.max_iterations or '(task default)'}  envs={args.num_envs or '(task default)'}")
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
