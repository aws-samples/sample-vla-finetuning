#!/usr/bin/env python3
"""
Submit a VLA RLVR job (RLinf PPO post-training of a pretrained pi0.5) to AWS Batch.

The RLVR counterpart of rl_launch.py. It builds the VRL_* container-override env that
vla_rl_bootstrap.py (the VlaRlJobDef command) reads, with the settings that ran the
LIBERO-Spatial PoC baked in as defaults, then calls Batch SubmitJob. Without --yes it
prints the plan and submits nothing.

Three modes, one per question you ask of a checkpoint:

  # 1. how good is the SFT checkpoint? (n = 200 on fixed reset states)
  python vla_rl_launch.py eval --stack PaiTrainingPlatform-RL-PatternA --yes

  # 2. post-train it with PPO, reward = task success (1 GPU, 64 trajectories / step)
  python vla_rl_launch.py train --steps 60 --stack PaiTrainingPlatform-RL-PatternA --yes

  # 3. how good is the PPO checkpoint? (same 200 reset states as 1)
  python vla_rl_launch.py eval --from-run <train job id> --step 60 \
    --stack PaiTrainingPlatform-RL-PatternA --yes

`train --resume-run <job id> --resume-step 60 --steps 120` continues a run.
`inspect` prints the image facts the other modes depend on (run it on a new image).

Measured on 1x L40S (g6e.8xlarge On-Demand): ~8 min per PPO step at 64 trajectories,
~10 min per n = 200 eval. Compare two evals with vla_rl_compare.py.

--job-queue / --job-definition come from the RlPatternA stack outputs (JobQueueArn /
VlaRlJobDefinitionArn); --stack reads them for you. Results go to the JobDef's
VRL_OUTPUT_S3 (<artifact bucket>/vla-rl/<job name>/) unless --output-s3 is given.
"""

import argparse
import shlex
import sys
import time

TRAIN_CONFIG = "libero_spatial_ppo_openpi_pi05"
EVAL_CONFIG = "libero_spatial_openpi_pi05_eval"
MODEL_REPO = "RLinf/RLinf-Pi05-LIBERO-SFT"
CACHE_DIR = "/mnt/efs/vla-rl"

# Per-GPU rollout shape that fit 1x L40S 48 GB next to the policy (16 envs x 4 epochs =
# 64 trajectories / step). More GPUs scale envs and the global batch, not the micro batch.
ENVS_PER_GPU = 16
ROLLOUT_EPOCHS = 4
MICRO_BATCH = 32
GLOBAL_BATCH_PER_GPU = 256
# 20 EGL-rendered eval envs fit beside the policy; 50 ran the GPU out of memory.
EVAL_ENVS = 20
# Per-GPU vCPU / memory (MiB) asked of Batch when --num-gpus > 1 (g6e.12xlarge = 4 GPUs).
VCPU_PER_GPU = 11
MEMORY_MIB_PER_GPU = 87500
MINUTES_PER_STEP_1GPU = 8.0


def run_dir(job_id, config=TRAIN_CONFIG):
    """EFS dir a train job writes (vla_rl_bootstrap.py logs to runs/<Batch job id>)."""
    return f"{CACHE_DIR}/runs/{job_id}/{config}"


def checkpoint_dir(job_id, step, config=TRAIN_CONFIG):
    return f"{run_dir(job_id, config)}/checkpoints/global_step_{step}"


def build_plan(args):
    """Return (job_name, container_overrides, timeout_seconds) for the parsed args.

    Pure: no AWS calls, so the test exercises exactly what gets submitted.
    """
    mode = args.mode
    stamp = time.strftime("%Y%m%d-%H%M")
    env = {"VRL_MODE": mode, "VRL_MODEL_REPO": args.model_repo}
    overrides = []
    resources = []
    timeout_h = args.timeout_hours

    if mode == "train":
        g = args.num_gpus
        overrides += [
            f"env.train.total_num_envs={ENVS_PER_GPU * g}",
            f"env.train.rollout_epoch={ROLLOUT_EPOCHS}",
            f"env.eval.total_num_envs={EVAL_ENVS}",
            f"actor.micro_batch_size={MICRO_BATCH}",
            f"actor.global_batch_size={GLOBAL_BATCH_PER_GPU * g}",
            "actor.enable_offload=True",
            "rollout.enable_offload=True",
            f"runner.max_epochs={args.steps}",
            f"runner.save_interval={args.save_interval}",
        ]
        if args.resume_run:
            overrides.append("runner.resume_dir="
                             + checkpoint_dir(args.resume_run, args.resume_step, args.config or TRAIN_CONFIG))
        if g > 1:
            resources = [
                {"type": "GPU", "value": str(g)},
                {"type": "VCPU", "value": str(VCPU_PER_GPU * g)},
                {"type": "MEMORY", "value": str(MEMORY_MIB_PER_GPU * g)},
            ]
        env["VRL_CONFIG"] = args.config or TRAIN_CONFIG
        if timeout_h is None:
            todo = args.steps - (args.resume_step if args.resume_run else 0)
            # 1.5x the 1-GPU measurement + 1 h for image pull and checkpoint upload.
            timeout_h = todo * MINUTES_PER_STEP_1GPU * 1.5 / 60 + 1
        name = f"vla-rl-train-{stamp}"
    elif mode == "eval":
        env["VRL_CONFIG"] = args.config or EVAL_CONFIG
        env["VRL_EVAL_BENCHMARK"] = "libero"
        overrides += [
            f"env.eval.total_num_envs={EVAL_ENVS}",
            f"env.eval.rollout_epoch={args.episodes // EVAL_ENVS}",
        ]
        if args.from_run:
            ckpt = f"{checkpoint_dir(args.from_run, args.step)}/actor/model_state_dict/full_weights.pt"
            # A PPO checkpoint carries the critic's value head; the strict state-dict
            # load fails unless the eval model is built with it.
            overrides += [
                f"runner.ckpt_path={ckpt}",
                "rollout.model.add_value_head=True",
                "++rollout.model.openpi.value_after_vlm=True",
            ]
        if timeout_h is None:
            timeout_h = 2
        name = f"vla-rl-eval-{'step%d' % args.step if args.from_run else 'sft'}-{stamp}"
    else:  # inspect
        if timeout_h is None:
            timeout_h = 1
        name = f"vla-rl-inspect-{stamp}"

    overrides += args.override
    if overrides:
        env["VRL_OVERRIDES"] = " ".join(overrides)
    name = args.job_name or name
    env["VRL_RUN_NAME"] = name
    if args.output_s3:
        env["VRL_OUTPUT_S3"] = args.output_s3.rstrip("/")

    container = {"environment": [{"name": k, "value": v} for k, v in env.items()]}
    if resources:
        container["resourceRequirements"] = resources
    return name, container, max(60, int(round(timeout_h * 3600)))


def stack_outputs(sess, stack):
    cfn = sess.client("cloudformation")
    outs = cfn.describe_stacks(StackName=stack)["Stacks"][0].get("Outputs", [])
    return {o["OutputKey"]: o["OutputValue"] for o in outs}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("mode", choices=["inspect", "eval", "train"])
    p.add_argument("--model-repo", default=MODEL_REPO, help="HF SFT checkpoint repo id.")
    p.add_argument("--config", default=None,
                   help=f"RLinf config name (default: train {TRAIN_CONFIG} / eval {EVAL_CONFIG}).")
    p.add_argument("--override", action="append", default=[], metavar="key=value",
                   help="Extra Hydra override, appended last (repeatable).")
    g = p.add_argument_group("train")
    g.add_argument("--steps", type=int, default=60, help="PPO steps (runner.max_epochs).")
    g.add_argument("--save-interval", type=int, default=20)
    g.add_argument("--num-gpus", type=int, default=1,
                   help="GPUs on one instance; scales envs and global batch (4 = g6e.12xlarge).")
    g.add_argument("--resume-run", default=None, metavar="JOB_ID",
                   help="Continue the train job with this Batch job id ...")
    g.add_argument("--resume-step", type=int, default=None, help="... from this saved step.")
    e = p.add_argument_group("eval")
    e.add_argument("--episodes", type=int, default=200,
                   help=f"Eval episodes (multiple of {EVAL_ENVS}; fixed reset states).")
    e.add_argument("--from-run", default=None, metavar="JOB_ID",
                   help="Evaluate a checkpoint of this train job (default: the SFT checkpoint). "
                        f"Assumes the run used {TRAIN_CONFIG}.")
    e.add_argument("--step", type=int, default=None, help="Which saved step of --from-run.")
    a = p.add_argument_group("aws")
    a.add_argument("--stack", default=None, help="RlPatternA stack name; reads the three ARNs below.")
    a.add_argument("--job-queue", default=None)
    a.add_argument("--job-definition", default=None, help="VlaRlJobDefinitionArn output.")
    a.add_argument("--output-s3", default=None, help="S3 prefix for run dirs (default: the JobDef's <artifact bucket>/vla-rl).")
    a.add_argument("--region", default=None)
    a.add_argument("--job-name", default=None)
    a.add_argument("--timeout-hours", type=float, default=None)
    a.add_argument("--yes", action="store_true", help="Submit. Without it, print the plan only.")
    args = p.parse_args(argv)

    if args.resume_run and args.resume_step is None:
        p.error("--resume-run needs --resume-step")
    if args.resume_run and args.resume_step >= args.steps:
        p.error("--steps is the total step count; it must exceed --resume-step")
    if args.from_run and args.step is None:
        p.error("--from-run needs --step")
    if args.episodes % EVAL_ENVS:
        p.error(f"--episodes must be a multiple of {EVAL_ENVS}")
    if args.num_gpus < 1:
        p.error("--num-gpus must be >= 1")
    return args


def main(argv=None):
    args = parse_args(argv)
    name, container, timeout_s = build_plan(args)

    print("=" * 64)
    print(f"VLA RLVR {args.mode}: {name}")
    for kv in container["environment"]:
        if kv["name"] == "VRL_OVERRIDES":
            for o in shlex.split(kv["value"]):
                print(f"  override  {o}")
        else:
            print(f"  {kv['name']:<20}{kv['value']}")
    for r in container.get("resourceRequirements", []):
        print(f"  resource  {r['type']}={r['value']}")
    print(f"  timeout   {timeout_s / 3600:.1f} h")
    if args.mode == "train":
        todo = args.steps - (args.resume_step if args.resume_run else 0)
        print(f"  estimate  ~{todo * MINUTES_PER_STEP_1GPU / 60:.1f} h at the 1-GPU rate "
              f"({MINUTES_PER_STEP_1GPU:g} min/step, 64 trajectories/step)")
    print("=" * 64)
    if not args.yes:
        print("Plan only. Add --yes to submit.")
        return 0

    try:
        import boto3
    except ImportError:
        print("ERROR: pip install boto3", file=sys.stderr)
        return 1
    sess = boto3.Session(region_name=args.region) if args.region else boto3.Session()
    queue, jobdef = args.job_queue, args.job_definition
    if args.stack:
        outs = stack_outputs(sess, args.stack)
        queue = queue or outs.get("JobQueueArn")
        jobdef = jobdef or outs.get("VlaRlJobDefinitionArn")
    if not queue or not jobdef:
        print("ERROR: need --stack, or --job-queue and --job-definition", file=sys.stderr)
        return 2
    resp = sess.client("batch").submit_job(
        jobName=name, jobQueue=queue, jobDefinition=jobdef,
        containerOverrides=container, timeout={"attemptDurationSeconds": timeout_s},
    )
    job_id = resp["jobId"]
    print(f"Submitted Batch job id: {job_id}")
    print(f"Track: aws batch describe-jobs --jobs {job_id} --region {sess.region_name} "
          f"--query 'jobs[0].status'")
    if args.mode == "train":
        print(f"Checkpoints: {run_dir(job_id)}/checkpoints/global_step_<N>/ (EFS) — "
              f"evaluate with: eval --from-run {job_id} --step <N>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
