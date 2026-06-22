#!/usr/bin/env python3
"""
Submit a VLA fine-tune as a managed SageMaker Training Job.

This is the **Pattern B** launcher (SageMaker Training Job): it wraps
the verified single-EC2 lerobot-train fine-tune (e.g. an `openarm-lift-pi05`
single-GPU run) into a managed job with Managed-Spot auto-resume and
automatic S3 checkpoint sync. The training logic itself lives in src/train.py and
is unchanged from the EC2 path — only the orchestration is managed now.

Quickstart (π0.5 probe, mirrors openarm-lift-pi05.yaml steps=2000):

  python launch.py \
    --policy pi05 \
    --pretrained-path lerobot/pi05_base \
    --dataset-s3 s3://my-bucket/openarm-lift/lerobot_dataset/ \
    --instance-type ml.g6e.4xlarge \
    --steps 2000 --batch-size 16 \
    --image-uri <ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com/pai/vla-ft:latest

Full-VLM full FT (Option 4 — 4xL40S, ~34 epoch on the 50-ep OpenArm set):

  python launch.py \
    --policy pi05 \
    --pretrained-path lerobot/pi05_base \
    --dataset-s3 s3://example-openarm-lift-dataset/lerobot_dataset/ \
    --instance-type ml.g6e.12xlarge \
    --steps 20000 --batch-size 4 \
    --no-spot --region us-west-2 \
    --hf-token-ssm /pai/hf-token --hf-token-ssm-region us-east-1 \
    --image-uri <ACCOUNT>.dkr.ecr.us-west-2.amazonaws.com/pai/vla-ft:latest

  absolute action (use_relative_actions): NOT passed — pi05's default is OFF, and
  both the verified openarm-lift-pi05 EC2 run and the PASSED smoke achieved the
  REQUIRED absolute-action behavior by OMITTING the flag (the A3 smoke log confirmed
  `use_relative_actions:False`). Do NOT add `--extra use_relative_actions=false`:
  the bare flag has no `policy.` prefix and was never in the verified lock, so
  draccus may reject it. Match the lock → omit.

  Multi-GPU is automatic on g6e.12xlarge (4xL40S): train.py detects SM_NUM_GPUS=4
  and wraps lerobot in `accelerate launch --multi_gpu --num_processes=4`. NOTE: this
  is full-VLM (NO --train-expert-only), so the 2B PaliGemma VLM trains too — needs
  the 48GB L40S (won't fit A10G 24GB). Effective batch = 4x4 = 16 (= the verified
  single-GPU smoke's effective batch), steps held at 20000 → same optimizer
  trajectory, ~4x faster wall-clock. compile_model stays off (default); the smoke
  showed torch.compile cudagraphs OOMs even 48GB.

The container (docker/Dockerfile, built + pushed by build.sh) has lerobot
pre-installed; src/ is shipped as source_dir so train.py is the entry point.

Outputs:
  - checkpoints  -> s3://<output>/<job>/checkpoints/   (live sync, Managed-Spot resumes from here)
  - final model  -> s3://<output>/<job>/output/model.tar.gz
The final-model S3 prefix is exactly the handoff point into vla-hub serving
(set vla-hub MODEL_CHECKPOINT_DIR to a mount of this prefix — see README §handoff).
"""

import argparse
import sys
import time


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Model / data
    p.add_argument("--policy", required=True,
                   help="LeRobot policy type: pi05, pi0, act, groot, smolvla, xvla, ...")
    p.add_argument("--pretrained-path", default=None,
                   help="Base checkpoint (HF id or s3://). e.g. lerobot/pi05_base")
    p.add_argument("--dataset-s3", required=True,
                   help="S3 URI of a LeRobot v3 dataset root (must contain meta/info.json)")

    # Compute
    p.add_argument("--instance-type", default="ml.g6e.4xlarge",
                   help="SageMaker instance type. Single-GPU L40S default (fits 3B + grad-ckpt).")
    p.add_argument("--instance-count", type=int, default=1,
                   help="Nodes. >1 enables torchrun multi-node (Pattern C-lite). Default 1.")
    p.add_argument("--num-gpus", type=int, default=None,
                   help="GPUs per node to train across (single-node multi-GPU via "
                        "accelerate). Default: auto-detect from SM_NUM_GPUS on the host "
                        "(4 on g6e.12xlarge/g5.12xlarge, 1 on g6e.4xlarge). Set to 1 to "
                        "force single-GPU on a multi-GPU instance. Effective batch = "
                        "batch_size x num_gpus; lerobot does NOT auto-scale steps/lr.")
    p.add_argument("--volume-size", type=int, default=300,
                   help="EBS GB. venv + base weights + dataset + checkpoints.")

    # Training knobs (mirror openarm-lift-pi05.yaml)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--save-freq", type=int, default=2000)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--gradient-checkpointing", default="true", choices=["true", "false"])
    p.add_argument("--freeze-vision-encoder", default="false", choices=["true", "false"])
    p.add_argument("--train-expert-only", default="false", choices=["true", "false"],
                   help="OOM fallback: freeze VLM, train action expert only (~24GB).")

    # LoRA / PEFT — the recommended way to fine-tune a FULL VLA on one GPU. lerobot-
    # native (cfg.peft @ d1b1c5c8), no fork: the base is frozen and only low-rank
    # adapters train, so the fp32 Adam state that OOMs a full-VLM full-FT collapses.
    # Mutually exclusive with --train-expert-only (both freeze the VLM). QLoRA (4-bit)
    # is NOT available — lerobot has no bitsandbytes path at this commit; use --lora.
    p.add_argument("--lora", default="false", choices=["true", "false"],
                   help="LoRA fine-tune: freeze the base, train low-rank adapters only. "
                        "Lets the FULL VLM be fine-tuned on one L40S without OOM. The saved "
                        "checkpoint is ADAPTER-ONLY (needs the base at load — see README).")
    p.add_argument("--lora-r", type=int, default=None,
                   help="LoRA rank (lerobot default 16). Higher = more trainable params.")
    p.add_argument("--lora-alpha", type=int, default=None,
                   help="LoRA scaling alpha (lerobot default = r; scaling = alpha / r).")
    p.add_argument("--lora-target-modules", default=None,
                   help="OVERRIDE which modules to adapt (regex/suffix). OMIT to use the "
                        "policy default (pi0/pi05: action-expert q/v + action/state MLPs).")

    # Early-stop / overfit guard (all opt-in; omit = verified path byte-identical).
    p.add_argument("--val-episodes", type=int, default=None,
                   help="Hold out the LAST N episodes as a validation set (lerobot then "
                        "trains on episodes [0..total-N-1]). Required for --select-best. "
                        "NOTE: the OpenArm set is only 50 episodes, so a small held-out "
                        "val-loss is noisy — best-checkpoint selection mitigates this by "
                        "comparing several checkpoints rather than trusting one number.")
    p.add_argument("--select-best", default="false", choices=["true", "false"],
                   help="After training, score every saved checkpoint on the held-out "
                        "val episodes and ship the lowest-val-loss one (early stopping by "
                        "checkpoint selection — the overfit guard). Requires --val-episodes; "
                        "lower --save-freq to keep several checkpoints (e.g. --save-freq 2000 "
                        "with --steps 20000 -> 10 checkpoints to choose from).")
    p.add_argument("--early-stop-patience", type=int, default=None,
                   help="Converged-cost lever: stop training (SIGTERM, current checkpoint "
                        "kept) once TRAIN loss fails to improve for this many consecutive "
                        "log-points (log_freq=200 steps). Saves GPU time on a plateau. "
                        "Independent of --val-episodes (watches train loss, not val).")
    p.add_argument("--early-stop-min-delta", type=float, default=None,
                   help="Min train-loss improvement to reset the patience counter "
                        "(default 0.0). Only meaningful with --early-stop-patience.")

    p.add_argument("--extra", action="append", default=[], metavar="key=value",
                   help="Passthrough lerobot-train flag, repeatable. e.g. --extra optimizer.lr=2.5e-5")

    # Spot / managed
    p.add_argument("--no-spot", action="store_true",
                   help="Disable Managed Spot (use On-Demand). Spot is ON by default.")
    p.add_argument("--max-run", type=int, default=86400, help="Max training seconds.")
    p.add_argument("--max-wait", type=int, default=172800,
                   help="Max wait incl. Spot interruption (>= max-run). Ignored with --no-spot.")

    # Plumbing
    p.add_argument("--image-uri", required=True, help="ECR image URI (build.sh output).")
    p.add_argument("--role", default=None,
                   help="SageMaker execution role ARN. Default: sagemaker.get_execution_role().")
    p.add_argument("--region", default=None)
    p.add_argument("--output-s3", default=None,
                   help="Base S3 for output+checkpoints. Default: sagemaker default bucket.")
    p.add_argument("--job-name", default=None, help="Override the auto job name.")
    p.add_argument("--wait", action="store_true", help="Block + stream logs until the job ends.")
    p.add_argument("--hf-token-secret", default=None,
                   help="Secrets Manager secret name holding an HF token (gated backbones, e.g. PaliGemma).")
    p.add_argument("--hf-token-ssm", default=None,
                   help="SSM Parameter (SecureString) holding an HF token, e.g. /pai/hf-token. "
                        "Use --hf-token-ssm-region if the parameter lives in another region.")
    p.add_argument("--hf-token-ssm-region", default=None,
                   help="Region of --hf-token-ssm (default: job region). e.g. the param may be in us-east-1.")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        import boto3
        import sagemaker
        from sagemaker.pytorch import PyTorch
    except ImportError:
        print("ERROR: pip install sagemaker boto3", file=sys.stderr)
        sys.exit(1)

    boto_sess = boto3.Session(region_name=args.region) if args.region else boto3.Session()
    sm_sess = sagemaker.Session(boto_session=boto_sess)
    region = boto_sess.region_name
    role = args.role or sagemaker.get_execution_role(sagemaker_session=sm_sess)

    job_name = args.job_name or f"vla-ft-{args.policy}-{time.strftime('%Y%m%d-%H%M%S')}"
    output_path = args.output_s3 or f"s3://{sm_sess.default_bucket()}/vla-ft"
    checkpoint_s3 = f"{output_path.rstrip('/')}/{job_name}/checkpoints"

    # Hyperparameters -> src/train.py (SageMaker JSON-encodes each value).
    hp = {
        "policy": args.policy,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "save_freq": args.save_freq,
        "dtype": args.dtype,
        "gradient_checkpointing": args.gradient_checkpointing,
        "freeze_vision_encoder": args.freeze_vision_encoder,
        "train_expert_only": args.train_expert_only,
        "job_name": job_name.replace("-", "_"),
    }
    if args.pretrained_path:
        hp["pretrained_path"] = args.pretrained_path
    if args.num_gpus is not None:
        hp["num_gpus"] = args.num_gpus
    # LoRA (only emit when on -> default jobs byte-identical; train.py enforces the
    # mutual exclusion with train_expert_only).
    if args.lora == "true":
        hp["lora"] = "true"
        if args.lora_r is not None:
            hp["lora_r"] = args.lora_r
        if args.lora_alpha is not None:
            hp["lora_alpha"] = args.lora_alpha
        if args.lora_target_modules is not None:
            hp["lora_target_modules"] = args.lora_target_modules
    # Early-stop / overfit guard (only emit when set -> default jobs unchanged).
    if args.val_episodes is not None:
        hp["val_episodes"] = args.val_episodes
    if args.select_best == "true":
        hp["select_best"] = args.select_best
    if args.early_stop_patience is not None:
        hp["early_stop_patience"] = args.early_stop_patience
    if args.early_stop_min_delta is not None:
        hp["early_stop_min_delta"] = args.early_stop_min_delta
    for kv in args.extra:
        if "=" not in kv:
            print(f"ERROR: --extra must be key=value, got: {kv}", file=sys.stderr)
            sys.exit(1)
        k, v = kv.split("=", 1)
        hp[k] = v

    # Gated backbones (PaliGemma for pi05) need an HF token. Inject via env from
    # Secrets Manager so it never lands in the job definition / CloudWatch.
    environment = {"PYTHONUNBUFFERED": "1"}
    token = None
    if args.hf_token_secret:
        secrets = boto_sess.client("secretsmanager")
        token = secrets.get_secret_value(SecretId=args.hf_token_secret)["SecretString"]
    elif args.hf_token_ssm:
        ssm = boto_sess.client("ssm", region_name=args.hf_token_ssm_region or region)
        token = ssm.get_parameter(Name=args.hf_token_ssm, WithDecryption=True)["Parameter"]["Value"]
    if token:
        environment["HF_TOKEN"] = token
        environment["HUGGING_FACE_HUB_TOKEN"] = token

    spot = not args.no_spot
    estimator = PyTorch(
        entry_point="train.py",
        source_dir="src",
        image_uri=args.image_uri,
        role=role,
        sagemaker_session=sm_sess,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        volume_size=args.volume_size,
        hyperparameters=hp,
        environment=environment,
        output_path=output_path,
        # Managed-Spot: SageMaker syncs /opt/ml/checkpoints <-> checkpoint_s3_uri
        # and auto-resumes after an interruption. train.py reads it back on restart.
        use_spot_instances=spot,
        max_run=args.max_run,
        max_wait=args.max_wait if spot else None,
        checkpoint_s3_uri=checkpoint_s3,
        checkpoint_local_path="/opt/ml/checkpoints",
        # torchrun across nodes when instance_count>1 (single-node: harmless).
        distribution={"torch_distributed": {"enabled": True}} if args.instance_count > 1 else None,
        disable_profiler=True,
    )

    print("=" * 64)
    print(f"Submitting VLA-FT job: {job_name}")
    print(f"  region        : {region}")
    print(f"  policy        : {args.policy}  base={args.pretrained_path}")
    print(f"  instance      : {args.instance_count} x {args.instance_type}  (spot={spot})")
    print(f"  dataset       : {args.dataset_s3}")
    print(f"  checkpoints   : {checkpoint_s3}   (Managed-Spot resume)")
    print(f"  final model   : {output_path}/{job_name}/output/model.tar.gz")
    print("=" * 64)

    # 'training' channel -> /opt/ml/input/data/training/ inside the container.
    estimator.fit(
        inputs={"training": args.dataset_s3},
        job_name=job_name,
        wait=args.wait,
    )

    if not args.wait:
        print(f"\nSubmitted. Track: aws sagemaker describe-training-job "
              f"--training-job-name {job_name} --region {region}")


if __name__ == "__main__":
    main()
