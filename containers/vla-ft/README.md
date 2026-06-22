# containers/vla-ft — generalized LeRobot VLA fine-tune container (Pattern B engine)

The IL-axis training engine, generalized from a verified π0.5 SageMaker fine-tune.
It is **backend-agnostic BYOC**: the same image runs under a
SageMaker Training Job (Pattern B) or AWS Batch (Pattern A). The platform's
`SharedBaseStack` provides the ECR repo (`pai/vla-ft`), data/artifact buckets, and
base IAM; `lib/il` provides the execution role and submits the job.

## What "generalize" means here (and what it does NOT touch)

The verified dependency **lock** (the image) is reused byte-identical — generalization
and feature work are only in the *source_dir code around it*, never the lock. The lock
is `docker/Dockerfile`; the `src/` entrypoint is shipped fresh each job (source_dir), so
adding opt-in features there needs **no image rebuild** and cannot drift the verified
stack.

| File | Status vs the original single-EC2 fine-tune | Note |
|---|---|---|
| `docker/Dockerfile` | **byte-identical** (sha256 verified) | the verified stack/lock; do not edit |
| `src/train.py` | **diverged** (early-stop layer added) | multi-policy + multi-GPU **+ opt-in early-stop / overfit guard** (see below). Default path (no early-stop flags) is behaviorally identical to the verified smoke. source_dir → no rebuild. |
| `launch.py` | **diverged** | platform binding (`--image-uri/--role/--output-s3`) + `--val-episodes/--select-best/--early-stop-*` flags |
| `build.sh` | **generalized** | default repo `pai/vla-ft` (was flat `vla-ft`); slash-safe IAM/CodeBuild names |

The container is **not pinned to π0.5**. `train.py`'s `POLICY_MAP` exposes act,
diffusion, vqbet, tdmpc, smolvla, pi0, pi05, pi0_fast, groot, xvla; policy /
pretrained-path / dataset / HP are all injected at launch. π0.5 is just the
verified smoke.

## The verified stack (do not drift)

`docker/Dockerfile` installs, on Ubuntu 24.04 / CUDA 12.8 / Python 3.12:
- `lerobot[pi,training] @ git+https://github.com/huggingface/lerobot.git@d1b1c5c8`
  (lerobot 0.5.2-dev — the exact commit the verified `openarm-lift-pi05` EC2 run used;
  registers `relative_actions_processor` which the pi05_base checkpoint references and
  which the 0.5.1 PyPI wheel lacks),
- the torch trio pinned to that run's `uv.lock`: **torch 2.11.0+cu128 /
  torchvision 0.26.0 / torchcodec 0.11.1** (ABI-matched — mismatching breaks
  libtorchcodec at the first video decode),
- `sagemaker-training` (BYOC contract: source_dir extraction, hyperparameters.json,
  artifact upload).

Smoke PASSED: job `vla-ft-pi05-20260614-160920` (steps=200, g5.4xlarge, On-Demand)
ran to Completed and produced `output/model.tar.gz` with the 7-file
`pretrained_model/`. Full provenance + the debug history that arrived at this lock
live in the git history.

## Multi-GPU (already built in)

`train.py:build_command` wraps lerobot in
`accelerate launch --multi_gpu --num_processes=N --num_machines=1 --mixed_precision=bf16`
when `num_gpus>1` (auto-detected from `SM_NUM_GPUS`, or `--num-gpus` override).
Single-GPU path is byte-identical to the verified smoke. `accelerate` is already in
the image's `[training]` extra → **no rebuild needed** for multi-GPU. lerobot does
not auto-scale steps/LR, so the caller holds effective batch (= `batch_size × N`)
and steps constant to keep the verified optimizer trajectory (e.g. batch 4 × 4 GPU =
16 = the smoke's effective batch).

## Early-stop / overfit guard (opt-in)

Two independent, opt-in levers. Both default OFF → a job that sets none of these flags
runs the verified path unchanged. lerobot has **no native early-stopping** (no
train/val split, no non-sim validation, no val-loss logging, no best-checkpoint — all
verified against commit `d1b1c5c8`), so this layer lives entirely in `train.py` and
**reuses lerobot's own factories — it does not fork lerobot or touch the image lock.**

**1. `--select-best` + `--val-episodes N` — overfit guard (the real early stopping).**
Holds out the **last N episodes** as a validation set (lerobot trains on episodes
`[0 .. total-N-1]` via the verified `--dataset.episodes` flag). After training, every
saved checkpoint is scored on those held-out episodes and the **lowest-val-loss one** is
staged as `model.tar.gz` — early stopping by checkpoint selection. Lower `--save-freq`
(e.g. `2000` with `--steps 20000` → 10 checkpoints) to give it a real choice. The
scorer reloads each checkpoint with lerobot's `TrainPipelineConfig.from_pretrained` +
`make_policy` + `make_pre_post_processors` (the exact config lerobot saved), runs a
no-grad `policy.forward` over the val set, and is **best-effort**: any failure falls
back to staging the latest checkpoint (training already succeeded — selection never
fails the job).

> The OpenArm set is only **50 episodes**, so a 5-episode held-out val-loss is noisy.
> Selecting the best of several checkpoints is deliberately more robust than trusting a
> single in-loop val number — and it pairs with `--train-expert-only` (freeze the 2.3B
> VLM, train only the ~0.3M action expert), which is the strongest small-data overfit
> guard and is the verified expert-only configuration.

**2. `--early-stop-patience P` [`--early-stop-min-delta D`] — converged-cost lever.**
Tails lerobot's `loss:X.XXX` train-loss log (emitted every `log_freq`=200 steps),
re-emitting every line so CloudWatch is unchanged. If train loss fails to improve by `D`
(default 0) for `P` consecutive log-points, sends the training subprocess **SIGTERM** so
lerobot stops after its current checkpoint, saving GPU time on a plateau. This watches
**train** loss (not val) — it's a cost lever, not the overfit guard.

```bash
# Overfit-guarded full FT: hold out 5 eps, keep 10 checkpoints, ship the best.
python launch.py --policy pi05 --pretrained-path lerobot/pi05_base \
  --dataset-s3 s3://.../lerobot_dataset/ \
  --instance-type ml.g6e.12xlarge --steps 20000 --batch-size 4 \
  --train-expert-only true --no-spot --region us-west-2 \
  --hf-token-ssm /pai/hf-token --hf-token-ssm-region us-east-1 \
  --image-uri <ECR>/pai/vla-ft:latest \
  --val-episodes 5 --save-freq 2000 --select-best true \
  --early-stop-patience 10            # optional: also stop early if train loss plateaus
```

## LoRA — fine-tune the FULL VLM on one GPU (the recommended big-model path)

`--lora` freezes the base model and trains only small low-rank adapters. This is
**lerobot-native** (`cfg.peft`, verified against commit `d1b1c5c8`) — **no fork, no
image-lock change** beyond adding the `peft` package to the image (the Dockerfile now
installs `lerobot[pi,training,peft]`; `peft` is inert unless a run sets `cfg.peft`).

**Why it matters.** Full-VLM **full** fine-tune of pi05 (2.3B VLM) OOMs on a 48 GB
L40S: the dominating term is the fp32 Adam **optimizer state** (~37 GB), not activations
— so more GPUs (DDP replicates) and smaller batches don't help. LoRA collapses that
optimizer state to a few MB (only the adapters have it), so the **whole VLM** fine-tunes
on a single L40S without OOM. It is the standard answer to "the model doesn't fit"; FSDP
(weight sharding) would require forking lerobot's checkpoint path and is not used here.

> QLoRA (4-bit frozen base) is **not available**: lerobot @ `d1b1c5c8` has no
> bitsandbytes / 4-bit load path, so it would need a fork. `--qlora` is rejected with a
> message pointing to `--lora` (the bf16-frozen base already fits one L40S for pi05).

**Flags** (all opt-in; omit `--lora` → verified path byte-identical):
- `--lora true` — enable (emits lerobot `--peft.method_type=LORA`). Mutually exclusive
  with `--train-expert-only` (both freeze the VLM); `train.py` errors if both are set.
- `--lora-r N` (lerobot default 16), `--lora-alpha M` (default = r; scaling = alpha/r).
- `--lora-target-modules '<regex>'` — **OMIT** to use the policy default. pi0/pi05 ship a
  built-in default (`modeling_pi05._get_default_peft_targets`) that adapts the **action
  expert's** q/v projections + the action/state projection MLPs — the right target for
  action fine-tuning. Pass a regex covering the gemma backbone layers only if you want to
  adapt the VLM itself.
- `--freeze-vision-encoder true` — **launcher-level power-user knob** (not on the
  `vla_ft_cli` / MCP front door, which exposes only the three modes above). Freezes just
  the SigLIP vision tower while the rest of a full-VLM fine-tune still trains — a middle
  ground between `--train-expert-only` and full-VLM. `train.py` honors it; the orchestrated
  path pins the verified default (`false`).

```bash
# Full-VLM LoRA fine-tune on ONE L40S — no OOM, no FSDP, no fork.
python launch.py --policy pi05 --pretrained-path lerobot/pi05_base \
  --dataset-s3 s3://.../lerobot_dataset/ \
  --instance-type ml.g6e.4xlarge --steps 20000 --batch-size 16 \
  --lora true --lora-r 32 --lora-alpha 64 \
  --no-spot --region us-west-2 \
  --hf-token-ssm /pai/hf-token --hf-token-ssm-region us-east-1 \
  --image-uri <ECR>/pai/vla-ft:latest
```

> **Checkpoint contract (important for the vla-hub handoff).** Under LoRA the saved
> `pretrained_model/` is **adapter-only** (`adapter_model.safetensors` +
> `adapter_config.json` + the policy `config.json`) — it does **not** contain the base
> weights. At load, lerobot (`make_policy`, `use_peft=true`) reads the adapter, resolves
> the base from `adapter_config.json`'s `base_model_name_or_path`, loads it, then overlays
> the adapter. So serving a LoRA checkpoint needs the **base reachable** (HF id
> `lerobot/pi05_base` or a mounted path). lerobot does NOT auto-merge; for a single
> self-contained artifact, merge with `PeftModel.merge_and_unload()` and re-save (not done
> automatically at this commit). A full-FT / expert-only checkpoint stays self-contained.

## Build + launch

```bash
# 1. Build + push to the platform ECR repo (CodeBuild by default).
./build.sh --region us-west-2
#   -> <ACCOUNT>.dkr.ecr.us-west-2.amazonaws.com/pai/vla-ft:latest

# 2. Submit a fine-tune (role + buckets come from the deployed platform stacks).
python launch.py \
  --policy pi05 --pretrained-path lerobot/pi05_base \
  --dataset-s3 s3://<pai-data-bucket>/openarm-lift/lerobot_dataset/ \
  --instance-type ml.g6e.12xlarge --steps 20000 --batch-size 4 \
  --no-spot --region us-west-2 \
  --hf-token-ssm /pai/hf-token --hf-token-ssm-region us-east-1 \
  --role <PatternB exec-role ARN from stack output> \
  --output-s3 s3://<pai-artifacts-bucket>/vla-ft \
  --image-uri <ACCOUNT>.dkr.ecr.us-west-2.amazonaws.com/pai/vla-ft:latest
```

- Dataset S3 URI must be a **LeRobot v3 dataset root** (`meta/info.json`); SageMaker
  downloads it to `/opt/ml/input/data/training/`.
- π0.5's PaliGemma backbone is gated → HF token required (`--hf-token-ssm` reads an
  SSM SecureString param, e.g. `/pai/hf-token`, optionally in another region).
- Full-VLM **full** FT does NOT fit a 48 GB L40S (fp32 Adam state ~37 GB → OOM). To
  fine-tune the full VLM, use **`--lora true`** (collapses the optimizer state; fits one
  L40S — see the LoRA section). `--train-expert-only true` (~24 GB, freezes the VLM) is
  the lighter alternative and fits A10G (g5).

## Outputs / vla-hub handoff

```
s3://<output>/<job>/checkpoints/          # live sync (Managed-Spot resumes here)
s3://<output>/<job>/output/model.tar.gz   # final LeRobot pretrained_model/
```

`vla-hub`'s `serve.py` loads from `MODEL_CHECKPOINT_DIR`, so serving a fine-tuned
policy is mount + env override — no serve.py change.
