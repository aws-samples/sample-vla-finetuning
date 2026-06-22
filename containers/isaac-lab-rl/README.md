# containers/isaac-lab-rl — Isaac Lab headless RL training container (RL Pattern A)

The RL-axis training engine: NVIDIA Isaac Lab **headless PPO** in simulation. Absorbed
from a verified Isaac Lab DCV + workshop infra template, but it
fills the gap that source left open — the source builds the *infra* (IAM / launch
template / SG) and runs training **interactively in a DCV desktop**, with the AWS Batch
Compute Environment / Job Queue / Job Definition documented as **manual console steps**.
This container + `lib/rl` automate that Batch path end to end.

The "intent" of an RL job is the **task id**, not a dataset: the environment and reward
are registered to the Isaac Lab task, and the simulator generates all experience on the
GPU. The reference task is `Isaac-Velocity-Rough-H1-v0` (Unitree H1 rough-terrain
locomotion, proprioception-only PPO, a few-MB MLP — sim-to-real transferable).

## Why rsl_rl (not skrl)

`Isaac-Velocity-Rough-H1-v0` registers **both** `rsl_rl_cfg_entry_point`
(`H1RoughPPORunnerCfg`, experiment `h1_rough`) and `skrl_cfg_entry_point`. We use
**rsl_rl** because it is the only Isaac Lab workflow with a **built-in ONNX export**:
`isaaclab_rl.rsl_rl.export_policy_as_onnx`, called by `rsl_rl/play.py`, writes
`<checkpoint_dir>/exported/policy.onnx` + `policy.pt`. The SKRL `play.py` has no ONNX
export. ROADMAP Phase 3's gate is "a trained policy comes out" with a deployable
artifact, so rsl_rl is the low-risk route. (Verified verbatim against the Isaac Lab
**v2.3.2** tag — see the session note for the source citations.)

## The verified stack (do not drift)

`docker/Dockerfile` builds on **Isaac Sim 4.5.0** + **Isaac Lab v2.3.2**:
- `FROM nvcr.io/nvidia/isaac-sim:4.5.0` — the NVIDIA Isaac Sim runtime (the same base
  the Isaac Lab workshop assets use).
- Clones `github.com/isaac-sim/IsaacLab` at tag `v2.3.2` (the README "Isaac Sim Version
  Dependency" table maps `v2.3.X` → Isaac Sim 4.5/5.0/5.1; v2.3.2 is the latest v2.3).
- `./isaaclab.sh -i` installs **all** RL frameworks by default (rsl_rl, skrl, sb3,
  rl_games, robomimic), so rsl_rl needs no separate install.
- Headless/root env: `ACCEPT_EULA=Y`, `PRIVACY_CONSENT=Y` (doc-cited), plus the
  NVIDIA Dockerfile-convention `OMNI_KIT_ALLOW_ROOT=1` / `NVIDIA_DRIVER_CAPABILITIES=all`.

The image is **task-generic** — the task id and all training knobs are injected at
submit time. Like vla-ft's Pattern A, the Batch entrypoint (`rl_train_bootstrap.py`) is
**not baked in**; the `lib/rl` Job Definition reads it at synth and injects it via
`python3 -c "<src>"`, so the image is never rebuilt to carry orchestration glue.


> **NGC pull note**: `nvcr.io/nvidia/isaac-sim:4.5.0` may require NGC credentials. If a
> build fails at the `FROM` pull, configure CodeBuild / local docker with an NGC API
> key (`docker login nvcr.io -u '$oauthtoken'`).

## Files

| File | Role |
|---|---|
| `docker/Dockerfile` | the verified RL stack (Isaac Sim 4.5.0 + Isaac Lab v2.3.2 + all RL frameworks) |
| `rl_train_bootstrap.py` | Batch glue: wire `logs/` → EFS (resume), run rsl_rl headless PPO, run `play.py` ONNX export, upload run dir → S3. stdlib + boto3 only; injected via `python3 -c`. |
| `rl_launch.py` | Batch `SubmitJob` wrapper — sets the `RL_*` container-override env the bootstrap reads. All queue/def/output values come from `RlPatternAStack` outputs. |
| `build.sh` | build + push to ECR `pai/isaac-lab-rl` (CodeBuild default, `--local` fallback). |

## The verified command forms (Isaac Lab v2.3.2)

```bash
# train (single-GPU, headless rsl_rl PPO)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-H1-v0 --headless [--num_envs N] [--max_iterations M] [--seed S] [--resume]

# train (multi-GPU, single node)
./isaaclab.sh -p -m torch.distributed.run --nnodes=1 --nproc_per_node=K \
  scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Rough-H1-v0 --headless --distributed

# export policy.pt + policy.onnx (writes <ckpt_dir>/exported/)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Rough-H1-v0 --headless --num_envs 32 --checkpoint <run>/model_<N>.pt
```

Checkpoints land at `logs/rsl_rl/h1_rough/<run>/model_<N>.pt`; the export writes
`logs/rsl_rl/h1_rough/<run>/exported/policy.onnx`. `rl_train_bootstrap.py` symlinks
`logs/` onto EFS so a Spot reclaim + Batch retry resumes (`--resume`) from the last
checkpoint, then uploads the whole run dir to `<output-s3>/<job>/output/`.

## Build + run

```bash
# 1. build the image (after SharedBaseStack created pai/isaac-lab-rl, or standalone)
./build.sh --region us-west-2

# 2. deploy the RL Pattern A stack (lib/rl/pattern-a-stack.ts), then submit:
python rl_launch.py \
  --task Isaac-Velocity-Rough-H1-v0 --max-iterations 3000 --num-envs 4096 \
  --job-queue      <RlPatternA JobQueueArn output> \
  --job-definition <RlPatternA JobDefinitionArn output> \
  --output-s3      <RlPatternA OutputS3Hint output>
```

## Status / caveats

- **Code-complete; real GPU run deferred.** The CDK stack + container + launcher
  synthesize and `py_compile`/`tsc`/`jest` clean. The first real RL run (ROADMAP Phase 3
  gate: "a reward+task goes in, a trained policy + ONNX comes out") needs GPU capacity
  and is the next real-GPU priority.
- The headless rsl_rl + play.py commands are the documented Isaac Lab v2.3.x forms,
  verified verbatim against the v2.3.2 tag, but have not yet executed under Batch in
  this account — treat the first run as an E2E smoke (like the IL Pattern A first run).
