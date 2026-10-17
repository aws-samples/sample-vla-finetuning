# containers/gr00t-n17 — GR00T N1.7 fine-tune container (IL GR00T Pattern A)

The platform's third IL training engine, alongside `vla-ft` (lerobot π0.5) and
`isaac-lab-rl`. It fine-tunes **NVIDIA GR00T N1.7** (a 3B VLA on a Cosmos-Reason2-2B
backbone) on a LeRobot v2.1 dataset and emits a **full merged HF checkpoint** that
`gr00t/eval/run_gr00t_server.py` loads for sim rollout. Built to deliver the WS1 G1
adapter fine-tune ("a G1 + GR00T N1.7 motion sim video").

The "intent" of a GR00T job is a **dataset + an embodiment tag** (like Pattern B), not a
lerobot policy. The reference run fine-tunes the registered `UNITREE_G1` embodiment (a
posttrain tag — not in the base checkpoint, so a fine-tune is required) on the G1
cloudwalk-teacher LeRobot dataset.

## The verified stack (do not drift)

`docker/Dockerfile` matches upstream's own install **exactly** — the M1 lesson was that
hand-assembling a dependency stack that drifts from the source's lock causes repeated
build failures.

- Source: `github.com/NVIDIA/Isaac-GR00T` @ **`65cc4a192e6d`** (N1.7 Early Access).
- `FROM nvidia/cuda:12.8.0-devel-ubuntu22.04` (upstream's base) · Python **==3.10.\***.
- Clones the repo at the pinned commit, then runs upstream's two-stage install:
  `uv sync --frozen --no-install-project --extra dev` (deps from the committed `uv.lock`;
  `--extra dev` pulls **boto3**, the bootstrap's only non-stdlib need) → `uv pip install
  -e . --no-deps` (the gr00t package). **No hand-pinned pip installs.** flash-attn comes
  from the pinned GitHub-release wheel declared in pyproject's `[tool.uv.sources]` — no
  source build. Locked pins: `torch==2.7.1`, `transformers==4.57.3`, `diffusers==0.35.1`,
  `peft==0.17.1`, `flash-attn==2.7.4.post1`.
- The base CUDA image has no streaming ENTRYPOINT (unlike the isaac-sim base the RL
  container resets with `ENTRYPOINT []`), so the `python3 -c` injection runs directly —
  same as vla-ft Pattern A on the same base.

The image is **task-generic** — dataset, steps, embodiment, and all knobs are injected at
submit time. The Batch entrypoint (`gr00t_train_bootstrap.py`) is **baked into the image**
(`COPY /workspace/gr00t_train_bootstrap.py`), and `lib/il/gr00t-pattern-a-stack.ts` runs it
with a short command (`python3 /workspace/gr00t_train_bootstrap.py`). It was formerly
injected **zlib-compressed** via `python3 -c "<stub>"`, but the AWR update-half grew the
packed bootstrap so that command + per-job AWR env overrides (Batch **sums** the command and
environment overrides at SubmitJob) exceeded the 8192-byte ECS container-override ceiling.
Baking it — the same move already used for `g1_finetune.py` / `gr00t_awr_resample.py` — drops
the command to ~40 bytes and restores the full 8192B budget for env overrides.

> **HF note**: `nvidia/GR00T-N1.7-3B` is an **ungated** ~6 GB HF model, downloaded at
> runtime. No HF token is required; `--hf-token-ssm` is an optional escape hatch if HF
> throttles anonymous pulls.

## GPU requirement (g6e L40S only — A10G OOMs)

GR00T N1.7's own hardware doc requires **40 GB+ VRAM**; the default frozen-backbone
fine-tune (tune_projector + tune_diffusion, LLM/visual frozen) peaks at **~35 GB/GPU**.
So `GrootPatternAStack` uses **g6e.4xlarge (1×L40S 48 GB) with NO g5 fallback** — A10G
24 GB would only schedule a guaranteed OOM. The CE is **On-Demand by default** (a single
multi-hour fine-tune with no EFS-resume; a Spot reclaim would restart from scratch).

## Files

| File | Role |
|---|---|
| `docker/Dockerfile` | the verified GR00T stack (Isaac-GR00T @ 65cc4a192e6d, `uv sync` from upstream's lock) |
| `docker/g1_finetune.py` | baked thin universal train entry. With `GROOT_ACTION_HORIZON` set it patches `config.model.action_horizon` (+ the model-config class, so the `from_pretrained`-built model's saved `config.json` also gets it) then re-runs `launch_finetune.py` via runpy; unset = byte-identical to launch_finetune. Needed for the UNITREE_G1 50-step horizon (no upstream flag). See action_horizon note below. |
| `docker/gr00t_awr_resample.py` | baked AWR/RWR episode resampler (the RL update-half). Invoked by the bootstrap ONLY when `GROOT_AWR_ADVANTAGES_S3` is set: upsamples high-advantage rollout episodes (multiplicity ∝ exp(adv/β)) in the synced LeRobot dataset before stats, then the byte-identical fine-tune runs on the reshaped data. Unset = never called. See AWR note below. stdlib + boto3 + pyarrow. |
| `docker/test_gr00t_awr_resample.py` | offline self-test for the resampler (pure AWR math + LeRobot v2.1 file surgery; no GPU/network). |
| `gr00t_train_bootstrap.py` | Batch glue: sync dataset S3→writable local, (optional AWR resample), generate `meta/stats.json` (+ `relative_stats.json`), run `launch_finetune.py` under a fail-fast liveness guard, upload the merged 3B ckpt → S3. stdlib + boto3; injected via `python3 -c`. |
| `gr00t_launch.py` | Batch `SubmitJob` wrapper — sets the `GROOT_*` container-override env the bootstrap reads. All queue/def/output values come from `GrootPatternAStack` outputs. |
| `test_gr00t_liveness_guard.py` | unit tests for the liveness guard (GR00T-verified signals). |
| `build.sh` | build + push to ECR `pai/gr00t-n17` (CodeBuild default, `--local` fallback). |

## The verified command forms (Isaac-GR00T @ 65cc4a192e6d)

```bash
# 0. stats (MANDATORY — the loader asserts meta/stats.json; G1 left_arm/right_arm are
#    RELATIVE so meta/relative_stats.json is needed too; writes both INTO the dataset)
python gr00t/data/stats.py --dataset-path <dataset> --embodiment-tag UNITREE_G1

# 1. fine-tune (single GPU, frozen-backbone = defaults: tune_projector + tune_diffusion)
CUDA_VISIBLE_DEVICES=0 python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B --dataset-path <dataset> \
  --embodiment-tag UNITREE_G1 --output-dir <out> \
  --max-steps 2000 --save-steps 1000 --global-batch-size 64

# multi-GPU (the trainer does NOT self-spawn torchrun)
torchrun --nproc_per_node=K gr00t/experiment/launch_finetune.py --num-gpus K ...
```

Checkpoints land at `<output-dir>/checkpoint-<N>/` (full merged 3B HF safetensors +
`experiment_cfg/` + processor). `gr00t_train_bootstrap.py` uploads the output dir to
`<output-s3>/<job>/output/`, and `run_gr00t_server.py --model-path <ckpt> --embodiment-tag
UNITREE_G1 --use-sim-policy-wrapper` loads any `checkpoint-<N>/` directly (the WS1 eval
contract).

## ★ action_horizon=50 (UNITREE_G1 full-body) — two-patch fix in `g1_finetune.py`

UNITREE_G1 full-body data uses `action delta_indices=range(50)` but the model config and the
base `GR00T-N1.7-3B` checkpoint both default `action_horizon=40`, and `launch_finetune.py`
exposes no flag to change it. `gr00t_launch.py --action-horizon 50` sets `GROOT_ACTION_HORIZON`
so the baked `g1_finetune.py` wrapper applies it. **GR00T sources `action_horizon` differently
for the processor vs the model, so the wrapper applies TWO patches** (verified @ 65cc4a192e6d):

1. **processor** — `setup.py::_create_dataset` reads `self.model_config.action_horizon` (the
   live config). Patch #1 = wrap `get_default_config` to set `cfg.model.action_horizon=50`.
2. **model** — `setup.py::_create_model` builds via `AutoModel.from_pretrained(base, ...)` whose
   override-kwarg list does NOT include `action_horizon`, so it rebuilds a fresh config from the
   base checkpoint's `config.json` (=40) onto `model.config`. Patch #1 cannot reach this. Patch
   #2 = wrap the model-config CLASS `__init__` to force `action_horizon=50` on every instance,
   incl. the `from_pretrained`-rebuilt one. action_horizon sizes NO weight, so this is weight-safe.

**Why both are required (the 2026-06-20 bug):** with only Patch #1, training ran at 50 (the
processor pads data to 50) but the saved `config.json`/`final_model_config.json` shipped
`action_horizon=40` (from `model.config`). At rollout the action head read 40 → emitted (1,40,7)
while the processor expected (50,7) → broadcast crash. The first full FT (`...103523`) had to be
hot-fixed by hand-editing the saved config 40→50 (`...103523-h50fix`). Patch #2 makes future G1
FTs save `config.json=50` directly. With `GROOT_ACTION_HORIZON` unset, neither patch fires.

## ★ AWR / RWR — advantage-weighted episode resampling (the RL update-half)

This image doubles as the **update-half of a decomposed online RL-FT loop** (RECAP/AWR family):
a separate rollout substrate (the eval AMI) runs the current policy in sim, scores each episode
with a reward-derived advantage, and writes `{a LeRobot dataset + advantages.json}` to S3. This
container then fine-tunes with those episodes **weighted by advantage**, pulling the policy
toward successful behaviour (e.g. the place step that imitation left at 0/10) without any RL
algorithm in the trainer.

**Why resampling, not per-transition loss weighting.** GR00T's action head reduces the
flow-matching loss to a scalar inside `model.forward`, `Gr00tTrainer.compute_loss` returns that
scalar, and the collator (`processing_gr00t_n1d7.py`) drops every non-modality key — so a
per-transition advantage can never reach the loss without a fragile multi-layer monkeypatch
that would only fail at GPU runtime (verified @65cc4a192e6d). Instead `gr00t_awr_resample.py`
materialises episode _i_ with multiplicity mᵢ ∝ exp((advᵢ−max)/β), then runs the
**byte-identical** verified fine-tune on the reshaped dataset. This is reward-weighted
regression at **episode** granularity — coarser than per-transition AWR, but faithful (zero
training-internal patching) and well matched to our sparse success/place signal.

**Env contract** (all optional; unset `GROOT_AWR_ADVANTAGES_S3` ⇒ the resampler is never
invoked ⇒ byte-identical to a plain GR00T fine-tune):

| env | meaning |
|---|---|
| `GROOT_AWR_ADVANTAGES_S3` | S3 uri to advantages JSON — `{"beta"?: β, "advantages": {"<episode_index>": adv, …}}` or a bare `{episode_index: adv}` map. Presence turns the run into AWR. |
| `GROOT_AWR_BETA` | temperature β (default 1.0; smaller ⇒ sharper upweighting). Precedence: CLI > env > JSON > 1.0. |
| `GROOT_AWR_MAX_DUP` | cap on copies per episode (default 8). |
| `GROOT_AWR_ALLOW_DROP` | set ⇒ episodes with adv < `GROOT_AWR_DROP_BELOW` are dropped (filtered BC). |
| `GROOT_AWR_DROP_BELOW` | drop threshold (default 0.0). |

The resampler physically duplicates each upsampled episode (rewriting the parquet
`episode_index` + global `index` columns, copying video files, appending `meta/episodes.jsonl`,
bumping `meta/info.json` counters) — no LeRobot loader constructor accepts an episode-list or
per-episode weight @65cc4a192e6d, so file surgery is the only faithful lever. Originals are
append-preserved (only `--allow-drop` rewrites `episodes.jsonl`). **1-shot gap:** the offline
test proves the reshaped dataset is self-consistent, not that the loader accepts it — that is
confirmed only by the first GPU run.

## Liveness guard (booted-but-idle → fast exit-42)

The guard (ported from the RL bootstrap, but with **GR00T-verified** signals — the RL
`Learning iteration` marker is rsl_rl-only) kills a trainer that booted but never started
learning. GR00T uses the HF transformers Trainer, which logs a per-step dict
`{'loss': ..., 'learning_rate': ...}` to stdout every `logging_steps` (GR00T sets 10) and
prints `Starting training...` to stderr before the loop; the guard merges stderr (`2>&1`)
and also watches for the first `checkpoint-<N>/` dir. If none appears within the deadline
(default 30 min; `--liveness-deadline`), it SIGTERMs the process group and exits 42 — the
JobDef maps 42 → `EXIT` (no retry; idle is deterministic).

## Build + run

```bash
# 1. build the image (after SharedBaseStack created pai/gr00t-n17, or standalone)
./build.sh --region us-west-2

# 2. deploy the stack, then SMOKE first (cheap; verifies checkpoint format), then full FT:
#    cd .. && cd .. && npx cdk deploy PaiTrainingPlatform-IL-GrootPatternA -c region=us-west-2
python gr00t_launch.py \
  --dataset-s3 s3://example-gr00t-g1-dataset/2026-06-17-0410/task-0/lerobot/ \
  --embodiment-tag UNITREE_G1 \
  --max-steps 60 --save-steps 50 \
  --job-queue <GrootPatternA JobQueueArn> --job-definition <GrootPatternA JobDefinitionArn> \
  --output-s3 <GrootPatternA OutputS3Hint>
# then re-run with --max-steps 2000 (full FT) after the smoke checkpoint format is confirmed.
```

## Status / caveats

- **Code-complete; real GPU run deferred** (build · synth · `tsc`/`jest`/`py_compile`
  clean). The image CodeBuild + a smoke (small `--max-steps`, checkpoint-format check)
  precede the full G1 adapter fine-tune — the order the order-form recommends.
- The train/stats command forms are the documented Isaac-GR00T forms, verified verbatim
  against `65cc4a192e6d`, but have not yet executed under Batch in this account — treat
  the first run as an E2E smoke (the lerobot LoRA-wrap-style GPU surprises only show on
  the first real run).
- **Output is a full merged 3B HF checkpoint** (not adapter-only), so the downstream
  `run_gr00t_server.py` load is self-contained — no base+adapter merge step needed.
