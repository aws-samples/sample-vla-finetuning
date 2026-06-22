# vla-ft — Roadmap

Build order and **verification level per phase**. Verification policy (user-set):
A and B get real end-to-end runs; C is code + `cdk synth` only (real multi-node
deploy deferred); RL gets one verified reference task at A/B level.

Legend: ☐ todo · ◐ in progress · ☑ done · ⊘ deferred-by-design

> The phase numbers below are the **build-dependency order** (shared base → IL → RL →
> scale layers). The **priority order** we actually work in is reframed to the
> north-star (easiest + most efficient VLA FT, IL & RL) — see next block.

---

## Reframed priority order (2026-06-15)

After the IL axis landed (A/B/C built) the work was reframed against the four-axis
wedge (Ease / Efficiency / Coverage / Backend-flexibility). Priority order:

| # | Reframed step | Axis | Maps to phase | Cost |
|---|---------------|------|---------------|------|
| 1 | **RL realized** — absorb Isaac Lab Batch PPO + 1 reference task | Coverage | **Phase 3** ◐ code built, real run pending | real GPU |
| 2 | **Unified launcher + smart defaults** — auto instance/backend pick, Spot default, pre-flight, quickstart | Ease | **Phase 6** ✅ built | 0 |
| 3 | **Pre-launch cost estimate + early-stop/expert-only defaulted** — turn built levers into default UX | Efficiency | **Phase 6** ✅ built | 0 |
| 4 | **Pattern B real deploy + 1 FT** — A is real-E2E; B is code-complete | Proof | **Phase 2b** | real GPU |
| 5 | **Positioning** — docs reflect north-star + `vla-ft` rebrand sweep | Positioning | **done** ✅ | 0 |

> Phase 6 (#2–3) shipped as `containers/vla-ft/vla_ft_cli.py` + `vla_ft_decide.py`
> (dry-run validated against live AWS; live `--yes` submit validated on the
> `vla-ft-pi05-20260616-014126` run).
> #1 RL is now **code-complete** (`containers/isaac-lab-rl/` + `lib/rl/pattern-a-stack.ts`,
> synth + jest green); the remaining real-GPU work is the first RL run + #4 Pattern B live FT.

> Note: the deterministic **Step Functions orchestrator** (Phase 4) is an
> **optional, scale-gated graduation layer**, not the headline — the backend-select
> logic ships in the launcher first (Phase 6). It is now **built (code + synth)**:
> both the launcher and the orchestrator import the *same* `vla_ft_decide` module, so the
> decision logic lives in one place. Deploy-gated until multi-job/multi-user scale is
> real; first live run deferred to the next GPU spend. See ARCHITECTURE §3.

---

## Phase 0 — Design  ☑

- ☑ Asset inventory (IL + RL + ingest) — done via a prior-art sweep.
- ☑ `README.md`, `docs/ARCHITECTURE.md`, `docs/ROADMAP.md`.
- ☑ Design of record framed: the platform is the IL **and** RL axes of a single
  VLA fine-tuning best-practice, with a deterministic orchestrator. Inventory/gap
  analysis preserved as the IL-axis rationale.

*Cost: ~0. Output: design of record. **Phase 0 complete.***

## Phase 1 — Shared CDK skeleton  ☑

- ☑ `bin/app.ts` + CDK project bootstrap (TypeScript: `cdk.json`, `package.json`,
  `tsconfig.json`, `jest.config.js`). CDK 2.180 / TS 5.7 / ts-node.
- ☑ `lib/shared/base-stack.ts` (`SharedBaseStack`, L2 constructs): VPC (2 AZ, NAT,
  S3 gateway endpoint) · EFS (encrypted, RETAIN) · 2 ECR repos (`pai/vla-ft`,
  `pai/isaac-lab-rl`, scan-on-push, untagged-expiry) · 2 S3 buckets (data + artifacts,
  RETAIN + versioned + SSE + public-blocked) · `jobBasePolicy` managed policy.
- ☑ `lib/shared/az-selector.ts` (**AzSelector** extracted from a verified Isaac Lab
  infra template, generalized to g6e→g5 fallback). Opt-in: base stack
  places no custom resource, so base synth needs no credentials/probe AMI.
- ☑ `cdk synth PaiTrainingPlatform-Base` green; `tsc --noEmit` clean; `jest` 8/8
  (incl. AzSelector construct unit-check).

*Cost: 0 (synth only). Gate: synth passes, AzSelector construct unit-checks. **✅ met.***

## Phase 2 — IL axis (the real-E2E core)  ◐

- ☑ **2a Pattern A** — AWS Batch + g6e Spot stack (adapt sample-embodied-ai-platform;
  port Batch construct to TS; **automate CE/JQ/JD** which the source left manual).
  **Code + real E2E both done.**
  - ☑ `lib/il/pattern-a-stack.ts` (`PatternAStack`): managed EC2 **Spot** CE
    (`SPOT_PRICE_CAPACITY_OPTIMIZED`, single-GPU g6e.4xl→g5.4xl fallback, scale-to-0)
    + JobQueue + GPU JobDefinition, all in CDK — the CE/JQ/JD both sources left
    manual, now automated. Reuses `base.vlaFtRepo` (same verified image as Pattern B),
    `base.fileSystem` (EFS checkpoints → Spot-reclaim resume), `base.jobBasePolicy`.
    No AzSelector (Spot CE is Batch's native capacity strategy).
  - ☑ Backend portability glue: `containers/vla-ft/batch_bootstrap.py` (stdlib+boto3,
    injected via `python3 -c` at synth → image byte-identical) stages dataset S3→local,
    fetches unchanged `train.py` from S3, runs it, uploads model in SageMaker's exact
    output layout. train.py needed **zero** changes (its `SM_HP_*` env fallback +
    env-overridable paths already cover Batch).
  - ☑ `containers/vla-ft/batch_launch.py` (Batch counterpart of launch.py: uploads
    train.py to S3, submits with `SM_HP_*` + dataset/output/HF-token container overrides).
  - ☑ `TrainingNotifications.addBatchJobRule()` (Batch job-state-change → same SNS topic).
  - ☑ `bin/app.ts` registers `PaiTrainingPlatform-IL-PatternA`. Gate: `tsc` clean ·
    `cdk synth` 3-stack green · `jest` 34/34 (+14) · `py_compile` clean. Cost 0.
  - ☑ **real deploy + one real fine-tune** (2026-06-15, us-west-2).
    `cdk deploy PatternA` → CE/JQ/JD live (Spot CE idle=0 vCPU). Validation job
    `vla-ft-pi05-20260615-155834` (g6e.4xl Spot, pi05 expert-only, steps=200):
    RUNNABLE→STARTING→RUNNING in ~6 min (Spot capacity ~2.5 min + 9.8GB image pull),
    **SUCCEEDED** in ~12.7 min RUNNING (~\$0.20 Spot). Proved the full backend-portability
    path: batch_bootstrap synced dataset S3→local + fetched train.py from S3 → the
    **unchanged train.py** ran under Batch (SM_HP_* env, same lerobot command as
    Pattern B) → loss 0.232 → staged 7 files → uploaded to
    `s3://pai-artifacts-.../vla-ft/<job>/output/pretrained_model/` (the flat 7-file
    layout, model.safetensors 8.7GB — same the sim eval consumes, no tar). Spot
    instance auto-terminated on idle (no lingering GPU cost). **2a gate met: dataset in,
    checkpoint out, same image as B.** (Pre-deploy em-dash IAM bug fixed: `c6e6a97`.)
- ◐ **2b Pattern B** — SM Training Job stack wrapping the **generalized `vla-ft`
  container**.
  - ☑ Absorb the verified single-EC2 fine-tune engine → `containers/vla-ft`: `docker/Dockerfile`
    **byte-identical** (sha256 verified — this is the verified dependency lock,
    must not drift). `src/train.py` was absorbed
    byte-identical, but has since gained the **opt-in early-stop feature**
    (`6c2e1bd`: `--select-best`/`--val-episodes` + `--early-stop-patience`), so it
    now diverges by ~296 net lines (354→650); `launch.py` gained the 4 matching
    flags. The early-stop additions are **default OFF**, so the verification path is
    behavior-identical to the verified smoke. `build.sh` generalized to the
    `pai/vla-ft` ECR repo (slash-safe IAM/CodeBuild names). train.py is already
    multi-policy (10 LeRobot types) + multi-GPU (`accelerate launch`), so no code
    change was needed for backend generalization itself.
  - ☑ `lib/il/pattern-b-stack.ts` (`PatternBStack`): SageMaker execution role
    (trusted by sagemaker.amazonaws.com) attaching the base `jobBasePolicy` +
    least-privilege CloudWatch Logs/metrics; reads the verified openarm-lift dataset
    bucket directly via `extraDatasetReadArns`. No VPC/AzSelector at this layer —
    Pattern B runs on SageMaker-managed infra. Outputs role ARN / image URI / output
    S3 for launch.py. `tsc` clean · `cdk synth` green (2 stacks) · `jest` 15/15.
  - ☐ **real deploy + one real fine-tune** (reuses the g6e quota already secured).
- ☐ **2c Ingest** — `hdf5_to_lerobot.py` generalized + event-driven trigger
  (manifest → S3 Event → converter). video→action adapter slot stubbed + clearly
  marked unimplemented.
- ◐ **2d GR00T N1.7 engine** — a **second IL model family** (NVIDIA GR00T N1.7, a 3B
  Cosmos VLA), the third IL training engine alongside the lerobot-π0.5 Pattern A/B.
  Delivers the WS1 Unitree-G1 adapter fine-tune ("a G1 + GR00T N1.7 motion sim video").
  **Code + synth done; image build + GPU run pending.**
  - ☑ `containers/gr00t-n17/` — `docker/Dockerfile` matches upstream's own install
    (Isaac-GR00T @ `65cc4a192e6d`, `uv sync --frozen` from the committed `uv.lock`,
    **not** hand-pinned); `gr00t_train_bootstrap.py`
    (sync dataset S3→writable local → **mandatory** `stats.py` [meta/stats.json +
    relative_stats.json for G1 RELATIVE left/right_arm] → `launch_finetune.py`
    frozen-backbone under a fail-fast liveness guard → upload merged 3B ckpt);
    `gr00t_launch.py` (`GROOT_*` contract); `build.sh` → `pai/gr00t-n17`.
  - ☑ `lib/il/gr00t-pattern-a-stack.ts` (`GrootPatternAStack`): Batch CE/JQ/JD on
    **g6e.4xlarge 1×L40S (no g5 fallback** — A10G OOMs a 3B model; frozen-backbone peak
    ~35 GB), **On-Demand default** (no EFS-resume → reclaim restarts), no EFS mount
    (ckpts on 300 GB local root), zlib bootstrap inject (<7000 B), liveness retry
    42→EXIT, gr00t-g1 bucket read IAM. `bin/app.ts` registers it (`-c grootUseSpot`).
  - ☑ Liveness guard ported from RL but **GR00T-verified** signals (NOT rsl_rl's): HF
    Trainer loss-dict (`logging_steps=10`) + stderr `Starting training` (2>&1) +
    `checkpoint-<N>/`. Gate: `tsc` · `jest` 87/87 (+17) · `cdk synth` 5-stack · py guard
    15/15 · `py_compile` · IAM descriptions ASCII-only.
  - ☐ **image CodeBuild → smoke (small `--max-steps`, ckpt format) → full G1 FT** (GPU,
    user confirm). Output = full merged 3B HF ckpt; downstream loads via
    `run_gr00t_server.py --embodiment-tag UNITREE_G1`.

*Cost: real GPU runs (A ~\$4, B ~\$96-scale, GR00T smoke ~\$2-5 + full FT ~\$5-20). Gate:
a dataset goes in, a checkpoint comes out, for A/B/GR00T. **2b/2d code complete (cost 0);
real runs pending.***

## Phase 3 — RL axis  ◐ code built, real run pending

- ☑ Absorb a verified Isaac Lab Batch headless-PPO infra template into `lib/rl` +
  `containers/isaac-lab-rl`. **Automate Batch CE/JQ/JD** — the exact CE/JQ/JD the source
  left as manual console steps (it automated only IAM / launch-template / SG). Done as
  `lib/rl/pattern-a-stack.ts` (`RlPatternAStack`), structurally mirroring the verified IL
  `PatternAStack`: managed Spot CE (g6e→g5 single-GPU fallback), JQ, GPU JobDef, EFS
  mount, isaac-lab-rl image, opt-in notifications.
- ☑ Reference RL task wired: **`Isaac-Velocity-Rough-H1-v0`** (Unitree H1 rough-terrain
  locomotion, PPO). Container = `containers/isaac-lab-rl/` (Isaac Sim 4.5.0 + Isaac Lab
  **v2.3.2**, all RL frameworks via `./isaaclab.sh -i`). Trainer = the **rsl_rl** workflow
  (not skrl) because only rsl_rl ships a built-in ONNX export
  (`isaaclab_rl.rsl_rl.export_policy_as_onnx` via `rsl_rl/play.py` → `exported/policy.onnx`
  + `policy.pt`). The H1 task registers both entry points, so rsl_rl is valid. Commands
  verified verbatim against the v2.3.2 tag.
- ☑ The RL **"intent" = the task id** (env + reward are registered to the task), plus
  optional Hydra-style reward/task overrides (`rl_launch.py --override key=value`) and
  iteration/env knobs. `rl_train_bootstrap.py` (Batch glue, injected via `python3 -c`)
  wires `logs/` → EFS (Spot-reclaim resume), runs headless PPO, runs play.py for ONNX,
  uploads the run dir → S3.
- ☐ **Real deploy + one real RL run** producing a policy + ONNX (the GPU gate). Treat the
  first run as an E2E smoke (the commands are documented-verified but unexecuted under
  Batch in this account). NGC pull credentials for `nvcr.io/nvidia/isaac-sim:4.5.0` may be
  needed at image-build time.

*Cost: 0 to build (synth + jest green, py_compile clean); real GPU for the first run.
Gate: a reward+task goes in, a trained policy + ONNX comes out. **Code complete; real
run pending.***

## Phase 4 — Deterministic orchestrator  ◐ code + synth built, real run deferred (scale-gated)

**Repositioned (2026-06-15):** no longer the headline. The classify→profile→decide
logic ships first in the **launcher** (Phase 6) as a smart default. This phase promotes
that *same* logic into a Step Functions state machine **only when multi-job/multi-user
scale justifies the added infrastructure** (durable retries, audit history, async
fan-out). See ARCHITECTURE §3.

- ☑ `lib/orchestrator/orchestrator-stack.ts`: Step Functions state machine + two Python
  3.12 Lambdas. **Plan** (`containers/vla-ft/orchestrator_plan.py`) = classify → profile
  → decide; it imports `vla_ft_decide` **verbatim** (the SAME rule-table module the
  Phase 6 launcher uses — zero re-implementation of the decision logic). **Submit**
  (`orchestrator_submit.py`) dispatches the
  backend faithfully to the verified launchers: IL/RL **Batch** submitted directly via
  boto3 (mirrors `batch_launch.py`/`rl_launch.py` on the container-override contract);
  **SageMaker (B)** handed off as the exact `launch.py` command (no SDK-estimator fork).
  A `Choice` on the plan's deterministic `runnable` flag routes Pattern C (code+synth
  only) to a recommendation; SNS publishes the outcome (reuses `TrainingNotifications`).
  HF token read in-Lambda + injected as job env (never into SFN execution history).
  `vla_ft_decide.py` gained the RL rule table (`profile_rl`/`decide_rl`) + `classify()`
  additively — the IL path and the Phase 6 launcher are byte-identical (30/30 unchanged).
- ☑ **Deploy-gated** (NOT in `bin/app.ts`, like the HyperPod stacks), so `test/
  orchestrator.test.ts` (10 tests) IS the synth gate. Gate: `tsc` clean · `jest` 66/66
  (+10) · `cdk synth` 4-stack green (orchestrator gated) · `py_compile` clean ·
  `test_vla_ft_decide.py` 48/48 (+18: classify, RL decide, plan, submit-handoff
  faithfulness). IAM descriptions + template ASCII-only (guarded by a test). Cost 0.
- ☐ End-to-end on GPU: one input intent → correct backend chosen → job submitted (one
  live IL and one live RL through the full state machine). Deferred to the next GPU spend
  (reuses Phase 2b/3 runs); the plan/submit logic is unit-verified, the live path is not.

*Gate: same input → same decision, reproducibly; live path for both axes. **Code + synth
done; deployed only when launcher-level orchestration is outgrown.***

## Phase 5 — Pattern C (HyperPod), pre-built  ◐ real-deploy deferred

- ☑ `lib/shared/hyperpod-cluster.ts` (`HyperPodCluster`): shared construct wrapping
  `sagemaker.CfnCluster` — Slurm orchestrator (self-contained at synth; EKS needs an
  external cluster ARN), instance groups, lifecycle config (`s3://sagemaker-` prefix),
  cluster role (`AmazonSageMakerClusterInstanceRolePolicy` + base `jobBasePolicy`),
  self-referencing SG for NCCL, `NodeRecovery: Automatic`. IL and RL differ only in
  instance-group sizing + image/lifecycle scripts, so the cluster is ONE construct.

- ☑ `lib/il/hyperpod-stack.ts` (`IlHyperPodStack`, default 2× ml.g6e.48xlarge) +
  `lib/rl/hyperpod-stack.ts` (`RlHyperPodStack`, default 2× ml.g6.12xlarge). Thin
  wrappers over the shared construct. Copy-paste base:
  `aws-samples/awsome-distributed-training` openvla (Slurm) / openvla-oft (EKS).
- ☑ `cdk synth` green — proven by `test/hyperpod.test.ts` (8 tests, both stacks synth
  to valid `AWS::SageMaker::Cluster`). The stacks are intentionally **NOT** in
  `bin/app.ts` (deploy gated), so the test IS the synth gate. `jest` 42/42 total.
- ⊘ Real multi-node deploy deferred (1-day cluster commit; g-series *cluster* quotas
  are 0 — open question). Access confirmed (`list-clusters` reachable).

*Cost: 0 (synth only). Gate: synth passes; ready to deploy when a multi-node job
justifies the cluster commit. **✅ met (code + synth).***

## Phase 6 — Ease + Efficiency: smart-default launcher  ☑ built (real-run validation pending)  (reframed priority #2–3)

The Ease and Efficiency axes (§6 of ARCHITECTURE). The patterns A/B/C and the
efficiency *levers* (AzSelector, Spot CE, early-stop, expert-only) already existed; this
phase turns them into a **one-command UX with smart defaults**, so the user never
hand-writes a launcher or picks an instance.

Built as **two thin Python files in `containers/vla-ft/` that call the UNCHANGED verified
launchers** (`launch.py` / `batch_launch.py` stay byte-identical — verified-lock safe):
- `vla_ft_decide.py` — the rule table as one **pure, stdlib-only** module (instance
  catalog + $/hr anchors, model registry, `profile()` + `decide()`). This is the single
  source Phase 4 would promote into Step Functions verbatim — no I/O, no re-implementation.
- `vla_ft_cli.py` — the one-command wrapper: profile → decide → resolve CFN outputs →
  pre-flight → cost estimate → subprocess-call the verified launcher.

- ☑ **Unified launcher** (`vla_ft_cli.py`): given `--dataset` + `--model` (IL) **or**
  `--task` (RL via `--intent rl`), auto-picks Pattern A/B/C **and** the instance type from
  the §3.1 rule table — a **smart default the user overrides** (`--backend`,
  `--instance-type`, `--full-vlm`, `--lora`; RL: `--num-envs`, `--max-iterations`,
  `--num-gpus`). **RL is now wired end-to-end** (`--intent rl` → `profile_rl`/`decide_rl`
  → resolve `RlPatternAStack` outputs → pre-flight → hand off to the unchanged
  `rl_launch.py`); dry-run-validated against the live RL stack. The decision lives in the
  one `vla_ft_decide.py` config module — IL & RL share it (the orchestrator imports the
  same module). This is the literal "easiest VLA FT, IL **and** RL" wedge.
- ☑ **Pre-flight checks**: HF token (SSM `get_parameter`, hard gate for gated pi-family
  backbones) + live capacity (free Spot Placement Score signal, advisory). Self-contained
  in boto3 (the BILLABLE ODCR ground-truth stays the separate
  `scripts/aws-gpu-region-probe --verify`). Service-Quotas check left as a follow-up
  (capacity SPS already surfaces the practical blocker).
- ☑ **Pre-launch cost estimate**: instance $/hr (real us-west-2 OD anchors: EC2 for A,
  SageMaker ML for B) × estimated wall-clock (anchored to the verified run:
  20000 steps / 19819 s on g6e.12xl L40S ≈ 1 s/step), Spot vs On-Demand, printed before
  submit. The OD estimate reproduces the verified run's real ~$72.
- ☑ **Efficient defaults**: **Spot on by default**; pi-family **expert-only by default**
  (matches the verified lock, fits one L40S, resists overfit) — `--full-vlm` opts out;
  `--select-best` wires the overfit-guard trio. Default OFF paths stay byte-identical.
- ☑ **Quickstart**: `python vla_ft_cli.py --quickstart [--yes]` runs end-to-end on the
  verified openarm-lift dataset (documented in README).
- ☑ **Gate met (cost 0)**: `py_compile` clean; `test_vla_ft_decide.py` **56/56** (rule
  table, overrides, cost anchor, budget, IL **and** RL CLI argv vs the unchanged launchers,
  orchestrator faithfulness); **real `--dry-run` against live CFN outputs** resolved
  role/image/queue, ran HF-token + SPS pre-flight, and printed the cost estimate — for
  Pattern A, Pattern B, an 8-GPU override, a tight-budget warning, **and the RL intent
  against the live `RlPatternAStack`** (single-GPU + 4-GPU + hyperpod-override routing).
  The launchers + Dockerfile + train.py verified unchanged.
- ☐ **Real-run validation**: one `--yes` submit (reuses Phase 2b's GPU spend) to confirm
  the quickstart launches end-to-end (dry-run proven; live submit pending the next GPU run).

*Cost: 0 to build (a real run to validate the quickstart reuses Phase 2b's GPU spend).
Gate: one command, dataset+model in → correct backend/instance auto-picked, cost shown,
job submitted — no hand-written launcher. **✅ met at the dry-run level; live submit pending.***

---

## Verification matrix

| Phase | Pattern | Verification |
|---|---|---|
| 2a | IL Batch (A) | real deploy + real fine-tune |
| 2b | IL SM Training Job (B) | real deploy + real fine-tune |
| 2d | IL GR00T N1.7 (Batch A) | ☑ `cdk synth` 5-stack + jest 87/87 + py guard 15/15; image build + smoke + full G1 FT pending |
| 3 | RL Batch (A/B) | real deploy + real RL run (1 reference task) |
| 4 | Orchestrator (Step Functions) | `cdk synth` + jest + py asserts; live IL/RL execution deferred to next GPU spend |
| 5 | HyperPod (C, IL & RL) | `cdk synth` only — real deploy deferred |
| 6 | Launcher UX (Ease/Efficiency) | ☑ `test_vla_ft_decide.py` 30/30 + real dry-run vs live CFN; live `--yes` submit pending |
| ingest | video→action | **unimplemented**, slot stubbed |

## Carried-over assets / quotas

- g6e training-job quota secured: g6e.4xl=4, g6e.12xl=2, **g6e.48xl=1 (8×L40S)**.
  p5.48xl=1 CASE_OPENED. Available for Phase 2b/3 real runs.
- Verified `vla-ft` container in ECR `vla-ft:latest` (π0.5 smoke PASSED
  job=`vla-ft-pi05-20260614-160920`). Source of `containers/vla-ft`.
- 8-way full-FT launch (g6e.48xl, batch 2, eff-batch 16) **on hold** until the
  platform absorbs the container (Phase 2b), then runnable through the platform.
