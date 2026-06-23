# vla-ft — Architecture

**North-star**: make VLA fine-tuning on AWS the **easiest and most efficient** it can
be, for **both IL and RL**. Everything below serves that goal; the wedge is four axes
(Ease / Efficiency / Coverage / Backend-flexibility — §6), not any single component.

**Status**: IL axis A/B/C built (A real-E2E verified, B real-FT verified, C synth).
Ease/Efficiency default-UX built (Phase 6, live-validated). RL axis Pattern A built
(code + synth + jest; first real GPU run pending), Pattern C synth. The first RL run and
Pattern B-scale RL are the next real-GPU work (see [`ROADMAP.md`](ROADMAP.md)).
Verified facts are marked; unverified assumptions are flagged `[확인 필요]`. The container
stack reused as the IL engine is fully verified (see §5 *Asset reuse*).

**Naming**: the product is **`vla-ft`**. The repo dir (`projects/pai-training-platform/`),
CDK stack names (`PaiTrainingPlatform-*`), ECR repo (`pai/vla-ft`), and job prefix
(`vla-ft-`) keep their current identifiers (renaming is a one-way door — gitlink,
deployed stacks, ledger). Docs use the product name `vla-ft`.

This document is the design of record. The roadmap (build order + verification
level per phase) lives in [`ROADMAP.md`](ROADMAP.md).

---

## 1. Problem restated

A robot-learning user has an *intent*, not an infrastructure plan. Two intents:

1. **Imitation (IL)** — "Here are demonstrations of the task; give me a policy that
   imitates them." Supervised fine-tuning of a VLA (π0.5, GR00T, ACT, …) on a
   LeRobot dataset. This is the verified `vla-ft` engine path.
2. **Reinforcement (RL)** — "Here is the task and what counts as success; learn a
   policy in simulation." Isaac Lab environment + reward → PPO (or similar) →
   trained policy + ONNX export. Required because **in a pure-sim setting you often
   cannot collect demonstrations at all**, and because the IL demonstration-gathering
   pipeline is still thin.

`vla-ft` turns either intent into a finished artifact while **picking the backend for
the user** (AWS Batch / SageMaker Training Job / SageMaker HyperPod) from data scale,
model memory, sim env count, and budget. The backend pick is a **smart default the user
can override** — implemented first as launcher decision logic, promotable to a
deterministic Step Functions orchestrator only when scale justifies it (§3).

```
                          ┌─────────────────────────────────────────────┐
   user intent  ───────►  │   LAUNCHER  (smart defaults + pre-flight)     │
   (IL data |  RL task)   │   classify → profile → decide → submit → notify│
                          │   [scale-gated → Step Functions orchestrator] │
                          └───────┬───────────────────────────┬──────────┘
                                  │ IL                         │ RL
                       ┌──────────▼─────────┐       ┌──────────▼──────────┐
                       │  INGEST (optional) │       │  Isaac Lab task pkg  │
                       │  *→ LeRobot v3     │       │  (env + reward)      │
                       └──────────┬─────────┘       └──────────┬──────────┘
                                  │                            │
                 ┌────────────────┴───────────┐   ┌────────────┴───────────┐
                 │  Pattern A  Batch + Spot    │   │  RL on Batch MNP / C   │
                 │  Pattern B  SM Training Job │   │  (Isaac Lab headless   │
                 │  Pattern C  HyperPod        │   │   PPO, NCCL multi-node)│
                 └────────────────┬───────────┘   └────────────┬───────────┘
                                  │                            │
                                  ▼                            ▼
                         S3 checkpoint/model           S3 policy + ONNX
                                  │
                                  └──► (optional) vla-hub serving hand-off
```

---

## 2. The two axes

### 2.1 IL axis — supervised VLA fine-tuning

- **Engine**: the generalized `vla-ft` container (LeRobot `lerobot-train`, BYOC).
  Backend-agnostic: the same image runs under Batch (A) or SM Training Job (B).
- **Input**: a LeRobot v3 dataset. If the user supplies something else (HDF5, rosbag,
  raw video), the **ingest** stage converts it first (§4).
- **Output**: a fine-tuned checkpoint (`pretrained_model/`), published to a promised
  S3 prefix, optionally mounted by `vla-hub` for serving.

### 2.2 RL axis — sim policy learning  *(Phase 3: code built, real run pending)*

- **Engine**: Isaac Lab headless training container (`containers/isaac-lab-rl/`) =
  Isaac Sim 4.5.0 + Isaac Lab **v2.3.2** (all RL frameworks via `./isaaclab.sh -i`). Runs
  on Batch single-GPU (RL Pattern A, `lib/rl/pattern-a-stack.ts`, built) or HyperPod for
  multi-node (RL Pattern C, synth-only). Pattern source: a verified Isaac Lab CDK +
  workshop infra template — we automated the Batch CE/JQ/JD it left manual.
  - **Trainer = rsl_rl, not SKRL.** The reference task registers both entry points, but
    only the **rsl_rl** workflow has a built-in ONNX export
    (`isaaclab_rl.rsl_rl.export_policy_as_onnx`, invoked by `rsl_rl/play.py`). SKRL's
    `play.py` does not export ONNX. Since the gate requires a deployable artifact, the
    container trains with rsl_rl and runs play.py to emit `exported/policy.onnx`.
- **Input (the RL "intent")**: an Isaac Lab **task id** — the environment cfg + reward
  are registered to the task, so the task *is* the success criteria. The verified
  reference task is **`Isaac-Velocity-Rough-H1-v0`** (Unitree H1 rough-terrain
  locomotion PPO, experiment `h1_rough`). Reward/task tweaks ride as Hydra-style
  overrides (`rl_launch.py --override key=value`); iteration/env counts as flags.
  User-supplied tasks plug into the same runner.
- **Output**: trained `model_<N>.pt` checkpoints + `exported/policy.onnx` + `policy.pt`,
  uploaded to S3 (`<output-s3>/<job>/output/<run>/`, sim2real-ready). `logs/` lives on
  EFS so a Spot reclaim + Batch retry resumes PPO (`--resume`) from the last checkpoint.

> **Why RL is not just "another Pattern"**: RL needs the simulator in the loop
> (Isaac Sim/Lab), thousands of parallel envs on the GPU, and a reward — a different
> container and a different success signal than supervised FT. It shares the
> *infrastructure* (VPC/EFS/AzSelector/Batch/HyperPod) with IL but not the engine.

---

## 3. Backend selection — smart default first, orchestrator when scale justifies

The defining decision is the same five-step pipeline regardless of where it runs:

1. **Classify** — IL vs RL from the input manifest (data artifact vs task package).
2. **Profile** — compute the deciding quantities:
   - IL: dataset size (episodes/frames/bytes), target policy → estimated VRAM
     (`model_params_GB × {lora 2.0, full_ft 12.0}`), expected wall-clock.
   - RL: num_envs, sim steps, single- vs multi-node need.
3. **Decide** — pure rule table → Pattern A / B / C (+ instance type). See §3.1.
   **Built** (`vla_ft_decide.decide()`).
4. **Submit** — launch the chosen backend (Batch job / SM Training Job / HyperPod job)
   with checkpoint+resume wired. **Built** for A/B (`vla_ft_cli.py` → the verified
   launchers); C is recommend-only (synth, not wired into `bin/app.ts`).
5. **Notify** — SNS on completion/failure; publish artifact S3 URI. **Built**
   (`lib/shared/notifications.ts`, EventBridge → SNS, deployed).

**Where this runs — two stages, deliberately:**

- **Stage 1 (now, BUILT): launcher decision logic.** The classify→profile→decide steps
  live in the launcher as a pure function over the rule table (§3.1). The user gets a
  **smart default** they can override (`--backend B`, `--instance-type …`). No Step
  Functions, no Lambda, no per-job state machine to debug. This is what serves the
  **Ease** axis with the least moving parts.
  **Shipped** as `containers/vla-ft/vla_ft_decide.py` (the pure rule-table module:
  `profile()` + `decide()`, stdlib only) called by `vla_ft_cli.py` (the one-command
  wrapper that resolves CFN outputs, pre-flights, estimates cost, then subprocess-calls
  the UNCHANGED `launch.py`/`batch_launch.py`). `vla_ft_decide.py` is the exact module a
  Stage-2 Lambda would import verbatim. See ROADMAP Phase 6.
- **Stage 2 (scale-gated, BUILT — code + synth): Step Functions orchestrator.** When the
  workload is multi-job / multi-user / needs durable retries, audit history, and async
  fan-out, the *same* rule table is promoted into a **deterministic Step Functions state
  machine** (Lambda per step, no LLM/Bedrock in the core path — same input → same
  decision → reproducible). This is an **optional graduation layer**, not the headline:
  it earns its complexity only when single-launcher orchestration stops being enough.
  **Shipped** as `lib/orchestrator/orchestrator-stack.ts` (Phase 4): a 2-Lambda state
  machine — `orchestrator_plan.py` (classify → profile → decide) imports
  `vla_ft_decide` **verbatim** (the exact module Stage 1 uses — zero re-implementation),
  and `orchestrator_submit.py` dispatches the backend **faithfully to the verified
  launchers**: IL/RL **Batch** is submitted directly via boto3 (mirroring
  `batch_launch.py`/`rl_launch.py` on the container-override contract), while
  **SageMaker (Pattern B)** is handed off as the exact `launch.py` command rather than
  forking the SDK estimator (the M1 hand-assembly trap).
  A `Choice` on the plan's deterministic `runnable` flag routes Pattern C (code+synth
  only) to a recommendation. The HF token is read inside the submit Lambda and injected
  as job env, so it never lands in the Step Functions execution history. **Deploy-gated**
  (not in `bin/app.ts`, like the HyperPod stacks); `test/orchestrator.test.ts` is the
  synth gate. No GPU run yet (a live IL + live RL through the machine is deferred to the
  next GPU spend — see ROADMAP Phase 4).

> Why this ordering: the heavy orchestrator does not, by itself, make fine-tuning
> *easier* — it adds infrastructure. Ease comes from smart defaults + pre-flight +
> one command. So the decision logic ships in the launcher first; Step Functions is a
> capacity/scale upgrade of the identical logic, gated on real need.

### 3.1 Decision rule table (initial, deterministic)

| Condition (IL) | Pattern | Instance |
|---|---|---|
| est. VRAM ≤ 48 GB **and** est. wall-clock ≤ ~2–4 h | **A** Batch Spot | g6e.4xlarge (1×L40S) |
| est. VRAM ≤ ~192 GB single node **or** wall-clock 8 h+ (needs auto-resume) | **B** SM Training Job | g6e.12xlarge / g6e.48xlarge / g5.48xlarge |
| exceeds single node | **C** HyperPod | g5.48xlarge×N / p5 |

| Condition (RL) | Pattern | Instance |
|---|---|---|
| single-GPU env count, short | **A** Batch | g6e.4xlarge |
| multi-GPU single node | **A/B** Batch MNP (1 node, N GPU) | g6e.12xlarge (4×L40S) |
| multi-node distributed PPO | **C** HyperPod / Batch MNP (N nodes) | g6.12xlarge×N |

> The thresholds are parameters, not hard-codes — they live in one config module so
> the rule table is auditable and tunable. Budget acts as a constraint (cap instance
> tier / force Spot) layered on top of the capability decision.

---

## 4. Ingest (IL input adapters)

The user's "imitation video" is the front door. Verified converters exist for some
formats; one link is missing.

| Input | Converter | Status |
|---|---|---|
| **LeRobot v3 dataset** | passthrough | ready |
| **HDF5** (sim/Isaac demos) | an `hdf5_to_lerobot.py` converter (verified sim-export path) | **verified code, directly reusable** |
| **teleop rosbag** | rosbag → LeRobot (robotis physical-ai-tools pattern) | pattern, OSS reference |
| **raw imitation video** | video → action labels → LeRobot | **GAP** — needs action-extraction layer (TwelveLabs `extract_actions` / Cosmos / retargeting) |

Ingest is event-driven: upload to S3 → manifest-complete event (apto edge-collection
pattern: `.done` marker / manifest PUT → S3 Event → SQS) → converter (Lambda for
light, Batch for heavy) → LeRobot v3 in S3 → triggers the IL path. The video→action
gap is exposed as a pluggable adapter slot, **explicitly marked unimplemented** so the
platform is honest about what it does and does not yet do.

---

## 5. Asset reuse map

What the platform absorbs vs references.

| Platform piece | Source asset | Reuse |
|---|---|---|
| Pattern B engine (container) | a verified single-EC2 π0.5 fine-tune | **absorb + generalize** → `containers/vla-ft` (policy/dataset/HP injected, not pinned to π0.5) |
| GPU capacity probe | a verified Isaac Lab `AzSelector` construct | **extract → `lib/shared`** (solves Spot insufficient-capacity for both axes) |
| RL Batch MNP + EFS + DCV + version-profiles | a verified Isaac Lab Batch/DCV infra template | **absorb → `lib/rl`** (Batch CE/JQ/JD automated, which the source left manual) |
| Pattern A Batch FT CDK + `dcv_construct.py` | `aws-samples/sample-embodied-ai-platform` (public, MIT-0, GR00T N1.5) | **adapt** (Python → TS port of the Batch construct) |
| Pattern B estimator pattern (10-policy draccus) | a reference SageMaker draccus estimator (pattern only, **not ours**) | **pattern only** — informs the generalized container; no fork |
| Pattern C HyperPod VLA | `aws-samples/awsome-distributed-training` openvla (Slurm) / openvla-oft (EKS) | **copy-paste starting point** (MIT-0) |
| Platform data model / multi-format import slot | a reference robotics data-platform design | **blueprint** (Terraform → CDK reimpl) |
| HDF5→LeRobot converter | a verified sim-export `hdf5_to_lerobot.py` | **direct reuse** |
| Trainium option (future) | a GR00T-on-Trainium/Neuron reference path | reference only |
| RL task/reward code | public Isaac Lab task/reward references (`rsl_rl`, reward-generation research) | **task source** — pick 1 verified reference task |
| Serving hand-off | `vla-hub` `MODEL_CHECKPOINT_DIR` env | unchanged; CDK injects env+mount |

---

## 6. Prior art & differentiation (vs VAMS / AWS Guidance)

A prior-art sweep (public web + AWS specialist guidance) ran *before* this build. It
found **partial overlap, not a duplicate**. Honest summary:

**Closest prior art — VAMS** (Visual Asset Management System,
`aws-solutions-library-samples`, Apache-2.0). VAMS ships **registered CDK pipeline
constructs**:
- Isaac Lab **RL** training+eval — `isaacLabTraining-construct.ts`: **AWS Batch**
  (G5/G6/G6E) + Step Functions `WAIT_FOR_TASK_TOKEN` + EFS, **rsl_rl hard-coded**,
  default task `Isaac-Cartpole-Direct-v0` (added 2026-01, has an AWS blog).
- **GR00T-N1.5-3B fine-tune** — `backendPipelines/genAi/nvidia/gr00t/`: LoRA + full FT,
  **LeRobot v2.1** input, Batch g6e.4xl/12xl.
- Cosmos.

VAMS has **no SageMaker Training Job, no HyperPod, and no IL/RL auto-select** — each
model is a separate hard-coded pipeline. So our **Pattern A (Batch)** and **Phase 3
RL (Isaac Lab on Batch)** substantively overlap what VAMS already does.

**Other prior art:**
- *AWS Guidance — AI-Driven Robotic Simulation and Training*: IL+RL+VLA in one
  architecture, but backend is **EKS + Trainium only** (no Batch/SM/HyperPod choice).
- *AWS Guidance — Physical AI for Robotics*: Batch (sim) + SM (retrain), **not VLA**,
  sim2real-edge focused.
- `aws-samples/sample-embodied-ai-platform` (Batch, **IL only**); other regional
  HyperPod IL+RL samples (IL and RL as *separate* samples); 2026-06-09 AWS blog (Isaac Lab RL on SM Training Job +
  HyperPod — **selection criteria written as prose, automation not built** — direct
  evidence of the gap we fill).

**The wedge — four axes, not one component.** No public or internal source combines
all four. (Each row notes overlap honestly.)

| Axis | `vla-ft` | Closest prior art | Genuinely differentiated? |
|------|----------|-------------------|---------------------------|
| **Ease** | one command; dataset+model → instance/backend auto-picked; pre-flight (quota/capacity/token); quickstart | VAMS = per-model CDK pipelines; 2026-06-09 blog = prose selection criteria, no automation | **Yes** — nobody ships the one-command smart-default UX for VLA FT |
| **Efficiency** | Spot + AzSelector default; instance auto-size; expert-only/early-stop default; pre-launch cost estimate | VAMS uses Spot Batch; none bundle AZ-probe + early-stop + cost preview as defaults | **Partial** — individual levers exist elsewhere; the *defaulted bundle* does not |
| **Coverage** | IL **and** RL behind one entry point | VAMS does both but as *separate* hard-coded pipelines; other samples = separate IL/RL samples; Guidances pick one backend | **Yes** — single intent surface over both |
| **Backend flexibility** | same intent → Batch **or** SM Training Job **or** HyperPod (A/B/C), with override | VAMS = Batch-only, per-model; Guidances = EKS/Trainium only or EKS only | **Yes** — A/B/C abstraction + HyperPod path is unique |

The reused parts (verified engine container, AzSelector) are real reuse, not
reinvention; the prior-art survey preceded the build.

**Honest caveats (do not oversell):**
- The **Ease/Efficiency default UX** (one-command launcher, pre-flight, pre-launch cost
  estimate) is **built and dry-run-validated** (`vla_ft_cli.py` + `vla_ft_decide.py`;
  `test_vla_ft_decide.py` 30/30 + a real dry-run against live CFN outputs), but the
  one-command path has **not yet been validated by a live `--yes` submit** (that reuses
  the next Phase 2b GPU run). The cost estimate is anchored to the verified run, not a
  per-job profiler. Service-Quotas pre-flight is a follow-up (capacity SPS already
  surfaces the practical blocker).
- The Batch single-pattern slice (Pattern A) **overlaps VAMS**; its value here is as one
  selectable backend behind the smart-default launcher, not as a standalone novelty.
- The **RL axis is code-built but not yet run on GPU** — RL Pattern A (Batch CE/JQ/JD +
  the Isaac Lab rsl_rl container + launcher) synthesizes and passes jest, but no RL job
  has executed under Batch in this account. Until the first real run produces a policy +
  ONNX, the Coverage claim is *built, not delivered*. (Note the genuine VAMS overlap: VAMS
  also runs Isaac Lab RL on Batch — §6 — so RL's differentiation lives in the IL→RL
  chaining + auto-select wedge, not in Batch RL alone.)
- The Step Functions orchestrator (the old single-wedge claim) is **optional and
  scale-gated** (§3), not the differentiator. It is now **built (code + synth, Phase 4
  `lib/orchestrator/`)** but **deploy-gated and not yet run on GPU** — a live IL + live
  RL execution through the state machine is deferred to the next GPU spend. The
  differentiator is the four-axis bundle above; the orchestrator is one way to scale the
  Ease axis, not the reason to exist.

### 6.1 vs a self-managed ParallelCluster / Slurm (and cross-cloud) operator

The prior-art table above is vs AWS samples. A different and increasingly common reader
already runs **self-managed EC2 ParallelCluster + Slurm**, often **across clouds**
(e.g. nebius / Azure for price or capacity arbitrage). For them the honest framing is not
"vla-ft is better" — it is **what vla-ft adds, and what it does not**.

A few facts first, so the comparison is grounded and not FUD (verified against AWS docs,
2026-06-23):

- **Control plane is $0 on both HyperPod and ParallelCluster** — you pay only for the EC2
  GPU capacity either way. So "self-managed PC is cheaper because there's no managed
  control-plane fee" is **false**; the real cost lever is Spot/instance-sizing/teardown
  discipline, which is exactly what vla-ft defaults.
- **HyperPod managed Spot resilience is EKS-only.** The managed Spot interruption
  handling announced 2025-11 ("up to 90%") and the managed node auto-recovery / health
  agent (GA for Slurm 2025-09-15) are HyperPod features, **not Slurm-on-ParallelCluster
  features**. On self-managed PC, Spot interruption handling, requeue, and node-health
  replacement are **yours to build and operate.** vla-ft does *not* deliver these through
  HyperPod-Slurm either — its resume story is **Batch retry (Pattern A) and SageMaker
  Managed Spot (Pattern B)**, which is a different mechanism, stated in §3 / the README.

**What vla-ft adds for a PC/Slurm operator:**
- **One-command backend auto-select** — it picks Batch / SM Training Job / HyperPod +
  instance for a given job instead of you maintaining a hand-tuned Slurm submit script per
  model size.
- **Defaulted Spot economics** — Spot on, AZ-capacity probe, instance auto-size,
  early-stop, and a **pre-launch cost estimate** — the levers a self-managed cluster
  exposes but does not default.
- **Resume-on-reclaim wiring you don't author** — the requeue/checkpoint-restore around a
  Spot reclaim is owned by the platform (Pattern A wired + a real run SUCCEEDED; Pattern B
  managed-spot code-complete), not a launcher you hand-write and maintain.
- **MCP self-serve** — an agent/research session can submit, check "is it really
  learning?", and read back a consistency-checked checkpoint without a human relaying
  order docs (see [`MCP-DESIGN.md`](MCP-DESIGN.md)).

**What vla-ft does NOT add (do not oversell):**
- **It is AWS-only.** It is **not a cross-cloud arbitrage replacement** — if you run on
  nebius/Azure for price or capacity, vla-ft does not span those; it makes the AWS leg
  easier, it does not unify your multi-cloud fleet.
- **It is not a ParallelCluster replacement.** If you already operate a tuned, trusted
  PC/Slurm pipeline, vla-ft is a *different abstraction* (managed Batch/SM schedulers),
  not a drop-in migration — and it does not give you Slurm semantics (gang scheduling,
  job arrays, fair-share). The "skip it when…" guidance in the README applies directly:
  a single fixed pipeline you trust does not need this.

---

## 7. Key decisions

- **North-star = easiest + most efficient VLA FT on AWS (IL & RL).** Every design
  choice is judged against Ease/Efficiency/Coverage/Backend-flexibility (§6). Features
  that add infrastructure without making fine-tuning *easier or cheaper* are deferred
  or made optional (e.g. the Step Functions orchestrator).
- **Backend-select ships in the launcher first; orchestrator is scale-gated.** The
  classify→profile→decide logic is a pure function over the rule table, exposed as a
  **smart default the user can override**. It is promoted to a deterministic Step
  Functions state machine (no LLM in the core path; auditable rule table; thresholds in
  one config module) **only when multi-job/multi-user scale justifies it** (§3). That
  promotion is now **built** (Phase 4 `lib/orchestrator/`): both the launcher and the
  orchestrator import the **same** `vla_ft_decide` module, so the decision logic lives in
  exactly one place and they can never diverge. The orchestrator stays deploy-gated until
  the scale need is real.
- **CDK language = TypeScript.** The highest-value, hardest-to-rewrite reusable
  assets (the Isaac Lab infra: AzSelector, Batch infra, version-profiles, DCV) are TS.
  `sample-embodied-ai-platform` is Python-CDK, but its Batch construct is small enough
  to port. One language across the platform beats straddling two.
- **Container stays BYOC and backend-agnostic.** The verified `vla-ft` image must
  run unchanged under both Batch (A) and SM Training Job (B). Generalization =
  parameterize policy/dataset/HP, **do not touch** the verified dependency stack
  (`lerobot @ git d1b1c5c8` + torch trio + py3.12/cu128).
- **Verification asymmetry** (user-set): A and B get **real E2E** (deploy + one real
  job). C gets **code + `cdk synth`** (real multi-node deploy deferred — HyperPod is
  reachable but needs a 1-day cluster commit). RL gets one verified
  reference task at the A/B level.
- **AzSelector everywhere.** Both axes hit GPU capacity limits; the probe-then-pick
  construct is shared, not duplicated.
- **Single CDK, multiple stacks.** `bin/app.ts` composes a shared base stack +
  per-pattern stacks so a user deploys only what a given run needs.

## 8. Open questions / to verify before promotion

- [ ] Confirm the reference SageMaker `train.py` draccus path actually runs pi05/groot
      (pattern we mirror in the generalized container).
- [ ] Pick the one verified RL reference task (workshop H1 vs a manipulation task).
- [ ] video→action extraction approach for the ingest gap (TwelveLabs vs Cosmos vs retarget).
- [ ] HyperPod cluster quota path (g-series cluster quotas are 0; g6e training-job
      quota auto-approved — is cluster usage similarly self-service?).
- [ ] Batch CE/JQ/JD full CDK automation (source left these manual).

**Related**: [`ROADMAP.md`](ROADMAP.md) (phased build + per-phase verification),
[`MCP-DESIGN.md`](MCP-DESIGN.md) (the MCP control surface).
