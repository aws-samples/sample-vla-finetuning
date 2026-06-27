#!/usr/bin/env python3
"""
vla-ft decision logic — the rule table as one pure, side-effect-free module.

This is the "smart default" brain behind the unified `vla_ft_cli.py` launcher
(ARCHITECTURE §3: backend-select ships in the launcher first, promotable to a
Step Functions orchestrator when scale justifies it). Keep it pure (stdlib only,
no boto3, no I/O): the SAME function the CLI calls today is what a Phase 4
classify→profile→decide Lambda would import verbatim — no re-implementation.

Three things live here, all data, all auditable:

  1. INSTANCES   — the GPU instance catalog: per-GPU memory, GPU count, and the
                   On-Demand $/hr anchors (EC2 for Pattern A/Batch, SageMaker ML
                   for Pattern B). Prices are real us-west-2 quotes (see source
                   note on each); they are the cost-estimate's only anchor.
  2. MODELS      — known VLA policies → parameter count (billions) + whether the
                   pi-family expert-only freeze applies (the verified-safe default).
  3. decide()    — pure function: (profile, budget, overrides) → a Decision
                   (Pattern A/B/C + instance + spot + a printable cost estimate).

Faithful to ARCHITECTURE §3.1 (the rule table) and to the hard-won verified
lessons (vla-ft session 2026-06-14..16):

  - **VRAM** is estimated with the formula
    `required_vram = params_billions × {lora 2.0, full_ft 12.0}`,
    with a distinct **expert_only** mode for pi-family (freeze the 2B VLM, train only the
    ~0.3B action expert — empirically ~24 GB on an L40S, the config that PASSED).
  - **DDP replicates the model per GPU** (accelerate --multi_gpu), so the deciding
    quantity is **per-GPU** replica footprint, NOT aggregate node memory. This is
    exactly why full-VLM pi05 full-FT OOM'd on a 48 GB L40S at batch 4 even with
    4 GPUs — more GPUs split the batch, not the replica. A replica that exceeds one
    L40S (48 GB) needs sharding (FSDP) → Pattern C.
  - **Efficient default** for the pi-family is **expert_only=true**: it matches the
    verified lock, fits one L40S, and resists overfit on the 50-episode set. The
    user overrides with `--full-vlm` when they truly want the 2B VLM to train too.

Nothing here launches anything; `vla_ft_cli.py` turns a Decision into a call to the
UNCHANGED verified launchers (launch.py / batch_launch.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ── 1. Instance catalog ─────────────────────────────────────────────────────────
#
# gpu_mem_gb is PER-GPU memory (L40S = 48, A10G = 24) — the DDP-replica budget.
# Aggregate node memory = gpus × gpu_mem_gb.
#
# Prices: real On-Demand us-west-2 quotes pulled from the AWS Price List API on
# 2026-06-12 (EC2) / 2026-06-15 (SageMaker). EC2 OD drives the Pattern A / Batch
# estimate; SageMaker ML OD drives Pattern B. They are anchors for a *pre-launch
# estimate* shown for confirmation, not a billing guarantee.

@dataclass(frozen=True)
class Instance:
    gpus: int
    gpu_mem_gb: int          # per-GPU memory (the DDP replica budget)
    gpu: str                 # GPU model (informational)
    ec2_od: float            # EC2 On-Demand $/hr (Pattern A / Batch)
    sm_od: float | None      # SageMaker Training ML $/hr (Pattern B); None = not offered/used

    @property
    def aggregate_mem_gb(self) -> int:
        return self.gpus * self.gpu_mem_gb


# Catalog limited to the types we have quota for + the verified fallbacks.
INSTANCES: dict[str, Instance] = {
    # 1×A10G 24 GB — Pattern A single-GPU Spot fallback when L40S capacity is short.
    "g5.4xlarge":   Instance(gpus=1, gpu_mem_gb=24, gpu="A10G", ec2_od=1.624,  sm_od=None),
    # 1×L40S 48 GB — Pattern A primary (short single-GPU fine-tunes / PoC).
    "g6e.4xlarge":  Instance(gpus=1, gpu_mem_gb=48, gpu="L40S", ec2_od=3.00424, sm_od=3.76),
    # 4×L40S — Pattern B default (the verified full-run instance, eff-batch 16).
    "g6e.12xlarge": Instance(gpus=4, gpu_mem_gb=48, gpu="L40S", ec2_od=10.49264, sm_od=13.12),
    # 8×L40S — Pattern B "go faster" option (~½ the wall-clock of 12xl).
    "g6e.48xlarge": Instance(gpus=8, gpu_mem_gb=48, gpu="L40S", ec2_od=30.13118, sm_od=37.66),
}

# Largest per-GPU memory we can place a single DDP replica on (L40S). A replica
# bigger than this can't run under accelerate DDP → needs FSDP/HyperPod (Pattern C).
MAX_SINGLE_GPU_MEM_GB = 48

# Pattern A Spot CE GPU floor. The Spot Compute Environment lists g6e.4xlarge (L40S
# 48 GB) primary → g5.4xlarge (A10G 24 GB) fallback, and Batch may place a Spot job on
# EITHER depending on capacity — there is NO per-job instance-type override in Batch, so
# a replica > this floor can silently land on the 24 GB g5 and OOM at step 0 (the
# 2026-06-23 pi05 full-VLM failure). The On-Demand CE is narrowed to L40S-only (48 GB),
# so the safe answer when a replica exceeds this floor is to route to On-Demand.
SPOT_GPU_FLOOR_GB = 24

# Spot discount factor for the estimate. Anchored to the REAL Pattern A run
# (g6e.4xl Spot, 2026-06-15: ~$0.20 for ~12.7 min RUNNING ⇒ ~$0.94/hr ⇒ ~31 % of
# the $3.004 OD). Spot fluctuates (~30–60 % of OD typically); we use ~0.35 and
# clearly label the result an estimate.
SPOT_FACTOR = 0.35

# Per-step wall-clock heuristic, by GPU model. Anchored to the verified full run
# (g6e.12xlarge, 20 000 steps, billable 19 819 s ⇒ ~0.99 s/step on L40S). Steps are
# fixed regardless of GPU count (more GPUs grow the effective batch, they do NOT
# shorten a step), so this is GPU-architecture-bound, not count-bound.
S_PER_STEP = {"L40S": 1.0, "A10G": 1.6}

# Effective batch we hold constant to preserve the verified optimizer trajectory
# (verified smoke: batch 4 × 4 GPU = 16). per-device batch = EFF_BATCH / num_gpus.
EFF_BATCH = 16


# ── 2. Model registry ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Model:
    params_b: float          # parameter count in billions (the VRAM formula's base)
    family: str              # "pi" | "groot" | "act" | "smolvla" | ...
    expert_only_capable: bool  # pi-family: freeze the VLM, train the action expert only


# Known LeRobot policy types. params_b feeds the best-practice VRAM formula.
MODELS: dict[str, Model] = {
    "pi05":    Model(params_b=3.3,  family="pi",      expert_only_capable=True),
    "pi0":     Model(params_b=3.3,  family="pi",      expert_only_capable=True),
    "pi0_fast": Model(params_b=3.3, family="pi",      expert_only_capable=True),
    "groot":   Model(params_b=3.0,  family="groot",   expert_only_capable=False),
    "smolvla": Model(params_b=0.45, family="smolvla", expert_only_capable=False),
    "xvla":    Model(params_b=1.0,  family="xvla",    expert_only_capable=False),
    "act":     Model(params_b=0.09, family="act",     expert_only_capable=False),
    "diffusion": Model(params_b=0.16, family="diffusion", expert_only_capable=False),
    "vqbet":   Model(params_b=0.10, family="vqbet",   expert_only_capable=False),
    "tdmpc":   Model(params_b=0.05, family="tdmpc",   expert_only_capable=False),
}

# Unknown policy → assume a 3B VLA (the common case here) so the estimate is
# conservative rather than absent.
DEFAULT_PARAMS_B = 3.0

# Fine-tune VRAM multipliers (× params_billions).
# QLoRA (4-bit, ~0.5×) is intentionally absent: lerobot @ d1b1c5c8 has no bitsandbytes /
# 4-bit load path, so the front door rejects --qlora rather than ship a non-QLoRA run.
VRAM_MULT = {"lora": 2.0, "full_ft": 12.0}


def expert_only_vram_gb(params_b: float) -> float:
    """Per-GPU VRAM for the pi-family expert-only freeze.

    The 2B VLM is frozen (bf16 forward only, ~2 bytes/param) and only the small
    action expert trains. Modeled as `params_b × 2 + 16` GB of headroom for
    activations + the expert's optimizer state. Anchored to the verified pi05
    expert-only run (~24 GB on a 48 GB L40S): 3.3 × 2 + 16 ≈ 22.6 GB."""
    return params_b * 2.0 + 16.0


# ── 3. Profile + Decision ─────────────────────────────────────────────────────────

@dataclass
class Profile:
    """The deciding quantities for one IL fine-tune (RL profiling is a Phase 3 add)."""
    model: str
    params_b: float
    ft_mode: str             # "expert_only" | "full_ft" | "lora"
    vram_per_gpu_gb: float   # one DDP replica's footprint — the deciding memory
    steps: int
    est_wall_clock_h: float  # rough estimate (steps × s/step), order-of-magnitude


@dataclass
class Decision:
    pattern: str             # "A" | "B" | "C"
    backend: str             # "batch" | "sagemaker" | "hyperpod"
    instance_type: str       # bare EC2 type (e.g. g6e.12xlarge)
    sm_instance_type: str    # ml.-prefixed for SageMaker (Pattern B); "" otherwise
    num_gpus: int
    per_device_batch: int    # EFF_BATCH / num_gpus (holds verified trajectory)
    spot: bool
    expert_only: bool
    price_per_hr: float      # the rate used for the estimate (spot-adjusted)
    est_cost_usd: float      # price_per_hr × est_wall_clock_h
    est_cost_od_usd: float   # same wall-clock at On-Demand (for comparison)
    rationale: str
    axis: str = "il"         # "il" | "rl" — which axis (and which pattern stack) this targets
    notes: list[str] = field(default_factory=list)


def resolve_ft_mode(model: str, full_vlm: bool, lora: bool) -> str:
    """Resolve the fine-tune mode from the model + the user's efficiency overrides.

    Efficient default (pi-family): expert_only — matches the verified lock, fits one
    L40S, resists overfit on small data. `full_vlm` opts into training the 2B VLM too.
    Non-pi models have no expert to isolate, so they default to full_ft.

    (No qlora mode: lerobot @ d1b1c5c8 has no 4-bit path, so the front door rejects
    --qlora outright rather than expose a mode that can't run — honest gating.)"""
    if lora:
        return "lora"
    m = MODELS.get(model)
    if m and m.expert_only_capable and not full_vlm:
        return "expert_only"
    return "full_ft"


def profile_run(model: str, steps: int, ft_mode: str) -> Profile:
    """Compute the deciding quantities for a run. Pure; no I/O."""
    m = MODELS.get(model)
    params_b = m.params_b if m else DEFAULT_PARAMS_B

    if ft_mode == "expert_only":
        vram = expert_only_vram_gb(params_b)
    else:
        vram = params_b * VRAM_MULT.get(ft_mode, VRAM_MULT["full_ft"])

    # Wall-clock: GPU-architecture-bound (L40S default). steps × s/step.
    wall_h = steps * S_PER_STEP["L40S"] / 3600.0
    return Profile(
        model=model,
        params_b=params_b,
        ft_mode=ft_mode,
        vram_per_gpu_gb=round(vram, 1),
        steps=steps,
        est_wall_clock_h=round(wall_h, 2),
    )


def _price(instance: Instance, pattern: str, spot: bool) -> float:
    """The $/hr rate used for the estimate. Pattern B → SageMaker ML rate; A → EC2.
    Spot applies SPOT_FACTOR (A's Batch CE is always Spot; B uses Managed Spot)."""
    base = instance.sm_od if (pattern == "B" and instance.sm_od) else instance.ec2_od
    return round(base * (SPOT_FACTOR if spot else 1.0), 4)


def decide(
    profile: Profile,
    *,
    budget_usd: float | None = None,
    backend_override: str | None = None,
    instance_override: str | None = None,
    spot: bool = True,
) -> Decision:
    """Pure rule-table decision: profile (+ budget + overrides) → Decision.

    Order of precedence (the user always wins — smart default, not a cage):
      1. explicit backend_override / instance_override,
      2. the §3.1 rule table over (per-GPU VRAM, wall-clock),
      3. budget as a constraint layered on top (force Spot / warn, never block).
    """
    notes: list[str] = []
    vram = profile.vram_per_gpu_gb
    wall = profile.est_wall_clock_h
    expert_only = profile.ft_mode == "expert_only"

    # ── pattern selection ──
    if backend_override:
        pattern = {"batch": "A", "sagemaker": "B", "hyperpod": "C",
                   "a": "A", "b": "B", "c": "C"}.get(backend_override.lower())
        if not pattern:
            raise ValueError(f"unknown backend override: {backend_override!r} "
                             f"(use batch|sagemaker|hyperpod)")
        rationale = f"backend forced to Pattern {pattern} (--backend {backend_override})"
    elif vram > MAX_SINGLE_GPU_MEM_GB:
        # One replica exceeds a single L40S → DDP can't place it; needs FSDP/HyperPod.
        pattern = "C"
        rationale = (f"per-GPU replica ~{vram:.0f} GB > {MAX_SINGLE_GPU_MEM_GB} GB "
                     f"(one L40S) → needs model sharding (FSDP) → Pattern C HyperPod")
        notes.append("A model replica larger than one GPU cannot run under accelerate "
                     "DDP (which replicates, not shards) — Pattern C shards it via FSDP2 "
                     "(accelerate --use_fsdp FULL_SHARD across nodes). Pattern C is a "
                     "deploy-gated reference (cdk -c enableHyperPod=true synthesizes the "
                     "cluster; launch with hyperpod_fsdp_launch.sh), NOT auto-submittable "
                     "from this planner — it's an operator-run sbatch on the HyperPod head, "
                     "not a Batch/SageMaker job. Or reduce footprint (expert-only / LoRA / "
                     "smaller model) to stay on a runnable single-node pattern.")
    elif vram <= MAX_SINGLE_GPU_MEM_GB and wall <= 4.0:
        # Fits one L40S AND short → cheapest single-GPU Spot tier.
        pattern = "A"
        rationale = (f"per-GPU ~{vram:.0f} GB ≤ 48 GB and est ~{wall:.1f} h ≤ 4 h "
                     f"→ Pattern A (Batch + g6e.4xlarge Spot), the cheap single-GPU tier")
    else:
        # Fits one L40S but long (8h+ / needs auto-resume) → managed SM Training Job.
        pattern = "B"
        rationale = (f"per-GPU ~{vram:.0f} GB ≤ 48 GB but est ~{wall:.1f} h (>4 h, wants "
                     f"checkpoint auto-resume) → Pattern B (SageMaker Training Job)")

    # ── instance selection ──
    if instance_override:
        bare = instance_override.replace("ml.", "")
        if bare not in INSTANCES:
            notes.append(f"instance {instance_override!r} not in the catalog — "
                         f"cost estimate may be unavailable.")
        instance_type = bare
    elif pattern == "A":
        instance_type = "g6e.4xlarge"   # CE primary (falls back to g5.4xlarge on capacity)
        notes.append("Pattern A instance is set by the Batch Compute Environment "
                     "(g6e.4xlarge primary → g5.4xlarge Spot fallback); batch_launch.py "
                     "takes no --instance-type.")
    elif pattern == "B":
        instance_type = "g6e.12xlarge"  # verified full-run instance (4×L40S, eff-batch 16)
        notes.append("Pattern B default g6e.12xlarge (4×L40S) is the verified full-run "
                     "instance; pass --instance-type ml.g6e.48xlarge for ~½ the wall-clock.")
    else:  # C
        instance_type = "g6e.48xlarge"
        notes.append("Pattern C (HyperPod Slurm, multi-node FSDP2) is a deploy-gated "
                     "reference: deploy the cluster with `cdk deploy "
                     "PaiTrainingPlatform-IL-HyperPod -c enableHyperPod=true "
                     "[-c hyperPodFsx=true]`, then launch with hyperpod_fsdp_launch.sh "
                     "(sbatch on the head). It is NOT auto-submitted by this planner — a "
                     "real multi-node run is operator-gated (standing cluster cost).")

    instance = INSTANCES.get(instance_type)
    num_gpus = instance.gpus if instance else 1
    per_device_batch = max(1, round(EFF_BATCH / num_gpus))
    sm_instance_type = f"ml.{instance_type}" if pattern == "B" else ""

    # ── GPU-floor guard (Pattern A only): keep a too-big replica off the 24 GB g5 ──
    # Batch picks the instance from the CE list (no per-job override), so a Spot job whose
    # replica exceeds the Spot CE's g5 fallback floor can OOM at step 0. The On-Demand CE
    # is L40S-only (48 GB), so auto-route there instead of letting Spot gamble on capacity.
    # This is a smart default (not a cage): an explicit instance_override is left untouched.
    if pattern == "A" and spot and not instance_override and vram > SPOT_GPU_FLOOR_GB:
        spot = False
        notes.append(
            f"per-GPU replica ~{vram:.0f} GB > {SPOT_GPU_FLOOR_GB} GB (the Spot CE's g5 "
            f"fallback) — auto-routed to the On-Demand queue (L40S-only, 48 GB) so Batch "
            f"can't place it on a 24 GB g5 and OOM at step 0. Pass spot=True + a smaller "
            f"footprint (expert_only / LoRA) to stay on Spot.")

    # ── cost estimate ──
    if instance:
        price = _price(instance, pattern, spot)
        price_od = _price(instance, pattern, spot=False)
        est_cost = round(price * wall, 2)
        est_cost_od = round(price_od * wall, 2)
    else:
        price, est_cost, est_cost_od = 0.0, 0.0, 0.0
        notes.append("no price anchor for the chosen instance — estimate unavailable.")

    # ── budget as a constraint (never blocks; two-way door, user confirms) ──
    if budget_usd is not None and est_cost_od > budget_usd:
        if not spot:
            notes.append(f"est On-Demand ${est_cost_od:.0f} exceeds budget "
                         f"${budget_usd:.0f} → recommend Spot (est ${round(price_od * SPOT_FACTOR * wall, 2):.0f}).")
        elif est_cost > budget_usd:
            notes.append(f"⚠️ even on Spot, est ${est_cost:.0f} exceeds budget "
                         f"${budget_usd:.0f} — consider fewer steps, expert-only, or a "
                         f"smaller instance.")

    return Decision(
        pattern=pattern,
        backend={"A": "batch", "B": "sagemaker", "C": "hyperpod"}[pattern],
        instance_type=instance_type,
        sm_instance_type=sm_instance_type,
        num_gpus=num_gpus,
        per_device_batch=per_device_batch,
        spot=spot,
        expert_only=expert_only,
        price_per_hr=price,
        est_cost_usd=est_cost,
        est_cost_od_usd=est_cost_od,
        rationale=rationale,
        notes=notes,
    )


def format_plan(profile: Profile, decision: Decision) -> str:
    """A human-readable, copy-pasteable summary of the resolved plan + cost estimate."""
    d = decision
    spot_label = "Spot" if d.spot else "On-Demand"
    lines = [
        "=" * 64,
        f"vla-ft plan — {profile.model} ({profile.params_b:g}B), {profile.ft_mode}",
        "=" * 64,
        f"  decision   : Pattern {d.pattern} ({d.backend})  on  {d.instance_type}"
        + (f"  [{d.sm_instance_type}]" if d.sm_instance_type else ""),
        f"  why        : {d.rationale}",
        f"  compute    : {d.num_gpus} GPU × per-device batch {d.per_device_batch} "
        f"(eff batch {d.per_device_batch * d.num_gpus}), {spot_label}",
        f"  est VRAM   : ~{profile.vram_per_gpu_gb:g} GB / GPU   (model replica footprint)",
        f"  est time   : ~{profile.est_wall_clock_h:g} h  ({profile.steps} steps)",
        f"  est cost   : ~${d.est_cost_usd:.0f}  ({spot_label} @ ${d.price_per_hr:g}/hr)"
        + (f"   ·  On-Demand ~${d.est_cost_od_usd:.0f}" if d.spot else ""),
    ]
    if d.notes:
        lines.append("  notes      :")
        lines += [f"    - {n}" for n in d.notes]
    lines.append("  ⚠️ cost/time are ESTIMATES (anchored to the verified run); "
                 "Spot fluctuates and capacity is time-sensitive.")
    lines.append("=" * 64)
    return "\n".join(lines)


def format_rl_plan(profile: RlProfile, decision: Decision) -> str:
    """A human-readable plan summary for an RL run (the RL analogue of format_plan).
    The RL intent is a TASK + env/iteration knobs — no model VRAM / per-device batch."""
    d = decision
    spot_label = "Spot" if d.spot else "On-Demand"
    envs = profile.num_envs if profile.num_envs is not None else "(task default)"
    iters = profile.max_iterations if profile.max_iterations is not None else "(task default)"
    time_str = f"~{profile.est_wall_clock_h:g} h" if profile.est_wall_clock_h else "unknown (no --max-iterations)"
    cost_str = (f"~${d.est_cost_usd:.0f}  ({spot_label} @ ${d.price_per_hr:g}/hr)"
                + (f"   ·  On-Demand ~${d.est_cost_od_usd:.0f}" if d.spot else "")
                ) if d.est_cost_usd else "unavailable (pass --max-iterations)"
    lines = [
        "=" * 64,
        f"vla-ft RL plan — {profile.task}",
        "=" * 64,
        f"  decision   : Pattern {d.pattern} ({d.backend})  on  {d.instance_type}",
        f"  why        : {d.rationale}",
        f"  compute    : {d.num_gpus} GPU, {spot_label}  (envs {envs}, iters {iters})",
        f"  est time   : {time_str}",
        f"  est cost   : {cost_str}",
    ]
    if d.notes:
        lines.append("  notes      :")
        lines += [f"    - {n}" for n in d.notes]
    lines.append("=" * 64)
    return "\n".join(lines)


# ── 4. RL axis (Phase 3) ────────────────────────────────────────────────────────────
#
# The RL "intent" is a TASK, not a dataset (the env + reward are registered to the task
# id), so the deciding quantities are different from IL: parallel env count, PPO
# iterations, and single- vs multi-node need — NOT model VRAM. ARCHITECTURE §3.1's RL
# rule table:
#
#   single-GPU env count, short      → Pattern A (Batch)            g6e.4xlarge
#   multi-GPU single node            → Pattern A (Batch MNP, N GPU) g6e.12xlarge
#   multi-node distributed PPO       → Pattern C (HyperPod)         g6.12xlarge×N
#
# Only RL Pattern A (Batch) is a runnable backend today (lib/rl/pattern-a-stack.ts);
# Pattern C is code+synth only. This block is the RL half of the SAME single-source rule
# table the orchestrator imports — it does NOT re-implement decision logic anywhere else.

RL_DEFAULT_TASK = "Isaac-Velocity-Rough-H1-v0"   # the verified reference task (H1 rough PPO)
RL_DEFAULT_EXPERIMENT = "h1_rough"

# Per-iteration wall-clock heuristic for the RL cost estimate. UNVERIFIED: unlike the IL
# anchor (a real g6e.12xl run), no RL job has executed on GPU in this account yet
# (ROADMAP Phase 3 — real run pending). Treat the RL cost estimate as order-of-magnitude
# only; every RL Decision carries a note saying so.
RL_S_PER_ITER = 1.5


@dataclass
class RlProfile:
    """The deciding quantities for one RL run (the RL analogue of Profile)."""
    task: str
    experiment_name: str
    num_envs: int | None        # parallel sim envs (None = task-registered default)
    max_iterations: int | None  # PPO iterations (None = task-registered default)
    num_gpus: int               # GPUs requested on the node
    multi_node: bool            # distributed PPO across >1 node
    est_wall_clock_h: float | None  # iterations × s/iter (None if iterations unset)


def profile_rl(
    task: str | None,
    *,
    num_envs: int | None = None,
    max_iterations: int | None = None,
    num_gpus: int = 1,
    multi_node: bool = False,
    experiment_name: str | None = None,
) -> RlProfile:
    """Compute the RL deciding quantities. Pure; no I/O."""
    iters = max_iterations
    wall = round(iters * RL_S_PER_ITER / 3600.0, 2) if iters else None
    return RlProfile(
        task=task or RL_DEFAULT_TASK,
        experiment_name=experiment_name or RL_DEFAULT_EXPERIMENT,
        num_envs=num_envs,
        max_iterations=iters,
        num_gpus=max(1, num_gpus),
        multi_node=multi_node,
        est_wall_clock_h=wall,
    )


def decide_rl(
    profile: RlProfile,
    *,
    budget_usd: float | None = None,
    backend_override: str | None = None,
    instance_override: str | None = None,
    spot: bool = True,
) -> Decision:
    """Pure RL rule-table decision: RlProfile (+ budget + overrides) → Decision.

    Same precedence as decide(): explicit override → rule table → budget constraint.
    The only runnable RL backend is Pattern A (Batch); multi-node maps to Pattern C
    (HyperPod), which is synth-only today (flagged in notes, like the IL C path)."""
    notes: list[str] = []

    # ── pattern selection (RL rule table) ──
    if backend_override:
        pattern = {"batch": "A", "hyperpod": "C", "a": "A", "c": "C",
                   # SageMaker is not an RL backend (no Isaac Sim managed image) — map B→A.
                   "sagemaker": "A", "b": "A"}.get(backend_override.lower())
        if not pattern:
            raise ValueError(f"unknown RL backend override: {backend_override!r} "
                             f"(use batch|hyperpod)")
        if backend_override.lower() in ("sagemaker", "b"):
            notes.append("SageMaker is not an RL backend (Isaac Sim needs our own GPU "
                         "node) — using Pattern A (Batch) instead.")
        rationale = f"RL backend forced to Pattern {pattern} (--backend {backend_override})"
    elif profile.multi_node:
        pattern = "C"
        rationale = ("multi-node distributed PPO → Pattern C (HyperPod / Batch MNP across "
                     "nodes)")
        notes.append("RL Pattern C (multi-node) shares the IL HyperPod cluster construct "
                     "(deploy-gated, -c enableHyperPod=true); the RL multi-node launch path "
                     "is not yet wired. Single-node multi-GPU (Pattern A with --num-gpus) is "
                     "the runnable path; use it unless the run genuinely needs >1 node.")
    elif profile.num_gpus > 1:
        pattern = "A"
        rationale = (f"{profile.num_gpus}-GPU single node → RL Pattern A (Batch MNP, "
                     f"{profile.num_gpus} GPU on one node via torchrun)")
    else:
        pattern = "A"
        rationale = "single-GPU env count → RL Pattern A (Batch + g6e.4xlarge Spot)"

    # ── instance selection ──
    if instance_override:
        bare = instance_override.replace("ml.", "")
        if bare not in INSTANCES:
            notes.append(f"instance {instance_override!r} not in the catalog — "
                         f"cost estimate may be unavailable.")
        instance_type = bare
    elif pattern == "C":
        instance_type = "g6e.48xlarge"
        notes.append("RL Pattern C (HyperPod) is a recommendation, not a runnable path yet.")
    elif profile.num_gpus > 1:
        instance_type = "g6e.12xlarge"   # 4×L40S for single-node multi-GPU PPO
        notes.append("Multi-GPU RL uses g6e.12xlarge (4×L40S); deploy RlPatternAStack with "
                     "this instance type and submit with --num-gpus.")
    else:
        instance_type = "g6e.4xlarge"    # CE primary (g6e.4xlarge → g5.4xlarge Spot fallback)
        notes.append("RL Pattern A instance is set by the Batch Compute Environment "
                     "(g6e.4xlarge primary → g5.4xlarge Spot fallback); rl_launch.py takes "
                     "no --instance-type.")

    instance = INSTANCES.get(instance_type)
    num_gpus = profile.num_gpus if profile.num_gpus > 1 else (instance.gpus if instance else 1)

    # ── cost estimate (UNVERIFIED for RL — see RL_S_PER_ITER) ──
    wall = profile.est_wall_clock_h
    if instance and wall is not None:
        price = _price(instance, pattern, spot)
        price_od = _price(instance, pattern, spot=False)
        est_cost = round(price * wall, 2)
        est_cost_od = round(price_od * wall, 2)
    else:
        price = _price(instance, pattern, spot) if instance else 0.0
        price_od = _price(instance, pattern, spot=False) if instance else 0.0
        est_cost, est_cost_od = 0.0, 0.0
        if wall is None:
            notes.append("RL wall-clock unknown (no --max-iterations) — cost estimate "
                         "unavailable; pass max_iterations for an estimate.")
    notes.append("⚠️ RL timing is UNVERIFIED — no RL job has run on GPU in this account "
                 "yet (ROADMAP Phase 3). The first run is an E2E smoke; treat cost/time "
                 "as order-of-magnitude.")

    # ── budget as a constraint (never blocks) ──
    if budget_usd is not None and est_cost_od > budget_usd > 0:
        if est_cost > budget_usd:
            notes.append(f"⚠️ even on Spot, est ${est_cost:.0f} exceeds budget "
                         f"${budget_usd:.0f} — consider fewer iterations or a smaller env count.")
        else:
            notes.append(f"est On-Demand ${est_cost_od:.0f} exceeds budget "
                         f"${budget_usd:.0f} → recommend Spot (est ${est_cost:.0f}).")

    return Decision(
        pattern=pattern,
        backend={"A": "batch", "C": "hyperpod"}[pattern],
        instance_type=instance_type,
        sm_instance_type="",          # RL has no SageMaker backend
        num_gpus=num_gpus,
        per_device_batch=0,           # N/A for RL (env count, not a per-device batch)
        spot=spot,
        expert_only=False,            # N/A for RL
        price_per_hr=price,
        est_cost_usd=est_cost,
        est_cost_od_usd=est_cost_od,
        rationale=rationale,
        axis="rl",
        notes=notes,
    )


# ── 5. GR00T axis (NVIDIA Isaac-GR00T N1.7 — the third IL engine) ────────────────────
#
# GR00T N1.7 is a separate training engine from lerobot: a 3B Cosmos VLA fine-tuned by
# NVIDIA's launch_finetune.py (baked in the gr00t-n17 image), with its own GROOT_* env
# contract (gr00t_launch.py) and its own checkpoint layout (sharded model-*.safetensors,
# never adapter-only). The "intent" is a LeRobot v2.1 dataset + a registered embodiment
# tag (e.g. UNITREE_G1), not a lerobot policy. So it gets its own profile/decide rather
# than being squeezed into the lerobot rule table.
#
# It runs on EXACTLY ONE deployed backend — GrootPatternAStack: AWS Batch on a single
# g6e.4xlarge (1×L40S 48 GB), On-Demand by default, NO g5/A10G fallback (24 GB OOMs a
# 3B Cosmos model). There is no Pattern B/C for GR00T, so decide_groot always returns
# Pattern A. This block is part of the SAME single-source rule table the orchestrator
# imports — it does not re-implement decision logic anywhere else.

GROOT_DEFAULT_BASE = "nvidia/GR00T-N1.7-3B"
GROOT_DEFAULT_EMBODIMENT = "UNITREE_G1"
GROOT_INSTANCE = "g6e.4xlarge"   # the only GrootPatternAStack tier (1×L40S, quota secured)

# UNITREE_G1 full-body uses a 50-step action horizon, but the base GR00T-N1.7-3B / model
# config default to 40 and launch_finetune.py exposes no flag to change it. Omitting the
# horizon ships the 40/50 mismatch that crashed the G1 rollout at step 0. So for G1 we
# smart-default action_horizon=50 (the baked g1_finetune.py wrapper sets it pre-build).
# Other embodiments whose data horizon matches the 40-wide head leave it unset.
GROOT_G1_ACTION_HORIZON = 50

# Per-step wall-clock anchor — the VERIFIED full G1 run (gr00t-n17-20260619-103523:
# 5000 steps, train_runtime 6389 s ⇒ ~1.28 s/step on a single L40S). Unlike RL this is a
# real on-GPU anchor, so the GR00T cost estimate is grounded (still labeled an estimate).
GROOT_S_PER_STEP = 1.28


@dataclass
class GrootProfile:
    """The deciding quantities for one GR00T N1.7 fine-tune (the GR00T analogue of Profile).
    The intent is a dataset + embodiment, so the deciding inputs are steps + horizon, not
    a per-device batch (GR00T uses a global batch the trainer splits internally)."""
    embodiment_tag: str
    base_model: str
    max_steps: int
    save_steps: int
    action_horizon: int | None   # None = leave the 40-wide base head (non-G1 embodiments)
    num_gpus: int
    global_batch: int
    est_wall_clock_h: float


def profile_groot(
    *,
    embodiment_tag: str | None = None,
    base_model: str | None = None,
    max_steps: int | None = None,
    save_steps: int | None = None,
    action_horizon: int | None = None,
    num_gpus: int = 1,
    global_batch: int | None = None,
) -> GrootProfile:
    """Compute the GR00T deciding quantities. Pure; no I/O.

    Smart default for the horizon: an explicit action_horizon wins; else UNITREE_G1 (full
    body) defaults to 50 (the verified-required value — omitting it ships the 40/50 step-0
    crash); other embodiments leave it unset (the base head is already 40-wide)."""
    emb = embodiment_tag or GROOT_DEFAULT_EMBODIMENT
    steps = max_steps if max_steps is not None else 2000   # gr00t_launch.py default
    if action_horizon is not None:
        horizon = action_horizon
    elif emb.upper().startswith("UNITREE_G1"):
        horizon = GROOT_G1_ACTION_HORIZON
    else:
        horizon = None
    wall = round(steps * GROOT_S_PER_STEP / 3600.0, 2)
    return GrootProfile(
        embodiment_tag=emb,
        base_model=base_model or GROOT_DEFAULT_BASE,
        max_steps=steps,
        save_steps=save_steps if save_steps is not None else 1000,
        action_horizon=horizon,
        num_gpus=max(1, num_gpus),
        global_batch=global_batch if global_batch is not None else 64,
        est_wall_clock_h=wall,
    )


def decide_groot(
    profile: GrootProfile,
    *,
    budget_usd: float | None = None,
    backend_override: str | None = None,
    instance_override: str | None = None,
    spot: bool = False,
) -> Decision:
    """Pure GR00T rule-table decision: GrootProfile (+ budget) → Decision.

    GR00T has exactly one deployed backend (GrootPatternAStack, Pattern A Batch on
    g6e.4xlarge), so unlike the IL/RL deciders there is no pattern selection — only the
    cost estimate + budget constraint vary. A backend override other than batch/A is
    rejected (no SageMaker/HyperPod GR00T stack exists). Spot defaults OFF, matching the
    stack default (a single multi-hour fine-tune; On-Demand avoids reclaim restarts)."""
    notes: list[str] = []

    if backend_override and backend_override.lower() not in ("batch", "a"):
        raise ValueError(f"unknown GR00T backend override: {backend_override!r} — GR00T runs "
                         f"only on Pattern A (Batch, GrootPatternAStack); there is no "
                         f"SageMaker/HyperPod GR00T backend.")

    instance_type = (instance_override.replace("ml.", "")
                     if instance_override else GROOT_INSTANCE)
    if instance_type != GROOT_INSTANCE:
        notes.append(f"GR00T's deployed CE is {GROOT_INSTANCE} (1×L40S, NO g5/A10G fallback "
                     f"— 24 GB OOMs a 3B Cosmos model); {instance_type!r} only changes the "
                     f"cost estimate, not where the job actually runs.")
    if profile.num_gpus > 1:
        notes.append(f"{profile.num_gpus}-GPU GR00T needs a multi-GPU CE (e.g. g6e.12xlarge); "
                     f"the default GrootPatternAStack is single-GPU g6e.4xlarge.")
    if profile.action_horizon == GROOT_G1_ACTION_HORIZON:
        notes.append("action_horizon=50 (UNITREE_G1 full-body) set via the baked "
                     "g1_finetune.py wrapper before model build — avoids the 40/50 step-0 crash.")
    elif profile.action_horizon is None and profile.embodiment_tag.upper().startswith("UNITREE_G1"):
        notes.append("⚠️ UNITREE_G1 without action_horizon=50 will ship the 40/50 mismatch "
                     "(rollout crashes at step 0). Pass action_horizon=50.")

    instance = INSTANCES.get(instance_type)
    num_gpus = profile.num_gpus
    wall = profile.est_wall_clock_h
    if instance:
        # GR00T runs on EC2 Batch → EC2 OD anchor (never the SageMaker rate).
        price = round(instance.ec2_od * (SPOT_FACTOR if spot else 1.0), 4)
        price_od = round(instance.ec2_od, 4)
        est_cost = round(price * wall, 2)
        est_cost_od = round(price_od * wall, 2)
    else:
        price, est_cost, est_cost_od = 0.0, 0.0, 0.0
        notes.append("no price anchor for the chosen instance — estimate unavailable.")

    if budget_usd is not None and est_cost_od > budget_usd > 0:
        notes.append(f"est ${est_cost_od:.0f} exceeds budget ${budget_usd:.0f} — consider "
                     f"fewer --max-steps (steps dominate GR00T cost).")

    return Decision(
        pattern="A",
        backend="batch",
        instance_type=instance_type,
        sm_instance_type="",          # GR00T has no SageMaker backend
        num_gpus=num_gpus,
        per_device_batch=0,           # N/A for GR00T (a global batch, split by the trainer)
        spot=spot,
        expert_only=False,            # N/A for GR00T (full merged 3B, never adapter-only)
        price_per_hr=price,
        est_cost_usd=est_cost,
        est_cost_od_usd=est_cost_od,
        rationale=(f"GR00T N1.7 fine-tune → Pattern A (Batch, GrootPatternAStack) on "
                   f"{instance_type} (1×L40S); the only deployed GR00T backend"),
        axis="gr00t",
        notes=notes,
    )


def format_groot_plan(profile: GrootProfile, decision: Decision) -> str:
    """A human-readable plan summary for a GR00T N1.7 fine-tune (the GR00T analogue of
    format_plan). The intent is a dataset + embodiment + step budget."""
    d = decision
    spot_label = "Spot" if d.spot else "On-Demand"
    horizon = (f"{profile.action_horizon}" if profile.action_horizon is not None
               else "(base 40-wide head)")
    lines = [
        "=" * 64,
        f"vla-ft GR00T plan — {profile.embodiment_tag} ({profile.base_model})",
        "=" * 64,
        f"  decision   : Pattern {d.pattern} ({d.backend})  on  {d.instance_type}",
        f"  why        : {d.rationale}",
        f"  compute    : {d.num_gpus} GPU, {spot_label}  (global batch {profile.global_batch})",
        f"  action-hzn : {horizon}",
        f"  est time   : ~{profile.est_wall_clock_h:g} h  ({profile.max_steps} steps, "
        f"save every {profile.save_steps})",
        f"  est cost   : ~${d.est_cost_usd:.0f}  ({spot_label} @ ${d.price_per_hr:g}/hr)",
    ]
    if d.notes:
        lines.append("  notes      :")
        lines += [f"    - {n}" for n in d.notes]
    lines.append("  ⚠️ output is a full merged 3B HF checkpoint (never adapter-only); "
                 "cost/time are ESTIMATES anchored to the verified G1 run.")
    lines.append("=" * 64)
    return "\n".join(lines)


# ── 6. Classification (IL vs RL vs GR00T) — the orchestrator's first step ────────────

def classify(intent: dict) -> str:
    """Classify an intent dict as 'il' | 'rl' | 'gr00t' — the orchestrator's classify step.

    Deterministic and explainable (no LLM, ARCHITECTURE §3): an explicit `intent`
    field wins; otherwise an `embodiment_tag` (the NVIDIA GR00T N1.7 engine's own
    intent — a LeRobot dataset adapted to a registered embodiment) → GR00T, a `dataset`
    alone (lerobot demonstrations) → IL, and a `task` (an Isaac Lab task id) → RL.

    Why embodiment_tag is the GR00T signal: both IL (lerobot) and GR00T take a dataset,
    so the dataset alone can't disambiguate. The embodiment tag is what only the GR00T
    trainer (launch_finetune.py) consumes — a dataset + embodiment_tag is unambiguously a
    GR00T fine-tune. A dataset alone stays the (more common) lerobot IL path."""
    explicit = (intent.get("intent") or "").lower()
    if explicit in ("il", "rl", "gr00t", "groot"):
        return "gr00t" if explicit in ("gr00t", "groot") else explicit
    if intent.get("embodiment_tag"):
        return "gr00t"
    if intent.get("dataset"):
        return "il"
    if intent.get("task"):
        return "rl"
    raise ValueError("cannot classify intent: provide intent='il'|'rl'|'gr00t', or a "
                     "'dataset' (IL) / 'dataset'+'embodiment_tag' (GR00T) / 'task' (RL) field.")
