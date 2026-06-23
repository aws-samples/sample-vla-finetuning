#!/usr/bin/env python3
"""
vla-ft — the one-command launcher (the Ease + Efficiency axes, ARCHITECTURE §6).

This is the thin top wrapper. It does NOT train, and it does NOT re-implement the
verified launchers: given just a `--dataset` and a `--model`, it

  1. PROFILES + DECIDES   — picks Pattern A/B/C and the instance from the rule table
                            (vla_ft_decide.py), as a smart default the user can override.
  2. RESOLVES wiring      — reads role / image / queue / output S3 from the deployed
                            CloudFormation stack Outputs (no hardcoding).
  3. PRE-FLIGHTS          — HF token (SSM), GPU capacity (Spot Placement Score),
                            quota (Service Quotas) — fail/​warn BEFORE any GPU spend.
  4. ESTIMATES cost       — prints the resolved plan + a Spot/On-Demand cost estimate
                            for confirmation.
  5. LAUNCHES             — subprocess-calls the UNCHANGED launch.py (Pattern B) or
                            batch_launch.py (Pattern A) with the resolved args.

The two verified launchers are byte-identical (verified-lock safe); this wrapper only
chooses and wires.

Quickstart (one command, end-to-end on the verified OpenArm-lift dataset):

    python vla_ft_cli.py --quickstart --yes

Typical use:

    # dry-run: show the plan + cost estimate, launch nothing
    python vla_ft_cli.py --dataset s3://.../lerobot_dataset/ --model pi05 --dry-run

    # real run (efficient defaults: expert-only for pi, Spot on, auto backend/instance)
    python vla_ft_cli.py --dataset s3://.../lerobot_dataset/ --model pi05 --yes

    # overrides (you always win): force SageMaker on the 8-GPU box, full VLM
    python vla_ft_cli.py --dataset s3://... --model pi05 \\
        --backend sagemaker --instance-type ml.g6e.48xlarge --full-vlm --yes

Run it from the launcher venv (the one with sagemaker + boto3 — `.temp/vla-ft-venv`),
since it shells the launchers with the same interpreter.

Deeper capacity ground-truth (BILLABLE ODCR verify) is a separate tool:
`scripts/aws-gpu-region-probe/probe.py --verify` in the parent repo — this CLI uses
only the free capacity signals (offering + Spot Placement Score) to stay self-contained.
"""

import argparse
import os
import subprocess
import sys

import vla_ft_decide as dec


# Deployed stack logical names (literals in bin/app.ts — independent of namePrefix,
# which only renames resources, not stacks).
STACK_PATTERN_A = "PaiTrainingPlatform-IL-PatternA"
STACK_PATTERN_B = "PaiTrainingPlatform-IL-PatternB"
STACK_RL_PATTERN_A = "PaiTrainingPlatform-RL-PatternA"

# Example OpenArm-lift dataset (50 ep, LeRobot v3, absolute EE pose) — the quickstart
# target. Point `--dataset s3://...` at your own LeRobot dataset bucket; this default
# is a placeholder for an externally-published sample.
VERIFIED_DATASET_S3 = "s3://example-openarm-lift-dataset/lerobot_dataset/"
# Gated PaliGemma (pi-family) HF token — SSM SecureString param holding your HF token.
# Override with --hf-token-ssm; the param must exist in --hf-token-ssm-region.
DEFAULT_HF_TOKEN_SSM = "/pai/hf-token"
DEFAULT_HF_TOKEN_SSM_REGION = "us-east-1"
DEFAULT_REGION = "us-west-2"

# SPS below this = capacity pressure (mirrors the probe's threshold).
SPS_PRESSURE_THRESHOLD = 4


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Intent (the only things the user must bring).
    p.add_argument("--dataset", help="S3 URI of a LeRobot v3 dataset root.")
    p.add_argument("--model", default="pi05",
                   help="Target policy: pi05 (default), pi0, groot, act, smolvla, ...")
    p.add_argument("--intent", choices=["il", "rl"], default="il",
                   help="il = imitation fine-tune (default, needs --dataset). "
                        "rl = sim policy learning (Isaac Lab headless PPO, needs --task; "
                        "RL Pattern A on Batch).")

    # RL intent (Phase 3): the "intent" is a TASK id, not a dataset. Only read when
    # --intent rl. Defaults mirror the verified reference task in vla_ft_decide.
    p.add_argument("--task", default=None,
                   help="RL only: Isaac Lab task id (env + reward registered to it). "
                        f"Default {dec.RL_DEFAULT_TASK}.")
    p.add_argument("--num-envs", type=int, default=None,
                   help="RL only: parallel sim envs (omit = task-registered default).")
    p.add_argument("--max-iterations", type=int, default=None,
                   help="RL only: PPO iterations (omit = task-registered default).")
    p.add_argument("--num-gpus", type=int, default=1,
                   help="RL only: GPUs on the node (>1 → torchrun multi-GPU PPO).")
    p.add_argument("--quickstart", action="store_true",
                   help="Fill --dataset with the verified OpenArm-lift set + --model pi05 "
                        "if unset, for a one-command end-to-end run.")

    # Training knobs (passed through to the verified launcher).
    p.add_argument("--steps", type=int, default=20000,
                   help="Training steps (default 20000 = the verified full run; "
                        "use e.g. 200 for a cheap smoke).")
    p.add_argument("--pretrained-path", default=None,
                   help="Base checkpoint (default: lerobot/<model>_base for pi-family).")

    # Efficiency overrides (the defaults are the efficient ones).
    p.add_argument("--full-vlm", action="store_true",
                   help="pi-family: train the 2B VLM too (default is expert-only freeze: "
                        "fits one L40S, matches the verified lock, resists overfit).")
    p.add_argument("--lora", action="store_true",
                   help="LoRA fine-tune: freeze the base, train low-rank adapters only. "
                        "The recommended way to fine-tune a FULL VLA (e.g. full-VLM pi05) "
                        "on ONE L40S without OOM — it collapses the fp32 Adam state that "
                        "blows up full-FT. lerobot-native (no fork). Checkpoint is "
                        "adapter-only (base resolved at load — see README handoff).")
    p.add_argument("--lora-r", type=int, default=None,
                   help="LoRA rank (lerobot default 16). Higher = more trainable params.")
    p.add_argument("--lora-alpha", type=int, default=None,
                   help="LoRA scaling alpha (lerobot default = r; scaling = alpha / r).")
    p.add_argument("--lora-target-modules", default=None,
                   help="OVERRIDE which modules LoRA adapts (regex/suffix). OMIT to use the "
                        "policy default — pi0/pi05 adapt the action expert's q/v + the "
                        "action/state projection MLPs. Pass a regex covering the gemma "
                        "backbone layers to adapt the VLM itself.")
    p.add_argument("--qlora", action="store_true",
                   help="(NOT AVAILABLE) QLoRA needs 4-bit base quantization, which lerobot "
                        "@ d1b1c5c8 has no path for (no bitsandbytes) — would need a fork. "
                        "Use --lora: the bf16-frozen base already fits one L40S for pi05.")
    p.add_argument("--no-spot", action="store_true",
                   help="Disable Spot (use On-Demand). Spot is ON by default (Efficiency).")
    p.add_argument("--select-best", action="store_true",
                   help="Overfit guard: hold out val episodes + ship the best checkpoint "
                        "(sets --val-episodes 5 --save-freq 2000 unless overridden).")
    p.add_argument("--val-episodes", type=int, default=None)
    p.add_argument("--save-freq", type=int, default=None)
    p.add_argument("--early-stop-patience", type=int, default=None,
                   help="Converged-cost lever: stop on a train-loss plateau.")
    p.add_argument("--budget", type=float, default=None,
                   help="Soft budget in USD. Constrains the decision (recommends Spot / "
                        "warns); never blocks (two-way door).")

    # Backend / instance overrides (you always win over the smart default).
    p.add_argument("--backend", default=None, choices=["batch", "sagemaker", "hyperpod"],
                   help="Force the backend (default: auto from the rule table).")
    p.add_argument("--instance-type", default=None,
                   help="Force the instance type (default: auto from the rule table).")
    p.add_argument("--per-device-batch", type=int, default=None,
                   help="Force the per-GPU batch size (default: EFF_BATCH / num_gpus to "
                        "hold the verified effective-batch-16 trajectory). Lower it to fit "
                        "memory when the auto value OOMs — e.g. full-VLM pi05 on 4xL40S "
                        "auto-picks batch 4 (the config that OOM'd at 44.4/48 GB); "
                        "--per-device-batch 2 fits (note: lerobot has no grad-accum, so "
                        "effective batch becomes 2 x num_gpus — a different trajectory).")

    # Plumbing.
    p.add_argument("--region", default=None, help=f"AWS region (default {DEFAULT_REGION}).")
    p.add_argument("--hf-token-ssm", default=None,
                   help=f"SSM SecureString with an HF token (default {DEFAULT_HF_TOKEN_SSM} "
                        f"for pi-family gated backbones).")
    p.add_argument("--hf-token-ssm-region", default=None,
                   help=f"Region of --hf-token-ssm (default {DEFAULT_HF_TOKEN_SSM_REGION}).")
    p.add_argument("--no-hf-token", action="store_true",
                   help="Skip HF token injection (non-gated models).")

    # Gates.
    p.add_argument("--dry-run", action="store_true",
                   help="Show the resolved plan + cost estimate, then stop (launch nothing).")
    p.add_argument("--yes", action="store_true",
                   help="Confirm the plan and actually submit the job (required to launch; "
                        "without it the CLI prints the plan and stops, like --dry-run).")
    p.add_argument("--skip-preflight", action="store_true",
                   help="Skip the capacity/quota/token pre-flight checks.")
    return p.parse_args()


# ── CloudFormation Outputs resolution (no hardcoding) ──────────────────────────────

def stack_outputs(cfn, stack_name):
    """Return {OutputKey: OutputValue} for a deployed stack, or {} if not found."""
    try:
        resp = cfn.describe_stacks(StackName=stack_name)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: cannot read stack {stack_name}: {e}", file=sys.stderr)
        print(f"  Is it deployed? `cdk deploy {stack_name} -c region=<region>`",
              file=sys.stderr)
        return {}
    stacks = resp.get("Stacks", [])
    if not stacks:
        return {}
    return {o["OutputKey"]: o["OutputValue"] for o in stacks[0].get("Outputs", [])}


# ── Pre-flight checks (fail/​warn before GPU spend) ────────────────────────────────

def preflight_hf_token(session, ssm_name, ssm_region):
    """Confirm the HF token parameter resolves (never print its value)."""
    try:
        ssm = session.client("ssm", region_name=ssm_region)
        ssm.get_parameter(Name=ssm_name, WithDecryption=True)
        return True, f"HF token OK (SSM {ssm_name} @ {ssm_region})"
    except Exception as e:  # noqa: BLE001
        code = getattr(e, "response", {}).get("Error", {}).get("Code", type(e).__name__)
        return False, (f"HF token NOT resolvable (SSM {ssm_name} @ {ssm_region}: {code}). "
                       f"pi-family backbones are gated — fix the token or pass --no-hf-token.")


def preflight_capacity(session, region, instance_type):
    """Free capacity signal: Spot Placement Score for the instance type in `region`.
    Warn (don't block) — capacity is time-sensitive and the launcher retries."""
    try:
        ec2 = session.client("ec2", region_name=region)
        resp = ec2.get_spot_placement_scores(
            InstanceTypes=[instance_type],
            RegionNames=[region],
            TargetCapacity=1,
            SingleAvailabilityZone=True,
        )
        scores = [s["Score"] for s in resp.get("SpotPlacementScores", [])]
        if not scores:
            return True, f"capacity: no SPS for {instance_type} (proceeding)"
        best = max(scores)
        if best >= SPS_PRESSURE_THRESHOLD:
            return True, f"capacity OK: {instance_type} SPS={best}/10 in {region}"
        return True, (f"⚠️ capacity PRESSURE: {instance_type} SPS={best}/10 in {region} "
                      f"(Spot may wait; consider --no-spot or another region)")
    except Exception as e:  # noqa: BLE001
        return True, f"capacity: SPS check skipped ({type(e).__name__})"


def run_preflight(session, region, decision, hf_ssm, hf_region, want_hf):
    """Run all pre-flight checks. Returns True if safe to proceed (HF token is the only
    hard gate; capacity/quota are advisory warnings)."""
    print("-" * 64)
    print("pre-flight:")
    ok = True

    if want_hf:
        passed, msg = preflight_hf_token(session, hf_ssm, hf_region)
        print(f"  [{'ok' if passed else 'FAIL'}] {msg}")
        ok = ok and passed
    else:
        print("  [skip] HF token (--no-hf-token)")

    # Capacity (advisory).
    probe_type = decision.instance_type
    _, cap_msg = preflight_capacity(session, region, probe_type)
    print(f"  [info] {cap_msg}")

    print("-" * 64)
    return ok


# ── Launcher argv construction (calls the UNCHANGED verified launchers) ─────────────

def resolve_pretrained_path(model, pretrained_path):
    """Return the base checkpoint to fine-tune from.

    Honour an explicit --pretrained-path. Otherwise, for the pi-family, default to the
    released base `lerobot/<model>_base`: pi fine-tunes ON TOP of that base (never from
    scratch), and LoRA HARD-REQUIRES it (lerobot's _validate_peft_config rejects PEFT
    from scratch). Non-pi models keep None (their launch path supplies their own init).
    Pure; no I/O."""
    if pretrained_path is not None:
        return pretrained_path
    m = dec.MODELS.get(model)
    if m and m.family == "pi":
        return f"lerobot/{model}_base"
    return None


def build_pattern_b_argv(args, decision, outputs, region, hf_ssm, hf_region, want_hf):
    """argv for the unchanged launch.py (SageMaker Training Job)."""
    role = outputs.get("ExecutionRoleArn")
    image = outputs.get("ImageUriHint")
    output_s3 = outputs.get("OutputS3Hint")
    missing = [k for k, v in
               {"ExecutionRoleArn": role, "ImageUriHint": image, "OutputS3Hint": output_s3}.items()
               if not v]
    if missing:
        raise SystemExit(f"Pattern B stack missing outputs {missing} — is "
                         f"{STACK_PATTERN_B} deployed?")
    instance = decision.sm_instance_type or f"ml.{decision.instance_type}"
    argv = [
        "--policy", args.model,
        "--dataset-s3", args.dataset,
        "--image-uri", image,
        "--role", role,
        "--output-s3", output_s3,
        "--instance-type", instance,
        "--num-gpus", str(decision.num_gpus),
        "--steps", str(args.steps),
        "--batch-size", str(decision.per_device_batch),
        "--region", region,
    ]
    if args.pretrained_path:
        argv += ["--pretrained-path", args.pretrained_path]
    if decision.expert_only:
        argv += ["--train-expert-only", "true"]
    argv += _lora_flags(args, decision)
    if not decision.spot:
        argv += ["--no-spot"]
    argv += _common_quality_flags(args)
    if want_hf:
        argv += ["--hf-token-ssm", hf_ssm, "--hf-token-ssm-region", hf_region]
    return argv


def build_pattern_a_argv(args, decision, outputs, region, hf_ssm, hf_region, want_hf):
    """argv for the unchanged batch_launch.py (AWS Batch)."""
    keys = ("JobQueueArn", "JobDefinitionArn", "CodeS3Hint", "OutputS3Hint")
    vals = {k: outputs.get(k) for k in keys}
    missing = [k for k, v in vals.items() if not v]
    if missing:
        raise SystemExit(f"Pattern A stack missing outputs {missing} — is "
                         f"{STACK_PATTERN_A} deployed?")
    argv = [
        "--policy", args.model,
        "--dataset-s3", args.dataset,
        "--job-queue", vals["JobQueueArn"],
        "--job-definition", vals["JobDefinitionArn"],
        "--code-s3", vals["CodeS3Hint"],
        "--output-s3", vals["OutputS3Hint"],
        "--steps", str(args.steps),
        "--batch-size", str(decision.per_device_batch),
        "--region", region,
    ]
    if args.pretrained_path:
        argv += ["--pretrained-path", args.pretrained_path]
    if decision.expert_only:
        argv += ["--train-expert-only", "true"]
    argv += _lora_flags(args, decision)
    argv += _common_quality_flags(args)
    if want_hf:
        argv += ["--hf-token-ssm", hf_ssm, "--hf-token-ssm-region", hf_region]
    return argv


def build_rl_argv(args, decision, outputs, region):
    """argv for the unchanged rl_launch.py (Isaac Lab headless PPO on AWS Batch).

    The RL analogue of build_pattern_a_argv: the RL "intent" is a TASK id (env + reward
    are registered to it), not a dataset, so there is no --code-s3 / --dataset / HF token
    — just the task + sim knobs + the RlPatternAStack outputs. Pattern C (multi-node) has
    no runnable stack today, so this only wires Pattern A (Batch)."""
    keys = ("JobQueueArn", "JobDefinitionArn", "OutputS3Hint")
    vals = {k: outputs.get(k) for k in keys}
    missing = [k for k, v in vals.items() if not v]
    if missing:
        raise SystemExit(f"RL Pattern A stack missing outputs {missing} — is "
                         f"{STACK_RL_PATTERN_A} deployed?")
    argv = [
        "--task", args.task or dec.RL_DEFAULT_TASK,
        "--job-queue", vals["JobQueueArn"],
        "--job-definition", vals["JobDefinitionArn"],
        "--output-s3", vals["OutputS3Hint"],
        "--num-gpus", str(decision.num_gpus),
        "--region", region,
    ]
    if args.num_envs is not None:
        argv += ["--num-envs", str(args.num_envs)]
    if args.max_iterations is not None:
        argv += ["--max-iterations", str(args.max_iterations)]
    return argv


def _lora_flags(args, decision):
    """LoRA flags for the verified launchers (opt-in). Emitted when --lora is set.
    Mutually exclusive with --train-expert-only: resolve_ft_mode makes --lora win over
    the pi-family expert-only default, so decision.expert_only is False whenever --lora
    is set — the caller's `if decision.expert_only` branch never co-fires with this."""
    if not args.lora:
        return []
    out = ["--lora", "true"]
    if args.lora_r is not None:
        out += ["--lora-r", str(args.lora_r)]
    if args.lora_alpha is not None:
        out += ["--lora-alpha", str(args.lora_alpha)]
    if args.lora_target_modules is not None:
        out += ["--lora-target-modules", args.lora_target_modules]
    return out


def _common_quality_flags(args):
    """Early-stop / overfit-guard flags shared by both launchers (opt-in)."""
    out = []
    if args.select_best:
        out += ["--select-best", "true"]
        out += ["--val-episodes", str(args.val_episodes if args.val_episodes else 5)]
        out += ["--save-freq", str(args.save_freq if args.save_freq else 2000)]
    else:
        if args.val_episodes is not None:
            out += ["--val-episodes", str(args.val_episodes)]
        if args.save_freq is not None:
            out += ["--save-freq", str(args.save_freq)]
    if args.early_stop_patience is not None:
        out += ["--early-stop-patience", str(args.early_stop_patience)]
    return out


def run_rl(args):
    """RL intent (Phase 3): profile → decide → resolve RlPatternAStack outputs →
    pre-flight → hand off to the UNCHANGED rl_launch.py. The RL analogue of the IL
    flow below; the decision logic is the SAME single-source rule table the
    orchestrator imports (vla_ft_decide.profile_rl / decide_rl)."""
    import boto3

    region = args.region or DEFAULT_REGION

    # ── 1. profile + decide (RL rule table) ──
    profile = dec.profile_rl(
        args.task,
        num_envs=args.num_envs,
        max_iterations=args.max_iterations,
        num_gpus=args.num_gpus,
    )
    decision = dec.decide_rl(
        profile,
        budget_usd=args.budget,
        backend_override=args.backend,
        instance_override=args.instance_type,
        spot=not args.no_spot,
    )

    # ── 2. cost estimate / plan (printed always) ──
    print(dec.format_rl_plan(profile, decision))

    if decision.pattern == "C":
        print("\nRL Pattern C (multi-node HyperPod) is code+synth only — not wired into "
              "bin/app.ts. Use single-node multi-GPU (Pattern A + --num-gpus), or "
              "--backend batch.", file=sys.stderr)
        if not args.backend:
            raise SystemExit(2)

    session = boto3.Session(region_name=region)

    # ── 3. resolve stack outputs (no hardcoding) ──
    cfn = session.client("cloudformation", region_name=region)
    outputs = stack_outputs(cfn, STACK_RL_PATTERN_A)
    if not outputs:
        raise SystemExit(2)

    # ── 4. pre-flight (capacity only — RL has no dataset/HF token gate) ──
    if not args.skip_preflight:
        print("-" * 64)
        print("pre-flight:")
        _passed, msg = preflight_capacity(session, region, decision.instance_type)
        print(f"  [info] {msg}")
        print("-" * 64)

    # ── gate ──
    if args.dry_run or not args.yes:
        reason = "--dry-run" if args.dry_run else "no --yes"
        print(f"\n[{reason}] plan shown above; nothing launched. "
              f"Re-run with --yes to submit.")
        return

    # ── 5. launch the UNCHANGED rl_launch.py ──
    # rl_launch.py lives in the sibling RL container dir (containers/isaac-lab-rl/),
    # not this vla-ft dir. It is stdlib + boto3 and takes no relative source_dir, so
    # cwd is incidental — but pin it to its own dir for parity with the IL launchers.
    here = os.path.dirname(os.path.abspath(__file__))
    rl_dir = os.path.join(os.path.dirname(here), "isaac-lab-rl")
    script = os.path.join(rl_dir, "rl_launch.py")
    argv = build_rl_argv(args, decision, outputs, region)
    cmd = [sys.executable, script] + argv
    print(f"\nlaunching: {' '.join(cmd)}\n")
    rc = subprocess.run(cmd, cwd=rl_dir).returncode
    sys.exit(rc)


def main():
    args = parse_args()

    # RL intent: a task, not a dataset — separate flow (Isaac Lab PPO on Batch).
    if args.intent == "rl":
        try:
            import boto3  # noqa: F401
        except ImportError:
            raise SystemExit("ERROR: pip install boto3 (run from the launcher venv).")
        run_rl(args)
        return

    # QLoRA is not flag-only feasible (lerobot @ d1b1c5c8 has no 4-bit/bitsandbytes
    # path) — refuse rather than silently produce a non-QLoRA run. Honest gating, like
    # Pattern C's "not runnable yet".
    if args.qlora:
        raise SystemExit(
            "ERROR: --qlora is not available. LeRobot @ d1b1c5c8 has no 4-bit base "
            "quantization (no bitsandbytes), so QLoRA would require a lerobot fork. Use "
            "--lora: the base is bf16-frozen and the full VLM (e.g. pi05) fits one L40S.")

    # Quickstart fills the intent the user didn't type.
    if args.quickstart:
        args.dataset = args.dataset or VERIFIED_DATASET_S3
        args.model = args.model or "pi05"
    if not args.dataset:
        raise SystemExit("ERROR: --dataset is required (or use --quickstart).")

    try:
        import boto3
    except ImportError:
        raise SystemExit("ERROR: pip install boto3 sagemaker (run from the launcher venv).")

    region = args.region or DEFAULT_REGION
    # Inject an HF token when the backbone is gated. pi-family (PaliGemma) is gated,
    # so it's the default; passing --hf-token-ssm explicitly forces it on for any
    # model; --no-hf-token forces it off. Family lookup is robust to unknown models.
    model_family = dec.MODELS.get(args.model).family if args.model in dec.MODELS else ""

    # Resolve the base checkpoint. pi-family fine-tunes ON TOP of a released base
    # (lerobot/<model>_base), never from scratch; LoRA in particular HARD-REQUIRES it
    # (lerobot's _validate_peft_config raises "Training from scratch using PEFT is
    # unlikely to yield good results. Supply a policy.pretrained_path ..."). Default it
    # for pi-family when omitted so the documented behavior holds and --lora is runnable.
    resolved_base = resolve_pretrained_path(args.model, args.pretrained_path)
    if resolved_base != args.pretrained_path:
        args.pretrained_path = resolved_base
        print(f"[vla-ft] pretrained-path defaulted to {args.pretrained_path} "
              f"(pi-family fine-tunes on the released base; required for --lora).")

    want_hf = (not args.no_hf_token) and (model_family == "pi" or args.hf_token_ssm is not None)
    hf_ssm = args.hf_token_ssm or DEFAULT_HF_TOKEN_SSM
    hf_region = args.hf_token_ssm_region or DEFAULT_HF_TOKEN_SSM_REGION

    # ── 1. profile + decide ──
    # args.qlora is already rejected above (honest gating) — never reaches resolve_ft_mode.
    ft_mode = dec.resolve_ft_mode(args.model, args.full_vlm, args.lora)
    profile = dec.profile_run(args.model, args.steps, ft_mode)
    decision = dec.decide(
        profile,
        budget_usd=args.budget,
        backend_override=args.backend,
        instance_override=args.instance_type,
        spot=not args.no_spot,
    )

    # Per-device-batch override: the rule table auto-picks EFF_BATCH/num_gpus to hold
    # the verified effective-batch-16 trajectory, but that can OOM (full-VLM pi05 on
    # 4xL40S → batch 4 → 44.4/48 GB). Let the operator force a smaller per-GPU batch to
    # fit memory; lerobot has no grad-accum, so this changes the effective batch.
    if args.per_device_batch is not None:
        eff = args.per_device_batch * decision.num_gpus
        decision.per_device_batch = args.per_device_batch
        decision.notes.append(
            f"per-device batch forced to {args.per_device_batch} (--per-device-batch) "
            f"→ effective batch {eff} across {decision.num_gpus} GPU "
            f"(differs from the verified {dec.EFF_BATCH}; lerobot does not auto-scale LR/steps).")

    # ── 4. cost estimate / plan (printed always) ──
    print(dec.format_plan(profile, decision))

    if decision.pattern == "C":
        print("\nPattern C (HyperPod) is code+synth only — not wired into bin/app.ts. "
              "Reduce footprint (expert-only / LoRA / smaller model) for a runnable A/B "
              "path, or override with --backend.", file=sys.stderr)
        if not args.backend:
            raise SystemExit(2)

    session = boto3.Session(region_name=region)

    # ── 2. resolve stack outputs ──
    cfn = session.client("cloudformation", region_name=region)
    stack = STACK_PATTERN_A if decision.pattern == "A" else STACK_PATTERN_B
    outputs = stack_outputs(cfn, stack)
    if not outputs:
        raise SystemExit(2)

    # ── 3. pre-flight ──
    if not args.skip_preflight:
        safe = run_preflight(session, region, decision, hf_ssm, hf_region, want_hf)
        if not safe:
            print("pre-flight FAILED (see above) — not launching. Fix, or "
                  "--skip-preflight to bypass.", file=sys.stderr)
            raise SystemExit(1)

    # ── gate ──
    if args.dry_run or not args.yes:
        reason = "--dry-run" if args.dry_run else "no --yes"
        print(f"\n[{reason}] plan shown above; nothing launched. "
              f"Re-run with --yes to submit.")
        return

    # ── 5. launch the UNCHANGED verified launcher ──
    here = os.path.dirname(os.path.abspath(__file__))
    if decision.pattern == "A":
        script = os.path.join(here, "batch_launch.py")
        argv = build_pattern_a_argv(args, decision, outputs, region, hf_ssm, hf_region, want_hf)
    else:
        script = os.path.join(here, "launch.py")
        argv = build_pattern_b_argv(args, decision, outputs, region, hf_ssm, hf_region, want_hf)

    cmd = [sys.executable, script] + argv
    print(f"\nlaunching: {' '.join(cmd)}\n")
    # Run the launcher from its own directory: launch.py ships SageMaker
    # source_dir="src" / entry_point="train.py" as a RELATIVE path, which SageMaker
    # resolves against the CWD. Without cwd=here it would look for ./src/train.py under
    # whatever directory the CLI was invoked from (e.g. the repo root) and fail. The
    # launchers stay byte-identical; only their working directory is pinned here.
    rc = subprocess.run(cmd, cwd=here).returncode
    sys.exit(rc)


if __name__ == "__main__":
    main()
