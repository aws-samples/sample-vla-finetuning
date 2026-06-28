#!/usr/bin/env python3
"""
Self-contained tests for the vla-ft decision logic + CLI argv construction.

No pytest dependency (the Python side of this repo is verified by py_compile +
standalone asserts; TS uses jest). Run:  python3 test_vla_ft_decide.py
Exits non-zero on the first failure. Covers the rule table (VRAM→pattern→instance),
overrides, the cost estimate's anchor, and that the CLI builds argv for the UNCHANGED
launchers (no network — the pure functions only).
"""

import argparse
import sys

import vla_ft_decide as dec
import vla_ft_cli as cli


PASS, FAIL = 0, 0


def check(cond, label):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {label}")
    else:
        FAIL += 1
        print(f"  FAIL {label}")


def _run(model, steps, full_vlm=False, lora=False, **decide_kw):
    mode = dec.resolve_ft_mode(model, full_vlm, lora)
    p = dec.profile_run(model, steps, mode)
    return p, dec.decide(p, **decide_kw)


# ── rule table: pattern selection ──
def test_rule_table():
    print("rule table:")
    # pi05 default = expert-only (~22.6 GB), short → A; long → B
    p, d = _run("pi05", 200)
    check(p.ft_mode == "expert_only", "pi05 default → expert-only freeze")
    check(d.pattern == "A" and d.instance_type == "g6e.4xlarge",
          "pi05 expert-only + 200 steps (short) → Pattern A g6e.4xlarge")

    p, d = _run("pi05", 20000)
    check(d.pattern == "B" and d.instance_type == "g6e.12xlarge",
          "pi05 expert-only + 20000 steps (long) → Pattern B g6e.12xlarge")
    check(d.num_gpus == 4 and d.per_device_batch == 4,
          "Pattern B g6e.12xl → 4 GPU, per-device batch 4 (eff batch 16)")

    # full-VLM pi05 = full_ft (3.3×12=39.6 GB) ≤ 48, long → B
    p, d = _run("pi05", 20000, full_vlm=True)
    check(p.ft_mode == "full_ft" and p.vram_per_gpu_gb == 39.6,
          "pi05 --full-vlm → full_ft, ~39.6 GB/GPU (best-practice formula)")
    check(d.pattern == "B", "full-VLM pi05 fits one L40S (39.6≤48) → Pattern B")

    # a replica > one L40S → Pattern C (FSDP/HyperPod)
    big = dec.Profile("big", 5.0, "full_ft", 60.0, 20000, 5.5)
    d = dec.decide(big)
    check(d.pattern == "C", "5B full_ft (~60 GB > 48) → Pattern C (sharding needed)")

    # small policy → A
    p, d = _run("smolvla", 2000)
    check(d.pattern == "A", "smolvla (~5.4 GB, short) → Pattern A")


# ── overrides (the user always wins) ──
def test_overrides():
    print("overrides:")
    p, d = _run("pi05", 200, backend_override="sagemaker")
    check(d.pattern == "B", "--backend sagemaker forces Pattern B even when A would fit")

    p, d = _run("pi05", 200, instance_override="ml.g6e.48xlarge")
    check(d.instance_type == "g6e.48xlarge",
          "--instance-type ml.g6e.48xlarge honored (ml. stripped)")

    p, d = _run("pi05", 20000, lora=True)
    check(p.ft_mode == "lora", "--lora overrides the expert-only default")
    # No qlora mode: the engine never resolves to "qlora" (lerobot @ d1b1c5c8 has no 4-bit
    # path), and VRAM_MULT carries no qlora multiplier — honest gating at the front door.
    check("qlora" not in dec.VRAM_MULT, "no qlora VRAM multiplier (4-bit unsupported)")


# ── LoRA: the recommended full-VLM-on-one-GPU path (lerobot-native, no fork) ──
def test_lora():
    print("LoRA (full VLM on one GPU, fork-free):")
    # LoRA VRAM = params_b × 2.0 (best-practice 'lora' multiplier). For pi05: 3.3×2 = 6.6 GB,
    # FAR under one L40S — so even the FULL VLM fits without OOM (vs full_ft's 39.6 GB that
    # OOM'd in practice at 44.4 GB). This is the whole point: LoRA collapses the fp32 Adam
    # state that made full-VLM full-FT impossible on an L40S.
    p, d = _run("pi05", 20000, full_vlm=True, lora=True)
    check(p.ft_mode == "lora", "--lora wins over --full-vlm (lora is the freeze strategy)")
    check(abs(p.vram_per_gpu_gb - 6.6) < 0.1, f"LoRA pi05 ~6.6 GB/GPU (got {p.vram_per_gpu_gb})")
    check(d.pattern in ("A", "B") and d.pattern != "C",
          "LoRA pi05 fits one L40S (6.6≤48) → never Pattern C (no sharding needed)")
    check(d.expert_only is False, "LoRA decision is not expert-only (different freeze)")

    # CLI argv: --lora emits the launcher flag AND does NOT co-emit --train-expert-only
    # (they're mutually exclusive — both freeze the VLM). r/alpha optional overrides.
    out_b = {"ExecutionRoleArn": "arn:role", "ImageUriHint": "img:latest",
             "OutputS3Hint": "s3://art/vla-ft"}
    argv = cli.build_pattern_b_argv(
        _fake_args(lora=True, lora_r=32, lora_alpha=64), d, out_b, "us-west-2",
        "/pai/hf-token", "us-east-1", want_hf=True)
    s = " ".join(argv)
    check("--lora true" in s, "B argv: --lora true emitted")
    check("--lora-r 32" in s and "--lora-alpha 64" in s, "B argv: lora r/alpha overrides wired")
    check("--train-expert-only true" not in s, "B argv: LoRA does NOT co-emit expert-only")

    # Pattern A path emits the same LoRA flags. r/alpha omitted → lerobot defaults (no flag).
    p_a, d_a = _run("pi05", 200, lora=True)  # short → A
    out_a = {"JobQueueArn": "arn:q", "JobDefinitionArn": "arn:jd",
             "CodeS3Hint": "s3://art/code/train.py", "OutputS3Hint": "s3://art/vla-ft"}
    argv_a = cli.build_pattern_a_argv(_fake_args(lora=True, steps=200), d_a, out_a,
                                      "us-west-2", "/x", "us-east-1", want_hf=False)
    sa = " ".join(argv_a)
    check("--lora true" in sa, "A argv: --lora true emitted")
    check("--lora-r" not in sa and "--lora-alpha" not in sa,
          "A argv: r/alpha omitted when unset (lerobot defaults apply)")
    check("--train-expert-only true" not in sa, "A argv: LoRA does NOT co-emit expert-only")

    # Default (no --lora) emits NO lora flags → verified path byte-identical.
    argv_def = cli.build_pattern_b_argv(_fake_args(), _run("pi05", 20000)[1], out_b,
                                        "us-west-2", "/x", "us-east-1", want_hf=False)
    check("--lora" not in " ".join(argv_def), "default (no --lora) emits no LoRA flags")

    # pretrained-path resolution: LoRA HARD-REQUIRES a base (lerobot rejects PEFT from
    # scratch), so pi-family must default to lerobot/<model>_base when omitted.
    check(cli.resolve_pretrained_path("pi05", None) == "lerobot/pi05_base",
          "pi05 base defaults to lerobot/pi05_base when --pretrained-path omitted")
    check(cli.resolve_pretrained_path("pi0", None) == "lerobot/pi0_base",
          "pi0 base defaults to lerobot/pi0_base")
    check(cli.resolve_pretrained_path("pi05", "s3://my/ckpt/") == "s3://my/ckpt/",
          "explicit --pretrained-path is honoured (not overridden)")
    check(cli.resolve_pretrained_path("act", None) is None,
          "non-pi (act) keeps None (its launch path supplies its own init)")


# ── cost estimate anchor ──
def test_cost_anchor():
    print("cost estimate (anchored to the verified run):")
    # Verified full run: g6e.12xl, 20000 steps, ~5.5 h, SageMaker OD $13.12/hr ≈ $72.
    p, d = _run("pi05", 20000)
    check(70 <= d.est_cost_od_usd <= 76,
          f"OD estimate ${d.est_cost_od_usd} ≈ verified $72 (5.5h × $13.12 SM ml.g6e.12xl)")
    check(d.est_cost_usd < d.est_cost_od_usd,
          f"Spot estimate ${d.est_cost_usd} < On-Demand ${d.est_cost_od_usd}")
    check(d.spot is True, "Spot is the default")

    # Pattern A real run anchor: g6e.4xl Spot, steps 200 ⇒ tiny.
    p, d = _run("pi05", 200)
    check(d.est_cost_usd < 1.0, f"200-step Pattern A estimate ${d.est_cost_usd} < $1")


# ── budget constraint (warns, never blocks) ──
def test_budget():
    print("budget constraint:")
    p, d = _run("pi05", 20000, budget_usd=10.0)
    has_warn = any("budget" in n or "exceed" in n for n in d.notes)
    check(has_warn, "tight budget produces a warning note")
    check(d.pattern == "B", "budget does NOT change the capability decision (still B)")


# ── CLI argv construction (pure; no network) ──
def _fake_args(**kw):
    base = dict(model="pi05", dataset="s3://b/ds/", pretrained_path=None, steps=20000,
                select_best=False, val_episodes=None, save_freq=None,
                early_stop_patience=None,
                # LoRA flags (only emitted into argv when args.lora is True).
                lora=False, lora_r=None, lora_alpha=None, lora_target_modules=None,
                # RL intent fields (only read by the RL path / build_rl_argv).
                task=None, num_envs=None, max_iterations=None, num_gpus=1)
    base.update(kw)
    return argparse.Namespace(**base)


def test_cli_argv():
    print("CLI argv (calls the UNCHANGED launchers):")
    p, d = _run("pi05", 20000)  # → Pattern B, expert-only, spot
    out_b = {"ExecutionRoleArn": "arn:role", "ImageUriHint": "img:latest",
             "OutputS3Hint": "s3://art/vla-ft"}
    argv = cli.build_pattern_b_argv(_fake_args(), d, out_b, "us-west-2",
                                    "/pai/hf-token", "us-east-1", want_hf=True)
    s = " ".join(argv)
    check("--policy pi05" in s, "B argv: --policy pi05")
    check("--train-expert-only true" in s, "B argv: expert-only flag emitted")
    check("--instance-type ml.g6e.12xlarge" in s, "B argv: ml.-prefixed instance")
    check("--batch-size 4" in s, "B argv: per-device batch 4")
    check("--hf-token-ssm /pai/hf-token" in s, "B argv: HF token wired")
    check("--no-spot" not in s, "B argv: Spot default (no --no-spot)")

    # Pattern A argv from PatternA outputs.
    p, d = _run("pi05", 200)  # → Pattern A
    out_a = {"JobQueueArn": "arn:q", "JobDefinitionArn": "arn:jd",
             "CodeS3Hint": "s3://art/vla-ft-code/train.py", "OutputS3Hint": "s3://art/vla-ft"}
    argv = cli.build_pattern_a_argv(_fake_args(steps=200), d, out_a, "us-west-2",
                                    "/pai/hf-token", "us-east-1", want_hf=True)
    s = " ".join(argv)
    check("--job-queue arn:q" in s, "A argv: job-queue from stack output")
    check("--job-definition arn:jd" in s, "A argv: job-definition from stack output")
    check("--code-s3 s3://art/vla-ft-code/train.py" in s, "A argv: code-s3 from output")

    # select-best wires the overfit-guard trio.
    argv = cli.build_pattern_b_argv(_fake_args(select_best=True), d if d.pattern == "B" else _run("pi05", 20000)[1],
                                    out_b, "us-west-2", "/x", "us-east-1", want_hf=False)
    s = " ".join(argv)
    check("--select-best true" in s and "--val-episodes 5" in s and "--save-freq 2000" in s,
          "--select-best wires val-episodes 5 + save-freq 2000")
    check("--hf-token-ssm" not in s, "want_hf=False omits HF token")

    # missing required output → clear failure.
    try:
        cli.build_pattern_b_argv(_fake_args(), _run("pi05", 20000)[1], {}, "us-west-2",
                                 "/x", "us-east-1", want_hf=False)
        check(False, "missing stack outputs raises SystemExit")
    except SystemExit:
        check(True, "missing stack outputs raises SystemExit")


def test_cli_rl_argv():
    """The unified CLI's RL path builds argv for the UNCHANGED rl_launch.py from the RL
    rule-table decision + RlPatternAStack outputs (the Phase 6 'IL & RL' unified-launcher
    claim — RL was previously a 'not yet runnable' raise)."""
    print("CLI RL argv (calls the UNCHANGED rl_launch.py):")
    pr = dec.profile_rl("Isaac-Velocity-Rough-H1-v0", max_iterations=3000, num_envs=4096)
    dr = dec.decide_rl(pr)
    out_rl = {"JobQueueArn": "arn:q", "JobDefinitionArn": "arn:jd",
              "OutputS3Hint": "s3://art/isaac-rl"}
    argv = cli.build_rl_argv(
        _fake_args(intent="rl", task="Isaac-Velocity-Rough-H1-v0",
                   max_iterations=3000, num_envs=4096),
        dr, out_rl, "us-west-2")
    s = " ".join(argv)
    check("--task Isaac-Velocity-Rough-H1-v0" in s, "RL argv: task wired")
    check("--job-queue arn:q" in s, "RL argv: job-queue from stack output")
    check("--job-definition arn:jd" in s, "RL argv: job-definition from stack output")
    check("--output-s3 s3://art/isaac-rl" in s, "RL argv: output-s3 from stack output")
    check("--max-iterations 3000" in s and "--num-envs 4096" in s, "RL argv: sim knobs wired")
    check("--num-gpus 1" in s, "RL argv: num-gpus from decision")
    # RL has no dataset / code-s3 / HF token.
    check("--dataset" not in s and "--code-s3" not in s and "--hf-token" not in s,
          "RL argv: no IL-only flags (dataset/code-s3/HF token)")

    # missing RL output → clear failure.
    try:
        cli.build_rl_argv(_fake_args(intent="rl"), dr, {}, "us-west-2")
        check(False, "RL: missing stack outputs raises SystemExit")
    except SystemExit:
        check(True, "RL: missing stack outputs raises SystemExit")


# ── Phase 4 orchestrator: classify + RL decide + plan/submit faithfulness ──
def test_orchestrator():
    print("orchestrator (Phase 4):")
    import orchestrator_plan as op
    import orchestrator_submit as osub

    # classify: explicit intent wins; else dataset→IL, task→RL.
    check(dec.classify({"intent": "rl", "dataset": "s3://x"}) == "rl",
          "classify: explicit intent overrides dataset")
    check(dec.classify({"dataset": "s3://x", "model": "pi05"}) == "il",
          "classify: dataset → IL")
    check(dec.classify({"task": "Isaac-Velocity-Rough-H1-v0"}) == "rl",
          "classify: task → RL")
    try:
        dec.classify({})
        check(False, "classify: empty intent raises")
    except ValueError:
        check(True, "classify: empty intent raises")

    # RL rule table: single-GPU short → A; multi-node → C; sagemaker override → A (no RL SM).
    pr = dec.profile_rl("Isaac-Velocity-Rough-H1-v0", max_iterations=3000, num_envs=4096)
    dr = dec.decide_rl(pr)
    check(dr.pattern == "A" and dr.axis == "rl", "RL: single-GPU → Pattern A, axis rl")
    check(dec.decide_rl(dec.profile_rl("t", multi_node=True)).pattern == "C",
          "RL: multi-node → Pattern C")
    check(dec.decide_rl(dec.profile_rl("t", num_gpus=4)).instance_type == "g6e.12xlarge",
          "RL: 4-GPU single node → g6e.12xlarge")
    check(dec.decide_rl(pr, backend_override="sagemaker").pattern == "A",
          "RL: sagemaker override maps to A (no RL SageMaker backend)")

    # plan(): IL pi05 20000 steps → Pattern B (5.6h > 4h), runnable; submit carries policy.
    plan_il = op.plan({"dataset": "s3://b/ds/", "model": "pi05", "steps": 20000})
    check(plan_il["axis"] == "il" and plan_il["pattern"] == "B" and plan_il["runnable"],
          "plan: IL pi05 20000 steps → Pattern B, runnable")
    check(plan_il["submit"]["policy"] == "pi05" and plan_il["submit"]["expert_only"],
          "plan: IL submit carries policy + expert-only default")

    # plan(): IL short → Pattern A (runnable). RL → Pattern A (runnable).
    check(op.plan({"dataset": "s3://b/ds/", "model": "pi05", "steps": 200})["pattern"] == "A",
          "plan: IL 200 steps → Pattern A")
    plan_rl = op.plan({"task": "Isaac-Velocity-Rough-H1-v0", "max_iterations": 3000})
    check(plan_rl["axis"] == "rl" and plan_rl["runnable"], "plan: RL → runnable Pattern A")

    # plan(): full-VLM pi05 → full_ft (3.3×12=39.6GB ≤ 48) but 5.6h > 4h → Pattern B, runnable.
    plan_fv = op.plan({"dataset": "s3://b/ds/", "model": "pi05", "full_vlm": True})
    check(plan_fv["pattern"] == "B" and plan_fv["runnable"] and not plan_fv["submit"]["expert_only"],
          "plan: full-VLM pi05 → Pattern B (full_ft 39.6GB ≤48), runnable, expert_only off")

    # plan(): LoRA carries through submit (lora True, expert_only False) — and the SM
    # handoff command emits --lora true without --train-expert-only.
    plan_lora = op.plan({"dataset": "s3://b/ds/", "model": "pi05", "lora": True,
                         "lora_r": 32, "lora_alpha": 64})
    check(plan_lora["submit"]["lora"] and not plan_lora["submit"]["expert_only"],
          "plan: --lora → submit.lora True, expert_only False")
    h_lora = osub._handoff_sagemaker(plan_lora["submit"], want_hf=True)
    cl = h_lora["handoff_command"]
    check("--lora true" in cl and "--lora-r 32" in cl and "--lora-alpha 64" in cl,
          "submit: SM handoff carries --lora + r/alpha")
    check("--train-expert-only true" not in cl, "submit: LoRA handoff omits expert-only")

    # plan(): job_name pins through submit (resume path) — omit → None (fresh run).
    plan_resume = op.plan({"dataset": "s3://b/ds/", "model": "pi05", "steps": 20000,
                           "job_name": "vla-ft-pi05-20260620-151444"})
    check(plan_resume["submit"]["job_name"] == "vla-ft-pi05-20260620-151444",
          "plan: job_name pins through IL submit (resume)")
    check(plan_il["submit"].get("job_name") is None,
          "plan: no job_name → submit.job_name None (fresh timestamped run)")

    # plan(): RL multi-node → Pattern C, NOT runnable → the Choice state routes to recommend.
    plan_c = op.plan({"task": "Isaac-Velocity-Rough-H1-v0", "multi_node": True})
    check(plan_c["pattern"] == "C" and not plan_c["runnable"],
          "plan: RL multi-node → Pattern C, NOT runnable (Choice routes to recommend)")

    # submit handoff (Pattern B): the resolved command is a FAITHFUL launch.py invocation
    # (no fork). Set env wiring; no boto3 needed for the handoff path.
    import os
    os.environ["B_IMAGE_URI"] = "428.dkr.ecr.us-west-2.amazonaws.com/pai/vla-ft:latest"
    os.environ["B_EXECUTION_ROLE"] = "arn:aws:iam::428:role/exec"
    os.environ["B_OUTPUT_S3"] = "s3://pai-artifacts/vla-ft"
    os.environ["HF_TOKEN_SSM"] = "/pai/hf-token"
    os.environ["HF_TOKEN_SSM_REGION"] = "us-east-1"
    h = osub._handoff_sagemaker(plan_il["submit"], want_hf=True)
    cmd = h["handoff_command"]
    check(h["status"] == "handoff" and "launch.py" in cmd, "submit: Pattern B → launch.py handoff (not forked)")
    check("--policy pi05" in cmd and "--instance-type ml.g6e.12xlarge" in cmd,
          "submit: handoff carries the resolved policy + ml.-prefixed instance")
    check("--train-expert-only true" in cmd and "--hf-token-ssm /pai/hf-token" in cmd,
          "submit: handoff carries expert-only + HF token (matches verified launch.py)")
    check("--batch-size 4" in cmd, "submit: handoff per-device batch = decision (eff-batch lock)")

    # ── image pin (reproducibility): a digest threads through plan → submit and PINS the
    #    Pattern B handoff's --image-uri (overriding the stack's mutable B_IMAGE_URI hint).
    DIGEST = "428.dkr.ecr.us-west-2.amazonaws.com/pai/vla-ft@sha256:574bb43"
    plan_pin = op.plan({"dataset": "s3://b/ds/", "model": "pi05", "steps": 20000,
                        "image_uri": DIGEST})
    check(plan_pin["submit"]["image_uri"] == DIGEST, "plan: image_uri threads into submit")
    check(plan_pin["pattern"] == "B" and f"image      : {DIGEST}" in plan_pin["plan_text"]
          and "launch.py --image-uri" in plan_pin["plan_text"],
          "plan: Pattern B shows the image as PINNED in plan_text")
    h_pin = osub._handoff_sagemaker(plan_pin["submit"], want_hf=True)
    check(f"--image-uri {DIGEST}" in h_pin["handoff_command"],
          "submit: caller image_uri (digest) overrides B_IMAGE_URI on the SM handoff")
    # without a pin, the handoff falls back to the stack's B_IMAGE_URI (':latest') — unchanged.
    check(":latest" in osub._handoff_sagemaker(plan_il["submit"], want_hf=True)["handoff_command"],
          "submit: no image_uri → handoff uses the stack B_IMAGE_URI (back-compat)")
    # Pattern A (Batch) cannot override the image per job → the pin is shown as ADVISORY,
    # never silently dropped (honest: Batch reads the image from the Job Definition).
    plan_pin_a = op.plan({"dataset": "s3://b/ds/", "model": "pi05", "steps": 200,
                          "image_uri": DIGEST})
    check(plan_pin_a["pattern"] == "A" and "ADVISORY" in plan_pin_a["plan_text"]
          and DIGEST in plan_pin_a["plan_text"],
          "plan: Pattern A shows image_uri as ADVISORY (Batch can't override per job)")

    # ── DDP OOM-margin warning (P4): full-VLM full-FT on 1 node × multi-GPU replicates the
    #    fp32 Adam state per GPU. WARN there; SILENT for expert_only / LoRA / single-GPU.
    plan_fv_warn = op.plan({"dataset": "s3://b/ds/", "model": "pi05", "full_vlm": True,
                            "steps": 20000})
    check(plan_fv_warn["submit"]["num_gpus"] > 1 and not plan_fv_warn["submit"]["expert_only"]
          and "WARNING" in plan_fv_warn["plan_text"] and "DDP" in plan_fv_warn["plan_text"],
          "plan: full-VLM full-FT multi-GPU 1-node → DDP OOM-margin WARNING")
    check("WARNING" not in op.plan({"dataset": "s3://b/ds/", "model": "pi05",
                                    "steps": 20000})["plan_text"],
          "plan: expert_only default (the safe path) → no DDP warning")
    check("WARNING" not in op.plan({"dataset": "s3://b/ds/", "model": "pi05", "lora": True,
                                    "steps": 20000})["plan_text"],
          "plan: LoRA (frozen base) → no DDP warning")


# ── Per-job timeout + dual-queue spot/od (the MCP-only-FT controls) ──────────────────────
def test_per_job_controls():
    """timeout_hours and spot are resolved PER JOB (no redeploy): the plan normalizes them
    into the submit dict, orchestrator_submit turns timeout_s into the SubmitJob `timeout`
    kwarg and selects the Spot vs On-Demand queue. Covers all three Batch engines."""
    print("per-job timeout + dual-queue (no-redeploy controls):")
    import orchestrator_plan as op
    import orchestrator_submit as osub
    import os

    # plan(): timeout_hours → submit.timeout_s (seconds), surfaced in plan_text. Omit → None.
    p_to = op.plan({"dataset": "s3://b/ds/", "model": "pi05", "steps": 20000, "timeout_hours": 24})
    check(p_to["submit"]["timeout_s"] == 86400, f"plan: timeout_hours=24 → 86400 s (got {p_to['submit']['timeout_s']})")
    check("timeout" in p_to["plan_text"] and "24 h" in p_to["plan_text"],
          "plan: per-job timeout shown in plan_text")
    check(op.plan({"dataset": "s3://b", "timeout_s": 3600})["submit"]["timeout_s"] == 3600,
          "plan: raw timeout_s passes through")
    check(op.plan({"dataset": "s3://b", "timeout_hours": 0.001})["submit"]["timeout_s"] == 60,
          "plan: tiny timeout floored at Batch's 60 s minimum")
    p_none = op.plan({"dataset": "s3://b/ds/", "model": "pi05", "steps": 200})
    check(p_none["submit"]["timeout_s"] is None, "plan: no timeout → submit.timeout_s None (JobDef default)")
    check("timeout    :" not in p_none["plan_text"], "plan: no timeout line when unset")
    # RL + GR00T carry the same knob.
    check(op.plan({"task": "Isaac-Velocity-Rough-H1-v0", "timeout_hours": 3})["submit"]["timeout_s"] == 10800,
          "plan: RL carries timeout_s")
    check(op.plan({"dataset": "s3://b", "embodiment_tag": "UNITREE_G1", "timeout_hours": 2})["submit"]["timeout_s"] == 7200,
          "plan: GR00T carries timeout_s")

    # _timeout_kwarg: SubmitJob shape, or {} to inherit the JobDef default.
    check(osub._timeout_kwarg({"timeout_s": 86400}) == {"timeout": {"attemptDurationSeconds": 86400}},
          "submit: timeout_s → SubmitJob timeout.attemptDurationSeconds")
    check(osub._timeout_kwarg({"timeout_s": None}) == {} and osub._timeout_kwarg({}) == {},
          "submit: no timeout_s → no timeout kwarg (JobDef default inherited)")

    # _submit_il_batch: spot selects the queue (Spot default vs On-Demand), and timeout_s
    # flows into SubmitJob. Intercept boto3 (no live AWS), capture the call.
    os.environ["IL_A_JOB_QUEUE"] = "arn:aws:batch:us-west-2:428:job-queue/il-spot"
    os.environ["IL_A_JOB_QUEUE_OD"] = "arn:aws:batch:us-west-2:428:job-queue/il-od"
    os.environ["IL_A_JOB_DEFINITION"] = "arn:aws:batch:us-west-2:428:job-definition/il:7"
    os.environ["IL_A_CODE_S3"] = "s3://pai-artifacts/vla-ft-code/train.py"
    os.environ["IL_A_OUTPUT_S3"] = "s3://pai-artifacts/vla-ft"
    os.environ.pop("HF_TOKEN_SSM", None)  # skip the HF read on this IL path

    captured = {}

    class _FakeS3:
        def upload_file(self, *a, **k): pass
        def put_object(self, **k): pass

    class _FakeBatch:
        def submit_job(self, **kw):
            captured.clear(); captured.update(kw)
            return {"jobId": "fake-il-id"}

    class _FakeSession:
        region_name = "us-west-2"
        def client(self, name, region_name=None):
            return _FakeBatch() if name == "batch" else _FakeS3()

    base_submit = {"policy": "pi05", "dataset_s3": "s3://b/ds/", "steps": 20000,
                   "per_device_batch": 16, "num_gpus": 1, "expert_only": False}

    # spot=False → On-Demand queue + timeout flows through.
    out_od = osub._submit_il_batch(_FakeSession(), "us-west-2",
                                   {**base_submit, "spot": False, "timeout_s": 86400}, want_hf=False)
    check(captured["jobQueue"].endswith("/il-od"), "submit: spot=False → On-Demand queue")
    check(captured.get("timeout") == {"attemptDurationSeconds": 86400}, "submit: timeout_s reaches SubmitJob")
    check(out_od["queue"] == "on-demand", "submit: return labels queue 'on-demand'")

    # spot=True (default) → Spot queue + no timeout kwarg when unset.
    out_sp = osub._submit_il_batch(_FakeSession(), "us-west-2", {**base_submit, "spot": True}, want_hf=False)
    check(captured["jobQueue"].endswith("/il-spot"), "submit: spot=True → Spot queue")
    check("timeout" not in captured, "submit: no timeout_s → no timeout kwarg (JobDef default)")
    check(out_sp["queue"] == "spot", "submit: return labels queue 'spot'")

    # Missing OD output (older stack) → graceful fall back to the Spot queue for spot=False.
    os.environ.pop("IL_A_JOB_QUEUE_OD", None)
    osub._submit_il_batch(_FakeSession(), "us-west-2", {**base_submit, "spot": False}, want_hf=False)
    check(captured["jobQueue"].endswith("/il-spot"),
          "submit: spot=False with no OD queue → falls back to Spot (no crash)")
    os.environ["IL_A_JOB_QUEUE_OD"] = "arn:aws:batch:us-west-2:428:job-queue/il-od"  # restore


def test_gpu_floor_guard():
    """A full-VLM replica (~40 GB) must NOT ride the Spot CE, whose g5 fallback (24 GB)
    OOMs it at step 0 — Batch has no per-job instance override, so the decision auto-routes
    such a run to the On-Demand (L40S-only) queue. Expert-only / LoRA (≤24 GB) stay on Spot,
    and an explicit instance_override is never second-guessed."""
    print("GPU-floor guard (full-VLM off the 24 GB Spot fallback):")

    # The guard is Pattern-A-only (the dual-queue tier). The Af7-2 launch path forces
    # backend=batch (Pattern A); without an override a long full-VLM run routes to Pattern B
    # (SageMaker, which picks its own instance — no g5 OOM risk), so the guard correctly
    # only matters when the user pins Batch.
    p_full = dec.profile_run("pi05", 20000, "full_ft")
    check(p_full.vram_per_gpu_gb > dec.SPOT_GPU_FLOOR_GB,
          f"pi05 full_ft replica {p_full.vram_per_gpu_gb} GB > {dec.SPOT_GPU_FLOOR_GB} GB floor")

    # pi05 full_vlm on Batch, spot=True → guard flips to On-Demand (the real Af7-2 recipe).
    _, d_full = _run("pi05", 20000, full_vlm=True, spot=True, backend_override="batch")
    check(d_full.pattern == "A", "backend=batch full-VLM → Pattern A")
    check(d_full.spot is False, "pi05 full_vlm (39.6 GB) on Batch auto-routed OFF Spot → On-Demand")
    check(any("On-Demand queue" in n and "24 GB" in n for n in d_full.notes),
          "guard explains the auto-route in notes")

    # expert-only (~22.6 GB ≤ 24 GB) fits the g5 fallback → stays on Spot.
    _, d_exp = _run("pi05", 20000, full_vlm=False, spot=True, backend_override="batch")
    check(d_exp.spot is True, "pi05 expert-only (22.6 GB ≤ 24 GB) stays on Spot")

    # smolvla (0.45B × 12 = 5.4 GB) trivially fits → Spot.
    _, d_small = _run("smolvla", 20000, spot=True, backend_override="batch")
    check(d_small.spot is True, "smolvla (small) stays on Spot")

    # An explicit instance_override is honored, not overridden by the guard.
    _, d_ovr = _run("pi05", 20000, full_vlm=True, spot=True, backend_override="batch",
                    instance_override="g6e.4xlarge")
    check(d_ovr.spot is True, "explicit instance_override left on Spot (user wins, not caged)")

    # spot=False already → guard is a no-op (no spurious note).
    _, d_od = _run("pi05", 20000, full_vlm=True, spot=False, backend_override="batch")
    check(d_od.spot is False and not any("auto-routed" in n for n in d_od.notes),
          "spot=False full-VLM: no redundant guard note")


# ── GR00T axis: classify + profile/decide + plan/submit faithfulness vs gr00t_launch.py ──
def test_groot():
    print("GR00T axis (NVIDIA Isaac-GR00T N1.7):")
    import orchestrator_plan as op
    import orchestrator_submit as osub
    import os

    # classify: embodiment_tag is the GR00T signal; explicit intent='gr00t' too. A dataset
    # WITHOUT an embodiment stays lerobot IL (the more common path).
    check(dec.classify({"dataset": "s3://b/ds/", "embodiment_tag": "UNITREE_G1"}) == "gr00t",
          "classify: dataset + embodiment_tag → gr00t")
    check(dec.classify({"intent": "gr00t", "dataset": "s3://b/ds/"}) == "gr00t",
          "classify: explicit intent=gr00t")
    check(dec.classify({"intent": "groot", "dataset": "s3://b/ds/"}) == "gr00t",
          "classify: intent=groot alias → gr00t")
    check(dec.classify({"dataset": "s3://b/ds/", "model": "pi05"}) == "il",
          "classify: dataset WITHOUT embodiment stays IL (lerobot)")

    # profile_groot: UNITREE_G1 auto-defaults action_horizon=50; other embodiments leave it
    # unset; explicit value wins. Wall-clock anchored to the verified G1 run (~1.28 s/step).
    pg = dec.profile_groot(embodiment_tag="UNITREE_G1", max_steps=5000)
    check(pg.action_horizon == 50, f"profile: UNITREE_G1 auto action_horizon=50 (got {pg.action_horizon})")
    check(abs(pg.est_wall_clock_h - round(5000 * 1.28 / 3600.0, 2)) < 1e-9,
          f"profile: ~1.28 s/step anchor → {pg.est_wall_clock_h} h for 5000 steps")
    check(dec.profile_groot(embodiment_tag="NEW_ARM", max_steps=2000).action_horizon is None,
          "profile: non-G1 embodiment leaves action_horizon unset (base 40-wide head)")
    check(dec.profile_groot(embodiment_tag="UNITREE_G1", action_horizon=40).action_horizon == 40,
          "profile: explicit action_horizon overrides the G1 default")

    # decide_groot: always Pattern A (the only deployed GR00T backend); Spot OFF by default;
    # EC2 (not SageMaker) price anchor; a non-batch backend override is rejected.
    dg = dec.decide_groot(pg)
    check(dg.pattern == "A" and dg.axis == "gr00t" and dg.backend == "batch",
          "decide: GR00T → Pattern A (Batch), axis gr00t")
    check(dg.instance_type == "g6e.4xlarge" and not dg.spot,
          "decide: g6e.4xlarge (1×L40S), On-Demand default")
    check(dg.est_cost_usd > 0 and abs(dg.price_per_hr - 3.00424) < 1e-3,
          f"decide: EC2 OD anchor $3.004/hr (got {dg.price_per_hr})")
    try:
        dec.decide_groot(pg, backend_override="sagemaker")
        check(False, "decide: sagemaker override raises (no GR00T SageMaker backend)")
    except ValueError:
        check(True, "decide: sagemaker override raises (no GR00T SageMaker backend)")

    # plan(): dataset + embodiment → gr00t axis, Pattern A, runnable; submit carries the
    # GROOT_* intent (embodiment, max_steps, action_horizon).
    plan_g = op.plan({"dataset": "s3://b/ds/", "embodiment_tag": "UNITREE_G1", "steps": 5000})
    check(plan_g["axis"] == "gr00t" and plan_g["pattern"] == "A" and plan_g["runnable"],
          "plan: GR00T → axis gr00t, Pattern A, runnable")
    s = plan_g["submit"]
    check(s["embodiment_tag"] == "UNITREE_G1" and s["max_steps"] == 5000 and s["action_horizon"] == 50,
          "plan: GR00T submit carries embodiment + max_steps + auto horizon 50")

    # submit (_submit_groot_batch): mirrors gr00t_launch.py's GROOT_* container-override
    # contract byte-faithfully. Set env wiring; intercept boto3 submit_job (no live AWS).
    os.environ["GROOT_A_JOB_QUEUE"] = "arn:aws:batch:us-west-2:428:job-queue/groot"
    os.environ["GROOT_A_JOB_DEFINITION"] = "arn:aws:batch:us-west-2:428:job-definition/groot:5"
    os.environ["GROOT_A_OUTPUT_S3"] = "s3://pai-artifacts/gr00t-n17"
    os.environ["HF_TOKEN_SSM"] = "/pai/hf-token"
    os.environ["HF_TOKEN_SSM_REGION"] = "us-east-1"

    captured = {}

    class _FakeBatch:
        def submit_job(self, **kw):
            captured.update(kw)
            return {"jobId": "fake-groot-id"}

    class _FakeSsm:
        def get_parameter(self, **kw):
            return {"Parameter": {"Value": "hf_FAKE_TOKEN"}}

    class _FakeSession:
        region_name = "us-west-2"

        def client(self, name, region_name=None):
            return _FakeBatch() if name == "batch" else _FakeSsm()

    out = osub._submit_groot_batch(_FakeSession(), "us-west-2", s, want_hf=True)
    check(out["axis"] == "gr00t" and out["status"] == "submitted"
          and out["job_name"].startswith("gr00t-n17-"),
          "submit: GR00T job submitted with gr00t-n17- prefix")
    check(captured["jobQueue"].endswith("/groot")
          and captured["jobDefinition"].endswith("groot:5"),
          "submit: GR00T uses the GROOT_A_* queue/jobdef wiring")
    env = {e["name"]: e["value"] for e in captured["containerOverrides"]["environment"]}
    check(env["GROOT_DATASET_S3"] == "s3://b/ds/" and env["GROOT_EMBODIMENT_TAG"] == "UNITREE_G1",
          "submit: GROOT_* env carries dataset + embodiment (matches gr00t_launch.py)")
    check(env["GROOT_MAX_STEPS"] == "5000" and env["GROOT_ACTION_HORIZON"] == "50",
          "submit: GROOT_MAX_STEPS + GROOT_ACTION_HORIZON=50 set")
    check(env["GROOT_OUTPUT_S3"] == f"s3://pai-artifacts/gr00t-n17/{out['job_name']}",
          "submit: GROOT_OUTPUT_S3 = <hint>/<job> (matches gr00t_launch.py)")
    check(env.get("HF_TOKEN") == "hf_FAKE_TOKEN" and env.get("HUGGING_FACE_HUB_TOKEN") == "hf_FAKE_TOKEN",
          "submit: HF token injected (gated Cosmos-Reason2-2B backbone is REQUIRED)")
    # dispatch routes a gr00t plan to _submit_groot_batch (no live boto3: spy the branch +
    # neutralize boto3.Session so submit()'s session build never touches AWS).
    import boto3 as _boto3
    routed = {}
    _orig_branch, _orig_sess = osub._submit_groot_batch, _boto3.Session
    try:
        _boto3.Session = lambda **kw: _FakeSession()  # noqa: ARG005
        osub._submit_groot_batch = lambda *a, **k: (routed.update(hf=k.get("want_hf")),
                                                    {"axis": "gr00t", "status": "submitted"})[1]
        osub_out = osub.submit(plan_g)
    finally:
        osub._submit_groot_batch, _boto3.Session = _orig_branch, _orig_sess
    check(osub_out["axis"] == "gr00t" and osub_out["status"] == "submitted",
          "submit dispatch: gr00t plan → _submit_groot_batch")
    check(routed.get("hf") is True,
          "submit dispatch: GR00T want_hf always True (gated backbone)")


def test_multinode_pattern_c():
    """num_nodes>1 is an explicit multi-node request → Pattern C (HyperPod FSDP2), which is
    a deploy-gated reference (NOT auto-submitted). Phase 5: the planner routes it, threads
    num_nodes through submit, and the envelope marks it not-runnable so the orchestrator
    Choice state returns a recommendation (the operator runs sbatch on the cluster)."""
    print("multi-node Pattern C (Phase 5 — deploy-gated FSDP2 routing):")
    import orchestrator_plan as op

    # num_nodes>1 → backend forced to hyperpod (Pattern C) even for a small replica.
    plan_mn = op.plan({"dataset": "s3://b/ds/", "model": "pi05", "steps": 20000, "num_nodes": 2})
    check(plan_mn["pattern"] == "C" and not plan_mn["runnable"],
          "num_nodes=2 → Pattern C, NOT runnable (operator-gated sbatch, not a Batch job)")
    check(plan_mn["submit"]["num_nodes"] == 2,
          "num_nodes threads through the IL submit dict (the launcher reads it as --nodes)")
    # The decision notes now describe the deploy-gated reference, not 'synth-only/not wired'.
    notes = " ".join(plan_mn["decision"]["notes"])
    check("enableHyperPod=true" in notes and "hyperpod_fsdp_launch.sh" in notes,
          "Pattern C notes give the deploy-gate + launcher (accurate, not 'synth-only')")
    check("not wired into bin/app.ts" not in notes,
          "stale 'not wired into bin/app.ts' language is gone")

    # num_nodes=1 (default) stays on the runnable single-node path (unchanged).
    plan_sn = op.plan({"dataset": "s3://b/ds/", "model": "pi05", "steps": 20000})
    check(plan_sn["pattern"] == "B" and plan_sn["runnable"] and plan_sn["submit"]["num_nodes"] == 1,
          "num_nodes default 1 → runnable single-node Pattern B (unchanged), submit num_nodes=1")

    # An explicit backend override still wins over the num_nodes→hyperpod inference.
    plan_ovr = op.plan({"dataset": "s3://b/ds/", "model": "pi05", "steps": 200,
                        "num_nodes": 2, "backend": "batch"})
    check(plan_ovr["pattern"] == "A",
          "explicit --backend batch overrides the num_nodes→C inference (user wins)")


def main():
    for t in (test_rule_table, test_overrides, test_lora, test_cost_anchor, test_budget,
              test_cli_argv, test_cli_rl_argv, test_orchestrator, test_per_job_controls,
              test_gpu_floor_guard, test_groot, test_multinode_pattern_c):
        t()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
