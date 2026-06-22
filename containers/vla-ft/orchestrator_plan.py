#!/usr/bin/env python3
"""
Phase 4 orchestrator — the classify → profile → decide Lambda (the "plan" step).

This is the Step Functions front door. It takes one *intent* (the same thing the
`vla_ft_cli.py` user types: a dataset + model for IL, or a task for RL) and turns it
into a concrete plan: which axis, which Pattern (A/B/C), which instance, the cost
estimate, and the normalized parameters the downstream `submit` Lambda needs.

★ Single-source guarantee (ARCHITECTURE §3, the whole reason this phase is cheap):
this handler does NOT re-implement any decision logic. It imports `vla_ft_decide`
verbatim — the SAME pure module the launcher CLI calls today — and only marshals
its dataclasses to/from the JSON Step Functions passes between states. If the rule
table changes, it changes in exactly one place and both the launcher and the
orchestrator move together.

Determinism (ARCHITECTURE §3, "deterministic Step Functions, no LLM"): every output
is a pure function of the input event. Same intent in → same plan out, reproducibly.
No boto3, no network here — capacity/quota pre-flight and the actual SubmitJob live in
the submit Lambda (it needs creds; this one does not).

Event (the Step Functions input, mirrors vla_ft_cli flags):
  {
    "intent":  "il" | "rl"        # optional; else inferred from dataset/task
    "dataset": "s3://.../lerobot_dataset/",   # IL
    "model":   "pi05",                        # IL (default pi05)
    "task":    "Isaac-Velocity-Rough-H1-v0",  # RL
    "steps":   20000, "max_iterations": 3000, "num_envs": 4096,
    "num_gpus": 1, "multi_node": false,
    "full_vlm": false, "lora": false,
    "budget_usd": 100, "backend": "sagemaker", "instance_type": "ml.g6e.48xlarge",
    "spot": true, "select_best": false, "early_stop_patience": null,
    "pretrained_path": null
  }

Return (consumed by the Choice + submit states):
  {
    "axis": "il"|"rl", "pattern": "A"|"B"|"C", "backend": "batch"|"sagemaker"|"hyperpod",
    "runnable": true|false,            # false = Pattern C (recommend-only, gate stops here)
    "decision": {...}, "profile": {...},
    "submit": {...},                   # normalized args for orchestrator_submit
    "plan_text": "...human-readable plan...",
  }
"""

from __future__ import annotations

import dataclasses

import vla_ft_decide as dec


def _il_plan(event: dict) -> dict:
    """Profile + decide for an IL fine-tune. Pure passthrough to vla_ft_decide."""
    model = event.get("model") or "pi05"
    steps = int(event.get("steps", 20000))
    ft_mode = dec.resolve_ft_mode(
        model,
        bool(event.get("full_vlm", False)),
        bool(event.get("lora", False)),
    )
    profile = dec.profile_run(model, steps, ft_mode)
    decision = dec.decide(
        profile,
        budget_usd=event.get("budget_usd"),
        backend_override=event.get("backend"),
        instance_override=event.get("instance_type"),
        spot=bool(event.get("spot", True)),
    )
    # The normalized submit contract for an IL job (the submit Lambda maps this onto the
    # verified launcher's interface — batch_launch.py for A, a recommend handoff for B/C).
    submit = {
        "axis": "il",
        "pattern": decision.pattern,
        "policy": model,
        # Pinning job_name reuses the same EFS checkpoint dir → resume after a timeout /
        # Spot reclaim (orchestrator_submit derives VLA_FT_CHECKPOINT_DIR from it). Omit →
        # a fresh timestamped name.
        "job_name": event.get("job_name"),
        "dataset_s3": event.get("dataset"),
        "pretrained_path": event.get("pretrained_path"),
        "steps": steps,
        "per_device_batch": decision.per_device_batch,
        "num_gpus": decision.num_gpus,
        "instance_type": decision.instance_type,
        "sm_instance_type": decision.sm_instance_type,
        "spot": decision.spot,
        "expert_only": decision.expert_only,
        # LoRA (full-VLM-on-one-GPU; mutually exclusive with expert_only — resolve_ft_mode
        # already makes --lora win over the expert-only default, so expert_only is False here).
        "lora": ft_mode == "lora",
        "lora_r": event.get("lora_r"),
        "lora_alpha": event.get("lora_alpha"),
        "lora_target_modules": event.get("lora_target_modules"),
        "select_best": bool(event.get("select_best", False)),
        "val_episodes": event.get("val_episodes"),
        "save_freq": event.get("save_freq"),
        "early_stop_patience": event.get("early_stop_patience"),
    }
    return _envelope("il", profile, decision, submit, dec.format_plan(profile, decision))


def _groot_plan(event: dict) -> dict:
    """Profile + decide for a GR00T N1.7 fine-tune. Pure passthrough to vla_ft_decide.

    The intent is a LeRobot dataset + a registered embodiment tag (e.g. UNITREE_G1). GR00T
    runs on the single deployed GrootPatternAStack (Batch, 1×L40S), so the decision is
    always Pattern A; the submit Lambda maps this onto gr00t_launch.py's GROOT_* contract."""
    profile = dec.profile_groot(
        embodiment_tag=event.get("embodiment_tag"),
        base_model=event.get("base_model"),
        max_steps=event.get("steps") if event.get("steps") is not None
        else event.get("max_steps"),
        save_steps=event.get("save_steps"),
        action_horizon=event.get("action_horizon"),
        num_gpus=int(event.get("num_gpus", 1)),
        global_batch=event.get("global_batch"),
    )
    decision = dec.decide_groot(
        profile,
        budget_usd=event.get("budget_usd"),
        backend_override=event.get("backend"),
        instance_override=event.get("instance_type"),
        spot=bool(event.get("spot", False)),  # GR00T defaults On-Demand (stack default)
    )
    submit = {
        "axis": "gr00t",
        "pattern": decision.pattern,
        "job_name": event.get("job_name"),  # pin → resume from EFS (see _il_plan)
        "dataset_s3": event.get("dataset"),
        "embodiment_tag": profile.embodiment_tag,
        "base_model": profile.base_model,
        "max_steps": profile.max_steps,
        "save_steps": profile.save_steps,
        "action_horizon": profile.action_horizon,
        "global_batch": profile.global_batch,
        "learning_rate": event.get("learning_rate"),
        "num_gpus": decision.num_gpus,
        "instance_type": decision.instance_type,
        "liveness_deadline_s": event.get("liveness_deadline_s"),
        "extra": event.get("extra") or [],
    }
    return _envelope("gr00t", profile, decision, submit,
                     dec.format_groot_plan(profile, decision))


def _rl_plan(event: dict) -> dict:
    """Profile + decide for an RL run. Pure passthrough to vla_ft_decide."""
    profile = dec.profile_rl(
        event.get("task"),
        num_envs=event.get("num_envs"),
        max_iterations=event.get("max_iterations"),
        num_gpus=int(event.get("num_gpus", 1)),
        multi_node=bool(event.get("multi_node", False)),
        experiment_name=event.get("experiment_name"),
    )
    decision = dec.decide_rl(
        profile,
        budget_usd=event.get("budget_usd"),
        backend_override=event.get("backend"),
        instance_override=event.get("instance_type"),
        spot=bool(event.get("spot", True)),
    )
    submit = {
        "axis": "rl",
        "pattern": decision.pattern,
        "job_name": event.get("job_name"),  # pin → resume from EFS (see _il_plan)
        "task": profile.task,
        "experiment_name": profile.experiment_name,
        "num_envs": profile.num_envs,
        "max_iterations": profile.max_iterations,
        "seed": event.get("seed"),
        "num_gpus": decision.num_gpus,
        "instance_type": decision.instance_type,
        "skip_export": bool(event.get("skip_export", False)),
        "overrides": event.get("overrides") or [],
    }
    return _envelope("rl", profile, decision, submit, _format_rl_plan(profile, decision))


def _envelope(axis: str, profile, decision, submit: dict, plan_text: str) -> dict:
    """Assemble the state output. `runnable` is the deterministic gate the Choice state
    branches on: Pattern C is code+synth only, so the machine stops at a recommendation
    rather than attempting an un-deployed backend."""
    return {
        "axis": axis,
        "pattern": decision.pattern,
        "backend": decision.backend,
        "runnable": decision.pattern in ("A", "B"),
        "profile": dataclasses.asdict(profile),
        "decision": dataclasses.asdict(decision),
        "submit": submit,
        "plan_text": plan_text,
    }


def _format_rl_plan(profile, decision) -> str:
    """Human-readable RL plan (the RL analogue of vla_ft_decide.format_plan)."""
    d = decision
    spot_label = "Spot" if d.spot else "On-Demand"
    iters = profile.max_iterations or "(task default)"
    envs = profile.num_envs or "(task default)"
    cost = (f"~${d.est_cost_usd:.0f}  ({spot_label} @ ${d.price_per_hr:g}/hr)"
            if d.est_cost_usd else "unavailable (set max_iterations)")
    lines = [
        "=" * 64,
        f"vla-ft RL plan — task {profile.task}",
        "=" * 64,
        f"  decision   : Pattern {d.pattern} ({d.backend})  on  {d.instance_type}",
        f"  why        : {d.rationale}",
        f"  compute    : {d.num_gpus} GPU, {spot_label}   envs={envs}  iters={iters}",
        f"  est cost   : {cost}",
    ]
    if d.notes:
        lines.append("  notes      :")
        lines += [f"    - {n}" for n in d.notes]
    lines.append("=" * 64)
    return "\n".join(lines)


def plan(event: dict) -> dict:
    """Classify then profile+decide. The pure core (testable without Lambda)."""
    axis = dec.classify(event)
    if axis == "gr00t":
        return _groot_plan(event)
    return _il_plan(event) if axis == "il" else _rl_plan(event)


def handler(event, context=None):  # noqa: ARG001 (Lambda signature)
    """AWS Lambda entry point. Step Functions passes the intent as `event`."""
    return plan(event)


if __name__ == "__main__":
    # Tiny manual smoke: `python3 orchestrator_plan.py` prints a couple of plans.
    import json

    for ev in (
        {"dataset": "s3://b/ds/", "model": "pi05"},
        {"task": "Isaac-Velocity-Rough-H1-v0", "max_iterations": 3000, "num_envs": 4096},
    ):
        out = plan(ev)
        print(out["plan_text"])
        print(json.dumps(out["submit"], indent=2))
