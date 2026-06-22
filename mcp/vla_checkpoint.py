#!/usr/bin/env python3
"""
vla-ft MCP — checkpoint inspection + the export-consistency gate (pure core).

`describe_checkpoint` is the tool that would have caught the bug that motivated this MCP:
the GR00T G1 checkpoint shipped with model `action_horizon=40` while its processor
expected `50`, and nobody noticed until the *consumer's* rollout crashed at step 0. The
fix is to inspect a finished checkpoint at export-read time and refuse to call it OK if
its two halves disagree.

This module is **pure**: it takes an already-listed set of object keys and the parsed
contents of the small JSON config files, and returns a verdict. All S3 I/O lives in
vla_aws.py. The key paths below are grounded in REAL checkpoints, not guessed:

  - GR00T model side  : `config.json` / `experiment_cfg/final_model_config.json`
                        → top-level `action_horizon`  (real value seen: 40)
  - GR00T proc. side  : `processor/processor_config.json`
                        → `processor_kwargs.max_action_horizon`  (real value: 50)
                        and per-embodiment `...<emb>/action/delta_indices` length.
                        Also `experiment_cfg/final_processor_config.json` → top-level
                        `max_action_horizon`.
  - lerobot pi05      : `pretrained_model/` — adapter-only (adapter_config.json +
                        adapter_model.safetensors, NO model.safetensors) vs self-contained
                        (model.safetensors present). Base from
                        adapter_config.json:base_model_name_or_path.

(Verified 2026-06-20 against s3://pai-artifacts-.../gr00t-n17-20260619-103523/output/.)
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ── checkpoint kind detection (by the files present) ─────────────────────────────────
def detect_kind(keys: list[str]) -> str:
    """Classify a checkpoint by its object basenames: 'gr00t' | 'lerobot' | 'unknown'.

    GR00T full merged HF ckpt → sharded `model-0000X-of-0000Y.safetensors` +
    `model.safetensors.index.json` at the prefix root. lerobot pi05 → a `pretrained_model/`
    dir (adapter or merged). Detection is on basenames so it works on a flat key list."""
    bases = {k.rsplit("/", 1)[-1] for k in keys}
    paths = "\n".join(keys)
    if any(b.startswith("model-") and b.endswith(".safetensors") for b in bases) \
            or "model.safetensors.index.json" in bases:
        return "gr00t"
    if "pretrained_model/" in paths or "adapter_config.json" in bases \
            or any("pretrained_model" in k for k in keys):
        return "lerobot"
    return "unknown"


# ── lerobot adapter-only vs merged ───────────────────────────────────────────────────
def is_adapter_only(keys: list[str]) -> bool:
    """True if the lerobot checkpoint is adapter-only (LoRA), False if self-contained.

    Adapter-only ⇔ `adapter_model.safetensors` present AND the full base
    `model.safetensors` ABSENT. A full-FT / expert-only ckpt ships `model.safetensors`
    and no adapter files. (README checkpoint-contract callout, verified.)"""
    bases = {k.rsplit("/", 1)[-1] for k in keys}
    has_adapter = "adapter_model.safetensors" in bases or "adapter_config.json" in bases
    has_full = "model.safetensors" in bases
    return has_adapter and not has_full


# ── GR00T horizon extraction ─────────────────────────────────────────────────────────
def groot_model_horizon(model_config: dict | None) -> int | None:
    """The model's action_horizon — top-level key of config.json / final_model_config.json."""
    if not model_config:
        return None
    v = model_config.get("action_horizon")
    return int(v) if isinstance(v, (int, float)) else None


def groot_processor_horizon(
    processor_config: dict | None,
    final_processor_config: dict | None = None,
    embodiment_tag: str | None = None,
) -> tuple[int | None, dict]:
    """The processor's expected action horizon, with evidence.

    Source order (most authoritative first):
      1. processor_config.json → processor_kwargs.max_action_horizon (the scalar the
         processor was built with — this is what setup.py injects from the model config).
      2. final_processor_config.json → top-level max_action_horizon (the experiment mirror).
      3. per-embodiment action.delta_indices length (the actual padded sequence length).
    Returns (horizon, evidence_dict). The per-embodiment delta length is reported as
    `delta_indices_len` so a caller can see the data's true horizon even if the scalar lies.
    """
    evidence: dict = {}
    horizon: int | None = None

    if processor_config:
        pk = processor_config.get("processor_kwargs") or {}
        mah = pk.get("max_action_horizon")
        if isinstance(mah, (int, float)):
            horizon = int(mah)
            evidence["processor_kwargs.max_action_horizon"] = horizon

        # Per-embodiment action delta_indices length (the data's true action horizon).
        mcfgs = pk.get("modality_configs") or {}
        if isinstance(mcfgs, dict):
            delta_lens: dict[str, int] = {}
            for emb, cfg in mcfgs.items():
                try:
                    di = cfg["action"]["delta_indices"]
                    if isinstance(di, list):
                        delta_lens[emb] = len(di)
                except (KeyError, TypeError):
                    continue
            if delta_lens:
                evidence["action_delta_indices_len_per_embodiment"] = delta_lens
                # If an embodiment tag was given, that one's length is the run's true horizon.
                if embodiment_tag and embodiment_tag.lower() in {e.lower() for e in delta_lens}:
                    for emb, ln in delta_lens.items():
                        if emb.lower() == embodiment_tag.lower():
                            evidence["selected_embodiment"] = emb
                            evidence["selected_embodiment_delta_len"] = ln
                            if horizon is None:
                                horizon = ln

    if horizon is None and final_processor_config:
        mah = final_processor_config.get("max_action_horizon")
        if isinstance(mah, (int, float)):
            horizon = int(mah)
            evidence["final_processor_config.max_action_horizon"] = horizon

    return horizon, evidence


# ── verdict ──────────────────────────────────────────────────────────────────────────
@dataclass
class CheckpointReport:
    """The describe_checkpoint answer. `consistency` is the headline gate."""
    prefix: str
    kind: str                              # 'gr00t' | 'lerobot' | 'unknown'
    consistency: str                       # 'OK' | 'MISMATCH' | 'UNKNOWN'
    adapter_only: bool | None = None       # lerobot only
    base_model: str | None = None
    model_action_horizon: int | None = None       # GR00T
    processor_action_horizon: int | None = None    # GR00T
    files: list[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    loadable_hint: str = ""                # how a consumer should load this
    notes: list[str] = field(default_factory=list)


def inspect_gr00t(
    prefix: str,
    keys: list[str],
    *,
    model_config: dict | None,
    processor_config: dict | None,
    final_model_config: dict | None = None,
    final_processor_config: dict | None = None,
    embodiment_tag: str | None = None,
) -> CheckpointReport:
    """Build the GR00T checkpoint report, including the model↔processor horizon gate."""
    notes: list[str] = []
    model_h = groot_model_horizon(model_config) or groot_model_horizon(final_model_config)
    proc_h, evidence = groot_processor_horizon(
        processor_config, final_processor_config, embodiment_tag)

    if model_h is not None and proc_h is not None:
        if model_h == proc_h:
            consistency = "OK"
        else:
            consistency = "MISMATCH"
            notes.append(
                f"action_horizon MISMATCH: model config.json says {model_h}, processor "
                f"expects {proc_h}. The action head will emit (1,{model_h},D) while the "
                f"processor expects ({proc_h},D) → broadcast crash at rollout step 0. "
                f"Hot-fix: edit config.json + experiment_cfg/final_model_config.json "
                f"action_horizon {model_h}->{proc_h} and re-export (the ...-h50fix pattern).")
    else:
        consistency = "UNKNOWN"
        notes.append("Could not read both horizons (missing config.json or processor_config "
                     ".json) — cannot verify consistency.")

    base = (model_config or {}).get("base_model") \
        or (model_config or {}).get("_name_or_path") or "nvidia/GR00T-N1.7-3B"

    return CheckpointReport(
        prefix=prefix, kind="gr00t", consistency=consistency,
        adapter_only=False, base_model=base,
        model_action_horizon=model_h, processor_action_horizon=proc_h,
        files=sorted({k.rsplit("/", 1)[-1] for k in keys}),
        evidence=evidence,
        loadable_hint=("run_gr00t_server.py --model-path <prefix> --embodiment-tag "
                       "<TAG> --use-sim-policy-wrapper  (full merged 3B HF ckpt)"),
        notes=notes,
    )


def inspect_lerobot(
    prefix: str,
    keys: list[str],
    *,
    adapter_config: dict | None = None,
    train_config: dict | None = None,
) -> CheckpointReport:
    """Build the lerobot pi05 report: adapter-only vs merged + the base it needs."""
    notes: list[str] = []
    adapter = is_adapter_only(keys)

    base = None
    if adapter_config:
        base = adapter_config.get("base_model_name_or_path")
    if not base and train_config:
        # train_config.json nests the launch pretrained_path under policy.pretrained_path.
        pol = train_config.get("policy") or {}
        base = pol.get("pretrained_path")
    if adapter and not base:
        base = "lerobot/pi05_base"  # the documented pi05 default
        notes.append("adapter_config.json not read; assuming base lerobot/pi05_base "
                     "(the pi05 default). Confirm before serving.")

    if adapter:
        consistency = "OK"  # adapter-only is valid; the only 'gotcha' is needing the base
        notes.append("Adapter-only (LoRA) checkpoint: the base weights are NOT included. "
                     f"To serve, the base ({base}) must be reachable; lerobot loads the "
                     "adapter over it (use_peft=true). lerobot does NOT auto-merge — for a "
                     "single self-contained artifact, merge with PeftModel.merge_and_unload().")
        hint = (f"load with lerobot make_policy(use_peft=true); base {base} must be reachable "
                f"(HF id or mounted path). NOT self-contained.")
    else:
        consistency = "OK"
        hint = "self-contained (model.safetensors present) — load directly, no base needed."

    return CheckpointReport(
        prefix=prefix, kind="lerobot", consistency=consistency,
        adapter_only=adapter, base_model=base,
        files=sorted({k.rsplit("/", 1)[-1] for k in keys}),
        loadable_hint=hint, notes=notes,
    )
