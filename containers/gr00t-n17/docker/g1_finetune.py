#!/usr/bin/env python3
"""Universal GR00T fine-tune entry that can set config.model.action_horizon (no upstream fork).

WHY THIS EXISTS
  UNITREE_G1 full-body fine-tune is an UNSUPPORTED upstream path: its data config
  (`unitree_g1_full_body_with_waist_height_nav_cmd`) uses action `delta_indices=range(50)`
  (a 50-step horizon), but the model config (`Gr00tN1d7Config.action_horizon`) and the base
  GR00T-N1.7-3B checkpoint both default to 40. `launch_finetune.py`'s `FinetuneConfig` has
  NO `action_horizon` field and never assigns `config.model.action_horizon`, so the processor
  is built with `max_action_horizon=40` and `processing_gr00t_n1d7.py:548` asserts:
      AssertionError: Action sequence length 50 exceeds max_action_horizon 40
  There is no flag / env / modality-config path to reach `config.model` — the only correct
  fix is to set `config.model.action_horizon` programmatically BEFORE the model is built.

TWO PATCHES ARE REQUIRED (verified vs NVIDIA/Isaac-GR00T @ 65cc4a192e6d — no source fork;
see README.md for the recipe notes):
  GR00T sources `action_horizon` DIFFERENTLY for the processor vs the model:
    - PROCESSOR (`gr00t/model/gr00t_n1d7/setup.py::_create_dataset`): built with
      `max_action_horizon=self.model_config.action_horizon`, where `model_config` is
      `self.config.model` — the live config object. Patch #1 (get_default_config) reaches it.
    - MODEL (`setup.py::_create_model`): built with `AutoModel.from_pretrained(GR00T-N1.7-3B,
      <override kwargs>)`. The override-kwarg list (tune_llm/tune_visual/tune_projector/
      tune_diffusion_model/tune_vlln/state_dropout_prob/backbone_trainable_params_fp32/
      load_bf16) does NOT include `action_horizon`. So `from_pretrained` rebuilds a FRESH
      `Gr00tN1d7Config` from the BASE checkpoint's config.json (action_horizon=40) and that 40
      lands on `model.config` — Patch #1 cannot reach it. `_create_model` then writes
      `final_model_config.json` from `model.config` (=40), and HF `Trainer.save_model` later
      serializes the top-level `config.json` from `model.config` (=40) too.
  RESULT WITHOUT PATCH #2: training runs at 50 (the processor pads data to 50; no weight is
  sized by action_horizon — `position_embedding` is sized by max_seq_len=1024), but the saved
  model `config.json`/`final_model_config.json` ship 40. At inference the action head reads
  config.action_horizon=40 and emits (1,40,7) while the processor expects (50,7) -> broadcast
  crash. (This is exactly what bit the gr00t-g1 rollout on job ...103523, 2026-06-20.)

  Patch #2 therefore patches the CONFIG CLASS so EVERY instance — including the fresh one
  `from_pretrained` builds — carries action_horizon=50. action_horizon sizes NO checkpoint
  weight (verified: no tensor has a 40/50 dim; position_embedding=[1024,1536]=max_seq_len), so
  forcing it on a model whose weights loaded from a 40-built base is weight-safe; it only makes
  the saved config.json match the 50-step regime the model was actually trained at.

HOW (mechanism, verified):
  - `launch_finetune.py` has no `def main()`; all logic runs under `if __name__ == "__main__"`,
    so we re-execute its __main__ body verbatim with `runpy.run_path(run_name="__main__")`
    (logic duplication = 0; runpy keeps argv[1:], so tyro.cli parses the same CLI args).
  - launch_finetune does `from gr00t.configs.base_config import get_default_config`, then
    `cfg = get_default_config().load_dict({data...})`. Patch #1 wraps the SOURCE-module attr
    `gr00t.configs.base_config.get_default_config` so `cfg.model.action_horizon` is set. From
    `cfg.model` we read its `__class__` (`Gr00tN1d7Config`) and apply Patch #2 to that class's
    `__init__` so any later `from_pretrained`-built instance also gets action_horizon forced.
  - `Gr00tN1d7Config.__init__(self, **kwargs)` sets attrs from kwargs and backfills dataclass
    defaults; there is NO `__post_init__` guard, so forcing the attr after the original
    `__init__` is safe (verified at 65cc4a192e6d).
  - GROOT_ACTION_HORIZON UNSET = neither patch fires = byte-identical to calling
    launch_finetune.py directly, so this is safe as a UNIVERSAL entry for every embodiment
    (only G1 full-body actually needs horizon=50).

NOTE: the base 3B action head is 40-wide but horizon-agnostic at the weight level, so loading
base weights into a horizon-50 config is fine; the diffusion head adapts during the
frozen-backbone fine-tune (tune_projector + tune_diffusion) to 50-step chunks.
"""
import os
import runpy

import gr00t.configs.base_config as _bc

# launch_finetune.py: the upstream training script we re-run verbatim under runpy. WORKDIR is
# /workspace (the Dockerfile clones Isaac-GR00T into it), so this absolute path is canonical.
_LAUNCH_FINETUNE = "/workspace/gr00t/experiment/launch_finetune.py"

_orig_get_default_config = _bc.get_default_config

# Guard so we patch the model-config class exactly once even if get_default_config is called
# more than once (idempotent; avoids stacking wrappers on __init__).
_model_config_class_patched = {"done": False}


def _force_action_horizon(cfg):
    """Set cfg.model.action_horizon = GROOT_ACTION_HORIZON (Patch #1, processor path)."""
    horizon = os.environ.get("GROOT_ACTION_HORIZON")
    if horizon:
        cfg.model.action_horizon = int(horizon)
    return cfg


def _patch_model_config_class(model_config):
    """Patch the model-config CLASS __init__ so every later instance (incl. the one
    AutoModel.from_pretrained rebuilds from the base checkpoint) gets action_horizon forced
    (Patch #2, model path). Derived from the live config's __class__ — no import-path guessing.
    """
    horizon = os.environ.get("GROOT_ACTION_HORIZON")
    if not horizon or _model_config_class_patched["done"]:
        return
    cls = type(model_config)
    _orig_init = cls.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        # Override AFTER the original __init__ (and any base-config.json kwargs) run, so the
        # value wins regardless of what from_pretrained loaded. action_horizon sizes no weight.
        self.action_horizon = int(horizon)

    cls.__init__ = _patched_init
    _model_config_class_patched["done"] = True


def _patched_get_default_config():
    cfg = _orig_get_default_config()
    _force_action_horizon(cfg)        # Patch #1: processor path (live config object).
    _patch_model_config_class(cfg.model)  # Patch #2: model path (from_pretrained-rebuilt config).
    return cfg


_bc.get_default_config = _patched_get_default_config

runpy.run_path(_LAUNCH_FINETUNE, run_name="__main__")
