"""
Distributed Checkpoint (DCP) save/load for multi-node FSDP2 — with the verified
gloo-coordinator race fix.

WHY THIS EXISTS (and why it's separate from train.py): the single-node lerobot path
(train.py) checkpoints via HuggingFace save_pretrained dirs and resumes from EFS — that
works and is unchanged. But a multi-node FSDP2 run (Pattern C) shards a 3B+ model's
weights + optimizer state across ranks; each rank holds only its shard, so a single-rank
HF save would be incomplete. PyTorch Distributed Checkpoint (torch.distributed.checkpoint,
"DCP") is the standard answer: every rank writes its shard in parallel to a shared path
(FSx Lustre), and load reshards on resume. This module is the DCP save/load helper the
Pattern C trainer calls; the single-node path never imports it.

★ THE GLOO-COORDINATOR RACE FIX (absorbed verbatim from awslabs/awsome-distributed-ai
DreamZero `dcp-save-gloo-coordinator.patch`, MIT-0, fetched 2026-06-27):

  dcp.save's post-write finalization broadcasts a multi-MB result object via
  broadcast_object_list. That helper picks its transport device from the process group's
  backend — CUDA when NCCL, CPU when gloo. With the DEFAULT (NCCL/CUDA) process group, the
  object broadcast runs on CUDA and RACES with NCCL comm teardown at the end of a long
  (100s+ GB) checkpoint write → a non-coordinator rank reads an all-zero buffer →
  `_pickle.UnpicklingError: invalid load key '\x00'`, AFTER every shard + .metadata are
  already on disk (so the checkpoint LOOKS written but the job dies, reproduced on 2x p5en).

  NOTE (from the patch): this is NOT fixed by upgrading torch — the synchronous save() /
  _save_state_dict() path is byte-identical through at least torch 2.8 and always
  broadcasts over the default PG. Only async_save uses a CPU/gloo backend.

  THE FIX: create a dedicated gloo process group (timeout 30 min) ONCE, cache it, and pass
  it as dcp.save(..., process_group=gloo_pg) so the finalization object-broadcast goes over
  CPU/gloo — immune to the CUDA/NCCL teardown race.

This module ports that fix faithfully into vla-ft's own DCP wrapper (we don't carry
RLinf's fsdp/strategy/base.py, so the patch is applied to the equivalent call site here).

Stdlib + torch only; torch is imported lazily so this file py_compiles and unit-tests
without a GPU/torch install (the pure helpers below are tested standalone).
"""

from __future__ import annotations

import datetime
import os


# Dedicated gloo PG timeout — 30 min, the DreamZero patch value (a 100GB+ DCP finalize on
# a slow shared FS can take many minutes; the default 10 min gloo timeout would abort it).
GLOO_PG_TIMEOUT = datetime.timedelta(minutes=30)

# Module-level cache for the gloo PG (the patch caches it on the class as `_dcp_gloo_pg`;
# we cache at module scope — the trainer process is long-lived and creates it once).
_DCP_GLOO_PG = None


def is_distributed() -> bool:
    """True iff torch.distributed is initialized with world_size > 1 — i.e. this is a
    genuine multi-rank run where DCP (and the gloo fix) apply. Single-node falls back to
    the HF-dir path in train.py and never calls into here. Import-guarded so a non-torch
    host (unit test) returns False rather than raising."""
    try:
        import torch.distributed as dist
    except Exception:  # noqa: BLE001 — no torch / no distributed build
        return False
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def get_dcp_gloo_pg():
    """Return the dedicated gloo process group for dcp.save finalization, creating it once.

    This is the heart of the DreamZero fix: a SEPARATE gloo (CPU) process group, so the
    post-write broadcast_object_list runs on CPU/gloo and never races NCCL teardown. Returns
    None when not distributed (single-rank → no PG needed) or if gloo group creation fails
    (we then fall back to the default PG — better a possible race than no checkpoint)."""
    global _DCP_GLOO_PG
    if not is_distributed():
        return None
    if _DCP_GLOO_PG is not None:
        return _DCP_GLOO_PG
    try:
        import torch.distributed as dist
        _DCP_GLOO_PG = dist.new_group(backend="gloo", timeout=GLOO_PG_TIMEOUT)
    except BaseException:  # noqa: BLE001 — match the patch's broad guard; fall back to default PG
        _DCP_GLOO_PG = None
    return _DCP_GLOO_PG


def dcp_save(state_dict: dict, checkpoint_dir: str) -> str:
    """Save a sharded training state to checkpoint_dir via torch DCP, with the gloo fix.

    Every rank writes its shard in parallel; the coordinator finalizes the .metadata. The
    gloo PG is passed to dcp.save so finalization broadcasts over CPU/gloo (the race fix).
    Returns the checkpoint_dir. Raises if torch DCP is unavailable (a multi-node trainer
    must have it)."""
    import torch.distributed.checkpoint as dcp

    os.makedirs(checkpoint_dir, exist_ok=True)
    gloo_pg = get_dcp_gloo_pg()
    # process_group=gloo_pg routes the finalization object-broadcast over CPU/gloo — the
    # verbatim DreamZero fix. When gloo_pg is None (single-rank / creation failed) dcp.save
    # uses the default PG, which is correct for the single-rank case and the documented
    # fallback for the rare gloo-creation failure.
    dcp.save({"state": state_dict}, checkpoint_id=checkpoint_dir, process_group=gloo_pg)
    return checkpoint_dir


def dcp_load(state_dict: dict, checkpoint_dir: str) -> dict:
    """Load a sharded training state from checkpoint_dir into state_dict (in place, the DCP
    contract — pass the freshly-constructed model/optim state dict to reshard into). Returns
    the populated state_dict. Load does not hit the finalization broadcast, so it needs no
    gloo PG. Raises if torch DCP is unavailable."""
    import torch.distributed.checkpoint as dcp

    payload = {"state": state_dict}
    dcp.load(payload, checkpoint_id=checkpoint_dir)
    return payload["state"]


def _reset_gloo_pg_for_test() -> None:
    """Test hook: clear the cached gloo PG so a test can re-exercise creation. Not used in
    production (the PG is created once and reused for the trainer's life)."""
    global _DCP_GLOO_PG
    _DCP_GLOO_PG = None
