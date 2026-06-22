#!/usr/bin/env python3
"""
vla-ft MCP — checkpoint registry (pure schema + merge logic).

`register_checkpoint` / `list_checkpoints` give a requesting session a memory across
runs: "what did I train, from which dataset+config, and is the export validated or
flagged bad?" The backing store is a **single S3 JSON manifest** (NOT DynamoDB) —
decided 2026-06-20:

  - The platform already writes everything to the artifacts bucket; the registry lives
    beside it (`s3://<artifacts>/registry/checkpoints.json`). No new service, no IAM
    table grants, no DynamoDB cost. The query volume is tiny (a handful of checkpoints),
    so a flat list read+filtered in memory is more than enough — DynamoDB's query power
    would be unused complexity.
  - It is **append/update by entry id** (job_name), so re-registering a checkpoint after
    re-validation just updates its `consistency` in place. Concurrent writers are rare
    (one MCP session at a time per consumer) and the server does a read-modify-write with
    an optional ETag check at the I/O layer.

This module is pure: it builds entries and merges them into an in-memory manifest dict.
vla_aws.py does the get/put.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


MANIFEST_VERSION = 1
REGISTRY_KEY = "registry/checkpoints.json"   # under the artifacts bucket


@dataclass
class CheckpointEntry:
    """One registered checkpoint. `id` is the job name (stable, unique per run)."""
    id: str                                # job_name, e.g. vla-ft-pi05-20260620-005746
    output_s3: str                         # the checkpoint prefix
    engine: str                            # 'lerobot' | 'gr00t' | 'rl'
    model: str | None = None               # pi05 / groot / task id (RL)
    dataset_s3: str | None = None
    intent: str = "il"                     # il | rl
    ft_mode: str | None = None             # expert_only | full_ft | lora
    steps: int | None = None
    action_horizon: int | None = None      # GR00T (the resolved/run horizon)
    adapter_only: bool | None = None       # lerobot LoRA
    base_model: str | None = None
    consistency: str = "UNKNOWN"           # OK | MISMATCH | UNKNOWN (from describe_checkpoint)
    registered_at: str | None = None       # ISO ts (stamped by the I/O layer — pure module is clock-free)
    notes: list[str] = field(default_factory=list)


def empty_manifest() -> dict:
    """A fresh, empty registry manifest."""
    return {"version": MANIFEST_VERSION, "checkpoints": []}


def upsert(manifest: dict, entry: CheckpointEntry) -> dict:
    """Insert or update `entry` (matched by id) in a COPY of the manifest. Pure.

    Returns a new manifest dict; the caller persists it. Update-in-place semantics let a
    consumer re-register a checkpoint after re-validating it (e.g. OK after a hot-fix)."""
    m = {"version": manifest.get("version", MANIFEST_VERSION),
         "checkpoints": list(manifest.get("checkpoints", []))}
    e = asdict(entry)
    for i, existing in enumerate(m["checkpoints"]):
        if existing.get("id") == entry.id:
            # Preserve the original registered_at unless the new entry sets one.
            if not e.get("registered_at") and existing.get("registered_at"):
                e["registered_at"] = existing["registered_at"]
            m["checkpoints"][i] = e
            return m
    m["checkpoints"].append(e)
    return m


def query(
    manifest: dict,
    *,
    engine: str | None = None,
    model: str | None = None,
    intent: str | None = None,
    consistency: str | None = None,
) -> list[dict]:
    """Filter the manifest's checkpoints by any combination of fields. Pure.

    Newest-first (by registered_at when present, else insertion order reversed)."""
    rows = list(manifest.get("checkpoints", []))

    def keep(r: dict) -> bool:
        if engine and r.get("engine") != engine:
            return False
        if model and r.get("model") != model:
            return False
        if intent and r.get("intent") != intent:
            return False
        if consistency and r.get("consistency") != consistency:
            return False
        return True

    out = [r for r in rows if keep(r)]
    out.sort(key=lambda r: r.get("registered_at") or "", reverse=True)
    return out
