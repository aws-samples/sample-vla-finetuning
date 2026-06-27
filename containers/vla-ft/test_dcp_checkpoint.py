#!/usr/bin/env python3
"""
Self-contained tests for dcp_checkpoint.py — the multi-node DCP helper with the verified
gloo-coordinator race fix. No torch / GPU needed: torch.distributed is STUBBED so the
gating + gloo-PG logic is exercised on a plain box (same convention as the other Python
suites here — pure-logic asserts, the real distributed path is verified on a multi-node run).

Run:  python3 test_dcp_checkpoint.py    (exits non-zero on first failure)
"""

import datetime
import sys
import types

import dcp_checkpoint as dcp


PASS, FAIL = 0, 0


def check(cond, label):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {label}")
    else:
        FAIL += 1
        print(f"  FAIL {label}")


class _StubDist:
    """Minimal torch.distributed stub. Records new_group calls so we can assert the gloo PG
    is created with backend='gloo' and the 30-min timeout."""
    def __init__(self, *, available=True, initialized=True, world_size=2,
                 new_group_raises=False):
        self._available = available
        self._initialized = initialized
        self._world_size = world_size
        self._new_group_raises = new_group_raises
        self.new_group_calls = []

    def is_available(self):
        return self._available

    def is_initialized(self):
        return self._initialized

    def get_world_size(self):
        return self._world_size

    def new_group(self, *, backend=None, timeout=None):
        self.new_group_calls.append({"backend": backend, "timeout": timeout})
        if self._new_group_raises:
            raise RuntimeError("simulated gloo group creation failure")
        return f"<gloo-pg backend={backend}>"


def _install_dist(stub):
    """Install a fake `torch.distributed` module so dcp_checkpoint's lazy imports hit it."""
    torch_mod = sys.modules.get("torch") or types.ModuleType("torch")
    torch_mod.distributed = stub
    sys.modules["torch"] = torch_mod
    sys.modules["torch.distributed"] = stub
    dcp._reset_gloo_pg_for_test()


def _uninstall_dist():
    sys.modules.pop("torch.distributed", None)
    sys.modules.pop("torch", None)
    dcp._reset_gloo_pg_for_test()


# ── gating: is_distributed ────────────────────────────────────────────────────────────
print("is_distributed gating:")
# No torch at all → False (the unit-test / single-node box).
_uninstall_dist()
check(dcp.is_distributed() is False, "no torch → is_distributed False (single-node path)")

# world_size 1 → False (single rank, even if 'distributed' is initialized).
_install_dist(_StubDist(world_size=1))
check(dcp.is_distributed() is False, "world_size 1 → False (single rank)")

# not initialized → False.
_install_dist(_StubDist(initialized=False))
check(dcp.is_distributed() is False, "not initialized → False")

# available + initialized + world_size>1 → True (a genuine multi-node run).
_install_dist(_StubDist(world_size=8))
check(dcp.is_distributed() is True, "available + init + ws8 → True (multi-node)")


# ── the gloo-coordinator fix: get_dcp_gloo_pg ─────────────────────────────────────────
print("gloo-coordinator PG (the DreamZero race fix):")
# Single-rank → None (no PG needed).
_install_dist(_StubDist(world_size=1))
check(dcp.get_dcp_gloo_pg() is None, "single-rank → no gloo PG (None)")

# Multi-rank → creates a gloo PG with backend='gloo' and the 30-min timeout.
stub = _StubDist(world_size=2)
_install_dist(stub)
pg = dcp.get_dcp_gloo_pg()
check(pg is not None, "multi-rank → gloo PG created")
check(len(stub.new_group_calls) == 1 and stub.new_group_calls[0]["backend"] == "gloo",
      "PG created with backend='gloo' (CPU transport, not NCCL/CUDA)")
check(stub.new_group_calls[0]["timeout"] == datetime.timedelta(minutes=30),
      "PG timeout = 30 min (the DreamZero patch value)")

# Cached: a second call does NOT create a second group (created once, reused).
pg2 = dcp.get_dcp_gloo_pg()
check(pg2 is pg and len(stub.new_group_calls) == 1, "gloo PG cached (created once, reused)")

# Creation failure → None (fall back to the default PG rather than no checkpoint).
stub_fail = _StubDist(world_size=2, new_group_raises=True)
_install_dist(stub_fail)
check(dcp.get_dcp_gloo_pg() is None, "gloo PG creation failure → None (graceful fallback)")


# ── dcp_save passes the gloo PG to dcp.save (the actual fix application) ───────────────
print("dcp_save wires the gloo PG into dcp.save:")
captured = {}


class _StubDcp:
    """Stub torch.distributed.checkpoint capturing the save() kwargs."""
    def save(self, payload, *, checkpoint_id=None, process_group=None):
        captured.clear()
        captured.update(payload=payload, checkpoint_id=checkpoint_id, process_group=process_group)

    def load(self, payload, *, checkpoint_id=None):
        captured.clear()
        captured.update(loaded=payload, checkpoint_id=checkpoint_id)
        return payload


def _install_dcp(stub_dist, stub_dcp):
    _install_dist(stub_dist)
    cp_mod = sys.modules.get("torch.distributed.checkpoint") or stub_dcp
    sys.modules["torch.distributed.checkpoint"] = stub_dcp
    # torch.distributed.checkpoint must be importable as `import torch.distributed.checkpoint`
    sys.modules["torch.distributed"].checkpoint = stub_dcp


import os
import tempfile

stub2 = _StubDist(world_size=4)
_install_dcp(stub2, _StubDcp())
with tempfile.TemporaryDirectory() as d:
    ckpt = os.path.join(d, "dcp_ckpt")
    out = dcp.dcp_save({"w": 1}, ckpt)
    check(out == ckpt and os.path.isdir(ckpt), "dcp_save creates the dir and returns it")
    check(captured["process_group"] is not None,
          "dcp_save passed a non-None process_group (the gloo PG) to dcp.save")
    check(captured["checkpoint_id"] == ckpt and captured["payload"] == {"state": {"w": 1}},
          "dcp_save wraps state under 'state' and uses checkpoint_id")

# Single-rank dcp_save → process_group None (default PG; no gloo needed, no race at ws1).
stub1 = _StubDist(world_size=1)
_install_dcp(stub1, _StubDcp())
with tempfile.TemporaryDirectory() as d:
    dcp.dcp_save({"w": 2}, os.path.join(d, "c"))
    check(captured["process_group"] is None, "single-rank dcp_save → process_group None (default PG)")

# dcp_load round-trips the state dict (in-place contract).
_install_dcp(_StubDist(world_size=2), _StubDcp())
with tempfile.TemporaryDirectory() as d:
    loaded = dcp.dcp_load({"w": 0}, os.path.join(d, "c"))
    check(loaded == {"w": 0}, "dcp_load returns the (resharded-in-place) state dict")

_uninstall_dist()
sys.modules.pop("torch.distributed.checkpoint", None)

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
