#!/usr/bin/env bash
# fix_managed_slurm_conf.sh — reconcile the verbatim awslabs base-config with HyperPod's
# CFN-native managed-Slurm slurm.conf, on the controller, at cluster create.
#
# WHY THIS EXISTS
#   In managed-Slurm mode (Orchestrator:{Slurm} on the cluster) HyperPod GENERATES
#   /opt/slurm/etc/slurm.conf itself, and that generated file carries both
#   `Include accounting.conf` and `TopologyPlugin=topology/tree`. The verbatim base-config
#   bundle, meanwhile, fills accounting.conf with `GresTypes=gpu` (setup_mariadb_accounting.sh)
#   and ships no topology.conf / switch definitions. The two were authored assuming the bundle
#   owns slurm.conf, so together they make slurmctld crash-loop at first start:
#     "No Assoc usage file to recover" / "GresTypes specified more than once" /
#     "No switches configured"
#   leaving the cluster InService but Slurm dead (sinfo empty).
#
# WHAT IT DOES
#   Replicates the manual unblock VALIDATED on the 2026-06-28 2-node smoke: comment the two
#   colliding directives in the LIVE slurm.conf, then restart slurmctld. It does NOT touch the
#   bundle's accounting.conf — slurm.conf is the file slurmctld actually loads, and commenting
#   `Include accounting.conf` there removes the duplicate GresTypes + the half-wired accounting
#   in one move (exactly what the manual fix did).
#
# PROPERTIES
#   - Controller-only: workers mask/rename slurmctld.service in start_slurm.sh, so the
#     `systemctl cat slurmctld.service` probe is a clean "am I the head node?" signal.
#   - Idempotent: a sentinel marker makes re-runs (enroot's later slurmctld restart, a node
#     reboot re-running on_create.sh) no-ops.
#   - Never fatal: returns 0 in every path so it cannot abort on_create.sh (which runs under
#     `set -e`). The caller also guards it.
#
# This file is OUR additive fix, copied into the staged bundle and called from on_create.sh's
# tail by stage-lifecycle.sh. The pinned base-config scripts are left byte-for-byte verbatim.
set -uo pipefail   # deliberately NOT -e: must never abort the caller

SLURM_CONF="${SLURM_CONF:-/opt/slurm/etc/slurm.conf}"
MARKER="# pai-managed-slurm-reconciled"

log() { echo "[fix_managed_slurm_conf] $*"; }

# --- Controller-only guard -----------------------------------------------------------------
# On workers start_slurm.sh renames slurmctld.service away, so `systemctl cat` fails there.
if ! systemctl cat slurmctld.service >/dev/null 2>&1; then
  log "no live slurmctld.service on this node (worker / not head) — nothing to do."
  exit 0
fi
if [[ ! -f "$SLURM_CONF" ]]; then
  log "no $SLURM_CONF present — nothing to do."
  exit 0
fi

# --- Idempotency ----------------------------------------------------------------------------
if grep -q "$MARKER" "$SLURM_CONF"; then
  log "already reconciled (marker present) — skipping."
  exit 0
fi

# --- Comment the two colliding directives (leave everything else byte-for-byte) -------------
# Broad enough to match both the relative form (`Include accounting.conf`) and a full-path
# form (`Include /opt/slurm/etc/accounting.conf`) should HyperPod emit one.
changed=0
if grep -Eq '^[[:space:]]*Include[[:space:]]+.*accounting\.conf' "$SLURM_CONF"; then
  sed -i.bak -E 's|^([[:space:]]*Include[[:space:]]+.*accounting\.conf.*)$|#\1  '"$MARKER"'|' "$SLURM_CONF"
  changed=1; log "commented: Include ...accounting.conf"
fi
if grep -Eq '^[[:space:]]*TopologyPlugin[[:space:]]*=[[:space:]]*topology/tree' "$SLURM_CONF"; then
  sed -i.bak -E 's|^([[:space:]]*TopologyPlugin[[:space:]]*=[[:space:]]*topology/tree.*)$|#\1  '"$MARKER"'|' "$SLURM_CONF"
  changed=1; log "commented: TopologyPlugin=topology/tree"
fi
rm -f "${SLURM_CONF}.bak"

if [[ "$changed" -eq 0 ]]; then
  # Neither directive present — managed slurm.conf may already be clean (or HyperPod changed
  # its template). Drop the marker so we don't rescan on every invocation, but do NOT restart.
  log "neither colliding directive found — slurm.conf may already be clean. No restart."
  printf '%s (no-op: directives absent)\n' "$MARKER" >> "$SLURM_CONF"
  exit 0
fi

# --- Restart slurmctld onto the reconciled config -------------------------------------------
log "restarting slurmctld onto reconciled slurm.conf ..."
if ! systemctl restart slurmctld; then
  log "WARN: 'systemctl restart slurmctld' returned non-zero — check 'journalctl -u slurmctld'."
  exit 0
fi
sleep 3
if systemctl is-active --quiet slurmctld; then
  log "OK: slurmctld active after reconciliation."
else
  log "WARN: slurmctld not active after restart — inspect 'journalctl -xeu slurmctld'."
fi
exit 0
