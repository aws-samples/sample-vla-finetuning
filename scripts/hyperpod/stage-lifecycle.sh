#!/usr/bin/env bash
# stage-lifecycle.sh — stage the HyperPod Slurm provisioning lifecycle bundle to S3.
#
# Pattern C (lib/il/hyperpod-stack.ts) points the cluster's ClusterLifeCycleConfig.OnCreate
# at s3://sagemaker-<region>-<account>/<prefix>/. HyperPod runs that bundle on EVERY node at
# cluster create to install/configure Slurm + mount FSx. The bundle MUST already be in S3
# before `cdk deploy` — an empty/missing prefix means each node's provisioning fails and the
# (multi-GPU, billing) cluster never forms Slurm. This script puts the verified bundle there.
#
# It does NOT hand-write the lifecycle scripts. It fetches the base-config directory VERBATIM
# from a PINNED commit of aws-samples/awsome-distributed-training (MIT-0) — the match-verified
# lock — and only generates the one cluster-specific file the bundle can't ship: the
# provisioning_parameters.json (group names + optional FSx coordinates).
#
# Usage:
#   scripts/hyperpod/stage-lifecycle.sh [--region us-west-2] [--prefix pai/il-hyperpod-lifecycle]
#       [--bucket sagemaker-<region>-<account>] [--worker-instance-type ml.g6e.48xlarge]
#       [--controller-group controller-machine] [--worker-group worker-group-1]
#       [--fsx-id fs-0abc... ]    # OMIT for an FSx-free first deploy (recommended smoke)
#       [--dry-run]
#
# The defaults match lib/il/hyperpod-stack.ts (namePrefix=pai, controller-machine /
# worker-group-1, ml.g6e.48xlarge workers). Pass --fsx-id only AFTER the FSx filesystem
# exists (see scripts/hyperpod/README.md "FSx chicken-and-egg").
set -euo pipefail

# --- Verified source pin (match-verified-lock). Bump deliberately, never float to main. ----
PIN="34a09f120908c2eff72e9363acc9c4be41f34760"
SRC_SUBDIR="1.architectures/5.sagemaker-hyperpod/LifecycleScripts/base-config"
TARBALL_URL="https://codeload.github.com/aws-samples/awsome-distributed-training/tar.gz/${PIN}"

# --- Defaults (mirror lib/il/hyperpod-stack.ts) --------------------------------------------
REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-west-2}}"
PREFIX="pai/il-hyperpod-lifecycle"
BUCKET=""
WORKER_INSTANCE_TYPE="ml.g6e.48xlarge"
CONTROLLER_GROUP="controller-machine"
WORKER_GROUP="worker-group-1"
FSX_ID=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --region) REGION="$2"; shift 2;;
    --prefix) PREFIX="$2"; shift 2;;
    --bucket) BUCKET="$2"; shift 2;;
    --worker-instance-type) WORKER_INSTANCE_TYPE="$2"; shift 2;;
    --controller-group) CONTROLLER_GROUP="$2"; shift 2;;
    --worker-group) WORKER_GROUP="$2"; shift 2;;
    --fsx-id) FSX_ID="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
if [[ -z "$BUCKET" ]]; then
  # HyperPod's AmazonSageMakerClusterInstanceRolePolicy only grants the sagemaker- prefix.
  BUCKET="sagemaker-${REGION}-${ACCOUNT}"
fi
S3_DEST="s3://${BUCKET}/${PREFIX}/"

echo "============================================================"
echo "HyperPod lifecycle staging"
echo "  source pin : ${PIN}"
echo "  dest       : ${S3_DEST}"
echo "  groups     : controller=${CONTROLLER_GROUP}  worker=${WORKER_GROUP} (${WORKER_INSTANCE_TYPE})"
echo "  fsx        : ${FSX_ID:-<none — FSx-free deploy>}"
echo "  dry-run    : ${DRY_RUN}"
echo "============================================================"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# --- 1. Fetch the verified base-config bundle VERBATIM from the pinned commit ---------------
echo "[1/4] fetching base-config @ ${PIN} ..."
curl -fsSL --max-time 120 "$TARBALL_URL" -o "${WORK}/src.tar.gz"
tar -xzf "${WORK}/src.tar.gz" -C "$WORK"
# The tarball's top-level dir is <repo>-<pin>; the repo was renamed (awsome-distributed-ai),
# so resolve it dynamically rather than hardcoding the name.
TOPDIR="$(tar -tzf "${WORK}/src.tar.gz" 2>/dev/null | head -1 | cut -d/ -f1)"
BUNDLE="${WORK}/${TOPDIR}/${SRC_SUBDIR}"
if [[ ! -f "${BUNDLE}/on_create.sh" || ! -f "${BUNDLE}/lifecycle_script.py" ]]; then
  echo "ERROR: expected lifecycle scripts not found under ${BUNDLE}" >&2
  exit 1
fi
echo "      ok — $(find "$BUNDLE" -type f | wc -l | tr -d ' ') files (incl. utils/, observability/)."

# --- 2. Do NOT generate provisioning_parameters.json (CFN-native managed Slurm) ------------
# lib/shared/hyperpod-cluster.ts declares the cluster with Orchestrator:{Slurm} + per-group
# SlurmConfig.NodeType. In that managed-Slurm mode HyperPod GENERATES provisioning_parameters.json
# itself (from the InstanceGroups/SlurmConfig declaration) and drops it into the lifecycle working
# dir, where the verified on_create.sh reads it by relative path. Staging our OWN copy alongside a
# Slurm Orchestrator Config is rejected at CREATE: "The LifeCycleConfig cannot include both a SLURM
# Orchestrator Config and a provisioning_parameters.json file simultaneously." So we stage the
# base-config bundle VERBATIM and let HyperPod own the parameters file. (--controller-group /
# --worker-group / --worker-instance-type / --fsx-id are accepted for back-compat but unused here;
# group names + FSx are declared in the CDK stack, not this file.)
echo "[2/4] managed-Slurm mode: provisioning_parameters.json is generated by HyperPod (not staged)."
if [[ -n "$FSX_ID" ]]; then
  echo "      note: --fsx-id is ignored in managed-Slurm mode — FSx is wired via the CDK stack" >&2
  echo "            (-c hyperPodFsx=true), not provisioning_parameters.json. See README." >&2
fi
# Belt-and-suspenders: if the upstream bundle ever ships one, drop it so it can't conflict.
rm -f "${BUNDLE}/provisioning_parameters.json"

# --- 3. Validate the bundle's bash entrypoints parse (cheap pre-deploy gate) ----------------
echo "[3/4] bash -n gate on entrypoints ..."
for s in on_create.sh mount_fsx.sh start_slurm.sh; do
  [[ -f "${BUNDLE}/${s}" ]] && bash -n "${BUNDLE}/${s}" && echo "      bash -n ok: ${s}"
done
python3 -m py_compile "${BUNDLE}/lifecycle_script.py" && echo "      py_compile ok: lifecycle_script.py"

# --- 4. Sync to S3 (the OnCreate sourceS3Uri the cluster reads at create) -------------------
# Exclude __pycache__: the step-3 py_compile gate leaves a local-Python-version .pyc in the
# bundle; it is a build artifact, not part of the verified-lock source, so keep it out of S3.
echo "[4/4] syncing to ${S3_DEST} ..."
if [[ "$DRY_RUN" == "1" ]]; then
  echo "      DRY-RUN — would: aws s3 sync ${BUNDLE}/ ${S3_DEST} --delete --exclude '__pycache__/*'"
  aws s3 sync "${BUNDLE}/" "$S3_DEST" --delete --exclude '__pycache__/*' --dryrun
else
  aws s3 sync "${BUNDLE}/" "$S3_DEST" --delete --exclude '__pycache__/*'
  echo "      done. verified base-config bundle (on_create.sh + installers + observability)"
  echo "      staged at ${S3_DEST}. provisioning_parameters.json is generated by HyperPod."
fi

echo "============================================================"
echo "Next: cdk deploy PaiTrainingPlatform-IL-HyperPod -c enableHyperPod=true \\"
[[ -n "$FSX_ID" ]] && echo "        -c hyperPodFsx=true \\"
echo "        -c region=${REGION}"
echo "See scripts/hyperpod/README.md for the full deploy + sbatch + teardown runbook."
echo "============================================================"
