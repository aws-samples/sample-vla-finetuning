#!/usr/bin/env bash
# Multi-node FSDP2 launch reference for Pattern C (SageMaker HyperPod, Slurm orchestrator).
#
# This is the layer Pattern C was missing: the cluster CDK (lib/il/hyperpod-stack.ts) stands
# up the nodes, but until now there was no launch path that actually runs a SHARDED (FSDP2)
# multi-node fine-tune across them. This script is that path — a Slurm batch script you
# `sbatch` on the HyperPod head node. It runs the UNCHANGED train.py (no fork): train.py's
# build_command detects the multi-node env (NNODES/NODE_RANK/MASTER_ADDR set below) and
# launches `accelerate launch --use_fsdp` across all ranks. It ties together:
#   - Phase 1 EFA fabric: the EFA/NCCL env (FI_PROVIDER=efa etc.) baked into the :efa image
#     (build.sh --efa) and re-asserted here, so cross-node NCCL collectives ride EFA RDMA.
#   - Phase 3 DCP: --fsdp_state_dict_type=SHARDED_STATE_DICT (set by train.py) so the FSDP
#     checkpoint is sharded; dcp_checkpoint.py carries the gloo-coordinator race fix for the
#     sharded save path. Checkpoints land on the FSx Lustre hot tier (→ S3 via the DRA).
#   - The verified torchrun/FSDP2 rendezvous from awslabs/awsome-distributed-ai (Cosmos3
#     JobSet `torchrun --rdzv_backend=c10d`), ported to Slurm's srun + accelerate.
#
# WHY accelerate-over-lerobot (not a raw FSDP trainer): vla-ft wraps lerobot via accelerate
# — that is the verified core, and accelerate natively does multi-node + FSDP. A parallel
# raw-FSDP train_fsdp.py would FORK the training loop the platform is built NOT to fork. So
# multi-node is an ADDITIVE branch in the same train.py the single-node path uses.
#
# WHY Slurm (not the source's EKS/JobSet): the whole platform is CDK + Slurm HyperPod;
# matching EKS+KubeRay/JobSet would introduce a K8s operational layer (a fork, not an
# absorption). HyperPod-Slurm gives the same multi-node FSDP2 + EFA + FSx + DCP via
# srun/accelerate. The EKS path is a documented future option (README "Pattern C → EKS").
#
# DEPLOY-GATED: runs on a HyperPod cluster (high standing cost) — only after
#   cdk deploy PaiTrainingPlatform-IL-HyperPod -c enableHyperPod=true -c hyperPodFsx=true
#
# Usage (on the HyperPod head node, after staging the image + dataset to FSx):
#   IMAGE_URI=<acct>.dkr.ecr.<region>.amazonaws.com/pai/vla-ft:efa \
#     sbatch --nodes=2 hyperpod_fsdp_launch.sh
# Env knobs (override at sbatch time):
#   IMAGE_URI      ECR uri of the EFA fabric image (build.sh --efa output)   [required]
#   FSX_MOUNT      Lustre mount on each node (lifecycle on_create.sh mounts) [/fsx]
#   DATASET_DIR    LeRobot v3 dataset dir under the FSx mount    [$FSX_MOUNT/datasets/lerobot]
#   OUTPUT_DIR     checkpoints/model dir under the FSx mount     [$FSX_MOUNT/checkpoints/<job>]
#   POLICY         lerobot policy (pi05/pi0/...)                              [pi05]
#   STEPS          training steps                                            [20000]
#   GPUS_PER_NODE  GPUs per node (g6e.48xlarge=8)                            [8]
#
#SBATCH --job-name=vla-ft-fsdp
#SBATCH --output=%x_%j.out
#SBATCH --exclusive
set -euo pipefail

# ── Cluster topology (Slurm fills these on the allocation) ────────────────────────────
NNODES="${SLURM_NNODES:-${NNODES:-2}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
# Rendezvous on the first node of the allocation (c10d backend — the verified Cosmos3 rdzv).
MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST:-localhost}" | head -n1)"
MASTER_PORT="${MASTER_PORT:-29500}"

IMAGE_URI="${IMAGE_URI:?set IMAGE_URI to the EFA fabric image (build.sh --efa output)}"
FSX_MOUNT="${FSX_MOUNT:-/fsx}"
DATASET_DIR="${DATASET_DIR:-${FSX_MOUNT}/datasets/lerobot}"
OUTPUT_DIR="${OUTPUT_DIR:-${FSX_MOUNT}/checkpoints/${SLURM_JOB_NAME:-vla-ft-fsdp}-${SLURM_JOB_ID:-local}}"
POLICY="${POLICY:-pi05}"
STEPS="${STEPS:-20000}"

# ── EFA / NCCL fabric (Phase 1) ───────────────────────────────────────────────────────
# Match the verified DreamZero/Cosmos3 multi-node manifests AND the :efa image's baked ENV.
# Re-asserted here so the launch is self-documenting + robust to an older image.
# NCCL_DEBUG=INFO so the logs PROVE EFA was selected (NET/OFI = EFA RDMA; NET/Socket = TCP
# fallback = misconfig to fix before paying for a multi-node run).
export FI_PROVIDER="${FI_PROVIDER:-efa}"
export FI_EFA_USE_DEVICE_RDMA="${FI_EFA_USE_DEVICE_RDMA:-1}"
export FI_EFA_FORK_SAFE="${FI_EFA_FORK_SAFE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET,ENV}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^docker,lo,veth}"

echo "============================================================"
echo "VLA-FT Pattern C — multi-node FSDP2 (HyperPod Slurm, unchanged train.py)"
echo "  nodes=${NNODES}  gpus/node=${GPUS_PER_NODE}  master=${MASTER_ADDR}:${MASTER_PORT}"
echo "  image=${IMAGE_URI}"
echo "  dataset=${DATASET_DIR}  output=${OUTPUT_DIR}  policy=${POLICY} steps=${STEPS}"
echo "  EFA: FI_PROVIDER=${FI_PROVIDER} RDMA=${FI_EFA_USE_DEVICE_RDMA}"
echo "============================================================"

# ── Launch: one task per node via srun; each runs the UNCHANGED train.py in the container ─
# srun fans one task to each node. Inside the container we export the multi-node env
# (NNODES/NODE_RANK/MASTER_ADDR/MASTER_PORT) that train.py.resolve_distributed() reads, then
# run train.py — which builds `accelerate launch --use_fsdp --num_machines=NNODES
# --machine_rank=NODE_RANK ...` and lerobot trains sharded across all ranks. NODE_RANK comes
# from Slurm's SLURM_NODEID. SageMaker's SM_* paths map onto the same train.py interface.
srun --ntasks="${NNODES}" --ntasks-per-node=1 --gpus-per-node="${GPUS_PER_NODE}" \
  bash -c '
    docker run --rm --gpus all --network host \
      --device /dev/infiniband --cap-add IPC_LOCK --ipc host \
      -v '"${FSX_MOUNT}"':'"${FSX_MOUNT}"' \
      -e FI_PROVIDER -e FI_EFA_USE_DEVICE_RDMA -e FI_EFA_FORK_SAFE \
      -e NCCL_DEBUG -e NCCL_DEBUG_SUBSYS -e NCCL_SOCKET_IFNAME \
      -e NNODES='"${NNODES}"' \
      -e NODE_RANK=${SLURM_NODEID} \
      -e MASTER_ADDR='"${MASTER_ADDR}"' \
      -e MASTER_PORT='"${MASTER_PORT}"' \
      -e SM_NUM_GPUS='"${GPUS_PER_NODE}"' \
      -e SM_CHANNEL_TRAINING='"${DATASET_DIR}"' \
      -e VLA_FT_CHECKPOINT_DIR='"${OUTPUT_DIR}"' \
      -e SM_HP_policy='"${POLICY}"' \
      -e SM_HP_steps='"${STEPS}"' \
      -e SM_HP_full_vlm=true \
      '"${IMAGE_URI}"' \
      python3 '"${FSX_MOUNT}"'/code/train.py
  '
# NOTE on train.py staging: the image is BYOC (train.py ships as source_dir, not baked), so
# the operator stages the verified src/train.py to ${FSX_MOUNT}/code/train.py once before
# sbatch (e.g. `aws s3 cp s3://<artifacts>/vla-ft-code/train.py ${FSX_MOUNT}/code/train.py`).
# Same S3-iterable train.py the Batch bootstrap fetches — no rebuild to iterate on it.

echo "[hyperpod-fsdp] srun returned $? — sharded checkpoints under ${OUTPUT_DIR} (FSx → S3 via DRA)."
