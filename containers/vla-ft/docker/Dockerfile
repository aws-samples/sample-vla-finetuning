# VLA-FT training container — BYOC for SageMaker Training Jobs (Pattern B).
#
# Stack matches the VERIFIED openarm-lift-pi05 EC2 run (which produced a working
# π0.5 checkpoint): lerobot 0.5.x line, Python 3.12, cu128 torch, transformers 5.x.
#   - Earlier this image used the pip wheel lerobot==0.4.4 on py3.11 with a
#     hand-pinned transformers. That FAILS for pi05: the 0.4.4 wheel's pi0/pi05
#     need a `transformers_replace` patch (`from transformers.models.siglip import
#     check`) that the wheel doesn't ship → "An incorrect transformer version is
#     used" at model build, no matter the transformers pin.
#   - lerobot 0.5.1 dropped that patch (stock transformers 5.3.0 works) and is a
#     clean PyPI wheel. We install `lerobot[pi]==0.5.1` and let pip resolve the
#     policy deps (transformers/scipy) + the core closure (datasets/accelerate/
#     av/wandb/...) — no hand-pinned dep wall to drift out of sync.
#
# The sagemaker-training toolkit makes this a valid SageMaker BYOC image (handles
# source_dir extraction, hyperparameters.json, env, artifact upload), so launch.py
# ships src/train.py as source_dir without baking it in. Same image is backend-
# portable to AWS Batch (Pattern A) via `python /opt/ml/code/train.py` + SM_* env.

# Ubuntu 24.04 ships Python 3.12 natively; CUDA 12.8 matches the cu128 torch index
# the verified EC2 run used (uv [tool.uv.index] pytorch-cu128).
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3.12-venv python3-pip \
    git git-lfs ffmpeg libsm6 libxext6 libgl1 libglx-mesa0 libglib2.0-0 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu 24.04 marks the system Python "externally managed" (PEP 668); we own this
# image so install globally without a venv.
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# torch TRIO pinned to the EXACT versions in the verified run's uv.lock (d1b1c5c):
# torch 2.11.0+cu128 / torchvision 0.26.0+cu128 / torchcodec 0.11.1. These are an
# ABI-matched set — torchcodec 0.11.1's libtorchcodec_core*.so links against torch
# 2.11's C++ ABI (symbol torch_dtype_float4_e2m1fn_x2). An earlier build pinned
# torch 2.7.1 but let torchcodec resolve to 0.11.1; the DataLoader then died with
# "Could not load libtorchcodec ... undefined symbol: torch_dtype_float4_e2m1fn_x2"
# at the first video-decode. Pinning the whole trio to the lock removes that skew.
# torch/torchvision come from the cu128 index; torchcodec is a PyPI wheel that links
# the system ffmpeg 6 (apt) at runtime via libtorchcodec_core6.so.
RUN pip install --no-cache-dir \
    torch==2.11.0 torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir torchcodec==0.11.1

# SageMaker training toolkit — BYOC contract (same as AWS DLCs).
RUN pip install --no-cache-dir sagemaker-training

# lerobot from the EXACT verified commit (d1b1c5c, reports version 0.5.2-dev) with
# the π0/π0.5 policy extra. This is the commit the openarm-lift-pi05 EC2 run used
# (`git checkout d1b1c5c` + `uv sync --extra pi --extra training`).
#   Why git, not the PyPI wheel: the lerobot/pi05_base checkpoint's saved processor
#   config references the 'relative_actions_processor' step, which was added AFTER
#   0.5.1 (the latest PyPI release; 0.5.2 was never published). Loading pi05_base on
#   0.5.1 dies with "Processor step 'relative_actions_processor' not found in
#   registry". d1b1c5c (0.5.1 + 113 commits) registers it, matching the checkpoint.
# Extras mirror the verified EC2 `uv sync --extra pi --extra training`, PLUS [peft]
# and [smolvla]:
#   [pi]       -> transformers (5.4–5.6) + scipy (the π0/π0.5 policy deps)
#   [training] -> lerobot[dataset] (datasets/pandas/pyarrow/av) + accelerate
#   [peft]     -> peft>=0.18,<1.0  (NOT pulled in by [pi]/[training] — added so LoRA works)
#   [smolvla]  -> transformers (same 5.4–5.6 as [pi]) + num2words>=0.5.14 + accelerate>=1.7
# In this commit datasets/accelerate are NOT core — they live in [dataset]/[training],
# and lerobot_train.py hard-requires them (require_package("datasets", extra="dataset")),
# so [pi] alone fails at import with "'datasets' is required but not installed".
# [peft] is ADDITIVE and inert unless a run sets cfg.peft (--peft.* / our --lora): lerobot
# only imports peft inside the `if cfg.peft is not None` wrap path (lerobot_train.py) and
# the use_peft load branch (policies/factory.py). Verified non-LoRA runs never touch it,
# so the default training path stays behavior-identical — but LoRA would ImportError
# without it ([pi,training] does not install peft). Pinned to peft's own extra range.
# [smolvla] is ADDITIVE and does NOT touch the pi05 verified lock: at d1b1c5c8 its only
# net-new dep is num2words (>=0.5.14,<0.6.0) — the SmolVLM processor imports it eagerly,
# so [pi,training,peft] alone dies at "Package 'num2words' is required to run SmolVLM
# processor" the moment SmolVLAPolicy.from_pretrained loads the processor. smolvla's other
# two deps are already satisfied: transformers (5.4–5.6, identical alias to [pi]) and
# accelerate (>=1.7, looser than [training]'s >=1.10 → no resolver change). So one image
# serves both the verified pi05 path AND smolvla — no separate smolvla-only image needed.
# torch is already satisfied (>=2.7,<2.12) so it stays cu128. [tool.uv.sources] is a uv
# directive pip ignores — harmless. policies.__init__ imports all policies eagerly.
RUN pip install --no-cache-dir \
    "lerobot[pi,training,peft,smolvla] @ git+https://github.com/huggingface/lerobot.git@d1b1c5c8"

ENV WANDB_MODE=offline
ENV PYTHONUNBUFFERED=1
# sagemaker-training discovers the entry point from source_dir at runtime;
# SAGEMAKER_PROGRAM is set by the estimator. No COPY of train.py here — launch.py
# ships src/ as source_dir so we iterate on train.py without rebuilding the image.

# ─────────────────────── EFA / NCCL fabric overlay (multi-node, opt-in) ───────────────
# Pattern C multi-node FSDP2 needs EFA RDMA for NCCL inter-node collectives. This overlay
# installs the EXACT verified fabric stack from awslabs/awsome-distributed-ai DreamZero
# (MIT-0, fetched 2026-06-27): EFA installer 1.47.0 (bundles libfabric 2.4.0amzn1.0 +
# aws-ofi-nccl 1.18.0 + OpenMPI) and GDRCopy 2.5.1. Version pins are copied VERBATIM from
# the verified source, not hand-assembled (the torch-trio ABI lesson, applied to fabric).
#
# Gated behind ENABLE_EFA (default 0) so the DEFAULT build stays byte-identical to the
# verified single-node pi05 lock — single-node Pattern A/B never touch EFA. The overlay
# is APPENDED after the pip layers so a default and an --build-arg ENABLE_EFA=1 build share
# every cached layer up to here. Build the multi-node image with:
#   docker build --build-arg ENABLE_EFA=1 -t <repo>:efa .   (or build.sh --efa)
#
# Deliberately UNLIKE DreamZero's overlay (which sits on a CUDA-12.4 base where RLinf
# source-builds torch + NCCL): we DO NOT source-build NCCL and DO NOT LD_PRELOAD one. This
# image's torch is the pip wheel 2.11.0+cu128, which BUNDLES its own cu128 NCCL
# (torch/lib/libnccl.so.2, located via the wheel's RPATH). aws-ofi-nccl is a NET *plugin*
# that the bundled NCCL auto-discovers on LD_LIBRARY_PATH — exactly the Cosmos3 AWS-DLC
# path ("we do NOT set NCCL_NET_PLUGIN to an NGC path; the version-matched aws-ofi-nccl is
# auto-discovered"). Source-building + preloading a second NCCL here would re-introduce the
# very torch<->NCCL ABI skew the trio pin above was added to kill.
ARG ENABLE_EFA=0
ARG EFA_INSTALLER_VERSION=1.47.0
ARG GDRCOPY_VERSION=v2.5.1
# One RUN, guarded on the ARG: a no-op echo when ENABLE_EFA != 1, so the default image
# gains only an empty cached layer and is otherwise unchanged.
RUN if [ "$ENABLE_EFA" = "1" ]; then set -eux; \
      apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl pkg-config libtool autoconf automake \
        libhwloc15 libevent-pthreads-2.1-7 && \
      git clone -b ${GDRCOPY_VERSION} https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy && \
      make -C /tmp/gdrcopy prefix=/opt/gdrcopy lib_install && rm -rf /tmp/gdrcopy && \
      cd /tmp && \
      curl -fsSL -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz && \
      tar -xf aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz && \
      cd aws-efa-installer && \
      ./efa_installer.sh -y -g -d --skip-kmod --skip-limit-conf --no-verify && \
      rm -rf /tmp/aws-efa-installer* /var/lib/apt/lists/* && \
      ( ls -la /opt/amazon/ofi-nccl/lib/x86_64-linux-gnu/libnccl-ofi*.so* || \
        ls -la /opt/amazon/ofi-nccl/lib/libnccl-ofi*.so* ); \
    else echo "ENABLE_EFA != 1 — skipping EFA overlay (single-node image)"; fi

# EFA / NCCL runtime env. INERT on a single-node host: when no EFA device is present (or the
# overlay was skipped) libfabric is simply never invoked — single-GPU runs init no NCCL, and
# multi-GPU DDP uses intra-node P2P/SHM, not the OFI net path. So these are safe to set
# unconditionally and match the verified DreamZero/Cosmos3 multi-node manifests verbatim.
#   - --skip-kmod: the EFA kernel driver lives on the host AMI, not the container.
#   - --no-verify / --skip-limit-conf: the build host has no EFA device / sysctl to probe.
# LD_LIBRARY_PATH puts the EFA + OFI-plugin + GDRCopy libs on the search path so the bundled
# NCCL auto-discovers the OFI NET plugin (libnccl-net.so) on a multi-node run; on a default
# image those dirs don't exist and are ignored.
ENV LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu:/opt/amazon/ofi-nccl/lib:/opt/gdrcopy/lib:${LD_LIBRARY_PATH:-/usr/local/cuda/lib64}
ENV PATH=/opt/amazon/efa/bin:/opt/gdrcopy/bin:${PATH}
# Verified EFA provider selection (DreamZero + Cosmos3 multi-node): force the EFA libfabric
# provider; DEVICE_RDMA + FORK_SAFE are the p5/g6e EFA defaults; HUGE_PAGE=0 per DreamZero.
ENV FI_PROVIDER=efa
ENV FI_EFA_USE_DEVICE_RDMA=1
ENV FI_EFA_FORK_SAFE=1
ENV FI_EFA_USE_HUGE_PAGE=0
# NB: we deliberately DO NOT set NCCL_TUNER_PLUGIN. aws-ofi-nccl 1.18 auto-registers its
# tuner when the net plugin loads (multi-node EFA image), so the explicit path is
# redundant there — and on the DEFAULT single-node image (no OFI plugin) an explicit path
# to a non-existent .so makes NCCL print a "could not open tuner plugin" WARN on every
# verified Pattern B multi-GPU run. Omitting it keeps the verified path log-clean.
