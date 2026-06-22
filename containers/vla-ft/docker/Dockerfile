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
# Extras mirror the verified EC2 `uv sync --extra pi --extra training`, PLUS [peft]:
#   [pi]       -> transformers (5.4–5.6) + scipy (the π0/π0.5 policy deps)
#   [training] -> lerobot[dataset] (datasets/pandas/pyarrow/av) + accelerate
#   [peft]     -> peft>=0.18,<1.0  (NOT pulled in by [pi]/[training] — added so LoRA works)
# In this commit datasets/accelerate are NOT core — they live in [dataset]/[training],
# and lerobot_train.py hard-requires them (require_package("datasets", extra="dataset")),
# so [pi] alone fails at import with "'datasets' is required but not installed".
# [peft] is ADDITIVE and inert unless a run sets cfg.peft (--peft.* / our --lora): lerobot
# only imports peft inside the `if cfg.peft is not None` wrap path (lerobot_train.py) and
# the use_peft load branch (policies/factory.py). Verified non-LoRA runs never touch it,
# so the default training path stays behavior-identical — but LoRA would ImportError
# without it ([pi,training] does not install peft). Pinned to peft's own extra range.
# torch is already satisfied (>=2.7,<2.12) so it stays cu128. [tool.uv.sources] is a uv
# directive pip ignores — harmless. policies.__init__ imports all policies eagerly.
RUN pip install --no-cache-dir \
    "lerobot[pi,training,peft] @ git+https://github.com/huggingface/lerobot.git@d1b1c5c8"

ENV WANDB_MODE=offline
ENV PYTHONUNBUFFERED=1
# sagemaker-training discovers the entry point from source_dir at runtime;
# SAGEMAKER_PROGRAM is set by the estimator. No COPY of train.py here — launch.py
# ships src/ as source_dir so we iterate on train.py without rebuilding the image.
