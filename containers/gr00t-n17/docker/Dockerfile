# GR00T N1.7 fine-tune container (Isaac-GR00T, Cosmos-Reason2-2B backbone), for AWS Batch.
#
# Third IL engine in the platform, alongside vla-ft (lerobot π0.5) and isaac-lab-rl.
# It fine-tunes NVIDIA GR00T N1.7 (a 3B VLA) on a LeRobot v2.1 dataset and emits a full
# merged HF checkpoint that `gr00t/eval/run_gr00t_server.py` loads for sim rollout.
#
# Install matches the VERIFIED upstream exactly (the M1 lesson: hand-assembling a dep
# stack instead of the source's own lock caused 6 consecutive build failures on vla-ft).
# Isaac-GR00T ships docker/Dockerfile that does `uv sync --frozen` from its committed
# uv.lock, so we do the same: clone the repo at the pinned commit and let uv install the
# exact locked closure (incl. the prebuilt flash-attn wheel declared in pyproject's
# [tool.uv.sources]) — NO hand-pinned pip installs to drift out of sync.
#
#
#   Source : github.com/NVIDIA/Isaac-GR00T @ 65cc4a192e6d  (N1.7 Early Access)
#   Base   : nvidia/cuda:12.8.0-devel-ubuntu22.04  (upstream's base)
#   Python : ==3.10.*  ·  uv 0.8.14 (upstream-pinned)  ·  torch 2.7.1 / transformers 4.57.3
#
# The Batch entrypoint (gr00t_train_bootstrap.py) is NOT baked in — it is injected at
# submit via `python3 -c "<zlib stub>"` (the JobDefinition command reads it at synth),
# exactly like vla-ft / isaac-lab-rl Pattern A, so this image stays task-agnostic and is
# never rebuilt to carry orchestration glue. The plain CUDA base has no streaming
# ENTRYPOINT (unlike the isaac-sim base), so — as proven by vla-ft Pattern A on the same
# cuda base — `python3 -c` injection runs directly; no `ENTRYPOINT []` reset is needed.
#
# ONE file IS baked: g1_finetune.py, a thin universal train entry that can set
# config.model.action_horizon (needed for the UNITREE_G1 full-body 50-step horizon, which
# launch_finetune.py exposes no flag for). It is ENGINE code, not orchestration glue, and
# baking it (vs writing it at runtime) keeps the zlib bootstrap payload under Batch's 8192B
# ECS override ceiling. With GROOT_ACTION_HORIZON unset it is byte-identical to
# launch_finetune.py.

FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PYTHONUNBUFFERED=1 \
    GROOT_COMMIT=65cc4a192e6d

# python3.10 (upstream's pin), git/git-lfs for the clone, build-essential for any source
# build uv may do, and the GL/ffmpeg libs the data/eval stack needs at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev python-is-python3 \
        git git-lfs curl ca-certificates build-essential \
        libgl1 libglib2.0-0 ffmpeg \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# uv, version-pinned to upstream's docker/Dockerfile (0.8.14), installed to /usr/local/bin.
RUN curl --retry 5 --retry-delay 10 --retry-max-time 300 -LsSf \
        https://astral.sh/uv/0.8.14/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

WORKDIR /workspace

# Clone Isaac-GR00T at the pinned commit. Full clone + checkout (the ref is a commit, not
# a tag, so --branch can't take it). The 3B weights are NOT in the repo — launch_finetune
# downloads nvidia/GR00T-N1.7-3B from HuggingFace at runtime (ungated, ~6 GB).
RUN git clone https://github.com/NVIDIA/Isaac-GR00T.git . \
    && git checkout ${GROOT_COMMIT}

# Install the locked dependency closure into /workspace/.venv, then the project editable.
# This is upstream's exact two-stage install:
#   1. uv sync --frozen --no-install-project --extra dev  → deps only, from uv.lock.
#      --extra dev pulls boto3 (the Batch bootstrap's only non-stdlib need) into the venv.
#   2. uv pip install -e . --no-deps                      → the gr00t package itself.
# --frozen requires uv.lock and never mutates it (reproducible). flash-attn (x86_64) comes
# from the pinned GitHub release wheel in [tool.uv.sources] — no --no-build-isolation dance.
ENV UV_PREVIEW=1 UV_HTTP_TIMEOUT=300 UV_CONCURRENT_DOWNLOADS=4
RUN uv sync --frozen --no-install-project --extra dev --no-cache
RUN uv pip install --python /workspace/.venv/bin/python -e . --no-deps

# Bake the universal train entry (action_horizon-aware). The bootstrap points the train
# subprocess at this when GROOT_ACTION_HORIZON is set (G1 full-body); otherwise it runs
# launch_finetune.py directly. It imports gr00t.* so it must sit beside the checkout, not
# replace it. The build context is containers/gr00t-n17/docker (Dockerfile + this file).
COPY g1_finetune.py /workspace/g1_finetune.py

# Make the venv the default interpreter. The injected `python3 -c "<bootstrap>"` command
# (and the launch_finetune.py / stats.py / g1_finetune.py subprocesses it spawns via
# sys.executable) then run inside the venv with the full gr00t stack + boto3. Headless GL
# for any render path.
ENV PATH="/workspace/.venv/bin:${PATH}" \
    VIRTUAL_ENV="/workspace/.venv" \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl

# Default to a shell; the Batch JobDefinition overrides this with `python3 -c "<stub>"`.
CMD ["/bin/bash"]
