# Isaac Lab RL training container (headless PPO via rsl_rl), for AWS Batch / HyperPod.
#
# Base = the NVIDIA Isaac Sim runtime; Isaac Lab is installed on top at a pinned tag.
# This mirrors a verified Isaac Lab workshop build (the userdata + workshop Dockerfile
# install path) but is self-contained: it clones IsaacLab at a fixed
# tag and runs the official installer at build time, instead of fetching workshop
# assets at instance boot.
#
#   Isaac Sim 4.5.0  +  Isaac Lab v2.3.2  (the README "Isaac Sim Version Dependency"
#   table maps v2.3.X -> Isaac Sim 4.5/5.0/5.1; v2.3.2 is the latest v2.3 tag).
#
# `./isaaclab.sh -i` installs ALL learning frameworks by default (rsl_rl, skrl, sb3,
# rl_games, robomimic), so rsl_rl (our ONNX-export path) needs no separate install.
#
# The Batch entrypoint (rl_train_bootstrap.py) is NOT baked in — it is injected at
# job submit via `python3 -c "<src>"` (the JobDefinition command reads it at synth),
# exactly like vla-ft's Pattern A, so this image stays generic across RL tasks and is
# never rebuilt to carry orchestration glue.

FROM nvcr.io/nvidia/isaac-sim:4.5.0

# Headless / root-in-container env. ACCEPT_EULA + PRIVACY_CONSENT are the
# doc-cited requirements to run the isaac-sim container non-interactively;
# OMNI_KIT_ALLOW_ROOT lets Kit run as root inside the container (NVIDIA Dockerfile
# convention). NVIDIA_DRIVER_CAPABILITIES exposes the GPU to the runtime.
# TERM=xterm gives `isaaclab.sh` a real terminfo entry: the installer runs `set -e`
# then `tabs 4` very early, and under BuildKit's no-TTY environment an unset/`ansi`
# TERM makes `tabs` fail with "'ansi+tabs': unknown terminal type" (the ansi+tabs
# terminfo building block lives in ncurses-term, not the base image's ncurses-base),
# which `set -e` turns into an aborted install. xterm + ncurses-term below fix it.
ENV ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    OMNI_KIT_ALLOW_ROOT=1 \
    NVIDIA_DRIVER_CAPABILITIES=all \
    DEBIAN_FRONTEND=noninteractive \
    TERM=xterm \
    ISAACLAB_TAG=v2.3.2

# git for the clone; python3-pip so the Batch JobDefinition command
# (`python3 -c "<rl_train_bootstrap.py>"`, which runs under the SYSTEM python3) has a
# place to install boto3. The bootstrap itself is stdlib + boto3 only; it delegates
# all Isaac Lab work to `isaaclab.sh -p` subprocesses, so the system interpreter does
# not need the Kit/torch stack.
# ncurses-term supplies the full terminfo database (incl. the `ansi+tabs` building
# block) so `isaaclab.sh`'s early `tabs 4` succeeds under a no-TTY docker build.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates python3-pip ncurses-term \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --no-cache-dir boto3

WORKDIR /workspace

# Clone Isaac Lab at the pinned tag and symlink the Isaac Sim install where the
# installer expects it (`_isaac_sim`), then install Isaac Lab + all RL frameworks.
#
# ★ Why this is NOT just `./isaaclab.sh -i` (regression fixed 2026-07-29):
# `isaaclab.sh -i` installs the core `source/isaaclab*` extensions through a
# `find -L source -exec bash -c 'install_isaaclab_extension "{}"' \;` loop. Each
# extension installs in a fresh `bash -c` subshell that does NOT inherit the script's
# top-level `set -e`, and `find -exec` discards the child's non-zero exit — so if the
# editable install of `source/isaaclab` (the dist that provides the `isaaclab` module
# train.py imports) fails, the loop keeps going and `-i` STILL returns 0. The build
# then succeeds with `isaaclab` never installed → a first-run `ModuleNotFoundError:
# No module named 'isaaclab'` (silent until this account's first RL run). Only
# isaaclab_rl / isaaclab_mimic install outside the loop under set -e, so a core-package
# failure is invisible at build time. The official docker/Dockerfile.base runs two
# prereqs BEFORE --install that this image was missing: `pip install toml` (every
# extension setup.py imports toml) and `tools/install_deps.py apt source` (apt-level
# build deps). We add those, then reinstall the core extensions EXPLICITLY (so any pip
# failure aborts THIS RUN and prints the real error), then hard-verify the import.
#
# ★★ The swallowed failure, root-caused (2026-07-29 CodeBuild 50bf1aaf log):
# `isaaclab.sh -i` transitively UPGRADES setuptools (70.3.0 -> 83.0.0) while resolving
# the RL frameworks. setuptools >= 81 REMOVED the bundled `pkg_resources`, so the very
# next editable install of `source/isaaclab` (whose legacy setup.py does
# `import pkg_resources`) fails its PEP-517 "get_requires_for_build_wheel" with
# `ModuleNotFoundError: No module named 'pkg_resources'`. In the original image this
# died inside the error-swallowing `find -exec` loop, leaving `isaaclab` uninstalled.
# Fix: after `-i`, re-pin setuptools < 81 (restores pkg_resources) in the kit python,
# then install the core extensions with --no-build-isolation so the wheel build uses
# that pinned setuptools instead of pip provisioning a fresh >=81 in an isolated overlay.
RUN git clone --depth 1 --branch ${ISAACLAB_TAG} https://github.com/isaac-sim/IsaacLab.git \
    && cd IsaacLab \
    && ln -s /isaac-sim _isaac_sim \
    && ./isaaclab.sh -p -m pip install toml \
    && ./isaaclab.sh -p tools/install_deps.py apt source \
    && ./isaaclab.sh -i \
    && ./isaaclab.sh -p -m pip install "setuptools<81" wheel \
    && ./isaaclab.sh -p -m pip install --no-build-isolation \
         -e source/isaaclab -e source/isaaclab_assets -e source/isaaclab_tasks \
    && ./isaaclab.sh -p -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('isaaclab') else 1)"

WORKDIR /workspace/IsaacLab

# The isaac-sim base image sets ENTRYPOINT ["/bin/sh","-c","/isaac-sim/runheadless.sh"]
# (it boots the GUI/WebRTC streaming Kit app). Because that ENTRYPOINT is a fixed
# `sh -c "<script>"`, the Batch JobDefinition command (`python3 -c "<bootstrap>"`)
# does NOT replace it — it is appended as ignored positional args ($0, $1, ...) to
# runheadless.sh, so the rsl_rl bootstrap never runs and the instance idles for hours
# in the streaming server (GPU ~10%, no checkpoints) while burning On-Demand cost.
# Reset ENTRYPOINT to [] so the Batch override executes directly. vla-ft's Pattern A
# uses the same `python3 -c` injection and works because its base image has no such
# ENTRYPOINT.
ENTRYPOINT []

# Default to an interactive shell; the Batch JobDefinition overrides this with
# `python3 -c "<rl_train_bootstrap.py>"`.
CMD ["/bin/bash"]
