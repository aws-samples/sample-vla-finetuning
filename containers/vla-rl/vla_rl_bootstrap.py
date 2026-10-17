"""
VLA RLVR — AWS Batch bootstrap (RL Pattern A, vendor RLinf image).

Runs RLinf (https://github.com/RLinf/RLinf, Apache-2.0) PPO/GRPO post-training of a
pretrained flow-matching VLA (pi0 / pi0.5) in LIBERO, where the reward is task success.
The image is RLinf's own published image, pinned by digest in the CDK stack; RLinf
mounts its code rather than baking it in, so this glue (1) clones RLinf at a pinned
commit onto EFS, (2) downloads the SFT checkpoint from Hugging Face onto EFS (both
cached across jobs), (3) runs train_embodied_agent.py (train) or
evaluations/eval_embodied_agent.py (eval) with Hydra overrides, and
(4) uploads the run's log dir to S3. Stdlib only; S3 upload uses boto3 when the
interpreter has it, else installs it into the EFS cache.

VRL_* contract (env, set at SubmitJob): VRL_MODE (inspect | eval | train),
VRL_CONFIG (train: an examples/embodiment/config name; eval: an
evaluations/<VRL_EVAL_BENCHMARK>/ config name, e.g. libero_spatial_openpi_pi05_eval),
VRL_EVAL_BENCHMARK (default libero), VRL_MODEL_REPO (HF repo id),
VRL_OUTPUT_S3, VRL_OVERRIDES (space-separated Hydra overrides), VRL_CACHE_DIR (EFS),
VRL_RLINF_SHA, VRL_RLINF_ENV (the image's switch_env name), VRL_ROBOT_PLATFORM.

inspect mode prints and uploads the image facts the other modes depend on
(interpreters, switch_env, GPU, disk) — run it first on a new image digest.
"""

import os
import shlex
import shutil
import subprocess
import sys
import time

RLINF_URL = "https://github.com/RLinf/RLinf.git"


def env(name, default=None):
    v = os.environ.get(name, "")
    return v if v != "" else default


def log(msg):
    print(f"[vla-rl] {msg}", flush=True)


def run(cmd, check=True, capture=False):
    log("$ " + (cmd if isinstance(cmd, str) else " ".join(cmd)))
    r = subprocess.run(
        cmd,
        shell=isinstance(cmd, str),
        executable="/bin/bash" if isinstance(cmd, str) else None,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        text=True,
    )
    if check and r.returncode != 0:
        raise SystemExit(f"command failed rc={r.returncode}")
    return r


def in_rlinf_env(script):
    """Run a bash script inside the image's RLinf virtualenv (login shell for switch_env)."""
    venv = env("VRL_RLINF_ENV", "openpi")
    return f"bash -lc {shlex.quote(f'source switch_env {venv} && {script}')}"


def _split_s3(uri):
    rest = uri[len("s3://"):]
    bucket, _, prefix = rest.partition("/")
    return bucket, prefix.rstrip("/")


def _boto3(cache):
    try:
        import boto3  # noqa: F401
    except ImportError:
        target = os.path.join(cache, "pylib")
        if not os.path.isdir(os.path.join(target, "boto3")):
            run([sys.executable, "-m", "pip", "install", "--quiet", "--target", target, "boto3"])
        sys.path.insert(0, target)
    import boto3

    return boto3.client("s3")


def upload_dir(cache, src, output_s3):
    if not output_s3 or not os.path.isdir(src):
        log(f"upload skipped (src={src} output={output_s3})")
        return 0
    s3 = _boto3(cache)
    bucket, prefix = _split_s3(output_s3)
    n = 0
    for root, _dirs, files in os.walk(src):
        for f in files:
            p = os.path.join(root, f)
            key = f"{prefix}/{os.path.relpath(p, src)}".lstrip("/")
            s3.upload_file(p, bucket, key)
            n += 1
    log(f"uploaded {n} files to s3://{bucket}/{prefix}/")
    return n


def inspect(cache, output_s3):
    probes = [
        "uname -m; cat /etc/os-release | head -2",
        "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv",
        "df -h / /mnt/efs | tail -n +1",
        "free -g | head -2; nproc",
        "which -a python python3 git hf huggingface-cli aws 2>&1",
        "python3 -c 'import sys; print(sys.version)'; python3 -c 'import boto3' && echo boto3:yes || echo boto3:no",
        "ls -la / /workspace /opt 2>&1 | head -60",
        "bash -lc 'type switch_env' 2>&1 | head -5; bash -ic 'type switch_env' 2>&1 | head -5; grep -rl switch_env /etc/profile.d /root/.bashrc /etc/bash.bashrc 2>/dev/null",
        in_rlinf_env("which python; python -c 'import sys,torch; print(sys.version, torch.__version__, torch.cuda.is_available())'"),
        in_rlinf_env("python -c 'import libero, rlinf' 2>&1; pip list 2>/dev/null | grep -iE \"^(torch|rlinf|libero|openpi|ray|huggingface|transformers|mujoco|robosuite) \""),
        "env | grep -vE 'AWS_(SECRET|SESSION|ACCESS)' | sort | head -80",
    ]
    out = []
    for p in probes:
        r = run(p, check=False, capture=True)
        out.append(f"### {p}\nrc={r.returncode}\n{r.stdout}\n")
    report = "\n".join(out)
    print(report, flush=True)
    os.makedirs(os.path.join(cache, "inspect"), exist_ok=True)
    path = os.path.join(cache, "inspect", f"inspect-{int(time.time())}.txt")
    with open(path, "w") as fh:
        fh.write(report)
    d = os.path.dirname(path)
    tmp = os.path.join(d, "upload")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    shutil.copy(path, os.path.join(tmp, "inspect.txt"))
    upload_dir(cache, tmp, output_s3)
    return 0


def ensure_rlinf(cache, sha):
    repo = os.path.join(cache, f"RLinf-{sha[:12]}")
    if os.path.isfile(os.path.join(repo, ".vrl-ready")):
        return repo
    shutil.rmtree(repo, ignore_errors=True)
    run(["git", "clone", "--quiet", RLINF_URL, repo])
    run(["git", "-C", repo, "checkout", "--quiet", sha])
    open(os.path.join(repo, ".vrl-ready"), "w").close()
    return repo


def ensure_model(cache, repo_id):
    d = os.path.join(cache, "models", repo_id.replace("/", "__"))
    if os.path.isfile(os.path.join(d, ".vrl-ready")):
        return d
    os.makedirs(d, exist_ok=True)
    py = (
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download(repo_id={repo_id!r}, local_dir={d!r})"
    )
    run(in_rlinf_env(f"python -c {shlex.quote(py)}"))
    open(os.path.join(d, ".vrl-ready"), "w").close()
    return d


def train_or_eval(mode, cache, output_s3):
    repo = ensure_rlinf(cache, env("VRL_RLINF_SHA"))
    model = ensure_model(cache, env("VRL_MODEL_REPO"))
    config = env("VRL_CONFIG")
    run_name = env("AWS_BATCH_JOB_ID", str(int(time.time())))
    log_dir = os.path.join(cache, "runs", run_name)
    os.makedirs(log_dir, exist_ok=True)
    embodied = os.path.join(repo, "examples", "embodiment")
    overrides = [f"runner.logger.log_path={log_dir}", f"rollout.model.model_path={model}"]
    if mode == "eval":
        # RLinf's eval configs put the full model preset on rollout.model and have no
        # actor; the train configs' runner.only_eval=True path reads rollout.model.*
        # (model_type, openpi, ...) that those configs only set on actor.model.
        entry = os.path.join(repo, "evaluations", "eval_embodied_agent.py")
        config_dir = os.path.join(repo, "evaluations", env("VRL_EVAL_BENCHMARK", "libero"))
    else:
        entry = os.path.join(embodied, "train_embodied_agent.py")
        config_dir = os.path.join(embodied, "config")
        overrides.append(f"actor.model.model_path={model}")
    overrides += shlex.split(env("VRL_OVERRIDES", ""))
    exports = (
        f"export EMBODIED_PATH={shlex.quote(embodied)} "
        f"PYTHONPATH={shlex.quote(repo)}:$PYTHONPATH "
        f"ROBOT_PLATFORM={shlex.quote(env('VRL_ROBOT_PLATFORM', 'LIBERO'))} "
        "MUJOCO_GL=${MUJOCO_GL:-egl} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}"
    )
    train = " ".join(
        [
            "python",
            shlex.quote(entry),
            "--config-path",
            shlex.quote(config_dir),
            "--config-name",
            shlex.quote(config),
        ]
        + [shlex.quote(o) for o in overrides]
    )
    r = run(in_rlinf_env(f"cd {shlex.quote(repo)} && {exports} && {train}"), check=False)
    log(f"trainer rc={r.returncode}")
    upload_dir(cache, log_dir, output_s3)
    return r.returncode


def main():
    mode = env("VRL_MODE", "inspect")
    cache = env("VRL_CACHE_DIR", "/mnt/efs/vla-rl")
    output_s3 = env("VRL_OUTPUT_S3")
    if output_s3 and env("AWS_BATCH_JOB_ID"):
        output_s3 = f"{output_s3.rstrip('/')}/{env('VRL_RUN_NAME', env('AWS_BATCH_JOB_ID'))}"
    os.makedirs(cache, exist_ok=True)
    log(f"mode={mode} cache={cache} output={output_s3}")
    if mode == "inspect":
        sys.exit(inspect(cache, output_s3))
    if mode in ("eval", "train"):
        sys.exit(train_or_eval(mode, cache, output_s3))
    raise SystemExit(f"unknown VRL_MODE={mode!r} (inspect | eval | train)")


if __name__ == "__main__":
    main()
