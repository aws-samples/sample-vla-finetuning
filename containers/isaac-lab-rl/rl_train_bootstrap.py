"""
Isaac Lab RL — AWS Batch bootstrap (RL Pattern A).

RL-axis counterpart of vla-ft's batch_bootstrap.py. The trainer (Isaac Lab v2.3.2) is
baked into the image (FROM isaac-sim:4.5.0), so this glue downloads no code: it (1)
symlinks Isaac Lab logs/ onto EFS so a Spot reclaim + retry resumes, (2) runs the
verified rsl_rl headless PPO under a fail-fast liveness guard, (3) runs rsl_rl play.py
(the only built-in ONNX export), (4) uploads the run dir to S3. rsl_rl not skrl:
H1-v0 registers both but only rsl_rl ships ONNX export. Commands are the v2.3.2 forms,
verified verbatim. Stdlib + boto3 only.

RL_* contract (env, set by Batch container overrides at SubmitJob; flags documented in
rl_launch.py): RL_TASK, RL_OUTPUT_S3, RL_EXPERIMENT_NAME, RL_CHECKPOINT_DIR (EFS; logs/
symlinked here → resume), RL_NUM_ENVS, RL_MAX_ITERATIONS, RL_SEED, RL_NUM_GPUS (>1 →
torchrun --distributed), RL_PLAY_ENVS, RL_EXTRA_OVERRIDES (Hydra), RL_ISAACLAB_DIR,
RL_SKIP_EXPORT, RL_LIVENESS_DEADLINE_S (guard grace seconds; default 1200; 0=off).

Liveness guard (the 5.5 h idle run): the first real RL submit sat "RUNNING" 5.5 h with
no checkpoints, stdout frozen at boot — the base ENTRYPOINT had swallowed this bootstrap
(fixed with ENTRYPOINT []); Batch saw "RUNNING" and never killed it, ~$17 burned for
zero training. Status != learning. The guard makes "booted but not learning" a fast
non-zero exit with no human watching.
"""

import os
import re
import shutil
import subprocess
import sys
import glob
import threading
import time


# --- verified Isaac Lab v2.3.2 rsl_rl script paths (relative to the checkout) ---
TRAIN_SCRIPT = "scripts/reinforcement_learning/rsl_rl/train.py"
PLAY_SCRIPT = "scripts/reinforcement_learning/rsl_rl/play.py"
# SIGTERM exit codes treated as graceful (Spot reclaim): bare -15 or 128+15.
GRACEFUL_RCS = (0, -15, 128 + 15)

# Liveness marker: rsl_rl OnPolicyRunner.log() prints "Learning iteration {it}/{tot}"
# once per completed PPO iteration (verified vs rsl-rl-lib 3.1.2, pinned by Isaac Lab
# v2.3.2). Clean "training started" signal — never during Isaac Sim/Kit boot. NOT
# "Mean reward:" (conditional, may skip iter 0). ANSI/centering precede it → search.
LIVENESS_RE = re.compile(r"Learning iteration\s+\d+/\d+")

# Sentinel RC for a guard kill — distinct from child SIGTERM (-15/143 = graceful Spot
# reclaim) so the CDK JobDef maps exit 42 → EXIT/no-retry (idle is deterministic).
LIVENESS_KILL_RC = 42

# Guard grace period (s) + poll cadence. 20 min ≈ 4x a healthy boot, ~16x < the 5.5 h
# burn; cleared the instant training starts so a fast boot is never penalized.
DEFAULT_LIVENESS_DEADLINE_S = 1200
POLL_INTERVAL_S = 1.0


def _split_s3(uri):
    """s3://bucket/key... -> (bucket, key)."""
    assert uri.startswith("s3://"), f"not an s3 uri: {uri}"
    rest = uri[len("s3://"):]
    bucket, _, key = rest.partition("/")
    return bucket, key


def sync_up(s3, src_root, output_s3):
    """Upload everything under src_root to output_s3/output/<relpath>."""
    bucket, prefix = _split_s3(output_s3.rstrip("/") + "/")
    base = prefix + "output/"
    n = 0
    for root, _dirs, files in os.walk(src_root):
        for f in files:
            local = os.path.join(root, f)
            rel = os.path.relpath(local, src_root)
            s3.upload_file(local, bucket, base + rel.replace(os.sep, "/"))
            n += 1
    print(f"[rl-bootstrap] uploaded {n} files -> s3://{bucket}/{base}", flush=True)
    return n


def _wire_logs_to_efs(isaaclab_dir, ckpt_dir):
    """Point <isaaclab>/logs at the EFS checkpoint dir so the rsl_rl run folder
    survives a Spot reclaim. If logs already exists and is not our symlink, move
    its contents under EFS once, then replace it with the symlink."""
    if not ckpt_dir:
        return
    os.makedirs(ckpt_dir, exist_ok=True)
    logs = os.path.join(isaaclab_dir, "logs")
    if os.path.islink(logs):
        return  # already wired (retry)
    if os.path.isdir(logs):
        # First run on a fresh image: relocate any seed content, then symlink.
        for entry in os.listdir(logs):
            shutil.move(os.path.join(logs, entry), os.path.join(ckpt_dir, entry))
        os.rmdir(logs)
    os.symlink(ckpt_dir, logs)
    print(f"[rl-bootstrap] logs/ -> {ckpt_dir} (EFS; resume survives Spot reclaim)", flush=True)


def _latest_run_dir(isaaclab_dir, experiment):
    """Newest run folder under logs/rsl_rl/<experiment>/, or None."""
    root = os.path.join(isaaclab_dir, "logs", "rsl_rl", experiment)
    runs = [d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)]
    return max(runs, key=os.path.getmtime) if runs else None


def _latest_checkpoint(run_dir):
    """Highest-numbered model_<N>.pt in run_dir (rsl_rl checkpoint naming), or None."""
    ckpts = glob.glob(os.path.join(run_dir, "model_*.pt"))
    if not ckpts:
        return None

    def _step(p):
        stem = os.path.basename(p)[len("model_"):-len(".pt")]
        return int(stem) if stem.isdigit() else -1

    return max(ckpts, key=_step)


def _build_train_cmd(isaaclab_sh, num_gpus):
    """rsl_rl headless PPO. Single-GPU = isaaclab.sh -p train.py; multi-GPU = the
    documented torchrun form wrapped through isaaclab.sh -p so the Kit env is set."""
    task = os.environ.get("RL_TASK", "Isaac-Velocity-Rough-H1-v0")
    cmd = [isaaclab_sh, "-p"]
    if num_gpus > 1:
        cmd += ["-m", "torch.distributed.run", "--nnodes=1", f"--nproc_per_node={num_gpus}"]
    cmd += [TRAIN_SCRIPT, "--task", task, "--headless"]
    if num_gpus > 1:
        cmd += ["--distributed"]
    if os.environ.get("RL_NUM_ENVS"):
        cmd += ["--num_envs", os.environ["RL_NUM_ENVS"]]
    if os.environ.get("RL_MAX_ITERATIONS"):
        cmd += ["--max_iterations", os.environ["RL_MAX_ITERATIONS"]]
    if os.environ.get("RL_SEED"):
        cmd += ["--seed", os.environ["RL_SEED"]]
    # Resume if an EFS run folder already exists (Spot retry). rsl_rl auto-picks the
    # latest run when --load_run is omitted.
    experiment = os.environ.get("RL_EXPERIMENT_NAME", "h1_rough")
    if _latest_run_dir(os.path.dirname(isaaclab_sh), experiment):
        cmd += ["--resume"]
    # Reward / task overrides (the RL "intent" knobs) appended bare (Hydra style).
    extra = os.environ.get("RL_EXTRA_OVERRIDES", "").split()
    cmd += extra
    return cmd


def _build_play_cmd(isaaclab_sh, checkpoint):
    """rsl_rl play.py — runs a short rollout and exports policy.pt + policy.onnx
    into <checkpoint_dir>/exported/ (the only documented built-in ONNX export)."""
    task = os.environ.get("RL_TASK", "Isaac-Velocity-Rough-H1-v0")
    play_envs = os.environ.get("RL_PLAY_ENVS", "32")
    return [
        isaaclab_sh, "-p", PLAY_SCRIPT,
        "--task", task, "--headless",
        "--num_envs", play_envs,
        "--checkpoint", checkpoint,
    ]


def _has_checkpoint(isaaclab_dir, experiment):
    """True once rsl_rl wrote any model_<N>.pt under logs/rsl_rl/<experiment>/.
    rsl-rl-lib 3.1.2 saves model_0.pt at the start of iteration 0 on a fresh run
    (it % save_interval == 0 with it=0), so this corroborates the stdout marker:
    a checkpoint on disk proves the loop was reached, immune to stdout buffering."""
    root = os.path.join(isaaclab_dir, "logs", "rsl_rl", experiment)
    return bool(glob.glob(os.path.join(root, "*", "model_*.pt")))


def _liveness_deadline_s():
    """Guard grace period in seconds (RL_LIVENESS_DEADLINE_S; default 20 min; 0=off)."""
    raw = os.environ.get("RL_LIVENESS_DEADLINE_S")
    if raw is None or raw == "":
        return DEFAULT_LIVENESS_DEADLINE_S
    try:
        return max(0, int(raw))
    except ValueError:
        print(f"[rl-bootstrap] WARNING: bad RL_LIVENESS_DEADLINE_S={raw!r}; using default "
              f"{DEFAULT_LIVENESS_DEADLINE_S}s", flush=True)
        return DEFAULT_LIVENESS_DEADLINE_S


def _run_train_with_liveness_guard(train_cmd, isaaclab_dir, experiment, deadline_s,
                                   now=time.monotonic, sleep=time.sleep):
    """Run the verified rsl_rl train command (byte-identical), re-emitting output so
    CloudWatch is unchanged, while a guard watches for the first sign of training. If
    neither a "Learning iteration N/M" line nor a model_*.pt checkpoint appears within
    deadline_s, the trainer is booted-but-idle: SIGTERM the process group and return
    LIVENESS_KILL_RC. deadline_s<=0 disables it. now/sleep injectable for tests."""
    import signal

    # New session so os.killpg signals the whole tree (isaaclab.sh shell, python
    # trainer, torchrun per-GPU children) at once.
    proc = subprocess.Popen(
        train_cmd, cwd=isaaclab_dir,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, start_new_session=True,
    )

    live = threading.Event()

    def _reader():
        # Must keep draining or the child blocks on a full pipe; re-emit every line.
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if not live.is_set() and LIVENESS_RE.search(line):
                live.set()
                print("[rl-bootstrap] liveness OK: 'Learning iteration' — disarmed.", flush=True)

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()

    if deadline_s > 0:
        start = now()
        while proc.poll() is None:
            if live.is_set():
                break
            # Checkpoint on disk disarms the guard too (marker may be buffered).
            if _has_checkpoint(isaaclab_dir, experiment):
                live.set()
                print("[rl-bootstrap] liveness OK: model_*.pt on disk — disarmed.", flush=True)
                break
            if now() - start > deadline_s:
                print(f"[rl-bootstrap] FATAL: no training progress within {deadline_s}s "
                      f"(no 'Learning iteration', no model_*.pt) — booted but not "
                      f"learning; killing so Batch fails fast.", flush=True)
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                reader.join(timeout=5)
                if proc.stdout:
                    proc.stdout.close()
                return LIVENESS_KILL_RC
            sleep(POLL_INTERVAL_S)

    proc.wait()
    reader.join(timeout=5)
    if proc.stdout:
        proc.stdout.close()
    return proc.returncode


def main():
    import boto3

    isaaclab_dir = os.environ.get("RL_ISAACLAB_DIR", "/workspace/IsaacLab")
    isaaclab_sh = os.path.join(isaaclab_dir, "isaaclab.sh")
    output_s3 = os.environ["RL_OUTPUT_S3"]
    experiment = os.environ.get("RL_EXPERIMENT_NAME", "h1_rough")
    ckpt_dir = os.environ.get("RL_CHECKPOINT_DIR")
    num_gpus = int(os.environ.get("RL_NUM_GPUS", "1"))

    print("=" * 60, flush=True)
    print("Isaac Lab RL — AWS Batch bootstrap (RL Pattern A)", flush=True)
    print(f"  task       : {os.environ.get('RL_TASK', 'Isaac-Velocity-Rough-H1-v0')}", flush=True)
    print(f"  experiment : {experiment}  (logs/rsl_rl/{experiment}/)", flush=True)
    print(f"  output     : {output_s3}", flush=True)
    print(f"  ckpt(EFS)  : {ckpt_dir}", flush=True)
    print(f"  num_gpus   : {num_gpus}", flush=True)
    print("=" * 60, flush=True)

    if not os.path.exists(isaaclab_sh):
        print(f"[rl-bootstrap] FATAL: {isaaclab_sh} not found — image missing Isaac Lab", flush=True)
        sys.exit(2)

    s3 = boto3.client("s3")

    # 1. Wire logs/ onto EFS (resume across Spot reclaim).
    _wire_logs_to_efs(isaaclab_dir, ckpt_dir)

    # 2. Train (rsl_rl headless PPO) under the fail-fast liveness guard so a
    #    booted-but-idle trainer (the 5.5 h ENTRYPOINT-swallow class) dies in minutes,
    #    not hours. The train command is byte-identical.
    train_cmd = _build_train_cmd(isaaclab_sh, num_gpus)
    deadline_s = _liveness_deadline_s()
    print(f"[rl-bootstrap] train: {' '.join(train_cmd)}", flush=True)
    print(f"[rl-bootstrap] liveness guard deadline: {deadline_s}s "
          f"({'OFF' if deadline_s == 0 else 'armed'})", flush=True)
    rc = _run_train_with_liveness_guard(train_cmd, isaaclab_dir, experiment, deadline_s)
    if rc == LIVENESS_KILL_RC:
        print(f"[rl-bootstrap] killed by liveness guard rc={rc} (booted, never learned)", flush=True)
        sys.exit(rc)
    if rc not in GRACEFUL_RCS:
        print(f"[rl-bootstrap] training failed rc={rc}", flush=True)
        sys.exit(rc)

    # 3. Locate the trained checkpoint.
    run_dir = _latest_run_dir(isaaclab_dir, experiment)
    if not run_dir:
        print(f"[rl-bootstrap] WARNING: no run dir under logs/rsl_rl/{experiment} — nothing to export/upload", flush=True)
        sys.exit(1)
    checkpoint = _latest_checkpoint(run_dir)
    print(f"[rl-bootstrap] run dir: {run_dir}  checkpoint: {checkpoint}", flush=True)

    # 4. Export policy.pt + policy.onnx via the rsl_rl play.py (best-effort: a failed
    #    export must not discard the trained checkpoint we already have).
    if checkpoint and os.environ.get("RL_SKIP_EXPORT", "false").lower() != "true":
        play_cmd = _build_play_cmd(isaaclab_sh, checkpoint)
        print(f"[rl-bootstrap] export: {' '.join(play_cmd)}", flush=True)
        prc = subprocess.run(play_cmd, cwd=isaaclab_dir).returncode
        if prc not in GRACEFUL_RCS:
            print(f"[rl-bootstrap] WARNING: ONNX export (play.py) rc={prc} — uploading checkpoints without exported/", flush=True)
        else:
            exported = os.path.join(os.path.dirname(checkpoint), "exported")
            onnx = os.path.join(exported, "policy.onnx")
            print(f"[rl-bootstrap] exported: {onnx} (exists={os.path.exists(onnx)})", flush=True)

    # 5. Upload the whole run dir (checkpoints + exported/) to S3.
    n = sync_up(s3, run_dir, output_s3)
    if n == 0:
        print("[rl-bootstrap] WARNING: nothing uploaded", flush=True)
    print("[rl-bootstrap] done.", flush=True)


if __name__ == "__main__":
    main()
