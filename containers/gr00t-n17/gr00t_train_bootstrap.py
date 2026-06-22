"""GR00T N1.7 fine-tune - AWS Batch bootstrap (IL GR00T Pattern A).

Glue: sync dataset, generate meta stats, run launch_finetune.py under a liveness guard,
upload the merged 3B HF checkpoint. Rationale, GROOT_* contract and verified signals live
in containers/gr00t-n17/README.md and git history (kept out of this payload to stay under
Batch's 8192B ECS override ceiling). Stdlib + boto3 only.
"""

import os
import re
import subprocess
import sys
import glob
import threading
import time


# Verified Isaac-GR00T @ 65cc4a192e6d script paths (relative to the checkout).
FINETUNE_SCRIPT = "gr00t/experiment/launch_finetune.py"
STATS_SCRIPT = "gr00t/data/stats.py"
# Baked universal entry (abs path in the image). When GROOT_ACTION_HORIZON is set it patches
# config.model.action_horizon then runpy's launch_finetune; otherwise it is byte-identical to
# launch_finetune. Needed for the UNITREE_G1 full-body 50-step horizon (no upstream flag).
G1_FINETUNE_SCRIPT = "/workspace/g1_finetune.py"
GRACEFUL_RCS = (0, -15, 128 + 15)  # 0, bare SIGTERM, 128+SIGTERM = graceful (Spot reclaim)
# Training markers (verified vs GR00T 65cc4a192e6d + transformers 4.57.3): per-step HF
# Trainer stdout dict ('loss' + 'learning_rate', logging_steps=10) or stderr "Starting
# training". The "***** Running training *****" banner is suppressed by GR00T, so not matched.
LIVENESS_RE = re.compile(r"'loss':\s*[\d.eE+-]+.*'learning_rate':|Starting training")
LIVENESS_KILL_RC = 42  # sentinel; CDK JobDef maps exit 42 -> EXIT/no-retry (idle is deterministic)
DEFAULT_LIVENESS_DEADLINE_S = 1800  # 30 min: covers cold 3B+6GB boot, far below a multi-hour idle
POLL_INTERVAL_S = 2.0


def _split_s3(uri):
    assert uri.startswith("s3://"), f"not an s3 uri: {uri}"
    rest = uri[len("s3://"):]
    bucket, _, key = rest.partition("/")
    return bucket, key


def sync_down(s3, dataset_s3, dest_root):
    # Download dataset_s3 -> dest_root (WRITABLE; stats.py writes back into meta/).
    bucket, prefix = _split_s3(dataset_s3.rstrip("/") + "/")
    paginator = s3.get_paginator("list_objects_v2")
    n = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue  # directory placeholder
            rel = key[len(prefix):]
            local = os.path.join(dest_root, rel)
            os.makedirs(os.path.dirname(local), exist_ok=True)
            s3.download_file(bucket, key, local)
            n += 1
    print(f"[gr00t-bootstrap] synced {n} dataset objects -> {dest_root}", flush=True)
    if n == 0:
        print(f"[gr00t-bootstrap] WARNING: no objects under {dataset_s3}", flush=True)
    return n


def sync_up(s3, src_root, output_s3):
    # Upload src_root -> output_s3/output/<relpath> (vla-ft/SageMaker output layout).
    bucket, prefix = _split_s3(output_s3.rstrip("/") + "/")
    base = prefix + "output/"
    n = 0
    for root, _dirs, files in os.walk(src_root):
        for f in files:
            local = os.path.join(root, f)
            rel = os.path.relpath(local, src_root)
            s3.upload_file(local, bucket, base + rel.replace(os.sep, "/"))
            n += 1
    print(f"[gr00t-bootstrap] uploaded {n} files -> s3://{bucket}/{base}", flush=True)
    return n


def _heal_modality_annotation(data_dir):
    # GR00T's loader reads each annotation from a parquet column named by its original_key.
    # LeRobot v2.1 stores the task as a `task_index` column, so the annotation entry needs
    # original_key="task_index" (as GR00T's demo_data/cube_to_bowl_5 has); producers that
    # emit an empty {} cause a KeyError. Patch it. See README / recipe doc for details.
    import json
    mpath = os.path.join(data_dir, "meta", "modality.json")
    if not os.path.exists(mpath):
        return
    with open(mpath) as f:
        m = json.load(f)
    ann = m.get("annotation")
    if not isinstance(ann, dict):
        return
    changed = False
    for sub, spec in ann.items():
        if isinstance(spec, dict) and not spec.get("original_key"):
            spec["original_key"] = "task_index"
            changed = True
    if changed:
        with open(mpath, "w") as f:
            json.dump(m, f, indent=2)
        print(f"[gr00t-bootstrap] patched modality.json annotation original_key=task_index",
              flush=True)


def _venv_python():
    cand = "/workspace/.venv/bin/python"  # venv carrying the gr00t stack
    return cand if os.path.exists(cand) else sys.executable


def _build_stats_cmd(py, groot_dir, dataset_dir, embodiment_tag, modality_cfg):
    # Verbatim `python gr00t/data/stats.py --dataset-path <p> --embodiment-tag <t>`;
    # writes meta/stats.json (+ relative_stats.json) into the dataset dir.
    cmd = [py, os.path.join(groot_dir, STATS_SCRIPT),
           "--dataset-path", dataset_dir,
           "--embodiment-tag", embodiment_tag]
    if modality_cfg:
        cmd += ["--modality-config-path", modality_cfg]
    return cmd


def _build_train_cmd(py, groot_dir, dataset_dir, output_dir, embodiment_tag, modality_cfg,
                     num_gpus):
    # launch_finetune.py with frozen-backbone defaults (tune_projector + tune_diffusion,
    # no --tune-* flags). Single GPU = python by path; multi = torchrun --nproc_per_node
    # (trainer does not self-spawn torchrun). Forms from examples/finetune.sh.
    # GROOT_ACTION_HORIZON set -> train via the baked g1_finetune.py wrapper (sets
    # config.model.action_horizon, then runpy launch_finetune); unset -> launch_finetune
    # directly (byte-identical). See g1_finetune.py / README for the why.
    script = (G1_FINETUNE_SCRIPT if os.environ.get("GROOT_ACTION_HORIZON")
              else os.path.join(groot_dir, FINETUNE_SCRIPT))
    if num_gpus > 1:
        cmd = [py, "-m", "torch.distributed.run", "--standalone",
               f"--nproc_per_node={num_gpus}", script]
    else:
        cmd = [py, script]
    cmd += [
        "--base-model-path", os.environ.get("GROOT_BASE_MODEL", "nvidia/GR00T-N1.7-3B"),
        "--dataset-path", dataset_dir,
        "--embodiment-tag", embodiment_tag,
        "--output-dir", output_dir,
        "--num-gpus", str(num_gpus),
        "--max-steps", os.environ.get("GROOT_MAX_STEPS", "2000"),
        "--save-steps", os.environ.get("GROOT_SAVE_STEPS", "1000"),
        "--global-batch-size", os.environ.get("GROOT_GLOBAL_BATCH", "64"),
        "--learning-rate", os.environ.get("GROOT_LEARNING_RATE", "1e-4"),
    ]
    if modality_cfg:
        cmd += ["--modality-config-path", modality_cfg]
    extra = os.environ.get("GROOT_EXTRA_ARGS", "").split()
    cmd += extra
    return cmd


def _has_checkpoint(output_dir):
    # True once a checkpoint-<N>/ dir exists (first at save_steps); buffering-immune corroboration.
    return bool(glob.glob(os.path.join(output_dir, "checkpoint-*")))


def _liveness_deadline_s():
    # Guard grace period in seconds (GROOT_LIVENESS_DEADLINE_S; default 30 min; 0=off).
    raw = os.environ.get("GROOT_LIVENESS_DEADLINE_S")
    if raw is None or raw == "":
        return DEFAULT_LIVENESS_DEADLINE_S
    try:
        return max(0, int(raw))
    except ValueError:
        print(f"[gr00t-bootstrap] WARNING: bad GROOT_LIVENESS_DEADLINE_S={raw!r}; default "
              f"{DEFAULT_LIVENESS_DEADLINE_S}s", flush=True)
        return DEFAULT_LIVENESS_DEADLINE_S


def _run_train_with_liveness_guard(train_cmd, groot_dir, output_dir, deadline_s,
                                   now=time.monotonic, sleep=time.sleep):
    # Run the verified launch_finetune command (byte-identical), re-emitting output (stderr
    # merged into stdout) while a guard watches for the first training sign. If neither
    # marker nor a checkpoint-<N>/ dir appears within deadline_s, the trainer is
    # booted-but-idle: SIGTERM the group, return LIVENESS_KILL_RC. deadline_s<=0 disables.
    import signal

    env = dict(os.environ)
    # Pin CUDA_VISIBLE_DEVICES=0 for single-GPU (avoids HF Trainer auto-DataParallel);
    # torchrun owns devices for multi-GPU, so only default it.
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")

    # New session so os.killpg signals the whole tree (trainer + torchrun children).
    proc = subprocess.Popen(
        train_cmd, cwd=groot_dir, env=env,
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
                print("[gr00t-bootstrap] liveness OK: training marker - disarmed.", flush=True)

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()

    if deadline_s > 0:
        start = now()
        while proc.poll() is None:
            if live.is_set():
                break
            # A checkpoint dir on disk disarms the guard too (marker may be buffered).
            if _has_checkpoint(output_dir):
                live.set()
                print("[gr00t-bootstrap] liveness OK: checkpoint-*/ on disk.", flush=True)
                break
            if now() - start > deadline_s:
                print(f"[gr00t-bootstrap] FATAL: no training progress within {deadline_s}s "
                      f"(no loss/learning_rate, 'Starting training', or checkpoint-*/) - "
                      f"booted but not learning; killing for fail-fast.", flush=True)
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

    dataset_s3 = os.environ["GROOT_DATASET_S3"]
    output_s3 = os.environ["GROOT_OUTPUT_S3"]
    groot_dir = os.environ.get("GROOT_GROOT_DIR", "/workspace")
    data_dir = os.environ.get("GROOT_DATA_DIR", "/opt/ml/input/data/training")
    output_dir = os.environ.get("GROOT_OUTPUT_DIR", "/opt/ml/model")
    embodiment_tag = os.environ.get("GROOT_EMBODIMENT_TAG", "UNITREE_G1")
    modality_cfg = os.environ.get("GROOT_MODALITY_CONFIG") or None
    num_gpus = int(os.environ.get("GROOT_NUM_GPUS", "1"))

    py = _venv_python()

    print("=" * 60, flush=True)
    print("GR00T N1.7 - AWS Batch bootstrap (IL GR00T Pattern A)", flush=True)
    print(f"  dataset={dataset_s3}  output={output_s3}", flush=True)
    print(f"  embodiment={embodiment_tag}  num_gpus={num_gpus}  python={py}", flush=True)
    print("=" * 60, flush=True)

    if not os.path.exists(os.path.join(groot_dir, FINETUNE_SCRIPT)):
        print(f"[gr00t-bootstrap] FATAL: {FINETUNE_SCRIPT} not under {groot_dir} - "
              f"image missing Isaac-GR00T", flush=True)
        sys.exit(2)

    s3 = boto3.client("s3")

    # 1. Sync dataset to a WRITABLE local dir (stats.py mutates meta/ in place).
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    n = sync_down(s3, dataset_s3, data_dir)
    if n == 0:
        print("[gr00t-bootstrap] FATAL: empty dataset", flush=True)
        sys.exit(1)

    # 1b. Heal a common producer omission: annotation original_key -> task_index.
    _heal_modality_annotation(data_dir)

    # 2. Generate stats (MANDATORY: loader asserts meta/stats.json + relative_stats.json).
    stats_cmd = _build_stats_cmd(py, groot_dir, data_dir, embodiment_tag, modality_cfg)
    print(f"[gr00t-bootstrap] stats: {' '.join(stats_cmd)}", flush=True)
    src = subprocess.run(stats_cmd, cwd=groot_dir).returncode
    if src != 0:
        print(f"[gr00t-bootstrap] FATAL: stats.py failed rc={src}", flush=True)
        sys.exit(src)
    meta = os.path.join(data_dir, "meta")
    print(f"[gr00t-bootstrap] stats.json={os.path.exists(os.path.join(meta, 'stats.json'))} "
          f"relative_stats.json={os.path.exists(os.path.join(meta, 'relative_stats.json'))}",
          flush=True)

    # 3. Fine-tune under the fail-fast liveness guard (train command byte-identical).
    train_cmd = _build_train_cmd(py, groot_dir, data_dir, output_dir, embodiment_tag,
                                 modality_cfg, num_gpus)
    deadline_s = _liveness_deadline_s()
    print(f"[gr00t-bootstrap] train: {' '.join(train_cmd)}", flush=True)
    print(f"[gr00t-bootstrap] liveness guard deadline: {deadline_s}s "
          f"({'OFF' if deadline_s == 0 else 'armed'})", flush=True)
    rc = _run_train_with_liveness_guard(train_cmd, groot_dir, output_dir, deadline_s)
    if rc == LIVENESS_KILL_RC:
        print(f"[gr00t-bootstrap] killed by liveness guard rc={rc} (booted, never learned)",
              flush=True)
        sys.exit(rc)
    if rc not in GRACEFUL_RCS:
        print(f"[gr00t-bootstrap] training failed rc={rc}", flush=True)
        sys.exit(rc)

    # 4. Upload checkpoint(s): full merged 3B HF ckpt (checkpoint-<N>/ + experiment_cfg/).
    if not os.listdir(output_dir):
        print(f"[gr00t-bootstrap] WARNING: {output_dir} empty - nothing to upload", flush=True)
        sys.exit(1)
    up = sync_up(s3, output_dir, output_s3)
    if up == 0:
        print("[gr00t-bootstrap] WARNING: nothing uploaded", flush=True)
    print("[gr00t-bootstrap] done.", flush=True)


if __name__ == "__main__":
    main()
