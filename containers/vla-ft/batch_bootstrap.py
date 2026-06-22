"""
Pattern A (AWS Batch) bootstrap for the vla-ft container.

The vla-ft image is BYOC for SageMaker (ships src/train.py as a source_dir, not
baked in). Batch lacks SageMaker's dataset download, hyperparameters, and model
upload, so this glue reproduces those three, then runs the unchanged train.py.
Image + train.py stay byte-identical to Pattern B; this is the only Pattern-A
code, injected via `python3 -c` (read at synth with readFileSync), so the image
is never rebuilt.. Stdlib + boto3 only.

NOTE: this command is `python3 -c "<this source>"` on the JobDefinition, and AWS
counts command + merged container-override env against an 8192B ceiling at submit.
Keep the source small — prose here is real budget. See test/il-pattern-a.test.ts.

Hyperparameters travel as a single S3 JSON file (VLA_FT_HP_S3), staged at the standard
hyperparameters.json path for the UNCHANGED train.py file branch — not per-knob SM_HP_*
override env, so the override no longer scales with hp (the LoRA regex kept tipping 8192).

Contract (env, set by the Batch container overrides at submit):
  VLA_FT_DATASET_S3      synced -> SM_CHANNEL_TRAINING
  VLA_FT_OUTPUT_S3       model dir uploaded here (.../<job>/output/)
  VLA_FT_CODE_S3         the verified trainer -> /opt/ml/code
  VLA_FT_HP_S3           hp JSON -> hyperparameters.json (unset -> SM_HP_* env fallback)
  SM_CHANNEL_TRAINING    dataset lands here (default /opt/ml/input/data/training)
  SM_MODEL_DIR           train.py stages final model here (default /opt/ml/model)
  VLA_FT_CHECKPOINT_DIR  EFS dir, survives Spot reclaim -> resume
  SM_NUM_GPUS            GPU count (train.py picks single vs accelerate)

On Spot reclaim Batch retries the SAME job (same EFS VLA_FT_CHECKPOINT_DIR);
train.py's resolve_resume() resumes from the checkpoint or clears a leftover
output_dir that holds none. Upload mirrors SageMaker's /opt/ml/model layout.
"""

import os
import subprocess
import sys


def _split_s3(uri):
    """s3://bucket/key... -> (bucket, key). Trailing slash preserved on key."""
    assert uri.startswith("s3://"), f"not an s3 uri: {uri}"
    rest = uri[len("s3://"):]
    bucket, _, key = rest.partition("/")
    return bucket, key


def sync_down(s3, dataset_s3, dest_root):
    """Download every object under dataset_s3 into dest_root, preserving the
    relative key path (LeRobot v3 layout: meta/ data/ videos/)."""
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
    print(f"[bootstrap] synced {n} dataset objects -> {dest_root}", flush=True)
    if n == 0:
        print(f"[bootstrap] WARNING: no objects under {dataset_s3}", flush=True)


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
    print(f"[bootstrap] uploaded {n} model files -> s3://{bucket}/{base}", flush=True)


def main():
    import boto3

    dataset_s3 = os.environ["VLA_FT_DATASET_S3"]
    output_s3 = os.environ["VLA_FT_OUTPUT_S3"]
    code_s3 = os.environ["VLA_FT_CODE_S3"]
    hp_s3 = os.environ.get("VLA_FT_HP_S3")  # hp JSON pointer (None -> SM_HP_* env fallback)
    channel = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    ckpt_dir = os.environ.get("VLA_FT_CHECKPOINT_DIR", "/opt/ml/checkpoints")

    s3 = boto3.client("s3")

    print("=" * 60, flush=True)
    print("VLA-FT — AWS Batch bootstrap (Pattern A)", flush=True)
    print(f"  dataset : {dataset_s3}", flush=True)
    print(f"  output  : {output_s3}", flush=True)
    print(f"  code    : {code_s3}", flush=True)
    print(f"  ckpt    : {ckpt_dir} (resume survives Spot reclaim)", flush=True)
    print("=" * 60, flush=True)

    # 1. Fetch the verified, unchanged trainer (single file, self-contained).
    os.makedirs("/opt/ml/code", exist_ok=True)
    cb, ck = _split_s3(code_s3)
    s3.download_file(cb, ck, "/opt/ml/code/train.py")

    # 1b. Fetch the hp JSON -> hyperparameters.json (train.py's file branch). Unset ->
    #     train.py falls back to SM_HP_* env. See module docstring.
    if hp_s3:
        os.makedirs("/opt/ml/input/config", exist_ok=True)
        hb, hk = _split_s3(hp_s3)
        s3.download_file(hb, hk, "/opt/ml/input/config/hyperparameters.json")
        print(f"[bootstrap] hyperparameters <- {hp_s3}", flush=True)

    # 2. Stage the dataset + the checkpoint/model dirs.
    os.makedirs(channel, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    sync_down(s3, dataset_s3, channel)

    # 3. Run the trainer. train.py reads SM_HP_* (env fallback), SM_CHANNEL_TRAINING,
    #    SM_MODEL_DIR, VLA_FT_CHECKPOINT_DIR, SM_NUM_GPUS — all already in os.environ,
    #    so it inherits them. SIGTERM(-15) is train.py's intentional early-stop exit.
    rc = subprocess.run([sys.executable, "/opt/ml/code/train.py"]).returncode
    if rc not in (0, -15, 128 + 15):
        print(f"[bootstrap] train.py failed rc={rc}", flush=True)
        sys.exit(rc)

    # 4. Upload the staged model dir to S3 (SageMaker-equivalent output layout).
    sync_up(s3, model_dir, output_s3)
    print("[bootstrap] done.", flush=True)


if __name__ == "__main__":
    main()
