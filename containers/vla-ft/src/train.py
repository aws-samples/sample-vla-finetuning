"""
SageMaker Training Job entrypoint for LeRobot VLA fine-tuning.

Wraps `lerobot-train` so an existing, verified single-EC2 fine-tune (e.g. an
`openarm-lift-pi05` single-GPU run) runs as a *managed* SageMaker Training Job
with no code change to the training itself. SageMaker provides:
  - automatic dataset download (S3 -> /opt/ml/input/data/training/)
  - automatic checkpoint S3 sync + Managed-Spot auto-resume
  - automatic model artifact upload (/opt/ml/model/ -> S3)
  - CloudWatch log delivery

This is **Pattern B** of the platform (SageMaker Training Job + Managed Spot).
The container is backend-agnostic: the same image also runs
under AWS Batch (Pattern A) — see README "backend portability".

LeRobot v0.5.x uses draccus (not Hydra) for config:
  - CLI format: --dotted.path=value
  - Entry point: python -m lerobot.scripts.lerobot_train

SageMaker standard paths (https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-running-container.html):
  /opt/ml/input/data/training/   - dataset (LeRobot v3 format), downloaded by SM
  /opt/ml/input/config/          - hyperparameters.json
  /opt/ml/checkpoints/           - checkpoint dir, continuously synced to checkpoint_s3_uri
  /opt/ml/model/                 - final model artifacts, tar.gz'd + uploaded to S3 on success
  /opt/ml/output/                - other output data (logs, etc.)

Hyperparameters (passed via estimator hyperparameters=, mirror openarm-lift-pi05.yaml):
  policy            pi05 | pi0 | act | groot | smolvla | xvla | ...   (required)
  pretrained_path   e.g. lerobot/pi05_base  (HF id or local; gated -> HF_TOKEN)
  steps             training steps (e.g. 2000 probe / 20000 full)
  batch_size        per-device batch (e.g. 16)
  save_freq         checkpoint cadence in steps (align with steps for a single final ckpt)
  dtype             bfloat16 (default) | float32
  gradient_checkpointing  true | false
  freeze_vision_encoder   true | false
  train_expert_only       true | false   (OOM fallback: freeze VLM, train action expert only)

  LoRA / PEFT (the recommended way to fine-tune a FULL VLA on one GPU — opt-in;
  OFF by default = verified path byte-identical). When lora=true the 2B+ VLM is
  FROZEN and only small low-rank adapters train, so the fp32 Adam optimizer state
  (the OOM-dominating term for full-VLM full-FT — ~37 GB on a 2.3B model) collapses
  to a few MB. This is lerobot-native (cfg.peft, commit d1b1c5c8): no fork, and
  save_checkpoint's PEFT branch writes adapter-only checkpoints safely.
  lora                    true | false. Enable a LoRA fine-tune (emits --peft.method_type=LORA).
                          Mutually exclusive with train_expert_only (both freeze the VLM;
                          LoRA additionally trains adapters, expert-only trains the expert).
  lora_r                  int (lerobot default 16). LoRA rank — higher = more trainable params.
  lora_alpha              int (lerobot default = r). LoRA scaling alpha (scaling = alpha / r).
  lora_target_modules     str. OVERRIDE which modules to adapt (regex / suffix list). OMIT to
                          use the policy's built-in default — pi0/pi05 default to the action
                          expert's q/v projections + the action/state projection MLPs
                          (modeling_pi05.py _get_default_peft_targets), which is the right
                          target for OpenArm-style action fine-tuning.
  NOTE: QLoRA (4-bit base) is NOT supported by lerobot at this commit (no bitsandbytes
  path), so there is no qlora hyperparameter here — use lora (the base is bf16-frozen,
  which already fits one L40S for pi05).

  Early-stop / overfit guard (all opt-in; OFF by default = verified path byte-identical):
  val_episodes            int >0. Hold out the LAST N episodes as a validation set:
                          lerobot trains only on episodes [0 .. total-N-1] (via the
                          verified --dataset.episodes flag). Enables select_best.
  select_best             true | false. After training, score EVERY saved checkpoint on
                          the held-out val episodes and stage the lowest-val-loss one as
                          the final model (early-stopping by checkpoint selection — the
                          overfit guard). Requires val_episodes; lower save_freq to keep
                          several checkpoints. Reuses lerobot's own factories (no fork).
  early_stop_patience     int >0. Converged-cost lever: tail lerobot's train-loss log
                          ('loss:X.XXX', every log_freq steps); if it doesn't improve by
                          early_stop_min_delta for this many consecutive log-points, send
                          the training subprocess SIGTERM (saves GPU time). Independent of
                          val_episodes (this watches TRAIN loss, not val).
  early_stop_min_delta    float >=0 (default 0.0). Min train-loss improvement to reset the
                          patience counter.

  Any other --key=value lerobot-train flag passes through verbatim (see PASSTHROUGH).
"""

import os

# Force unbuffered stdout/stderr — without this Python block-buffers when piped
# and CloudWatch loses the tail if the process crashes before the buffer flushes.
os.environ["PYTHONUNBUFFERED"] = "1"

import json
import subprocess
import sys

# ── SageMaker standard paths (overridable by SM_* env on the training host) ──
SM_CHANNEL_TRAINING = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
SM_MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
SM_OUTPUT_DIR = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
SM_HP_FILE = "/opt/ml/input/config/hyperparameters.json"
# checkpoint_local_path in the estimator maps this dir to checkpoint_s3_uri.
# lerobot writes/reads checkpoints here so Managed-Spot resume finds them.
SM_CHECKPOINT_DIR = os.environ.get("VLA_FT_CHECKPOINT_DIR", "/opt/ml/checkpoints")

# LeRobot policy types we expose (draccus --policy.type values).
POLICY_MAP = {
    "act": "act",
    "diffusion": "diffusion",
    "vqbet": "vqbet",
    "tdmpc": "tdmpc",
    "smolvla": "smolvla",
    "pi0": "pi0",
    "pi05": "pi05",
    "pi0_fast": "pi0_fast",
    "groot": "groot",
    "xvla": "xvla",
}

# Policies whose lerobot config accepts --policy.dtype / --policy.gradient_checkpointing.
# The pi-family (PI0Config / PI0FASTConfig) defines these fields; SmolVLAConfig and the
# classic policies (act/diffusion/...) do NOT, and draccus raises a DecodingError if the
# flags are passed ("The fields `dtype`, `gradient_checkpointing` are not valid for
# SmolVLAConfig"). smolvla_base is already bf16 + 0.45B, so gradient checkpointing is
# unnecessary and dropping the flags is safe. Gate emission on membership here.
PI_FAMILY_POLICIES = {"pi0", "pi05", "pi0_fast"}

# Hyperparameters we consume directly (not forwarded as raw draccus flags).
# Everything else becomes a --key=value passthrough so any lerobot-train flag works.
CONSUMED_KEYS = {
    "policy",
    "pretrained_path",
    "steps",
    "batch_size",
    "save_freq",
    "dtype",
    "gradient_checkpointing",
    "freeze_vision_encoder",
    "train_expert_only",
    "lora",
    "lora_r",
    "lora_alpha",
    "lora_target_modules",
    "job_name",
    "resume",
    "num_gpus",
    "val_episodes",
    "select_best",
    "early_stop_patience",
    "early_stop_min_delta",
}

# SageMaker-injected keys to strip (never valid lerobot flags).
SAGEMAKER_KEYS = {
    "sagemaker_program",
    "sagemaker_submit_directory",
    "sagemaker_region",
    "sagemaker_container_log_level",
    "sagemaker_job_name",
    "sagemaker_estimator_module",
    "sagemaker_estimator_class_name",
}


def load_hyperparameters():
    """Load hyperparameters from SageMaker config or SM_HP_* env fallback."""
    if os.path.exists(SM_HP_FILE):
        with open(SM_HP_FILE) as f:
            # SageMaker stores each value as json.dumps(value), so json.load here
            # gives the values still JSON-encoded (a string '"pi05"', not 'pi05').
            raw = json.load(f)
            return {k: _coerce(v) for k, v in raw.items()}

    # Fallback: SageMaker also sets SM_HP_<name> env vars (these are NOT json-encoded).
    hp = {}
    for key, value in os.environ.items():
        if key.startswith("SM_HP_"):
            hp[key[6:].lower()] = _coerce(value)
    return hp


def _coerce(v):
    """Return a plain CLI-flag string for a SageMaker hyperparameter value.

    SageMaker serializes each hyperparameter with json.dumps, so the value read
    back from hyperparameters.json is double-encoded: the string "pi05" arrives as
    '"pi05"' (quotes embedded) and the int 200 as '200'. Passing '"pi05"' straight
    to draccus yields `--policy.type="pi05"`, which lerobot rejects as an invalid
    choice. So decode one JSON layer when present, then stringify for the CLI.
    (SM_HP_* env vars are not json-encoded — json.loads then fails and we keep v.)"""
    if isinstance(v, str):
        try:
            v = json.loads(v)  # '"pi05"' -> 'pi05'; '200' -> 200; 'true' -> True
        except (ValueError, json.JSONDecodeError):
            return v  # plain string that isn't valid JSON (e.g. a path)
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def resolve_resume(output_dir, forced_resume):
    """Decide whether to append --resume=true, and clean up a stale output_dir.

    On a Spot reclaim / host-termination, Batch retries the SAME job (same job_name
    -> same VLA_FT_CHECKPOINT_DIR on EFS), so the previous attempt's output_dir
    survives. lerobot's validate() REFUSES to reuse an existing output_dir unless
    --resume=true, so EVERY retry after an interruption crashes unless this is wired:

      (A) a checkpoint survived (reclaim AFTER save_freq) -> resume from it.
      (B) output_dir exists but holds NO checkpoint (reclaim BEFORE the first save,
          e.g. step < save_freq -- the openarm 79695a9c case) -> there is nothing to
          resume, yet the leftover dir still trips lerobot's "already exists" guard,
          so clear it for a fresh start.

    NOTE the layout: lerobot writes into output_dir = SM_CHECKPOINT_DIR/run, so its
    checkpoints land at output_dir/checkpoints/<step|last>/pretrained_model/. The
    earlier code probed SM_CHECKPOINT_DIR/checkpoints (one level too high), which is
    why resume never engaged. Resumability is now read from the SAME path
    _stage_final_model uses (_latest_checkpoint), so the two can never disagree.

    lerobot's resume contract (verified vs the pinned d1b1c5c8 source) needs MORE
    than --resume=true: configs/train.py validate() raises
    "A config_path is expected when resuming a run" unless --config_path also points
    at the checkpoint's train_config.json FILE. validate() then derives
    policy_dir = config_path.parent (the pretrained_model dir) and
    checkpoint_path = policy_dir.parent (.../checkpoints/last, where the training
    state lives), so the FILE path is mandatory (the dir would shift both one level).

    Returns the resume checkpoint's pretrained_model dir (truthy) when --resume=true
    should be appended, else None (and clears a stale, checkpoint-less output_dir)."""
    resume_ckpt = _latest_checkpoint(os.path.join(output_dir, "checkpoints"))
    if resume_ckpt:
        print(f"[resume] found checkpoint {resume_ckpt} -> resuming.", flush=True)
        return resume_ckpt
    if forced_resume:
        print(
            f"[resume] resume=true requested but no resumable checkpoint under "
            f"{output_dir} -- starting fresh.",
            flush=True,
        )
    if os.path.isdir(output_dir):
        import shutil
        print(
            f"[resume] clearing leftover output_dir {output_dir} (no resumable "
            f"checkpoint -- reclaim before first save) for a fresh start.",
            flush=True,
        )
        shutil.rmtree(output_dir, ignore_errors=True)
    return None


def _dataset_total_episodes():
    """Total episode count from the LeRobot v3 dataset's meta/info.json.

    Used to compute the train/val split: with val_episodes=N we train lerobot on
    episodes [0 .. total-N-1] and hold out the LAST N for validation. Returns None
    if info.json is missing/unreadable (caller then skips the split)."""
    info_json = os.path.join(SM_CHANNEL_TRAINING, "meta", "info.json")
    try:
        with open(info_json) as f:
            return int(json.load(f).get("total_episodes"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def held_out_split(hp):
    """Return (train_episodes, val_episodes_list) for the val_episodes hyperparameter.

    Splits by holding out the LAST N episodes (deterministic, no shuffling — so the
    same split is reproducible for both the training subprocess and the post-training
    val scoring). Returns (None, None) when val_episodes is unset/invalid or the
    dataset is too small to spare a held-out set."""
    n = hp.get("val_episodes")
    if n is None:
        return None, None
    try:
        n = int(str(n))
    except (TypeError, ValueError):
        return None, None
    total = _dataset_total_episodes()
    if not total or n <= 0 or n >= total:
        if n:
            print(
                f"WARNING: val_episodes={n} invalid for total_episodes={total} — "
                f"skipping held-out split (training on all episodes).",
                flush=True,
            )
        return None, None
    train_eps = list(range(0, total - n))
    val_eps = list(range(total - n, total))
    return train_eps, val_eps


def resolve_num_gpus(hp):
    """Number of GPUs to train across on this node.

    Single-node multi-GPU only (Pattern B). Priority:
      1. explicit `num_gpus` hyperparameter (caller override),
      2. SM_NUM_GPUS (SageMaker sets this from the instance type — e.g. 4 on
         ml.g6e.12xlarge / ml.g5.12xlarge, 1 on ml.g6e.4xlarge),
      3. default 1.
    Returns an int >= 1."""
    if "num_gpus" in hp:
        try:
            n = int(str(hp["num_gpus"]))
            if n >= 1:
                return n
        except (TypeError, ValueError):
            pass
    try:
        return max(1, int(os.environ.get("SM_NUM_GPUS", "1")))
    except (TypeError, ValueError):
        return 1


def resolve_distributed():
    """Multi-node topology from the environment (Pattern C — HyperPod multi-node FSDP2).

    The HyperPod-Slurm launcher (hyperpod_fsdp_launch.sh) sets NNODES / NODE_RANK /
    MASTER_ADDR / MASTER_PORT from Slurm. SageMaker multi-node sets its own (the launch.py
    instance_count>1 path) — read those too as a fallback. Returns a dict
    {nnodes, node_rank, main_ip, main_port} when this is a genuine >1-node run, else None
    (the single-node / single-node-multi-GPU path is unchanged → verified lock intact)."""
    def _int(env, default=None):
        try:
            return int(os.environ[env])
        except (KeyError, TypeError, ValueError):
            return default

    nnodes = _int("NNODES") or _int("SM_HOST_COUNT")
    if not nnodes or nnodes <= 1:
        return None
    node_rank = _int("NODE_RANK", 0)
    main_ip = os.environ.get("MASTER_ADDR") or os.environ.get("SM_MASTER_ADDR")
    main_port = _int("MASTER_PORT", 29500)
    if not main_ip:
        return None  # no rendezvous endpoint → can't form the group; fall back to single-node
    return {"nnodes": nnodes, "node_rank": node_rank, "main_ip": main_ip, "main_port": main_port}


def build_command(hp):
    """Translate consumed hyperparameters + passthroughs into a lerobot-train command.

    Mirrors the verified single-EC2 `openarm-lift-pi05` invocation [5/6].

    Single-GPU (num_gpus==1): `python -m lerobot.scripts.lerobot_train ...` — byte-
    identical to the PASSED smoke; the verified path is left untouched.

    Multi-GPU (num_gpus>1): prepend `accelerate launch --multi_gpu` per the lerobot
    docs (huggingface.co/docs/lerobot/multi_gpu_training). lerobot auto-detects the
    accelerate context (DDP: model replicated per GPU, batch split across GPUs).
    Caveats baked in by the caller, NOT here:
      - effective batch = batch_size x num_gpus (lerobot does NOT auto-scale),
      - lerobot does NOT auto-scale steps/LR — to keep the verified single-GPU
        numerics, launch.py holds effective-batch & steps constant (see launch.py).
    mixed precision is bf16 via accelerate (NOT --policy.use_amp, which lerobot
    ignores under accelerate)."""
    policy = hp.get("policy", "act")
    policy_type = POLICY_MAP.get(policy, policy)

    dataset_root = SM_CHANNEL_TRAINING  # SM downloads the 'training' channel here
    # lerobot validates output_dir doesn't exist (unless resume). SM pre-creates
    # /opt/ml/checkpoints, so train into a subdir of it (so checkpoints sync to S3).
    output_dir = os.path.join(SM_CHECKPOINT_DIR, "run")

    num_gpus = resolve_num_gpus(hp)
    dist = resolve_distributed()
    if dist:
        # Pattern C — multi-node FSDP2 via accelerate (NO lerobot fork: accelerate natively
        # does multi-node + FSDP, and lerobot auto-detects the accelerate context exactly as
        # it does for single-node --multi_gpu). FSDP (not DDP) is what makes a model whose
        # replica exceeds one GPU trainable: it SHARDS params+optimizer+grads across all
        # ranks (vla_ft_decide routes >48 GB replicas here). The EFA fabric (Phase 1 :efa
        # image) carries the inter-node NCCL collectives; DCP (dcp_checkpoint.py) handles the
        # sharded checkpoint. Effective batch = per-device batch × num_gpus × nnodes; the
        # caller holds it constant to preserve the verified optimizer trajectory.
        total_procs = num_gpus * dist["nnodes"]
        launcher = [
            "accelerate", "launch",
            "--use_fsdp",
            f"--num_processes={total_procs}",
            f"--num_machines={dist['nnodes']}",
            f"--machine_rank={dist['node_rank']}",
            f"--main_process_ip={dist['main_ip']}",
            f"--main_process_port={dist['main_port']}",
            "--mixed_precision=bf16",
            # FULL_SHARD = ZeRO-3 (params+grads+optimizer sharded) — the setting that fits a
            # >1-GPU replica; matches the source FSDP2 shard_degree=-1 (full sharding).
            "--fsdp_sharding_strategy=FULL_SHARD",
            "--fsdp_state_dict_type=SHARDED_STATE_DICT",
            "-m", "lerobot.scripts.lerobot_train",
        ]
    elif num_gpus > 1:
        # accelerate launch --multi_gpu --num_processes=N --mixed_precision=bf16 \
        #     -m lerobot.scripts.lerobot_train ...
        launcher = [
            "accelerate", "launch",
            "--multi_gpu",
            f"--num_processes={num_gpus}",
            "--num_machines=1",
            "--mixed_precision=bf16",
            "-m", "lerobot.scripts.lerobot_train",
        ]
    else:
        launcher = [sys.executable, "-m", "lerobot.scripts.lerobot_train"]

    cmd = [
        *launcher,
        "--dataset.repo_id=local",
        f"--dataset.root={dataset_root}",
        f"--policy.type={policy_type}",
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        f"--output_dir={output_dir}",
        f"--job_name={hp.get('job_name', 'vla_ft')}",
        "--wandb.enable=false",
    ]

    # Consumed scalar knobs (only emit when provided).
    if "pretrained_path" in hp:
        cmd.append(f"--policy.pretrained_path={hp['pretrained_path']}")
    # --policy.dtype / --policy.gradient_checkpointing exist only on the pi-family
    # config (see PI_FAMILY_POLICIES). Emitting them for smolvla / act / etc. trips a
    # draccus DecodingError, so skip them for any non-pi policy. The flags are harmless
    # to drop for smolvla_base (already bf16; checkpointing not needed at 0.45B).
    pi_family = policy_type in PI_FAMILY_POLICIES
    if "dtype" in hp and pi_family:
        cmd.append(f"--policy.dtype={hp['dtype']}")
    if "steps" in hp:
        cmd.append(f"--steps={hp['steps']}")
    if "batch_size" in hp:
        cmd.append(f"--batch_size={hp['batch_size']}")
    if "save_freq" in hp:
        cmd.append(f"--save_freq={hp['save_freq']}")
    if str(hp.get("gradient_checkpointing", "")).lower() == "true" and pi_family:
        cmd.append("--policy.gradient_checkpointing=true")
    if "freeze_vision_encoder" in hp:
        cmd.append(f"--policy.freeze_vision_encoder={hp['freeze_vision_encoder']}")
    if str(hp.get("train_expert_only", "")).lower() == "true":
        cmd.append("--policy.train_expert_only=true")

    # LoRA / PEFT: the recommended full-VLM-on-one-GPU path. Enabling cfg.peft makes
    # lerobot freeze the base and train low-rank adapters only — collapsing the fp32
    # Adam state that OOMs a full-VLM full-FT on a 48 GB L40S. lerobot-native (no fork);
    # passing ANY --peft.* subfield instantiates the optional PeftConfig (draccus), and
    # --peft.method_type=LORA is the explicit, source-accurate enable (the field default
    # is "LORA"; we set it so intent is unambiguous in the command + CloudWatch log).
    if str(hp.get("lora", "")).lower() == "true":
        if str(hp.get("train_expert_only", "")).lower() == "true":
            # Both freeze the VLM; running both is contradictory (expert-only trains the
            # expert weights, LoRA trains adapters). Fail fast rather than ship a confused
            # run — the caller should pick one.
            print("ERROR: lora=true and train_expert_only=true are mutually exclusive "
                  "(both freeze the VLM). Choose LoRA (adapters) OR expert-only.", flush=True)
            sys.exit(2)
        cmd.append("--peft.method_type=LORA")
        if "lora_r" in hp:
            cmd.append(f"--peft.r={hp['lora_r']}")
        if "lora_alpha" in hp:
            cmd.append(f"--peft.lora_alpha={hp['lora_alpha']}")
        # target_modules: OMIT to use the policy's built-in default. pi0/pi05 define a
        # default in modeling (gemma_expert q/v + action/state projections), so most runs
        # need no override; non-pi policies without a default would require this.
        if "lora_target_modules" in hp:
            cmd.append(f"--peft.target_modules={hp['lora_target_modules']}")

    # Held-out validation split: train lerobot on the FIRST (total - val_episodes)
    # episodes only, reserving the last val_episodes for post-training val scoring.
    # Uses lerobot's verified --dataset.episodes list flag (format: [0, 1, 2, ...]).
    train_eps, _val_eps = held_out_split(hp)
    if train_eps is not None:
        cmd.append(f"--dataset.episodes=[{', '.join(str(e) for e in train_eps)}]")

    # Managed-Spot / Batch-retry resume: if a checkpoint survived an interruption,
    # continue from it; otherwise clear any leftover output_dir so lerobot's
    # "already exists" guard doesn't crash a fresh retry (the 79695a9c failure mode).
    # lerobot requires BOTH --resume=true AND --config_path=<ckpt>/train_config.json
    # (validate() derives policy_dir/checkpoint_path from that FILE path; verified vs
    # the pinned d1b1c5c8 source). The other --key=value args above stay: lerobot loads
    # the saved config as the base and applies them as overrides, and a Batch resume
    # reuses the SAME job_name -> byte-identical args to the run that wrote the ckpt.
    resume_ckpt = resolve_resume(output_dir, str(hp.get("resume", "")).lower() == "true")
    if resume_ckpt:
        cmd.append("--resume=true")
        cmd.append(f"--config_path={os.path.join(resume_ckpt, 'train_config.json')}")

    # Passthrough: any remaining --key=value the caller set.
    for key, value in hp.items():
        if key in CONSUMED_KEYS or key in SAGEMAKER_KEYS:
            continue
        cmd.append(f"--{key}={value}")

    return cmd


# ── GPU telemetry (the DCGM-pattern signal, on the existing CloudWatch channel) ──────
# Inlined (not a sibling module) because train.py is shipped self-contained: Pattern A's
# bootstrap downloads ONLY this one file from S3, and Pattern B ships src/ as source_dir —
# a sibling import would fail at runtime. Stdlib + nvidia-smi only.

# nvidia-smi fields pulled per sample (bare numbers via nounits), in this order.
_GPU_QUERY_FIELDS = "utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit"
GPU_TELEMETRY_INTERVAL_S = 30


def _gpu_sample_line():
    """One nvidia-smi sample → a parseable '[gpu-telemetry] ...' line, or None if no GPU.

    Aggregates across visible GPUs: util mean/min, summed mem used/total + %, max temp,
    summed power draw/limit. Never raises (telemetry must not disturb training)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={_GPU_QUERY_FIELDS}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    rows = [[c.strip() for c in ln.split(",")] for ln in out.stdout.splitlines() if ln.strip()]
    if not rows:
        return None

    utils, mem_used, mem_total, temps, p_draw, p_limit = [], 0.0, 0.0, [], 0.0, 0.0
    for r in rows:
        def _num(idx):
            try:
                return float(r[idx])
            except (IndexError, ValueError):
                return None
        u = _num(0)
        if u is not None:
            utils.append(u)
        mu, mt, t = _num(1), _num(2), _num(3)
        if mu is not None:
            mem_used += mu
        if mt is not None:
            mem_total += mt
        if t is not None:
            temps.append(t)
        pd, pl = _num(4), _num(5)
        if pd is not None:
            p_draw += pd
        if pl is not None:
            p_limit += pl
    if not utils:
        return None

    mem_pct = (mem_used / mem_total * 100.0) if mem_total else 0.0
    parts = ["[gpu-telemetry]", f"gpus={len(rows)}",
             f"util_mean={sum(utils) / len(utils):.0f}%", f"util_min={min(utils):.0f}%",
             f"mem_used={mem_used:.0f}MiB", f"mem_total={mem_total:.0f}MiB",
             f"mem_pct={mem_pct:.0f}%"]
    if temps:
        parts.append(f"temp_max={max(temps):.0f}C")
    if p_limit:
        parts.append(f"power={p_draw:.0f}/{p_limit:.0f}W")
    return " ".join(parts)


def start_gpu_telemetry():
    """Start a daemon thread printing a GPU-saturation line every ~30 s. Default-on;
    opt-out with VLA_FT_GPU_TELEMETRY=0. No-op (no thread) on a CPU box / when nvidia-smi
    is absent. A priming sample runs inline so the first reading lands immediately and an
    absent GPU short-circuits without spawning a thread."""
    if os.environ.get("VLA_FT_GPU_TELEMETRY", "1").strip().lower() in ("0", "false", "no"):
        return
    try:
        interval = max(5, int(os.environ.get("VLA_FT_GPU_TELEMETRY_INTERVAL_S",
                                             str(GPU_TELEMETRY_INTERVAL_S))))
    except (TypeError, ValueError):
        interval = GPU_TELEMETRY_INTERVAL_S

    try:
        first = _gpu_sample_line()
    except Exception:  # noqa: BLE001 — telemetry must never raise into training
        first = None
    if first is None:
        return  # no GPU / nvidia-smi unavailable → quiet no-op
    print(first, flush=True)
    print(f"[gpu-telemetry] GPU saturation sampling ON (every {interval}s → CloudWatch).",
          flush=True)

    import threading
    import time

    def _loop():
        while True:
            time.sleep(interval)
            try:
                line = _gpu_sample_line()
            except Exception:  # noqa: BLE001
                line = None
            if line is None:
                return
            print(line, flush=True)

    threading.Thread(target=_loop, name="gpu-telemetry", daemon=True).start()


def main():
    print("=" * 60, flush=True)
    print("VLA-FT — SageMaker Training Job (LeRobot / draccus)", flush=True)
    print("=" * 60, flush=True)

    subprocess.run([sys.executable, "--version"], check=False)
    try:
        import lerobot
        print(f"LeRobot version: {lerobot.__version__}", flush=True)
    except ImportError:
        print("ERROR: lerobot not installed in the container image", flush=True)
        sys.exit(1)

    # Keep everything on AWS — no HuggingFace Hub pulls of the dataset.
    # (pretrained_path base weights may still need HF_TOKEN; see launch.py.)
    os.environ.setdefault("HF_HUB_OFFLINE", "0")  # base weights may pull; dataset is local
    # expandable_segments reduces allocator fragmentation (PyTorch OOM hint).
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # GPU telemetry → CloudWatch (the DCGM-pattern signal on this platform's existing log
    # channel). A daemon thread prints a parseable [gpu-telemetry] line every ~30 s so the
    # MCP liveness check can see GPU util/mem, not just the step/loss line — closing the
    # cold-load-warmup false positive and the idle-burn false negative. Default-on, opt-out
    # via VLA_FT_GPU_TELEMETRY=0; best-effort (no GPU / nvidia-smi absent → quiet no-op).
    start_gpu_telemetry()

    hp = load_hyperparameters()
    print(f"Hyperparameters: {json.dumps(hp, indent=2)}", flush=True)
    print(f"Training data:   {SM_CHANNEL_TRAINING}", flush=True)
    print(f"Checkpoint dir:  {SM_CHECKPOINT_DIR} (synced to checkpoint_s3_uri)", flush=True)
    print(f"Model output:    {SM_MODEL_DIR}", flush=True)

    # Sanity: LeRobot v3 dataset must have meta/info.json.
    info_json = os.path.join(SM_CHANNEL_TRAINING, "meta", "info.json")
    if not os.path.exists(info_json):
        print(
            f"WARNING: {info_json} not found — is the 'training' channel a "
            f"LeRobot v3.0 dataset root? Contents: "
            f"{os.listdir(SM_CHANNEL_TRAINING) if os.path.isdir(SM_CHANNEL_TRAINING) else '<missing>'}",
            flush=True,
        )

    cmd = build_command(dict(hp))
    print(f"\nCommand: {' '.join(cmd)}", flush=True)
    print("=" * 60, flush=True)

    returncode = _run_training(cmd, hp)
    # SIGTERM (-15) is how the converged early-stop ends lerobot intentionally; the
    # checkpoint at that point is valid, so treat it as success (not a job failure).
    if returncode not in (0, -15, 128 + 15):
        print(f"Training failed with exit code {returncode}", flush=True)
        sys.exit(returncode)

    # Stage the final model into /opt/ml/model so SageMaker tars + uploads it as the
    # canonical artifact. With select_best, pick the lowest-val-loss checkpoint
    # (overfit guard); otherwise stage lerobot's latest ('last').
    _stage_final_model(hp)
    print(f"\nTraining complete. Model artifacts staged in {SM_MODEL_DIR}", flush=True)


def _run_training(cmd, hp):
    """Run the lerobot training subprocess, returning its exit code.

    Default (no early_stop_patience): stream output straight through — behaviorally
    identical to the verified subprocess.run path.

    early_stop_patience set (converged-cost lever, 목표②): tail stdout line by line,
    re-emitting every line so CloudWatch is unchanged, while parsing lerobot's
    'loss:X.XXX' train-loss (emitted every log_freq steps). If the loss fails to
    improve by min_delta for `patience` consecutive log-points, send SIGTERM so
    lerobot stops after writing its current checkpoint. This watches TRAIN loss, so
    it only saves GPU once training has plateaued — it is NOT the overfit guard
    (that's select_best on val loss)."""
    patience = _as_int(hp.get("early_stop_patience"))
    if not patience or patience <= 0:
        # Verified path: merge stderr into stdout, stream directly to CloudWatch.
        return subprocess.run(cmd, stderr=subprocess.STDOUT).returncode

    min_delta = _as_float(hp.get("early_stop_min_delta")) or 0.0
    print(
        f"[early-stop] converged train-loss watch ON: patience={patience} "
        f"log-points, min_delta={min_delta}",
        flush=True,
    )

    import re
    import signal

    loss_re = re.compile(r"\bloss:([0-9]+\.?[0-9]*)")
    best = float("inf")
    stale = 0
    stopping = False

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        sys.stdout.write(line)  # re-emit verbatim — CloudWatch sees the full log
        sys.stdout.flush()
        if stopping:
            continue
        m = loss_re.search(line)
        if not m:
            continue
        try:
            loss = float(m.group(1))
        except ValueError:
            continue
        if loss < best - min_delta:
            best = loss
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(
                    f"[early-stop] train loss plateaued at {best:.3f} for {stale} "
                    f"log-points — sending SIGTERM to stop after current checkpoint.",
                    flush=True,
                )
                proc.send_signal(signal.SIGTERM)
                stopping = True
    return proc.wait()


def _as_int(v):
    try:
        return int(str(v))
    except (TypeError, ValueError):
        return None


def _as_float(v):
    try:
        return float(str(v))
    except (TypeError, ValueError):
        return None


def _list_step_checkpoints(run_dir):
    """All saved step-checkpoint 'pretrained_model' dirs under run_dir, step-ascending.
    lerobot layout: <run>/checkpoints/<zero-padded step>/pretrained_model/."""
    if not os.path.isdir(run_dir):
        return []
    steps = sorted((d for d in os.listdir(run_dir) if d.isdigit()), key=int)
    out = []
    for s in steps:
        pm = os.path.join(run_dir, s, "pretrained_model")
        if os.path.isdir(pm):
            out.append((int(s), pm))
    return out


def _latest_checkpoint(run_dir):
    """lerobot's final checkpoint ('last' symlink, else highest step)."""
    last = os.path.join(run_dir, "last", "pretrained_model")
    if os.path.isdir(last):
        return last
    ckpts = _list_step_checkpoints(run_dir)
    return ckpts[-1][1] if ckpts else None


def _stage_final_model(hp):
    """Copy the chosen checkpoint to /opt/ml/model/ for SageMaker artifact upload.

    Default: stage lerobot's latest checkpoint (verified behavior).
    select_best + val_episodes (목표① overfit guard): score every saved checkpoint on
    the held-out val episodes and stage the lowest-val-loss one (early stopping by
    checkpoint selection). On ANY failure in the optional selection, fall back to the
    latest checkpoint — training already succeeded, so selection must never fail the job."""
    import shutil

    run_dir = os.path.join(SM_CHECKPOINT_DIR, "run", "checkpoints")

    src = None
    if str(hp.get("select_best", "")).lower() == "true":
        _, val_eps = held_out_split(hp)
        if not val_eps:
            print(
                "WARNING: select_best=true but no valid held-out split "
                "(set val_episodes) — staging latest checkpoint instead.",
                flush=True,
            )
        else:
            try:
                src = _select_best_checkpoint(run_dir, val_eps)
            except Exception as e:  # noqa: BLE001 — selection is best-effort
                import traceback
                print(f"WARNING: best-checkpoint selection failed ({e}); "
                      f"staging latest checkpoint instead.", flush=True)
                traceback.print_exc()

    if not src:
        src = _latest_checkpoint(run_dir)

    if not src:
        print(f"WARNING: no final checkpoint found under {run_dir}", flush=True)
        return

    dest = os.path.join(SM_MODEL_DIR, "pretrained_model")
    os.makedirs(SM_MODEL_DIR, exist_ok=True)
    if os.path.exists(dest):
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    n = sum(len(files) for _, _, files in os.walk(dest))
    print(f"Staged final checkpoint {src} -> {dest} ({n} files)", flush=True)


def _select_best_checkpoint(run_dir, val_eps):
    """Return the pretrained_model dir of the lowest held-out-val-loss checkpoint.

    Reuses lerobot's OWN factories and the exact config it wrote to each checkpoint
    (train_config.json) — no dependency/version hand-assembly, no lerobot fork. The
    only new logic is a no-grad forward loop that mirrors lerobot_train.py's batch
    handling (uint8->float, preprocessor, policy.forward) verbatim. Runs single-process
    on cuda:0 (the accelerate subprocess has already exited).

    Returns None if there are no checkpoints to score (caller falls back to latest)."""
    ckpts = _list_step_checkpoints(run_dir)
    if not ckpts:
        print(f"WARNING: no step-checkpoints under {run_dir} to score.", flush=True)
        return None

    import torch
    from pathlib import Path
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets import make_dataset
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.utils.collate import lerobot_collate_fn

    print(
        f"[select_best] scoring {len(ckpts)} checkpoint(s) on held-out val episodes "
        f"{val_eps[0]}..{val_eps[-1]} ({len(val_eps)} eps)",
        flush=True,
    )

    # Reload the EXACT config lerobot wrote (deps/policy/dataset shape all baked in),
    # then redirect it at the held-out episodes and the local dataset root.
    ref_ckpt = ckpts[-1][1]
    cfg = TrainPipelineConfig.from_pretrained(ref_ckpt)
    cfg.dataset.root = SM_CHANNEL_TRAINING
    cfg.dataset.episodes = list(val_eps)
    cfg.env = None  # ensure no sim-env path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build the val dataset + dataloader ONCE (stats/shape are identical across
    # checkpoints of one run); only the policy weights change per checkpoint.
    dataset = make_dataset(cfg)
    collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None
    val_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=collate_fn,
    )
    cam_keys = dataset.meta.camera_keys

    def val_loss_for(pm_dir):
        # Load policy weights from this checkpoint (from_pretrained calls .eval()).
        cfg.policy.pretrained_path = Path(pm_dir)
        policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
        # Reload the processor from this checkpoint. We override ONLY device — NOT the
        # normalizer stats. lerobot uses the checkpoint's SAVED stats when no stats
        # override is given (normalize_processor.py docstring), so each checkpoint is
        # scored with the exact normalization it was trained/saved with. Do not add a
        # 'normalizer_processor': {'stats': ...} override here — it would silently
        # replace the saved stats and corrupt the val-loss comparison.
        preprocessor, _post = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=pm_dir,
            preprocessor_overrides={"device_processor": {"device": device.type}},
        )
        policy.eval()
        total, nb = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                # Mirror lerobot_train.py:460-464 batch handling verbatim.
                for ck in cam_keys:
                    if ck in batch and batch[ck].dtype == torch.uint8:
                        batch[ck] = batch[ck].to(dtype=torch.float32) / 255.0
                batch = preprocessor(batch)
                loss, _out = policy.forward(batch)
                total += float(loss.item())
                nb += 1
        del policy
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return total / nb if nb else float("inf")

    best_dir, best_loss, best_step = None, float("inf"), None
    for step, pm in ckpts:
        loss = val_loss_for(pm)
        flag = ""
        if loss < best_loss:
            best_dir, best_loss, best_step = pm, loss, step
            flag = "  <- best"
        print(f"[select_best] step {step:>7}: val_loss={loss:.4f}{flag}", flush=True)

    print(
        f"[select_best] selected step {best_step} (val_loss={best_loss:.4f}) "
        f"out of {len(ckpts)} checkpoints.",
        flush=True,
    )
    return best_dir


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001 — top-level guard for CloudWatch trace
        print(f"\n{'=' * 60}", flush=True)
        print(f"FATAL: unhandled exception in train.py: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)
