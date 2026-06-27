#!/usr/bin/env python3
"""
Self-contained tests for train.py's Batch/Spot resume reconciliation.

No pytest (same convention as test_vla_ft_decide.py). Run:
  python3 test_train_resume.py
Exits non-zero on the first failure.

Locks the fix for the 79695a9c failure mode: after a Spot reclaim / host-term,
Batch retries the SAME job, so the previous attempt's output_dir survives on EFS.
lerobot's validate() refuses to reuse an existing output_dir unless --resume=true,
so the retry must:
  (A) resume when a checkpoint survived (reclaim after save_freq), and
  (B) clear a leftover output_dir that holds NO checkpoint (reclaim before the first
      save) so a fresh retry doesn't crash on lerobot's "already exists" guard.
Also asserts the verified fresh-start path is a no-op (no dir, no resume, no clear).
"""

import importlib.util
import os
import shutil
import sys
import tempfile

_spec = importlib.util.spec_from_file_location(
    "train", os.path.join(os.path.dirname(__file__), "src", "train.py")
)
train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train)


PASS, FAIL = 0, 0


def check(cond, label):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {label}")
    else:
        FAIL += 1
        print(f"  FAIL {label}")


def _make_ckpt(output_dir, step):
    """Create a lerobot-shaped checkpoint: <output>/checkpoints/<step>/pretrained_model/."""
    pm = os.path.join(output_dir, "checkpoints", step, "pretrained_model")
    os.makedirs(pm, exist_ok=True)
    open(os.path.join(pm, "model.safetensors"), "w").close()
    return pm


def test_case_a_checkpoint_exists_resumes():
    print("case A: checkpoint survived a reclaim -> resume, keep dir:")
    with tempfile.TemporaryDirectory() as root:
        output_dir = os.path.join(root, "run")
        pm = _make_ckpt(output_dir, "0002000")
        resumed = train.resolve_resume(output_dir, forced_resume=False)
        check(resumed == pm, "returns the pretrained_model ckpt path (truthy -> --resume=true)")
        check(os.path.isdir(output_dir), "leaves output_dir intact (lerobot resumes into it)")
        check(
            os.path.isdir(os.path.join(output_dir, "checkpoints", "0002000")),
            "checkpoint not deleted",
        )


def test_case_a_last_symlink_resumes():
    print("case A': 'last' checkpoint present -> resume:")
    with tempfile.TemporaryDirectory() as root:
        output_dir = os.path.join(root, "run")
        pm = os.path.join(output_dir, "checkpoints", "last", "pretrained_model")
        os.makedirs(pm, exist_ok=True)
        resumed = train.resolve_resume(output_dir, forced_resume=False)
        check(resumed == pm, "'last' dir detected as resumable (returns its path)")


def test_case_b_dir_only_clears():
    print("case B: output_dir exists but holds NO checkpoint (the 79695a9c case):")
    with tempfile.TemporaryDirectory() as root:
        output_dir = os.path.join(root, "run")
        # Reclaim before first save: lerobot created the run dir (and maybe an empty
        # checkpoints/) but never wrote a checkpoint.
        os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
        open(os.path.join(output_dir, "train_config.json"), "w").close()
        resumed = train.resolve_resume(output_dir, forced_resume=False)
        check(resumed is None, "returns None (do NOT resume — nothing to resume)")
        check(
            not os.path.isdir(output_dir),
            "clears the leftover output_dir so lerobot starts fresh (no FileExistsError)",
        )


def test_forced_resume_without_ckpt_still_clears():
    print("case B': resume=true requested but no checkpoint -> fresh start, clear dir:")
    with tempfile.TemporaryDirectory() as root:
        output_dir = os.path.join(root, "run")
        os.makedirs(output_dir, exist_ok=True)
        resumed = train.resolve_resume(output_dir, forced_resume=True)
        check(resumed is None, "forced resume can't conjure a checkpoint — returns None")
        check(not os.path.isdir(output_dir), "still clears the empty leftover dir")


def test_fresh_start_is_noop():
    print("verified fresh path: no output_dir at all -> no resume, no error:")
    with tempfile.TemporaryDirectory() as root:
        output_dir = os.path.join(root, "run")  # never created
        resumed = train.resolve_resume(output_dir, forced_resume=False)
        check(resumed is None, "returns None on a clean fresh start")
        check(not os.path.isdir(output_dir), "does not create the dir (lerobot does)")


def test_resume_emits_config_path_file():
    print("lerobot contract: resume must yield --config_path=<ckpt>/train_config.json (FILE):")
    with tempfile.TemporaryDirectory() as root:
        output_dir = os.path.join(root, "run")
        pm = _make_ckpt(output_dir, "0008000")
        resume_ckpt = train.resolve_resume(output_dir, forced_resume=False)
        # The caller appends --config_path=<resume_ckpt>/train_config.json. validate()
        # then derives policy_dir=config_path.parent (=pretrained_model) and
        # checkpoint_path=policy_dir.parent (=checkpoints/0008000). Assert that math.
        config_path = os.path.join(resume_ckpt, "train_config.json")
        check(config_path.endswith("/pretrained_model/train_config.json"),
              "config_path is the train_config.json FILE inside pretrained_model")
        check(os.path.dirname(config_path) == pm,
              "config_path.parent == the pretrained_model dir (policy_dir)")
        check(os.path.basename(os.path.dirname(os.path.dirname(config_path))) == "0008000",
              "config_path.parent.parent == the step checkpoint dir (checkpoint_path)")


def test_hp_s3_roundtrip_matches_env_path():
    print("hp-S3 root fix: launcher JSON (file branch) == old SM_HP_* env branch:")
    import json
    # The hp dict the launchers build (plain string values) — include the heaviest case:
    # the long LoRA vision-tower target regex that used to blow the 8192 override.
    lora_regex = (
        r"(.*\.gemma_expert\..*\.self_attn\.(q|v)_proj|model\.(state_proj|action_in_proj|"
        r"action_out_proj|action_time_mlp_in|action_time_mlp_out)|.*\.vision_tower\..*\."
        r"(q_proj|v_proj|fc1|fc2))"
    )
    hp = {
        "policy": "pi05", "steps": "20000", "batch_size": "16", "save_freq": "2000",
        "dtype": "bfloat16", "gradient_checkpointing": "true",
        "freeze_vision_encoder": "false", "train_expert_only": "false",
        "job_name": "vla_ft_pi05_20260620_151444", "pretrained_path": "lerobot/pi05_base",
        "lora": "true", "lora_r": "32", "lora_alpha": "64",
        "lora_target_modules": lora_regex, "val_episodes": "5", "select_best": "true",
        "early_stop_patience": "10",
    }

    saved_file, saved_env = train.SM_HP_FILE, dict(os.environ)
    try:
        # (1) FILE branch: launcher writes {k: json.dumps(v)} (SageMaker double-encode).
        with tempfile.TemporaryDirectory() as d:
            hp_file = os.path.join(d, "hyperparameters.json")
            with open(hp_file, "w") as f:
                json.dump({k: json.dumps(v) for k, v in hp.items()}, f)
            train.SM_HP_FILE = hp_file
            for k in [e for e in os.environ if e.startswith("SM_HP_")]:
                del os.environ[k]
            hp_from_file = train.load_hyperparameters()

        # (2) ENV branch: the OLD wire format — SM_HP_<NAME>=<plain value>, no file.
        train.SM_HP_FILE = os.path.join(d, "does-not-exist.json")
        for k, v in hp.items():
            os.environ[f"SM_HP_{k.upper()}"] = v
        hp_from_env = train.load_hyperparameters()
    finally:
        train.SM_HP_FILE = saved_file
        os.environ.clear()
        os.environ.update(saved_env)

    check(hp_from_file == hp_from_env,
          "resolved hp dict identical across file (new) and env (old) wire formats")
    check(hp_from_file.get("policy") == "pi05" and hp_from_file.get("steps") == "20000",
          "values decode to plain strings (not '\"pi05\"' / quoted)")
    check(hp_from_file.get("lora_target_modules") == lora_regex,
          "the long LoRA regex round-trips through the JSON file byte-identical")
    # The command train.py builds is a pure function of this hp dict, so identical hp -> identical command.
    check(train.build_command(dict(hp_from_file)) == train.build_command(dict(hp_from_env)),
          "build_command output identical (verified-lock: same training invocation)")


def test_path_alignment_with_stage_final():
    print("path alignment: resume reads the SAME checkpoints/ that _stage_final_model writes:")
    with tempfile.TemporaryDirectory() as root:
        output_dir = os.path.join(root, "run")
        pm = _make_ckpt(output_dir, "0004000")
        # _list_step_checkpoints (used by _stage_final_model) and resolve_resume must
        # agree on the layout: <output_dir>/checkpoints/<step>/pretrained_model/.
        ckpts = train._list_step_checkpoints(os.path.join(output_dir, "checkpoints"))
        check(ckpts and ckpts[-1][1] == pm, "_list_step_checkpoints finds the same ckpt path")
        check(
            train.resolve_resume(output_dir, forced_resume=False) == pm,
            "resolve_resume agrees it is resumable (returns the same ckpt path)",
        )


def test_multinode_fsdp_branch():
    print("Pattern C: multi-node env → accelerate --use_fsdp (single-node byte-identical):")
    # Snapshot + clear the distributed/SM env so the single-node assertion is clean.
    saved = {k: os.environ.get(k) for k in
             ("NNODES", "NODE_RANK", "MASTER_ADDR", "MASTER_PORT", "SM_NUM_GPUS", "SM_HOST_COUNT")}
    try:
        for k in saved:
            os.environ.pop(k, None)
        hp = {"policy": "pi05", "steps": "20000", "full_vlm": "true", "job_name": "x"}

        # (1) Single-node (no NNODES): the verified path — plain `python -m lerobot...`,
        #     NO accelerate, NO FSDP flags. resolve_distributed() is None.
        check(train.resolve_distributed() is None, "no NNODES → resolve_distributed None")
        single = " ".join(train.build_command(dict(hp)))
        check("accelerate launch" not in single and "--use_fsdp" not in single,
              "single-node command is the verified plain launcher (no accelerate/FSDP)")

        # (2) Single-node multi-GPU (SM_NUM_GPUS=4, still 1 node): accelerate --multi_gpu,
        #     NOT --use_fsdp (DDP, the existing Pattern B path — unchanged).
        os.environ["SM_NUM_GPUS"] = "4"
        mg = " ".join(train.build_command(dict(hp)))
        check("--multi_gpu" in mg and "--use_fsdp" not in mg,
              "single-node 4-GPU → accelerate --multi_gpu (DDP), not FSDP")
        os.environ.pop("SM_NUM_GPUS", None)

        # (3) Multi-node (NNODES=2, 8 GPU/node): accelerate --use_fsdp, FULL_SHARD, the
        #     machine_rank/main_process_ip from the env, num_processes = gpus × nodes.
        os.environ.update(NNODES="2", NODE_RANK="1", MASTER_ADDR="10.0.0.5",
                          MASTER_PORT="29500", SM_NUM_GPUS="8")
        d = train.resolve_distributed()
        check(d == {"nnodes": 2, "node_rank": 1, "main_ip": "10.0.0.5", "main_port": 29500},
              "resolve_distributed reads NNODES/NODE_RANK/MASTER_ADDR/PORT")
        mn = " ".join(train.build_command(dict(hp)))
        check("accelerate launch" in mn and "--use_fsdp" in mn, "multi-node → accelerate --use_fsdp")
        check("--num_machines=2" in mn and "--machine_rank=1" in mn,
              "num_machines + machine_rank from the env")
        check("--num_processes=16" in mn, "num_processes = 8 gpu × 2 nodes = 16")
        check("--main_process_ip=10.0.0.5" in mn and "--main_process_port=29500" in mn,
              "rendezvous endpoint wired from MASTER_ADDR/PORT")
        check("--fsdp_sharding_strategy=FULL_SHARD" in mn and "--fsdp_state_dict_type=SHARDED_STATE_DICT" in mn,
              "FULL_SHARD (ZeRO-3) + SHARDED_STATE_DICT (DCP-shaped checkpoint)")

        # (4) Multi-node env but NO MASTER_ADDR → can't rendezvous → falls back to single-node.
        os.environ.pop("MASTER_ADDR", None)
        check(train.resolve_distributed() is None,
              "NNODES set but no MASTER_ADDR → None (safe single-node fallback, no half-formed group)")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def main():
    for t in (
        test_case_a_checkpoint_exists_resumes,
        test_case_a_last_symlink_resumes,
        test_case_b_dir_only_clears,
        test_forced_resume_without_ckpt_still_clears,
        test_fresh_start_is_noop,
        test_path_alignment_with_stage_final,
        test_resume_emits_config_path_file,
        test_hp_s3_roundtrip_matches_env_path,
        test_multinode_fsdp_branch,
    ):
        t()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
