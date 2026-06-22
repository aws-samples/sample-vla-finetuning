#!/usr/bin/env python3
"""
Self-contained tests for the vla-ft MCP pure cores (status / checkpoint / registry).

No pytest, no network — same convention as containers/vla-ft/test_vla_ft_decide.py:
standalone asserts over the pure functions, using log lines + config dicts captured from
REAL jobs/checkpoints (2026-06-20). The boto3 layer (vla_aws) and the tool wiring are
verified separately against live AWS (see the session log / verify_live.py). Run:

    python3 test_mcp.py     # exits non-zero on the first failure
"""

import sys

import vla_status as st
import vla_checkpoint as ck
import vla_registry as rg


PASS, FAIL = 0, 0


def check(cond, label):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {label}")
    else:
        FAIL += 1
        print(f"  FAIL {label}")


# ── engine classification ────────────────────────────────────────────────────────────
print("engine classification:")
check(st.classify_engine("vla-ft-pi05-20260620-122229") == "lerobot", "vla-ft- -> lerobot")
check(st.classify_engine("gr00t-n17-20260619-103523") == "gr00t", "gr00t-n17- -> gr00t")
check(st.classify_engine("isaac-rl-20260617-181938") == "rl", "isaac-rl- -> rl")
check(st.classify_engine("") == "lerobot", "empty -> lerobot (default)")


# ── lerobot progress parsing (REAL line from job vla-ft-pi05-20260620-122229) ────────
print("lerobot progress parsing:")
LEROBOT_LINE = ("INFO 2026-06-20 03:39:27 ot_train.py:489 step:200 smpl:3K ep:17 "
                "epch:0.34 loss:0.710 grdn:0.700 lr:3.8e-06 updt_s:2.326 data_s:0.009")
p = st.parse_progress(LEROBOT_LINE, "lerobot")
check(p.latest_step == 200, f"step parsed = 200 (got {p.latest_step})")
check(abs(p.latest_loss - 0.710) < 1e-9, f"loss parsed = 0.710 (got {p.latest_loss})")
check(p.saw_progress_line, "saw_progress_line True on a real step line")

# multiple lines → newest step wins
two = LEROBOT_LINE + "\n" + LEROBOT_LINE.replace("step:200", "step:400").replace("loss:0.710", "loss:0.369")
p2 = st.parse_progress(two, "lerobot")
check(p2.latest_step == 400 and abs(p2.latest_loss - 0.369) < 1e-9,
      f"newest of two lines wins (step {p2.latest_step}, loss {p2.latest_loss})")

# K-suffix expansion
check(st._expand_k("3K") == 3000 and st._expand_k("200") == 200, "K-suffix expand")

# boot noise → no progress
boot = "Loading model from: lerobot/pi05_base\n✓ Loaded state dict\nWrapped pi05 with PEFT (LoraConfig)"
pb = st.parse_progress(boot, "lerobot")
check(not pb.saw_progress_line and pb.latest_step is None, "boot lines -> no progress")


# ── GR00T progress parsing (HF Trainer dict) ─────────────────────────────────────────
print("GR00T progress parsing:")
GROOT_LINE = "{'loss': 0.0761, 'grad_norm': 0.5, 'learning_rate': 1e-05, 'epoch': 0.92}"
g = st.parse_progress(GROOT_LINE, "gr00t")
check(abs(g.latest_loss - 0.0761) < 1e-9, f"loss = 0.0761 (got {g.latest_loss})")
check(abs(g.latest_epoch - 0.92) < 1e-9, f"epoch = 0.92 (got {g.latest_epoch})")
check(g.saw_progress_line, "GR00T loss+lr dict -> progress")
# eval-only dict (no learning_rate) must NOT count as progress
ev = st.parse_progress("{'eval_loss': 0.5, 'epoch': 1.0}", "gr00t")
check(not ev.saw_progress_line, "eval_loss dict (no lr) -> NOT progress")


# ── RL progress parsing (rsl_rl) ─────────────────────────────────────────────────────
print("RL progress parsing:")
r = st.parse_progress("Learning iteration 137/1500\nMean reward: 12.4", "rl")
check(r.latest_step == 137 and r.total_steps == 1500, f"iter 137/1500 (got {r.latest_step}/{r.total_steps})")
check(abs(r.latest_reward - 12.4) < 1e-9, f"reward 12.4 (got {r.latest_reward})")


# ── status verdict logic (the RUNNING != learning gate) ──────────────────────────────
print("status verdict:")
v_run_learn = st.build_verdict(
    job_name="vla-ft-pi05-x", job_id="id1", batch_status="RUNNING", elapsed_s=300,
    progress=st.parse_progress(LEROBOT_LINE, "lerobot"))
check(v_run_learn.learning and v_run_learn.liveness_ok, "RUNNING + step line -> liveness_ok")

v_run_idle = st.build_verdict(
    job_name="isaac-rl-x", job_id="id2", batch_status="RUNNING", elapsed_s=2000,
    progress=st.parse_progress("Isaac Sim Full Streaming App is loaded", "rl"),
    liveness_deadline_s=1200)
check(not v_run_idle.learning and not v_run_idle.liveness_ok,
      "RUNNING + NO progress -> NOT liveness_ok (the idle-burn signal)")
check(any("stall" in n.lower() for n in v_run_idle.notes),
      "idle past deadline -> probable-stall note")

v_done = st.build_verdict(job_name="j", job_id="i", batch_status="SUCCEEDED",
                          elapsed_s=6389, progress=st.Progress(engine="gr00t"))
check(v_done.liveness_ok, "SUCCEEDED -> liveness_ok (finished)")
v_fail = st.build_verdict(job_name="j", job_id="i", batch_status="FAILED",
                          elapsed_s=160, progress=st.Progress(engine="lerobot"))
check(not v_fail.liveness_ok, "FAILED -> not ok")
v_pend = st.build_verdict(job_name="j", job_id="i", batch_status="RUNNABLE",
                          elapsed_s=None, progress=st.Progress(engine="lerobot"))
check(v_pend.liveness_ok and not v_pend.learning, "RUNNABLE -> ok-but-waiting, not learning")
# checkpoint corroborates learning even without a parsed line
v_ckpt = st.build_verdict(job_name="j", job_id="i", batch_status="RUNNING", elapsed_s=500,
                          progress=st.Progress(engine="lerobot"), has_checkpoint=True)
check(v_ckpt.learning, "RUNNING + checkpoint present -> learning")


# ── checkpoint kind detection ────────────────────────────────────────────────────────
print("checkpoint kind detection:")
GROOT_KEYS = ["config.json", "model-00001-of-00003.safetensors",
              "model.safetensors.index.json", "experiment_cfg/final_model_config.json",
              "processor/processor_config.json", "checkpoint-5000/config.json"]
check(ck.detect_kind(GROOT_KEYS) == "gr00t", "sharded model-*.safetensors -> gr00t")
LORA_KEYS = ["pretrained_model/adapter_config.json", "pretrained_model/adapter_model.safetensors",
             "pretrained_model/config.json", "pretrained_model/train_config.json"]
check(ck.detect_kind(LORA_KEYS) == "lerobot", "pretrained_model/adapter -> lerobot")
check(ck.is_adapter_only(LORA_KEYS), "adapter + no model.safetensors -> adapter_only")
MERGED_KEYS = ["pretrained_model/model.safetensors", "pretrained_model/config.json"]
check(not ck.is_adapter_only(MERGED_KEYS), "model.safetensors present -> NOT adapter_only")


# ── GR00T horizon consistency gate (REAL keys from gr00t-n17-20260619-103523) ────────
print("GR00T horizon gate (real config keys):")
# Real model config.json (action_horizon=40) and processor (max_action_horizon=50, G1 delta=50).
MODEL_CFG_40 = {"action_horizon": 40, "model_type": "Gr00tN1d7"}
PROC_CFG_50 = {"processor_kwargs": {
    "max_action_horizon": 50,
    "modality_configs": {
        "unitree_g1_full_body_with_waist_height_nav_cmd": {
            "action": {"delta_indices": list(range(50))}},
        "oxe_droid_relative_eef_relative_joint": {
            "action": {"delta_indices": list(range(40))}},
    }}}
rep_bad = ck.inspect_gr00t("s3://b/p/", GROOT_KEYS, model_config=MODEL_CFG_40,
                           processor_config=PROC_CFG_50,
                           embodiment_tag="unitree_g1_full_body_with_waist_height_nav_cmd")
check(rep_bad.model_action_horizon == 40, f"model horizon 40 (got {rep_bad.model_action_horizon})")
check(rep_bad.processor_action_horizon == 50, f"processor horizon 50 (got {rep_bad.processor_action_horizon})")
check(rep_bad.consistency == "MISMATCH", "40 vs 50 -> MISMATCH (the rollout-step-0 bug)")
check(any("MISMATCH" in n for n in rep_bad.notes), "mismatch note explains the broadcast crash")
check(rep_bad.evidence.get("action_delta_indices_len_per_embodiment", {})
      .get("unitree_g1_full_body_with_waist_height_nav_cmd") == 50, "G1 delta len 50 in evidence")

# the h50fix: model corrected to 50 -> OK
MODEL_CFG_50 = {"action_horizon": 50, "model_type": "Gr00tN1d7"}
rep_ok = ck.inspect_gr00t("s3://b/p-h50fix/", GROOT_KEYS, model_config=MODEL_CFG_50,
                          processor_config=PROC_CFG_50)
check(rep_ok.consistency == "OK", "50 vs 50 (h50fix) -> OK")
check(rep_ok.adapter_only is False, "GR00T is never adapter-only")


# ── lerobot adapter report (the OpenArm LoRA contract) ───────────────────────────────
print("lerobot adapter report:")
ADAPTER_CFG = {"base_model_name_or_path": "lerobot/pi05_base", "r": 32, "lora_alpha": 64}
rep_lora = ck.inspect_lerobot("s3://b/p/", LORA_KEYS, adapter_config=ADAPTER_CFG)
check(rep_lora.adapter_only and rep_lora.base_model == "lerobot/pi05_base",
      "adapter-only + base lerobot/pi05_base")
check(any("base" in n.lower() and "reachable" in n.lower() for n in rep_lora.notes),
      "adapter note warns base must be reachable")
rep_merged = ck.inspect_lerobot("s3://b/p2/", MERGED_KEYS)
check(rep_merged.adapter_only is False and "self-contained" in rep_merged.loadable_hint,
      "merged -> self-contained hint")


# ── registry (S3 manifest, pure merge/query) ─────────────────────────────────────────
print("registry:")
m = rg.empty_manifest()
e1 = rg.CheckpointEntry(id="vla-ft-pi05-A", output_s3="s3://b/A/output/", engine="lerobot",
                        model="pi05", intent="il", ft_mode="lora", adapter_only=True,
                        consistency="OK", registered_at="2026-06-20T00:00:00Z")
m = rg.upsert(m, e1)
check(len(m["checkpoints"]) == 1, "upsert adds one")
# update in place by id (re-validated)
e1b = rg.CheckpointEntry(id="vla-ft-pi05-A", output_s3="s3://b/A/output/", engine="lerobot",
                         model="pi05", consistency="OK", notes=["re-validated"])
m = rg.upsert(m, e1b)
check(len(m["checkpoints"]) == 1 and m["checkpoints"][0]["notes"] == ["re-validated"],
      "upsert by id updates in place (no dup)")
check(m["checkpoints"][0]["registered_at"] == "2026-06-20T00:00:00Z",
      "original registered_at preserved on update")
e2 = rg.CheckpointEntry(id="gr00t-G1", output_s3="s3://b/G/output/", engine="gr00t",
                        model="groot", intent="il", consistency="MISMATCH",
                        registered_at="2026-06-21T00:00:00Z")
m = rg.upsert(m, e2)
q_gr = rg.query(m, engine="gr00t")
check(len(q_gr) == 1 and q_gr[0]["id"] == "gr00t-G1", "query by engine=gr00t")
q_ok = rg.query(m, consistency="OK")
check(len(q_ok) == 1 and q_ok[0]["id"] == "vla-ft-pi05-A", "query by consistency=OK")
check(rg.query(m)[0]["id"] == "gr00t-G1", "newest-first by registered_at")


print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
