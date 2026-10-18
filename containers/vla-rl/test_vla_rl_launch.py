"""
Unit tests for the RLVR launcher (vla_rl_launch.py) and eval comparison (vla_rl_compare.py).

The launcher tests pin the submitted overrides to the ones the LIBERO-Spatial PoC ran
with, so a refactor cannot silently change what a default `train` / `eval` does. No AWS,
no GPU: build_plan() and the statistics are pure.

Run:  python3 test_vla_rl_launch.py     (exit 0 = all pass)
"""

import os
import tempfile
import unittest

import vla_rl_compare as cmp
import vla_rl_launch as launch


def plan(*argv):
    args = launch.parse_args(list(argv))
    name, container, timeout_s = launch.build_plan(args)
    env = {kv["name"]: kv["value"] for kv in container["environment"]}
    return name, env, container.get("resourceRequirements", []), timeout_s


class TrainPlanTest(unittest.TestCase):
    def test_default_train_is_the_poc_run(self):
        _, env, res, _ = plan("train")
        self.assertEqual(env["VRL_MODE"], "train")
        self.assertEqual(env["VRL_CONFIG"], "libero_spatial_ppo_openpi_pi05")
        self.assertEqual(env["VRL_MODEL_REPO"], "RLinf/RLinf-Pi05-LIBERO-SFT")
        self.assertEqual(env["VRL_OVERRIDES"].split(), [
            "env.train.total_num_envs=16", "env.train.rollout_epoch=4",
            "env.eval.total_num_envs=20", "actor.micro_batch_size=32",
            "actor.global_batch_size=256", "actor.enable_offload=True",
            "rollout.enable_offload=True", "runner.max_epochs=60", "runner.save_interval=20",
        ])
        self.assertEqual(res, [])

    def test_four_gpus_scale_envs_and_batch_not_micro_batch(self):
        _, env, res, _ = plan("train", "--num-gpus", "4")
        ov = env["VRL_OVERRIDES"].split()
        self.assertIn("env.train.total_num_envs=64", ov)
        self.assertIn("actor.global_batch_size=1024", ov)
        self.assertIn("actor.micro_batch_size=32", ov)
        self.assertEqual(res, [{"type": "GPU", "value": "4"}, {"type": "VCPU", "value": "44"},
                               {"type": "MEMORY", "value": "350000"}])

    def test_resume_points_at_the_saved_step(self):
        _, env, _, _ = plan("train", "--steps", "120", "--resume-run", "JOB", "--resume-step", "60")
        self.assertIn("runner.resume_dir=/mnt/efs/vla-rl/runs/JOB/libero_spatial_ppo_openpi_pi05"
                      "/checkpoints/global_step_60", env["VRL_OVERRIDES"].split())
        self.assertIn("runner.max_epochs=120", env["VRL_OVERRIDES"].split())

    def test_timeout_covers_only_the_remaining_steps(self):
        _, _, _, full = plan("train", "--steps", "120")
        _, _, _, rest = plan("train", "--steps", "120", "--resume-run", "J", "--resume-step", "60")
        self.assertGreater(full, rest)
        self.assertGreaterEqual(rest, 60 * launch.MINUTES_PER_STEP_1GPU * 60)

    def test_never_sets_expandable_segments(self):
        # CUDA IPC weight sync then needs pidfd_getfd, which the container denies.
        for argv in (["train"], ["train", "--num-gpus", "4"], ["eval"]):
            _, env, _, _ = plan(*argv)
            self.assertNotIn("PYTORCH_CUDA_ALLOC_CONF", env)
            self.assertNotIn("expandable_segments", " ".join(env.values()))

    def test_user_overrides_go_last(self):
        _, env, _, _ = plan("train", "--override", "actor.optim.lr=1e-5")
        self.assertEqual(env["VRL_OVERRIDES"].split()[-1], "actor.optim.lr=1e-5")


class EvalPlanTest(unittest.TestCase):
    def test_sft_eval_is_200_fixed_episodes(self):
        name, env, _, _ = plan("eval")
        self.assertEqual(env["VRL_CONFIG"], "libero_spatial_openpi_pi05_eval")
        self.assertEqual(env["VRL_EVAL_BENCHMARK"], "libero")
        self.assertEqual(env["VRL_OVERRIDES"].split(),
                         ["env.eval.total_num_envs=20", "env.eval.rollout_epoch=10"])
        self.assertIn("sft", name)

    def test_checkpoint_eval_loads_the_value_head(self):
        name, env, _, _ = plan("eval", "--from-run", "JOB", "--step", "120")
        ov = env["VRL_OVERRIDES"].split()
        self.assertIn("runner.ckpt_path=/mnt/efs/vla-rl/runs/JOB/libero_spatial_ppo_openpi_pi05"
                      "/checkpoints/global_step_120/actor/model_state_dict/full_weights.pt", ov)
        self.assertIn("rollout.model.add_value_head=True", ov)
        self.assertIn("++rollout.model.openpi.value_after_vlm=True", ov)
        self.assertIn("step120", name)
        self.assertEqual(env["VRL_RUN_NAME"], name)


class ArgCheckTest(unittest.TestCase):
    def bad(self, *argv):
        with self.assertRaises(SystemExit):
            launch.parse_args(list(argv))

    def test_rejects_inconsistent_args(self):
        self.bad("train", "--resume-run", "J")                       # no step
        self.bad("train", "--resume-run", "J", "--resume-step", "60")  # steps 60 <= 60
        self.bad("eval", "--from-run", "J")                          # no step
        self.bad("eval", "--episodes", "50")                         # not a multiple of 20
        self.bad("train", "--num-gpus", "0")


class CompareTest(unittest.TestCase):
    def test_parses_the_rlinf_log_line(self):
        line = ("\x1b[36m(EnvGroup(rank=0) pid=1763)\x1b[0m [INFO 04:41:30 EnvGroup-Rank-0]"
                "[utils.py:264] [libero eval] task_id=3, trial_id=7, success=False")
        self.assertEqual(cmp.parse_trials(line), {(3, 7): False})

    def test_reproduces_the_poc_statistics(self):
        # Base 163/200 vs step 120 177/200 with 25 gained / 11 lost discordant pairs.
        self.assertAlmostEqual(cmp.fisher_one_sided(163, 177, 200, 200), 0.034, places=3)
        self.assertAlmostEqual(cmp.mcnemar_one_sided(25, 11), 0.014, places=3)

    def test_no_change_is_not_significant(self):
        self.assertGreater(cmp.fisher_one_sided(100, 100, 200, 200), 0.5)
        self.assertEqual(cmp.mcnemar_one_sided(0, 0), 1.0)

    def test_compare_pairs_by_task_and_trial(self):
        before = {(0, 0): False, (0, 1): True, (1, 0): True, (1, 1): False}
        after = {(0, 0): True, (0, 1): True, (1, 0): False, (1, 1): False, (2, 0): True}
        r = cmp.compare(before, after)
        self.assertEqual((r["n"], r["gained"], r["lost"], r["unpaired"]), (4, 1, 1, 1))
        self.assertEqual(r["per_task"][0], (1, 2, 2))

    def test_reads_a_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.txt")
            with open(p, "w") as f:
                f.write("task_id=0, trial_id=0, success=True\ntask_id=1, trial_id=0, success=False\n")
            self.assertEqual(cmp.load(p), {(0, 0): True, (1, 0): False})


if __name__ == "__main__":
    unittest.main(verbosity=2)
