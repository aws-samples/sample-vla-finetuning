"""
Unit tests for the RL bootstrap's fail-fast liveness guard (rl_train_bootstrap.py).

These verify the guard that turns the 5.5 h "booted but never learning" idle-run into a
fast non-zero exit, WITHOUT touching the verified train command. They use a tiny fake
trainer (a child python that prints scripted lines / sleeps) so no GPU, Isaac Sim, or
network is involved — the guard is plain stdlib (subprocess + threading + regex + glob).

Run:  python3 test_rl_liveness_guard.py     (exit 0 = all pass)
"""

import os
import sys
import tempfile
import textwrap
import time
import unittest

import rl_train_bootstrap as boot


def _fake_trainer(body):
    """A train_cmd that runs an inline python program (the 'trainer')."""
    return [sys.executable, "-c", textwrap.dedent(body)]


class LivenessRegexTest(unittest.TestCase):
    def test_matches_rsl_rl_iteration_banner(self):
        # The real rsl-rl-lib 3.1.2 banner is wrapped in ANSI + centering spaces.
        line = " \033[1m Learning iteration 0/1500 \033[0m \n"
        self.assertRegex(line, boot.LIVENESS_RE)

    def test_matches_mid_run_iteration(self):
        self.assertRegex("Learning iteration 137/1500", boot.LIVENESS_RE)

    def test_does_not_match_boot_noise(self):
        for noise in [
            "Isaac Sim Full Streaming App is loaded",
            "Simulation App Startup Complete",
            "[ext: omni.kit.window.title-1.1.4] startup",
            "Mean reward: 1.23",  # conditional line — deliberately NOT our signal
            "loading iteration data",
        ]:
            self.assertNotRegex(noise, boot.LIVENESS_RE, noise)


class DeadlineParseTest(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.pop("RL_LIVENESS_DEADLINE_S", None)

    def tearDown(self):
        if self._saved is not None:
            os.environ["RL_LIVENESS_DEADLINE_S"] = self._saved
        else:
            os.environ.pop("RL_LIVENESS_DEADLINE_S", None)

    def test_default_when_unset(self):
        self.assertEqual(boot._liveness_deadline_s(), boot.DEFAULT_LIVENESS_DEADLINE_S)

    def test_explicit_value(self):
        os.environ["RL_LIVENESS_DEADLINE_S"] = "300"
        self.assertEqual(boot._liveness_deadline_s(), 300)

    def test_zero_disables(self):
        os.environ["RL_LIVENESS_DEADLINE_S"] = "0"
        self.assertEqual(boot._liveness_deadline_s(), 0)

    def test_bad_value_falls_back_to_default(self):
        os.environ["RL_LIVENESS_DEADLINE_S"] = "not-a-number"
        self.assertEqual(boot._liveness_deadline_s(), boot.DEFAULT_LIVENESS_DEADLINE_S)


class HasCheckpointTest(unittest.TestCase):
    def test_false_then_true_when_model_pt_appears(self):
        with tempfile.TemporaryDirectory() as d:
            exp = "h1_rough"
            self.assertFalse(boot._has_checkpoint(d, exp))
            run = os.path.join(d, "logs", "rsl_rl", exp, "2026-01-01_00-00-00")
            os.makedirs(run)
            self.assertFalse(boot._has_checkpoint(d, exp))  # dir but no ckpt yet
            open(os.path.join(run, "model_0.pt"), "w").close()
            self.assertTrue(boot._has_checkpoint(d, exp))


class GuardKillsIdleTrainerTest(unittest.TestCase):
    """The core regression: a trainer that boots, prints boot noise, then goes silent
    (exactly the ENTRYPOINT-swallow failure) must be killed at the deadline with the
    sentinel RC — not blocked on forever."""

    def test_idle_trainer_is_killed_with_sentinel_rc(self):
        with tempfile.TemporaryDirectory() as d:
            # Trainer that prints boot lines then sleeps "forever" without ever training.
            trainer = _fake_trainer(
                """
                import time, sys
                print("Isaac Sim Full Streaming App is loaded", flush=True)
                time.sleep(60)   # idle — never prints 'Learning iteration', no ckpt
                """
            )
            t0 = time.monotonic()
            rc = boot._run_train_with_liveness_guard(
                trainer, isaaclab_dir=d, experiment="h1_rough", deadline_s=2,
            )
            elapsed = time.monotonic() - t0
            self.assertEqual(rc, boot.LIVENESS_KILL_RC)
            # Must have killed near the deadline, not waited out the 60 s sleep.
            self.assertLess(elapsed, 20, f"guard took too long: {elapsed:.1f}s")


class GuardPassesHealthyTrainerTest(unittest.TestCase):
    def test_training_line_disarms_guard_and_run_completes(self):
        with tempfile.TemporaryDirectory() as d:
            # Healthy trainer: boots, prints a Learning iteration line, exits 0.
            trainer = _fake_trainer(
                """
                import sys
                print("Isaac Sim Full Streaming App is loaded", flush=True)
                print("\\033[1m Learning iteration 0/1500 \\033[0m", flush=True)
                print("Learning iteration 1/1500", flush=True)
                sys.exit(0)
                """
            )
            rc = boot._run_train_with_liveness_guard(
                trainer, isaaclab_dir=d, experiment="h1_rough", deadline_s=10,
            )
            self.assertEqual(rc, 0)

    def test_checkpoint_on_disk_disarms_guard_even_without_stdout_marker(self):
        with tempfile.TemporaryDirectory() as d:
            run = os.path.join(d, "logs", "rsl_rl", "h1_rough", "run0")
            os.makedirs(run)
            # Trainer prints NO training marker, but writes a checkpoint then keeps
            # running briefly. The filesystem corroboration must disarm the guard so it
            # is NOT killed despite a short deadline.
            ckpt = os.path.join(run, "model_0.pt")
            trainer = _fake_trainer(
                f"""
                import time
                open({ckpt!r}, "w").close()   # checkpoint appears immediately
                time.sleep(3)                 # then "trains" a little and exits
                """
            )
            t0 = time.monotonic()
            rc = boot._run_train_with_liveness_guard(
                trainer, isaaclab_dir=d, experiment="h1_rough", deadline_s=2,
            )
            elapsed = time.monotonic() - t0
            # NOT killed (rc != sentinel); the child exited 0 on its own after sleeping.
            self.assertEqual(rc, 0)
            # And it ran past the 2 s deadline because the guard was disarmed by the ckpt.
            self.assertGreater(elapsed, 2.5, f"guard killed a healthy run: {elapsed:.1f}s")


class GuardDisabledTest(unittest.TestCase):
    def test_deadline_zero_never_kills(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = _fake_trainer(
                """
                import time
                print("Isaac Sim Full Streaming App is loaded", flush=True)
                time.sleep(2)   # idle but short; with guard off it must run to completion
                """
            )
            rc = boot._run_train_with_liveness_guard(
                trainer, isaaclab_dir=d, experiment="h1_rough", deadline_s=0,
            )
            self.assertEqual(rc, 0)


class GuardPreservesFailureRcTest(unittest.TestCase):
    def test_real_training_failure_propagates_its_rc(self):
        with tempfile.TemporaryDirectory() as d:
            # Trainer that DID start learning, then crashed with a non-zero rc. The guard
            # must surface that rc (not the sentinel, not 0).
            trainer = _fake_trainer(
                """
                import sys
                print("Learning iteration 0/1500", flush=True)
                sys.exit(7)
                """
            )
            rc = boot._run_train_with_liveness_guard(
                trainer, isaaclab_dir=d, experiment="h1_rough", deadline_s=10,
            )
            self.assertEqual(rc, 7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
