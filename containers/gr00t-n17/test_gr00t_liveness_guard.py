"""
Unit tests for the GR00T bootstrap's fail-fast liveness guard (gr00t_train_bootstrap.py).

These verify the guard that turns a "booted but never learning" idle run into a fast
non-zero exit, WITHOUT touching the verified launch_finetune command. They use a tiny
fake trainer (a child python that prints scripted lines / sleeps) so no GPU, HF download,
or network is involved — the guard is plain stdlib (subprocess + threading + regex + glob).

The signals are GR00T-specific (verified vs Isaac-GR00T 65cc4a192e6d + transformers
4.57.3), NOT the rsl_rl signals:
  - stdout per-step HF Trainer dict `{'loss': ..., 'learning_rate': ...}` (logging_steps=10)
  - stderr `Starting training...` (GR00T experiment.py, merged via 2>&1)
  - checkpoint-<N>/ dir on disk (first at save_steps)

Run:  python3 test_gr00t_liveness_guard.py     (exit 0 = all pass)
"""

import os
import sys
import tempfile
import textwrap
import time
import unittest

import gr00t_train_bootstrap as boot


def _fake_trainer(body):
    """A train_cmd that runs an inline python program (the 'trainer')."""
    return [sys.executable, "-c", textwrap.dedent(body)]


class LivenessRegexTest(unittest.TestCase):
    def test_matches_hf_trainer_loss_dict(self):
        # The HF default per-step log line (transformers 4.57.3, logging_steps=10).
        line = "{'loss': 1.234, 'grad_norm': 0.5, 'learning_rate': 1e-05, 'epoch': 0.01}"
        self.assertRegex(line, boot.LIVENESS_RE)

    def test_matches_loss_dict_with_train_accuracy(self):
        # GR00T injects 'train_accuracy' into the same dict; loss+learning_rate still present.
        line = "{'loss': 0.42, 'grad_norm': 1.1, 'learning_rate': 9e-05, 'train_accuracy': 0.7, 'epoch': 0.5}"
        self.assertRegex(line, boot.LIVENESS_RE)

    def test_matches_starting_training_marker(self):
        # GR00T's own stderr marker (root logger INFO), seen because we merge 2>&1.
        line = "2026-06-18 02:30:00 - INFO - 🚀 Starting training..."
        self.assertRegex(line, boot.LIVENESS_RE)

    def test_does_not_match_boot_or_load_noise(self):
        for noise in [
            "Loading checkpoint shards: 100%|##########| 2/2",
            "Downloading nvidia/GR00T-N1.7-3B",
            "Some weights of the model checkpoint were not used",
            "***** Running training *****",  # transformers banner is SUPPRESSED in GR00T → must NOT match
            "  Num examples = 40,180",
            "Resolving data files",
            "{'eval_loss': 0.1, 'epoch': 1.0}",  # an eval dict has no learning_rate → not a train step
        ]:
            self.assertNotRegex(noise, boot.LIVENESS_RE, noise)


class DeadlineParseTest(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.pop("GROOT_LIVENESS_DEADLINE_S", None)

    def tearDown(self):
        if self._saved is not None:
            os.environ["GROOT_LIVENESS_DEADLINE_S"] = self._saved
        else:
            os.environ.pop("GROOT_LIVENESS_DEADLINE_S", None)

    def test_default_when_unset(self):
        self.assertEqual(boot._liveness_deadline_s(), boot.DEFAULT_LIVENESS_DEADLINE_S)

    def test_explicit_value(self):
        os.environ["GROOT_LIVENESS_DEADLINE_S"] = "600"
        self.assertEqual(boot._liveness_deadline_s(), 600)

    def test_zero_disables(self):
        os.environ["GROOT_LIVENESS_DEADLINE_S"] = "0"
        self.assertEqual(boot._liveness_deadline_s(), 0)

    def test_bad_value_falls_back_to_default(self):
        os.environ["GROOT_LIVENESS_DEADLINE_S"] = "not-a-number"
        self.assertEqual(boot._liveness_deadline_s(), boot.DEFAULT_LIVENESS_DEADLINE_S)


class HasCheckpointTest(unittest.TestCase):
    def test_false_then_true_when_checkpoint_dir_appears(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(boot._has_checkpoint(d))
            os.makedirs(os.path.join(d, "checkpoint-1000"))
            self.assertTrue(boot._has_checkpoint(d))


class GuardKillsIdleTrainerTest(unittest.TestCase):
    """The core regression: a trainer that boots, prints load noise, then goes silent
    (the idle-run class) must be killed at the deadline with the sentinel RC."""

    def test_idle_trainer_is_killed_with_sentinel_rc(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = _fake_trainer(
                """
                import time
                print("Loading checkpoint shards: 100%", flush=True)
                time.sleep(60)   # idle — never logs a loss dict, no checkpoint dir
                """
            )
            t0 = time.monotonic()
            rc = boot._run_train_with_liveness_guard(
                trainer, groot_dir=d, output_dir=d, deadline_s=2,
            )
            elapsed = time.monotonic() - t0
            self.assertEqual(rc, boot.LIVENESS_KILL_RC)
            self.assertLess(elapsed, 20, f"guard took too long: {elapsed:.1f}s")


class GuardPassesHealthyTrainerTest(unittest.TestCase):
    def test_loss_dict_disarms_guard_and_run_completes(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = _fake_trainer(
                """
                import sys
                print("Loading checkpoint shards: 100%", flush=True)
                print("{'loss': 1.0, 'grad_norm': 0.5, 'learning_rate': 1e-05, 'epoch': 0.0}", flush=True)
                sys.exit(0)
                """
            )
            rc = boot._run_train_with_liveness_guard(
                trainer, groot_dir=d, output_dir=d, deadline_s=10,
            )
            self.assertEqual(rc, 0)

    def test_starting_training_marker_disarms_guard(self):
        with tempfile.TemporaryDirectory() as d:
            # Marker on stderr — proves the 2>&1 merge in the guard surfaces it.
            trainer = _fake_trainer(
                """
                import sys, time
                print("2026-06-18 - INFO - Starting training...", file=sys.stderr, flush=True)
                time.sleep(3)
                sys.exit(0)
                """
            )
            t0 = time.monotonic()
            rc = boot._run_train_with_liveness_guard(
                trainer, groot_dir=d, output_dir=d, deadline_s=2,
            )
            elapsed = time.monotonic() - t0
            self.assertEqual(rc, 0)
            self.assertGreater(elapsed, 2.5, f"guard killed a healthy run: {elapsed:.1f}s")

    def test_checkpoint_dir_disarms_guard_even_without_stdout_marker(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "checkpoint-1000")
            trainer = _fake_trainer(
                f"""
                import time, os
                os.makedirs({ckpt!r})   # checkpoint dir appears immediately
                time.sleep(3)
                """
            )
            t0 = time.monotonic()
            rc = boot._run_train_with_liveness_guard(
                trainer, groot_dir=d, output_dir=d, deadline_s=2,
            )
            elapsed = time.monotonic() - t0
            self.assertEqual(rc, 0)
            self.assertGreater(elapsed, 2.5, f"guard killed a healthy run: {elapsed:.1f}s")


class GuardDisabledTest(unittest.TestCase):
    def test_deadline_zero_never_kills(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = _fake_trainer(
                """
                import time
                print("Loading checkpoint shards", flush=True)
                time.sleep(2)
                """
            )
            rc = boot._run_train_with_liveness_guard(
                trainer, groot_dir=d, output_dir=d, deadline_s=0,
            )
            self.assertEqual(rc, 0)


class GuardPreservesFailureRcTest(unittest.TestCase):
    def test_real_training_failure_propagates_its_rc(self):
        with tempfile.TemporaryDirectory() as d:
            # Trainer that DID start learning, then crashed non-zero. The guard surfaces
            # that rc (not the sentinel, not 0).
            trainer = _fake_trainer(
                """
                import sys
                print("{'loss': 1.0, 'grad_norm': 0.5, 'learning_rate': 1e-05, 'epoch': 0.0}", flush=True)
                sys.exit(7)
                """
            )
            rc = boot._run_train_with_liveness_guard(
                trainer, groot_dir=d, output_dir=d, deadline_s=10,
            )
            self.assertEqual(rc, 7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
