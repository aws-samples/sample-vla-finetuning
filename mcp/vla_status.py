#!/usr/bin/env python3
"""
vla-ft MCP — job-status enrichment, the pure parsing core.

`get_job_status` exists to answer ONE question a requesting session keeps asking and
the platform kept answering by hand: **"RUNNING ≠ learning — is it actually training?"**
Diagnosing the RL 5.5 h idle burn took 10+ SSM calls. This module turns the raw signals
(Batch state + the tail of CloudWatch logs + checkpoint listing) into a single verdict.

It is **pure**: it takes already-fetched strings/lists and returns a dict. All boto3 /
CloudWatch / S3 I/O lives in vla_aws.py; the server wires them. That split keeps the
parsing logic unit-testable against captured log lines (which is exactly how the
regexes below were verified — against the real running job's logs).

Three engines, three different progress dialects (all verified against real logs):

  - **lerobot (pi05 / vla-ft)** — a `key:value` token line emitted every log_freq
    steps, e.g.
      `INFO ... ot_train.py:489 step:200 smpl:3K ep:17 epch:0.34 loss:0.710 grdn:0.700 lr:3.8e-06`
    (verified against job vla-ft-pi05-20260620-122229). step / loss are anchored
    independently so a partial or reordered line still parses.
  - **GR00T N1.7 (HF Trainer)** — a Python dict per log line, e.g.
      `{'loss': 0.42, 'grad_norm': 1.1, 'learning_rate': 9e-05, 'epoch': 0.5}`
    (logging_steps=10). We require BOTH 'loss' and 'learning_rate' so an eval-loss dict
    or the transformers banner never reads as progress.
  - **RL (rsl_rl, isaac-lab-rl)** — `Learning iteration N/M`, once per PPO iteration.

`classify_engine` picks the dialect from the job name prefix (the launchers' own
convention: vla-ft- / gr00t-n17- / isaac-rl-), so the caller need not say which.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ── engine detection (by the launcher's job-name prefix) ─────────────────────────────
def classify_engine(job_name: str) -> str:
    """Map a job name to its log dialect: 'lerobot' | 'gr00t' | 'rl'.

    The prefixes are the launchers' own convention (batch_launch.py / gr00t_launch.py /
    rl_launch.py): vla-ft- → lerobot, gr00t-n17- → gr00t, isaac-rl- → rl. Unknown →
    'lerobot' (the most common case here), so enrichment degrades gracefully."""
    n = (job_name or "").lower()
    if n.startswith("gr00t"):
        return "gr00t"
    if n.startswith("isaac-rl") or n.startswith("rl-"):
        return "rl"
    return "lerobot"


# ── progress regexes (verified against real logs) ────────────────────────────────────
# lerobot: anchor each field independently. step/smpl/ep may carry a 'K' suffix; loss is
# a plain float; lr is scientific. (train.py's own early-stop watcher uses the same loss
# token: re.compile(r"\bloss:([0-9]+\.?[0-9]*)").)
_LEROBOT_STEP = re.compile(r"\bstep:(\d+\.?\d*K?)", re.IGNORECASE)
_LEROBOT_LOSS = re.compile(r"\bloss:([0-9]+\.?[0-9eE+-]*)")

# GR00T HF Trainer dict line — require both keys so eval-only dicts / banners don't match.
_GROOT_LIVE = re.compile(r"'loss':\s*[\d.eE+-]+.*'learning_rate':")
_GROOT_LOSS = re.compile(r"'loss':\s*([\d.eE+-]+)")
_GROOT_EPOCH = re.compile(r"'epoch':\s*([\d.]+)")

# RL rsl_rl iteration marker (may be preceded by ANSI/centering codes → not anchored).
_RL_ITER = re.compile(r"Learning iteration\s+(\d+)/(\d+)")
_RL_REWARD = re.compile(r"Mean reward:\s*([\-\d.]+)")


def _expand_k(tok: str) -> int:
    """Expand a lerobot 'K'-suffixed integer token ('3K' → 3000, '200' → 200)."""
    tok = tok.strip()
    if tok.upper().endswith("K"):
        return int(round(float(tok[:-1]) * 1000))
    return int(round(float(tok)))


@dataclass
class Progress:
    """The latest learning signal scraped from a log tail. None = not seen yet."""
    engine: str
    latest_step: int | None = None        # lerobot step / RL iteration (GR00T: None, uses epoch)
    total_steps: int | None = None        # RL only (N/M); None otherwise
    latest_loss: float | None = None
    latest_epoch: float | None = None     # GR00T (no integer step counter in the dict)
    latest_reward: float | None = None    # RL only
    saw_progress_line: bool = False       # a real training line (not just boot) was seen


def parse_progress(log_tail: str, engine: str) -> Progress:
    """Scan a chunk of CloudWatch log text and return the LATEST progress signal.

    Pure. `log_tail` is whatever the caller fetched (newest events). We scan all lines and
    keep the last match of each field, so the most recent step/loss wins regardless of how
    many lines were passed."""
    p = Progress(engine=engine)
    if not log_tail:
        return p

    if engine == "gr00t":
        for m in _GROOT_LOSS.finditer(log_tail):
            p.latest_loss = float(m.group(1))
            p.saw_progress_line = True
        ep = None
        for m in _GROOT_EPOCH.finditer(log_tail):
            ep = m.group(1)
        if ep is not None:
            p.latest_epoch = float(ep)
        # Only count it as progress if a real loss+lr dict line appeared (not just a banner).
        if not _GROOT_LIVE.search(log_tail):
            p.saw_progress_line = p.saw_progress_line and False
        else:
            p.saw_progress_line = True
        return p

    if engine == "rl":
        last = None
        for m in _RL_ITER.finditer(log_tail):
            last = m
        if last:
            p.latest_step = int(last.group(1))
            p.total_steps = int(last.group(2))
            p.saw_progress_line = True
        r = None
        for m in _RL_REWARD.finditer(log_tail):
            r = m.group(1)
        if r is not None:
            p.latest_reward = float(r)
        return p

    # lerobot (default)
    s = None
    for m in _LEROBOT_STEP.finditer(log_tail):
        s = m.group(1)
    if s is not None:
        p.latest_step = _expand_k(s)
        p.saw_progress_line = True
    ll = None
    for m in _LEROBOT_LOSS.finditer(log_tail):
        ll = m.group(1)
    if ll is not None:
        try:
            p.latest_loss = float(ll)
        except ValueError:
            pass
    return p


@dataclass
class StatusVerdict:
    """The enriched answer to 'is it really training?'. `liveness_ok` is the headline."""
    job_name: str
    job_id: str
    engine: str
    batch_status: str                      # SUBMITTED|PENDING|RUNNABLE|STARTING|RUNNING|SUCCEEDED|FAILED
    elapsed_s: int | None
    learning: bool                         # a progress line was seen in the recent tail
    liveness_ok: bool                      # RUNNING and learning, OR a terminal-success state
    latest_step: int | None = None
    total_steps: int | None = None
    latest_loss: float | None = None
    latest_epoch: float | None = None
    latest_reward: float | None = None
    last_log_ts: int | None = None         # epoch ms of the newest log event seen
    output_s3: str | None = None
    summary: str = ""                      # one-line human verdict
    notes: list[str] = field(default_factory=list)


# Batch lifecycle buckets.
_TERMINAL_OK = {"SUCCEEDED"}
_TERMINAL_BAD = {"FAILED"}
_PRE_RUN = {"SUBMITTED", "PENDING", "RUNNABLE", "STARTING"}


def build_verdict(
    *,
    job_name: str,
    job_id: str,
    batch_status: str,
    elapsed_s: int | None,
    progress: Progress,
    last_log_ts: int | None = None,
    output_s3: str | None = None,
    liveness_deadline_s: int | None = None,
    has_checkpoint: bool = False,
) -> StatusVerdict:
    """Combine Batch state + parsed progress into the headline verdict. Pure.

    The rule is the one the human applies by hand:
      - terminal SUCCEEDED            → liveness_ok (it finished)
      - terminal FAILED               → not ok
      - RUNNING + a progress line     → ok, learning
      - RUNNING + NO progress line    → the dangerous case (RUNNING ≠ learning): not ok,
        and if elapsed exceeds the liveness deadline, call it a probable stall.
      - pre-run (RUNNABLE/STARTING/…) → ok-but-waiting (capacity), not yet learning
    """
    learning = progress.saw_progress_line or has_checkpoint
    notes: list[str] = []
    status = (batch_status or "").upper()

    if status in _TERMINAL_OK:
        liveness_ok = True
        summary = f"{job_name}: SUCCEEDED."
    elif status in _TERMINAL_BAD:
        liveness_ok = False
        summary = f"{job_name}: FAILED — inspect logs (see get_job_status notes / CloudWatch)."
    elif status == "RUNNING":
        liveness_ok = learning
        if learning:
            bits = []
            if progress.latest_step is not None:
                bits.append(f"step {progress.latest_step}"
                            + (f"/{progress.total_steps}" if progress.total_steps else ""))
            if progress.latest_epoch is not None:
                bits.append(f"epoch {progress.latest_epoch:g}")
            if progress.latest_loss is not None:
                bits.append(f"loss {progress.latest_loss:g}")
            if progress.latest_reward is not None:
                bits.append(f"reward {progress.latest_reward:g}")
            summary = f"{job_name}: RUNNING and learning ({', '.join(bits) or 'progress seen'})."
        else:
            summary = (f"{job_name}: RUNNING but NO learning signal in the recent log tail "
                       f"— RUNNING != learning.")
            notes.append("No step/loss/iteration line found yet. If the job has been RUNNING "
                         "for more than a few minutes with no progress, it may be stalled "
                         "(the bootstrap liveness guard will SIGTERM it at its deadline).")
            if liveness_deadline_s and elapsed_s and elapsed_s > liveness_deadline_s:
                notes.append(f"elapsed {elapsed_s}s exceeds the liveness deadline "
                             f"{liveness_deadline_s}s — probable stall.")
    elif status in _PRE_RUN:
        liveness_ok = True
        summary = (f"{job_name}: {status} — waiting for capacity/placement (no GPU cost while "
                   f"pending). Not learning yet; poll again shortly.")
    else:
        liveness_ok = False
        summary = f"{job_name}: {status or 'UNKNOWN'} — unrecognized state."

    return StatusVerdict(
        job_name=job_name, job_id=job_id, engine=progress.engine,
        batch_status=status, elapsed_s=elapsed_s, learning=learning, liveness_ok=liveness_ok,
        latest_step=progress.latest_step, total_steps=progress.total_steps,
        latest_loss=progress.latest_loss, latest_epoch=progress.latest_epoch,
        latest_reward=progress.latest_reward, last_log_ts=last_log_ts,
        output_s3=output_s3, summary=summary, notes=notes,
    )
