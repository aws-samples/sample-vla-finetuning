#!/usr/bin/env python3
"""
Compare two LIBERO evals of the same reset states: is the after-checkpoint better?

Each eval (vla_rl_launch.py eval) logs one line per episode,
`[libero eval] task_id=<t>, trial_id=<r>, success=<True|False>`, and with fixed reset
states the same (task, trial) pair is the same initial scene in both runs. So beside the
unpaired one-sided Fisher exact test this reports the paired one-sided McNemar exact test
on the discordant pairs, which is the sharper of the two here.

  python vla_rl_compare.py <before> <after>

<before> / <after> = a Batch job id (reads its CloudWatch log stream) or a text file of
those lines. Stdlib only, except boto3 for the job-id form.
"""

import argparse
import os
import re
import sys
from math import comb

LINE = re.compile(r"task_id=(\d+), trial_id=(\d+), success=(True|False)")


def parse_trials(text):
    """{(task, trial): success}. Later lines win, so a re-logged episode counts once."""
    return {(int(t), int(r)): s == "True" for t, r, s in LINE.findall(text)}


def fisher_one_sided(k_before, k_after, n_before, n_after):
    """P(after >= k_after | margins), hypergeometric: is after's success rate higher?"""
    total, succ = n_before + n_after, k_before + k_after
    denom = comb(total, n_after)
    return sum(comb(succ, k) * comb(total - succ, n_after - k)
               for k in range(k_after, min(succ, n_after) + 1)) / denom


def mcnemar_one_sided(gained, lost):
    """Exact binomial on discordant pairs: P(X >= gained), X ~ Bin(gained + lost, 0.5)."""
    m = gained + lost
    if m == 0:
        return 1.0
    return sum(comb(m, k) for k in range(gained, m + 1)) / 2 ** m


def compare(before, after):
    shared = sorted(set(before) & set(after))
    if not shared:
        raise ValueError("no (task, trial) pair is in both evals")
    b = sum(before[k] for k in shared)
    a = sum(after[k] for k in shared)
    gained = sum(1 for k in shared if after[k] and not before[k])
    lost = sum(1 for k in shared if before[k] and not after[k])
    n = len(shared)
    tasks = sorted({t for t, _ in shared})
    per_task = {t: (sum(before[k] for k in shared if k[0] == t),
                    sum(after[k] for k in shared if k[0] == t),
                    sum(1 for k in shared if k[0] == t)) for t in tasks}
    return {
        "n": n, "before": b, "after": a, "gained": gained, "lost": lost,
        "fisher_p": fisher_one_sided(b, a, n, n),
        "mcnemar_p": mcnemar_one_sided(gained, lost),
        "per_task": per_task,
        "unpaired": len(set(before) ^ set(after)),
    }


def load(src, region=None):
    if os.path.exists(src):
        with open(src) as f:
            return parse_trials(f.read())
    import boto3  # job-id form only
    sess = boto3.Session(region_name=region) if region else boto3.Session()
    job = sess.client("batch").describe_jobs(jobs=[src])["jobs"]
    if not job:
        raise SystemExit(f"{src}: neither a file nor a Batch job id")
    stream = job[0].get("container", {}).get("logStreamName")
    if not stream:
        raise SystemExit(f"{src}: job has no log stream yet (status {job[0]['status']})")
    logs = sess.client("logs")
    kw = {"logGroupName": "/aws/batch/job", "logStreamNames": [stream],
          "filterPattern": '"[libero eval]"'}
    text = []
    while True:
        page = logs.filter_log_events(**kw)
        text += [e["message"] for e in page["events"]]
        if not page.get("nextToken"):
            break
        kw["nextToken"] = page["nextToken"]
    return parse_trials("\n".join(text))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("before")
    p.add_argument("after")
    p.add_argument("--region", default=None)
    args = p.parse_args(argv)
    r = compare(load(args.before, args.region), load(args.after, args.region))
    n = r["n"]
    print(f"n = {n} paired episodes"
          + (f" ({r['unpaired']} unpaired dropped)" if r["unpaired"] else ""))
    print(f"success  before {r['before'] / n:.3f} ({r['before']})  after {r['after'] / n:.3f} "
          f"({r['after']})  delta {100 * (r['after'] - r['before']) / n:+.1f} pt")
    print(f"paired   fail->success {r['gained']}  success->fail {r['lost']}")
    print(f"one-sided p  Fisher {r['fisher_p']:.3f}  McNemar {r['mcnemar_p']:.3f}")
    print("per task (before -> after / trials):")
    for t, (b, a, k) in r["per_task"].items():
        print(f"  task {t:>2}: {b:>3} -> {a:>3} / {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
