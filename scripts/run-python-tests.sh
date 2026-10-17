#!/usr/bin/env bash
# Run every Python unit test in the repo. Each test imports its sibling modules, so it
# runs from its own directory. Needs only boto3 (no AWS credentials, no GPU).
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
fail=0
for t in \
  containers/vla-ft/test_vla_ft_decide.py \
  containers/vla-ft/test_train_resume.py \
  containers/vla-ft/test_dcp_checkpoint.py \
  containers/isaac-lab-rl/test_rl_liveness_guard.py \
  containers/gr00t-n17/test_gr00t_liveness_guard.py \
  mcp/test_mcp.py; do
  echo "== $t"
  (cd "$(dirname "$t")" && "$PY" "$(basename "$t")")
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "FAILED ($rc): $t"
    fail=1
  fi
done
exit "$fail"
