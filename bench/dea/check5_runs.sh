#!/bin/bash
# native_check5.py once per set of options:
#   bash bench/dea/check5_runs.sh <rundir> <binary> "<options>" "<options>" ...
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
dir=$1; bin=$2; shift 2
for o in "$@"; do
  echo "== $o"
  # shellcheck disable=SC2086
  bash "$ROOT/bench/dea/py.sh" "$ROOT/bench/dea/native_check5.py" "$dir" "$bin" $o 2>&1 | \
    grep -E '^2p|^3p1h|^4p2h|^5p3h|stopped|Error|error|native: option deasig'
done
