#!/bin/bash
# native_check5.py piece by piece: bash bench/dea/check5_pieces.sh <rundir> <binary> [pieces ...] [-- extra args]
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
dir=$1; bin=$2; shift 2
pieces=(); extra=()
while [ $# -gt 0 ]; do
  if [ "$1" = "--" ]; then shift; extra=("$@"); break; fi
  pieces+=("$1"); shift
done
[ ${#pieces[@]} -eq 0 ] && pieces=(pp ring hh x23 x14 e1x)
for p in "${pieces[@]}"; do
  echo "== $p"
  bash "$ROOT/bench/dea/py.sh" "$ROOT/bench/dea/native_check5.py" "$dir" "$bin" --piece "$p" "${extra[@]}" 2>&1 | grep -E '^5p3h|stopped|Error|error'
done
