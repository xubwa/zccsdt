#!/bin/bash
# Runs the native program in a run directory:
#   bash bench/dea/run_native.sh <rundir> <binary> <threads> <output file> [ranks]
# (ranks: through mpirun -np, the MPI binary)
dir=$1; bin=$2; thr=$3; out=$4; np=$5
cd "$dir" || exit 1
export OMP_NUM_THREADS=$thr
if [ -n "$np" ]; then
  mpirun -np "$np" "$bin" > "$out" 2>&1
else
  "$bin" > "$out" 2>&1
fi
echo "exit $?"
