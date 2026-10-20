#!/bin/bash
#SBATCH -p parallel -A lcheng24 -N 1 -n 1 -c 32 --mem=120G -t 12:00:00
#SBATCH -J zc_dea_nat
#
# EOM-DEA-CCSDT of O2, the native program (cfour/xzccsdt) against the Python
# sector (zccsdt.cc.dea_sector) on one node, the reference first: the Python
# program's ground state, three sigma vectors and the roots of 2 Omega = 0, 2
# (bench/dea/sector_bench.py), then the native program on the same
# Hamiltonian (a run directory of bench/dea/native_rundir.py in node-local
# /tmp: CCSDT and the roots of %xzdea).
#
#   sbatch native_bench.sbatch SNAP NPZ FREEZE OUTDIR [ESTATE_THR] [MEM_GB] [PYTHON=0|1]
#
#   SNAP     a snapshot of the repository (git archive) with src/zccsdt/lib/*.so
#            and the native binary SNAP/xzccsdt
#   NPZ      the Hamiltonian file;  FREEZE  spinors frozen (common.load)
#   OUTDIR   where the outputs go (the json of the Python run, the native output, XZDEA)
#   ESTATE_THR  %eomthrs of the native run (the largest element of a residual; default 1e-10,
#            about the residual norm 1e-9 of the Python run)
#   MEM_GB   MEMORY_SIZE of the native run (default 100)
#   PYTHON   1 (default): the Python reference first; 0: the native program alone
set -u
snap=$1
npz=$2
freeze=$3
outdir=$4
thr=${5:-1e-10}
memgb=${6:-100}
dopy=${7:-1}
tag=$(basename "$npz" .npz)
mkdir -p "$outdir"
export OMP_NUM_THREADS=32
export PYTHONPATH=$snap/src:$HOME/apps/socutils-git-env
echo "node $(hostname), job $SLURM_JOB_ID, $(date)"; lscpu | grep 'Model name'
cd "$snap"
if [ "$dopy" = 1 ]; then
  echo "=== Python sector (the reference)"
  /usr/bin/time -v python -u bench/dea/sector_bench.py "$npz" "$outdir/py_$tag.json" --freeze "$freeze" \
    --backends sector --nsigma 3 --roots --tol 1e-9 --mem 100000 > "$outdir/py_$tag.log" 2> "$outdir/py_$tag.time"
  grep -v "pyscf\|PYSCF" "$outdir/py_$tag.log" | tail -40
  grep "Maximum resident\|Elapsed" "$outdir/py_$tag.time"
fi
echo "=== native: the run directory in /tmp"
d=/tmp/zc_dea_nat_$SLURM_JOB_ID
mkdir -p "$d"
python -u bench/dea/native_rundir.py "$npz" "$d" --freeze "$freeze" --mem-gb "$memgb" --cc-conv 10 --estate-conv 9 \
  > "$outdir/rundir_$tag.log" 2>&1
grep -v "pyscf\|PYSCF" "$outdir/rundir_$tag.log" | tail -3
# the threshold of the roots: %eomthrs
python - "$d/ZMAT" "$thr" <<'EOF'
import sys
p, thr = sys.argv[1], sys.argv[2]
s = open(p).read().rstrip('\n') + '\n\n%%eomthrs\n%s\n\n' % thr
open(p, 'w').write(s.replace('%%eomthrs', '%eomthrs'))
EOF
echo "=== native: CCSDT and EOM-DEA"
(cd "$d" && /usr/bin/time -v "$snap/xzccsdt" > xzccsdt.out 2> time.txt)
cp "$d/xzccsdt.out" "$outdir/native_$tag.out"
cp "$d/time.txt" "$outdir/native_$tag.time"
cp "$d/XZDEA" "$outdir/native_$tag.XZDEA" 2> /dev/null
grep "E(CCSDT)\|converged in\|sigma vectors:\|per sigma\|of them\|takes\|work array used\|peak resident\|intermediates of\|wall time\|EOM-DEA " "$outdir/native_$tag.out"
sed -n '/the states (two/,/XZDEA has/p' "$outdir/native_$tag.out"
grep "Maximum resident\|Elapsed" "$outdir/native_$tag.time"
# (the run directory: its files, then the directory)
rm -f "$d"/moin* "$d"/PPPH* "$d"/XZ* "$d"/EOM* "$d"/* 2> /dev/null
rmdir "$d"
echo "done $(date)"
