#!/bin/bash
#SBATCH -p parallel -A lcheng24 -N 1 --ntasks-per-node=1 -c 48 --mem=180G --exclusive
#SBATCH -J zc_dea5_bench
#
# EOM-DEA with complete 5p3h, one node: the Python sector first (o2_dea5.py
# --sigmas: the ground state and sigma vectors of random vectors), then the
# native program (serial binary, 48 threads) in node-local /tmp on the same
# Hamiltonian - a few iterations per target (ESTATE_MAXCYC), the times of
# its sigma vectors.  docs/native.md, section 21.
#
#   sbatch -t 3:00:00 native5_bench.sbatch SNAP NPZ RUNDIR TAG [MAXCYC] [PYSIGMAS]
#
#   SNAP      a snapshot (bench/dea/snapshot.sh)
#   NPZ       the Hamiltonian file of the Python run
#   RUNDIR    the run directory of the native program (native_rundir.py --level 5p3h), copied to /tmp
#   MAXCYC    ESTATE_MAXCYC of the native run (default 3)
#   PYSIGMAS  sigma vectors of the Python run (default 2; 0: no Python run)
zargs=("$@")
source /data/lcheng24/intel/oneapi/setvars.sh --force > /dev/null 2>&1
set -- "${zargs[@]}"
snap=$1
npz=$2
rdir=$3
tag=$4
maxcyc=${5:-3}
pys=${6:-2}
export PYTHONPATH=$snap/src:$HOME/apps/socutils-git-env:${PYTHONPATH:-}
echo "node $(hostname), job $SLURM_JOB_ID, $(date)"
lscpu | grep 'Model name'
if [ "$pys" -gt 0 ]; then
  export OMP_NUM_THREADS=48
  /usr/bin/time -v python -u "$snap/bench/dea/o2_dea5.py" "$npz" "/tmp/py_$tag.json" --sigmas "$pys" --targets 0 0 \
    --mem 160000 > "$rdir.py.$tag.log" 2> "$rdir.py.$tag.time"
  grep -v -i plugin "$rdir.py.$tag.log" | grep -v pyscf | tail -20
  grep "Elapsed\|Maximum resident" "$rdir.py.$tag.time"
fi
work=/tmp/dea5_$SLURM_JOB_ID
mkdir -p "$work"
cp -r "$rdir"/. "$work"/
cd "$work" || exit 1
avail=$(awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo)
memw=$(( (avail - 6000) * 1000000 / 8 ))
[ "$memw" -gt 20000000000 ] && memw=20000000000
echo "MemAvailable $avail MB; MEMORY_SIZE $memw words"
python -u "$snap/bench/dea/jobarc_flags.py" . 97=$maxcyc 36=$memw
export OMP_NUM_THREADS=48
/usr/bin/time -v "$snap/xzccsdt" > "xzccsdt.$tag.out" 2> "xzccsdt.$tag.time"
echo "exit $?"
grep "CCSDT correlation\|converged in\|sigma vectors:\|per sigma\|5p3h\|of them\|work array used\|peak resident\|intermediates of the ground\|subspace\|a vector" "xzccsdt.$tag.out"
grep "Elapsed\|Maximum resident" "xzccsdt.$tag.time"
cp "xzccsdt.$tag.out" "xzccsdt.$tag.time" "$rdir"/
[ -f XZAMPS ] && cp XZAMPS "$rdir"/ 2> /dev/null
cd /
# (the files of the run in /tmp/dea5_<job>; no subdirectories)
rm -f /tmp/dea5_"$SLURM_JOB_ID"/*
rmdir /tmp/dea5_"$SLURM_JOB_ID"
echo "done $(date)"
