#!/bin/bash
#SBATCH -p parallel -A lcheng24 -N 2 --ntasks-per-node=1 -c 32 --mem=120G -t 6:00:00
#SBATCH -J zc_dea_mpi
#
# EOM-DEA-CCSDT of the native program over MPI ranks (cfour/xzccsdt,
# docs/native.md, section 20.6): one run directory on the shared file
# system (bench/dea/native_rundir.py), the CCSDT amplitudes once (one rank),
# then the roots of %xzdea with 1 rank x 32 threads and 2 ranks x 16 threads
# on the first node, 2 ranks x 32 threads and 4 ranks x 16 threads on both.
#
#   sbatch native_mpi.sbatch SNAP NPZ FREEZE RUNDIR [ESTATE_THR] [MEM_GB]
#
#   SNAP    a snapshot with the MPI binary SNAP/xzccsdt_mpi and the Python package (native_rundir.py)
#   RUNDIR  on a file system that both nodes see (made here if it has no ZMAT)
# (no set -u: setvars.sh of oneAPI reads unset variables)
snap=$1
npz=$2
freeze=$3
rdir=$4
thr=${5:-1e-10}
memgb=${6:-100}
export PYTHONPATH=$snap/src:$HOME/apps/socutils-git-env:${PYTHONPATH:-}
source /data/lcheng24/intel/oneapi/setvars.sh --force > /dev/null 2>&1
export PATH=$PATH:/cm/shared/apps/slurm/current/bin
export I_MPI_PIN_DOMAIN=omp
nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
echo "nodes ${nodes[*]}, job $SLURM_JOB_ID, $(date)"
if [ ! -f "$rdir/ZMAT" ]; then
  mkdir -p "$rdir"
  python -u "$snap/bench/dea/native_rundir.py" "$npz" "$rdir" --freeze "$freeze" --mem-gb "$memgb" --cc-conv 10 \
    --estate-conv 9 > "$rdir/rundir.log" 2>&1
  printf '\n%%eomthrs\n%s\n\n' "$thr" >> "$rdir/ZMAT"
fi
cd "$rdir"
run() {   # tag ranks ranks-per-node threads hosts
  export OMP_NUM_THREADS=$4
  echo "=== $1: $2 ranks, $3 per node, $4 threads, hosts $5"
  /usr/bin/time -v mpirun -np "$2" -ppn "$3" -hosts "$5" "$snap/xzccsdt_mpi" > "xzccsdt.$1.out" 2> "xzccsdt.$1.time"
  echo "exit $?"
  grep "sigma vectors:\|per sigma\|the vector to the ranks\|work array used\|peak resident\|EOM-DEA \|over " "xzccsdt.$1.out"
  sed -n '/the states (two/,/XZDEA has/p' "xzccsdt.$1.out" | grep "^ *[-0-9]"
  grep "Elapsed\|Maximum resident" "xzccsdt.$1.time"
  cp XZDEA "XZDEA.$1" 2> /dev/null
}
# the amplitudes (XZAMPS) once, one rank, all cores of the first node
if [ ! -f XZAMPS ]; then
  run ccsdt 1 1 32 "${nodes[0]}"
fi
run r1x32 1 1 32 "${nodes[0]}"
run r2x16 2 2 16 "${nodes[0]}"
run n2r2x32 2 1 32 "${nodes[0]},${nodes[1]}"
run n2r4x16 4 2 16 "${nodes[0]},${nodes[1]}"
echo "done $(date)"
