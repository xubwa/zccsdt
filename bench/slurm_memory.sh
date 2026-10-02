#!/bin/bash -l
#SBATCH --job-name=zcmem
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -A lcheng24
#SBATCH --partition=parallel
#SBATCH --mem=8G
# Spin-adapted CCSDT under the memory limit of the job (the cgroup of --mem):
# what does not fit in it goes to the node-local scratch directory.
# usage: sbatch --mem=6G -o bench/logs/%x-%j.out bench/slurm_memory.sh MOL BASIS [NITER] [NCORE]
#   ZC_Q=1: the (Q) correction as well;  ZC_MAXMEM (MB), ZC_PLAN, ZC_INCORE,
#   ZC_DIIS_SPACE as in bench/profile_sa.py
# The partition allows 4000 MB per core: --mem above 32G needs more cores (-c).
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-PASSIVE}
# scratch files (DIIS vectors, T3, integrals) on the local disk of the node
export ZCCSDT_TMPDIR=/tmp/zccsdt_$SLURM_JOB_ID
export PYSCF_TMPDIR=$ZCCSDT_TMPDIR
mkdir -p $ZCCSDT_TMPDIR
trap 'rm -rf $ZCCSDT_TMPDIR' EXIT
# ZC_ROOT: run from a snapshot of the repository (default: the working tree)
cd ${ZC_ROOT:-/home/xwang405/apps/zccsdt}
echo "host $(hostname)  $(date)  cpus $(taskset -cp $$ | sed 's/.*: //')  mem ${SLURM_MEM_PER_NODE} MB"
lscpu | grep 'Model name'
df -h /tmp | tail -1
export PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1
python -c "
from zccsdt import memory
c = memory.cgroup_available()
print('memory: node %.1f GB, cgroups leave %s GB, default budget %.1f GB'
      % (memory.node_available() / 1e9, c and round(c / 1e9, 1), memory.budget() / 1e9))"
/usr/bin/time -v python -u bench/profile_sa.py "$@" 2>&1 | grep -v -E "^\s+(Average|Major|Minor|Voluntary|Involuntary|Swaps|File system|Socket|Signals|Page size|Percent)"
echo "done $(date)"
