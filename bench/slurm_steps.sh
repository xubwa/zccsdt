#!/bin/bash -l
#SBATCH --job-name=zcsteps
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -A lcheng24
#SBATCH --partition=parallel
#SBATCH --mem=30G
# Several runs in one job, one after the other on the same cores and the same
# local disk, each as a job step with its own memory limit (its own memory
# cgroup): the way to compare a run in memory with runs under limits.
# usage: sbatch -o bench/logs/%x-%j.out bench/slurm_steps.sh RUNS
#   each line of the file RUNS:  MEM TAG COMMAND...
#   e.g.  4G c2h4_4G python -u bench/profile_sa.py c2h4 cc-pvtz 3
#   (# starts a comment; variables may be set in front of the command)
# The partition allows 4000 MB per core: --mem above 32G needs more cores (-c).
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-PASSIVE}
# scratch files (DIIS vectors, T3, integrals) on the local disk of the node
export ZCCSDT_TMPDIR=/tmp/zccsdt_$SLURM_JOB_ID
export PYSCF_TMPDIR=$ZCCSDT_TMPDIR
mkdir -p $ZCCSDT_TMPDIR
trap 'rm -rf $ZCCSDT_TMPDIR' EXIT
RUNS=$(readlink -f "$1")
# ZC_ROOT: run from a snapshot of the repository (default: the working tree)
cd ${ZC_ROOT:-/home/xwang405/apps/zccsdt}
echo "host $(hostname)  $(date)  cpus $(taskset -cp $$ | sed 's/.*: //')  mem ${SLURM_MEM_PER_NODE} MB"
lscpu | grep 'Model name'
df -h /tmp | tail -1
export PYTHONPATH=src:$PYTHONPATH PYTHONDONTWRITEBYTECODE=1
# (the login shell of a batch job may not have the Slurm commands in its PATH)
SRUN=$(command -v srun || echo /cm/shared/apps/slurm/current/bin/srun)
# what the memory cgroup of a step counted (the limit is on the step, the
# parent of the cgroup of its task): times the limit was hit, page faults
# that had to read from the disk
STAT='cg=$(dirname /sys/fs/cgroup/memory$(grep memory /proc/self/cgroup | cut -d: -f3)); echo "cgroup: limit $(cat $cg/memory.limit_in_bytes) failcnt $(cat $cg/memory.failcnt) max_usage $(cat $cg/memory.max_usage_in_bytes) $(grep -E "^total_(pgmajfault|pgfault) " $cg/memory.stat | tr "\n" " ")"'
grep -v -E '^\s*(#|$)' "$RUNS" | while read -r mem tag cmd; do
    echo "=== $tag  --mem=$mem  $(date +%T)  load $(cut -d' ' -f1-3 /proc/loadavg)"
    echo "    $cmd"
    t0=$(date +%s)
    $SRUN -n1 -c $OMP_NUM_THREADS --mem=$mem bash -c "$cmd; $STAT" < /dev/null 2>&1 \
        | grep -v -E "pyscf plugins|site-packages/pyscf|shciscf/pyscf|PYSCF_EXT_PATH"
    echo "=== end $tag  $(( $(date +%s) - t0 )) s"
done
echo "done $(date)"
