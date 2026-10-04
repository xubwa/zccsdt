#!/bin/bash -l
#SBATCH --job-name=zccsd
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -A lcheng24
#SBATCH --partition=parallel
#SBATCH --mem=30G
# The spinor CCSD and (T) copied from socutils (zccsdt.zccsd), one run of
# bench/profile_zccsd.py per argument, one after the other on the same node.
# usage: sbatch -o bench/logs/%x-%j.out bench/slurm_zccsd.sh 'MOL BASIS VARIANT [NITER] [T_ALGS]' ...
#   a run may end with ' @ VAR=VALUE ...': environment of that run only
# ZC_CFOUR='h2o:PVTZ h2o:PVQZ': CFOUR's xx2cccsd first, on the same node
#   (bench/cfour/slurm_x2ccc.sh without its zccsdt part); ZC_CFOUR_CCSD: the
#   same for molecules run with CALC=CCSD (no (T))
# ZC_ROOT: snapshot of the repository; scratch files in node-local /tmp
# ZC_EXTRA: a command run in the repository at the end (BLAS checks)
# The zccsdt runs have OMP_WAIT_POLICY=PASSIVE: the OpenMP kernels and a
# BLAS with threads of its own (numpy's OpenBLAS) take turns on the same
# cores, and OpenMP threads that spin while waiting slow the BLAS down.
# The partition allows 4000 MB per core: --mem=30G keeps the allocation at the
# 8 requested cores (a larger --mem silently adds cores).
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
root=${ZC_ROOT:-/home/xwang405/apps/zccsdt}
scr=/tmp/zccsdt_zccsd_${SLURM_JOB_ID:-local$$}
mkdir -p $scr
export PYSCF_TMPDIR=$scr
echo "host $(hostname)  $(date)  threads $OMP_NUM_THREADS  root $root  scratch $scr"
lscpu | grep 'Model name'
if [ -n "$ZC_CFOUR" ]; then
  ZC_NITER=0 bash $root/bench/cfour/slurm_x2ccc.sh $ZC_CFOUR
fi
if [ -n "$ZC_CFOUR_CCSD" ]; then
  CFOUR_CALC=CCSD ZC_NITER=0 bash $root/bench/cfour/slurm_x2ccc.sh $ZC_CFOUR_CCSD
fi
export OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-PASSIVE}
for run in "$@"; do
  echo "=== zccsdt.zccsd: $run  $(date)"
  spec=${run%% @ *}
  envs=; [ "$spec" != "$run" ] && envs=${run#* @ }
  (cd $root && PYTHONPATH=src:$PYTHONPATH env $envs /usr/bin/time -f "wall %e s  max RSS %M KB" \
     python -u bench/profile_zccsd.py $spec) 2>&1 | grep --line-buffered -v "plugins found\|PYSCF_EXT_PATH\|^/home/"
  df -h $scr | tail -n 1
  rm -rf $scr/*
done
rm -rf $scr
if [ -n "$ZC_EXTRA" ]; then
  echo "=== $ZC_EXTRA  $(date)"
  (cd $root && eval "$ZC_EXTRA") 2>&1 | grep --line-buffered -v "plugins found\|PYSCF_EXT_PATH\|^/home/"
fi
echo "done $(date)"
