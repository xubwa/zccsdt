#!/bin/bash -l
#SBATCH --job-name=ao2mo
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -A lcheng24
#SBATCH --partition=parallel
#SBATCH --mem=30G
# Spinor AO-to-MO transformations side by side (bench/profile_ao2mo.py): one
# run per argument, one after the other on the same node.
# usage: sbatch -o bench/logs/%x-%j.out bench/slurm_ao2mo.sh 'MOL BASIS [METHODS] [BLOCKS]' ...
# ZC_ROOT: snapshot of the repository; SOCUTILS: directory that holds the
# socutils package with lib/ao2mo/nrr_fast (default ~/apps/socutils-git-env)
# ZC_CFOUR='h2o:PVTZ ...': CFOUR's chain up to xx2cvtran first, on the same
#   node (bench/cfour/slurm_x2ccc.sh; its transformation makes the integrals
#   with at least one occupied index, no <ab||cd>)
# The partition allows 4000 MB per core: --mem=30G keeps the allocation at the
# 8 requested cores (a larger --mem silently adds cores).
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
root=${ZC_ROOT:-/home/xwang405/apps/zccsdt}
scr=/tmp/zccsdt_ao2mo_${SLURM_JOB_ID:-local$$}
mkdir -p $scr
export PYSCF_TMPDIR=$scr
echo "host $(hostname)  $(date)  threads $OMP_NUM_THREADS  root $root  scratch $scr"
lscpu | grep 'Model name'
if [ -n "$ZC_CFOUR" ]; then
  CFOUR_PROGS='xjoda xsox2catm xsox2chf xx2cvtran' CFOUR_CALC=CCSD ZC_NITER=0 \
    bash $root/bench/cfour/slurm_x2ccc.sh $ZC_CFOUR
fi
for run in "$@"; do
  echo "=== $run  $(date)"
  (cd $root && PYTHONPATH=src:${SOCUTILS:-$HOME/apps/socutils-git-env}:$PYTHONPATH \
     /usr/bin/time -f "wall %e s  max RSS %M KB" python -u bench/profile_ao2mo.py $run) 2>&1 \
     | grep --line-buffered -v "plugins found\|PYSCF_EXT_PATH\|^/home/"
  rm -rf $scr/*
done
rm -rf $scr
echo "done $(date)"
