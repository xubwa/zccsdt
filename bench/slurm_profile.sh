#!/bin/bash -l
#SBATCH --job-name=zccsdt
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -A lcheng24
#SBATCH --partition=parallel
#SBATCH --mem=30G
# usage: sbatch -o bench/logs/%x-%j.out bench/slurm_profile.sh MOL BASIS KIND NITER [FROZEN]
# threads = cores of the allocation (default 8; override with sbatch -c N --mem=...)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-PASSIVE}
# ZC_ROOT: run from a snapshot of the repository (default: the working tree)
cd ${ZC_ROOT:-/home/xwang405/apps/zccsdt}
echo "host $(hostname)  $(date)"
lscpu | grep 'Model name'
PYTHONPATH=src:$PYTHONPATH python bench/profile_iter.py "$@"
echo "done $(date)"
