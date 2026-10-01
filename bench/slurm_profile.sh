#!/bin/bash -l
#SBATCH --job-name=zccsdt
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
#SBATCH -A lcheng24
#SBATCH --partition=parallel
#SBATCH --mem=180G
# usage: sbatch -o bench/logs/%x-%j.out bench/slurm_profile.sh MOL BASIS KIND NITER [FROZEN]
export OMP_NUM_THREADS=48
export MKL_NUM_THREADS=48
export OPENBLAS_NUM_THREADS=48
cd /home/xwang405/apps/zccsdt
echo "host $(hostname)  $(date)"
lscpu | grep 'Model name'
PYTHONPATH=src:$PYTHONPATH python bench/profile_iter.py "$@"
echo "done $(date)"
