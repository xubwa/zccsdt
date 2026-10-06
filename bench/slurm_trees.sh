#!/bin/bash -l
#SBATCH --job-name=trees
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -A lcheng24
#SBATCH --partition=parallel
#SBATCH --mem=30G
# One command in several trees of the repository in turn, on the same node
# and cores: A B A B ... - two versions of the code under the same conditions.
# usage: sbatch -o bench/logs/%x-%j.out bench/slurm_trees.sh 'ROOT_A ROOT_B' ROUNDS COMMAND...
#   COMMAND runs in each ROOT with PYTHONPATH=src and ZC_ROOT=ROOT, e.g.
#   python -u bench/profile_frozen.py hi dyallv3z,unc-ccpvtz 36 1
# environment: passed on (ZC_MAXMEM, ZC_VIRCUT, ZC_SCF, ZC_HAM, ...);
#   SOCUTILS: the directory with socutils-git as 'socutils' (default
#   ~/apps/socutils-git-env), for ZC_SCF=socutils.  Set ZC_SPINORS to a file
#   for the spinors: the first run makes it, the others read it (random
#   amplitudes are in the basis of the spinors, and every SCF mixes the two
#   spinors of a Kramers pair differently).
# More than 8 cores: sbatch --cpus-per-task=16 --mem=60G (4000 MB per core).
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
roots=$1; rounds=$2; shift 2
scr=/tmp/zccsdt_trees_${SLURM_JOB_ID:-local$$}
mkdir -p $scr
export PYSCF_TMPDIR=$scr ZCCSDT_TMPDIR=$scr
echo "host $(hostname)  $(date)  threads $OMP_NUM_THREADS  scratch $scr"
lscpu | grep 'Model name'
echo "command: $*"
for r in $(seq $rounds); do
  for root in $roots; do
    echo "=== $root  round $r  $(date)"
    (cd $root && ZC_ROOT=$root PYTHONPATH=src:${SOCUTILS:-$HOME/apps/socutils-git-env}:$PYTHONPATH \
       /usr/bin/time -f "wall %e s  max RSS %M KB" "$@") 2>&1 \
       | grep --line-buffered -v "plugins found\|PYSCF_EXT_PATH\|^/home/\|new_energy_algo"
  done
done
rmdir $scr          # (the scratch files are unlinked when they are opened)
echo "done $(date)"
