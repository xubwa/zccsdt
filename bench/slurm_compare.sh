#!/bin/bash -l
#SBATCH --job-name=cmp
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -A lcheng24
#SBATCH --partition=parallel
#SBATCH --mem=30G
# Same-allocation comparison of zccsdt and CFOUR: both programs run back to
# back on the same cores of one job.
# usage: sbatch -o bench/logs/%x-%j.out bench/slurm_compare.sh \
#            MOL PYSCF_BASIS CFOUR_BASIS ZC_KIND CC_PROG REF [NITER]
#   closed shell (RCCSDT, R-form integrals):  ... h2o cc-pvqz PVQZ ghf NCC RHF
#   open shell (UCCSDT, U-form integrals):    ... nh2 cc-pvtz PVTZ uhf ECC UHF
#   ZC_SA=1: closed-shell spin-adapted RCCSDT (bench/profile_sa.py) instead
#   ZC_NCORE=n with CFOUR_FC=ON: n frozen core orbitals in both programs
mol=$1; pbasis=$2; cbasis=$3; kind=$4; prog=$5; ref=$6; niter=${7:-3}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-PASSIVE}
# ZC_ROOT: run from a snapshot of the repository (default: the working tree)
cd ${ZC_ROOT:-/home/xwang405/apps/zccsdt}
echo "host $(hostname)  $(date)"
lscpu | grep 'Model name'
echo "cpus: $(taskset -cp $$)"
flip=0; [ "$kind" = ghf ] && flip=1
if [ "${ZC_SA:-0}" = 1 ]; then
    echo "=== zccsdt spin-adapted RCCSDT $mol/$pbasis"
    PYTHONPATH=src:$PYTHONPATH python bench/profile_sa.py $mol $pbasis $niter ${ZC_NCORE:-0}
else
    echo "=== zccsdt $mol/$pbasis kind=$kind flip=$flip"
    ZC_RU=1 ZC_FLIP=$flip ZC_MEM_MB=${ZC_MEM_MB:-8000} ZC_VBATCH=48 PYTHONPATH=src:$PYTHONPATH \
        python bench/profile_iter.py $mol $pbasis $kind $niter
fi
echo "=== CFOUR $prog $ref $mol/$cbasis"
bash bench/cfour/slurm_cfour.sh $mol $cbasis $prog $ref $((niter + 2))
keep=/home/xwang405/.claude/jobs/7ba4fcf7/tmp/cfour_bench/${mol}_${cbasis}_${prog}_${ref}_${SLURM_JOB_ID}
grep -E "^ +[0-9]+ +-?[0-9]+\.[0-9]{6}" $keep/out | tail -$((niter + 2))
echo "done $(date)"
