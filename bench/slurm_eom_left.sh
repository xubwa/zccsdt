#!/bin/bash -l
#SBATCH --job-name=zc_leftb
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -A lcheng24
#SBATCH --partition=parallel
#SBATCH --mem=30G
# Left-hand vectors of EOM-EE-CCSDT against the right-hand ones, one node.
#   usage: sbatch -o bench/logs/%x-%j.out bench/slurm_eom_left.sh
#   ZC_ROOT: snapshot of the repository (with src/zccsdt/lib/*.so)
#   EL_PARTS: which parts run (default "h2o hbr hbr2 hbr_dense"; qz2, qz0: 26 + 120 spinors -
#   sbatch -c 24 --mem=90G, EB_MAXMEM=80000)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
cd ${ZC_ROOT:-/home/xwang405/apps/zccsdt}
ROOT=$(pwd)
export PYTHONPATH=$ROOT/src:$HOME/apps/socutils-git-env:$PYTHONPATH
S=/tmp/zccsdt_eoml_${SLURM_JOB_ID:-$$}
mkdir -p $S
export ZCCSDT_TMPDIR=$S PYSCF_TMPDIR=$S
echo "host $(hostname)  $(date)  threads $OMP_NUM_THREADS"
lscpu | grep 'Model name'
filter() { grep -v -i -E "plugins found|site-packages/pyscf|shciscf|PYSCF_EXT_PATH"; }
parts=${EL_PARTS:-"h2o hbr hbr2 hbr_dense"}
for part in $parts; do
case $part in
h2o)       echo "=== H2O / cc-pVTZ, packed block  $(date)"; python -u bench/eom_left_bench.py h2o 2 2>&1 | filter ;;
hbr)       echo "=== HBr / cc-pVTZ-DK, label blocks, shift 0  $(date)"; python -u bench/eom_left_bench.py hbr 2 0 2>&1 | filter ;;
hbr2)      echo "=== HBr / cc-pVTZ-DK, label blocks, shift 2  $(date)"; python -u bench/eom_left_bench.py hbr 2 2 2>&1 | filter ;;
hbr_dense) echo "=== HBr / cc-pVTZ-DK, packed block  $(date)"; python -u bench/eom_left_bench.py hbr_dense 3 2>&1 | filter ;;
qz2)       echo "=== HBr / cc-pVQZ-DK (26 + 120), label blocks, shift 2  $(date)"; python -u bench/eom_left_bench.py hbr_qz 2 2 2>&1 | filter ;;
qz0)       echo "=== HBr / cc-pVQZ-DK (26 + 120), label blocks, shift 0  $(date)"; python -u bench/eom_left_bench.py hbr_qz 2 0 2>&1 | filter ;;
esac
done
find "$S" -delete
echo "done $(date)"
