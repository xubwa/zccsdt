#!/bin/bash -l
#SBATCH --job-name=lamrefs
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -A lcheng24
#SBATCH --partition=parallel
#SBATCH --mem=30G
# CCSDT(Q)_Lambda of MRCC and of CFOUR (CC_PROG=NCC) for the inputs
# bench/refs/mrcc/*/MINP.ql.bohr and bench/refs/cfour/*/ZMAT.ql.bohr, and the
# dense reference of zccsdt (bench/lambda_ccsdt.py) for the same molecules,
# one after the other on one node.
# usage: ZC_ROOT=<repository> sbatch -o <log> bench/slurm_lambda_refs.sh
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
ROOT=${ZC_ROOT:-/home/xwang405/apps/zccsdt}
export ZC_ROOT=$ROOT
export ZCCSDT_TMPDIR=/tmp/zccsdt_$SLURM_JOB_ID
export PYSCF_TMPDIR=$ZCCSDT_TMPDIR
export ZC_LOGS=${ZC_LOGS:-$ROOT/bench/logs}
mkdir -p $ZCCSDT_TMPDIR $ZC_LOGS
trap 'rm -rf $ZCCSDT_TMPDIR' EXIT
cd $ROOT
echo "host $(hostname)  $(date)"
for m in bh_sto3g h2o_sto3g h2o_631g; do
    bash bench/mrcc_nr.sh bench/refs/mrcc/$m/MINP.ql.bohr ${m}_ql
done
export PATH=${CFOUR_BIN:-/home/xwang405/apps/cfour-x2cso/bin}:$PATH
# (no BH: with SYMMETRY=OFF the SCF of CFOUR converges to another state)
for m in h2o_sto3g h2o_631g; do
    run=$ZCCSDT_TMPDIR/cfour_$m
    mkdir -p $run
    cp bench/refs/GENBAS $run/GENBAS
    cp bench/refs/cfour/$m/ZMAT.ql.bohr $run/ZMAT
    (cd $run && xcfour > out 2>&1)
    echo "== CFOUR NCC $m"
    grep -E "E\(SCF\)|CCSDT.* energy|\(Q\)|Lambda|LAMBDA|converged" $run/out | tail -30
    cp $run/out $ZC_LOGS/cfour_${m}_ql-$SLURM_JOB_ID.out
done
export PYTHONPATH=$ROOT/src:$HOME/apps/socutils-git-env:$PYTHONPATH
python -u bench/lambda_ccsdt.py bh STO-3G field
python -u bench/lambda_ccsdt.py h2o STO-3G field
python -u bench/lambda_ccsdt.py h2o STO-3G complex field
python -u bench/lambda_ccsdt.py h2o 6-31G complex
python -u bench/lambda_ccsdt.py h2o 6-31G field
echo "done $(date)"
