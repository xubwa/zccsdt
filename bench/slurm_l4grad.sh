#!/bin/bash -l
#SBATCH --job-name=l4grad
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -A lcheng24
#SBATCH --partition=parallel
#SBATCH --mem=30G
# The relaxed density of CCSDTQ from the dense reference against finite
# fields (bench/l4grad_ccsdtq.py) and against CFOUR's ncc (CALC=CCSDTQ,
# PROPS=FIRST_ORDER, the inputs bench/refs/cfour/*/ZMAT.ccsdtq_props.bohr),
# one after the other on one node.
# usage: ZC_ROOT=<snapshot of the repository> ZC_SET=small|x2c|mid|large sbatch -o <log> bench/slurm_l4grad.sh
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
ROOT=${ZC_ROOT:-/home/xwang405/apps/zccsdt}
export ZCCSDT_TMPDIR=/tmp/zccsdt_$SLURM_JOB_ID
export PYSCF_TMPDIR=$ZCCSDT_TMPDIR
mkdir -p $ZCCSDT_TMPDIR
trap 'rm -rf $ZCCSDT_TMPDIR' EXIT
cd $ROOT
echo "host $(hostname)  $(date)  set ${ZC_SET:-small}"
source /data/lcheng24/intel/oneapi/setvars.sh --force > /dev/null 2>&1
CFOUR=${CFOUR_BIN:-/home/xwang405/apps/cfour-x2cso/bin}

cfour () {
    run=$ZCCSDT_TMPDIR/cfour_$1
    mkdir -p $run
    cp bench/refs/GENBAS $run/GENBAS
    cp bench/refs/cfour/$1/ZMAT.ccsdtq_props.bohr $run/ZMAT
    (cd $run && PATH=$CFOUR:$PATH xcfour > out 2>&1)
    echo "== CFOUR NCC CCSDTQ $1"
    grep -E "E\(SCF\)=|Total CCSDTQ energy|iterations converged" $run/out
    grep -A 8 "Properties computed from the correlated density" $run/out | grep -E "X =|XX =|XY ="
    if [ -n "$ZC_LOGS" ]; then cp $run/out $ZC_LOGS/cfour_$1_ccsdtq_props-$SLURM_JOB_ID.out; fi
}

export PYTHONPATH=$ROOT/src:$HOME/apps/socutils-git-env:$PYTHONPATH
run () {
    echo "== zccsdt $*"
    python -u bench/l4grad_ccsdtq.py "$@" 2>&1 | grep --line-buffered -v -E "pyscf plugins|site-packages/pyscf|shciscf|bench/pyscf|PYSCF_EXT_PATH"
}

if [ "${ZC_SET:-small}" = small ]; then
    cfour h2o_sto3g
    cfour nh3_sto3g
    run h2o:STO-3G:real
    run bh:STO-3G:real
    run bh:STO-3G:complex
    run lih:STO-3G:complex steps=2e-3,1e-3,5e-4,2.5e-4
    run bh:STO-3G:x2c
    run bh:STO-3G:complex 2 2
    run bh:STO-3G:x2c 2 2
    run nh3:STO-3G:real
    run nh3:STO-3G:complex 2 0
elif [ "$ZC_SET" = x2c ]; then
    run bh:STO-3G:x2c
    run bh:STO-3G:x2c 2 2
elif [ "$ZC_SET" = mid ]; then
    run h2o:6-31G:real 2 4
else
    cfour h2o_631g
    run h2o:6-31G:real verbose=1
fi
echo "done $(date)"
