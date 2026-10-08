#!/bin/bash -l
#SBATCH --job-name=qlgrad
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -A lcheng24
#SBATCH --partition=parallel
#SBATCH --mem=30G
# The first derivatives of CCSDT(Q)_Lambda next to those of CCSDT and CCSDT(Q)
# (bench/profile_ql_gradient.py), one run after the other on one node: H2O /
# cc-pVTZ as spin orbitals and HBr / cc-pVTZ-DK with X2C spinors, 28 frozen,
# with the other stencils of zeta; then H2O again under a tight max_memory.
# usage: ZC_TREE=<snapshot of the tree> sbatch -o <log> bench/slurm_ql_gradient.sh
# ZC_ONLY: the parts to run, of "h2o hbr tight" (default: all)
# ZC_TIGHT: max_memory in MB of the tight run (default 4000)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-PASSIVE}
export ZCCSDT_TMPDIR=$(mktemp -d /tmp/zccsdt_qlgrad_XXXXXX)
export PYSCF_TMPDIR=$ZCCSDT_TMPDIR
trap 'rm -rf "$ZCCSDT_TMPDIR"' EXIT
ONLY=${ZC_ONLY:-h2o hbr tight}
BASE_PYTHONPATH=$PYTHONPATH
echo "host $(hostname)  job $SLURM_JOB_ID  $(date)  load $(cut -d' ' -f1-3 /proc/loadavg)"
lscpu | grep 'Model name'

run() {     # max_memory, stencils, arguments of the script
  local mem=$1 stencils=$2
  shift 2
  echo "=== $(basename $ZC_TREE): $*  (max_memory $mem MB)  load $(cut -d' ' -f1-3 /proc/loadavg)  $(date)"
  ( cd $ZC_TREE && PYTHONPATH=$ZC_TREE/src:$HOME/apps/socutils-git-env:$BASE_PYTHONPATH ZC_MAXMEM=$mem \
      ZC_STENCILS=$stencils \
      /usr/bin/time -f "elapsed %e s, maximum resident %M kB" python -u bench/profile_ql_gradient.py "$@" 2>&1 \
      | grep -v "plugins found\|PYSCF_EXT_PATH\|^/home/" )
}

if [[ " $ONLY " == *" h2o "* ]]; then
  run 26000 1 h2o cc-pvtz ghf
fi
if [[ " $ONLY " == *" hbr "* ]]; then
  run 26000 1 hbr cc-pvtz-dk x2c 28
fi
if [[ " $ONLY " == *" tight "* ]]; then
  run ${ZC_TIGHT:-4000} "" h2o cc-pvtz ghf
fi
echo "done $(date)  load $(cut -d' ' -f1-3 /proc/loadavg)"
