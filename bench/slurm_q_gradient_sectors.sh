#!/bin/bash -l
#SBATCH --job-name=qgrad2
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -A lcheng24
#SBATCH --partition=parallel
#SBATCH --mem=30G
# The first derivatives of CCSDT(Q) before and after a change, one after the
# other on the same node (bench/profile_q_gradient.py of each tree): H2O /
# cc-pVTZ as spin orbitals and HBr / cc-pVTZ-DK with X2C spinors, 28 frozen,
# in both trees; in the new one also HBr / cc-pVQZ-DK and H2O under a tight
# max_memory.
# usage: ZC_BEFORE=<snapshot of the old tree> ZC_AFTER=<snapshot of the new one> \
#        sbatch -o <log> bench/slurm_q_gradient_sectors.sh
# ZC_ONLY: the parts to run, of "before after qz tight" (default: all)
# ZC_TIGHT: max_memory in MB of the tight run (default 4000)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-PASSIVE}
export ZCCSDT_TMPDIR=$(mktemp -d /tmp/zccsdt_qgrad_XXXXXX)
export PYSCF_TMPDIR=$ZCCSDT_TMPDIR
trap 'rm -rf "$ZCCSDT_TMPDIR"' EXIT
ONLY=${ZC_ONLY:-before after qz tight}
BASE_PYTHONPATH=$PYTHONPATH
echo "host $(hostname)  job $SLURM_JOB_ID  $(date)  load $(cut -d' ' -f1-3 /proc/loadavg)"
lscpu | grep 'Model name'

run() {     # tree, max_memory, arguments of the script
  local root=$1 mem=$2
  shift 2
  echo "=== $(basename $root): $*  (max_memory $mem MB)  load $(cut -d' ' -f1-3 /proc/loadavg)  $(date)"
  ( cd $root && PYTHONPATH=$root/src:$HOME/apps/socutils-git-env:$BASE_PYTHONPATH ZC_MAXMEM=$mem \
      /usr/bin/time -f "elapsed %e s, maximum resident %M kB" python -u bench/profile_q_gradient.py "$@" 2>&1 \
      | grep -v "plugins found\|PYSCF_EXT_PATH\|^/home/" )
}

if [[ " $ONLY " == *" before "* ]]; then
  run $ZC_BEFORE 26000 h2o cc-pvtz ghf
  run $ZC_BEFORE 26000 hbr cc-pvtz-dk x2c 28
fi
if [[ " $ONLY " == *" after "* ]]; then
  run $ZC_AFTER 26000 h2o cc-pvtz ghf
  run $ZC_AFTER 26000 hbr cc-pvtz-dk x2c 28
fi
if [[ " $ONLY " == *" qz "* ]]; then
  run $ZC_AFTER 26000 hbr cc-pvqz-dk x2c 28
fi
if [[ " $ONLY " == *" tight "* ]]; then
  run $ZC_AFTER ${ZC_TIGHT:-4000} h2o cc-pvtz ghf
fi
echo "done $(date)  load $(cut -d' ' -f1-3 /proc/loadavg)"
