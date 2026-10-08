#!/bin/bash -l
#SBATCH --job-name=qgrad
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -A lcheng24
#SBATCH --partition=parallel
#SBATCH --mem=30G
# The first derivatives of CCSDT(Q): time and memory next to (Q) and to the
# properties of CCSDT (bench/profile_q_gradient.py) for H2O / cc-pVTZ as spin
# orbitals and HBr / cc-pVTZ-DK with X2C spinors, 28 frozen; then the dipole
# moment of CFOUR's ncc for H2O / cc-pVTZ (CALC=CCSDT(Q), PROPS=FIRST_ORDER)
# with the primitives and the geometry of PySCF (bench/cfour/genbas.py).
# usage: ZC_ROOT=<snapshot of the repository> ZC_KEEP=<directory for the CFOUR output> \
#        sbatch -o <log> bench/slurm_q_gradient.sh
# ZC_ONLY: the parts to run, of "h2o hbr cfour" (default: all)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-PASSIVE}
ROOT=${ZC_ROOT:-/home/xwang405/apps/zccsdt}
KEEP=${ZC_KEEP:-$ROOT/bench/logs}
export ZCCSDT_TMPDIR=$(mktemp -d /tmp/zccsdt_qgrad_XXXXXX)
export PYSCF_TMPDIR=$ZCCSDT_TMPDIR
trap 'rm -rf "$ZCCSDT_TMPDIR"' EXIT
cd $ROOT
export PYTHONPATH=$ROOT/src:$HOME/apps/socutils-git-env:$PYTHONPATH
echo "host $(hostname)  job $SLURM_JOB_ID  $(date)  load $(cut -d' ' -f1-3 /proc/loadavg)"
lscpu | grep 'Model name'
export ZC_MAXMEM=${ZC_MAXMEM:-26000}
if [[ " ${ZC_ONLY:-h2o hbr cfour} " == *" h2o "* ]]; then
  python -u bench/profile_q_gradient.py h2o cc-pvtz ghf 2>&1 | grep -v "plugins found\|PYSCF_EXT_PATH\|^/home/"
  echo "load $(cut -d' ' -f1-3 /proc/loadavg)  $(date)"
fi
if [[ " ${ZC_ONLY:-h2o hbr cfour} " == *" hbr "* ]]; then
  python -u bench/profile_q_gradient.py hbr cc-pvtz-dk x2c 28 2>&1 | grep -v "plugins found\|PYSCF_EXT_PATH\|^/home/"
  echo "load $(cut -d' ' -f1-3 /proc/loadavg)  $(date)"
fi
if [[ " ${ZC_ONLY:-h2o hbr cfour} " == *" cfour "* ]]; then
  run=$ZCCSDT_TMPDIR/cfour
  mkdir -p $run $KEEP
  python bench/cfour/genbas.py h2o cc-pvtz $run/GENBAS $run/geom.txt $run/basis.txt 2>&1 \
    | grep -v "plugins found\|PYSCF_EXT_PATH\|^/home/"
  cat > $run/ZMAT <<ZEOF
h2o CCSDT(Q) first-order properties, cc-pVTZ of PySCF
$(cat $run/geom.txt)

*CFOUR(CALC=CCSDT(Q),BASIS=SPECIAL,REF=RHF,CC_PROG=NCC
COORDINATES=CARTESIAN,UNITS=BOHR,SYMMETRY=${CFOUR_SYM:-ON}
FROZEN_CORE=OFF,SCF_CONV=11,CC_CONV=10,CC_MAXCYC=200,LINEQ_CONV=10
SCF_MAXCYC=200,MEMORY_SIZE=20,MEM_UNIT=GB
PROPS=FIRST_ORDER)

$(cat $run/basis.txt)

ZEOF
  (
    source /data/lcheng24/intel/oneapi/setvars.sh --force > /dev/null 2>&1
    export PATH=${CFOUR_BIN:-/home/xwang405/apps/cfour-x2cso/bin}:$PATH
    cd $run && /usr/bin/time -v xcfour > out 2> time.txt
  )
  cp $run/out $KEEP/cfour_h2o_ccpvtz_qprops-$SLURM_JOB_ID.out
  cp $run/ZMAT $KEEP/cfour_h2o_ccpvtz_qprops-$SLURM_JOB_ID.ZMAT
  echo "== CFOUR ncc, H2O / cc-pVTZ (SYMMETRY=${CFOUR_SYM:-ON})"
  grep -E "E\(SCF\)=|Total CCSDT|CCSDT\(Q\) corr|iterations converged|executable x(ncc|dens|vscf|vtran|intprc) finished|required" $run/out
  grep -A1 "Components of electric dipole moment" $run/out
  grep -E "Elapsed|Maximum resident" $run/time.txt
fi
echo "done $(date)"
