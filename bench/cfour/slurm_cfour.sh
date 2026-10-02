#!/bin/bash -l
#SBATCH --job-name=cfour
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -A lcheng24
#SBATCH --partition=parallel
#SBATCH --mem=30G
# usage: sbatch bench/cfour/slurm_cfour.sh MOL BASIS CC_PROG REF MAXCYC
# CFOUR_FC=ON: frozen core (default OFF);  CFOUR_MEM_GB: MEMORY_SIZE (default 24)
# CFOUR_CALC='CCSDT(Q)': calculation level (default CCSDT);  CFOUR_CONV: CC_CONV (default 10)
# CFOUR_SYM=ON: point-group symmetry (default OFF, as zccsdt has none)
# CFOUR_BIN: directory of the CFOUR executables (default: the cfour-x2cso build)
# The partition allows 4000 MB per core: --mem=30G keeps the allocation at the
# 8 requested cores (a larger --mem silently adds cores).
mol=$1; basis=$2; prog=$3; ref=$4; maxcyc=${5:-6}; mult=1
# threads = cores of the allocation (default 8; override with sbatch -c N --mem=...)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PATH=${CFOUR_BIN:-/home/xwang405/apps/cfour-x2cso/bin}:$PATH
tag=${mol}_${basis}_${prog}_${ref}_${SLURM_JOB_ID:-local$$}
# CFOUR work files go to node-local disk (they reach tens of GB; the home
# quota is small); only out / time.txt / ZMAT are kept
keep=/home/xwang405/.claude/jobs/7ba4fcf7/tmp/cfour_bench/$tag
run=/tmp/zccsdt_cfour_$tag
mkdir -p $run $keep && cd $run
cp /home/xwang405/apps/cfour-x2cso/basis/GENBAS GENBAS
case $mol in
  h2o) geom="O 0.0 0.0 0.0
H 0.0 0.757 0.587
H 0.0 -0.757 0.587";;
  n2) geom="N 0.0 0.0 0.0
N 0.0 0.0 1.0977";;
  nh2) geom="N 0.0 0.0 0.0
H 0.0 0.8036 0.6347
H 0.0 -0.8036 0.6347"; mult=2;;
  o2) geom="O 0.0 0.0 0.0
O 0.0 0.0 1.2075"; mult=3;;
  h2s) geom="S 0.0 0.0 0.0
H 0.0 0.9618 0.9272
H 0.0 -0.9618 0.9272";;
  ch2o) geom="C 0.0 0.0 0.0
O 0.0 0.0 1.205
H 0.0 0.9447 -0.5847
H 0.0 -0.9447 -0.5847";;
  c2h4) geom="C 0.000000 0.000000 0.669500
C 0.000000 0.000000 -0.669500
H 0.000000 0.929781 1.232595
H 0.000000 -0.929781 1.232595
H 0.000000 0.929781 -1.232595
H 0.000000 -0.929781 -1.232595";;
  c6h6) geom="C 1.397000 0.000000 0.000000
C 0.698500 1.209837 0.000000
C -0.698500 1.209837 0.000000
C -1.397000 0.000000 0.000000
C -0.698500 -1.209837 0.000000
C 0.698500 -1.209837 0.000000
H 2.481000 0.000000 0.000000
H 1.240500 2.148609 0.000000
H -1.240500 2.148609 0.000000
H -2.481000 0.000000 0.000000
H -1.240500 -2.148609 0.000000
H 1.240500 -2.148609 0.000000";;
esac
cat > ZMAT <<ZEOF
$mol CCSDT $basis
$geom

*CFOUR(CALC=${CFOUR_CALC:-CCSDT},BASIS=$basis,REF=$ref,CC_PROG=$prog,MULT=$mult
COORDINATES=CARTESIAN,UNITS=ANGSTROM,SYMMETRY=${CFOUR_SYM:-OFF}
FROZEN_CORE=${CFOUR_FC:-OFF},SCF_CONV=10,CC_CONV=${CFOUR_CONV:-10},CC_MAXCYC=$maxcyc
SCF_MAXCYC=200,MEMORY_SIZE=${CFOUR_MEM_GB:-24},MEM_UNIT=GB)

ZEOF
echo "host $(hostname) $(date) run dir $run (kept: $keep)"
/usr/bin/time -v xcfour > out 2> time.txt
cp out time.txt ZMAT $keep/
cd / && rm -rf $run
grep -E "executable x(ncc|ecc|vcc|vtran|intprc) finished|Walltime|It\. |\(Q\)" $keep/out | tail -40
grep -E "Maximum resident" $keep/time.txt
echo "done $(date)"
