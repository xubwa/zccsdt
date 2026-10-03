#!/bin/bash -l
#SBATCH --job-name=x2ccc
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -A lcheng24
#SBATCH --partition=parallel
#SBATCH --mem=30G
# CFOUR's two-component CCSD(T) (xx2cccsd) and zccsdt's spinor CCSD on the
# same molecules, one after the other on the same node and cores.
# usage: sbatch -o bench/logs/%x-%j.out bench/cfour/slurm_x2ccc.sh MOL:BASIS [MOL:BASIS ...]
#   MOL: h2o | n2 | c6h6   BASIS: CFOUR name (PVDZ, PVTZ, PVQZ); zccsdt uses cc-pVXZ
# CFOUR: RELATIVISTIC=X2C1E, X2CKEY=SOATMX, ABCDTYPE=AOBASIS, all electrons,
#   C1, run as xjoda; xsox2catm; xsox2chf; xx2cvtran; xx2cccsd (no xcfour driver)
#   CFOUR_MEM_GB: MEMORY_SIZE (default 24); CFOUR_MAXCYC (default 50);
#   CFOUR_CONV: CC_CONV, max residual element (default 7)
# zccsdt: X2C-1e from PySCF (same orbital counts, not the same Hamiltonian),
#   bench/profile_iter.py with ZC_METHOD=ccsd; ZC_NITER timed iterations
#   (default 3; 0 skips zccsdt); ZC_ROOT: snapshot of the repository
# The partition allows 4000 MB per core: --mem=30G keeps the allocation at the
# 8 requested cores (a larger --mem silently adds cores).
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
root=${ZC_ROOT:-/home/xwang405/apps/zccsdt}
cfour=${CFOUR_BIN:-/home/xwang405/apps/cfour-x2cso/bin}
echo "host $(hostname)  $(date)  threads $OMP_NUM_THREADS  root $root"
lscpu | grep 'Model name'

geometry() {
  case $1 in
    h2o) echo "O 0.0 0.0 0.0
H 0.0 0.757 0.587
H 0.0 -0.757 0.587";;
    n2) echo "N 0.0 0.0 0.0
N 0.0 0.0 1.0977";;
    c6h6) echo "C 1.397000 0.000000 0.000000
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
}

for job in "$@"; do
  mol=${job%%:*}; basis=${job##*:}
  tag=${mol}_${basis}_x2c_${SLURM_JOB_ID:-local$$}
  # CFOUR work files on node-local disk (the home quota is small); only the
  # outputs are kept
  keep=/home/xwang405/.claude/jobs/7ba4fcf7/tmp/cfour_bench/$tag
  run=/tmp/zccsdt_cfour_$tag
  mkdir -p $run $keep && cd $run
  cp ${cfour%/bin}/basis/GENBAS GENBAS
  geom=$(geometry $mol)
  # BASIS=SPECIAL with one line per atom: xsox2chf reads the basis from ZMAT
  basislines=$(echo "$geom" | awk -v b=$basis '{print $1 ":" b}')
  cat > ZMAT <<ZEOF
$mol X2C-CCSD(T) $basis
$geom

*CFOUR(CALC=CCSD(T),BASIS=SPECIAL
SYM=OFF,COORDINATES=CARTESIAN,UNITS=ANGSTROM,CHARGE=0,MULT=1
RELATIVISTIC=X2C1E,X2CKEY=SOATMX,ABCDTYPE=AOBASIS
SCF_CONV=10,SCF_MAXCYC=200
CC_CONV=${CFOUR_CONV:-7},CC_MAXCYC=${CFOUR_MAXCYC:-50}
MEMORY_SIZE=${CFOUR_MEM_GB:-24},MEM_UNIT=GB)

$basislines

ZEOF
  echo "=== CFOUR $mol/$basis  $(date)  run dir $run (kept: $keep)"
  for p in xjoda xsox2catm xsox2chf xx2cvtran xx2cccsd; do
    /usr/bin/time -f "$p wall %e s  user %U s  max RSS %M KB" $cfour/$p > $p.out 2> $p.time
    status=$?
    tail -n 1 $p.time
    cp $p.out $p.time $keep/
    if [ $status -ne 0 ] || grep -q "Something very wrong" $p.out; then
      echo "$p FAILED"; tail -n 20 $p.out; break
    fi
  done
  cp ZMAT $keep/
  python $root/bench/cfour/x2ccc_summary.py $keep/xx2cccsd.out
  cd / && rm -rf $run
  if [ "${ZC_NITER:-3}" != 0 ]; then
    pybasis=cc-$(echo $basis | tr 'A-Z' 'a-z')
    echo "=== zccsdt CCSD $mol/$pybasis x2c  $(date)"
    (cd $root && ZC_METHOD=ccsd ZC_PACKED=1 PYTHONPATH=src:$PYTHONPATH OMP_WAIT_POLICY=PASSIVE \
       /usr/bin/time -f "zccsdt wall %e s  max RSS %M KB" \
       python bench/profile_iter.py $mol $pybasis x2c ${ZC_NITER:-3}) 2>&1 | grep -v "^cycle"
  fi
done
echo "done $(date)"
