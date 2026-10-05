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
#   CFOUR_CALC: CCSD(T) (default) or CCSD; CFOUR_MEM_GB: MEMORY_SIZE (default 24);
#   CFOUR_MAXCYC (default 50); CFOUR_PROGS: the programs to run (default: all
#   five; 'xjoda xsox2catm xsox2chf xx2cvtran' stops after the integrals;
#   with 'xx2chbar xx2clcc' after xx2cccsd also Hbar and the Lambda equations);
#   CFOUR_CONV: CC_CONV, max residual element (default 7; the Lambda equations
#   use it too); CFOUR_SCF_CONV: SCF_CONV, max change of the density (default
#   10); the Z-vector equations: LINEQ_CONV (default 7) through CFOUR_EXTRA
#   CFOUR_EXTRA: more keywords, comma separated (PROPS=FIRST_ORDER: xx2cccsd then
#   also makes the (T) terms of the Lambda equations and of the densities, and
#   'xx2chbar xx2clcc xx2cccdenocbp xx2czvec xsox2cprop' give the CCSD(T) dipole)
#   CFOUR_TAG: suffix of the directory of the kept outputs (two runs of a
#   molecule in one job); CFOUR_KEEP: that directory itself
#   CFOUR_MULT: the multiplicity (default 1; 2: REFERENCE=UHF, and the SCF
#   with the damping an open shell needs to converge)
#   CFOUR_POST: a command run in the directory of the CFOUR run after its
#   programs, before the directory is removed (bench/cfour/props.py pack)
#   CFOUR_PYBASIS: a PySCF basis name, or 'heavy,light' (bench/spinors.py):
#   the geometry (any molecule of bench/spinors.py) and the basis then come
#   from PySCF's data (bench/cfour/genbas.py writes the GENBAS entries; the
#   geometry in bohr, as PySCF converts it), and BASIS in MOL:BASIS is only
#   a label
#   CFOUR_SECTIONS: sections of the ZMAT after the basis, with \n for the
#   line ends ('%amfiMethod*\n0\n1\n1\n0': xx2camf, to be named in
#   CFOUR_PROGS after xjoda, then makes the atomic mean-field spin-orbit
#   integrals - without xx2camf the Hamiltonian is X2C-1e; '%vircut\n100.0':
#   no virtual spinors above 100 Hartree; frozen core: DROPMO=1>n in
#   CFOUR_EXTRA, n Kramers pairs)
# zccsdt: X2C-1e from PySCF (the same Hamiltonian as this chain without
#   xx2camf; with BASIS names CFOUR converts the Angstrom of the geometry
#   with its own constant, 2e-7 off PySCF's: CFOUR_PYBASIS for comparisons),
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
  tag=${mol}_${basis}_x2c_${SLURM_JOB_ID:-local$$}${CFOUR_TAG:+_$CFOUR_TAG}
  # CFOUR work files on node-local disk (the home quota is small); only the
  # outputs are kept
  keep=${CFOUR_KEEP:-/home/xwang405/.claude/jobs/7ba4fcf7/tmp/cfour_bench/$tag}
  run=/tmp/zccsdt_cfour_$tag
  mkdir -p $run $keep && cd $run
  cp ${cfour%/bin}/basis/GENBAS GENBAS
  # BASIS=SPECIAL with one line per atom: xsox2chf reads the basis from ZMAT
  if [ -n "$CFOUR_PYBASIS" ]; then
    (cd $root && PYTHONPATH=src:$PYTHONPATH python bench/cfour/genbas.py $mol "$CFOUR_PYBASIS" \
       $run/GENBAS $run/geom.txt $run/basis.txt) 2>&1 | grep -v "plugins found\|PYSCF_EXT_PATH\|^/home/"
    geom=$(cat geom.txt); basislines=$(cat basis.txt); units=BOHR
  else
    geom=$(geometry $mol); units=ANGSTROM
    basislines=$(echo "$geom" | awk -v b=$basis '{print $1 ":" b}')
  fi
  # (a line of the ZMAT has 80 columns: the extra keywords one per line)
  extra=; [ -n "$CFOUR_EXTRA" ] && extra=$'\n'$(echo "$CFOUR_EXTRA" | tr ',' '\n')
  scf="SCF_CONV=${CFOUR_SCF_CONV:-10},SCF_MAXCYC=200"
  [ "${CFOUR_MULT:-1}" != 1 ] && \
    scf="REFERENCE=UHF,SCF_CONV=${CFOUR_SCF_CONV:-10}"$'\n'"SCF_MAXCYC=600,SCF_DAMPING=500,SCF_EXPSTART=30"
  cat > ZMAT <<ZEOF
$mol X2C-${CFOUR_CALC:-CCSD(T)} $basis
$geom

*CFOUR(CALC=${CFOUR_CALC:-CCSD(T)},BASIS=SPECIAL
SYM=OFF,COORDINATES=CARTESIAN,UNITS=$units,CHARGE=0,MULT=${CFOUR_MULT:-1}
RELATIVISTIC=X2C1E,X2CKEY=SOATMX,ABCDTYPE=AOBASIS
$scf
CC_CONV=${CFOUR_CONV:-7},CC_MAXCYC=${CFOUR_MAXCYC:-50}$extra
MEMORY_SIZE=${CFOUR_MEM_GB:-24},MEM_UNIT=GB)

$basislines

$(printf '%b' "${CFOUR_SECTIONS:-}")

ZEOF
  echo "=== CFOUR $mol/$basis  $(date)  run dir $run (kept: $keep)"
  for p in ${CFOUR_PROGS:-xjoda xsox2catm xsox2chf xx2cvtran xx2cccsd}; do
    # xx2camf is C++: where the libstdc++ of the environment is too old for
    # it, the one of the Python installation is loaded first
    pre=
    [ -n "$(ldd $cfour/$p 2>&1 | grep 'GLIBCXX.*not found')" ] && \
      pre="env LD_PRELOAD=${CFOUR_STDCXX:-/home/xwang405/apps/conda/lib/libstdc++.so.6}"
    /usr/bin/time -f "$p wall %e s  user %U s  max RSS %M KB" $pre $cfour/$p > $p.out 2> $p.time
    status=$?
    tail -n 1 $p.time
    cp $p.out $p.time $keep/
    if [ $status -ne 0 ] || grep -q "Something very wrong" $p.out; then
      echo "$p FAILED"; tail -n 20 $p.out; break
    fi
  done
  cp ZMAT MOL $keep/
  [ -n "$CFOUR_POST" ] && (eval "$CFOUR_POST") 2>&1 | grep -v "plugins found\|PYSCF_EXT_PATH\|^/home/"
  grep -E "Integral transformation Elapsed|MO Integral sorting Elapsed|AO Integral Elapsed" \
    $keep/xx2cvtran.out | sed 's/^ */  xx2cvtran: /; s/  */ /g'
  [ -f $keep/xsox2chf.out ] && grep "Final SCF Energy" $keep/xsox2chf.out | tail -n 1 | awk '{print "  xsox2chf: E(SCF) " $5 " after " $4 " cycles"}'
  [ -f $keep/xx2cccsd.out ] && python $root/bench/cfour/x2ccc_summary.py $keep/xx2cccsd.out
  # the Lambda equations (CFOUR_PROGS with xx2chbar xx2clcc after xx2cccsd)
  [ -f $keep/xx2clcc.out ] && echo "  xx2clcc: $(grep -c 'CCSD cycle  *[0-9]' $keep/xx2clcc.out) Lambda cycles"
  if [ -f $keep/xsox2cprop.out ]; then
    for kind in "Hartree-Fock Properties" "Coupled-cluster Properties" "Unrelaxed Coupled-cl[a-z]* Properties"; do
      grep -A 11 "^ \*  *$kind" $keep/xsox2cprop.out | grep "mu z" | tail -n 1 | sed "s/^ */  xsox2cprop: ${kind%% Prop*} dipole, total (a.u., Debye) /"
    done
  fi
  cd / && rm -rf $run
  if [ "${ZC_NITER:-3}" != 0 ]; then
    pybasis=cc-$(echo $basis | tr 'A-Z' 'a-z')
    echo "=== zccsdt CCSD $mol/$pybasis x2c  $(date)"
    (cd $root && ZC_METHOD=ccsd ZC_PACKED=1 PYTHONPATH=src:$PYTHONPATH OMP_WAIT_POLICY=PASSIVE \
       /usr/bin/time -f "zccsdt wall %e s  max RSS %M KB" \
       python -u bench/profile_iter.py $mol $pybasis x2c ${ZC_NITER:-3}) 2>&1 | grep --line-buffered -v "^cycle"
  fi
done
echo "done $(date)"
