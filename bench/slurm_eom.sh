#!/bin/bash -l
#SBATCH --job-name=eom
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -A lcheng24
#SBATCH --partition=parallel
#SBATCH --mem=30G
# EOM-EE-CCSDT: MRCC first, then zccsdt, on the same node and cores.
#   usage: sbatch -o bench/logs/%x-%j.out bench/slurm_eom.sh
#   ZC_ROOT: snapshot of the repository (with src/zccsdt/lib/*.so)
#   EOM_PARTS: which parts run (default "mrcc_h2o zc_h2o mrcc_hbr zc_hbr")
#
# H2O / cc-pVTZ, all electrons.  MRCC at its best for a closed shell: its own
#   RHF and integrals, C2v, spin adaptation; the lowest singlet and the lowest
#   triplet (irrep "B2" of its numbering; nsing=2, ntrip=1).  zccsdt: the same
#   basis (CFOUR's GENBAS, O:PVTZ / H:PVTZ) and geometry, RHF as GHF spin
#   orbitals without any symmetry; four roots = the triplet and the singlet.
# HBr / cc-pVTZ-DK, X2C spinors, 28 frozen, 8 + 78 active.  MRCC through its
#   interface to DIRAC on the integrals written by zccsdt.dirac (no spin
#   adaptation; real integrals), nstate=4.  zccsdt through the front end: in
#   the blocks of the symmetry labels (the totally symmetric states) and on
#   the packed block (all states).
MRCCDIR=${MRCCDIR:-/home/czhan119/code/cfour-git/bin}
GENBAS=${EOM_GENBAS:-/home/czhan119/code/cfour-git/basis/GENBAS}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
cd ${ZC_ROOT:-/home/xwang405/apps/zccsdt}
ROOT=$(pwd)
export PYTHONPATH=$ROOT/src:$HOME/apps/socutils-git-env:$PYTHONPATH
S=/tmp/zccsdt_eom_${SLURM_JOB_ID:-$$}
mkdir -p $S
export ZCCSDT_TMPDIR=$S PYSCF_TMPDIR=$S
parts=${EOM_PARTS:-"mrcc_h2o zc_h2o mrcc_hbr zc_hbr"}
echo "host $(hostname)  $(date)  threads $OMP_NUM_THREADS"
lscpu | grep 'Model name'
echo "cpus: $(taskset -cp $$)"
filter() { grep -v -i -E "plugins found|site-packages/pyscf|shciscf|PYSCF_EXT_PATH"; }

for part in $parts; do
case $part in
mrcc_h2o)
  echo "=== MRCC H2O / cc-pVTZ (C2v, RHF), CCSDT + lowest singlet and triplet  $(date)"
  R=$S/mrcc_h2o; mkdir -p $R; cp $GENBAS $R/GENBAS
  cat > $R/MINP <<'EOF'
basis=PVTZ
calc=CCSDT
nsing=2
ntrip=1
symm=B2
core=corr
mem=24GB
scftype=RHF
scftol=12
cctol=8
ccmaxit=200
unit=bohr
geom=xyz
3

O 0.000000000000000 0.000000000000000 0.000000000000000
H 0.000000000000000 1.430522676295752 1.109269235119691
H 0.000000000000000 -1.430522676295752 1.109269235119691
EOF
  (cd $R; t0=$(date +%s); PATH=$MRCCDIR:$PATH dmrcc > out.dmrcc 2>&1; echo "dmrcc: $(( $(date +%s) - t0 )) s in all")
  grep -E "Computational point group|Number of CPUs|OpenMP|Allocation of|Total number of determinants|FINAL HARTREE|Iteration has converged|Total .*CC.* energy|Spin multiplicity:  |Starting right-hand" $R/out.dmrcc
  echo "-- wall time lines of MRCC's iterations (first 4, and every converged state):"
  grep -n -E "Wall time|Iteration has converged" $R/out.dmrcc | sed -n '1,6p'
  grep -B3 "Iteration has converged" $R/out.dmrcc | grep -E "Wall|converged"
  cp $R/out.dmrcc ${EOM_LOGS:-/home/xwang405/apps/zccsdt/bench/logs}/eom_mrcc_h2o-${SLURM_JOB_ID:-local}.out 2>/dev/null
  ;;
zc_h2o)
  echo "=== zccsdt H2O / cc-pVTZ, spin orbitals without symmetry  $(date)"
  EB_GENBAS=$GENBAS python -u bench/eom_bench.py h2o 4 2>&1 | filter
  ;;
mrcc_hbr)
  echo "=== MRCC (iface=dirac) HBr / cc-pVTZ-DK X2C, 8 + 78 spinors, on integrals of zccsdt.dirac  $(date)"
  D=$S/hbr; mkdir -p $D
  python -u bench/dirac/write_integrals.py x2c 'Br 0 0 0; H 0 0 1.4145' cc-pvtz-dk $D 28 2>&1 | filter
  ZC_ROOT=$ROOT MRCC_HF_ONLY=1 MRCC_MEM=24GB bash bench/dirac/mrcc_rel.sh $D 'CCSDT'
  MRCC_MEM=24GB bash bench/dirac/mrcc_eom.sh $D 'CCSDT' 4
  R=$D/full/eom_ccsdt_4
  grep -n -E "Wall time|Iteration has converged" $R/out.dmrcc | sed -n '1,6p'
  grep -B3 "Iteration has converged" $R/out.dmrcc | grep -E "Wall|converged"
  cp $R/out.dmrcc ${EOM_LOGS:-/home/xwang405/apps/zccsdt/bench/logs}/eom_mrcc_hbr-${SLURM_JOB_ID:-local}.out 2>/dev/null
  ;;
zc_hbr)
  echo "=== zccsdt HBr / cc-pVTZ-DK X2C, 28 frozen, label blocks  $(date)"
  python -u bench/eom_bench.py hbr 3 2>&1 | filter
  echo "=== zccsdt HBr, packed block  $(date)"
  python -u bench/eom_bench.py hbr_dense 6 2>&1 | filter
  ;;
esac
done
find "$S" -delete
echo "done $(date)"
