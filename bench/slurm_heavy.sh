#!/bin/bash -l
#SBATCH --job-name=heavy
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -A lcheng24
#SBATCH --partition=parallel
#SBATCH --mem=30G
# CCSD(T) of a molecule with a heavy atom, as it is run in practice
# (uncontracted basis, Gaussian nuclei, X2CAMF, frozen core): CFOUR's
# two-component coupled cluster first, with the same Hamiltonian, primitives
# and geometry, then zccsdt on the same node and cores, then the finite
# differences of zccsdt's own energy for its relaxed dipole moment.
# usage: sbatch -o bench/logs/%x-%j.out bench/slurm_heavy.sh MOL BASIS NCORE
#   MOL, BASIS: bench/spinors.py, e.g. hi dyallv3z,unc-ccpvtz (heavy,light)
#   NCORE: frozen Kramers pairs (CFOUR: DROPMO=1>NCORE; zccsdt: 2 NCORE spinors)
# environment:
#   HV_AMF: the flags of CFOUR's %amfiMethod - average of configurations,
#     Gaunt, gauge, two-electron picture change -, default '0 1 1 0' (the
#     Breit term in the atomic mean field); 'none': X2C-1e, without xx2camf
#   HV_VIRCUT: virtual spinors above this orbital energy (Hartree) frozen
#     too: the energies only (zccsdt has no relaxed density with them)
#   HV_PROPS=0: the energies only; HV_CFOUR=0: without CFOUR; HV_FF=0:
#     without the finite differences; HV_MAXMEM: max_memory of zccsdt in MB
#     (default 24000; CFOUR's MEMORY_SIZE is HV_CFOUR_GB, default 24);
#   HV_T_ENERGY=0: zccsdt without the run of the (T) energy alone
#   ZC_ROOT: snapshot of the repository
# Thresholds: CFOUR SCF_CONV 10, CC_CONV 9, LINEQ_CONV 9; zccsdt 1e-9 in the
# orbital gradient of the SCF, 1e-11 in the CCSD energy (1e-9 in t, lambda).
# More memory than 30G takes more cores (4000 MB each on this partition):
# sbatch --cpus-per-task=16 --mem=60G with HV_CFOUR_GB=48 HV_MAXMEM=48000.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
root=${ZC_ROOT:-/home/xwang405/apps/zccsdt}
mol=$1; basis=$2; ncore=$3
label=$(echo $basis | tr 'a-z,' 'A-Z_' | tr -cd 'A-Z0-9_')
amf=${HV_AMF:-0 1 1 0}
echo "host $(hostname)  $(date)  threads $OMP_NUM_THREADS  root $root"
lscpu | grep 'Model name'
echo "$mol $basis: $ncore frozen Kramers pairs; atomic mean field: $amf; virtual cutoff: ${HV_VIRCUT:-none}"

props=${HV_PROPS:-1}; [ -n "$HV_VIRCUT" ] && props=0
progs="xjoda"; sections=; ham=x2c1e
if [ "$amf" != none ]; then
  set -- $amf
  progs="$progs xx2camf"; sections="%amfiMethod*\n$1\n$2\n$3\n$4\n\n"; ham=x2camf
  [ $1 = 1 ] && ham=$ham+aoc; [ $2 = 1 ] && ham=$ham+gaunt
  [ $3 = 1 ] && ham=$ham+breit; [ $4 = 1 ] && ham=$ham+pcc
fi
[ -n "$HV_VIRCUT" ] && sections="$sections%vircut\n$HV_VIRCUT\n\n"
progs="$progs xsox2catm xsox2chf xx2cvtran xx2cccsd"
extra="NUC_MODEL=GAUSSIAN,LINEQ_CONV=9"
[ $ncore -gt 0 ] && extra="$extra,DROPMO=1>$ncore"
if [ $props = 1 ]; then
  progs="$progs xx2chbar xx2clcc xx2cccdenocbp xx2czvec xsox2cprop"
  extra="$extra,PROPS=FIRST_ORDER"
fi
if [ "${HV_CFOUR:-1}" != 0 ]; then
  ZC_ROOT=$root ZC_NITER=0 CFOUR_PYBASIS=$basis CFOUR_SCF_CONV=10 CFOUR_CONV=9 \
    CFOUR_MAXCYC=100 CFOUR_MEM_GB=${HV_CFOUR_GB:-24} CFOUR_EXTRA="$extra" \
    CFOUR_SECTIONS="$sections" CFOUR_PROGS="$progs" bash $root/bench/cfour/slurm_x2ccc.sh $mol:$label
fi

scr=/tmp/zccsdt_heavy_${SLURM_JOB_ID:-local$$}
keep=/home/xwang405/.claude/jobs/7ba4fcf7/tmp/heavy/${mol}_${label}_${SLURM_JOB_ID:-local$$}
mkdir -p $scr $keep
export PYSCF_TMPDIR=$scr OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-PASSIVE}
export ZC_NUCMOD=G ZC_HAM=$ham ZC_SCF=socutils ZC_FROZEN=$((2 * ncore)) ZC_MAXMEM=${HV_MAXMEM:-24000}
[ -n "$HV_VIRCUT" ] && export ZC_VIRCUT=$HV_VIRCUT
cd $root
noise="plugins found\|PYSCF_EXT_PATH\|^/home/\|new_energy_algo\|nuclear model\|^True$\|^False$\|DHF energy\|fractional occupation\|^$"
echo "=== zccsdt $mol/$basis $ham  $(date)  scratch $scr"
ZC_CONV=1e-11 ZC_SCF_GRAD=1e-9 ZC_RELAXED=$props ZC_T_ENERGY=${HV_T_ENERGY:-1} PYTHONPATH=src:$PYTHONPATH \
  /usr/bin/time -f "wall %e s  max RSS %M KB" python -u bench/profile_t_relaxed.py $mol $basis 2>&1 \
  | grep --line-buffered -v "$noise\|^cycle = " | tee $keep/zccsdt.out
if [ $props = 1 ] && [ "${HV_FF:-1}" != 0 ]; then
  analytic=$(grep -o "CCSD(T) relaxed [0-9.-]*" $keep/zccsdt.out | awk '{print $3}')
  echo "=== finite differences of the energy  $(date)"
  ZC_ANALYTIC=$analytic PYTHONPATH=src:$PYTHONPATH /usr/bin/time -f "wall %e s  max RSS %M KB" \
    python -u bench/check_relaxed_ff.py $mol $basis 2>&1 | grep --line-buffered -v "$noise" | tee $keep/ff.out
fi
df -h $scr | tail -n 1
rmdir $scr
echo "done $(date)"
