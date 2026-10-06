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
#   HV_MULT=2: a doublet (CFOUR: MULT=2, REFERENCE=UHF; zccsdt: the spin of
#     bench/spinors.py).  CFOUR's SCF then starts from the density of the
#     SCF of zccsdt with the spin on the axis of the molecule (drguess,
#     diguess): its own guess ends in a state with the spin off the axis,
#     which with spin-orbit coupling is not a solution of the SCF - it does
#     not converge, and the relaxed density of such a state means nothing
#     (BaF: 23 degrees off after 600 cycles).  HV_GUESS=0: CFOUR's own
#     guess; zccsdt then runs twice, in CFOUR's state and on the axis.
#   HV_SOCUTILS: the directory with socutils-git as 'socutils' (default
#     ~/apps/socutils-git-env; it has prop/cfour.py, which reads CFOUR's
#     operators). With CFOUR and the properties, all one-electron properties
#     of xsox2cprop are compared: CFOUR's operators with the densities of
#     zccsdt (bench/cfour/props.py), zccsdt in CFOUR's frame.
#   HV_FF_OPERATOR: matrices of CFOUR for the finite differences in the
#     place of the dipole operator, each with its step: 'UTM10Z:1e-3
#     SOX2CEFD:1e-6' (the magnetic moment along z, the effective electric
#     field: operators that are odd under time reversal, for a doublet)
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
scr=/tmp/zccsdt_heavy_${SLURM_JOB_ID:-local$$}
keep=/home/xwang405/.claude/jobs/7ba4fcf7/tmp/heavy/${mol}_${label}_${SLURM_JOB_ID:-local$$}
mkdir -p $scr $keep
socutils=${HV_SOCUTILS:-/home/xwang405/apps/socutils-git-env}
compare=0
if [ "${HV_CFOUR:-1}" != 0 ]; then
  post=
  if [ $props = 1 ]; then
    compare=1
    post="PYTHONPATH=$socutils:$root/src:\$PYTHONPATH python $root/bench/cfour/props.py pack $mol $basis $scr/cfour_ops.npz $keep/geometry.txt"
  fi
  guess=
  [ "${HV_MULT:-1}" != 1 ] && [ "${HV_GUESS:-1}" = 1 ] && \
    guess="ZC_NUCMOD=G ZC_HAM=$ham ZC_SCF=socutils ZC_AXIAL=1 PYTHONPATH=$socutils:$root/src:\$PYTHONPATH python $root/bench/cfour/props.py guess $mol $basis $keep/geometry.txt"
  ZC_ROOT=$root ZC_NITER=0 CFOUR_PYBASIS=$basis CFOUR_SCF_CONV=10 CFOUR_CONV=9 \
    CFOUR_MAXCYC=100 CFOUR_MEM_GB=${HV_CFOUR_GB:-24} CFOUR_EXTRA="$extra" \
    CFOUR_MULT=${HV_MULT:-1} CFOUR_KEEP=$keep CFOUR_POST="$post" CFOUR_GUESS="$guess" \
    CFOUR_SECTIONS="$sections" CFOUR_PROGS="$progs" bash $root/bench/cfour/slurm_x2ccc.sh $mol:$label
  # zccsdt in the frame of CFOUR: its operators are in it
  [ -s $keep/geometry.txt ] && export ZC_GEOMETRY=$keep/geometry.txt
fi
export PYSCF_TMPDIR=$scr OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-PASSIVE}
export ZC_NUCMOD=G ZC_HAM=$ham ZC_SCF=socutils ZC_FROZEN=$((2 * ncore)) ZC_MAXMEM=${HV_MAXMEM:-24000}
[ -n "$HV_VIRCUT" ] && export ZC_VIRCUT=$HV_VIRCUT
cd $root
noise="plugins found\|PYSCF_EXT_PATH\|^/home/\|new_energy_algo\|nuclear model\|^True$\|^False$\|DHF energy\|fractional occupation\|^$"
export PYTHONPATH=$socutils:src:$PYTHONPATH
[ $props = 1 ] && export ZC_DENSITIES=$scr/zccsdt_densities.npz

zccsdt() {      # $1: file for the output; $2: 0 without the (T) energy alone
  ZC_CONV=1e-11 ZC_SCF_GRAD=1e-9 ZC_RELAXED=$props ZC_T_ENERGY=$2 \
    /usr/bin/time -f "wall %e s  max RSS %M KB" python -u bench/profile_t_relaxed.py $mol $basis 2>&1 \
    | grep --line-buffered -v "$noise\|^cycle = " | tee $keep/$1
  if [ $compare = 1 ] && [ -f $scr/cfour_ops.npz ] && [ -f $scr/zccsdt_densities.npz ]; then
    echo "=== the properties of xsox2cprop: CFOUR's operators, the densities of zccsdt  $(date)"
    python bench/cfour/props.py compare $scr/cfour_ops.npz $scr/zccsdt_densities.npz $keep/xsox2cprop.out \
      2>&1 | grep -v "$noise" | tee $keep/props_$1
  fi
}

if [ "${HV_MULT:-1}" != 1 ] && [ -f $scr/cfour_ops.npz ] && [ "${HV_GUESS:-1}" = 1 ]; then
  # a doublet, CFOUR's SCF started on the axis: from its density, the same
  # Kramers partner
  export ZC_AXIAL=1
  echo "=== zccsdt $mol/$basis $ham, from the density of CFOUR's SCF  $(date)  scratch $scr"
  ZC_DM0=$scr/cfour_ops.npz:SCFDM zccsdt zccsdt.out ${HV_T_ENERGY:-1}
elif [ "${HV_MULT:-1}" != 1 ] && [ -f $scr/cfour_ops.npz ]; then
  # CFOUR's own guess: first its state, from the density of its SCF, then
  # the state with the spin on the axis of the molecule, which is the one a
  # property that is odd under time reversal is defined for and the one the
  # finite differences need
  echo "=== zccsdt $mol/$basis $ham, from the density of CFOUR's SCF  $(date)  scratch $scr"
  ZC_DM0=$scr/cfour_ops.npz:SCFDM zccsdt zccsdt_cfour_state.out ${HV_T_ENERGY:-1}
  export ZC_AXIAL=1
  echo "=== zccsdt $mol/$basis $ham, the spin on the axis  $(date)"
  zccsdt zccsdt.out 0
else
  echo "=== zccsdt $mol/$basis $ham  $(date)  scratch $scr"
  zccsdt zccsdt.out ${HV_T_ENERGY:-1}
fi
if [ $props = 1 ] && [ "${HV_FF:-1}" != 0 ]; then
  echo "=== finite differences of the energy  $(date)"
  if [ -n "$HV_FF_OPERATOR" ]; then
    for item in $HV_FF_OPERATOR; do
      ZC_FF_OPERATOR=$scr/cfour_ops.npz:${item%%:*} /usr/bin/time -f "wall %e s  max RSS %M KB" \
        python -u bench/check_relaxed_ff.py $mol $basis ${item##*:} 2>&1 \
        | grep --line-buffered -v "$noise" | tee -a $keep/ff.out
    done
  else
    analytic=$(grep -o "CCSD(T) relaxed [0-9.-]*" $keep/zccsdt.out | awk '{print $3}')
    ZC_ANALYTIC=$analytic /usr/bin/time -f "wall %e s  max RSS %M KB" \
      python -u bench/check_relaxed_ff.py $mol $basis 2>&1 \
      | grep --line-buffered -v "$noise" | tee $keep/ff.out
  fi
fi
df -h $scr | tail -n 1
# (HV_KEEP_OPS=1: the operators and the densities kept, for small molecules)
[ "${HV_KEEP_OPS:-0}" = 1 ] && cp $scr/cfour_ops.npz $scr/zccsdt_densities.npz $keep/
rm -f $scr/cfour_ops.npz $scr/zccsdt_densities.npz
rmdir $scr
echo "done $(date)"
