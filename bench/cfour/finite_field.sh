#!/bin/bash
# CFOUR's two-component CCSD(T) energies in a finite electric field along z,
# and their central differences: the relaxed dipole moment as the derivative
# of CFOUR's own energy, to hold against what its property programs
# (xx2cccdenocbp, xx2czvec, xsox2cprop) print - with dropped virtual spinors
# (%vircut) the two differ once most of them are dropped (docs/benchmarks.md,
# "Frozen virtual spinors").
# usage: bash bench/cfour/finite_field.sh MOL BASIS NCORE [VIRCUT] [AMF]
#   MOL, BASIS: as bench/slurm_heavy.sh (h2o unc-ccpvdz; hi dyallv2z,unc-ccpvdz)
#   NCORE: frozen Kramers pairs (DROPMO=1>NCORE; 0: none)
#   VIRCUT: %vircut in Hartree (default: no virtual spinors dropped)
#   AMF: the four flags of %amfiMethod*, e.g. '0 1 1 0' (default: X2C-1e)
# ZFIELD=n adds -n x 1e-7 times the dipole operator SOX2CDPZ to the
# one-electron Hamiltonian of xsox2chf; the fields here are 0.002 and 0.004
# a.u.  The derivative of the total energy is the total dipole moment, and
# with the nuclear part - the derivative of E(SCF) plus the SCF value that
# xsox2cprop prints for the electrons - the electronic part follows.
# Runs where it is called (a few threads for a small molecule; a batch job
# for a large one): OMP_NUM_THREADS, ZC_ROOT (snapshot of the repository),
# FF_KEEP (directory for the outputs, default ./cfour_ff).
mol=$1; basis=$2; ncore=$3; cut=$4; amf=${5:-none}
root=${ZC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
keep0=${FF_KEEP:-$PWD/cfour_ff}
label=$(echo $basis | tr 'a-z,' 'A-Z_' | tr -cd 'A-Z0-9_')
progs="xjoda"; sections=
if [ "$amf" != none ]; then
  set -- $amf
  progs="$progs xx2camf"; sections="%amfiMethod*\n$1\n$2\n$3\n$4\n\n"
fi
[ -n "$cut" ] && sections="$sections%vircut\n$cut\n\n"
progs="$progs xsox2catm xsox2chf xx2cvtran xx2cccsd"
mkdir -p $keep0
for fld in 20000 -20000 40000 -40000; do
  extra="NUC_MODEL=GAUSSIAN,ZFIELD=$fld"
  [ "$ncore" -gt 0 ] && extra="$extra,DROPMO=1>$ncore"
  keep=$keep0/${mol}_${label}_cut${cut:-none}_f$fld
  mkdir -p $keep
  ZC_ROOT=$root ZC_NITER=0 CFOUR_PYBASIS=$basis CFOUR_SCF_CONV=11 CFOUR_CONV=10 CFOUR_MAXCYC=200 \
    CFOUR_MEM_GB=${CFOUR_MEM_GB:-8} CFOUR_EXTRA="$extra" CFOUR_KEEP=$keep CFOUR_TAG=f$fld \
    CFOUR_SECTIONS="$sections" CFOUR_PROGS="$progs" bash $root/bench/cfour/slurm_x2ccc.sh $mol:$label \
    > $keep/run.log 2>&1
  echo "ZFIELD $fld: $(grep -h 'E_corr(CCSD)' $keep/run.log)  $(grep -h 'E(SCF)' $keep/run.log | head -n 1)"
done | tee $keep0/${mol}_${label}_cut${cut:-none}.txt
python - $keep0/${mol}_${label}_cut${cut:-none}.txt <<'PY'
import re
import sys
E = {}
for line in open(sys.argv[1]):
    num = r'(-?\d+\.\d+(?:[EeDd][-+]?\d+)?)'
    m = re.search(r'ZFIELD (-?\d+): E_corr\(CCSD\) %s; \(T\) correction %s; CCSD\(T\) energy\s+%s.*E\(SCF\) %s'
                  % (num, num, num, num), line)
    if m:
        f = int(m.group(1))
        ec, et, etot, escf = (float(x.replace('D', 'E')) for x in m.groups()[1:])
        E[f] = {'SCF': escf, 'CCSD': escf + ec, 'CCSD(T)': etot}
eps = 2e-3
print('central differences of the energy, Richardson (a.u.): the total dipole moment along z')
for k in ('SCF', 'CCSD', 'CCSD(T)'):
    d1 = (E[20000][k] - E[-20000][k]) / (2 * eps)
    d2 = (E[40000][k] - E[-40000][k]) / (4 * eps)
    print('  %-8s %.9f   (0.002: %.9f, 0.004: %.9f)' % (k, (4 * d1 - d2) / 3, d1, d2))
print('the electronic part: minus the nuclear one, which is the SCF line above plus the SCF '
      'value of SOX2CDPZ that xsox2cprop prints')
PY
