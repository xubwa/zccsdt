#!/bin/bash
# MRCC, nonrelativistic, with its own integrals and SCF, in a scratch
# directory of the node - the reference for molecules where MRCC can use spin
# and point-group symmetry (zccsdt runs an RHF reference as spin orbitals):
#
#   usage: bench/mrcc_nr.sh MINP [TAG]      e.g. bench/refs/mrcc/h2o_ccpvdz/MINP.tqp.sym.bohr
#
# GENBAS of bench/refs goes next to the input.  Prints the group, the numbers
# of excitations, the energies and the wall time of the iterations; the
# output is kept as bench/logs/TAG-<job>.out.
set -e
MRCCDIR=${MRCCDIR:-/home/czhan119/code/cfour-git/bin}
ROOT=${ZC_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
LOGS=${ZC_LOGS:-/home/xwang405/apps/zccsdt/bench/logs}
export PATH=$MRCCDIR:$PATH
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${OMP_NUM_THREADS:-8}
minp=$(readlink -f "$1"); tag=${2:-mrcc_nr}
run=${ZCCSDT_TMPDIR:-/tmp/zccsdt_mrcc_nr_$$}/$tag
mkdir -p $run
cp $ROOT/bench/refs/GENBAS $run/GENBAS
cp $minp $run/MINP
cd $run
t0=$(date +%s)
dmrcc > out.dmrcc 2>&1 || true
echo "== $tag: $(( $(date +%s) - t0 )) s in all"
grep -E "Computational point group|Number of irreps|Number of CPUs|Allocation of|Number of [0-9]-fold|Total number of determinants|Iteration has converged|Total .* energy|FINAL HARTREE|Perturbative" out.dmrcc || true
echo "-- wall time [min] at the first, second and last iteration, and at the end:"
grep -E "Wall time" out.dmrcc | sed -n '1p;2p' ; grep -B2 "Iteration has converged" out.dmrcc | grep "Wall time" || true
grep -E "Wall time" out.dmrcc | tail -1
cp out.dmrcc $LOGS/$tag-${SLURM_JOB_ID:-local}.out 2>/dev/null || true
