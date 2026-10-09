#!/bin/bash
# Excited states of MRCC (LR-CC = EOM-CC energies) through its interface to
# DIRAC, on MO integrals written by zccsdt.dirac - the reference of
# EOM-CCSD / EOM-CCSDT on spinors, on the same integrals as zccsdt:
#
#   python bench/dirac/write_integrals.py x2c 'Br 0 0 0; H 0 0 1.4145' cc-pvdz-dk DIR 28 16
#   bench/dirac/mrcc_rel.sh DIR 'CCSDT'          # makes DIR/full (complete MDCINT, fort.55, MINP)
#   bench/dirac/mrcc_eom.sh DIR 'CCSDT' 4        # ground state + 3 excited states
#
# nstate counts the ground state.  MRCC looks for the states one after the
# other from its own guesses: they need not be the lowest ones.  Results:
# DIR/full/eom_<calc>_<nstate>/out.dmrcc ("Total LR-CC.. energy").
set -e
MRCCDIR=${MRCCDIR:-/home/czhan119/code/cfour-git/bin}
export PATH=$MRCCDIR:$PATH
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${OMP_NUM_THREADS:-8}
src=$(readlink -f "$1"); calc=${2:-CCSDT}; ns=${3:-4}
full=$src/full
[ -f $full/fort.55 ] || { echo "run bench/dirac/mrcc_rel.sh $1 'CCSDT' first"; exit 1; }
tag=$(echo "$calc" | tr 'A-Z()' 'a-z__' | sed 's/_$//')
R=$full/eom_${tag}_$ns
mkdir -p $R && cp $full/MRCONEE $full/MDCINT $full/fort.55 $R/
printf 'iface=dirac\ncalc=%s\nmem=%s\ncctol=11\nnstate=%s\nrefdet=vector\n%s\n' "$calc" "${MRCC_MEM:-4GB}" "$ns" "$(sed -n 5p $full/MINP)" > $R/MINP
cd $R
t0=$(date +%s)
dmrcc > out.dmrcc 2>&1 || true
echo "== $calc, nstate=$ns: $(( $(date +%s) - t0 )) s"
grep -E "Total .*CC.* energy|Excitation energy \[eV|Iteration has converged" out.dmrcc
