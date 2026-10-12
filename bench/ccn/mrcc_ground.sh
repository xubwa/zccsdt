#!/bin/bash
# MRCC's closed-shell CC3 / CC4 (and the full models) for H2O with its own
# SCF and integrals - the reference of the ground states of docs/ccn.md:
#
#   bench/ccn/mrcc_ground.sh OUTDIR [BASIS ..]       (default: STO-3G 6-31G)
#
# CALCS (default "CC3 CC4 CCSD CCSDT CCSDTQ").  The input is
# bench/refs/mrcc/h2o_631g/MINP.eom.bohr without its excited states: this
# MRCC (release of 2018) has none for the approximate models ("Approximate
# CC methods are not implemented for excited states!"), its calc=CC2 stops
# in the integral program (density fitting), and through iface=dirac CC3 and
# CC4 are refused ("CCn methods are not implemented for ROHF reference!").
W=$(cd "$(dirname "$0")/../.." && pwd)
out=$(readlink -f "$1"); shift
mkdir -p "$out"
export ZC_LOGS=$out
for basis in ${@:-STO-3G 6-31G}; do
  for c in ${CALCS:-CC3 CC4 CCSD CCSDT CCSDTQ}; do
    tag=h2o_$(echo $basis | tr -d '-' | tr 'A-Z' 'a-z')_$(echo $c | tr 'A-Z' 'a-z')
    sed -e "s/^basis=.*/basis=$basis/" -e "s/^calc=.*/calc=$c/" -e "s/^cctol=.*/cctol=12/" -e "/^nsing/d" -e "/^ntrip/d" \
      $W/bench/refs/mrcc/h2o_631g/MINP.eom.bohr > $out/MINP.$tag
    echo "#### $tag"
    bash $W/bench/mrcc_nr.sh $out/MINP.$tag $tag 2>&1 | grep -E "Total .* energy|FINAL|converged|^=="
  done
done
