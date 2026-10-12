#!/bin/bash
# CFOUR's nonrelativistic CC2 / CCSD / CC3 with their EOM (linear-response)
# excitation energies (xvcc, xvee) for H2O, RHF, no point group - the
# reference of zccsdt.CC2 / CCSD / CC3 and eom_ee (docs/ccn.md, section 7):
#
#   bench/ccn/cfour_eom.sh OUTDIR [BASIS ..]        (default: STO-3G 6-31G)
#   python bench/ccn/compare_cfour.py OUTDIR 6-31G
#
# CALCS (default "CC2 CCSD CC3"), NROOT (5 singlets).  Geometry in bohr, as
# bench/refs/mrcc/h2o_631g/MINP.eom.bohr; the basis entries are CFOUR's
# (those of 6-31G are in bench/refs/GENBAS too).  The runs are made in
# ZCCSDT_TMPDIR (default /tmp); OUTDIR gets h2o_<BASIS>_<CALC>.out.
BIN=${CFOUR_BIN:-$HOME/apps/cfour-x2cso/bin}
export PATH=$BIN:$PATH
out=$(readlink -f "$1"); shift
mkdir -p "$out"
for basis in ${@:-STO-3G 6-31G}; do
  for c in ${CALCS:-CC2 CCSD CC3}; do
    tag=h2o_${basis}_$c
    R=${ZCCSDT_TMPDIR:-/tmp}/cfour_ccn_$tag
    mkdir -p "$R" && cd "$R" && find . -maxdepth 1 -type f -delete
    cp ${BIN%/bin}/basis/GENBAS GENBAS
    {
      echo "H2O $c excitation energies"
      echo "O 0.000000000000000 0.000000000000000 0.000000000000000"
      echo "H 0.000000000000000 1.430522676295752 1.109269235119691"
      echo "H 0.000000000000000 -1.430522676295752 1.109269235119691"
      echo
      echo "*CFOUR(CALC=$c,BASIS=$basis"
      echo "SYM=OFF,COORDINATES=CARTESIAN,UNITS=BOHR,CHARGE=0,MULT=1"
      echo "REF=RHF,FROZEN_CORE=OFF,EXCITE=EOMEE,ESTATE_SYM=${NROOT:-5}"
      echo "SCF_CONV=11,CC_CONV=11,LINEQ_CONV=10,ESTATE_CONV=9,ESTATE_MAXCYC=200"
      echo "CC_MAXCYC=200,MEMORY_SIZE=4,MEM_UNIT=GB)"
      echo
    } > ZMAT
    xcfour > out 2>&1
    echo "#### $tag (exit $?)"
    grep -E "E\(SCF\)=|Converged eigenvalue|Total EOMEE" out | tail -12
    cp out "$out/$tag.out"
  done
done
