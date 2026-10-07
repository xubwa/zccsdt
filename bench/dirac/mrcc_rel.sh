#!/bin/bash
# MRCC through its interface to DIRAC (iface=dirac) on the MO integrals of a
# DIRAC calculation, with the original (Q) and (P):
#
#   usage: bench/dirac/mrcc_rel.sh DIR 'CALC'      e.g.  ... run/bh 'CCSDT(Q)'
#
# DIR has MRCONEE and MDCINT: those of DIRAC (pam --get="MRCONEE MDCINT",
# .4INDEX, a real group given explicitly in the .mol file: C2v, D2 or D2h -
# with the linear supersymmetry that DIRAC finds itself MRCC stops at
# 'Invalid spatial symmetry'), or the integrals of a PySCF calculation
# written by zccsdt.dirac (bench/dirac/write_integrals.py; zccsdt on the same
# files: bench/dirac/run_integrals.py).  Three things that the interface
# does not do:
#
# - DIRAC 23 writes one of the integrals related by a permutation; this MRCC
#   (release of 2018) does not complete the list and its reference energy is
#   wrong.  The complete MDCINT is written first (zccsdt.dirac.write_mdcint).
# - MRCC takes the reference for a non-HF one (HF = 0 in fort.56) and then
#   gives the (Q)/A, (Q)/B, (P)/A, (P)/B of Kallay and Gauss (2008), which
#   are not the original corrections even for canonical HF spinors.  The run
#   through dmrcc makes fort.56; its 12th entry is set to 1 and goldstone,
#   xmrcc and mrcc are run again.
# - dirac_mointegral_export (MRCC's) writes fort.55 and a sample MINP whose
#   occupation line (refdet=vector) is taken over.
#
# Results: DIR/full/mrcc_<calc>/out.mrcc (HF = 1) and out.dmrcc (HF = 0).
set -e
MRCCDIR=${MRCCDIR:-/home/czhan119/code/cfour-git/bin}
ROOT=${ZC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
export PATH=$MRCCDIR:$PATH
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${OMP_NUM_THREADS:-8}
src=$(readlink -f "$1"); calc=$2
tag=$(echo "$calc" | tr 'A-Z()' 'a-z__' | sed 's/_$//')
full=$src/full
if [ ! -f $full/MDCINT ]; then
  mkdir -p $full
  (cd $ROOT && PYTHONPATH=src:$PYTHONPATH python - "$src" "$full" <<'PY'
import shutil, sys
from zccsdt import dirac
src, out = sys.argv[1:3]
x = dirac.read(src + '/MRCONEE', src + '/MDCINT')
print('%d integrals in the complete list' % dirac.write_mdcint(out + '/MDCINT', x))
shutil.copy(src + '/MRCONEE', out + '/MRCONEE')
PY
  ) 2>&1 | grep -v -i -E "plugins found|site-packages/pyscf|shciscf/pyscf|PYSCF_EXT_PATH"
  (cd $full && dirac_mointegral_export mrcc > export.log 2>&1)
fi
R=$full/mrcc_$tag
mkdir -p $R && cp $full/MRCONEE $full/MDCINT $full/fort.55 $R/
printf 'iface=dirac\ncalc=%s\nmem=%s\ncctol=12\nrefdet=vector\n%s\n' "$calc" "${MRCC_MEM:-4GB}" "$(sed -n 5p $full/MINP)" > $R/MINP
cd $R
dmrcc > out.dmrcc 2>&1 || true
python - <<'PY'
import re
lines = open('fort.56').read().split('\n')
a, b = [m.span() for m in re.finditer(r'\S+', lines[0])][11]
lines[0] = lines[0][:a] + '1' + lines[0][b:]            # HF: the reference is the Hartree-Fock determinant
open('fort.56', 'w').write('\n'.join(lines))
PY
goldstone > out.goldstone 2>&1
xmrcc > out.xmrcc 2>&1
mrcc > out.mrcc 2>&1
echo "== $calc with the reference as Hartree-Fock:"
grep -E "Total .* energy|Energy of reference" out.mrcc
echo "== the variants of the interface (HF = 0):"
grep -E "Total .*/[AB] energy" out.dmrcc || true
