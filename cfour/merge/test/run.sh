#!/bin/bash
#
# run.sh: the test of xzccsdt in CFOUR's two-component chain.
#
#   test/run.sh CFOUR_BIN [XZCCSDT]
#
#   CFOUR_BIN  the bin directory of the CFOUR build (xjoda, xsox2catm, xsox2chf,
#              xx2cvtran, and xzccsdt unless XZCCSDT names another binary)
#
# HF / 6-31G (CFOUR's GENBAS entries, unchanged), X2C-1e, all electrons
# (10 + 12 spinors), the SCF in blocks of m_j (%dglin: the symmetry labels of
# dgcosym, real arithmetic), CALC=CCSDT.  In a new directory below the
# current one (or $XZTEST_DIR): xjoda, xsox2catm, xsox2chf, xx2cvtran,
# xzccsdt; then xzccsdt once more, which must take the amplitudes of the
# first run (file XZAMPS) instead of iterating.  Compared with the reference
# numbers below: E(SCF) and E(CCSDT) to 1e-8 Eh.  Prints PASS or FAIL;
# exit status 0 on PASS.  A few seconds (OMP_NUM_THREADS threads).
#
set -u
ref_scf=-100.064473416710
ref_cc=-100.196412399000
tol=1e-8

here=$(cd "$(dirname "$0")" && pwd)
[ $# -ge 1 ] || { sed -n '2,18p' "$0"; exit 2; }
bin=$(cd "$1" && pwd) || exit 2
xz=${2:-$bin/xzccsdt}
for p in xjoda xsox2catm xsox2chf xx2cvtran; do
  [ -x "$bin/$p" ] || { echo "run.sh: no $bin/$p"; exit 2; }
done
[ -x "$xz" ] || { echo "run.sh: no $xz"; exit 2; }
xz=$(cd "$(dirname "$xz")" && pwd)/$(basename "$xz")

d=$(mktemp -d "${XZTEST_DIR:-$PWD}/xzccsdt_test.XXXXXX") || exit 2
cp "$here/ZMAT" "$here/GENBAS" "$d/"
cd "$d" || exit 2
echo "run.sh: in $d"
for p in xjoda xsox2catm xsox2chf xx2cvtran; do
  "$bin/$p" > "$p.out" 2>&1 || { echo "FAIL: $p stopped (exit $?), see $d/$p.out"; exit 1; }
done
"$xz" > xzccsdt.out 2>&1 || { echo "FAIL: xzccsdt stopped (exit $?), see $d/xzccsdt.out"; exit 1; }
"$xz" > xzccsdt.2.out 2>&1 || { echo "FAIL: xzccsdt (second run) stopped (exit $?), see $d/xzccsdt.2.out"; exit 1; }

val() { awk -v k="$2" '$1 == k { v = $NF } END { print v }' "$1"; }
bad=0
check() {   # what value reference
  local ok
  ok=$(awk -v a="$2" -v b="$3" -v t=$tol 'BEGIN { d = a - b; if (d < 0) d = -d; print (a != "" && d <= t) ? 1 : 0 }')
  if [ "$ok" = 1 ]; then
    printf '  %-34s %20s   reference %20s   ok\n' "$1" "$2" "$3"
  else
    printf '  %-34s %20s   reference %20s   DIFFERS\n' "$1" "${2:-missing}" "$3"; bad=1
  fi
}
check 'E(SCF)' "$(val xzccsdt.out 'E(SCF)')" $ref_scf
check 'E(CCSDT)' "$(val xzccsdt.out 'E(CCSDT)')" $ref_cc
check 'E(CCSDT), XZAMPS taken' "$(val xzccsdt.2.out 'E(CCSDT)')" $ref_cc
grep -q 'symmetry labels: the file dgcosym' xzccsdt.out || { echo "  the labels of dgcosym were not used"; bad=1; }
grep -q 'taken, the CCSDT iterations are skipped' xzccsdt.2.out || { echo "  the second run did not take XZAMPS"; bad=1; }
grep -q 'CCSDT converged in' xzccsdt.2.out && { echo "  the second run iterated"; bad=1; }
if [ $bad = 0 ]; then echo "PASS"; else echo "FAIL (outputs in $d)"; fi
exit $bad
