#!/bin/bash
# CFOUR's spinor EOM-CCSD (xx2ceomcc) against the EOM-CCSD of zccsdt on
# CFOUR's spinors and integrals (docs/eom.md, "CFOUR's spinor EOM-CCSD"):
#
#   bench/cfour/eom_ccsd.sh TAG "GEOMETRY lines" "BASIS lines" NROOT [DROPMO [ALG]]
#   e.g.  bench/cfour/eom_ccsd.sh hbr $'BR 0.0 0.0 0.0\nH 0.0 0.0 1.4145' $'BR:PVDZ_DK\nH:PVDZ_DK' 4 '1>14'
#
# Two runs of CFOUR in EOM_DIR (default /tmp/zccsdt_eom_cfour), since
# xx2ceomcc wants ABCDTYPE=AOBASIS and the <ab||cd> file that zccsdt reads is
# written with ABCDTYPE=STANDARD only:
#   cfour_TAG_ao : xjoda xsox2catm xsox2chf xx2cvtran xx2cccsd xx2chbar xx2ceomcc
#   cfour_TAG_std: xjoda xsox2catm xsox2chf xx2cvtran
# then bench/cfour/eom_ccsd_compare.py.  ZMAT: ESTATE_CONV=8 (largest element
# of the residual), EOMFOLLOW=OVERLAP, %eomnroot NROOT, %eomalg ALG (0: every
# root in a subspace of its own from its own unit vector; 1, 2: one subspace).
W=$(cd "$(dirname "$0")/../.." && pwd)
T=${EOM_DIR:-/tmp/zccsdt_eom_cfour}
BIN=${CFOUR_BIN:-$HOME/apps/cfour-x2cso/bin}
tag=$1; geom=$2; basis=$3; nroot=${4:-4}; drop=$5; alg=${6:-0}
export PYTHONPATH=$W/src:$HOME/apps/socutils-git-env:$PYTHONPATH
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for kind in ao std; do
  R=$T/cfour_${tag}_$kind
  mkdir -p $R; cd $R
  cp ${BIN%/bin}/basis/GENBAS GENBAS
  abcd=AOBASIS; [ $kind = std ] && abcd=STANDARD
  {
    echo "$tag X2C EOM-CCSD"
    echo "$geom"
    echo
    echo "*CFOUR(CALC=CCSD,BASIS=SPECIAL"
    echo "SYM=OFF,COORDINATES=CARTESIAN,UNITS=ANGSTROM,CHARGE=0,MULT=1"
    echo "RELATIVISTIC=X2C1E,X2CKEY=SOATMX,ABCDTYPE=$abcd"
    [ -n "$drop" ] && echo "DROPMO=$drop"
    echo "SCF_CONV=10,SCF_MAXCYC=200"
    echo "CC_CONV=10,CC_MAXCYC=100"
    echo "ESTATE_CONV=8,ESTATE_MAXCYC=80,EOMFOLLOW=OVERLAP"
    echo "MEMORY_SIZE=4,MEM_UNIT=GB)"
    echo
    echo "$basis"
    echo
    echo "%eomnroot"
    echo "$nroot"
    echo "%eomalg"
    echo "$alg"
    echo
  } > ZMAT
  progs="xjoda xsox2catm xsox2chf xx2cvtran xx2cccsd xx2chbar xx2ceomcc"
  [ $kind = std ] && progs="xjoda xsox2catm xsox2chf xx2cvtran"
  cd $W
  python -u -m zccsdt.cfour --chain --no-cc --bin $BIN --programs "$progs" $R 2>&1 | grep -v -i -E "plugins found|site-packages/pyscf|shciscf|PYSCF_EXT_PATH" | tail -12
done
python -u bench/cfour/eom_ccsd_compare.py $T/cfour_${tag}_std $T/cfour_${tag}_ao $((nroot + 2)) $((nroot + 4)) 2>&1 | grep -v -i -E "plugins found|site-packages/pyscf|shciscf|PYSCF_EXT_PATH|Warning|print\\("
