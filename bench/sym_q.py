'''(Q) in symmetry sectors against the dense (Q), one after the other on the
same amplitudes: a linear molecule on the z axis with X2C spinors (labels
2 m_j).

usage: python bench/sym_q.py GEOM BASIS FROZEN [NVIR]
       e.g.  python bench/sym_q.py 'Br 0 0 0; H 0 0 1.4145' cc-pvtz-dk 28
'''
import sys

import numpy as np
from pyscf import gto, x2c

import zccsdt

geom, basis, frozen = sys.argv[1], sys.argv[2], int(sys.argv[3])
nvir = int(sys.argv[4]) if len(sys.argv) > 4 else None
mol = gto.M(atom=geom, basis=basis, verbose=0)
mf = x2c.UHF(mol)
mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-8
mf.kernel()
order = np.argsort(mf.mo_energy)
nocc = int((mf.mo_occ > 0).sum())
cc = zccsdt.GCCSDT(mf, frozen=list(order[:frozen]) + ([] if nvir is None else list(order[nocc + nvir:])))
cc.verbose = 4
cc.conv_tol, cc.conv_tol_normt = 1e-10, 1e-8
cc.kernel()
out = {}
for sym in (True, False, True):
    cc.symmetry = sym
    cc.ccsdt_q()
    out[sym] = (cc.e_q5, cc.e_q)
print('sectors - dense: E[Q] %.1e  E(Q) %.1e' % tuple(np.subtract(out[True], out[False])))
