'''CCSDT, CCSDT(Q), CCSDTQ and CCSDTQ(P) through zccsdt.CCSDTQ in the blocks
of the symmetry labels and without them: a linear molecule on the z axis
with X2C spinors (labels 2 m_j), the whole calculation once each way.

usage: python bench/sym_all.py GEOM BASIS FROZEN [NVIR]
       e.g.  python bench/sym_all.py 'Br 0 0 0; H 0 0 1.4145' cc-pvtz-dk 28
'''
import sys
import time

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
out = {}
for sym in (True, False):
    t0 = time.time()
    cc = zccsdt.CCSDTQ(mf, frozen=list(order[:frozen]) + ([] if nvir is None else list(order[nocc + nvir:])))
    cc.verbose, cc.symmetry = 4, sym
    cc.conv_tol, cc.conv_tol_normt = 1e-10, 1e-8
    cc.kernel()
    q = cc.ccsdt_q()
    p = cc.ccsdtq_p()
    out[sym] = (cc.e_ccsdt, cc.e_corr, q, p)
    print('== symmetry %s: %.1f s in all' % (sym, time.time() - t0), flush=True)
print('blocks - dense: E(CCSDT) %.1e  E(CCSDTQ) %.1e  E(Q) %.1e  E(P) %.1e' % tuple(np.subtract(out[True], out[False])))
