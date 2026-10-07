#!/usr/bin/env python
'''
CCSDT(Q), CCSDTQ and CCSDTQ(P) on two-component X2C spinors: HBr with the
core frozen and the virtual space cut to its 16 lowest spinors (frozen as a
list, as in PySCF).  The molecule is not on the z axis: complex spinors.
'''

import numpy as np
from pyscf import gto, x2c
import zccsdt

mol = gto.M(atom='Br 0 0 0; H 0.988 0.745 0.686', basis='cc-pvdz-dk', verbose=4)
mf = x2c.UHF(mol)
mf.conv_tol = 1e-12
mf.kernel()

order = np.argsort(mf.mo_energy)
nocc = int((mf.mo_occ > 0).sum())
frozen = list(order[:28]) + list(order[nocc + 16:])     # the core; all but 16 virtual spinors

mycc = zccsdt.CCSDTQ(mf, frozen=frozen)     # zccsdt.GCCSDTQ
mycc.kernel()                               # CCSDT, then CCSDTQ
e_q = mycc.ccsdt_q()                        # (Q) of the CCSDT step
e_p = mycc.ccsdtq_p()                       # (P)
print('E(CCSDT)     = %.10f' % mycc.e_ccsdt)
print('E(CCSDT(Q))  = %.10f' % (mycc.e_ccsdt + e_q))
print('E(CCSDTQ)    = %.10f' % mycc.e_tot)
print('E(CCSDTQ(P)) = %.10f' % mycc.e_tot_p)
