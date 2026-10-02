#!/usr/bin/env python
'''
Closed-shell CCSDT(Q) on an RHF reference (spin-adapted engine).
'''

from pyscf import gto, scf
import zccsdt

mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis='cc-pvtz', verbose=4)
mf = scf.RHF(mol).run()

mycc = zccsdt.CCSDT(mf)             # zccsdt.RCCSDT for an RHF reference
mycc.set_frozen()                   # freeze the chemical core
mycc.kernel()
e_q = mycc.ccsdt_q()
print('E(CCSDT) = %.12f   E(CCSDT(Q)) = %.12f' % (mycc.e_tot, mycc.e_tot_q))
