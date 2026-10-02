#!/usr/bin/env python
'''
CCSDT on two-component X2C spinors (complex arithmetic, no spin blocks).
'''

from pyscf import gto, x2c
import zccsdt

mol = gto.M(atom='S 0 0 0; H 0 0.9618 0.9272; H 0 -0.9618 0.9272', basis='cc-pvdz',
            verbose=4)
mf = x2c.UHF(mol).run()

mycc = zccsdt.CCSDT(mf, frozen=10)  # zccsdt.GCCSDT; 10 core spinors frozen
mycc.run()
