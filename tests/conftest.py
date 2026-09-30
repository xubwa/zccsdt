import functools
import numpy as np
import pytest
from pyscf import gto, scf, x2c

from zccsdt import eris as E

WATER = 'O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587'


@functools.lru_cache(maxsize=None)
def water_ghf(basis='6-31g'):
    mol = gto.M(atom=WATER, basis=basis, verbose=0)
    rhf = scf.RHF(mol).run(conv_tol=1e-12)
    return scf.addons.convert_to_ghf(rhf)


@functools.lru_cache(maxsize=None)
def water_x2c(basis='6-31g'):
    mol = gto.M(atom=WATER, basis=basis, verbose=0)
    return x2c.UHF(mol).run(conv_tol=1e-12)


@functools.lru_cache(maxsize=None)
def three_electron(kind='ghf', basis='6-31g'):
    '''Li atom (3 electrons, doublet) as UHF -> GHF, or X2C spinor.'''
    mol = gto.M(atom='Li 0 0 0', basis=basis, spin=1, verbose=0)
    if kind == 'x2c':
        return x2c.UHF(mol).run(conv_tol=1e-12)
    uhf = scf.UHF(mol).run(conv_tol=1e-12)
    return scf.addons.convert_to_ghf(uhf)


def socutils_available():
    try:
        import socutils.cc.zccsdt  # noqa: F401
        return True
    except Exception:
        return False
