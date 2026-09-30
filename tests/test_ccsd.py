import numpy as np
import pytest
from pyscf.cc import gccsd

from zccsdt import eris as E
from zccsdt.cc import solver
from conftest import water_ghf, water_x2c, socutils_available

TIGHT = solver.Options(conv_tol=1e-12, conv_tol_normt=1e-10)


def _pyscf_gccsd(mf, frozen, mo_coeff=None):
    m = gccsd.GCCSD(mf, frozen=frozen, mo_coeff=mo_coeff)
    m.conv_tol = 1e-12
    m.max_cycle = 200
    m.conv_tol_normt = 1e-10
    m.kernel()
    assert m.converged
    return m.e_corr


def test_ccsd_real_vs_pyscf():
    mf = water_ghf()
    r = solver.kernel(E.from_scf(mf, frozen=2), 'ccsd', TIGHT)
    assert r.converged
    assert abs(r.e_corr - _pyscf_gccsd(mf, 2)) < 1e-10


def test_ccsd_complex_rotated_vs_pyscf():
    '''RHF -> GHF -> complex unitary (occ-occ, vir-vir, alpha/beta mixed).'''
    mf = water_ghf()
    C = E.rotate_occ_vir(mf.mo_coeff, mf.mo_occ, seed=3, frozen=2)
    e = E.from_scf(mf, mo_coeff=C, frozen=2)
    r = solver.kernel(e, 'ccsd', TIGHT)
    assert r.converged
    assert abs(r.e_corr.imag) < 1e-10
    assert abs(r.e_corr - _pyscf_gccsd(mf, 2, mo_coeff=C)) < 1e-10
    assert abs(r.e_corr - _pyscf_gccsd(mf, 2)) < 1e-10


@pytest.mark.socutils
@pytest.mark.skipif(not socutils_available(), reason='socutils not importable')
def test_ccsd_x2c_vs_socutils():
    from pyscf import gto
    from socutils.scf import spinor_hf
    from socutils.cc.zccsd import ZCCSD
    from conftest import WATER
    mol = gto.M(atom=WATER, basis='6-31g', verbose=0)
    mf = spinor_hf.SCF(mol).run(conv_tol=1e-12)
    ref = ZCCSD(mf, frozen=2)
    ref.conv_tol = 1e-12
    ref.conv_tol_normt = 1e-9
    ref.verbose = 0
    ref.kernel()
    r = solver.kernel(E.from_scf(mf, frozen=2), 'ccsd', TIGHT)
    assert abs(r.e_corr - ref.e_corr) < 1e-9
