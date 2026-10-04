'''Orbital relaxation of the spinor CCSD (zccsdt.zccsd.zvector): the relaxed
one-particle density against the derivative of E(SCF) + E_corr with respect to
a one-particle perturbation, the SCF repeated in the field.'''

import numpy as np
import pytest
from pyscf import gto, x2c
from pyscf.cc import gccsd_rdm

from zccsdt.zccsd import DirectZCCSD, zvector


@pytest.fixture(scope='module')
def mol():
    return gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis='6-31g', verbose=0)


def scf(mol, eps=0.0, V=None):
    mf = x2c.UHF(mol)
    if V is not None:
        h0 = mf.get_hcore()
        mf.get_hcore = lambda *args: h0 + eps * V
    mf.conv_tol = 1e-12
    mf.kernel()
    assert mf.converged
    return mf


def ccsd(mf, kr):
    mycc = DirectZCCSD(mf)
    mycc.verbose, mycc.conv_tol, mycc.conv_tol_normt, mycc.kramers = 0, 1e-12, 1e-9, kr
    mycc.max_cycle = 200
    mycc.kernel()
    assert mycc.converged
    return mycc


@pytest.mark.parametrize('kr', [False, None])
def test_relaxed_density_is_energy_derivative(mol, kr):
    '''The energy from the densities of t and lambda; the orbital gradient
    of the Lagrangian (anti-Hermitian, no occupied-occupied and no
    virtual-virtual block); the relaxed density with the dipole operator of
    the 2c AO basis against the central difference of E(SCF) + E_corr in the
    field, which the unrelaxed density misses by 2e-3.'''
    mf = scf(mol)
    mycc = ccsd(mf, kr)
    eris = mycc.eris
    assert bool(eris.kramers) == (kr is None)
    C, o = eris.mo_coeff, eris.nocc
    l1, l2 = mycc.solve_lambda()
    dm1 = gccsd_rdm.make_rdm1(mycc, mycc.t1, mycc.t2, l1, l2)
    dm2 = gccsd_rdm.make_rdm2(mycc, mycc.t1, mycc.t2, l1, l2)
    h = C.conj().T @ mf.get_hcore() @ C
    eri = mol.intor('int2e_spinor')
    eri = np.einsum('mp,nq,mnls,lr,st->pqrt', C.conj(), C, eri, C.conj(), C, optimize=True)
    e = np.einsum('pq,qp', h, dm1) + 0.5 * np.einsum('pqrs,pqrs', eri, dm2) + mol.energy_nuc()
    assert abs(e - mf.e_tot - mycc.e_corr) < 1e-9
    G = zvector.orbital_gradient(h, eri, dm1, dm2)
    assert abs(G + G.conj().T).max() < 1e-10
    assert abs(G[:o, :o]).max() < 1e-8 and abs(G[o:, o:]).max() < 1e-8
    assert abs(G[o:, :o]).max() > 1e-4

    relaxed, unrelaxed, z = zvector.relaxed_rdm1_dense(mycc, mf, l1, l2, tol=1e-10)
    assert abs(unrelaxed - dm1).max() < 1e-12 and abs(relaxed - relaxed.conj().T).max() < 1e-12
    R, eai = zvector.response(mf, C, eris.mo_energy, o)
    assert abs(R(z) + G[o:, :o]).max() < 1e-9
    V = mol.intor('int1e_r_spinor')[2]
    Vmo = C.conj().T @ V @ C
    eps = 1e-4
    ep, em = [(m.e_tot + ccsd(m, kr).e_corr) for m in (scf(mol, eps, V), scf(mol, -eps, V))]
    derivative = (ep - em) / (2 * eps)
    assert abs(derivative - np.einsum('pq,qp', Vmo, relaxed).real) < 2e-7
    assert abs(derivative - np.einsum('pq,qp', Vmo, unrelaxed).real) > 1e-3
