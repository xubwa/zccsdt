'''CC4: T4 stored (ccn.cc4_kernel, on the T4 engine of CCSDTQ) and without
stored T4 (zccsdt.cc.cc4fly), with the folded excitation energies:

- against the dense reference (ccn_ref) and MRCC (H2O / STO-3G);
- the limit: the full quadruples equation is CCSDTQ;
- four electrons: CCSDTQ is exact, CC4 is not;
- the pieces of the route without T4 against the T4 engine (the four sums,
  the rows of the right-hand side), real and complex;
- the folded roots against the eigenvalues of the dense CC4 Jacobian.'''
from math import comb

import numpy as np
import pytest
from pyscf import fci, gto, scf

import zccsdt
from conftest import water_ghf
from test_ccsdtq_ref import _scf, _spin_mixed
from zccsdt import eris as E
from zccsdt.cc import cc4fly, ccn, ccn_ref as C, ccsdtq, ccsdtq_ref as R, t4engine, t4pack as P

TIGHT = dict(conv_tol=1e-13, conv_tol_normt=1e-10)


@pytest.mark.parametrize('case', ['lih', 'h2o-fc-z', 'h2o'])
def test_stored_against_reference_and_mrcc(case):
    if case == 'lih':
        mf, frozen, cplx = scf.addons.convert_to_ghf(_scf('lih')), 0, False
    else:
        mf, frozen, cplx = water_ghf('sto-3g'), (2 if 'fc' in case else 0), case.endswith('z')
    eris = E.from_scf(mf, mo_coeff=_spin_mixed(mf)[0] if cplx else mf.mo_coeff, frozen=frozen)
    res = ccn.cc4_kernel(eris, **TIGHT)
    fock, g, mo_e, no = R.integrals(res.eris)
    ref = C.kernel(fock, g, mo_e, no, 'cc4')
    assert res.converged and abs(res.e_corr - ref.e_corr) < 1e-11
    fly = ccn.cc4_kernel(eris, store_t4=False, **TIGHT)
    assert fly.t4 is None and abs(fly.e_corr - res.e_corr) < 1e-12
    if case == 'h2o':           # MRCC, closed-shell CC4 on its own integrals (the same basis and SCF)
        assert abs(res.e_tot - (-75.012639672408)) < 1e-10
    if case == 'lih':           # the full quadruples equation: CCSDTQ
        lim = ccn.cc4_kernel(eris, model='ccsdtq', **TIGHT)
        full = ccsdtq.kernel(eris, ccsdtq.Options(**TIGHT))
        assert abs(lim.e_corr - full.e_corr) < 1e-12 and 1e-10 < abs(full.e_corr - res.e_corr)


def test_four_electrons():
    '''H4 / STO-3G (4 + 4 spin orbitals, one quadruple that conserves S_z):
    CCSDTQ is FCI, CC4 is not.'''
    mol = gto.M(atom='H 0 0 0; H 0 0 0.9; H 0 1.1 0; H 0 1.0 1.0', basis='sto-3g', verbose=0)
    rhf = scf.RHF(mol).run(conv_tol=1e-13)
    e_fci = fci.FCI(rhf).kernel()[0]
    eris = E.from_scf(scf.addons.convert_to_ghf(rhf))
    full = ccsdtq.kernel(eris, ccsdtq.Options(**TIGHT))
    cc4 = ccn.cc4_kernel(eris, **TIGHT)
    assert abs(eris.e_hf + full.e_corr - e_fci) < 1e-10
    assert abs(np.asarray(full.t4)).max() > 1e-6
    assert 1e-8 < abs(cc4.e_tot - e_fci) < 1e-3


@pytest.mark.parametrize('cplx', [False, True])
def test_pieces_of_the_route_without_t4(cplx):
    '''H2O / STO-3G (10 + 4): the four sums of a random T4 and the whole pass
    at the CC4 amplitudes against the T4 engine; the rows against the
    closed form of the stored route.'''
    mf = water_ghf('sto-3g')
    eris = E.from_scf(mf, mo_coeff=_spin_mixed(mf)[0] if cplx else mf.mo_coeff)
    res = ccn.cc4_kernel(eris, conv_tol=1e-12, conv_tol_normt=1e-9)
    er = res.eris
    no = er.nocc
    nv = er.fock.shape[0] - no
    e = np.asarray(er.fock).diagonal().real
    ints = t4engine.Dressed.from_eris(er, res.t1)
    rng = np.random.default_rng(1)
    shape = (comb(no, 4), comb(nv, 4))
    T4 = rng.standard_normal(shape) + (1j * rng.standard_normal(shape) if cplx else 0)
    z2, z3 = np.zeros_like(res.t2), np.zeros_like(res.t3)
    _, r3, r2 = t4engine.residual(z2, z3, T4, ints)
    back = cc4fly.Back(ints, T4.dtype)
    R2, R3 = np.zeros_like(r2), np.zeros((comb(no, 3), comb(nv, 3)), dtype=T4.dtype)
    for I in range(shape[0]):
        back(I, np.ascontiguousarray(T4[I]), R2, R3)
    assert abs(R2 - r2).max() < 1e-13 * abs(r2).max() and abs(R3 - r3.reshape(R3.shape)).max() < 1e-13 * abs(r3).max()
    rows = []
    F2, F3 = cc4fly.run(ints, e, res.t2, res.t3, rows=rows)
    d4 = e[:no][P.tuples(no, 4)].sum(axis=1)[:, None] - P.tuple_sums(e[no:], 4)[None, :]
    T4c = ccn.cc4_rhs(res.t2, res.t3, ints) / d4
    assert max(abs(r - T4c[I]).max() for I, r in rows) < 1e-12 * abs(T4c).max()
    _, s3, s2 = t4engine.residual(z2, z3, T4c, ints)
    assert abs(F2 - s2).max() < 1e-14 and abs(F3 - s3.reshape(F3.shape)).max() < 1e-14


@pytest.mark.parametrize('cplx', [False, True])
def test_folded_roots(cplx):
    '''LiH / STO-3G (4 + 8): the roots of the folded problem without stored
    T4 against the eigenvalues of the dense CC4 Jacobian.'''
    mf = scf.addons.convert_to_ghf(_scf('lih'))
    eris = E.from_scf(mf, mo_coeff=_spin_mixed(mf)[0] if cplx else mf.mo_coeff)
    res = ccn.cc4_kernel(eris, store_t4=False, **TIGHT)
    fock, g, mo_e, no = R.integrals(res.eris)
    ref = C.kernel(fock, g, mo_e, no, 'cc4')
    w = np.sort(C.eigenvalues('cc4', ref.amps, fock, g, no).real)
    e, sol = cc4fly.eom(res.eris, res.t1, res.t2, res.t3, nroots=4, conv_tol=1e-9)
    assert sol.converged.all() and max(abs(w - x).min() for x in e) < 1e-9
    assert abs(e[0] - w[w > 1e-6][0]) < 1e-9


def test_front_end():
    mf = water_ghf('sto-3g')
    cc = zccsdt.CC4(mf, frozen=2)       # (spin orbitals of the GHF object: the 1s pair)
    cc.verbose = 0
    cc.conv_tol, cc.conv_tol_normt = 1e-12, 1e-9
    cc.kernel()
    ref = ccn.cc4_kernel(E.from_scf(mf, frozen=2), **TIGHT)
    assert cc.converged and abs(cc.e_corr - ref.e_corr) < 1e-10 and cc.model == 'cc4'
    e, res = cc.eom_ee(nroots=2, conv_tol=1e-7)
    assert len(e) == 2 and res.converged.all()
