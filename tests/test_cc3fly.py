'''CC3 without stored triples (zccsdt.cc.cc3fly, csrc/cc3_fly.c; docs/ccn.md
sections 9, 10) against CC3 with T3 stored - the same equations, the triples
made per batch of occupied triples:

- the T3 terms of the singles and doubles equations at random amplitudes
  against pass 1 over the stored triples of the closed form, real and
  complex, with one occupied triple per batch and with all in one;
- the ground state, the folded sigma vector at a frequency (against the
  stored Jacobian folded by hand) and the self-consistent roots: packed
  layout and the blocks of the symmetry labels (LiH, X2C), complex spin
  orbitals (H2O);
- the front end: store_t3 = False.'''
import numpy as np
import pytest
from pyscf import gto, scf, x2c

import zccsdt
from conftest import water_ghf
from test_ccsdtq_ref import _scf, _spin_mixed
from zccsdt import eris as E
from zccsdt.cc import cc3fly, ccn, eom, eom_opt, solver, t3pack, zeta_opt

pytestmark = pytest.mark.skipif(not cc3fly.available(), reason='kernel library without csrc/cc3_fly.c')
TIGHT = dict(conv_tol=1e-13, conv_tol_normt=1e-10)


def _fold(gs, r1, r2, omega):
    '''A_PP r + A_PQ (omega - A_QQ)^-1 A_QP r from the Jacobian with R3 stored.'''
    eo3, ev3 = eom.denominators(gs)[2]
    s1, s2, s3 = eom_opt.sigma(gs, r1, r2, np.zeros(gs.eng.size, dtype=gs.dtype))
    r3 = s3.copy()
    for _, _, where, shape, I, A in zeta_opt._blocks(gs.eris, gs.dtype):
        r3[where] = (r3[where].reshape(shape) / (omega - (ev3[A][None, :] - eo3[I][:, None]))).ravel()
    b1, b2, _ = eom_opt.sigma(gs, np.zeros_like(r1), np.zeros_like(r2), r3)
    return s1 + b1, s2 + b2


@pytest.mark.parametrize('cplx', [False, True])
def test_spin_orbitals(cplx):
    '''H2O / STO-3G, frozen core (8 + 4), and LiH / STO-3G (4 + 8): ground
    state, the step at amplitudes that solve nothing for two sizes of the
    work array, the folded sigma vector.'''
    for mf, frozen in ((water_ghf('sto-3g'), 2), (scf.addons.convert_to_ghf(_scf('lih')), 0)):
        eris = E.from_scf(mf, mo_coeff=_spin_mixed(mf)[0] if cplx else mf.mo_coeff, frozen=frozen)
        eris = ccn.semicanonicalize(eris)[0]        # (the amplitudes of solver.kernel refer to these)
        a = solver.kernel(eris, 'cc3', solver.Options(**TIGHT))
        b = solver.kernel(eris, 'cc3', solver.Options(store_t3=False, **TIGHT))
        assert b.t3 is None and b.converged and abs(a.e_corr - b.e_corr) < 1e-12
        assert abs(a.t2 - b.t2).max() < 1e-10 and b.t1.dtype == (np.complex128 if cplx else np.float64)
        # one step from other amplitudes, with the smallest work array (one occupied triple
        # per batch) and with the default: T3 is that of the closed form at these amplitudes
        opts = solver.Options()
        t1, t2 = 1.3 * a.t1, 0.8 * a.t2
        ref = None
        for mem in (1e-6, 2000.0):
            n1, n2, n3 = zccsdt.cc.ccsdt_opt.update_amps(opts, t1, t2, None, eris, mem_mb=mem, model='cc3')
            assert n3 is None
            if ref is None:
                ref = (n1, n2)
            assert abs(n1 - ref[0]).max() < 1e-13 and abs(n2 - ref[1]).max() < 1e-13
        # (the stored route at T3 = closed form: one step from T3 = 0 gives it, a second uses it)
        om, vm = t3pack.maps_for(eris)
        _, _, t3 = zccsdt.cc.ccsdt_opt.update_amps(opts, t1, t2, np.zeros(om.ntri * vm.ntri, dtype=n1.dtype), eris, model='cc3')
        s1, s2, _ = zccsdt.cc.ccsdt_opt.update_amps(opts, t1, t2, t3, eris, model='cc3')
        assert abs(s1 - ref[0]).max() < 1e-12 and abs(s2 - ref[1]).max() < 1e-12
        gs = eom_opt.prepare(eris, a.t1, a.t2, a.t3, model='cc3')
        gf = eom_opt.prepare(eris, a.t1, a.t2, None, model='cc3', fly=True)
        assert gf.fly and not gf.with_t3 and gf.change < 1e-8
        rng = np.random.default_rng(3)
        r1 = rng.standard_normal((om.n, vm.n)) + (1j * rng.standard_normal((om.n, vm.n)) if cplx else 0)
        r2 = rng.standard_normal((om.npair, vm.npair)) + (1j * rng.standard_normal((om.npair, vm.npair)) if cplx else 0)
        for omega in (0.0, 0.3):
            w1, w2 = _fold(gs, r1, r2, omega)
            f1, f2, f3 = eom_opt.sigma(gf, r1, r2, omega=omega)
            assert f3 is None and abs(f1 - w1).max() < 1e-11 * abs(w1).max() and abs(f2 - w2).max() < 1e-11 * abs(w2).max()
        with pytest.raises(ValueError):
            eom_opt.sigma(gf, r1, r2)


@pytest.fixture(scope='module')
def lih_x2c():
    mol = gto.M(atom='Li 0 0 0; H 0 0 1.5957', basis='6-31g', verbose=0)
    mf = x2c.UHF(mol)
    mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-9
    mf.kernel()
    out = {}
    for sym in (True, False):
        for store in (True, False):
            cc = zccsdt.CC3(mf)
            cc.symmetry, cc.verbose, cc.store_t3 = sym, 0, store
            cc.conv_tol, cc.conv_tol_normt = 1e-13, 1e-10
            cc.kernel()
            out[sym, store] = cc
    return out


@pytest.mark.parametrize('sym', [True, False])
def test_label_blocks_and_front_end(lih_x2c, sym):
    '''LiH / 6-31G, X2C (4 + 18; real): in the blocks of the labels the
    batches compute the allowed blocks only, and everything agrees with T3
    stored and with the packed layout.'''
    a, b = lih_x2c[sym, True], lih_x2c[sym, False]
    assert b.t3 is None and b.converged and b.t2.dtype == np.float64
    assert abs(a.e_corr - b.e_corr) < 1e-12 and abs(a.e_corr - lih_x2c[not sym, False].e_corr) < 1e-12
    eris = b._eris
    om, vm = t3pack.maps_for(eris)
    fl = cc3fly.Fly(om, vm, np.float64, *zccsdt.cc.ccsdt_opt.fock_canonical(eris, om, vm))
    if sym:     # 6 classes; the batches of the largest group need a fraction of the dense work array
        assert fl.ncls > 1 and max(fl.per_i.values()) < 0.5 * (3 * vm.n * vm.npair + vm.ntri)
    else:
        assert fl.ncls == 1 and fl.per_i[0] == 3 * vm.n * vm.npair + vm.ntri
    assert sum(i1 - i0 for i0, i1, j, k in fl.batches()) <= om.ntri
    gs = eom_opt.prepare(a._eris, a.t1, a.t2, a.t3, model='cc3')
    gf = eom_opt.prepare(eris, b.t1, b.t2, None, model='cc3', fly=True)
    rng = np.random.default_rng(5)
    m1, m2 = gs.masks(0) if sym else (1.0, 1.0)
    r1 = rng.standard_normal((om.n, vm.n)) * m1
    r2 = rng.standard_normal((om.npair, vm.npair)) * m2
    w1, w2 = _fold(gs, r1, r2, 0.15)
    f1, f2, _ = eom_opt.sigma(gf, r1, r2, omega=0.15)
    assert abs(f1 - w1).max() < 1e-11 * abs(w1).max() and abs(f2 - w2).max() < 1e-11 * abs(w2).max()
    e_s, res_s = a.eom_ee(nroots=6, conv_tol=1e-9, **({'shift': 0} if sym else {}))
    e_f, res_f = b.eom_ee(nroots=3, conv_tol=1e-9)
    assert res_f.converged.all() and res_f.r3 is None and b.e_ee is e_f
    assert max(abs(e_s - x).min() for x in e_f) < 1e-9 and abs(e_f[0] - e_s[0]) < 1e-9
    with pytest.raises(NotImplementedError):
        b.eom_ee(nroots=2, shift=2)
    with pytest.raises(NotImplementedError):
        b.eom_ip(nroots=2)


def test_semicanonical_front_end_and_solver():
    '''An SCF converged with its noise in the blocks of F: the front end
    rotates the spinors before the integrals are made, solver.kernel turns
    the given integrals; the two and the batch route agree to 1e-12, and
    the off-diagonal elements the front end saw are reported.'''
    mol = gto.M(atom='Li 0 0 0; H 0 0 1.5957', basis='6-31g', verbose=0)
    mf = x2c.UHF(mol)
    mf.conv_tol = 1e-7                  # (a loose SCF: noise of 1e-5 in the blocks)
    mf.kernel()
    out = {}
    for sym in (True, False):
        cc = zccsdt.CC3(mf)
        cc.symmetry, cc.verbose = sym, 0
        cc.conv_tol, cc.conv_tol_normt = 1e-13, 1e-10
        cc.kernel()
        assert cc.semicanonical is not None and cc._eris.semicanonical is not None
        out[sym] = cc
    a = out[True]
    eris = E.from_scf(mf)
    f = np.asarray(eris.fock)
    no = eris.nocc
    assert abs(f[:no, :no] - np.diag(f.diagonal()[:no])).max() > 1e-9
    for store in (True, False):
        r = solver.kernel(eris, 'cc3', solver.Options(store_t3=store, **TIGHT))
        assert r.rotation is not None and abs(r.e_corr - a.e_corr) < 1e-12 and abs(out[False].e_corr - a.e_corr) < 1e-12
