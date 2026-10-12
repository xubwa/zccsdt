'''CC2 for spinors on Cholesky vectors (zccsdt.zccsd.cc2; DirectZCCSD.cc2(),
cc2_ee()) against the CC2 on the integrals of the spinors (zccsdt.cc.ccn,
the dense reference ccn_ref) - with vectors that reproduce the AO integrals
to rounding the two are the same calculation:

- the singles residual at a random t1, the ground state, the amplitudes;
- the folded sigma vector and the self-consistent roots against the CC2
  Jacobian with the doubles stored;
- batches and chunks sized by max_memory give the same numbers, and a
  budget below the three-index arrays is an error with the numbers;
- the error of the decomposition at the default max_error.'''
import numpy as np
import pytest
from pyscf import gto, x2c

from zccsdt import eris as E
from zccsdt.cc import ccn_ref as C, ccsdt, ccsdtq_ref as R, eom, eom_opt, solver
from zccsdt.zccsd import DirectZCCSD, cc2


@pytest.fixture(scope='module')
def lih():
    mol = gto.M(atom='Li 0 0 0; H 0 0 1.5957', basis='6-31g', verbose=0)
    mf = x2c.UHF(mol)
    mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-9
    mf.kernel()
    eris = E.from_scf(mf, packed_vvvv=False)
    res = solver.kernel(E.from_scf(mf), 'cc2', solver.Options(conv_tol=1e-13, conv_tol_normt=1e-11))
    if res.rotation is not None:     # (solver.kernel solves for semicanonical spinors: back to those of mf)
        from zccsdt.cc import ccn
        res.t1, res.t2 = ccn.back_t1(res.t1, res.rotation), ccn.back_t2(res.t2, res.rotation)
    return mol, mf, eris, res, cc2.cholesky_exact(mol)


def _engine(lih, **kw):
    mol, mf, eris, res, L = lih
    mog = np.vstack(mol.sph2spinor_coeff()) @ mf.mo_coeff
    return cc2.Engine(mog, eris.nocc, eris.fock, L, e_hf=mf.e_tot, **kw)


def test_residual_and_ground_state(lih):
    mol, mf, eris, res, L = lih
    fock, g, mo_e, no = R.integrals(eris)
    n = len(mo_e)
    eng = _engine(lih)
    assert eng.nchol == L.shape[0] == 63 and (eng.o, eng.v) == (4, 18)
    rng = np.random.default_rng(1)
    t1 = 0.05 * (rng.standard_normal((no, n - no)) + 1j * rng.standard_normal((no, n - no)))
    # the doubles in closed form from the dense dressed integrals, and the singles residual with them
    ge, _ = ccsdt._t1_dress(g, fock, t1, no, n)
    t2 = ge[no:, no:, :no, :no].transpose(2, 3, 0, 1) / R.denominator(mo_e, no, 2)
    ref = C.residuals('cc2', [t1, t2], fock, g, no)
    assert abs(ref[1]).max() < 1e-9
    R1, e = eng.residual(t1)
    assert abs(R1 - ref[0]).max() < 1e-12 and abs(e - R.energy(t1, t2, fock, g, no)) < 1e-13
    ec, t1c = eng.kernel(conv_tol=1e-13, conv_tol_normt=1e-11)
    assert eng.converged and abs(ec - res.e_corr.real) < 1e-11
    assert abs(t1c - res.t1).max() < 1e-9 and abs(eng.t2() - res.t2).max() < 1e-9
    # small batches of the pair loop and chunks of the vectors: the same numbers
    small = _engine(lih)
    small.plan.update(chunk=7, nj=2)
    R1b, eb = small.residual(t1)
    assert abs(R1b - R1).max() < 1e-13 and abs(eb - e) < 1e-14
    with pytest.raises(MemoryError, match='three-index arrays'):
        _engine(lih, max_memory=1e-3)


def test_sigma_and_roots(lih):
    mol, mf, eris, res, L = lih
    fock, g, mo_e, no = R.integrals(eris)
    nv = len(mo_e) - no
    eng = _engine(lih)
    eng.kernel(conv_tol=1e-13, conv_tol_normt=1e-11)
    # the folded matrix from the Jacobian on packed amplitudes with the doubles stored:
    # A_11 r1 + A_12 (omega - A_22)^-1 A_21 r1 by two of its sigma vectors
    em = E.from_scf(mf)
    gs = eom_opt.prepare(em, res.t1, res.t2, None, model='cc2')
    d2 = eom.denominators(gs)[1].reshape(gs.om.npair, gs.vm.npair)
    rng = np.random.default_rng(2)
    for omega in (0.0, 0.12, 0.3):
        r1 = rng.standard_normal((no, nv)) + 1j * rng.standard_normal((no, nv))
        s1, s2, _ = eom_opt.sigma(gs, r1, np.zeros_like(d2, dtype=complex))
        b1, _, _ = eom_opt.sigma(gs, np.zeros_like(r1), s2 / (omega - d2))
        want = s1 + b1
        assert abs(eng.sigma(r1, omega) - want).max() < 1e-9 * abs(want).max()
    small = _engine(lih)
    small.plan.update(chunk=5, nj=1)
    small.kernel(conv_tol=1e-13, conv_tol_normt=1e-11)
    assert abs(small.sigma(r1, 0.3) - eng.sigma(r1, 0.3)).max() < 1e-12
    # the roots: those of the Jacobian with the doubles stored (zccsdt.cc.eom); the lowest
    # level is a triplet (three roots), then a singlet
    stored = eom.kernel(em, res.t1, res.t2, None, nroots=8, conv_tol=1e-9, model='cc2')
    e, sol = eng.eom_ee(nroots=6, conv_tol=1e-9)
    assert sol.converged.all() and max(abs(stored.e - x).min() for x in e) < 1e-9
    assert abs(e[:3] - stored.e[0]).max() < 1e-9 and abs(e[3] - stored.e[3]) < 1e-9


def test_front_end(lih):
    '''DirectZCCSD: exact vectors, a frozen pair, and the decomposition at
    the default max_error (1e-5): 2e-7 Eh in the energy, 5e-7 in the
    excitation energies of LiH.'''
    mol, mf, eris, res, L = lih
    cc = DirectZCCSD(mf, ladder='chol', cderi=L)
    cc.verbose = 0
    cc.conv_tol, cc.conv_tol_normt = 1e-13, 1e-11
    e, t1 = cc.cc2()
    assert abs(e - res.e_corr.real) < 1e-11 and cc.e_cc2 == e and t1.shape == (4, 18)
    w, sol = cc.cc2_ee(nroots=4, conv_tol=1e-9)
    ref = eom.kernel(E.from_scf(mf), res.t1, res.t2, None, nroots=6, conv_tol=1e-9, model='cc2')
    assert sol.converged.all() and max(abs(ref.e - x).min() for x in w) < 1e-9
    default = DirectZCCSD(mf, ladder='chol')
    default.verbose = 0
    default.conv_tol, default.conv_tol_normt = 1e-13, 1e-11
    e5, _ = default.cc2()
    w5, _ = default.cc2_ee(nroots=4, conv_tol=1e-9)
    assert default._cc2.nchol < 63 and 1e-9 < abs(e5 - e) < 1e-6 and abs(w5 - w).max() < 2e-6
    # a frozen pair: against the CC2 on the integrals of the active spinors
    fz = DirectZCCSD(mf, frozen=2, ladder='chol', cderi=L)
    fz.verbose = 0
    fz.conv_tol, fz.conv_tol_normt = 1e-13, 1e-11
    ef, _ = fz.cc2()
    ref_f = solver.kernel(E.from_scf(mf, frozen=2), 'cc2', solver.Options(conv_tol=1e-13, conv_tol_normt=1e-11))
    assert abs(ef - ref_f.e_corr.real) < 1e-11 and abs(ef - e) > 1e-6
