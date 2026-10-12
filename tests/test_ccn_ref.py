'''The definitions of CC2, CC3 and CC4 (zccsdt.cc.ccn_ref, docs/ccn.md):

- the equations are the commutators they are named after, by second-quantized
  operators in a small Fock space (tests/fockspace.py), for a complex
  Hamiltonian whose Fock matrix is neither diagonal nor block diagonal and
  complex amplitudes that solve nothing;
- the Jacobian is the derivative of the residuals;
- limits: two electrons (CCSD exact, CC2 not), three electrons (CCSDT exact,
  CC3 not), and the models are ordered in the correlation energy they miss;
- the folded eigenproblem has the eigenvalues of the full matrix;
- H2O / STO-3G against MRCC (CC2, CC3, CC4; its closed-shell code on its own
  integrals).'''
import numpy as np
import pytest
from pyscf import gto, scf

import fockspace as F
from conftest import water_ghf
from test_lambda_ref import _model
from zccsdt import eris as E
from zccsdt.cc import ccn_ref as C, ccsdtq_ref as R, eom_ref, qlgrad_ref


def _case(model, seed=3):
    n = C.NRANK[model]
    no, nv = (4, 4) if n == 4 else (3, 4)
    f, v = _model(no, nv)
    rng = np.random.default_rng(seed)
    amps = [F.complex_antisym(rng, no, nv, k + 1, s) for k, s in enumerate((0.2, 0.1, 0.05, 0.03)[:n])]
    return no, nv, f, v, amps, F.Fock(no, nv)


@pytest.mark.parametrize('model', ['cc2', 'cc3', 'cc4'])
def test_equations_are_the_commutators(model):
    '''The equation of the highest rank n is <mu_n| [F, T_n] + [V~, T_(n-1)]
    (+ 1/2 [[V~, T2], T2] for CC4; V~ itself for CC2) |0> with V~ =
    exp(-T1) V exp(T1) and F the Fock operator as it is; the equations
    below are the projections of exp(-T) H exp(T).'''
    no, nv, f, v, amps, fk = _case(model)
    n = len(amps)
    res = C.residuals(model, amps, f, v, no)
    ex = F.Left(fk, f, v, amps)
    full = ex.residuals()
    off = 0
    for k in range(n - 1):                      # the lower ranks: the full equations
        u = res[k].ravel() if k == 0 else C.Q.unique(res[k], k + 1).ravel()
        assert abs(u - full[off:off + u.size]).max() < 1e-12
        off += u.size
    ref = fk.ref.astype(complex)
    T = [fk.excitation(t).tocsr() for t in amps]
    Fop, V = F.one_body(fk, f).tocsr(), fk.two_body(v).tocsr()
    t1 = F.Left(fk, f, v, amps[:1])             # exp(-T1) X exp(T1)

    def dressed(x):
        return t1.exp(V @ t1.exp(x, 1), -1)

    def comm(A, B, x):
        return A(B @ x) - B @ A(x)
    top = Fop @ (T[n - 1] @ ref) - T[n - 1] @ (Fop @ ref)
    if n == 2:
        top = top + dressed(ref)
    else:
        top = top + comm(dressed, T[n - 2], ref)
    if n == 4:
        top = top + 0.5 * (comm(dressed, T[1], T[1] @ ref) - T[1] @ comm(dressed, T[1], ref))
    want = fk_project(fk, top, n)
    assert abs(res[n - 1] - want).max() < 1e-12
    # the other choices of the Fock operator differ for this Fock matrix (f_ia is not zero)
    one = C.residuals(model, amps, f, v, no, fock_top='one')[n - 1]
    top1 = top + t1.exp(Fop @ t1.exp(T[n - 1] @ ref, 1), -1) - T[n - 1] @ t1.exp(Fop @ t1.exp(ref, 1), -1) \
        - (Fop @ (T[n - 1] @ ref) - T[n - 1] @ (Fop @ ref))
    assert abs(one - fk_project(fk, top1, n)).max() < 1e-12
    assert abs(one - res[n - 1]).max() > 1e-5
    assert abs(C.residuals(model, amps, f, v, no, fock_top='dressed')[n - 1] - one).max() > 1e-6


def fk_project(fk, vec, k):
    '''Fock.project for a complex vector.'''
    return fk.project(vec.real, k) + 1j * fk.project(vec.imag, k)


@pytest.mark.parametrize('model', ['cc2', 'cc3', 'cc4'])
def test_jacobian_is_the_derivative(model):
    no, nv, f, v, amps, _ = _case(model)
    rng = np.random.default_rng(5)
    r = [F.complex_antisym(rng, no, nv, k + 1) for k in range(len(amps))]
    s = C.sigma(model, r, amps, f, v, no)
    d = qlgrad_ref.directional(lambda a: C.residuals(model, a, f, v, no), list(amps), list(r))
    for a, b in zip(s, d):
        assert abs(a - b).max() < 1e-10 * max(1.0, abs(b).max())


def _fci(mf):
    '''The eigenvalues of H among the determinants of the electrons of a
    GHF / X2C object, by the operators of the Fock space of its spin
    orbitals (a few of them).'''
    eris = E.from_scf(mf, packed_vvvv=False)
    fock, g, mo_e, no = R.integrals(eris)
    fk = F.Fock(no, len(mo_e) - no)
    H = (F.one_body(fk, fock) + fk.two_body(g)).toarray()
    N = sum((fk.ad[p] @ fk.a[p]) for p in range(fk.n)).diagonal()
    sel = np.flatnonzero(abs(N - no) < 1e-9)
    w = np.linalg.eigvalsh(H[np.ix_(sel, sel)])
    return eris, w - (fk.ref @ (H @ fk.ref))


def test_two_and_three_electrons():
    '''He / 6-31G (2 + 2 spin orbitals): CCSD is exact, CC2 is not.
    H3 / 6-31G (3 + 9; with 3 + 3 no triple conserves S_z): CCSDT is exact, CC3 is not, and the eigenvalues
    of the CCSDT Jacobian are the excitation energies of FCI while those of
    CC3 are near them.'''
    he = scf.addons.convert_to_ghf(scf.RHF(gto.M(atom='He', basis='6-31g', verbose=0)).run(conv_tol=1e-13))
    eris, w = _fci(he)
    fock, g, mo_e, no = R.integrals(eris)
    ccsd = C.kernel(fock, g, mo_e, no, 'ccsd')
    cc2 = C.kernel(fock, g, mo_e, no, 'cc2')
    assert abs(ccsd.e_corr - w[0]) < 1e-10 and 1e-4 < abs(cc2.e_corr - w[0]) < 1e-2
    om = C.eigenvalues('ccsd', ccsd.amps, fock, g, no).real
    assert abs(np.sort(om) - (w[1:] - w[0])).max() < 1e-8
    h3 = gto.M(atom='H 0 0 0; H 0 0 0.9; H 0 0.8 1.7', basis='6-31g', spin=1, verbose=0)
    eris, w = _fci(scf.addons.convert_to_ghf(scf.UHF(h3).run(conv_tol=1e-13)))
    fock, g, mo_e, no = R.integrals(eris)
    ccsdt = C.kernel(fock, g, mo_e, no, 'ccsdt')
    cc3 = C.kernel(fock, g, mo_e, no, 'cc3')
    assert abs(ccsdt.e_corr - w[0]) < 1e-10 and 1e-8 < abs(cc3.e_corr - w[0]) < 1e-3
    w3 = C.eigenvalues('cc3', cc3.amps, fock, g, no).real
    wt = C.eigenvalues('ccsdt', ccsdt.amps, fock, g, no).real
    assert abs(np.sort(wt) - (w[1:] - w[0])).max() < 1e-8
    assert 1e-8 < abs(w3[:6] - wt[:6]).max() < 1e-1


@pytest.fixture(scope='module')
def water():
    '''H2O / STO-3G, all electrons (10 + 4): the ground states of the six
    models.'''
    eris = E.from_scf(water_ghf('sto-3g'), packed_vvvv=False)
    fock, g, mo_e, no = R.integrals(eris)
    out = {m: C.kernel(fock, g, mo_e, no, m) for m in ('cc2', 'ccsd', 'cc3', 'ccsdt', 'cc4', 'ccsdtq')}
    return eris, out


def test_water_against_mrcc(water):
    '''MRCC (release of 2018; closed-shell code, its own SCF and integrals,
    the same basis), totals to 1e-10.  MRCC has no excited states for the
    approximate models and no CCn for the reference of its interface to
    DIRAC: docs/ccn.md.'''
    eris, res = water
    mrcc = {'cc2': -74.998753128797, 'ccsd': -75.012530625524, 'cc3': -75.012598223465,
            'ccsdt': -75.012623761491, 'cc4': -75.012639672408, 'ccsdtq': -75.012647118993}
    assert abs(eris.e_hf + 74.963063129729) < 1e-9
    for m, ref in mrcc.items():
        assert res[m].converged
        assert abs(eris.e_hf + res[m].e_corr - ref) < 1e-10, m
    # each model of the ladder is nearer to the next full one than the one before
    e = {m: res[m].e_corr for m in res}
    assert abs(e['cc3'] - e['ccsdt']) < abs(e['ccsd'] - e['ccsdt'])
    assert abs(e['cc4'] - e['ccsdtq']) < abs(e['ccsdt'] - e['ccsdtq'])


@pytest.mark.parametrize('model', ['cc2', 'cc3', 'cc4'])
def test_folded_eigenproblem(water, model):
    '''The omega-dependent eigenproblem in the ranks below the highest,
    A_PP + A_PQ (omega - A_QQ)^-1 A_QP, iterated to self-consistency from
    the eigenvalues of A_PP, gives eigenvalues of the full Jacobian; A_QQ
    is the diagonal of the differences of the orbital energies.'''
    if model == 'cc4':
        # (a model Hamiltonian of 4 + 4 spin orbitals with one quadruple; its Fock matrix
        # is not diagonal, and A_QQ is the Fock operator on the quadruples)
        no = 4
        fock, g = _model(4, 4)
        mo_e = fock.diagonal().real.copy()
        amps = C.kernel(fock, g, mo_e, no, 'cc4').amps
    else:
        eris, res = water
        fock, g, mo_e, no = R.integrals(eris)
        amps = res[model].amps
    n = len(amps)
    M = C.matrix(model, amps, fock, g, no)
    sizes = eom_ref.sizes(no, len(mo_e) - no, n)
    nlow = sum(sizes[:-1])
    d = eom_ref.diagonal(mo_e, no, n)[nlow:]
    if model != 'cc4':
        assert abs(M[nlow:, nlow:] - np.diag(d)).max() < 1e-10
    w = np.linalg.eigvals(M)
    w = np.sort(w.real)
    w0 = np.sort(np.linalg.eigvals(M[:nlow, :nlow]).real)
    for k in (0, 3, 6):
        om, it = C.folded_root(M, nlow, w0[k])
        assert abs(w - om.real).min() < 1e-10 and it < 40
