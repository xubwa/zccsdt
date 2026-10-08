'''The dense reference of the first derivatives of CCSDT(Q)_Lambda
(zccsdt.cc.qlgrad_ref; docs/lambda.md, section 12):

- the derivatives of e = sum z4 r4 / D4 with respect to t2, t3, l2, l3, the
  integrals and the orbital energies against differences of the dense
  function, real and complex (holomorphic: complex numbers are compared);
- the Jacobian of the amplitude equations on a vector and the second
  derivative of the Lagrangian of CCSDT contracted with it, against
  differences of the residuals, and one against the other;
- a model Hamiltonian (complex Hermitian, not diagonal in f): the Lagrangian
  is stationary in t, lambda, zeta and lambda', and its densities give the
  derivative of E(CCSDT) + e with the amplitude and the Lambda equations
  solved again for every Hamiltonian, at fixed orbitals;
- the relaxed dipole derivative against E(SCF) + E_corr(CCSDT) + E(Q)_L with
  the SCF repeated in the field, with complex, frozen and dropped spin
  orbitals.'''
import numpy as np
import pytest

import fockspace as F
from test_qgrad_ref import _fd, _molecule
from zccsdt import eris as E
from zccsdt.cc import ccsdtq_ref as R, ccsdtq_terms as Q, lambda_ref as LR, qgrad_ref as G, qlgrad_ref as QL


@pytest.mark.parametrize('cplx', [False, True])
def test_derivatives_of_the_correction(cplx):
    '''de = sum_(i<j,a<b) (s2 dt2 + q2 dl2) + sum (s3 dt3 + q3 dl3) + sum d2
    d<pq||rs>, and dE(Q)_L = sum de d e_p, for amplitudes that solve nothing
    and integrals that are not Hermitian: four-point differences of the
    dense e along random directions.  For Hermitian integrals e is the
    number of lambda_ref.quadruples_lambda.'''
    no, nv = 5, 5
    n = no + nv
    rng = np.random.default_rng(3)
    if cplx:
        mk = lambda k, s: F.complex_antisym(rng, no, nv, k, s)             # noqa: E731
    else:
        mk = lambda k, s: s * F.random_antisym(rng, no, nv, k)              # noqa: E731

    def mg():           # antisymmetric in (p,q) and in (r,s), nothing else
        x = rng.standard_normal((n,) * 4) + (1j * rng.standard_normal((n,) * 4) if cplx else 0)
        x = x - x.transpose(1, 0, 2, 3)
        return x - x.transpose(0, 1, 3, 2)
    t2, t3, l2, l3, g = mk(2, 0.2), mk(3, 0.1), mk(2, 0.2), mk(3, 0.1), 0.3 * mg()
    assert abs(g - g.conj().transpose(2, 3, 0, 1)).max() > 0.1
    e = np.concatenate([-1 - rng.random(no), 1 + rng.random(nv)])
    gr = QL.quadruples_gradient(t2, t3, l2, l3, g, e, no)
    f = QL.energy
    assert abs(gr.e - f(t2, t3, l2, l3, g, e, no)) < 1e-12 * abs(gr.e)
    assert np.iscomplexobj(gr.s3) == cplx and np.iscomplexobj(gr.q3) == cplx
    d2, d3, dg, de = mk(2, 1.0), mk(3, 1.0), mg(), rng.standard_normal(n)
    checks = [(lambda x: f(t2 + x * d2, t3, l2, l3, g, e, no), np.einsum('ijab,ijab', gr.s2, d2) / 4),
              (lambda x: f(t2, t3 + x * d3, l2, l3, g, e, no), np.einsum('ijkabc,ijkabc', gr.s3, d3) / 36),
              (lambda x: f(t2, t3, l2 + x * d2, l3, g, e, no), np.einsum('ijab,ijab', gr.q2, d2) / 4),
              (lambda x: f(t2, t3, l2, l3 + x * d3, g, e, no), np.einsum('ijkabc,ijkabc', gr.q3, d3) / 36),
              (lambda x: f(t2, t3, l2, l3, g + x * dg, e, no), np.einsum('pqrs,pqrs', gr.d2, dg)),
              (lambda x: f(t2, t3, l2, l3, g, e + x * de, no).real, gr.de @ de)]
    for fx, analytic in checks:
        assert abs(analytic) > 1e-2
        assert abs(_fd(fx, 1e-3) - analytic) < 2e-9 * abs(analytic)
    for s in (gr.s2, gr.q2):
        assert abs(s + s.transpose(1, 0, 2, 3)).max() < 1e-13 * abs(s).max()
    for s in (gr.s3, gr.q3):
        assert abs(s + s.transpose(0, 1, 2, 4, 3, 5)).max() < 1e-13 * abs(s).max()
    # Hermitian integrals: the energy of lambda_ref, which conjugates lambda twice
    gh = g + g.conj().transpose(2, 3, 0, 1)
    want = LR.quadruples_lambda(t2, t3, l2, l3, gh, e, no)[1]
    assert abs(f(t2, t3, l2, l3, gh, e, no).real - want) < 1e-12 * abs(want)
    # the source of zeta is what T4 = r4 / D4 adds to the doubles and triples equations of CCSDTQ
    x, d = R._pieces(QL.RIGHT, 4, {'v': g, 't2': t2, 't3': t3}, e, no)
    arrays = {'v': g, 't4': Q.expand(sum(x.values()) / d, no, nv, 4)}
    for k, q in ((2, gr.q2), (3, gr.q3)):
        want = Q.asym(Q.total(Q.GROUPS['R%d: [V,T4]' % k][4], arrays, no, k), k)
        assert abs(want - q).max() < 1e-12 * abs(q).max()


def _model(no, nv, seed=1, scale=0.04):
    '''A complex Hermitian Hamiltonian with a gap, not diagonal in f (t1 is
    not small): (f, <pq||rs>), as in test_lambda_ref with more correlation.'''
    n = no + nv
    rng = np.random.default_rng(seed)
    v = scale * F.hermitian_integrals(rng, n)
    e = np.concatenate([-1.0 - 0.3 * rng.random(no), 1.0 + 0.3 * rng.random(nv)])
    x = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return np.diag(e) + 0.05 * (x + x.conj().T), v


def _solve(f, v, mo_e, no):
    r = R.kernel(f, v, mo_e, no, 'ccsdt', conv_tol=1e-14, conv_tol_normt=1e-12)
    rl = LR.kernel(f, v, mo_e, no, r.amps, conv_tol_normt=1e-12)
    assert r.converged and rl.converged
    return r.amps, rl.lams


@pytest.fixture(scope='module')
def model():
    no, nv = 4, 5
    f, v = _model(no, nv)
    mo_e = f.diagonal().real
    amps, lams = _solve(f, v, mo_e, no)
    return no, nv, f, v, mo_e, amps, lams, QL.kernel(f, v, mo_e, no, amps, lams, 1e-12)


def test_directional_derivatives(model):
    '''The Jacobian of the amplitude equations on a vector and the second
    derivative of the Lagrangian on it: the five-point formula with step 1
    is the derivative (differences with small steps agree to their own
    error); three evaluations in the dressed formalism give the same
    Jacobian; and zeta . (J^T lambda) = lambda . (J zeta) with J^T lambda
    the part of the Lambda residuals that is linear in lambda.'''
    no, nv, f, v, mo_e, amps, lams, _ = model
    rng = np.random.default_rng(5)
    zeta = [F.complex_antisym(rng, no, nv, k, 0.1) for k in (1, 2, 3)]

    def small(fx, h=1e-2):
        at = [fx([t + x * h * z for t, z in zip(amps, zeta)]) for x in (1, -1, 2, -2)]
        return [(8 * (a - b) - (c - d)) / (12 * h) for a, b, c, d in zip(*at)]
    J = QL.jacobian(zeta, amps, f, v, no)
    for a, b, c in zip(J, small(lambda t: R.residuals(t, f, v, no)), QL.jacobian_dressed(zeta, amps, f, v, no)):
        assert abs(a).max() > 0.1 and abs(a - b).max() < 1e-12 and abs(a - c).max() < 1e-13
    W = QL.lambda_response(zeta, amps, lams, f, v, no)
    for a, b in zip(W, small(lambda t: LR.residuals(lams, t, f, v, no))):
        assert abs(a).max() > 1e-3 and abs(a - b).max() < 1e-12
    W0 = LR.residuals([0 * x for x in lams], amps, f, v, no)
    W1 = LR.residuals(lams, amps, f, v, no)
    lhs = sum(np.sum(z * (a - b)) / k for z, a, b, k in zip(zeta, W1, W0, (1, 4, 36)))
    rhs = sum(np.sum(x * j) / k for x, j, k in zip(lams, J, (1, 4, 36)))
    assert abs(lhs) > 1e-3 and abs(lhs - rhs) < 1e-14


def test_lagrangian_is_stationary(model):
    '''F = L(t, lambda') + e(t, lambda) + zeta . W(t, lambda) at the solution
    of the four sets of equations: its value is E(CCSDT) + e, and its
    derivatives along random directions of t, lambda, zeta and lambda'
    vanish - none of them does when zeta or lambda-bar are left out.'''
    no, nv, f, v, mo_e, amps, lams, T = model
    rng = np.random.default_rng(7)
    e_cc = np.trace(f[:no, :no]) - 0.5 * np.einsum('ijij', v[:no, :no, :no, :no]) + R.energy(amps[0], amps[1], f, v, no)
    assert abs(T.e) > 3e-5 and abs(T.e.imag) > 1e-6
    assert abs(QL.lagrangian(f, v, mo_e, no, amps, lams, T.zeta, T.lams) - e_cc - T.e) < 1e-12
    zero = [0 * z for z in T.zeta]
    for which in range(4):
        d = [F.complex_antisym(rng, no, nv, k, 1.0) for k in (1, 2, 3)]

        def fx(x, zeta=T.zeta, lams2=T.lams):
            args = [amps, lams, zeta, lams2]
            args[which] = [a + x * b for a, b in zip(args[which], d)]
            return QL.lagrangian(f, v, mo_e, no, *args)
        assert abs(_fd(fx, 1e-3)) < 2e-11
        if which == 0:      # in t: without zeta, without lambda-bar
            assert abs(_fd(lambda x: fx(x, zeta=zero), 1e-3)) > 1e-6
            assert abs(_fd(lambda x: fx(x, lams2=lams), 1e-3)) > 1e-6
        if which == 1:      # in lambda: without zeta
            assert abs(_fd(lambda x: fx(x, zeta=zero), 1e-3)) > 1e-6


def test_densities_are_derivatives(model):
    '''dE/dx of E(CCSDT) + e, t and lambda solved again for every
    Hamiltonian at fixed orbitals, for a one-electron and a two-electron
    perturbation (complex Hermitian) and for the orbital energies of the
    denominators: the traces with the densities of the Lagrangian.  The
    part of e alone, which for a one-electron perturbation is all zeta and
    lambda-bar, is compared too.'''
    no, nv, f, v, mo_e, amps, lams, T = model
    n = no + nv
    rng = np.random.default_rng(9)

    def energies(ff, vv, ee):
        a, x = _solve(ff, vv, mo_e, no)
        e_cc = np.trace(ff[:no, :no]) - 0.5 * np.einsum('ijij', vv[:no, :no, :no, :no]) + R.energy(a[0], a[1], ff, vv, no)
        return np.array([e_cc, QL.energy(a[1], a[2], x[1], x[2], vv, ee, no)])
    x = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    one = x + x.conj().T
    two = F.hermitian_integrals(rng, n)
    fock2 = np.einsum('piqi->pq', two[:, :no, :, :no])          # h stays: f changes with <pq||rs>
    c1, c2 = LR.densities(amps, lams, no)                       # CCSDT alone
    cases = [(one, 0 * two, np.einsum('pq,pq', one, c1), np.einsum('pq,pq', one, T.d1)),
             (fock2, two, 0.25 * np.einsum('pqrs,pqrs', two, c2),
              0.25 * np.einsum('pqrs,pqrs', two, T.d2) + np.einsum('pqrs,pqrs', two, T.q.d2))]
    for df, dv, ccsdt, total in cases:
        fd = [_fd(lambda x: energies(f + x * df, v + x * dv, mo_e), eps) for eps in (4e-4, 2e-4)]
        fd = (16 * fd[1] - fd[0]) / 15
        assert abs(total - ccsdt) > 1e-5 and abs((total - ccsdt).imag) > 1e-7
        assert abs(fd[0] - ccsdt) < 1e-10 * abs(ccsdt)
        assert abs(fd[1] - (total - ccsdt)) < 1e-6 * abs(total - ccsdt)
    de = rng.standard_normal(n)
    fd = _fd(lambda x: energies(f, v, mo_e + x * de)[1].real, 1e-3)
    assert abs(T.q.de @ de) > 1e-5 and abs(fd - T.q.de @ de) < 1e-11


def _energy(make_scf, eps, nfrozen, ndrop):
    '''E(SCF) + E_corr(CCSDT) + E(Q)_L in the field, all dense.'''
    mf, C = make_scf(eps)
    fock, g, mo_e, o = R.integrals(E.from_scf(mf, mo_coeff=C))
    a = slice(nfrozen, len(mo_e) - ndrop)
    fa, ga, ea, oa = fock[a, a], g[a, a, a, a], mo_e[a], o - nfrozen
    amps, lams = _solve(fa, ga, ea, oa)
    return (mf.e_tot + float(np.real(R.energy(amps[0], amps[1], fa, ga, oa)))
            + LR.quadruples_lambda(amps[1], amps[2], lams[1], lams[2], ga, ea, oa)[1])


@pytest.mark.parametrize('name,cplx,nfrozen,ndrop', [('h2o', False, 0, 0), ('bh', True, 2, 2)])
def test_relaxed_dipole_against_finite_field(name, cplx, nfrozen, ndrop):
    '''The trace of z with the relaxed density of CCSDT(Q)_Lambda is the
    derivative of E(SCF) + E_corr(CCSDT) + E(Q)_L, the SCF repeated and
    Lambda solved again in the field (four-point differences).  The
    densities of CCSDT and of CCSDT(Q) are other numbers.'''
    mol, make_scf, z = _molecule(name, cplx)
    mf, C = make_scf(0.0)
    zmo = C.conj().T @ z @ C
    rel = QL.relaxed_rdm1(mf, C, nfrozen, ndrop)
    assert abs(rel.relaxed - rel.relaxed.conj().T).max() < 1e-13
    assert abs(np.trace(rel.relaxed) - mol.nelectron) < 1e-10
    value = np.einsum('pq,qp', zmo, rel.relaxed).real
    fd = _fd(lambda x: _energy(make_scf, x, nfrozen, ndrop), 1e-3)
    print('%s %s %d frozen %d dropped: relaxed %.12f finite field %.12f, difference %.1e'
          % (name, 'complex' if cplx else 'real', nfrozen, ndrop, value, fd, value - fd))
    assert abs(value - fd) < 2e-10
    for with_q in (False, True):
        other = G.relaxed_rdm1(mf, C, nfrozen, ndrop, with_q=with_q, amps=rel.amps)
        assert abs(np.einsum('pq,qp', zmo, other.relaxed).real - value) > (1e-7 if with_q else 1e-6)


@pytest.mark.parametrize('name', ['lih', 'bh'])
def test_gradient_among_occupied_and_among_virtual(name):
    '''The orbital gradient of CCSDT(Q)_Lambda among the occupied and among
    the virtual spin orbitals (complex ones), from the densities of the
    Lagrangian - those of lambda', of zeta and of e:

    - the whole of it is (e_q - e_p) times the Hermitian part of the
      Fock-matrix density of (Q)_Lambda, transposed, as for CCSDT(Q);
    - the part without the densities of e follows from the sources of the
      two equations, s = de/dt with t and q = de/dlambda with lambda: L and
      zeta . W do not change when t, lambda, zeta, lambda' and the integrals
      are rotated together.'''
    mol, make_scf, z = _molecule(name, True)
    mf, C = make_scf(0.0)
    rel = QL.relaxed_rdm1(mf, C)
    fock, g, mo_e, o = R.integrals(E.from_scf(mf, mo_coeff=C))
    t2, t3, l2, l3 = rel.amps[1], rel.amps[2], rel.lams[1], rel.lams[2]
    gam = QL.fock_density(t2, t3, l2, l3, g, mo_e, o)
    q = rel.terms.q
    assert abs(gam.diagonal().real - q.de).max() < 1e-14
    want = (mo_e[None, :] - mo_e[:, None]) * 0.5 * (gam.T + gam.conj())
    es = np.einsum
    nh = np.zeros_like(want)
    nh[:o, :o] = (-0.5 * es('jkab,ikab->ji', q.s2, t2) - es('jklabc,iklabc->ji', q.s3, t3) / 12
                  + 0.5 * es('ikab,jkab->ji', q.q2, l2) + es('iklabc,jklabc->ji', q.q3, l3) / 12)
    nh[o:, o:] = (0.5 * es('ijac,ijbc->ba', q.s2, t2) + es('ijkacd,ijkbcd->ba', q.s3, t3) / 12
                  - 0.5 * es('ijbc,ijac->ba', q.q2, l2) - es('ijkbcd,ijkacd->ba', q.q3, l3) / 12)
    lag = rel.gradient - rel.gradient_e
    size = [0.0, 0.0]
    for s in (slice(0, o), slice(o, None)):
        assert abs(rel.gradient[s, s] - want[s, s]).max() < 1e-11
        assert abs(lag[s, s] - 0.5 * (nh - nh.conj().T)[s, s]).max() < 1e-11
        size = [max(size[0], abs(want[s, s]).max()), max(size[1], abs(lag[s, s]).max())]
    assert size[0] > 1e-9 and size[1] > 1e-8
