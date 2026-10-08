'''The tables of the Lambda equations of CCSDTQ, of its densities and of its
orbital gradient (zccsdt.cc.lambda4_terms): against their two derivations
(zccsdt.cc.lambda_derive with nmax = 4: Wick's theorem, and the derivatives
of the Lagrangian made of the table of the amplitude equations), against
the tables of CCSDT, and against second-quantized operators in a Fock space
of 6 + 6 spin orbitals (tests/fockspace.py), for complex amplitudes that
solve nothing.'''

import numpy as np
import pytest

import fockspace as F
from zccsdt.cc import (ccsdt, ccsdtq_terms as Q, lambda4_ref as LR, lambda4_terms as L, lambda_derive as D,
                       lambda_terms as L3)

COUNTS = {1: 66, 2: 63, 3: 35, 4: 16}


def _with(terms, kind):
    return sum(1 for _, f in terms if f[0][0] == kind)


def test_tables_equal_derivations():
    '''Every table is what its derivation gives, coefficient by coefficient
    (rational numbers), and the two derivations of the Lambda equations
    agree: all contractions of <0| (1 + Lambda) (H tau T ..)_C |0> for tau
    of rank 1 .. 4 and T = T2 + T3 + T4, and the Lagrangian of the table of
    the amplitude equations of CCSDTQ differentiated with respect to t2, t3
    and t4 and one-index transformed for t1.'''
    for rank, n in COUNTS.items():
        wick = D.lambda_by_wick(rank, nmax=4)
        assert L.LAMBDA[rank] == wick and len(wick) == n
        assert D.lambda_by_differentiation(rank, nmax=4) == wick
    assert [_with(L.LAMBDA[1], 'l%d' % k) for k in (1, 2, 3, 4)] == [6, 13, 20, 26]
    assert [_with(L.LAMBDA[2], 'l%d' % k) for k in (1, 2, 3, 4)] == [3, 12, 19, 28]
    assert [_with(L.LAMBDA[3], 'l%d' % k) for k in (1, 2, 3, 4)] == [1, 3, 12, 19]
    assert [_with(L.LAMBDA[4], 'l%d' % k) for k in (1, 2, 3, 4)] == [0, 1, 3, 12]
    assert [len(L.quadruple(L.LAMBDA[k])) for k in (1, 2, 3, 4)] == [29, 28, 19, 12]
    assert L.DENSITY == D.densities(nmax=4)
    assert sum(len(v) for v in L.DENSITY.values()) == 76       # one for every term of the Lagrangian
    assert L.GRADIENT == D.one_index(excitation=False, nmax=4) and len(L.GRADIENT) == 168
    blocks = L.gradient_blocks()
    assert sum(len(v) for v in blocks.values()) == 168
    assert sorted(blocks) == ['f.oo', 'f.vo', 'f.vv', 'oo.oo', 'oo.vo', 'vo.oo', 'vo.vo', 'vo.vv',
                              'vv.oo', 'vv.vo', 'vv.vv']
    # the Lagrangian itself: the contractions of Wick's theorem are the table
    wick = D.collect([(c, f, '') for c, f in D.lagrangian(wick=True, nmax=4)], antisym=False)['']
    assert wick == D.collect([(c, f, '') for c, f in D.lagrangian(nmax=4)], antisym=False)['']
    assert len(wick) == 76


def test_ccsdt_limit():
    '''Without the terms of t4 and l4 the tables are those of CCSDT, in
    the same order; what is left of LAMBDA[4] is <0| (Lambda_2 + Lambda_3)
    H |Q>: l2 <kl||cd>, l3 f[l,d] and the two connected products of l3 with
    the integrals (the left-hand quadruples of CCSDT(Q)_Lambda, and their
    term of f[i,a]).'''
    lam, dens, grad = L.ccsdt_limit()
    for rank in (1, 2, 3):
        assert lam[rank] == L3.LAMBDA[rank]
    assert dens == L3.DENSITY and list(dens) == list(L3.DENSITY)
    assert grad == L3.GRADIENT
    assert sorted(f for _, f in lam[4]) == sorted([
        (('l2', 'ijab'), ('v', 'klcd')), (('l3', 'ijkabc'), ('f', 'ld')),
        (('l3', 'ijmabc'), ('v', 'kldm')), (('l3', 'ijkabe'), ('v', 'elcd'))])
    # the same numbers: the dense evaluation with t4 = l4 = 0
    no, nv = 4, 4
    f, v, amps, lams = _random(no, nv, 5)
    amps[3][:], lams[3][:] = 0, 0
    got = LR.residuals(lams, amps, f, v, no)
    from zccsdt.cc import lambda_ref as LR3
    want = LR3.residuals(lams[:3], amps[:3], f, v, no)
    for k in range(3):
        assert abs(got[k] - want[k]).max() < 1e-13 * abs(want[k]).max()
    for a, b in zip(LR.densities(amps, lams, no), LR3.densities(amps[:3], lams[:3], no)):
        assert abs(a - b).max() < 1e-13 * abs(b).max()


def test_products_in_slabs(monkeypatch):
    '''The evaluation for large arrays - every product with t4 or l4 for
    one value of an occupied index at a time - gives the residuals, the
    densities and the gradient of the plain one.'''
    no, nv = 5, 4
    f, v, amps, lams = _random(no, nv, 7)
    want = (LR.residuals(lams, amps, f, v, no), LR.densities(amps, lams, no),
            LR.orbital_gradient(f, v, no, amps, lams, dressed=True))
    monkeypatch.setattr(L, 'SLABS', 1)
    got = (LR.residuals(lams, amps, f, v, no), LR.densities(amps, lams, no),
           LR.orbital_gradient(f, v, no, amps, lams, dressed=True))
    for a, b in zip(want[0] + list(want[1]) + [want[2]], got[0] + list(got[1]) + [got[2]]):
        assert abs(a - b).max() < 1e-13 * abs(a).max()


SCALES = ((1, 0.3), (2, 0.2), (3, 0.1), (4, 0.05))


def _random(no, nv, seed, hermitian=True):
    rng = np.random.default_rng(seed)
    n = no + nv
    v = F.hermitian_integrals(rng, n)
    f = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    f = f + f.conj().T
    if not hermitian:
        v = v + 0.3j * F.hermitian_integrals(rng, n)
        f = f + 0.3 * (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    amps = [F.complex_antisym(rng, no, nv, k, s) for k, s in SCALES]
    lams = [F.complex_antisym(rng, no, nv, k, s) for k, s in SCALES]
    return f, v, amps, lams


@pytest.mark.parametrize('no,nv,hermitian', [(6, 6, True), (5, 6, False), (6, 4, False)])
def test_tables_equal_operators(no, nv, hermitian):
    '''For complex t1 .. t4, l1 .. l4 that solve nothing and complex f and
    <pq||rs> (Hermitian or not: nothing in the tables is conjugated):

        LAMBDA[k]  = <0| (1 + Lambda) [Hbar, tau_k] |0>,   k = 1 .. 4,
        densities  = <0| (1 + Lambda) exp(-T) p+ q exp(T) |0>, .. p+ q+ s r ..,
        Lagrangian = sum h d1 + 1/4 sum <pq||rs> d2,
        GRADIENT   = <0| (1 + Lambda) exp(-T2-T3-T4) [a+_i a_a, H~] exp(T2+T3+T4) |0>,

    and the same commutator with the bare H from the densities.  6 + 6
    spin orbitals: every term of the tables has at most six occupied and
    six virtual letters, so none of them is restricted there (the space in
    which the amplitude equations were checked).'''
    f, v, amps, lams = _random(no, nv, 3, hermitian)
    fk = F.Fock(no, nv)
    ex = F.Left(fk, f, v, amps)
    lam = ex.vector(lams)
    want = ex.tensors(ex.lambda_residual(lam))
    got = LR.residuals(lams, amps, f, v, no)
    for k in range(4):
        assert abs(got[k] - want[k]).max() < 1e-12 * abs(want[k]).max(), k
    d1, d2 = LR.densities(amps, lams, no)
    e1, e2 = ex.rdm1(lam), ex.rdm2(lam)
    assert abs(d1 - e1).max() < 1e-12 * abs(e1).max()
    assert abs(d2 - e2).max() < 1e-12 * abs(e2).max()
    lagrangian = ex.bra(lam) @ ex.hbar(fk.ref)
    assert abs(LR.energy(f, v, no, d1, d2) - lagrangian) < 1e-12 * abs(lagrangian)
    # the orbital gradient, bare: [a+_i a_a, H] between the two states
    bra = ex.exp(ex.bra(lam), -1, left=True)
    want = np.zeros((nv, no), dtype=complex)
    for a in range(nv):
        for i in range(no):
            E = fk.ad[i] @ fk.a[no + a]
            want[a, i] = bra @ ((E @ ex.H - ex.H @ E) @ ex.right)
    got = LR.orbital_gradient(f, v, no, amps, lams)
    assert abs(got - want).max() < 1e-12 * abs(want).max()
    # and of the dressed basis: T1 in the Hamiltonian
    t1 = F.Left(fk, f, v, amps[:1])
    t234 = F.Left(fk, f, v, amps[1:])
    bra = t234.exp(ex.bra(lam), -1, left=True)
    ket = t234.right
    want = np.zeros((nv, no), dtype=complex)
    for a in range(nv):
        for i in range(no):
            E = fk.ad[i] @ fk.a[no + a]
            want[a, i] = bra @ (E @ t1.hbar(ket) - t1.hbar(E @ ket))
    got = LR.orbital_gradient(f, v, no, amps, lams, dressed=True)
    assert abs(got - want).max() < 1e-12 * abs(want).max()
    # the Lambda_1 equation is the other half: the one-index transformation
    # of the dressed integrals by a+_a a_i with the dressed densities
    ge, fdr = ccsdt._t1_dress(v, f, amps[0], no, no + nv)
    hd = fdr - np.einsum('piqi->pq', ge[:, :no, :, :no])
    G = LR.one_index(hd, ge, *LR.dressed_densities(amps, lams, no))
    r1 = LR.residuals(lams, amps, f, v, no)[0]
    assert abs(G[no:, :no].T - r1).max() < 1e-12 * abs(r1).max()
    assert abs(-G[:no, no:].T - got).max() < 1e-12 * abs(got).max()


def test_every_term_counts():
    '''No term of a table is a combination of the others (none is written
    twice in another form), and none vanishes: the terms as vectors, for
    two sets of random amplitudes and integrals in 6 + 6 spin orbitals,
    have full rank.'''
    no = nv = 6
    cols = {}
    for seed in range(2):
        f, v, amps, lams = _random(no, nv, 20 + seed)
        arrays = LR._arrays(f, v, no, amps, lams)
        for rank in (1, 2, 3, 4):
            ext = Q.EXT[rank]
            for n, (_, fac) in enumerate(L.LAMBDA[rank]):
                x = L.product(fac, arrays, no, ext)
                x = x if rank == 1 else Q.asym_unique(x, rank)
                cols.setdefault(('lambda', rank, n), []).append(x.ravel())
        for n, (_, fac) in enumerate(L.GRADIENT):
            cols.setdefault(('gradient', 0, n), []).append(L.product(fac, arrays, no, 'ai').ravel())
    for name, rank, count in [('lambda', k, n) for k, n in COUNTS.items()] + [('gradient', 0, 168)]:
        A = np.array([np.concatenate(cols[name, rank, n]) for n in range(count)]).T
        s = np.linalg.svd(A / np.linalg.norm(A, axis=0), compute_uv=False)
        assert s[-1] > 1e-6 * s[0], (name, rank, s[-1] / s[0])
