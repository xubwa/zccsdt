'''The tables of the Lambda equations of CCSDT, of the densities and of the
orbital gradient (zccsdt.cc.lambda_terms): against their two derivations
(zccsdt.cc.lambda_derive: Wick's theorem, and the derivatives of the
Lagrangian made of the table of the amplitude equations) and against
second-quantized operators in a small Fock space (tests/fockspace.py), for
complex amplitudes that solve nothing.'''

import numpy as np
import pytest

import fockspace as F
from zccsdt.cc import ccsdt, ccsdtq_terms as Q, lambda_derive as D, lambda_ref as LR, lambda_terms as L


def test_tables_equal_derivations():
    '''Every table is what its derivation gives, coefficient by coefficient
    (rational numbers), and the two derivations of the Lambda equations
    agree: all contractions of <0| (1 + Lambda) (H tau T ..)_C |0>, and the
    Lagrangian of the table of the amplitude equations differentiated with
    respect to t2 and t3 and one-index transformed for t1.'''
    for rank, n in ((1, 37), (2, 35), (3, 16)):
        wick = D.lambda_by_wick(rank)
        assert L.LAMBDA[rank] == wick and len(wick) == n
        assert D.lambda_by_differentiation(rank) == wick
    assert L.DENSITY == D.densities()
    assert sum(len(v) for v in L.DENSITY.values()) == 38       # one for every term of the Lagrangian
    assert L.GRADIENT == D.one_index(excitation=False) and len(L.GRADIENT) == 81
    blocks = L.gradient_blocks()
    assert sum(len(v) for v in blocks.values()) == 81
    assert sorted(blocks) == ['f.oo', 'f.vo', 'f.vv', 'oo.oo', 'oo.vo', 'vo.oo', 'vo.vo', 'vo.vv',
                              'vv.oo', 'vv.vo', 'vv.vv']
    # the Lagrangian itself: the contractions of Wick's theorem are the table
    wick = D.collect([(c, f, '') for c, f in D.lagrangian(wick=True)], antisym=False)['']
    assert wick == D.collect([(c, f, '') for c, f in D.lagrangian()], antisym=False)['']
    assert len(wick) == 38


def test_canonical_form():
    '''The canonical form does not depend on how a term is written: summed
    letters renamed, indices permuted within the antisymmetric groups (with
    the sign), external letters permuted under the antisymmetrizer; and a
    term that its symmetry makes vanish is recognized.'''
    a = D.canonical(1, (('l3', 'ijmabe'), ('v', 'ekcm')), 'ijkabc', True)
    b = D.canonical(1, (('v', 'fjno'[0] + 'k' + 'cn'), ('l3', 'jinbaf')), 'ijkabc', True)
    assert a == b                                              # two transpositions: the same sign
    c = D.canonical(1, (('l3', 'jimabe'), ('v', 'ekcm')), 'ijkabc', True)
    assert c == (-a[0],) + a[1:]
    d = D.canonical(1, (('l3', 'kjmabe'), ('v', 'eicm')), 'ijkabc', True)
    assert d == (-a[0],) + a[1:]                               # A takes the transposition (i k) back with a sign
    assert D.canonical(1, (('l2', 'mnef'), ('f', 'ia'), ('t2', 'mnef')), 'ia', False)[0] == 1
    # two factors of one kind: either may be the first
    e = D.canonical(1, (('l3', 'mnoefg'), ('v', 'ajbh'), ('t2', 'ijef'), ('t3', 'mnobgh')), 'ai', False)
    forms = [D.canonical(1, (('l2', 'mnef'), ('v', 'ijab'), tuple(x.split('_')), tuple(y.split('_'))), '', False)
             for x, y in (('t2_imae', 't2_jnbf'), ('t2_jnbf', 't2_imae'), ('t2_nibf', 't2_mjea'),
                          ('t2_nibf', 't2_jmea'))]
    assert forms[0] == forms[1] == forms[2] and forms[0][0] == 1
    assert forms[3] == (-1,) + forms[0][1:]
    assert e[1] in [f for _, f in L.GRADIENT]
    # a pair of factors symmetric in two letters, contracted with an
    # antisymmetric pair: the term vanishes
    assert D.canonical(1, (('l2', 'mnab'), ('v', 'mcij'), ('v', 'ncij')), 'abc', False) is None


def _random(no, nv, seed, hermitian=True):
    rng = np.random.default_rng(seed)
    n = no + nv
    v = F.hermitian_integrals(rng, n)
    f = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    f = f + f.conj().T
    if not hermitian:
        v = v + 0.3j * F.hermitian_integrals(rng, n)
        f = f + 0.3 * (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    amps = [F.complex_antisym(rng, no, nv, k, s) for k, s in ((1, 0.3), (2, 0.2), (3, 0.1))]
    lams = [F.complex_antisym(rng, no, nv, k, s) for k, s in ((1, 0.3), (2, 0.2), (3, 0.1))]
    return f, v, amps, lams


@pytest.mark.parametrize('no,nv,hermitian', [(4, 5, True), (5, 4, False), (5, 5, True)])
def test_tables_equal_operators(no, nv, hermitian):
    '''For complex t1, t2, t3, l1, l2, l3 that solve nothing and complex f
    and <pq||rs> (Hermitian or not: nothing in the tables is conjugated):

        LAMBDA[k]  = <0| (1 + Lambda) [Hbar, tau_k] |0>,
        densities  = <0| (1 + Lambda) exp(-T) p+ q exp(T) |0>, .. p+ q+ s r ..,
        Lagrangian = sum h d1 + 1/4 sum <pq||rs> d2,
        GRADIENT   = <0| (1 + Lambda) exp(-T2-T3) [a+_i a_a, H~] exp(T2+T3) |0>,

    and the same commutator with the bare H from the densities.'''
    f, v, amps, lams = _random(no, nv, 3, hermitian)
    fk = F.Fock(no, nv)
    ex = F.Left(fk, f, v, amps)
    lam = ex.vector(lams)
    want = ex.tensors(ex.lambda_residual(lam))
    got = LR.residuals(lams, amps, f, v, no)
    for k in range(3):
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
    t23 = F.Left(fk, f, v, amps[1:])
    bra = t23.exp(ex.bra(lam), -1, left=True)
    ket = t23.right
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
    twice in another form), and none vanishes: the terms as vectors, for a
    few sets of random amplitudes and integrals, have full rank.'''
    no = nv = 5
    cols = {}
    for seed in range(3):
        f, v, amps, lams = _random(no, nv, 20 + seed)
        arrays = LR._arrays(f, v, no, amps, lams)
        for rank in (1, 2, 3):
            ext = Q.EXT[rank]
            for n, (_, fac) in enumerate(L.LAMBDA[rank]):
                x = L.product(fac, arrays, no, ext)
                x = x if rank == 1 else Q.asym_unique(x, rank)
                cols.setdefault(('lambda', rank, n), []).append(x.ravel())
        for n, (_, fac) in enumerate(L.GRADIENT):
            cols.setdefault(('gradient', 0, n), []).append(L.product(fac, arrays, no, 'ai').ravel())
    for name, rank, count in (('lambda', 1, 37), ('lambda', 2, 35), ('lambda', 3, 16), ('gradient', 0, 81)):
        A = np.array([np.concatenate(cols[name, rank, n]) for n in range(count)]).T
        s = np.linalg.svd(A / np.linalg.norm(A, axis=0), compute_uv=False)
        assert s[-1] > 1e-6 * s[0], (name, rank, s[-1] / s[0])
