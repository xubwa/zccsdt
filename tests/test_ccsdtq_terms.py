'''The term table of the spin-orbital CCSDTQ equations and of the
perturbative corrections (zccsdt.cc.ccsdtq_terms) against second-quantized
operators in a small Fock space (tests/fockspace.py).'''

import numpy as np
import pytest

import fockspace as F
from zccsdt.cc import ccsdt, ccsdtq_terms as Q


def one_body(fk, f):
    '''sum f[p,q] p+ q (the normal-ordered operator up to a constant).'''
    op = None
    for p in range(fk.n):
        for q in range(fk.n):
            x = f[p, q] * (fk.ad[p] @ fk.a[q])
            op = x if op is None else op + x
    return op


def commutator(H, ref, *Ts):
    '''[..[[H, T_1], T_2].., T_m] |ref> for commuting T's (m = 0: H |ref>).'''
    out = 0
    for mask in range(1 << len(Ts)):            # the T's to the left of H
        x = ref
        for bit, T in enumerate(Ts):
            if not mask >> bit & 1:
                x = T @ x
        x = H @ x
        nleft = 0
        for bit, T in enumerate(Ts):
            if mask >> bit & 1:
                x = T @ x
                nleft += 1
        out = out + (-1) ** nleft * x
    return out


def project(fk, vec, k):
    '''<Phi_I^A| vec> for sorted I and A: [C(no,k), C(nv,k)].'''
    occ, vir = Q.tuples(fk.no, k), Q.tuples(fk.nv, k)
    out = np.zeros((len(occ), len(vir)), dtype=np.result_type(vec, np.float64))
    for x, o in enumerate(occ):
        base = fk.ref
        for i in o:
            base = fk.a[i] @ base
        for y, v in enumerate(vir):
            det = base
            for a in reversed(v):
                det = fk.ad[fk.no + a] @ det
            out[x, y] = det @ vec               # (the determinants are real)
    return out


@pytest.fixture(scope='module')
def space():
    '''6 + 6 spin orbitals: where every group of terms has full rank.'''
    no, nv = 6, 6
    rng = np.random.default_rng(7)
    fk = F.Fock(no, nv)
    arrays = {'v': F.random_integrals(rng, no + nv)}
    f = rng.standard_normal((no + nv,) * 2)
    arrays['f'] = f + f.T
    ops = {'v': fk.two_body(arrays['v']), 'f': one_body(fk, arrays['f'])}
    for k in (2, 3, 4):
        arrays['t%d' % k] = F.random_antisym(rng, no, nv, k)
        ops['t%d' % k] = fk.excitation(arrays['t%d' % k])
    return fk, arrays, ops


def test_groups_equal_commutators(space):
    '''Every group of terms is the connected commutator it is named after,
    and its representatives are linearly independent (no topology is
    written twice, none is missing).'''
    fk, arrays, ops = space
    assert sum(len(g[4]) for name, g in Q.GROUPS.items() if name[:2] != 'R5') == 5 + 14 + 22 + 34
    for name, (k, h, kinds, sym, terms) in Q.GROUPS.items():
        want = project(fk, commutator(ops[h], fk.ref, *[ops[t] for t in kinds]) / sym, k)
        cols = [Q.asym_unique(Q.product(fac, arrays, fk.no, k), k) for _, fac in terms]
        got = sum(float(c) * x for (c, _), x in zip(terms, cols))
        assert abs(got - want).max() < 1e-9 * abs(want).max(), name
        A = np.array([x.ravel() for x in cols]).T
        assert np.linalg.matrix_rank(A, tol=1e-8 * abs(A).max()) == len(terms), name


def test_left_hand_quantities(space):
    '''<rank k| V T_n |0> for n = k - 2 (disconnected) and k - 1.'''
    fk, arrays, ops = space
    for name, (k, src, terms) in Q.LEFT.items():
        want = project(fk, ops['v'] @ (ops[src] @ fk.ref), k)
        got = Q.asym_unique(Q.total(terms, arrays, fk.no, k), k)
        assert abs(got - want).max() < 1e-9 * abs(want).max(), name
    # the connected ones are the same contractions as in the residuals
    assert Q.LEFT['Z4.T3'][2] == Q.GROUPS['R4: [V,T3]'][4]
    assert Q.LEFT['Z5.T4'][2] == Q.GROUPS['R5: [V,T4]'][4]


def test_table_equals_wick_derivation():
    '''Wick's theorem (zccsdt.cc.wick) gives the terms of the table: for
    every group all contractions of the quasi-particle annihilators of H
    with the cluster operators, collected by topology - the same
    topologies as the table has, and the coefficients of the table.  (The
    sign between two contractions of one topology is that of a relabeling;
    it is taken from one antisymmetrized element.)'''
    from zccsdt.cc import wick
    no, nv = 7, 7
    rng = np.random.default_rng(5)
    arrays = {'v': F.random_integrals(rng, no + nv)}
    f = rng.standard_normal((no + nv,) * 2)
    arrays['f'] = f + f.T
    for k in (2, 3, 4):
        arrays['t%d' % k] = F.random_antisym(rng, no, nv, k)
    ncontr = 0
    for group, (rank, kind, kinds, sym, table) in Q.GROUPS.items():
        tops = wick.by_topology(wick.derive(kind, [int(k[1]) for k in kinds], rank, sym))
        tab = {wick.topology(fac): (c, fac) for c, fac in table}
        assert len(tab) == len(table) and set(tab) == set(tops), group
        occ = tuple(sorted(rng.choice(no, rank, replace=False)))
        vir = tuple(sorted(rng.choice(nv, rank, replace=False)))
        for topo, (c, fac) in tab.items():
            unit = Q.element([(1, fac)], arrays, no, rank, occ, vir)
            got = sum(float(cw) * Q.element([(1, fw)], arrays, no, rank, occ, vir)
                      for cw, fw in tops[topo])
            assert abs(got - float(c) * unit) < 1e-9 * abs(unit), (group, topo)
            # all contractions of a topology have the same weight
            assert len({abs(cw) for cw, _ in tops[topo]}) == 1, (group, topo)
            assert abs(c) == abs(tops[topo][0][0]) * len(tops[topo]), (group, topo)
            ncontr += len(tops[topo])
    assert ncontr == 4061


def _hermitian_integrals(rng, n):
    x = rng.standard_normal((n,) * 4) + 1j * rng.standard_normal((n,) * 4)
    x = x - x.transpose(1, 0, 2, 3)
    x = x - x.transpose(0, 1, 3, 2)
    return x + x.transpose(2, 3, 0, 1).conj()


def _complex_antisym(rng, no, nv, k, scale):
    return scale * (F.random_antisym(rng, no, nv, k) + 1j * F.random_antisym(rng, no, nv, k))


@pytest.mark.parametrize('no,nv,with_t4', [(5, 5, True), (4, 6, True), (5, 4, False)])
def test_residuals_equal_similarity_transformed_hamiltonian(no, nv, with_t4):
    '''R_k = <Phi_k| exp(-T) H exp(T) |0> for T = T1 + T2 + T3 + T4 with
    complex amplitudes and complex Hermitian F and V: the T1 dressing of
    zccsdt.cc.ccsdt and all terms of the table together (without T4: the
    CCSDT equations).'''
    n = no + nv
    rng = np.random.default_rng(11)
    fk = F.Fock(no, nv)
    v = _hermitian_integrals(rng, n)
    f = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    f = f + f.conj().T
    t1 = 0.3 * (rng.standard_normal((no, nv)) + 1j * rng.standard_normal((no, nv)))
    amps = {'t2': _complex_antisym(rng, no, nv, 2, 0.2), 't3': _complex_antisym(rng, no, nv, 3, 0.1)}
    if with_t4:
        amps['t4'] = _complex_antisym(rng, no, nv, 4, 0.05)
    T = fk.excitation(t1)
    for a in amps.values():
        T = T + fk.excitation(a)
    H = one_body(fk, f) + fk.two_body(v)

    def exp(sign, x):
        out, term = x.astype(complex), x.astype(complex)
        for k in range(1, n + 1):
            term = sign * (T @ term) / k
            out = out + term
        return out
    hbar = exp(-1, H @ exp(1, fk.ref))
    ge, fdr = ccsdt._t1_dress(v, f, t1, no, n)
    arrays = dict(amps, v=ge, f=fdr)
    for k in range(1, 5 if with_t4 else 4):
        want = project(fk, hbar, k)
        got = Q.residual(k, arrays, no, with_t4=with_t4)
        got = got if k == 1 else Q.unique(got, k)
        assert abs(got - want).max() < 1e-10 * abs(want).max(), k
    if not with_t4:
        return
    # the helpers: the antisymmetrizer at the unique elements, and back
    x = Q.total(Q.residual_terms(4), arrays, no, 4)
    full = Q.asym(x, 4)
    assert abs(Q.asym_unique(x, 4) - Q.unique(full, 4)).max() < 1e-12 * abs(full).max()
    assert abs(Q.expand(Q.unique(full, 4), no, nv, 4) - full).max() < 1e-12 * abs(full).max()
