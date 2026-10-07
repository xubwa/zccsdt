'''The kernels on packed antisymmetric tuples (csrc/tnpack.c through
zccsdt.cc.t4pack) against the index tables of the numpy versions.'''
from math import comb

import numpy as np
import pytest

from zccsdt.cc import t4pack as P

pytestmark = pytest.mark.skipif(not P.USE_C, reason='libzc_tnpack is not built')


@pytest.fixture
def both(monkeypatch):
    '''run(f): f() with the kernels and with the tables.'''
    def run(f):
        out = []
        for use in (True, False):
            monkeypatch.setattr(P, 'USE_C', use)
            out.append(f())
        return out
    return run


def _rnd(rng, dtype, *shape):
    x = rng.standard_normal(shape)
    return (x + 1j * rng.standard_normal(shape)).astype(dtype) if dtype == np.complex128 else x


SPLITS = [(7, 2, 2), (7, 1, 3), (8, 3, 2), (8, 1, 4), (8, 2, 3), (6, 1, 2), (6, 1, 1), (9, 3, 3), (7, 3, 1),
          (7, 2, 1), (5, 4, 1)]


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('n,k1,k2', SPLITS)
def test_reduce_and_expand(both, dtype, n, k1, k2):
    '''The antisymmetrizing reduction of X[g1, g2] and the expansion of
    packed tuples into t[g1, g2]: whole and on a block of X, with an outer
    axis in every place.'''
    rng = np.random.default_rng(n * 100 + k1 * 10 + k2)
    n1, n2, nA = comb(n, k1), comb(n, k2), comb(n, k1 + k2)
    for rows, cols in ((None, None), ((1, n1 - 1), (2, n2)), ((0, max(1, n1 // 2)), (1, max(2, n2 // 3)))):
        b0, b1 = rows or (0, n1)
        c0, c1 = cols or (0, n2)
        X, T = _rnd(rng, dtype, b1 - b0, c1 - c0), _rnd(rng, dtype, nA)

        def flat():
            R = P.tn_reduce(np.zeros(nA, dtype=dtype), X, n, k1, k2, rows, cols, scale=0.7)
            Y = P.tn_expand(np.full(X.shape, 9.0, dtype=dtype), T, n, k1, k2, rows, cols, scale=-1.3)
            return R, Y
        (Rc, Yc), (Rt, Yt) = both(flat)
        assert abs(Rc - Rt).max() < 1e-13 and abs(Yc - Yt).max() < 1e-13
        for outer in (0, 1, 2):
            shape = [b1 - b0, c1 - c0]
            shape.insert(outer, 3)
            X3, T3 = _rnd(rng, dtype, *shape), _rnd(rng, dtype, 3, nA)

            def with_outer():
                R = P.tn_reduce(np.zeros((3, nA), dtype=dtype), X3, n, k1, k2, rows, cols, outer=outer)
                Y = P.tn_expand(np.full(shape, 9.0, dtype=dtype), T3, n, k1, k2, rows, cols, outer=outer)
                return R, Y
            (Rc, Yc), (Rt, Yt) = both(with_outer)
            assert abs(Rc - Rt).max() < 1e-13 and abs(Yc - Yt).max() < 1e-13, outer


def test_reduce_is_the_antisymmetrizer():
    '''tn_reduce of X[(ab),(cd)] is the antisymmetrizer over a, b, c, d of
    the array that has X at sorted pairs - the tables themselves against
    the dense antisymmetrizer.'''
    from zccsdt.cc import ccsdtq_terms as Q
    n = 6
    rng = np.random.default_rng(5)
    pairs = P.tuples(n, 2)
    X = rng.standard_normal((len(pairs), len(pairs)))
    full = np.zeros((1, 1, 1, 1, n, n, n, n))
    full[0, 0, 0, 0, pairs[:, 0][:, None], pairs[:, 1][:, None], pairs[:, 0], pairs[:, 1]] = X
    want = 0
    from itertools import permutations
    quads = P.tuples(n, 4)
    for p in permutations(range(4)):
        want = want + Q._sign(p) * full[0, 0, 0, 0][tuple(quads[:, p[s]] for s in range(4))]
    got = P.tn_reduce(np.zeros(len(quads)), X, n, 2, 2)
    assert abs(got - want).max() < 1e-13


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_rows_sums_energy(both, dtype):
    rng = np.random.default_rng(1)
    M = _rnd(rng, dtype, 11, 40)
    rows, sgn = np.array([3, 0, 10, 3, 7]), np.array([1, -1, 0, -1, 1])
    for o in both(lambda: P.tn_rows(np.empty((5, 25), dtype=dtype), M, rows, sgn, (5, 30))):
        assert abs(o - M[rows, 5:30] * sgn[:, None]).max() == 0
    e = rng.standard_normal(9)
    for k in (1, 2, 3, 4, 5):
        a, b = both(lambda: P.tuple_sums(e, k))
        assert abs(a - b).max() < 1e-13 and len(a) == comb(9, k)
    z, r1, r2 = (_rnd(rng, dtype, 1000) for _ in range(3))
    ev = rng.uniform(1, 2, 1000)
    a, b = both(lambda: P.energy2(z, r1, r2, ev, -3.0))
    assert abs(a[0] - b[0]) < 1e-11 and abs(a[1] - b[1]) < 1e-11
    x = _rnd(rng, dtype, 7, 9)
    assert not P.zero(x).any()


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_two_sided(dtype):
    '''tn_expand2 / tn_reduce2 with the entries of the occupied splits
    (occ_expand, occ_reduce) against the numpy view and reduce: in the
    natural layout [g1o, g2o, g1v, g2v] and in the layout of a matrix
    product [g1o, g1v, g2o, g2v], whole and in batches; the occupied side
    alone (tn_axpy_rows) and the virtual side alone.'''
    rng = np.random.default_rng(0)
    no, nv = 6, 7
    T = _rnd(rng, dtype, comb(no, 4), comb(nv, 4))
    for ko, kv in (((1, 3), (1, 3)), ((2, 2), (2, 2)), ((1, 3), (2, 2)), ((2, 2), (1, 3)), ((1, 3), (3, 1))):
        so, sv = P.split(no, *ko), P.split(nv, *kv)
        for jo, jv in ((None, None), ((1, min(4, so.n2)), (2, min(9, sv.n2)))):
            b0, b1 = jo or (0, so.n2)
            c0, c1 = jv or (0, sv.n2)
            want = P.view(T, so, sv, jo, jv)
            X = np.empty(want.shape, dtype=dtype)
            st = [x // X.itemsize for x in X.strides]
            P.tn_expand2(X, T, nv, kv[0], kv[1], P.occ_expand(so, st[0], st[1], jo), st[2], st[3], cols=(c0, c1))
            assert abs(X - want).max() < 1e-13, (ko, kv, jo)
            Y = np.empty((so.n1, sv.n1, b1 - b0, c1 - c0), dtype=dtype)
            st = [x // Y.itemsize for x in Y.strides]
            P.tn_expand2(Y, T, nv, kv[0], kv[1], P.occ_expand(so, st[0], st[2], jo), st[1], st[3], cols=(c0, c1))
            assert abs(Y - want.transpose(0, 2, 1, 3)).max() < 1e-13
            Z = _rnd(rng, dtype, *want.shape)
            ref = P.reduce(np.zeros_like(T), Z, so, sv, jo, jv, scale=0.7)
            st = [x // Z.itemsize for x in Z.strides]
            got = P.tn_reduce2(np.zeros_like(T), Z, nv, kv[0], kv[1], P.occ_reduce(so, st[0], st[1], jo),
                               st[2], st[3], cols=(c0, c1), scale=0.7)
            assert abs(got - ref).max() < 1e-12, (ko, kv, jo)
            Zt = np.ascontiguousarray(Z.transpose(0, 2, 1, 3))
            st = [x // Zt.itemsize for x in Zt.strides]
            got = P.tn_reduce2(np.zeros_like(T), Zt, nv, kv[0], kv[1], P.occ_reduce(so, st[0], st[2], jo),
                               st[1], st[3], cols=(c0, c1), scale=0.7)
            assert abs(got - ref).max() < 1e-12
    so, sv = P.split(no, 2, 2), P.split(nv, 2, 2)
    for jv in (None, (3, 20)):
        c0, c1 = jv or (0, T.shape[1])
        Z = _rnd(rng, dtype, so.n1, so.n2, c1 - c0)
        ref = P.reduce(np.zeros_like(T), Z, so, None, jv=jv)
        row, off, sg = P.occ_reduce(so, so.n2 * (c1 - c0), c1 - c0)
        assert abs(P.tn_axpy_rows(np.zeros_like(T), Z, row, off, sg, col0=c0) - ref).max() < 1e-12
    # half of the splits of the occupied quadruples (the first pair has the smallest index)
    Z = _rnd(rng, dtype, so.n1, so.n2, T.shape[1])
    half = P.tn_axpy_rows(np.zeros_like(T), Z, *P.occ_reduce(so, so.n2 * T.shape[1], T.shape[1], only=(0, 1, 2)))
    rest = P.tn_axpy_rows(np.zeros_like(T), Z, *P.occ_reduce(so, so.n2 * T.shape[1], T.shape[1], only=(3, 4, 5)))
    assert abs(half + rest - P.reduce(np.zeros_like(T), Z, so, None)).max() < 1e-12
    for I in range(len(T)):
        assert (so.idx1[I, :3] < so.idx2[I, :3]).all() and (so.idx1[I, 3:] > so.idx2[I, 3:]).all()
    Z = _rnd(rng, dtype, 5, sv.n1, sv.n2)
    ref = P.reduce(np.zeros_like(T), Z, None, sv, jo=(3, 8))
    got = P.tn_reduce2(np.zeros_like(T), Z, nv, 2, 2, (np.arange(3, 8), np.arange(5) * sv.n1 * sv.n2, np.ones(5)),
                       sv.n2, 1)
    assert abs(got - ref).max() < 1e-12
