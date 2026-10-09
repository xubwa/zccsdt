import numpy as np
import pytest

from zccsdt.cc import t3pack, ccsdt

NO, NV = 5, 7


@pytest.fixture(params=['c', 'numpy'])
def impl(request, monkeypatch):
    if request.param == 'c' and not t3pack.USE_C:
        pytest.skip('C kernels not built')
    monkeypatch.setattr(t3pack, 'USE_C', request.param == 'c')
    return request.param


def _rand(shape, rng, cplx):
    x = rng.standard_normal(shape)
    return x + 1j * rng.standard_normal(shape) if cplx else x


def _asym_pair(x, ax):
    return x - np.swapaxes(x, ax, ax + 1)


def _full_t3(rng, cplx):
    t = ccsdt.fullasym(_rand((NO, NO, NO, NV, NV, NV), rng, cplx))
    return t if cplx else t.real.copy()


def _pack(t3):
    oi, vi = ccsdt._t3_tri(NO, NV)
    return t3[oi[:, 0][:, None], oi[:, 1][:, None], oi[:, 2][:, None],
              vi[:, 0], vi[:, 1], vi[:, 2]]


@pytest.mark.parametrize('cplx', [False, True])
def test_gathers(cplx, impl):
    rng = np.random.default_rng(1)
    t3 = _full_t3(rng, cplx)
    T3p = _pack(t3)
    om, vm = t3pack.triple_map(NO), t3pack.triple_map(NV)
    pairs = np.array([0, 3, 7, 9])
    T = t3pack.gather_half(T3p, om, vm, pairs)
    P, Q = om.pairs[pairs], vm.pairs
    ref = t3[:, P[:, 0], P[:, 1]][:, :, :, Q[:, 0], Q[:, 1]]   # (m, P, d, Q)
    np.testing.assert_allclose(T, ref.transpose(1, 2, 0, 3), atol=1e-13)
    rows = np.array([0, 4, 5])
    Tv = t3pack.gather_vir_half(T3p, vm, rows)
    tri = om.triples[rows]
    ref = t3[tri[:, 0], tri[:, 1], tri[:, 2]][:, Q[:, 0], Q[:, 1], :]
    np.testing.assert_allclose(Tv, ref.transpose(1, 0, 2), atol=1e-13)


@pytest.mark.parametrize('cplx', [False, True])
def test_reductions(cplx, impl):
    '''P(i/jk)P(a/bc) of X[i, a, jk, bc] (antisymmetric within the pairs).'''
    rng = np.random.default_rng(2)
    X = _asym_pair(_asym_pair(_rand((NO, NV, NO, NO, NV, NV), rng, cplx), 2), 4)
    full = X.transpose(0, 2, 3, 1, 4, 5)                         # [i,j,k,a,b,c]
    full = (full - full.transpose(1, 0, 2, 3, 4, 5) - full.transpose(2, 1, 0, 3, 4, 5))
    full = (full - full.transpose(0, 1, 2, 4, 3, 5) - full.transpose(0, 1, 2, 5, 4, 3))
    om, vm = t3pack.triple_map(NO), t3pack.triple_map(NV)
    Xh = X[:, :, om.pairs[:, 0], om.pairs[:, 1]][..., vm.pairs[:, 0], vm.pairs[:, 1]]
    R = np.zeros((om.ntri, vm.ntri), dtype=Xh.dtype)
    for batch in (np.arange(0, 4), np.arange(4, om.npair)):
        Xv = t3pack.reduce_vir(Xh[:, :, batch].transpose(2, 0, 1, 3), vm)
        t3pack.scatter_occ(R, Xv, om, batch)
    np.testing.assert_allclose(R, _pack(full), atol=1e-12)
    # split layout Z[P, a, i, Q]
    R2 = np.zeros_like(R)
    for batch in (np.arange(0, 3), np.arange(3, om.npair)):
        Z = np.ascontiguousarray(Xh[:, :, batch].transpose(2, 1, 0, 3))
        t3pack.scatter_occ(R2, t3pack.reduce_vir_split(Z, vm, NO), om, batch)
    np.testing.assert_allclose(R2, _pack(full), atol=1e-12)
    # fused reduce + scatter
    R3 = np.zeros_like(R)
    for batch in (np.arange(0, 3), np.arange(3, om.npair)):
        Z = np.ascontiguousarray(Xh[:, :, batch].transpose(2, 1, 0, 3))
        t3pack.add_reduce_split(R3, Z, om, vm, batch)
    np.testing.assert_allclose(R3, _pack(full), atol=1e-12)
    # pair-first variant: Y[I, Q, c] with P(c/ab)
    Y = _asym_pair(_rand((3, NV, NV, NV), rng, cplx), 1)[:, vm.pairs[:, 0], vm.pairs[:, 1]]
    Yf = np.zeros((3, NV, NV, NV), dtype=Y.dtype)
    Yf[:, vm.pairs[:, 0], vm.pairs[:, 1]] = Y
    Yf[:, vm.pairs[:, 1], vm.pairs[:, 0]] = -Y
    ref = Yf - Yf.transpose(0, 3, 2, 1) - Yf.transpose(0, 1, 3, 2)
    want = ref[:, vm.triples[:, 0], vm.triples[:, 1], vm.triples[:, 2]]
    np.testing.assert_allclose(t3pack.reduce_vir_pair_first(Y, vm), want, atol=1e-12)
    Racc = np.zeros((5, vm.ntri), dtype=Y.dtype)
    t3pack.add_reduce_pair_first(Racc, np.ascontiguousarray(Y.transpose(1, 0, 2)), vm,
                                 np.array([4, 0, 2]))
    np.testing.assert_allclose(Racc[[4, 0, 2]], want, atol=1e-12)
    assert not Racc[[1, 3]].any()


def test_occ_operator():
    rng = np.random.default_rng(3)
    F = rng.standard_normal((NO, NO))
    om = t3pack.triple_map(NO)

    def apply(t):   # -P(i/jk) sum_m F[m,i] t[m,j,k]
        x = np.einsum('mi,Jmjk->Jijk', F, t)
        return -(x - x.transpose(0, 2, 1, 3) - x.transpose(0, 3, 2, 1))
    O = t3pack.occ_operator(om, apply)
    t3 = _full_t3(rng, False)
    x = np.einsum('mi,mjkabc->ijkabc', F, t3)
    ref = -(x - x.transpose(1, 0, 2, 3, 4, 5) - x.transpose(2, 1, 0, 3, 4, 5))
    np.testing.assert_allclose(O @ _pack(t3), _pack(ref), atol=1e-12)


@pytest.mark.parametrize('cplx', [False, True])
@pytest.mark.parametrize('o, labels, ncls', [
    (5, None, None), (8, None, None), (12, None, None), (8, [0] * 4 + [1] * 4, None),
    (9, [0, 0, 1, 1, 1, 2, 2, 2, 2], 3), (12, [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], 4)])
def test_occ_operator_forms(o, labels, ncls, cplx):
    '''The operator of the hh ladder and F_oo: its matrix written down
    directly and the pair-driven form against the matrix obtained by
    applying the operator to the packed basis; the transposed operator; dense
    and in the blocks of a label.'''
    from zccsdt.cc import t3kern
    rng = np.random.default_rng(4)
    om = t3pack.TripleMap(o) if labels is None else t3pack.TripleMap(o, labels, ncls)
    W = _asym_pair(_asym_pair(_rand((o,) * 4, rng, cplx), 0), 2)
    F = _rand((o, o), rng, cplx)
    O = t3pack.occ_operator(om, lambda t: t3pack.occ_apply(t, W, F))
    np.testing.assert_allclose(t3pack.occ_operator_direct(om, W, F), O, atol=1e-12)
    blocks = [(0, om.ntri)] if labels is None else \
        [(int(om.tri_off[g]), int(om.tri_off[g + 1])) for g in range(om.ntg)]
    for lo, hi in blocks:
        np.testing.assert_allclose(t3pack.occ_operator_direct(om, W, F, (lo, hi)),
                                   O[lo:hi, lo:hi], atol=1e-12)
    if not t3kern.USE_C:
        return
    for trans in (False, True):
        ref = O.T if trans else O
        for mode in ('matrix', 'pair'):
            op = t3kern.OccOperator(om, W, F, W.dtype, trans=trans, mode=mode)
            assert op.mode == mode
            for lo, hi in blocks:
                if hi == lo:
                    continue
                T, R0 = _rand((hi - lo, 203), rng, cplx), _rand((hi - lo, 203), rng, cplx)
                R = op.apply(T, R0.copy(), (lo, hi))
                np.testing.assert_allclose(R, R0 + ref[lo:hi, lo:hi] @ T, atol=1e-12, rtol=1e-12)
                assert op.nbytes((lo, hi)) > 0


@pytest.mark.parametrize('cplx', [False, True])
def test_pair_antisym_add(cplx, impl):
    rng = np.random.default_rng(5)
    n, nx, ny = 6, 4, 3
    X = _rand((n, nx, n, ny), rng, cplx)                 # layout [a, x, b, y]
    sa, sx, sb, sy = (s // X.itemsize for s in X.strides)
    pa, pb = np.triu_indices(n, 1)
    want = 0.5 + 2.0 * (X[pa, :, pb, :] - X[pb, :, pa, :])
    out = np.full((len(pa), nx, ny), 0.5, dtype=X.dtype)
    t3pack.pair_antisym_add(out, X, t3pack.triple_map(n), (sa, sb, sx, sy), alpha=2.0)
    np.testing.assert_allclose(out, want, atol=1e-13)


@pytest.mark.parametrize('cplx', [False, True])
def test_pair_batch_and_t1_contract(cplx, impl):
    rng = np.random.default_rng(6)
    n, ncol, no = 7, 5, 3
    vm = t3pack.triple_map(n)
    pa, pb = vm.pairs[:, 0], vm.pairs[:, 1]
    Yfull = _rand((n, n, ncol), rng, cplx)
    want = -(Yfull[pa, pb] - Yfull[pb, pa])
    out = np.zeros((vm.npair, ncol), dtype=Yfull.dtype)
    for x0 in (0, 3, 6):
        x1 = min(x0 + 3, n)
        t3pack.pair_antisym_batch_add(out, Yfull[x0:x1], vm, x0, alpha=-1.0)
    np.testing.assert_allclose(out, want, atol=1e-13)
    V = _rand((4, vm.npair), rng, cplx)
    t1 = _rand((no, n), rng, cplx)
    full = np.zeros((4, n, n), dtype=V.dtype)
    full[:, pa, pb] = V
    full[:, pb, pa] = -V
    out = np.ones((4, n, no), dtype=V.dtype)
    t3pack.pair_t1_contract(out, V, t1, vm)
    np.testing.assert_allclose(out, 1 + np.einsum('qef,if->qei', full, t1), atol=1e-13)


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_axpby_and_add_transposed(dtype):
    rng = np.random.default_rng(7)

    def rand(*shape):
        x = rng.standard_normal(shape)
        return (x + 1j * rng.standard_normal(shape)).astype(dtype) if dtype == np.complex128 \
            else x.astype(dtype)

    A, B = rand(7, 6, 5, 4), rand(6, 4, 7, 5)
    out, ref = A.copy(), A.copy()
    t3pack.axpby(out[:, 1:5, ::-1], B[1:5].transpose(2, 0, 3, 1), 0.5, 2.0)
    ref[:, 1:5, ::-1] = 2.0 * ref[:, 1:5, ::-1] + 0.5 * B[1:5].transpose(2, 0, 3, 1)
    assert abs(out - ref).max() < 1e-14
    o2 = rand(7, 4)
    t3pack.axpby(o2, A[:, 2, 3, :], -1.0, 0.0)          # beta = 0 overwrites
    assert abs(o2 + A[:, 2, 3, :]).max() == 0
    X, Y = rand(70, 45), rand(45, 70)
    ref = X + 0.25 * Y.T
    t3pack.add_transposed(X, Y, 0.25)
    assert abs(X - ref).max() < 1e-14


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_flip_sym3(dtype, monkeypatch):
    '''X + flip(X) with orbital / pair flip maps: C kernel (strided view) vs numpy.'''
    rng = np.random.default_rng(11)
    om = t3pack.triple_map(6, np.array([0] * 3 + [1] * 3))
    vm = t3pack.triple_map(8, np.array([0] * 4 + [1] * 4))
    of, Pf, sP = t3pack.flip_pairs(om)
    vf, Qf, sQ = t3pack.flip_pairs(vm)
    assert np.all(of[of] == np.arange(6)) and np.all(Qf[Qf] == np.arange(vm.npair))
    X = rng.standard_normal((om.npair, 8, 6)).astype(dtype)
    if dtype == np.complex128:
        X = X + 1j * rng.standard_normal(X.shape)
    args = ((vf, Pf, of), (np.ones(8), sP, np.ones(6)), np.arange(4))
    a = X.copy()
    t3pack.flip_sym3(a.transpose(1, 0, 2), *args)
    monkeypatch.setattr(t3pack, 'USE_C', False)
    b = X.copy()
    t3pack.flip_sym3(b.transpose(1, 0, 2), *args)
    assert abs(a - b).max() < 1e-14
    # the result is flip-invariant
    c = b.transpose(1, 0, 2)
    assert abs(c - sP[None, :, None] * c[np.ix_(vf, Pf, of)]).max() < 1e-14
