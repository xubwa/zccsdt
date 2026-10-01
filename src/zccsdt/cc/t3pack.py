'''Index maps, gathers and antisymmetrizing reductions for packed T3.

Packed layout: T3p[I, A], I over occupied triples i<j<k, A over virtual
triples a<b<c, both enumerated lexicographically (the order of
``ccsdt._t3_tri``). Pairs p<q are enumerated in ``numpy.triu_indices`` order.

Half expansion: every triple is written as (free index, pair) in three ways,
    (p<q<s) = (p; qs) with sign +, (q; ps) with sign -, (s; pq) with sign +.
``TripleMap.tri[x, P]`` and ``TripleMap.sgn[x, P]`` give the triple and sign
for a free index x and a pair P (sign 0 if x is in P), so that
    t[x, P...] = sgn[x, P] * t[tri[x, P]].
The antisymmetrizer P(x/P) applied to X[x, P] (X antisymmetric within P) is
evaluated on packed triples as
    R[I] = sum_r sgn_r X[free_r(I), pair_r(I)].
'''

import ctypes
import os

import numpy as np

_CACHE = {}


def _load_lib():
    path = os.environ.get('ZCCSDT_T3PACK_LIB') or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'lib', 'libzc_t3pack.so')
    try:
        return ctypes.CDLL(path)
    except OSError:
        return None


_lib = _load_lib()
USE_C = _lib is not None       # set False to force the numpy implementations


def _p(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def _sfx(dtype):
    if dtype == np.float64:
        return 'd'
    if dtype == np.complex128:
        return 'z'
    raise TypeError(dtype)


_I64 = ctypes.c_int64


class TripleMap:
    '''Pairs p<q and triples p<q<s of n indices with their index maps.

    Without ``spins`` the enumeration is lexicographic.  With ``spins`` (one
    0/1 = alpha/beta label per index, alpha indices first) pairs and triples
    are grouped by their number of beta indices (stable within a group), so
    Ms-conserving blocks are contiguous ranges: pair_off[g] .. pair_off[g+1]
    and tri_off[g] .. tri_off[g+1] hold the members with g beta indices.
    All other maps only depend on this enumeration.'''

    def __init__(self, n, spins=None):
        self.n = n
        pi, pj = np.triu_indices(n, 1)
        tri = np.array([(i, j, k) for i in range(n) for j in range(i + 1, n)
                        for k in range(j + 1, n)], dtype=np.int64).reshape(-1, 3)
        if spins is None:
            self.spins = None
        else:
            spins = np.asarray(spins, dtype=np.int64)
            assert spins.shape == (n,) and np.all(np.diff(spins) >= 0), \
                'spins must be 0/1 with alpha indices first'
            self.spins = spins
            order = np.argsort(spins[pi] + spins[pj], kind='stable')
            pi, pj = pi[order], pj[order]
            if len(tri):
                tri = tri[np.argsort(spins[tri].sum(axis=1), kind='stable')]
        self.npair = len(pi)
        self.pairs = np.stack((pi, pj), axis=1)
        pidx = -np.ones((n, n), dtype=np.int64)
        pidx[pi, pj] = np.arange(self.npair)
        self.pidx = pidx
        self.triples = tri
        self.ntri = len(tri)
        tidx = -np.ones((n, n, n), dtype=np.int64)
        if self.ntri:
            tidx[tri[:, 0], tri[:, 1], tri[:, 2]] = np.arange(self.ntri)
        if spins is not None:
            self.pair_grp = spins[pi] + spins[pj]
            self.tri_grp = spins[tri].sum(axis=1) if self.ntri else np.zeros(0, np.int64)
            self.pair_off = np.searchsorted(self.pair_grp, np.arange(4))
            self.tri_off = np.searchsorted(self.tri_grp, np.arange(5))

        # (free, pair) -> triple, sign
        self.tri = np.zeros((n, self.npair), dtype=np.int64)
        self.sgn = np.zeros((n, self.npair), dtype=np.int8)
        for P, (j, k) in enumerate(self.pairs):
            for x in range(n):
                if x == j or x == k:
                    continue
                if x < j:
                    self.tri[x, P], self.sgn[x, P] = tidx[x, j, k], 1
                elif x < k:
                    self.tri[x, P], self.sgn[x, P] = tidx[j, x, k], -1
                else:
                    self.tri[x, P], self.sgn[x, P] = tidx[j, k, x], 1

        # triple -> (free_r, pair_r, sign_r), r = 0, 1, 2
        if self.ntri:
            p, q, s = tri[:, 0], tri[:, 1], tri[:, 2]
            self.free = np.stack((p, q, s))
            self.pair = np.stack((pidx[q, s], pidx[p, s], pidx[p, q]))
        else:
            self.free = np.zeros((3, 0), dtype=np.int64)
            self.pair = np.zeros((3, 0), dtype=np.int64)
        self.rsgn = (1, -1, 1)
        # contiguous copies for the C kernels
        self.tri_c = np.ascontiguousarray(self.tri)
        self.sgn_c = np.ascontiguousarray(self.sgn)
        self.free_c = np.ascontiguousarray(self.free)
        self.pair_c = np.ascontiguousarray(self.pair)
        self.pa_c = np.ascontiguousarray(self.pairs[:, 0])
        self.pb_c = np.ascontiguousarray(self.pairs[:, 1])
        self.pidx_c = np.ascontiguousarray(self.pidx)
        # pair blocks (a whole-range block, or the alpha-alpha / alpha-beta /
        # beta-beta blocks) and, per block, pidx in block-local numbering
        # (-1 for pairs outside the block)
        if spins is None:
            self.blocks = [(0, self.npair)]
        else:
            self.blocks = [(int(self.pair_off[h]), int(self.pair_off[h + 1])) for h in range(3)]
        self.pidx_blk = []
        for lo, hi in self.blocks:
            loc = np.where((pidx >= lo) & (pidx < hi), pidx - lo, -1)
            self.pidx_blk.append(np.ascontiguousarray(loc, dtype=np.int64))


def triple_map(n, spins=None):
    key = (n, None if spins is None else tuple(int(x) for x in spins))
    if key not in _CACHE:
        _CACHE[key] = TripleMap(n, spins)
    return _CACHE[key]


def maps_for(eris):
    '''(om, vm) for an integral object: spin-grouped enumerations if the
    orbitals carry spin labels (eris.spin_occ / eris.spin_vir), otherwise
    lexicographic.'''
    so = getattr(eris, 'spin_occ', None)
    sv = getattr(eris, 'spin_vir', None)
    nocc = eris.nocc
    nvir = np.asarray(eris.fock).shape[0] - nocc
    return triple_map(nocc, so), triple_map(nvir, sv)


def gather_half(T3p, om, vm, pairs):
    '''Half-expanded T[P, d, m, Q] = t3[m, P_0, P_1, d, Q_0, Q_1] for the
    occupied pairs ``pairs`` (array of pair indices), all d, m and Q.
    Shape (len(pairs), nvir, nocc, nvpair).  In this layout T is a matrix
    (P d) x (m Q), so contractions over (P, d) or over (m, Q) are single
    GEMMs without any transpose of T.'''
    nb = len(pairs)
    rows = np.ascontiguousarray(om.tri[:, pairs].T)        # (nb, o)
    sgo = np.ascontiguousarray(om.sgn[:, pairs].T)
    if USE_C:
        T3p = np.ascontiguousarray(T3p)
        out = np.empty((nb, vm.n, om.n, vm.npair), dtype=T3p.dtype)
        getattr(_lib, 'zc_gather_half_' + _sfx(T3p.dtype))(
            _p(T3p), _I64(T3p.shape[1]), _p(rows), _p(sgo), _p(vm.tri_c),
            _p(vm.sgn_c), _I64(nb), _I64(om.n), _I64(vm.n), _I64(vm.npair),
            _p(out))
        return out
    X = T3p[rows.ravel()].reshape(nb, om.n, -1)  # (nb, o, nvtri), contiguous rows
    out = np.empty((nb, vm.n, om.n, vm.npair), dtype=T3p.dtype)
    for d in range(vm.n):
        np.take(X, vm.tri[d], axis=2, out=out[:, d])
        out[:, d] *= vm.sgn[d]
    out *= sgo[:, None, :, None]
    return out


def gather_vir_half(T3p, vm, rows):
    '''T[Q, I, c] = t3[rows[I], Q_0, Q_1, c] for occupied triples ``rows``.
    Shape (nvpair, len(rows), nvir), pair-major for the ladder GEMM.'''
    if USE_C:
        T3p = np.ascontiguousarray(T3p)
        rows = np.ascontiguousarray(rows, dtype=np.int64)
        out = np.empty((vm.npair, len(rows), vm.n), dtype=T3p.dtype)
        getattr(_lib, 'zc_gather_vir_half_' + _sfx(T3p.dtype))(
            _p(T3p), _I64(T3p.shape[1]), _p(rows), _I64(len(rows)),
            _p(vm.tri_c), _p(vm.sgn_c), _I64(vm.n), _I64(vm.npair), _p(out))
        return out
    X = np.take(T3p[rows], vm.tri.T, axis=1)     # (nb, nvpair, v)
    X *= vm.sgn.T
    return np.ascontiguousarray(X.transpose(1, 0, 2))


def reduce_vir(X, vm):
    '''P(c/ab)-type reduction over the virtual axes.  X[..., d, Q] with d the
    free index and Q the pair (antisymmetric) -> R[..., A].'''
    lead = X.shape[:-2]
    Xf = X.reshape(lead + (vm.n * vm.npair,))
    R = None
    for r in range(3):
        idx = vm.free[r] * vm.npair + vm.pair[r]
        part = np.take(Xf, idx, axis=-1)
        if vm.rsgn[r] < 0:
            part *= -1
        R = part if R is None else R + part
    return R


def reduce_vir_pair_first(X, vm):
    '''Same as reduce_vir for X[..., Q, d] (pair axis before free axis).'''
    lead = X.shape[:-2]
    Xf = X.reshape(lead + (vm.npair * vm.n,))
    R = None
    for r in range(3):
        idx = vm.pair[r] * vm.n + vm.free[r]
        part = np.take(Xf, idx, axis=-1)
        if vm.rsgn[r] < 0:
            part *= -1
        R = part if R is None else R + part
    return R


_SPLIT_CACHE = {}


def reduce_vir_split(Z, vm, n_mid):
    '''P(a/bc)-type reduction for Z[B, a, x, Q] (free virtual a, any middle
    index x of length n_mid, virtual pair Q) -> R[B, x, A].  Gathers straight
    from this layout through precomputed flat indices (no transpose).'''
    key = (id(vm), n_mid)
    if key not in _SPLIT_CACHE:
        x = np.arange(n_mid)[:, None]
        _SPLIT_CACHE[key] = [
            (vm.free[r][None, :] * (n_mid * vm.npair) + x * vm.npair
             + vm.pair[r][None, :]) for r in range(3)]
    idx = _SPLIT_CACHE[key]
    Zf = Z.reshape(Z.shape[0], -1)
    R = np.take(Zf, idx[0], axis=1)
    R -= np.take(Zf, idx[1], axis=1)
    R += np.take(Zf, idx[2], axis=1)
    return R


def scatter_occ(R, Xv, om, pairs, scale=1.0):
    '''R[tri(x, P)] += scale * P(x/P) Xv[P, x, :] for the pairs in ``pairs``.
    Xv has shape (len(pairs), nocc, ncol).  For a fixed position r of the free
    index in the target triple, the map (x, P) -> triple is injective, so each
    r is a collision-free fancy-index update.'''
    tri = om.tri[:, pairs].T                     # (nb, o)
    sgn = om.sgn[:, pairs].T
    x = np.arange(om.n)[None, :]
    P = om.pairs[pairs][:, None, :]
    # position of x in its sorted triple: 0 if x < P0, 1 if P0 < x < P1, 2 if x > P1
    pos = (x > P[..., 0]).astype(np.int8) + (x > P[..., 1])
    valid = sgn != 0
    for r in range(3):
        m = valid & (pos == r)
        if not m.any():
            continue
        ps, xs = np.nonzero(m)
        R[tri[ps, xs]] += (scale * sgn[ps, xs])[:, None] * Xv[ps, xs]


def occ_operator(om, apply):
    '''Dense matrix O[I, J] of a linear map acting on the occupied indices of
    antisymmetric 3-index tensors, built by applying ``apply`` (a function
    t[J, i, j, k] -> (L t)[J, i, j, k], batched over J) to the packed basis.'''
    n, nt = om.n, om.ntri
    B = np.zeros((nt, n, n, n))
    J = np.arange(nt)
    p, q, s = om.triples.T
    for perm, sign in (((p, q, s), 1), ((q, s, p), 1), ((s, p, q), 1),
                       ((q, p, s), -1), ((p, s, q), -1), ((s, q, p), -1)):
        B[(J,) + perm] = sign
    LB = apply(B)
    return LB[:, p, q, s].T                       # O[I, J]


def add_reduce_pair_first(R, X, vm, rows):
    '''R[rows] += P(c/ab) X, with X[Q, I, c] (pair-major, as produced by the
    ladder GEMM on gather_vir_half).'''
    if USE_C and R.flags.c_contiguous:
        X = np.ascontiguousarray(X, dtype=R.dtype)
        rows = np.ascontiguousarray(rows, dtype=np.int64)
        getattr(_lib, 'zc_reduce_pair_first_' + _sfx(R.dtype))(
            _p(R), _I64(R.shape[1]), _p(X), _p(rows), _I64(len(rows)),
            _I64(vm.n), _I64(vm.npair), _p(vm.free_c), _p(vm.pair_c))
        return
    R[rows] += reduce_vir_pair_first(X.transpose(1, 0, 2), vm)


def _batch_targets(om, pairs):
    '''Target triples touched by the pairs in a batch, each with up to three
    (local pair, free index, sign) entries; unused entries have pair -1.'''
    tri = om.tri[:, pairs].T                     # (nb, o)
    sgn = om.sgn[:, pairs].T
    ps, xs = np.nonzero(sgn)
    I = tri[ps, xs]
    order = np.argsort(I, kind='stable')
    ps, xs, I = ps[order], xs[order], I[order]
    s = sgn[ps, xs].astype(np.float64)
    tgt, start, counts = np.unique(I, return_index=True, return_counts=True)
    slot = np.arange(len(I)) - np.repeat(start, counts)
    eP = -np.ones((len(tgt), 3), dtype=np.int64)
    ex = np.zeros((len(tgt), 3), dtype=np.int64)
    es = np.zeros((len(tgt), 3))
    row = np.repeat(np.arange(len(tgt)), counts)
    eP[row, slot] = ps
    ex[row, slot] = xs
    es[row, slot] = s
    return tgt, eP, ex, es


def add_reduce_split(R, Z, om, vm, pairs, scale=1.0):
    '''R += scale * P(x/P) P(a/Q) Z[P, a, x, Q] for the pairs in ``pairs``
    (reduce_vir_split followed by scatter_occ, fused in C).'''
    if USE_C and R.flags.c_contiguous:
        Z = np.ascontiguousarray(Z, dtype=R.dtype)
        tgt, eP, ex, es = _batch_targets(om, pairs)
        es *= scale
        getattr(_lib, 'zc_reduce_scatter_' + _sfx(R.dtype))(
            _p(R), _I64(R.shape[1]), _p(Z), _I64(om.n), _I64(vm.n),
            _I64(vm.npair), _I64(len(tgt)), _p(tgt), _p(eP), _p(ex), _p(es),
            _p(vm.free_c), _p(vm.pair_c))
        return
    scatter_occ(R, reduce_vir_split(Z, vm, om.n), om, pairs, scale)


def pair_antisym_add(out, X, vm, strides, alpha=1.0):
    '''out[Q, x, y] += alpha * (X[a, b, x, y] - X[b, a, x, y]) for the pairs
    Q = (a, b) of the map ``vm``, where X[a, b, x, y] is addressed through
    element strides (sa, sb, sx, sy) of the array X (any layout).
    out: C-contiguous (npair, nx, ny).'''
    nx, ny = out.shape[1], out.shape[2]
    sa, sb, sx, sy = strides
    if USE_C and hasattr(_lib, 'zc_pair_antisym_add_d') and out.flags.c_contiguous \
            and X.dtype == out.dtype:
        base = np.ascontiguousarray(X)
        al = np.array(alpha, dtype=out.dtype)
        getattr(_lib, 'zc_pair_antisym_add_' + _sfx(out.dtype))(
            _p(out), _p(base), _I64(vm.npair), _p(vm.pa_c), _p(vm.pb_c), _I64(nx),
            _I64(ny), _I64(sa), _I64(sb), _I64(sx), _I64(sy), _p(al))
        return out
    flat = np.ascontiguousarray(X).ravel()
    pa, pb = vm.pairs[:, 0], vm.pairs[:, 1]
    x = np.arange(nx)[:, None]
    y = np.arange(ny)[None, :]
    off = x * sx + y * sy
    out += alpha * (flat[(pa * sa + pb * sb)[:, None, None] + off]
                    - flat[(pb * sa + pa * sb)[:, None, None] + off])
    return out


def pair_antisym_batch_add(out, Y, vm, x0, alpha=1.0, block=None):
    '''out[(a,b), :] += alpha (Y[a-x0, b] if x0 <= a < x1) - alpha (Y[b-x0, a]
    if x0 <= b < x1) for the pairs of ``vm`` (rows of out), or for the pairs
    of block ``block`` of ``vm`` (out then holds only that block's rows);
    Y has shape (x1 - x0, n, ncol).'''
    x1 = x0 + Y.shape[0]
    ncol = out.shape[1]
    n = vm.n
    lo, hi = (0, vm.npair) if block is None else vm.blocks[block]
    pa, pb = vm.pa_c[lo:hi], vm.pb_c[lo:hi]
    if USE_C and hasattr(_lib, 'zc_pair_antisym_batch_add_d') and out.flags.c_contiguous:
        Y = np.ascontiguousarray(Y, dtype=out.dtype)
        al = np.array(alpha, dtype=out.dtype)
        getattr(_lib, 'zc_pair_antisym_batch_add_' + _sfx(out.dtype))(
            _p(out), _p(Y), _I64(n), _I64(hi - lo), _p(np.ascontiguousarray(pa)),
            _p(np.ascontiguousarray(pb)), _I64(ncol), _I64(x0), _I64(x1), _p(al))
        return out
    ina = (pa >= x0) & (pa < x1)
    inb = (pb >= x0) & (pb < x1)
    q = np.arange(hi - lo)
    out[q[ina]] += alpha * Y[pa[ina] - x0, pb[ina]]
    out[q[inb]] -= alpha * Y[pb[inb] - x0, pa[inb]]
    return out


def pair_t1_contract(out, V, t1, vm, block=0):
    '''out[Q, e, i] += sum_f V[Q, e, f] t1[i, f] for V antisymmetric in (e, f),
    with the (e<f) columns of V in the numbering of pair block ``block`` of
    ``vm`` (the whole range for maps without spins); out: (nq, n, no).'''
    nq = V.shape[0]
    no, n = t1.shape
    pidx = vm.pidx_blk[block]
    if USE_C and hasattr(_lib, 'zc_pair_t1_contract_d') and out.flags.c_contiguous:
        dt = out.dtype
        getattr(_lib, 'zc_pair_t1_contract_' + _sfx(dt))(
            _p(out), _p(np.ascontiguousarray(V, dtype=dt)),
            _p(np.ascontiguousarray(t1, dtype=dt)), _I64(nq), _I64(n),
            _I64(V.shape[1]), _p(pidx), _I64(no))
        return out
    full = np.zeros((nq, n, n), dtype=np.result_type(V, out))
    e, f = np.nonzero(pidx >= 0)
    full[:, e, f] = V[:, pidx[e, f]]
    full[:, f, e] = -V[:, pidx[e, f]]
    out += full @ t1.T
    return out
