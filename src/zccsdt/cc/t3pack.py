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
    All other maps only depend on this enumeration.

    The labels need not be spins: any label 0 .. ncls - 1 that adds (the
    classes of 2 m_j of a linear molecule, shifted and halved) with the
    indices in the order of their labels; the groups are then the 2 ncls - 1
    sums of two labels and the 3 ncls - 2 sums of three.'''

    def __init__(self, n, spins=None, ncls=None):
        self.n = n
        pi, pj = np.triu_indices(n, 1)
        tri = np.array([(i, j, k) for i in range(n) for j in range(i + 1, n)
                        for k in range(j + 1, n)], dtype=np.int64).reshape(-1, 3)
        if spins is None:
            self.spins = None
        else:
            spins = np.asarray(spins, dtype=np.int64)
            self.ncls = max(2, int(spins.max(initial=0)) + 1) if ncls is None else int(ncls)
            self.npg, self.ntg = 2 * self.ncls - 1, 3 * self.ncls - 2
            assert spins.shape == (n,) and np.all(np.diff(spins) >= 0) \
                and (n == 0 or (spins[0] >= 0 and spins[-1] < self.ncls)), \
                'labels must be 0 .. ncls - 1 (spins: 0/1), in that order'
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
            self.pair_off = np.searchsorted(self.pair_grp, np.arange(self.npg + 1))
            self.tri_off = np.searchsorted(self.tri_grp, np.arange(self.ntg + 1))

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
            self.blocks = [(int(self.pair_off[h]), int(self.pair_off[h + 1])) for h in range(self.npg)]
        self.pidx_blk = []
        for lo, hi in self.blocks:
            loc = np.where((pidx >= lo) & (pidx < hi), pidx - lo, -1)
            self.pidx_blk.append(np.ascontiguousarray(loc, dtype=np.int64))


def triple_map(n, spins=None, ncls=None):
    key = (n, None if spins is None else tuple(int(x) for x in spins), ncls)
    if key not in _CACHE:
        _CACHE[key] = TripleMap(n, spins, ncls)
    return _CACHE[key]


def classes_of(eris):
    '''(labels of the occupied, of the virtual spinors, number of classes)
    as the grouped maps take them, from the symmetry labels of an integral
    object: shifted to start at 0 and divided by their common step.  None if
    there are no labels, if they add modulo something, or if the spinors are
    not in the order of their labels.'''
    lab = getattr(eris, 'labels', None)
    if lab is None or getattr(eris, 'modulus', 0) or not getattr(eris, 'use_labels', True):
        return None
    lab = np.asarray(lab, dtype=np.int64)
    no = eris.nocc
    if np.any(np.diff(lab[:no]) < 0) or np.any(np.diff(lab[no:]) < 0):
        return None
    step = int(np.gcd.reduce(np.diff(np.unique(lab)))) if len(np.unique(lab)) > 1 else 1
    k = (lab - lab.min()) // step
    return k[:no], k[no:], int(k.max()) + 1


def maps_for(eris):
    '''(om, vm) for an integral object: spin-grouped enumerations if the
    orbitals carry spin labels (eris.spin_occ / eris.spin_vir), otherwise
    lexicographic.'''
    so = getattr(eris, 'spin_occ', None)
    sv = getattr(eris, 'spin_vir', None)
    nocc = eris.nocc
    nvir = np.asarray(eris.fock).shape[0] - nocc
    if so is None:
        cls = classes_of(eris)
        if cls is not None:     # symmetry labels: the same block structure as spins
            return triple_map(nocc, cls[0], cls[2]), triple_map(nvir, cls[1], cls[2])
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


def gather_vir_half(T3p, vm, rows, out=None):
    '''T[Q, I, c] = t3[rows[I], Q_0, Q_1, c] for occupied triples ``rows``.
    Shape (nvpair, len(rows), nvir), pair-major for the ladder GEMM (written
    into ``out`` if given).'''
    if USE_C:
        T3p = np.ascontiguousarray(T3p)
        rows = np.ascontiguousarray(rows, dtype=np.int64)
        if out is None:
            out = np.empty((vm.npair, len(rows), vm.n), dtype=T3p.dtype)
        assert out.shape == (vm.npair, len(rows), vm.n) and out.flags.c_contiguous \
            and out.dtype == T3p.dtype
        getattr(_lib, 'zc_gather_vir_half_' + _sfx(T3p.dtype))(
            _p(T3p), _I64(T3p.shape[1]), _p(rows), _I64(len(rows)),
            _p(vm.tri_c), _p(vm.sgn_c), _I64(vm.n), _I64(vm.npair), _p(out))
        return out
    X = np.take(T3p[rows], vm.tri.T, axis=1)     # (nb, nvpair, v)
    X *= vm.sgn.T
    X = np.ascontiguousarray(X.transpose(1, 0, 2))
    if out is None:
        return X
    out[...] = X
    return out


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


def occ_apply(t, Woooo, Foo):
    '''(L t)[J,i,j,k] = P(k/ij) 1/2 sum_mn W[m,n,i,j] t[J,m,n,k]
                      - P(i/jk) sum_m F[m,i] t[J,m,j,k]:
    the hh ladder and the F_oo term of the T3 equation on full tensors (the
    reference of occ_operator_direct).'''
    x = 0.5 * np.einsum('mnij,Jmnk->Jijk', Woooo, t, optimize=True)
    out = x - x.transpose(0, 3, 2, 1) - x.transpose(0, 1, 3, 2)
    y = np.einsum('mi,Jmjk->Jijk', Foo, t, optimize=True)
    out -= y - y.transpose(0, 2, 1, 3) - y.transpose(0, 3, 2, 1)
    return out


def occ_operator_direct(om, Woooo, Foo, rows=None, dtype=None):
    '''The matrix O[I, J] of occ_apply on packed occupied triples, written
    down element by element (3 (C(o,2) + o) terms per row, C kernel) instead
    of applied to the packed basis (occ_operator, o^8 work and o^6-sized
    temporaries).  ``rows`` = (lo, hi): the diagonal block of the triples
    lo <= I < hi only (a block of the symmetry labels).'''
    dtype = np.dtype(np.result_type(Woooo, Foo) if dtype is None else dtype)
    lo, hi = (0, om.ntri) if rows is None else rows
    if not (USE_C and hasattr(_lib, 'zc_occ_operator_d')) or dtype not in (np.float64, np.complex128):
        O = occ_operator(om, lambda t: occ_apply(t, Woooo, Foo))
        return np.ascontiguousarray(O[lo:hi, lo:hi], dtype=dtype)
    n = hi - lo
    O = np.empty((n, n), dtype=dtype)
    loc = np.full(om.ntri, -1, dtype=np.int64)
    loc[lo:hi] = np.arange(n)
    W = np.ascontiguousarray(Woooo, dtype=dtype)
    F = np.ascontiguousarray(Foo, dtype=dtype)
    getattr(_lib, 'zc_occ_operator_' + _sfx(dtype))(
        _p(O), _I64(n), _p(np.arange(lo, hi, dtype=np.int64)), _p(loc), _I64(om.n),
        _I64(om.npair), _I64(om.ntri), _p(om.tri_c), _p(om.sgn_c), _p(om.free_c),
        _p(om.pair_c), _p(om.pa_c), _p(om.pb_c), _p(W), _p(F))
    return O


def add_reduce_pair_first(R, X, vm, rows, pairs=None):
    '''R[rows] += P(c/ab) X, with X[Q, I, c] (pair-major, as produced by the
    ladder GEMM on gather_vir_half).  ``pairs`` = (q0, q1): X holds the pair
    rows q0 <= Q < q1 only (a slab of the operator) and the terms of the
    other pairs are left out.'''
    if USE_C and R.flags.c_contiguous:
        X = np.ascontiguousarray(X, dtype=R.dtype)
        rows = np.ascontiguousarray(rows, dtype=np.int64)
        if pairs is None:
            getattr(_lib, 'zc_reduce_pair_first_' + _sfx(R.dtype))(
                _p(R), _I64(R.shape[1]), _p(X), _p(rows), _I64(len(rows)),
                _I64(vm.n), _I64(vm.npair), _p(vm.free_c), _p(vm.pair_c))
        else:
            getattr(_lib, 'zc_reduce_pair_first_rows_' + _sfx(R.dtype))(
                _p(R), _I64(R.shape[1]), _p(X), _p(rows), _I64(len(rows)),
                _I64(vm.n), _I64(pairs[0]), _I64(pairs[1]), _p(vm.free_c), _p(vm.pair_c))
        return
    if pairs is not None:
        full = np.zeros((vm.npair,) + X.shape[1:], dtype=X.dtype)
        full[pairs[0]:pairs[1]] = X
        X = full
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


def pair_antisym_add(out, X, vm, strides, alpha=1.0, pairs=None):
    '''out[Q, x, y] += alpha * (X[a, b, x, y] - X[b, a, x, y]) for the pairs
    Q = (a, b) of the map ``vm`` (or of the explicit lists pairs = (pa, pb)),
    where X[a, b, x, y] is addressed through element strides (sa, sb, sx, sy)
    of the array X (any layout).  out: C-contiguous (npair, nx, ny).'''
    nx, ny = out.shape[1], out.shape[2]
    sa, sb, sx, sy = strides
    if pairs is None:
        pa, pb = vm.pa_c, vm.pb_c
    else:
        pa, pb = (np.ascontiguousarray(x, dtype=np.int64) for x in pairs)
    if USE_C and hasattr(_lib, 'zc_pair_antisym_add_d') and out.flags.c_contiguous \
            and X.dtype == out.dtype:
        base = np.ascontiguousarray(X)
        al = np.array(alpha, dtype=out.dtype)
        getattr(_lib, 'zc_pair_antisym_add_' + _sfx(out.dtype))(
            _p(out), _p(base), _I64(len(pa)), _p(pa), _p(pb), _I64(nx),
            _I64(ny), _I64(sa), _I64(sb), _I64(sx), _I64(sy), _p(al))
        return out
    flat = np.ascontiguousarray(X).ravel()
    x = np.arange(nx)[:, None]
    y = np.arange(ny)[None, :]
    off = x * sx + y * sy
    out += alpha * (flat[(pa * sa + pb * sb)[:, None, None] + off]
                    - flat[(pb * sa + pa * sb)[:, None, None] + off])
    return out


def axpby(out, inp, alpha=1.0, beta=1.0):
    '''out = beta * out + alpha * inp for arrays (or strided views) of the same
    shape with at most 4 axes, in a C kernel that reads the numpy strides
    (beta = 0 overwrites).  The first two axes are split over threads and the
    last axis is the inner loop, so order the axes accordingly.'''
    assert out.shape == inp.shape and out.ndim <= 4
    # (small arrays: numpy on this thread costs less than waking the others)
    if not (USE_C and hasattr(_lib, 'zc_axpby4_d') and inp.dtype == out.dtype
            and out.dtype in (np.float64, np.complex128)) or out.size < 65536:
        if beta == 0:
            out[...] = alpha * inp
        else:
            if beta != 1:
                out *= beta
            out += alpha * inp
        return out
    pad = 4 - out.ndim
    isz = out.itemsize
    # (unit axes after the first one: a 2-D array is split over its rows)
    k = 1 if out.ndim >= 2 else 0

    def padded(t, fill):
        return np.array(t[:k] + (fill,) * pad + t[k:], dtype=np.int64)
    n = padded(out.shape, 1)
    os_ = padded(tuple(x // isz for x in out.strides), 0)
    is_ = padded(tuple(x // isz for x in inp.strides), 0)
    al = np.array(alpha, dtype=out.dtype)
    be = np.array(beta, dtype=out.dtype)
    getattr(_lib, 'zc_axpby4_' + _sfx(out.dtype))(
        ctypes.c_void_p(out.ctypes.data), ctypes.c_void_p(inp.ctypes.data), _p(n),
        _p(os_), _p(is_), _p(al), _p(be))
    return out


def _ccsd_c(*arrays):
    return USE_C and hasattr(_lib, 'zc_pack_last2_d') and all(
        a.flags.c_contiguous and a.dtype == arrays[0].dtype for a in arrays) \
        and arrays[0].dtype in (np.float64, np.complex128)


def pack_last2(Y, vm):
    '''Y[..., a, b] for the pairs (a, b) of ``vm``: (..., npair), threaded.'''
    Y = np.asarray(Y)
    lead = Y.shape[:-2]
    out = np.empty(lead + (vm.npair,), dtype=Y.dtype)
    if _ccsd_c(Y):
        getattr(_lib, 'zc_pack_last2_' + _sfx(Y.dtype))(
            _p(out), _p(Y), _I64(int(np.prod(lead))), _I64(vm.n), _I64(vm.npair),
            _p(vm.pa_c), _p(vm.pb_c), _p(np.ones(1, dtype=Y.dtype)), ctypes.c_int(0))
    else:
        out[...] = Y[..., vm.pairs[:, 0], vm.pairs[:, 1]]
    return out


def add_asym_last2(R, Y, vm, alpha=1.0):
    '''R[..., Q] += alpha (Y[..., a, b] - Y[..., b, a]) for the pairs Q = (a, b).'''
    if _ccsd_c(R, Y):
        getattr(_lib, 'zc_pack_last2_' + _sfx(R.dtype))(
            _p(R), _p(Y), _I64(R.size // max(vm.npair, 1)), _I64(vm.n), _I64(vm.npair),
            _p(vm.pa_c), _p(vm.pb_c), _p(np.array([alpha], dtype=R.dtype)), ctypes.c_int(1))
    else:
        a, b = vm.pairs[:, 0], vm.pairs[:, 1]
        R += alpha * (Y[..., a, b] - Y[..., b, a])
    return R


def add_asym_pack4(R, X, om, vm, axes=(0, 1, 2, 3), alpha=1.0):
    '''R[P, Q] += alpha (x[i,j,a,b] - x[j,i,a,b] - x[i,j,b,a] + x[j,i,b,a]) for
    P = (i, j) of ``om`` and Q = (a, b) of ``vm``, where x is the 4-index
    array X with its axes ``axes`` taken as (i, j, a, b) (no transposed copy).'''
    if _ccsd_c(R, X):
        st = [X.strides[ax] // X.itemsize for ax in axes]
        getattr(_lib, 'zc_asym_pack4_' + _sfx(R.dtype))(
            _p(R), _p(X), _I64(om.npair), _I64(vm.npair), _p(om.pa_c), _p(om.pb_c),
            _p(vm.pa_c), _p(vm.pb_c), *[_I64(s) for s in st],
            _p(np.array([alpha], dtype=R.dtype)))
    else:
        x = X.transpose(axes)
        i, j = om.pairs[:, 0][:, None], om.pairs[:, 1][:, None]
        a, b = vm.pairs[:, 0], vm.pairs[:, 1]
        R += alpha * (x[i, j, a, b] - x[j, i, a, b] - x[i, j, b, a] + x[j, i, b, a])
    return R


def t2_update(t2, R, om, vm, eo, ev):
    '''t2 + A(R) / D: the Jacobi step of T2 from the packed residual R[(i<j),
    (a<b)], with A(R) its antisymmetric (o, o, v, v) array and
    D[i,j,a,b] = (eo[i] - ev[a]) + (eo[j] - ev[b]).'''
    dt = np.result_type(t2, R)
    t2 = np.ascontiguousarray(t2, dtype=dt)
    R = np.ascontiguousarray(R, dtype=dt)
    if _ccsd_c(t2, R):
        out = np.empty_like(t2)
        getattr(_lib, 'zc_t2_update_' + _sfx(dt))(
            _p(out), _p(t2), _p(R), _I64(om.n), _I64(vm.n), _I64(om.npair), _I64(vm.npair),
            _p(om.pa_c), _p(om.pb_c), _p(vm.pa_c), _p(vm.pb_c),
            _p(np.ascontiguousarray(eo, dtype=np.float64)),
            _p(np.ascontiguousarray(ev, dtype=np.float64)))
        return out
    i, j = om.pairs[:, 0][:, None], om.pairs[:, 1][:, None]
    a, b = vm.pairs[:, 0], vm.pairs[:, 1]
    d = R / ((eo[i] - ev[a]) + (eo[j] - ev[b]))
    out = t2.copy()
    out[i, j, a, b] += d
    out[j, i, a, b] -= d
    out[i, j, b, a] -= d
    out[j, i, b, a] += d
    return out


def vvoo_ket_add(G, B, t1, om):
    '''G[Q, P] += Y[Q, i, j] - Y[Q, j, i] for the pairs P = (i, j) of ``om``,
    Y[Q, i, j] = sum_e t1[i, e] B[Q, e, j], one Q at a time (threads over Q).'''
    if _ccsd_c(G, B, np.ascontiguousarray(t1, dtype=G.dtype)):
        getattr(_lib, 'zc_vvoo_ket_' + _sfx(G.dtype))(
            _p(G), _p(B), _p(np.ascontiguousarray(t1, dtype=G.dtype)), _I64(G.shape[0]),
            _I64(t1.shape[1]), _I64(t1.shape[0]), _I64(om.npair), _p(om.pa_c), _p(om.pb_c))
    else:
        Y = np.matmul(t1, B)
        G += Y[:, om.pairs[:, 0], om.pairs[:, 1]] - Y[:, om.pairs[:, 1], om.pairs[:, 0]]
    return G


def flip_pairs(m):
    '''alpha <-> beta flip of a spin-grouped map with equal numbers of alpha
    and beta indices: (orbital map, pair map, pair re-sorting sign).'''
    na = int(np.sum(m.spins == 0))
    assert 2 * na == m.n, 'spin flip needs equal numbers of alpha and beta'
    r = np.arange(m.n)
    orb = np.where(r < na, r + na, r - na)
    fp = orb[m.pairs]
    sgn = np.where(fp[:, 0] < fp[:, 1], 1.0, -1.0)
    pair = m.pidx[fp.min(axis=1), fp.max(axis=1)]
    return orb, np.ascontiguousarray(pair, dtype=np.int64), sgn


def flip_sym3(X, f, s, rep):
    '''X <- X + flip(X) in place for a 3-index array or strided view, with
    flip(X)[i,j,k] = s0[i] s1[j] s2[k] X[f0[i], f1[j], f2[k]] (involutions f,
    flip-invariant signs s); rep: indices of axis 0, one per orbit of f0.'''
    f = [np.ascontiguousarray(x, dtype=np.int64) for x in f]
    s = [np.ascontiguousarray(x, dtype=np.float64) for x in s]
    if USE_C and hasattr(_lib, 'zc_flip_sym3_d') and X.dtype in (np.float64, np.complex128):
        isz = X.itemsize
        n = np.array(X.shape, dtype=np.int64)
        st = np.array([x // isz for x in X.strides], dtype=np.int64)
        rep = np.ascontiguousarray(rep, dtype=np.int64)
        getattr(_lib, 'zc_flip_sym3_' + _sfx(X.dtype))(
            ctypes.c_void_p(X.ctypes.data), _p(n), _p(st), _p(f[0]), _p(f[1]), _p(f[2]),
            _p(s[0]), _p(s[1]), _p(s[2]), _p(rep), _I64(len(rep)))
        return X
    sg = s[0][:, None, None] * s[1][None, :, None] * s[2][None, None, :]
    X[...] = X + sg * X[np.ix_(f[0], f[1], f[2])]
    return X


def add_transposed(out, inp, alpha=1.0):
    '''out += alpha * inp.T for 2-D arrays (out: C-contiguous (nr, nc),
    inp: C-contiguous (nc, nr)), with a blocked C kernel.'''
    nr, nc = out.shape
    assert inp.shape == (nc, nr)
    if USE_C and hasattr(_lib, 'zc_add_transposed_d') and out.flags.c_contiguous \
            and inp.flags.c_contiguous and inp.dtype == out.dtype:
        al = np.array(alpha, dtype=out.dtype)
        getattr(_lib, 'zc_add_transposed_' + _sfx(out.dtype))(
            _p(out), _p(inp), _I64(nr), _I64(nc), _I64(nc), _I64(nr), _p(al))
        return out
    out += alpha * inp.T
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
