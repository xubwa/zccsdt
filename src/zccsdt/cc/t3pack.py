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

import numpy as np

_CACHE = {}


class TripleMap:
    def __init__(self, n):
        self.n = n
        pi, pj = np.triu_indices(n, 1)
        self.npair = len(pi)
        self.pairs = np.stack((pi, pj), axis=1)
        pidx = -np.ones((n, n), dtype=np.int64)
        pidx[pi, pj] = np.arange(self.npair)
        self.pidx = pidx
        tri = np.array([(i, j, k) for i in range(n) for j in range(i + 1, n)
                        for k in range(j + 1, n)], dtype=np.int64).reshape(-1, 3)
        self.triples = tri
        self.ntri = len(tri)
        tidx = -np.ones((n, n, n), dtype=np.int64)
        if self.ntri:
            tidx[tri[:, 0], tri[:, 1], tri[:, 2]] = np.arange(self.ntri)

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


def triple_map(n):
    if n not in _CACHE:
        _CACHE[n] = TripleMap(n)
    return _CACHE[n]


def gather_half(T3p, om, vm, pairs):
    '''Half-expanded T[m, P, d, Q] = t3[m, P_0, P_1, d, Q_0, Q_1] for the
    occupied pairs ``pairs`` (array of pair indices), all m, d and Q.
    Shape (nocc, len(pairs), nvir, nvpair).'''
    rows = om.tri[:, pairs]                      # (o, nb)
    sgo = om.sgn[:, pairs]
    X = T3p[rows.ravel()]                        # (o*nb, nvtri), contiguous rows
    X = np.take(X, vm.tri, axis=1)               # (o*nb, v, nvpair)
    X *= vm.sgn
    X *= sgo.reshape(-1, 1, 1)
    return X.reshape(om.n, len(pairs), vm.n, vm.npair)


def gather_vir_half(T3p, vm, rows):
    '''T[I, Q, c] = t3[I, Q_0, Q_1, c] for occupied triples ``rows``.
    Shape (len(rows), nvpair, nvir).'''
    X = np.take(T3p[rows], vm.tri.T, axis=1)     # (nb, nvpair, v)
    X *= vm.sgn.T
    return X


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


def scatter_occ(R, Xv, om, pairs, scale=1.0):
    '''R[tri(x, P)] += scale * P(x/P) Xv[x, P, :] for the pairs in ``pairs``.
    Xv has shape (nocc, len(pairs), ncol).  For a fixed position r of the free
    index in the target triple, the map (x, P) -> triple is injective, so each
    r is a collision-free fancy-index update.'''
    tri = om.tri[:, pairs]
    sgn = om.sgn[:, pairs]
    x = np.arange(om.n)[:, None]
    P = om.pairs[pairs][None, :, :]
    # position of x in its sorted triple: 0 if x < P0, 1 if P0 < x < P1, 2 if x > P1
    pos = (x > P[..., 0]).astype(np.int8) + (x > P[..., 1])
    valid = sgn != 0
    for r in range(3):
        m = valid & (pos == r)
        if not m.any():
            continue
        xs, ps = np.nonzero(m)
        R[tri[xs, ps]] += (scale * sgn[xs, ps])[:, None] * Xv[xs, ps]


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
