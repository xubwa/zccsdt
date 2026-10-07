'''Packed antisymmetric amplitudes of any rank and their views.

An amplitude t[i<j<.., a<b<..] is stored on its unique block only,
``T[I, A]`` with I and A the positions of the sorted occupied and virtual
tuples (lexicographic, as itertools.combinations).  A contraction needs the
indices of a tuple in two groups - one that is summed or that a product
carries, and the rest -, in every way the tuple can be split:

    (p<q<r<s) = (p | qrs) = -(q | prs) = ..   or   (pq | rs) = -(pr | qs) = ..

Split(n, k1, k2) has the tables of one kind of split of the sorted
(k1+k2)-tuples of range(n):

    idx1[J, s], idx2[J, s], sgn[J, s]    the s-th split of tuple J: positions of
                                         its two groups among the sorted k1-
                                         and k2-tuples, and the sign of
                                         t[group 1, group 2] = sgn * t[sorted]
    pos[J1, J2], psg[J1, J2]             the inverse: the tuple that two groups
                                         make (psg = 0 where they overlap)

view() builds t[group 1 occ, group 2 occ, group 1 vir, group 2 vir] from the
packed block - the operand of a matrix product -, and reduce() adds the
antisymmetrized result of a product back onto the packed block: for every
split of the target, sgn * X[its groups].  Both take ranges of the second
groups, so that a product can be done in batches of columns.

These are the numpy versions (fancy indexing, one thread): the reference of
the kernels and good for small cases.
'''

import functools
from itertools import combinations
from math import comb

import numpy as np


@functools.lru_cache(maxsize=None)
def tuples(n, k):
    '''The sorted k-tuples of range(n): [C(n,k), k] (lexicographic).'''
    return np.array(list(combinations(range(n), k)), dtype=np.int64).reshape(comb(n, k), k)


@functools.lru_cache(maxsize=None)
def ranks(n, k):
    '''r[p, q, ..] = position of the sorted tuple p<q<.. (-1 elsewhere).'''
    r = np.full((n,) * k, -1, dtype=np.int64)
    t = tuples(n, k)
    r[tuple(t[:, x] for x in range(k))] = np.arange(len(t))
    return r


class Split:
    '''The sorted (k1+k2)-tuples of range(n) as a group of k1 and one of k2.'''

    def __init__(self, n, k1, k2):
        self.n, self.k1, self.k2 = n, k1, k2
        k = k1 + k2
        t = tuples(n, k)
        self.size, self.n1, self.n2 = len(t), comb(n, k1), comb(n, k2)
        r1, r2 = ranks(n, k1), ranks(n, k2)
        subsets = list(combinations(range(k), k1))
        self.nsplit = len(subsets)
        self.idx1 = np.empty((self.size, self.nsplit), dtype=np.int64)
        self.idx2 = np.empty((self.size, self.nsplit), dtype=np.int64)
        self.sgn = np.empty((self.size, self.nsplit), dtype=np.int8)
        for s, first in enumerate(subsets):
            rest = [x for x in range(k) if x not in first]
            # (first, rest) as a permutation of the sorted tuple
            inversions = sum(1 for x in first for y in rest if y < x)
            self.idx1[:, s] = r1[tuple(t[:, x] for x in first)]
            self.idx2[:, s] = r2[tuple(t[:, x] for x in rest)]
            self.sgn[:, s] = (-1) ** inversions
        self.pos = np.zeros((self.n1, self.n2), dtype=np.int64)
        self.psg = np.zeros((self.n1, self.n2), dtype=np.int8)
        rows = np.arange(self.size)[:, None]
        self.pos[self.idx1, self.idx2] = rows
        self.psg[self.idx1, self.idx2] = self.sgn
        # the targets of split s by their second group, for batches of it
        self._order = [np.argsort(self.idx2[:, s], kind='stable') for s in range(self.nsplit)]
        self._sorted = [self.idx2[o, s] for s, o in enumerate(self._order)]

    def select(self, s, b0, b1):
        '''The tuples whose s-th split has its second group in b0:b1.'''
        lo, hi = np.searchsorted(self._sorted[s], (b0, b1))
        return self._order[s][lo:hi]


@functools.lru_cache(maxsize=None)
def split(n, k1, k2):
    return Split(n, k1, k2)


def view(T, so, sv, jo=None, jv=None):
    '''t[g1o, g2o, g1v, g2v] of the packed block T[I, A] for the second
    groups in jo = (b0, b1) and jv = (c0, c1) (default: all).  A split may be
    None: that index stays the packed tuple, [I, ..] or [.., A], and jo / jv
    is then a range of tuples.'''
    if so is None:
        b0, b1 = jo or (0, T.shape[0])
    else:
        b0, b1 = jo or (0, so.n2)
    if sv is None:
        c0, c1 = jv or (0, T.shape[1])
    else:
        c0, c1 = jv or (0, sv.n2)
    if so is None and sv is None:
        return T[b0:b1, c0:c1]
    if so is None:
        return T[b0:b1][:, sv.pos[:, c0:c1]] * sv.psg[None, :, c0:c1]
    rows = so.pos[:, b0:b1]
    if sv is None:
        return T[rows, c0:c1] * so.psg[:, b0:b1, None]
    out = T[rows[:, :, None, None], sv.pos[None, None, :, c0:c1]]
    out *= so.psg[:, b0:b1, None, None] * sv.psg[None, None, :, c0:c1]
    return out


def reduce(R, X, so, sv, jo=None, jv=None, scale=1.0):
    '''R[I, A] += scale * sum over the splits (s, s') of (I, A) of
    sgn * X[g1o, g2o - b0, g1v, g2v - c0], for the splits whose second groups
    are in the batches jo = (b0, b1), jv = (c0, c1) of X.  A split may be
    None: X has the packed tuple there, X[I - b0, ..] or X[.., A - c0].'''
    if so is None:
        b0, b1 = jo or (0, R.shape[0])
        osplits = [(np.arange(b0, b1), None)]
    else:
        b0, b1 = jo or (0, so.n2)
        osplits = [(so.select(s, b0, b1), s) for s in range(so.nsplit)]
    if sv is None:
        c0, c1 = jv or (0, R.shape[1])
        vsplits = [(np.arange(c0, c1), None)]
    else:
        c0, c1 = jv or (0, sv.n2)
        vsplits = [(sv.select(s, c0, c1), s) for s in range(sv.nsplit)]
    for Is, s in osplits:
        if not len(Is):
            continue
        if s is None:
            Xs, fo = X[Is - b0], None
        else:
            Xs, fo = X[so.idx1[Is, s], so.idx2[Is, s] - b0], so.sgn[Is, s]
        for As, sp in vsplits:
            if not len(As):
                continue
            if sp is None:
                vals = Xs[:, As - c0] * scale
            else:
                vals = Xs[:, sv.idx1[As, sp], sv.idx2[As, sp] - c0] * (scale * sv.sgn[As, sp])
            if fo is not None:
                vals = vals * fo[:, None]
            R[Is[:, None], As[None, :]] += vals
    return R


def pack(t):
    '''The unique block [C(no,k), C(nv,k)] of a full antisymmetric array.'''
    k = t.ndim // 2
    occ, vir = tuples(t.shape[0], k), tuples(t.shape[-1], k)
    return t[tuple(occ[:, x][:, None] for x in range(k)) + tuple(vir[:, x][None, :] for x in range(k))]


def unpack(T, no, nv, k):
    '''The full antisymmetric array of a packed block.'''
    from itertools import permutations
    occ, vir = tuples(no, k), tuples(nv, k)
    out = np.zeros((no,) * k + (nv,) * k, dtype=T.dtype)

    def sign(p):
        return (-1) ** sum(1 for x in range(k) for y in range(x + 1, k) if p[x] > p[y])
    for p in permutations(range(k)):
        for q in permutations(range(k)):
            out[tuple(occ[:, p[x]][:, None] for x in range(k))
                + tuple(vir[:, q[x]][None, :] for x in range(k))] = sign(p) * sign(q) * T
    return out
