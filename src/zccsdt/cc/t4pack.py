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

These are the numpy versions (fancy indexing, one thread, tables of the size
of the tuples): the reference of the kernels and good for small cases.

tn_expand(), tn_reduce() and tn_rows() are the kernels (csrc/tnpack.c): one
group pair at a time - a packed row as t[g1, g2] and the antisymmetrizing
reduction of X[g1, g2] onto the sorted tuples -, threaded and without
tables; they fall back on the tables where the library is not built.
'''

import ctypes
import functools
import os
from itertools import combinations
from math import comb

import numpy as np


def _load_lib():
    path = os.environ.get('ZCCSDT_TNPACK_LIB') or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lib', 'libzc_tnpack.so')
    try:
        return ctypes.CDLL(path)
    except OSError:
        return None


_lib = _load_lib()
USE_C = _lib is not None        # set False to force the numpy implementations
_SFX = {np.dtype(np.float64): 'd', np.dtype(np.complex128): 'z'}
_I64, _INT, _DBL = ctypes.c_int64, ctypes.c_int, ctypes.c_double


def _p(a):
    return ctypes.c_void_p(a.ctypes.data)


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


def fold(G, sp, axis1, axis2):
    '''sum over the splits of the sorted tuples of ``sp`` of
    sgn * G[.., group 1, .., group 2, ..] (the two groups on axis1 and
    axis2): the antisymmetrizer of an array that is antisymmetric within
    each group.  The tuple is the first axis of the result, the other axes
    follow in their order.'''
    G = np.moveaxis(G, (axis1, axis2), (0, 1))
    out = 0
    for s in range(sp.nsplit):
        out = out + G[sp.idx1[:, s], sp.idx2[:, s]] * sp.sgn[:, s].reshape((-1,) + (1,) * (G.ndim - 2))
    return out


# ------------------------------------------------- kernels on one group pair

def _layout(X, outer):
    '''(strides of group 1, group 2 and the outer index in elements, number
    of outer indices) of X[g1, g2] or of X with one more axis ``outer``.'''
    assert X.flags.c_contiguous
    st = [x // X.itemsize for x in X.strides]
    if X.ndim == 2:
        assert outer is None
        return st[0], st[1], 0, 1
    assert X.ndim == 3 and outer in (0, 1, 2)
    g = [ax for ax in range(3) if ax != outer]
    return st[g[0]], st[g[1]], st[outer], X.shape[outer]


def _ranges(n, k1, k2, rows, cols):
    return (rows or (0, comb(n, k1))) + (cols or (0, comb(n, k2)))


def tn_reduce(R, X, n, k1, k2, rows=None, cols=None, scale=1.0, outer=None):
    '''R[A] += scale * sum over the splits (g1 | g2) of the sorted
    (k1+k2)-tuples A of range(n) of sgn * X[g1 - rows[0], g2 - cols[0]], for
    the groups inside the block of X: rows = (b0, b1) of the k1-tuples and
    cols = (c0, c1) of the k2-tuples (default: all).  X may have one more
    axis ``outer`` (in any place); R is then [outer, A].'''
    s1, s2, so, nout = _layout(X, outer)
    b0, b1, c0, c1 = _ranges(n, k1, k2, rows, cols)
    nA = comb(n, k1 + k2)
    assert R.flags.c_contiguous and R.dtype == X.dtype and R.shape == ((nA,) if outer is None else (nout, nA))
    if USE_C:
        getattr(_lib, 'zc_tn_reduce_' + _SFX[R.dtype])(
            _p(R), _I64(nout), _I64(nA), _INT(n), _INT(k1), _INT(k2), _I64(b0), _I64(b1), _I64(c0),
            _I64(c1), _p(X), _I64(s1), _I64(s2), _I64(so), _DBL(scale))
        return R
    sp = split(n, k1, k2)
    Xo = X[None] if outer is None else np.moveaxis(X, outer, 0)
    Ro = R.reshape(nout, nA)
    for s in range(sp.nsplit):
        i1, i2 = sp.idx1[:, s], sp.idx2[:, s]
        sel = np.where((i1 >= b0) & (i1 < b1) & (i2 >= c0) & (i2 < c1))[0]
        Ro[:, sel] += scale * sp.sgn[sel, s] * Xo[:, i1[sel] - b0, i2[sel] - c0]
    return R


def tn_expand(X, T, n, k1, k2, rows=None, cols=None, scale=1.0, outer=None):
    '''X[g1 - rows[0], g2 - cols[0]] = scale * t[g1, g2] of the packed tuples
    T[A] (zero where the groups share an index), the inverse layout of
    tn_reduce: X [b1 - b0, c1 - c0], with one more axis ``outer`` for
    T [outer, A].'''
    s1, s2, so, nout = _layout(X, outer)
    b0, b1, c0, c1 = _ranges(n, k1, k2, rows, cols)
    nA = comb(n, k1 + k2)
    assert T.flags.c_contiguous and T.dtype == X.dtype and T.shape == ((nA,) if outer is None else (nout, nA))
    if USE_C:
        _lib.zc_tn_zero(_p(X), _I64(X.nbytes))
        getattr(_lib, 'zc_tn_expand_' + _SFX[X.dtype])(
            _p(T), _I64(nout), _I64(nA), _INT(n), _INT(k1), _INT(k2), _I64(b0), _I64(b1), _I64(c0),
            _I64(c1), _p(X), _I64(s1), _I64(s2), _I64(so), _DBL(scale))
        return X
    sp = split(n, k1, k2)
    Xo = X[None] if outer is None else np.moveaxis(X, outer, 0)
    Xo[...] = scale * T.reshape(nout, nA)[:, sp.pos[b0:b1, c0:c1]] * sp.psg[b0:b1, c0:c1]
    return X


def _entries(row, xoff, osgn):
    row = np.ascontiguousarray(row, dtype=np.int64)
    xoff = np.ascontiguousarray(xoff, dtype=np.int64)
    osgn = np.ascontiguousarray(osgn, dtype=np.float64)
    assert row.shape == xoff.shape == osgn.shape and row.ndim == 1
    return row, xoff, osgn


def tn_reduce2(R, X, n, k1, k2, entries, s1, s2, rows=None, cols=None, scale=1.0):
    '''R[row, A] += scale * osgn * sum over the splits (g1 | g2) of the sorted
    (k1+k2)-tuples A of sgn * X.flat[xoff + (g1 - rows[0]) * s1 + (g2 - cols[0]) * s2]
    for every entry (row, xoff, osgn) of ``entries`` (three arrays): the
    reduction of a product on two groups of occupied and two groups of
    virtual indices - an entry per split of an occupied tuple (occ_reduce()),
    the splits of the virtual tuples here.  R: [rows, A]; X: any layout, as
    the strides (in elements) and the offsets say.'''
    row, xoff, osgn = _entries(*entries)
    b0, b1, c0, c1 = _ranges(n, k1, k2, rows, cols)
    assert R.flags.c_contiguous and X.flags.c_contiguous and R.dtype == X.dtype and R.ndim == 2
    assert R.shape[1] == comb(n, k1 + k2)
    order = np.argsort(row, kind='stable')          # (the kernel wants the entries of a row together)
    row, xoff, osgn = row[order], xoff[order], osgn[order]
    getattr(_lib, 'zc_tn_reduce2_' + _SFX[R.dtype])(
        _p(R), _I64(R.shape[1]), _I64(len(row)), _p(row), _p(xoff), _p(osgn), _INT(n), _INT(k1), _INT(k2),
        _I64(b0), _I64(b1), _I64(c0), _I64(c1), _p(X), _I64(s1), _I64(s2), _DBL(scale))
    return R


def tn_expand2(X, M, n, k1, k2, entries, s1, s2, rows=None, cols=None, scale=1.0):
    '''X.flat[xoff + (g1 - rows[0]) * s1 + (g2 - cols[0]) * s2] = scale * osgn * t[g1, g2]
    of the packed tuples M[row] for every entry (row, xoff, osgn): rows of a
    packed block on two groups of virtual indices, each at its place
    (occ_expand(): the rows that two groups of occupied indices make).  X
    is zeroed first.'''
    row, xoff, osgn = _entries(*entries)
    b0, b1, c0, c1 = _ranges(n, k1, k2, rows, cols)
    assert M.flags.c_contiguous and X.flags.c_contiguous and M.dtype == X.dtype and M.ndim == 2
    assert M.shape[1] == comb(n, k1 + k2)
    _lib.zc_tn_zero(_p(X), _I64(X.nbytes))
    getattr(_lib, 'zc_tn_expand2_' + _SFX[X.dtype])(
        _p(M), _I64(M.shape[1]), _I64(len(row)), _p(row), _p(xoff), _p(osgn), _INT(n), _INT(k1), _INT(k2),
        _I64(b0), _I64(b1), _I64(c0), _I64(c1), _p(X), _I64(s1), _I64(s2), _DBL(scale))
    return X


def tn_expand1(X, M, n, k2, entries, s1, cols=None, scale=1.0):
    '''tn_expand2 for the splits (a | K) with one index in the first group
    and X contiguous in the second (s2 = 1): the kernel walks through the
    block cols = (c0, c1) of the k2-tuples K instead of through all tuples -
    for products in many batches of columns.'''
    row, xoff, osgn = _entries(*entries)
    c0, c1 = cols or (0, comb(n, k2))
    assert M.flags.c_contiguous and X.flags.c_contiguous and M.dtype == X.dtype and M.ndim == 2
    assert M.shape[1] == comb(n, k2 + 1)
    _lib.zc_tn_zero(_p(X), _I64(X.nbytes))
    getattr(_lib, 'zc_tn_expand1_' + _SFX[X.dtype])(
        _p(M), _I64(M.shape[1]), _I64(len(row)), _p(row), _p(xoff), _p(osgn), _INT(n), _INT(k2),
        _I64(c0), _I64(c1), _p(X), _I64(s1), _DBL(scale))
    return X


def tn_reduce1(R, X, n, k2, entries, s1, cols=None, scale=1.0):
    '''tn_reduce2 for the splits (a | K) with one index in the first group
    and X contiguous in the second: the slabs of the entries of a row are
    summed first, then reduced once, walking through the block
    cols = (c0, c1) of the k2-tuples.'''
    row, xoff, osgn = _entries(*entries)
    c0, c1 = cols or (0, comb(n, k2))
    assert R.flags.c_contiguous and X.flags.c_contiguous and R.dtype == X.dtype and R.ndim == 2
    assert R.shape[1] == comb(n, k2 + 1)
    order = np.argsort(row, kind='stable')
    row, xoff, osgn = row[order], xoff[order], osgn[order]
    getattr(_lib, 'zc_tn_reduce1_' + _SFX[R.dtype])(
        _p(R), _I64(R.shape[1]), _I64(len(row)), _p(row), _p(xoff), _p(osgn), _INT(n), _INT(k2),
        _I64(c0), _I64(c1), _p(X), _I64(s1), _DBL(scale))
    return R


def tn_axpy_rows(R, X, tgt, src, sgn, col0=0):
    '''R[tgt[e], col0 + c] += sgn[e] * X.flat[src[e] + c] for c < X.shape[-1]:
    rows of X added to rows of R - the reduction over the splits of the
    occupied tuples alone (occ_reduce() with the packed virtual tuple last
    in X).'''
    tgt, src, sgn = _entries(tgt, src, sgn)
    ncol = X.shape[-1]
    assert R.flags.c_contiguous and X.flags.c_contiguous and R.dtype == X.dtype and R.ndim == 2
    assert col0 + ncol <= R.shape[1]
    getattr(_lib, 'zc_tn_axpy_rows_' + _SFX[R.dtype])(
        ctypes.c_void_p(R.ctypes.data + col0 * R.itemsize), _I64(R.shape[1]), _I64(len(tgt)), _p(tgt),
        _p(src), _p(sgn), _p(X), _I64(ncol))
    return R


def occ_reduce(so, s1, s2, jo=None, only=None):
    '''The entries (row, offset, sign) of the reduction over the splits of
    the occupied tuples of ``so``: for every tuple I and split s whose
    second group is in jo = (b0, b1), X at idx1 * s1 + (idx2 - b0) * s2.
    ``only``: the splits to take (default: all).'''
    b0, b1 = jo or (0, so.n2)
    row, off, sg = [], [], []
    for s in (range(so.nsplit) if only is None else only):
        Is = so.select(s, b0, b1)
        row.append(Is)
        off.append(so.idx1[Is, s] * s1 + (so.idx2[Is, s] - b0) * s2)
        sg.append(so.sgn[Is, s])
    return np.concatenate(row), np.concatenate(off), np.concatenate(sg).astype(np.float64)


def occ_expand(so, s1, s2, jo=None):
    '''The entries (row, offset, sign) of t[g1, g2] for all first groups and
    the second groups in jo = (b0, b1) of the occupied split ``so``: the
    packed row that the two groups make, X at g1 * s1 + (g2 - b0) * s2.'''
    b0, b1 = jo or (0, so.n2)
    g1, g2 = np.meshgrid(np.arange(so.n1), np.arange(b0, b1), indexing='ij')
    sg = so.psg[:, b0:b1].ravel()
    keep = sg != 0
    return (so.pos[:, b0:b1].ravel()[keep], (g1 * s1 + (g2 - b0) * s2).ravel()[keep],
            sg[keep].astype(np.float64))


def tn_rows(out, M, rows, sgn, cols=None):
    '''out[r] = sgn[r] * M[rows[r], c0:c1]: rows of a packed block with their
    signs (sgn 0: zeros).'''
    c0, c1 = cols or (0, M.shape[1])
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    sgn = np.ascontiguousarray(sgn, dtype=np.int8)
    assert out.flags.c_contiguous and out.shape == (len(rows), c1 - c0) and out.dtype == M.dtype
    if USE_C and M.flags.c_contiguous:
        getattr(_lib, 'zc_tn_rows_' + _SFX[out.dtype])(
            _p(M), _I64(M.shape[1]), _p(rows), _p(sgn), _I64(len(rows)), _I64(c0), _I64(c1), _p(out),
            _I64(out.shape[1]))
        return out
    np.multiply(M[rows, c0:c1], sgn[:, None], out=out)
    return out


def zero(a):
    '''a[...] = 0 on all threads.'''
    if USE_C and a.flags.c_contiguous:
        _lib.zc_tn_zero(_p(a), _I64(a.nbytes))
    else:
        a[...] = 0
    return a


def tuple_sums(e, k):
    '''sum_t e[a_t] of the sorted k-tuples of range(len(e)): [C(n,k)].'''
    e = np.ascontiguousarray(e, dtype=np.float64)
    if USE_C:
        out = np.empty(comb(len(e), k))
        _lib.zc_tn_tuple_sums(_p(e), _INT(len(e)), _INT(k), _p(out))
        return out
    return e[tuples(len(e), k)].sum(axis=1)


def energy2(z, r1, r2, ev, eo):
    '''(sum Re(conj(z) (r1 + r2)) / (eo - ev), sum Re(conj(r1) (r1 + r2)) / (eo - ev)).'''
    if USE_C:
        e = np.zeros(2)
        getattr(_lib, 'zc_tn_energy2_' + _SFX[z.dtype])(_p(z), _p(r1), _p(r2), _p(ev), _DBL(eo),
                                                       _I64(len(z)), _p(e))
        return float(e[0]), float(e[1])
    r = (r1 + r2) / (eo - ev)
    return float(np.vdot(z, r).real), float(np.vdot(r1, r).real)


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
