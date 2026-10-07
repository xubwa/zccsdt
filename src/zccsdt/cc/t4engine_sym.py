'''The T4 equation of CCSDTQ in blocks of a symmetry label: zccsdt.cc.t4engine
for spinors that carry a label of an Abelian group which adds without a
modulus (2 m_j of a linear molecule; no symmetry is the case of one label).

The labels are those of zccsdt.cc.t3pack.classes_of(): 0 .. ncls - 1, the
occupied and the virtual spinors each in the order of their labels, so that
the spinors of one label are a range.  An amplitude vanishes unless the
labels of its occupied indices sum to those of its virtual ones:

    T4, residual    one dense block per total label M: [occupied quadruples of
                    label M, virtual quadruples of label M], both in
                    lexicographic order (Layout; to_packed() / from_packed()
                    for the packed array [C(no,4), C(nv,4)] of the dense code)
    T3              the blocks of the CCSDT step (zccsdt.cc.t3engine.T3Layout):
                    one per label, the triples in lexicographic order

Every product of zccsdt.cc.t4engine is done in the blocks that the labels
leave - the operands with a pair or a triple of virtual indices in the
classes of their labels - and reduced into the blocks of the residual by
kernels that walk through the virtual quadruples of one label
(csrc/t4sym.c):

- pp ladder: per label M the rows in batches, T4 as [(ef), (I, cd)] for the
  pairs (ef) of one label q: one product with the block of the ladder
  operator per (M, q), and a thin one for the T4 terms of R2, R3 and Xo.
- ring: per occupied triple J the blocks t(m J; e K) with the label
  k_m - k_e and the triples K of one label, m outside J, times that block
  of the ring operator; the thin products of the T4 terms of R3 and Xv are
  rows of the same product.
- hh ladder: a matrix of the size of the occupied quadruples of one label
  times the block.
- the products with a six-index intermediate and of T3 with T3: per
  occupied quadruple one block X[(ab), (cd)] per label of (ab) and one
  X[a, (bcd)] per label of a, each the sum over the splits of the quadruple
  of small products on blocks of the operands; these, and the products
  that make the intermediates of T3, are tables of products that a kernel
  runs on all threads (zc_t4s_tasks).

What is kept in blocks besides T4 and its residual: the six-index
intermediate Xv (for every occupied pair and label of its first index, with
the rows of T3 that the same product needs above it), T3 expanded on one
virtual index and a pair, the intermediate of T3 x T3 and the T4 term of R3
before its reduction.  Dense (they are of the size of T2 times a few
indices): the dressed integrals as zccsdt.cc.t4engine.Dressed has them -
<vv||vv> on pairs included -, the intermediates of T2 and Xo.

Labels with a modulus are not done here: zccsdt.cc.t3pack.classes_of()
gives None for them and the callers use the dense code.
'''

import ctypes
import os
from math import comb

import numpy as np

from zccsdt.blas import gemm
from zccsdt.cc import t3kern, t4engine, t4pack as P
from zccsdt.timing import timer

BATCH_BYTES = 1 << 30           # an operand or a product of one batch, at most
PIECES = t4engine.PIECES
_I64, _INT = ctypes.c_int64, ctypes.c_int
_BUFS = ('t12', 'oovv', 'z', 'oovoG', 'g', 'wvA', 'gB', 'la13', 'y3A', 't3p', 'r13', 'xo', 't2A', 'x22',
         'x13', 'oovoA', 'vo', 'wrB', 't2xq', 'mx', 'wblk', 't2px', 'u3')
_B = {name: k for k, name in enumerate(_BUFS)}


def _load_lib():
    path = os.environ.get('ZCCSDT_T4SYM_LIB') or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lib', 'libzc_t4sym.so')
    try:
        return ctypes.CDLL(path)
    except OSError:
        return None


_lib = _load_lib()
if _lib is not None:
    _lib.zc_t4s_quads.restype = ctypes.c_int64


def _p(a):
    return ctypes.c_void_p(a.ctypes.data)


def classes(eris):
    '''(labels of the occupied, of the virtual spinors, number of classes)
    for the blocks, or None: no labels, labels with a modulus, spinors not
    in the order of their labels, fewer than four spinors of a kind, or the
    kernels are not built.'''
    from zccsdt.cc import t3pack
    if _lib is None or not t3kern.USE_C or not hasattr(t3kern._lib, 'zc_gemm_d'):
        return None
    cls = t3pack.classes_of(eris)
    if cls is None or len(cls[0]) < 4 or len(cls[1]) < 4 or len(cls[1]) > 32767:
        return None
    return cls


def available(eris):
    return classes(eris) is not None


def _by_class(lab, n):
    '''(members of every class 0 .. n - 1, position of every element in its
    class, the members in the order of the classes, where each class starts).'''
    lab = np.asarray(lab, dtype=np.int64)
    order = np.argsort(lab, kind='stable')
    off = np.searchsorted(lab[order], np.arange(n + 1))
    pos = np.empty(len(lab), dtype=np.int64)
    pos[order] = np.arange(len(lab)) - off[lab[order]]
    return [order[off[c]:off[c + 1]] for c in range(n)], pos, order, off


class _Tab:
    '''A table of products for zc_t4s_tasks, in groups with one target.'''

    def __init__(self):
        self.rows, self.alpha, self.piece, self.gstart, self.key = [], [], [], [0], []

    def term(self, piece, a, aoff, lda, ta, b, boff, ldb, c, coff, ldc, m, n, k, alpha):
        if m > 0 and n > 0 and k > 0:
            self.rows.append((_B[a], aoff, lda, ta, _B[b], boff, ldb, _B[c], coff, ldc, m, n, k))
            self.alpha.append(alpha)
            self.piece.append(piece)

    def group(self, key=0):
        if len(self.rows) > self.gstart[-1]:
            self.gstart.append(len(self.rows))
            self.key.append(key)

    def done(self):
        self.group()
        self.tab = np.array(self.rows, dtype=np.int64).reshape(-1, 13)
        self.alpha = np.array(self.alpha, dtype=np.float64)
        self.piece = np.array(self.piece, dtype=np.int64)
        self.gstart = np.array(self.gstart, dtype=np.int64)
        self.key = np.array(self.key[:len(self.gstart) - 1], dtype=np.int64)
        self._sel = {}
        del self.rows
        return self

    def select(self, pieces=None, keys=None):
        '''(gstart, tab, alpha) of the terms of ``pieces`` (numbers; None:
        all) in the groups with keys in keys = (k0, k1) (None: all).'''
        g0, g1 = (0, len(self.key)) if keys is None else np.searchsorted(self.key, keys)
        if pieces is None:
            return self.gstart[g0:g1 + 1], self.tab, self.alpha
        pieces = tuple(sorted(pieces))
        if pieces not in self._sel:
            keep = np.isin(self.piece, pieces)
            cum = np.concatenate(([0], np.cumsum(keep)))
            self._sel[pieces] = (cum[self.gstart], np.ascontiguousarray(self.tab[keep]),
                                 np.ascontiguousarray(self.alpha[keep]))
        gs, tab, al = self._sel[pieces]
        return np.ascontiguousarray(gs[g0:g1 + 1]), tab, al


_PIECE = {name: k for k, name in enumerate(PIECES)}
_LAYOUTS = {}


def layout_for(eris):
    '''The Layout of the labels of ``eris`` (kept), or None (classes()).'''
    cls = classes(eris)
    if cls is None:
        return None
    key = (tuple(int(x) for x in cls[0]), tuple(int(x) for x in cls[1]), int(cls[2]))
    if key not in _LAYOUTS:
        _LAYOUTS[key] = Layout(*cls)
    return _LAYOUTS[key]


class Layout:
    '''The blocks of T4 for the labels ``ko`` and ``kv`` (0 .. ncls - 1,
    ascending) and the tables of the engine.

    sectors        the total labels M with a block
    nI[M], nA[M]   occupied and virtual quadruples of label M
    boff[M], size  where the block [nI[M], nA[M]] starts in the flat array
    quads[M]       the virtual quadruples of label M, [nA[M], 4] (int16)
    O4cls[M]       the occupied ones as positions among the sorted quadruples
    '''

    def __init__(self, ko, kv, ncls):
        self.ko, self.kv = np.asarray(ko, dtype=np.int64), np.asarray(kv, dtype=np.int64)
        self.no, self.nv, self.ncls = no, nv, ncls = len(ko), len(kv), int(ncls)
        assert np.all(np.diff(self.ko) >= 0) and np.all(np.diff(self.kv) >= 0)
        self.n2, self.n3, self.n4 = 2 * ncls - 1, 3 * ncls - 2, 4 * ncls - 3
        self.o0 = np.searchsorted(self.ko, np.arange(ncls + 1))
        self.v0 = np.searchsorted(self.kv, np.arange(ncls + 1))
        self.nO, self.nV = np.diff(self.o0), np.diff(self.v0)
        self.mix0 = self.o0[:-1] + self.v0[:-1]     # the classes of occupied and virtual spinors mixed
        self.o2, self.o3, self.o4 = (P.tuples(no, k) for k in (2, 3, 4))
        self.kO2, self.kO3, self.kO4 = (self.ko[t].sum(axis=1) for t in (self.o2, self.o3, self.o4))
        self.O2cls, self.posO2, self.o2perm, self.o2off = _by_class(self.kO2, self.n2)
        self.O3cls, self.posO3, _, _ = _by_class(self.kO3, self.n3)
        self.O4cls, self.posO4, _, _ = _by_class(self.kO4, self.n4)
        self.nO2 = np.diff(self.o2off)
        self.nJ = np.array([len(x) for x in self.O3cls])
        self.nI = np.array([len(x) for x in self.O4cls])
        v2 = P.tuples(nv, 2)
        self.kQ = self.kv[v2].sum(axis=1)
        self.Q2cls, posQ, self.qperm, self.qoff = _by_class(self.kQ, self.n2)
        self.nQ = np.diff(self.qoff)
        self.va, self.vb = (np.ascontiguousarray(v2[self.qperm, x]) for x in (0, 1))    # pairs by class
        self.posQ = np.zeros((nv, nv), dtype=np.int32)
        self.posQ[v2[:, 0], v2[:, 1]] = posQ
        v3 = P.tuples(nv, 3)
        self.A3cls, pos3, _, _ = _by_class(self.kv[v3].sum(axis=1), self.n3)
        self.pos3 = np.ascontiguousarray(pos3, dtype=np.int32)
        self.tri = [np.ascontiguousarray(v3[idx], dtype=np.int16) for idx in self.A3cls]
        self.nK = np.array([len(x) for x in self.A3cls])
        self.kv32 = np.ascontiguousarray(self.kv, dtype=np.int32)
        self.posV = np.ascontiguousarray(np.arange(nv) - self.v0[self.kv], dtype=np.int32)
        self.quads, self.nA = [], np.zeros(self.n4, dtype=np.int64)
        for M in range(self.n4):
            n = _lib.zc_t4s_quads(_INT(nv), _p(self.kv32), _INT(M), None) if self.nI[M] else 0
            q = np.empty((n, 4), dtype=np.int16)
            if n:
                _lib.zc_t4s_quads(_INT(nv), _p(self.kv32), _INT(M), _p(q))
            self.quads.append(q)
            self.nA[M] = n
        self.sectors = [M for M in range(self.n4) if self.nI[M] and self.nA[M]]
        self.boff = np.concatenate(([0], np.cumsum(self.nI * self.nA)))
        self.size = int(self.boff[-1])
        self.t3off = np.concatenate(([0], np.cumsum(self.nJ * self.nK)))
        self.o12, self.o13, self.o22 = P.split(no, 1, 2), P.split(no, 1, 3), P.split(no, 2, 2)
        self._cache = {}
        self._offsets()

    # ---------------------------------------------------------------- storage
    def zeros(self, dtype):
        return np.zeros(self.size, dtype=dtype)

    def block(self, T4, M):
        return T4[self.boff[M]:self.boff[M + 1]].reshape(self.nI[M], self.nA[M])

    def lex(self, M):
        '''Positions of the virtual quadruples of label M among all sorted
        quadruples.'''
        n = self.nv
        y = n - 1 - self.quads[M].astype(np.int64)
        return (comb(n, 4) - 1 - y[:, 0] * (y[:, 0] - 1) * (y[:, 0] - 2) * (y[:, 0] - 3) // 24
                - y[:, 1] * (y[:, 1] - 1) * (y[:, 1] - 2) // 6 - y[:, 2] * (y[:, 2] - 1) // 2 - y[:, 3])

    def to_packed(self, T4):
        '''The packed array [C(no,4), C(nv,4)] of the blocks (zeros elsewhere).'''
        out = np.zeros((comb(self.no, 4), comb(self.nv, 4)), dtype=T4.dtype)
        for M in self.sectors:
            out[np.ix_(self.O4cls[M], self.lex(M))] = self.block(T4, M)
        return out

    def from_packed(self, T4p):
        '''The blocks of a packed array (what is outside them is dropped).'''
        T4p = np.asarray(T4p).reshape(comb(self.no, 4), comb(self.nv, 4))
        out = np.zeros(self.size, dtype=T4p.dtype)
        for M in self.sectors:
            self.block(out, M)[...] = T4p[np.ix_(self.O4cls[M], self.lex(M))]
        return out

    def t3_blocks(self, T3):
        '''The blocks [nJ[k], nK[k]] of T3 in the storage of the CCSDT step,
        as views.'''
        T3 = np.asarray(T3).ravel()
        assert T3.size == self.t3off[-1]
        return [T3[self.t3off[k]:self.t3off[k + 1]].reshape(self.nJ[k], self.nK[k]) for k in range(self.n3)]

    def t3_from_packed(self, T3p):
        '''The blocks of the packed T3 [C(no,3), C(nv,3)], flat.'''
        T3p = np.asarray(T3p).reshape(comb(self.no, 3), comb(self.nv, 3))
        return np.concatenate([T3p[np.ix_(self.O3cls[k], self.A3cls[k])].ravel() for k in range(self.n3)])

    def t3_to_packed(self, T3):
        out = np.zeros((comb(self.no, 3), comb(self.nv, 3)), dtype=np.asarray(T3).dtype)
        for k, b in enumerate(self.t3_blocks(T3)):
            out[np.ix_(self.O3cls[k], self.A3cls[k])] = b
        return out

    # ---------------------------------------------------------------- offsets
    def _offsets(self):
        ncls, n2, n3, nO, nV, nQ, nK, nJ = self.ncls, self.n2, self.n3, self.nO, self.nV, self.nQ, self.nK, self.nJ
        no, nP = self.no, len(self.o2)
        # T3 on one virtual index and a pair: (k, p) -> [V_p, J of label k, pairs of label k - p]
        self.t12off = np.full((n3, ncls), -1, dtype=np.int64)
        n = 0
        for k in range(n3):
            for p in range(ncls):
                if nJ[k] and nK[k] and nV[p] and 0 <= k - p < n2 and nQ[k - p]:
                    self.t12off[k, p] = n
                    n += nV[p] * nJ[k] * nQ[k - p]
        self.t12size = n
        # Z(n f; ij; ab): ((ij), n, label of f) -> [f, (ab)]
        self.zoff = np.full((nP, no, ncls), -1, dtype=np.int64)
        n = 0
        for Pq in range(nP):
            for x in range(no):
                for ph in range(ncls):
                    q = self.kO2[Pq] + ph - self.ko[x]
                    if nV[ph] and 0 <= q < n2 and nQ[q]:
                        self.zoff[Pq, x, ph] = n
                        n += nV[ph] * nQ[q]
        self.zsize = n
        # t(i mn; K): (i, label of (mn)) -> [(mn), K]
        self.t3poff = np.full((no, n2), -1, dtype=np.int64)
        n = 0
        for i in range(no):
            for c in range(n2):
                if self.nO2[c] and self.ko[i] + c < n3 and nK[self.ko[i] + c]:
                    self.t3poff[i, c] = n
                    n += self.nO2[c] * nK[self.ko[i] + c]
        self.t3psize = n
        # t(m kl; K) above Xv(e; kl; K): ((kl), label of m and e) -> [m | e, K]
        self.r13off = np.full((nP, ncls), -1, dtype=np.int64)
        n = 0
        for Pq in range(nP):
            for mu in range(ncls):
                k = self.kO2[Pq] + mu
                if nO[mu] + nV[mu] and k < n3 and nK[k]:
                    self.r13off[Pq, mu] = n
                    n += (nO[mu] + nV[mu]) * nK[k]
        self.r13size = n
        # the blocks of one row of a sector: X[(ab), (cd)] by the label of (ab), X[a, (bcd)] by that of a
        self.off22 = np.full((self.n4, n2), -1, dtype=np.int64)
        self.off13 = np.full((self.n4, ncls), -1, dtype=np.int64)
        self.n22, self.n13 = np.zeros(self.n4, dtype=np.int64), np.zeros(self.n4, dtype=np.int64)
        for M in self.sectors:
            n = 0
            for q in range(n2):
                if nQ[q] and 0 <= M - q < n2 and nQ[M - q]:
                    self.off22[M, q] = n
                    n += nQ[q] * nQ[M - q]
            self.n22[M] = n
            n = 0
            for p in range(ncls):
                if nV[p] and 0 <= M - p < n3 and nK[M - p]:
                    self.off13[M, p] = n
                    n += nV[p] * nK[M - p]
            self.n13[M] = n
        # <d m||e f> as [(e d), f]: (m, label of e, of d); Wr(m, d, e, k) as [(e d), m]: (k, .., ..)
        self.vooff = np.full((no, ncls, ncls), -1, dtype=np.int64)
        self.wroff = np.full((no, ncls, ncls), -1, dtype=np.int64)
        n = nw = 0
        for m in range(no):
            for mu in range(ncls):
                for ka in range(ncls):
                    ph = ka + self.ko[m] - mu
                    c = mu + self.ko[m] - ka
                    if not (nV[mu] and nV[ka]):
                        continue
                    if 0 <= ph < ncls and nV[ph]:
                        self.vooff[m, mu, ka] = n
                        n += nV[mu] * nV[ka] * nV[ph]
                    if 0 <= c < ncls and nO[c]:
                        self.wroff[m, mu, ka] = nw
                        nw += nV[mu] * nV[ka] * nO[c]
        self.vosize, self.wrsize = n, nw
        # <bc||ef> as [(bc) e, f]: (label of (bc), of e)
        self.wboff = np.full((n2, ncls), -1, dtype=np.int64)
        n = 0
        for be in range(n2):
            for mu in range(ncls):
                if nQ[be] and nV[mu] and 0 <= be - mu < ncls and nV[be - mu]:
                    self.wboff[be, mu] = n
                    n += nQ[be] * nV[mu] * nV[be - mu]
        self.wbsize = n
        # the two sets of products that Xv of one occupied pair is reduced from: per label
        # of the pair, (label of e, of d) -> [e, d, (bc)] and [(bc), e, d]
        self.mxoff = np.full((n2, ncls, ncls), -1, dtype=np.int64)
        self.mxsize = np.zeros(n2, dtype=np.int64)
        for kP in range(n2):
            n = 0
            for mu in range(ncls):
                for ka in range(ncls):
                    be = kP + mu - ka
                    if (nV[mu] and nV[ka] and 0 <= be < n2 and nQ[be] and kP + mu < n3 and nK[kP + mu]
                            and self.nO2[kP]):
                        self.mxoff[kP, mu, ka] = n
                        n += nV[mu] * nV[ka] * nQ[be]
            self.mxsize[kP] = n

    def cached(self, key, make):
        if key not in self._cache:
            self._cache[key] = make()
        return self._cache[key]

    # ---------------------------------------------------------------- entries
    def ent_rows(self, M, r0, r1, kind):
        '''The entries of the rows r0:r1 of sector M themselves: the blocks of
        ``kind`` '22' / '13' row after row, or 'pp' - [(ab), (row, (cd))].'''
        nb = r1 - r0
        row, sgn = np.arange(r0, r1, dtype=np.int64), np.ones(nb)
        if kind == '13':
            ok = self.off13[M] >= 0
            nc = np.where(ok, self.nK[np.clip(M - np.arange(self.ncls), 0, self.n3 - 1)], 0)
            xoff = np.where(ok[None, :], np.arange(nb)[:, None] * self.n13[M] + self.off13[M][None, :], -1)
            s1 = np.broadcast_to(nc, xoff.shape)
        else:
            ok = self.off22[M] >= 0
            nc = np.where(ok, self.nQ[np.clip(M - np.arange(self.n2), 0, self.n2 - 1)], 0)
            if kind == '22':
                xoff = np.where(ok[None, :], np.arange(nb)[:, None] * self.n22[M] + self.off22[M][None, :], -1)
                s1 = np.broadcast_to(nc, xoff.shape)
            else:
                xoff = np.where(ok[None, :], self.off22[M][None, :] * nb + np.arange(nb)[:, None] * nc[None, :], -1)
                s1 = np.broadcast_to(nc * nb, xoff.shape)
        return (row, sgn, np.ascontiguousarray(xoff, dtype=np.int64), np.ascontiguousarray(s1, dtype=np.int64),
                np.ones(xoff.shape, dtype=np.int64))

    def ent_t12(self, k):
        '''The entries of the triples J of label k for T3 as [e, J, (cd)].'''
        def make():
            nJ, ncls = self.nJ[k], self.ncls
            ok = self.t12off[k] >= 0
            nq = np.where(ok, self.nQ[np.clip(k - np.arange(ncls), 0, self.n2 - 1)], 0)
            xoff = np.where(ok[None, :], self.t12off[k][None, :] + np.arange(nJ)[:, None] * nq[None, :], -1)
            return (np.arange(nJ, dtype=np.int64), np.ones(nJ), np.ascontiguousarray(xoff, dtype=np.int64),
                    np.ascontiguousarray(np.broadcast_to(nq * nJ, xoff.shape), dtype=np.int64),
                    np.ones(xoff.shape, dtype=np.int64))
        return self.cached(('t12', k), make)

    def ent_xv(self, kP, mu):
        '''The entries of the rows e of label ``mu`` of Xv of a pair of label
        kP: the two sets of products, [e, d, (bc)] and [(bc), e, d] behind.'''
        def make():
            ncls, nV, nQ = self.ncls, self.nV, self.nQ
            ne = nV[mu]
            xoff = np.full((2 * ne, ncls), -1, dtype=np.int64)
            s1, s2 = np.ones((2 * ne, ncls), dtype=np.int64), np.ones((2 * ne, ncls), dtype=np.int64)
            e = np.arange(ne)
            for ka in range(ncls):
                o = self.mxoff[kP, mu, ka]
                if o < 0:
                    continue
                be = kP + mu - ka
                xoff[:ne, ka], s1[:ne, ka] = o + e * nV[ka] * nQ[be], nQ[be]
                xoff[ne:, ka], s2[ne:, ka] = self.mxsize[kP] + o + e * nV[ka], ne * nV[ka]
            return (np.concatenate((e, e)).astype(np.int64), np.ones(2 * ne), xoff, s1, s2)
        return self.cached(('xv', kP, mu), make)

    # ---------------------------------------------------------------- the ring
    def ring(self, J):
        '''The blocks of the ring of the occupied triple J: for every label
        d = k_m - k_e of a pair (m outside J, e) with triples K of label
        k_J + d a dict with the rows and columns of the stacked operator, the
        places of the thin products, and the offsets of t(m J; e K) and of
        the product in their buffers; and the entries of the sectors.'''
        def make():
            no, nv, ncls, ko, o12, o13 = self.no, self.nv, self.ncls, self.ko, self.o12, self.o13
            kJ = self.kO3[J]
            rest = [m for m in range(no) if m not in self.o3[J]]
            blocks, moff, xoff = {}, 0, 0
            for d in range(-(ncls - 1), ncls):
                kK = kJ + d
                if not (0 <= kK < self.n3 and self.nK[kK]):
                    continue
                cols, base = [], {}
                for m in rest:
                    c = ko[m] - d
                    if 0 <= c < ncls and self.nV[c]:
                        base[m] = len(cols)
                        cols.extend(m * nv + np.arange(self.v0[c], self.v0[c + 1]))
                if not cols:
                    continue
                nr = len(cols)
                rows = list(cols)
                fov = -1
                if d == 0:
                    fov = len(rows)
                    rows.append(no * nv)
                thin = []
                for s in range(3):
                    n, Pq, sg = int(o12.idx1[J, s]), int(o12.idx2[J, s]), float(o12.sgn[J, s])
                    c = d + ko[n]
                    if not 0 <= c < ncls:
                        continue
                    r0 = len(rows)
                    b = no * nv + 1 + n * (no + nv)
                    rows.extend(b + np.arange(self.o0[c], self.o0[c + 1]))
                    r1 = len(rows)
                    rows.extend(b + no + np.arange(self.v0[c], self.v0[c + 1]))
                    # the rows k of the first part go to R3[(k ij), K], those of the second to Xv
                    tgt = [(int(self.posO3[o12.pos[k, Pq]]), sg * float(o12.psg[k, Pq]))
                           for k in range(self.o0[c], self.o0[c + 1])]
                    xv = self.r13off[Pq, c] + self.nO[c] * self.nK[kK] if self.nV[c] else -1
                    thin.append((r0, r1, len(rows), tgt, xv, sg))
                blocks[d] = dict(kK=kK, nK=int(self.nK[kK]), nr=nr, ntot=len(rows), fov=fov, thin=thin,
                                 rows=np.array(rows, dtype=np.int64), cols=np.array(cols, dtype=np.int64),
                                 base=base, moff=moff, xoff=xoff)
                moff += nr * self.nK[kK]
                xoff += len(rows) * self.nK[kK]
            ents = {}
            for m in rest:
                I = int(o13.pos[m, J])
                M = int(self.kO4[I])
                if M not in self.sectors:
                    continue
                xm, xx = np.full(ncls, -1, dtype=np.int64), np.full(ncls, -1, dtype=np.int64)
                s1 = np.ones(ncls, dtype=np.int64)
                for p in range(ncls):
                    b = blocks.get(int(ko[m] - p))
                    if b is not None and m in b['base']:
                        xm[p] = b['moff'] + b['base'][m] * b['nK']
                        xx[p] = b['xoff'] + b['base'][m] * b['nK']
                        s1[p] = b['nK']
                ents.setdefault(M, []).append((int(self.posO4[I]), float(o13.psg[m, J]), xm, xx, s1))
            return blocks, int(moff), int(xoff), ents
        return self.cached(('ring', J), make)

    def ring_entries(self, Js):
        '''The entries of a batch of occupied triples per sector, the blocks
        of the triples one after the other: {M: (row, sgn, xoff of the
        operand, xoff of the product, s1, s2)}, and where the operand and
        the product of each triple start.'''
        out, mb, xb = {}, [], []
        m0 = x0 = 0
        for J in Js:
            _, msz, xsz, ents = self.ring(J)
            mb.append(m0)
            xb.append(x0)
            for M, es in ents.items():
                for row, sg, xm, xx, s1 in es:
                    out.setdefault(M, []).append((row, sg, np.where(xm >= 0, xm + m0, -1),
                                                  np.where(xx >= 0, xx + x0, -1), s1))
            m0 += msz
            x0 += xsz
        tabs = {}
        for M, es in out.items():
            es.sort(key=lambda e: e[0])
            tabs[M] = (np.array([e[0] for e in es], dtype=np.int64), np.array([e[1] for e in es]),
                       np.ascontiguousarray([e[2] for e in es], dtype=np.int64),
                       np.ascontiguousarray([e[3] for e in es], dtype=np.int64),
                       np.ascontiguousarray([e[4] for e in es], dtype=np.int64),
                       np.ones((len(es), self.ncls), dtype=np.int64))
        return tabs, mb, xb, m0, x0

    # ---------------------------------------------------------------- tables of products
    def tasks_t3(self):
        '''The products of T3 with one integral: Z, G, and the T3 terms of
        Wv and Wo.'''
        def make():
            no, nv, ncls, ko, kO2, o12 = self.no, self.nv, self.ncls, self.ko, self.kO2, self.o12
            nO, nV, nQ, nJ, o0, v0, qoff = self.nO, self.nV, self.nQ, self.nJ, self.o0, self.v0, self.qoff
            nP, nQt = len(self.o2), len(self.kQ)
            z, g, wv, wo = _Tab(), _Tab(), _Tab(), _Tab()

            def t12(x, Pq, p):
                # (offset of t(x Pq; e, .) for the e of label p, leading dimension, pairs, sign)
                sg = int(o12.psg[x, Pq])
                J = int(o12.pos[x, Pq])
                kJ = self.kO3[J]
                if not sg or not 0 <= p < ncls or self.t12off[kJ, p] < 0:
                    return None
                nq = int(nQ[kJ - p])
                return int(self.t12off[kJ, p] + self.posO3[J] * nq), int(nJ[kJ] * nq), nq, sg
            for Pq in range(nP):
                for n in range(no):
                    for ph in range(ncls):
                        if self.zoff[Pq, n, ph] < 0:
                            continue
                        q = kO2[Pq] + ph - ko[n]
                        for x in range(no):
                            ep = ko[x] + ko[n] - ph
                            b = t12(x, Pq, ep)
                            if b is None:
                                continue
                            z.term(0, 'oovv', ((x * no + n) * nv + v0[ep]) * nv + v0[ph], nv, 1, 't12', b[0], b[1],
                                   'z', self.zoff[Pq, n, ph], nQ[q], nV[ph], nQ[q], nV[ep], b[3])
                        z.group()
                for l in range(no):
                    for c in range(ncls):
                        be = kO2[Pq] - c + ko[l]
                        if not (nO[c] and 0 <= be < self.n2 and nQ[be]):
                            continue
                        for x in range(no):
                            ep = c + ko[x] - ko[l]
                            b = t12(x, Pq, ep)
                            if b is None:
                                continue
                            g.term(0, 'oovoG', ((x * no + l) * no + o0[c]) * nv + v0[ep], nv, 0, 't12', b[0], b[1],
                                   'g', ((Pq * no + l) * no + o0[c]) * nQt + qoff[be], nQt, nO[c], nQ[be], nV[ep],
                                   b[3])
                        g.group()
                for p in range(ncls):
                    c = kO2[Pq] - p
                    if not (nV[p] and 0 <= c < ncls and nO[c]):
                        continue
                    for x in range(no):
                        b = t12(x, Pq, p)
                        if b is None:
                            continue
                        wo.term(0, 't12', b[0], b[1], 0, 'gB', (x * nQt + qoff[self.kO3[o12.pos[x, Pq]] - p]) * no
                                + o0[c], no, 'la13', (Pq * nv + v0[p]) * (no + nv) + self.mix0[c], no + nv,
                                nV[p], nO[c], b[2], -b[3])
                    wo.group()
            for x in range(no):
                for ep in range(ncls):
                    be = ko[x] + ep
                    if not (nV[ep] and be < self.n2 and nQ[be]):
                        continue
                    for Pq in range(nP):
                        ph = kO2[Pq] - ep
                        b = t12(x, Pq, ph)
                        if b is None:
                            continue
                        m, n = self.o2[Pq]
                        wv.term(0, 't12', b[0], b[1], 1, 'oovv', ((m * no + n) * nv + v0[ph]) * nv + v0[ep], nv,
                                'wvA', (x * nQt + qoff[be]) * nv + v0[ep], nv, nQ[be], nV[ep], nV[ph], -b[3])
                    wv.group()
            return z.done(), g.done(), wv.done(), wo.done()
        return self.cached('t3', make)

    def tasks_xv(self):
        '''The first term of Xv, for all occupied pairs, and per pair the
        products that the two other terms are reduced from.'''
        def make():
            no, nv, ncls, ko, kO2, o12 = self.no, self.nv, self.ncls, self.ko, self.kO2, self.o12
            nO, nV, nQ, nK, o0, v0, qoff = self.nO, self.nV, self.nQ, self.nK, self.o0, self.v0, self.qoff
            nP, nQt = len(self.o2), len(self.kQ)
            first, per = _Tab(), []
            for Pq in range(nP):
                k, l = (int(x) for x in self.o2[Pq])
                kP = kO2[Pq]
                tab = _Tab()
                for mu in range(ncls):
                    if self.r13off[Pq, mu] < 0 or not nV[mu]:
                        continue
                    kK = kP + mu
                    for a, b, sg in ((l, k, -1.0), (k, l, 1.0)):
                        c = mu + ko[a]
                        if c < self.n2 and self.nO2[c] and self.t3poff[b, c] >= 0:
                            first.term(0, 'oovoA', (a * nv + v0[mu]) * nP + self.o2off[c], nP, 0, 't3p',
                                       self.t3poff[b, c], nK[kK], 'r13', self.r13off[Pq, mu] + nO[mu] * nK[kK],
                                       nK[kK], nV[mu], nK[kK], self.nO2[c], sg)
                    first.group()
                    for ka in range(ncls):
                        o = self.mxoff[kP, mu, ka]
                        if o < 0:
                            continue
                        be = kP + mu - ka
                        for m in range(no):
                            sg = int(o12.psg[m, Pq])
                            ph = ka + ko[m] - mu
                            if not sg or self.vooff[m, mu, ka] < 0:
                                continue
                            J = int(o12.pos[m, Pq])
                            kJ = self.kO3[J]
                            if self.t12off[kJ, ph] < 0:
                                continue
                            tab.term(0, 'vo', self.vooff[m, mu, ka], nV[ph], 0, 't12',
                                     self.t12off[kJ, ph] + self.posO3[J] * nQ[be], self.nJ[kJ] * nQ[be],
                                     'mx', o, nQ[be], nV[mu] * nV[ka], nQ[be], nV[ph], sg)
                        for a, b, sg in ((k, l, -1.0), (l, k, 1.0)):
                            c = mu + ko[a] - ka
                            if self.wroff[a, mu, ka] >= 0:
                                tab.term(0, 'wrB', self.wroff[a, mu, ka], nO[c], 0, 't2xq',
                                         (b * no + o0[c]) * nQt + qoff[be], nQt, 'mx', o, nQ[be],
                                         nV[mu] * nV[ka], nQ[be], nO[c], sg)
                        tab.group()
                        ph = be - mu
                        if 0 <= ph < ncls and self.wboff[be, mu] >= 0:
                            tab.term(0, 'wblk', self.wboff[be, mu], nV[ph], 0, 't2px',
                                     (Pq * nv + v0[ph]) * nv + v0[ka], nv, 'mx', self.mxsize[kP] + o, nV[ka],
                                     nQ[be] * nV[mu], nV[ka], nV[ph], 0.5)
                        tab.group()
                per.append(tab.done())
            return first.done(), per
        return self.cached('xv', make)

    def tasks_rows(self, M):
        '''The products of the rows of sector M: X[(ab), (cd)] (b1 and
        T3 x T3 with a ring) and X[a, (bcd)] (b2 and T3 x T3 with a ladder).'''
        def make():
            no, nv, ncls, ko, kO2 = self.no, self.nv, self.ncls, self.ko, self.kO2
            o12, o13, o22 = self.o12, self.o13, self.o22
            nO, nV, nQ, nK, nJ, o0, v0, qoff = self.nO, self.nV, self.nQ, self.nK, self.nJ, self.o0, self.v0, self.qoff
            nP, nQt = len(self.o2), len(self.kQ)
            t22, t13 = _Tab(), _Tab()
            b1, b2, h, r = (_PIECE[x] for x in ('b1', 'b2', 't3t3h', 't3t3r'))
            for row, I in enumerate(self.O4cls[M]):
                for q in range(self.n2):
                    if self.off22[M, q] < 0:
                        continue
                    coff, nc = row * self.n22[M] + self.off22[M, q], nQ[M - q]
                    for s in range(4):
                        x, J, sg = int(o13.idx1[I, s]), int(o13.idx2[I, s]), float(o13.sgn[I, s])
                        p, kJ = q - ko[x], M - ko[x]
                        if not 0 <= p < ncls:
                            continue
                        if self.t12off[kJ, p] >= 0:
                            t22.term(b1, 'wvA', (x * nQt + qoff[q]) * nv + v0[p], nv, 0, 't12',
                                     self.t12off[kJ, p] + self.posO3[J] * nc, nJ[kJ] * nc, 'x22', coff, nc,
                                     nQ[q], nc, nV[p], -sg)
                        t22.term(b1, 't2A', (x * nQt + qoff[q]) * no + o0[p], no, 0, 'xo',
                                 (J * no + o0[p]) * nQt + qoff[M - q], nQt, 'x22', coff, nc, nQ[q], nc, nO[p], sg)
                    for s in range(3):
                        P1, P2, sg = int(o22.idx1[I, s]), int(o22.idx2[I, s]), float(o22.sgn[I, s])
                        for n in range(no):
                            ps = int(o12.psg[n, P2])
                            ph = q - kO2[P1] + ko[n]
                            if not ps or not 0 <= ph < ncls or self.zoff[P1, n, ph] < 0:
                                continue
                            J = int(o12.pos[n, P2])
                            kJ = self.kO3[J]
                            if self.t12off[kJ, ph] < 0:
                                continue
                            t22.term(r, 'z', self.zoff[P1, n, ph], nQ[q], 1, 't12',
                                     self.t12off[kJ, ph] + self.posO3[J] * nc, nJ[kJ] * nc, 'x22', coff, nc,
                                     nQ[q], nc, nV[ph], sg * ps)
                    t22.group(row)
                for p in range(ncls):
                    if self.off13[M, p] < 0:
                        continue
                    coff, nc = row * self.n13[M] + self.off13[M, p], nK[M - p]
                    for s in range(6):
                        P1, P2, sg = int(o22.idx1[I, s]), int(o22.idx2[I, s]), float(o22.sgn[I, s])
                        mu = kO2[P1] - p
                        if 0 <= mu < ncls and self.r13off[P2, mu] >= 0:
                            t13.term(b2, 'la13', (P1 * nv + v0[p]) * (no + nv) + self.mix0[mu], no + nv, 0, 'r13',
                                     self.r13off[P2, mu], nc, 'x13', coff, nc, nV[p], nc, nO[mu] + nV[mu], sg)
                    for s in range(4):
                        x, J, sg = int(o13.idx1[I, s]), int(o13.idx2[I, s]), float(o13.sgn[I, s])
                        c = M - ko[x] - p
                        if 0 <= c < self.n2 and self.t3poff[x, c] >= 0:
                            t13.term(h, 'y3A', (J * nv + v0[p]) * nP + self.o2off[c], nP, 0, 't3p',
                                     self.t3poff[x, c], nc, 'x13', coff, nc, nV[p], nc, self.nO2[c], -sg)
                    t13.group(row)
            return t22.done(), t13.done()
        return self.cached(('rows', M), make)

    def pp_rows(self, q):
        '''The rows of the thin operator of the pp ladder for the pairs (ef)
        of label q: the occupied pairs of label q (R2), then for every n the
        m and the c with k_m + k_n = k_c + k_n = q (Xo, R3):
        (rows, [(first row of m, class, first row of c)] by n).'''
        def make():
            n0 = int(self.nO2[q])
            per = []
            for n in range(self.no):
                c = q - self.ko[n]
                if 0 <= c < self.ncls:
                    per.append((n0, int(c), n0 + int(self.nO[c])))
                    n0 += int(self.nO[c] + self.nV[c])
                else:
                    per.append(None)
            return n0, per
        return self.cached(('pp', q), make)


# ------------------------------------------------------------------ kernels

def _sfx(a):
    return P._SFX[a.dtype]


def _x13(lay, mode, R, M, ent, X):
    row, sgn, xoff, s1, s2 = ent
    assert R.flags.c_contiguous and X.flags.c_contiguous and R.dtype == X.dtype
    getattr(_lib, 'zc_t4s_x13_' + _sfx(R))(
        _INT(mode), _p(R), _I64(R.shape[1]), _p(lay.quads[M]), _I64(lay.nA[M]), _I64(len(row)), _p(row), _p(sgn),
        _p(xoff), _p(s1), _p(s2), _INT(lay.ncls), _INT(lay.nv), _p(lay.kv32), _p(lay.posV), _p(lay.pos3), _p(X))


def _x22(lay, mode, R, M, ent, X):
    row, sgn, xoff, s1, s2 = ent
    assert R.flags.c_contiguous and X.flags.c_contiguous and R.dtype == X.dtype
    getattr(_lib, 'zc_t4s_x22_' + _sfx(R))(
        _INT(mode), _p(R), _I64(R.shape[1]), _p(lay.quads[M]), _I64(lay.nA[M]), _I64(len(row)), _p(row), _p(sgn),
        _p(xoff), _p(s1), _p(s2), _INT(lay.n2), _INT(lay.nv), _p(lay.kv32), _p(lay.posQ), _p(X))


def _x12(lay, mode, R, k, ent, X):
    row, sgn, xoff, s1, s2 = ent
    assert R.flags.c_contiguous and X.flags.c_contiguous and R.dtype == X.dtype
    getattr(_lib, 'zc_t4s_x12_' + _sfx(R))(
        _INT(mode), _p(R), _I64(R.shape[1]), _p(lay.tri[k]), _I64(lay.nK[k]), _I64(len(row)), _p(row), _p(sgn),
        _p(xoff), _p(s1), _p(s2), _INT(lay.ncls), _INT(lay.nv), _p(lay.kv32), _p(lay.posV), _p(lay.posQ), _p(X))


def _run(tasks, bufs, dtype, pieces=None, keys=None, cshift=0):
    '''The products of a table on the arrays ``bufs`` (by name).'''
    gs, tab, al = tasks.select(pieces, keys)
    if len(gs) < 2 or gs[-1] == gs[0]:
        return
    arr = (ctypes.c_void_p * len(_BUFS))()
    for name, a in bufs.items():
        assert a.flags.c_contiguous and a.dtype == dtype, name
        arr[_B[name]] = a.ctypes.data
    sfx = P._SFX[np.dtype(dtype)]
    fn = ctypes.cast(getattr(t3kern._lib, 'zc_gemm_' + sfx), ctypes.c_void_p)
    with timer('tasks'):
        getattr(_lib, 'zc_t4s_tasks_' + sfx)(fn, _I64(len(gs) - 1), _p(gs), _p(tab), _p(al), arr, _I64(cshift))


def jacobi(lay, T4, R4, eo, ev):
    '''R4 <- T4 + R4 / D4 in the blocks (no array of the denominators).'''
    ev = np.ascontiguousarray(ev, dtype=np.float64)
    for M in lay.sectors:
        eo4 = np.ascontiguousarray(eo[lay.o4[lay.O4cls[M]]].sum(axis=1), dtype=np.float64)
        R, T = lay.block(R4, M), lay.block(T4, M)
        getattr(_lib, 'zc_t4s_jacobi_' + _sfx(R))(_p(R), _p(T), _I64(lay.nI[M]), _p(lay.quads[M]),
                                                    _I64(lay.nA[M]), _p(eo4), _p(ev))
    return R4


def _batches(n, step):
    step = int(max(1, step))
    return [(x, min(n, x + step)) for x in range(0, n, step)]


def _E(sub, *ops):
    return np.einsum(sub, *ops, optimize=True)


def _pair_one_body(F, a, b):
    '''F on the pairs (a[x] < b[x]): [x, y] = F[a,a'] d(b,b') - F[a,b'] d(b,a')
    - F[b,a'] d(a,b') + F[b,b'] d(a,a').'''
    A, B, a_, b_ = a[:, None], b[:, None], a[None, :], b[None, :]
    return (F[A, a_] * (B == b_) - F[A, b_] * (B == a_) - F[B, a_] * (A == b_) + F[B, b_] * (A == a_))


# ------------------------------------------------------------------ the residual

def residual(lay, t2, T3, T4, ints, batch_bytes=None, only=None, out=None):
    '''(R4, the T4 terms of R3, the T4 term of R2) as
    zccsdt.cc.t4engine.residual, in the blocks of ``lay`` (Layout): T4 and
    R4 flat arrays of its blocks, T3 and R3 in the blocks of the CCSDT step
    (flat), R2 packed [C(no,2), C(nv,2)]; t2[i,j,a,b] and the dressed
    integrals ``ints`` (zccsdt.cc.t4engine.Dressed) are full arrays.
    ``only``: the pieces to evaluate (tests), of PIECES.  ``out``: the array
    of R4 (default: a new one).'''
    def want(*pieces):
        return only is None or any(p in only for p in pieces)
    batch_bytes = batch_bytes or BATCH_BYTES
    no, nv, ncls, ko = lay.no, lay.nv, lay.ncls, lay.ko
    assert (ints.nocc, ints.nvir) == (no, nv)
    dtype = np.result_type(T4, T3, t2, ints.oovv, ints.f)
    isz = np.dtype(dtype).itemsize
    nP, nQt, nI3 = len(lay.o2), len(lay.kQ), len(lay.o3)
    nO, nV, nQ, nK, nJ, o0, v0, qoff = lay.nO, lay.nV, lay.nQ, lay.nK, lay.nJ, lay.o0, lay.v0, lay.qoff
    o12, o13, o22 = lay.o12, lay.o13, lay.o22
    i2, j2 = lay.o2[:, 0], lay.o2[:, 1]
    aq, bq = lay.va, lay.vb                     # the virtual pairs in the order of their labels

    def c_(x):
        return np.ascontiguousarray(x, dtype=dtype)

    def qs(q):
        return slice(qoff[q], qoff[q + 1])
    T4 = np.ascontiguousarray(np.asarray(T4).ravel(), dtype=dtype)
    T3b = [c_(b) for b in lay.t3_blocks(T3)]
    if out is None:
        R4 = np.zeros(lay.size, dtype=dtype)
    else:
        assert out.shape == (lay.size,) and out.dtype == dtype and out.flags.c_contiguous
        R4 = P.zero(out)
    R3 = np.zeros(lay.t3off[-1], dtype=dtype)
    R3b = lay.t3_blocks(R3)
    R2 = np.zeros((nP, nQt), dtype=dtype)       # (the pairs by class; put in their order at the end)
    bufs = {}

    f = np.asarray(ints.f)
    foo, fov, fvv = f[:no, :no], c_(f[:no, no:]), f[no:, no:]
    oovv = c_(ints.oovv)
    oovv_pq = c_(oovv[i2, j2][:, aq, bq])                               # [(mn), (ef)]
    oovv_xq = c_(oovv[:, :, aq, bq])                                    # [m, n, (ef)]
    oovo = np.asarray(ints.oovo)
    oovo_pp = c_(oovo[i2, j2])                                          # [(mn), e, l]
    vovv = np.asarray(ints.vovv)
    vovv_q = c_(vovv[:, :, aq, bq])                                     # [d, m, (ef)]
    t2_pq = c_(t2[i2, j2][:, aq, bq])                                   # [(ij), (ab)]
    t2_xq = c_(t2[:, :, aq, bq])                                        # [i, m, (ab)]
    t2_px = c_(t2[i2, j2])                                              # [(ij), a, e]
    bufs.update(oovv=oovv, t2xq=t2_xq, t2px=t2_px)

    # ------------------------------------------------ intermediates of T2
    with timer('t4.imds'):
        Fv = fvv - 0.5 * _E('mnef,mndf->de', oovv, t2)
        Fo = foo + 0.5 * _E('mnef,lnef->ml', oovv, t2)
        Whh = np.asarray(ints.oooo) + 0.5 * _E('mnef,klef->mnkl', oovv, t2)
        Whh_p = c_(Whh[:, :, i2, j2])                                   # [m, n, (jk)]
        Wr = c_(np.asarray(ints.ovvo) + _E('mnef,lndf->mdel', oovv, t2))        # [m, d, e, l]
        Wh = c_(Whh[i2, j2][:, i2, j2] - t4engine._one_body_pairs(Fo, no) / 3)  # [(mn), (ij)]
        # Wv[(cd), e, l] without its T3 term, as [l, (cd), e]
        Wv = c_(np.asarray(ints.vvvo_p)[lay.qperm])
        gemm(t2_pq.T, oovo_pp.reshape(nP, nv * no), Wv.reshape(nQt, nv * no), beta=1.0)
        x = _E('dmef,lmce->cdfl', vovv, t2)
        Wv += x[aq, bq]
        Wv -= x[bq, aq]
        del x
        Wv += _E('me,lmQ->Qel', fov, t2_xq)
        bufs['wvA'] = c_(Wv.transpose(2, 0, 1))
        del Wv
        # Wo[m, d, (kl)] without its T3 term, with t2[(ij), a, e] beside it: [(ij), a, m | e], the
        # m and the e of one label next to each other
        Wo = np.array(np.asarray(ints.ovoo)[:, :, i2, j2], dtype=dtype, order='C', copy=True)
        x = _E('nmek,lnde->mdkl', oovo, t2)
        Wo += (x - x.transpose(0, 1, 3, 2))[:, :, i2, j2]
        Wo -= _E('dmQ,PQ->mdP', vovv_q, t2_pq)
        Wo -= _E('me,Pde->mdP', fov, t2_px)
        la13 = np.empty((nP, nv, no + nv), dtype=dtype)
        for c in range(ncls):
            m0 = lay.mix0[c]
            la13[:, :, m0:m0 + nO[c]] = Wo[o0[c]:o0[c + 1]].transpose(2, 1, 0)
            la13[:, :, m0 + nO[c]:m0 + nO[c] + nV[c]] = t2_px[:, :, v0[c]:v0[c + 1]]
        bufs['la13'] = la13
        del x, Wo

    # ------------------------------------------------ T3 with one integral
    with timer('t4.t3pass'):
        # t(J; e, cd): [e of label p, J of label k, (cd) of label k - p]
        T12 = np.zeros(max(lay.t12size, 1), dtype=dtype)
        for k in range(lay.n3):
            if nJ[k] and nK[k]:
                _x12(lay, 0, T3b[k], k, lay.ent_t12(k), T12)
        bufs['t12'] = T12
        tz, tg, twv, two = lay.tasks_t3()
        bufs['oovoG'] = c_(oovo.transpose(1, 3, 0, 2))                  # [x, l, m, e]
        bufs['gB'] = c_(oovv_xq.transpose(1, 2, 0))                     # [x, (ef), m]
        G = np.zeros((nP, no, no, nQt), dtype=dtype)                    # [(jk), l, m, (cd)]
        bufs['g'] = G
        _run(tg, bufs, dtype)
        _run(twv, bufs, dtype)          # Wv += 1/2 <mn||ef> t(mnl, fcd)
        _run(two, bufs, dtype)          # Wo -= 1/2 <mn||ef> t(nkl, efd)
        del bufs['oovoG'], bufs['gB']
        if want('t3t3r'):
            Z = np.zeros(max(lay.zsize, 1), dtype=dtype)                # Z(nf;ij;ab) as [f, (ab)]
            bufs['z'] = Z
            _run(tz, bufs, dtype)
        # Y(mn;jkl;d) = sum_{e<f} <mn||ef> t(jkl,efd) as [(jkl), d, (mn) by class];
        # Hs[d, m, (jkl), c] = sum_{e<f} <dm||ef> t(jkl; ef, c)
        Y3 = np.zeros((nI3, nv, nP), dtype=dtype)
        Hs = np.zeros((nv, no, nI3, nv), dtype=dtype)
        mm_, dd_ = (x.ravel() for x in np.meshgrid(np.arange(no), np.arange(nv), indexing='ij'))
        for k in range(lay.n3):
            for p in range(ncls):
                o = lay.t12off[k, p]
                if o < 0:
                    continue
                r = k - p
                Tb = T12[o:o + nV[p] * nJ[k] * nQ[r]].reshape(nV[p] * nJ[k], nQ[r])
                if lay.nO2[r]:
                    y = gemm(Tb, c_(oovv_pq[lay.O2cls[r]][:, qs(r)].T))
                    Y3[lay.O3cls[k], v0[p]:v0[p + 1], lay.o2off[r]:lay.o2off[r + 1]] = \
                        y.reshape(nV[p], nJ[k], -1).transpose(1, 0, 2)
                sel = np.where(lay.kv[dd_] + ko[mm_] == r)[0]
                if len(sel):
                    h = gemm(Tb, c_(vovv_q[dd_[sel], mm_[sel]][:, qs(r)].T)).reshape(nV[p], nJ[k], len(sel))
                    Hs[dd_[sel][None, None, :], mm_[sel][None, None, :], lay.O3cls[k][None, :, None],
                       np.arange(v0[p], v0[p + 1])[:, None, None]] = h
        bufs['y3A'] = Y3
        # the part of Xo without T4, as [(jkl), m, (cd)]:
        # P(l/jk) [ <mn||el> t(njk,ecd) + 1/2 Whh(mn,jk) t(nl,cd) ] - <dm||ef> t(jkl,efc) + (c <-> d)
        Xo = np.empty((nI3, no, nQt), dtype=dtype)
        t2_n = t2_xq.reshape(no, no * nQt)
        for m in range(no):
            Gm = c_(G[:, :, m].transpose(1, 0, 2))                      # [l, (jk), (cd)]
            Gm += 0.5 * gemm(c_(Whh_p[m].T), t2_n).reshape(nP, no, nQt).transpose(1, 0, 2)
            Xo[:, m] = P.fold(Gm, o12, 0, 1)
            Xo[:, m] -= (Hs[bq, m, :, aq] - Hs[aq, m, :, bq]).T
        bufs['xo'] = Xo
        del G, Gm, Hs, bufs['g']
        # t(i mn; K) as [(mn) of one label, K], and t(m kl; K) for the m of one label above Xv
        T3P = np.zeros(max(lay.t3psize, 1), dtype=dtype)
        for i in range(no):
            for c in range(lay.n2):
                o = lay.t3poff[i, c]
                if o >= 0 and nJ[ko[i] + c]:
                    k, mn = ko[i] + c, lay.O2cls[c]
                    T3P[o:o + len(mn) * nK[k]].reshape(len(mn), nK[k])[...] = \
                        T3b[k][lay.posO3[o12.pos[i, mn]]] * o12.psg[i, mn][:, None]
        bufs['t3p'] = T3P
        R13 = np.zeros(max(lay.r13size, 1), dtype=dtype)
        for Pq in range(nP):
            for mu in range(ncls):
                o = lay.r13off[Pq, mu]
                if o >= 0 and nO[mu] and nJ[lay.kO2[Pq] + mu]:
                    k, ms = lay.kO2[Pq] + mu, np.arange(o0[mu], o0[mu + 1])
                    R13[o:o + nO[mu] * nK[k]].reshape(nO[mu], nK[k])[...] = \
                        T3b[k][lay.posO3[o12.pos[ms, Pq]]] * o12.psg[ms, Pq][:, None]
        bufs['r13'] = R13

    # ------------------------------------------------ the ladder operator in blocks
    with timer('t4.imds'):
        vpp = np.asarray(ints.vvvv_pp)
        Wpp = []
        for q in range(lay.n2):
            idx = lay.Q2cls[q]
            W = np.array(vpp[np.ix_(idx, idx)], dtype=dtype, order='C')
            if lay.nO2[q] and nQ[q]:
                oc = lay.O2cls[q]
                gemm(c_(t2_pq[oc][:, qs(q)].T), c_(oovv_pq[oc][:, qs(q)]), W, beta=1.0)
            Wpp.append(W)

    # ---- the part of Xv without T4, for all pairs: [e, (bcd)] below the rows of T3 in R13
    if want('b2'):
        with timer('t4.xv'):
            first, per = lay.tasks_xv()
            bufs['oovoA'] = c_(oovo_pp.transpose(2, 1, 0)[:, :, lay.o2perm])    # [l, e, (mn) by class]
            _run(first, bufs, dtype)    # - P(kl) sum_{m<n} <mn||el> t(mnk, bcd)
            del bufs['oovoA']
            VO = np.empty(max(lay.vosize, 1), dtype=dtype)
            WrB = np.empty(max(lay.wrsize, 1), dtype=dtype)
            for m in range(no):
                for mu in range(ncls):
                    for ka in range(ncls):
                        es, ds = slice(v0[mu], v0[mu + 1]), slice(v0[ka], v0[ka + 1])
                        o = lay.vooff[m, mu, ka]
                        if o >= 0:
                            ph = ka + ko[m] - mu
                            blk = vovv[ds, m, es, v0[ph]:v0[ph + 1]].transpose(1, 0, 2)
                            VO[o:o + blk.size] = blk.ravel()
                        o = lay.wroff[m, mu, ka]
                        if o >= 0:
                            c = mu + ko[m] - ka
                            blk = Wr[o0[c]:o0[c + 1], ds, es, m].transpose(2, 1, 0)
                            WrB[o:o + blk.size] = blk.ravel()
            Wb = np.empty(max(lay.wbsize, 1), dtype=dtype)
            for be in range(lay.n2):
                for mu in range(ncls):
                    o = lay.wboff[be, mu]
                    if o < 0:
                        continue
                    e = np.arange(v0[mu], v0[mu + 1])[:, None]
                    f_ = np.arange(v0[be - mu], v0[be - mu + 1])[None, :]
                    col = lay.posQ[np.minimum(e, f_), np.maximum(e, f_)]
                    blk = Wpp[be][:, np.where(e == f_, 0, col)] * np.sign(f_ - e)
                    Wb[o:o + blk.size] = blk.ravel()
            bufs.update(vo=VO, wrB=WrB, wblk=Wb)
            Mx = np.empty(max(2 * int(lay.mxsize.max()), 1), dtype=dtype)
            bufs['mx'] = Mx
            for Pq in range(nP):
                kP = lay.kO2[Pq]
                if not lay.mxsize[kP]:
                    continue
                # P(d/bc) [ <dm||ef> t(m kl; f, bc) + P(kl) sum_m Wr(m,d,e,k) t(ml,bc) ] and
                # P(d/bc) 1/2 sum_f Wpp(bc,ef) t(kl,fd)
                P.zero(Mx[:2 * lay.mxsize[kP]])
                _run(per[Pq], bufs, dtype)
                for mu in range(ncls):
                    o = lay.r13off[Pq, mu]
                    if o < 0 or not nV[mu] or not np.any(lay.mxoff[kP, mu] >= 0):
                        continue
                    k = kP + mu
                    o += nO[mu] * nK[k]
                    _x12(lay, 1, R13[o:o + nV[mu] * nK[k]].reshape(nV[mu], nK[k]), k, lay.ent_xv(kP, mu), Mx)
            del bufs['vo'], bufs['wrB'], bufs['wblk'], bufs['mx'], VO, WrB, Wb, Mx
    # Wq = Wpp + F_vv on pairs / 3
    for q in range(lay.n2):
        if nQ[q]:
            Wpp[q] += _pair_one_body(Fv, aq[qs(q)], bq[qs(q)]) / 3
    Wq = Wpp

    # ---- pp ladder with F_vv, the rows of a sector in batches; from the same view of T4
    # (virtual indices in two pairs) the T4 part of Xo and the T4 terms of R2 and R3
    if want('pp', 'r2', 'b1'):
        with timer('t4.pp'):
            U3 = np.zeros(max(lay.t12size, 1), dtype=dtype) if want('b1') else None     # [c, J, (ab)]
            thin = want('r2', 'b1')
            Ls = {}
            if thin:
                for q in range(lay.n2):
                    nrow, per = lay.pp_rows(q)
                    if not nQ[q] or not nrow:
                        continue
                    L = np.zeros((nrow, nQ[q]), dtype=dtype)
                    L[:lay.nO2[q]] = oovv_pq[lay.O2cls[q]][:, qs(q)]
                    for n in range(no):
                        if per[n] is not None:
                            r0, c, r1 = per[n]
                            L[r0:r1] = oovv_xq[o0[c]:o0[c + 1], n, qs(q)]       # <mn||ef>
                            L[r1:r1 + nV[c]] = -vovv_q[v0[c]:v0[c + 1], n, qs(q)]   # - <cn||ef>
                    Ls[q] = L
            for M in lay.sectors:
                T, R = lay.block(T4, M), lay.block(R4, M)
                n22 = int(lay.n22[M])
                if not n22:
                    continue
                Is = lay.O4cls[M]
                for r0, r1 in _batches(lay.nI[M], batch_bytes // (isz * n22)):
                    nb = r1 - r0
                    ent = lay.ent_rows(M, r0, r1, 'pp')
                    Tv = P.zero(np.empty(nb * n22, dtype=dtype))
                    with timer('expand'):
                        _x22(lay, 0, T, M, ent, Tv)
                    X = np.empty(nb * n22, dtype=dtype) if want('pp') else None
                    for q in range(lay.n2):
                        o = lay.off22[M, q]
                        if o < 0:
                            continue
                        nc = nQ[M - q]
                        Tq = Tv[o * nb:(o + nQ[q] * nc) * nb].reshape(nQ[q], nb * nc)
                        if want('pp'):
                            with timer('gemm'):
                                gemm(Wq[q], Tq, X[o * nb:(o + nQ[q] * nc) * nb].reshape(nQ[q], nb * nc))
                        if q not in Ls:
                            continue
                        with timer('gemm'):
                            O = gemm(Ls[q], Tq).reshape(-1, nb, nc)
                        per = lay.pp_rows(q)[1]
                        cs = qs(M - q)
                        for r in range(r0, r1):
                            I = Is[r]
                            if want('r2'):
                                for s in range(6):
                                    P1 = o22.idx1[I, s]
                                    if lay.kO2[P1] == q:
                                        R2[o22.idx2[I, s], cs] += o22.sgn[I, s] * O[lay.posO2[P1], r - r0]
                            if want('b1'):
                                # Xo(m;jkl;cd) += <mn||ef> t(n jkl; ef, cd);
                                # R3 -= P(c/ab) <cm||ef> t(m ijk; ef, ab)
                                for s in range(4):
                                    n, J, sg = o13.idx1[I, s], o13.idx2[I, s], o13.sgn[I, s]
                                    if per[n] is None:
                                        continue
                                    a0, c, a1 = per[n]
                                    if nO[c]:
                                        Xo[J, o0[c]:o0[c + 1], cs] += sg * O[a0:a1, r - r0]
                                    ou = lay.t12off[lay.kO3[J], c]
                                    if ou >= 0:
                                        U3[ou:ou + nV[c] * nJ[lay.kO3[J]] * nc].reshape(nV[c], -1, nc)[
                                            :, lay.posO3[J]] += sg * O[a1:a1 + nV[c], r - r0]
                    if want('pp'):
                        with timer('reduce'):
                            _x22(lay, 1, R, M, ent, X)
                    del Tv, X, Tq
            if want('b1'):
                for k in range(lay.n3):
                    if nJ[k] and nK[k]:
                        _x12(lay, 1, R3b[k], k, lay.ent_t12(k), U3)
            del U3, Ls
    del Wq, Wpp

    # ---- ring, for every occupied triple J = (jkl) and label of (m, e) the block
    # t(m J; e K), m outside J; from the same view of T4 the T4 part of Xv and the T4
    # terms of R3 with f_ov and <mn||ek>
    if want('ring', 'b2') and lay.sectors:
        with timer('t4.ring'):
            S = np.empty((no * nv + 1 + no * (no + nv), no * nv), dtype=dtype)
            S[:no * nv] = Wr.transpose(3, 1, 0, 2).reshape(no * nv, no * nv)    # [(i a), (m e)]
            S[no * nv] = fov.ravel()
            # per n: - 1/2 <mn||ek> as [k, (m e)] above 1/2 <mn||ef> as [e, (m f)]
            S[no * nv + 1:].reshape(no, no + nv, no * nv)[:, :no] = \
                -0.5 * oovo.transpose(1, 3, 0, 2).reshape(no, no, no * nv)
            S[no * nv + 1:].reshape(no, no + nv, no * nv)[:, no:] = \
                0.5 * oovv.transpose(1, 2, 0, 3).reshape(no, nv, no * nv)
            sizes = [lay.ring(J)[1:3] for J in range(nI3)]
            Js, batches, n = [], [], 0
            for J in range(nI3):
                if Js and (n + sum(sizes[J])) * isz > batch_bytes:
                    batches.append(Js)
                    Js, n = [], 0
                Js.append(J)
                n += sum(sizes[J])
            batches.append(Js)
            for Js in batches:
                tabs, mb, xb, msz, xsz = lay.cached(('ringb', tuple(Js)), lambda: lay.ring_entries(Js))
                if not msz:
                    continue
                Mb = P.zero(np.empty(msz, dtype=dtype))
                Xb = np.empty(xsz, dtype=dtype)
                with timer('expand'):
                    for M, (row, sgn, xm, xx, s1, s2) in tabs.items():
                        _x13(lay, 0, lay.block(T4, M), M, (row, sgn, xm, s1, s2), Mb)
                for J, m0, x0 in zip(Js, mb, xb):
                    kJ = lay.kO3[J]
                    for d, b in lay.ring(J)[0].items():
                        nK_, nr, ntot = b['nK'], b['nr'], b['ntot']
                        rows = b['rows'] if want('b2') else b['rows'][:nr + (b['fov'] >= 0)]
                        if not want('ring'):
                            rows = rows[nr:]
                        if not len(rows):
                            continue
                        Xj = Xb[x0 + b['xoff']:x0 + b['xoff'] + ntot * nK_].reshape(ntot, nK_)
                        first = 0 if want('ring') else nr
                        with timer('gemm'):
                            gemm(c_(S[np.ix_(rows, b['cols'])]),
                                 Mb[m0 + b['moff']:m0 + b['moff'] + nr * nK_].reshape(nr, nK_),
                                 Xj[first:first + len(rows)])
                        if want('ring') and b['fov'] >= 0:
                            R3b[kJ][lay.posO3[J]] += Xj[b['fov']]
                        if want('b2'):
                            for a0, a1, a2, tgt, xv, sg in b['thin']:
                                for k, (rowJ, sgk) in enumerate(tgt):
                                    if sgk:
                                        R3b[b['kK']][rowJ] += sgk * Xj[a0 + k]
                                if xv >= 0:
                                    R13[xv:xv + (a2 - a1) * nK_].reshape(a2 - a1, nK_)[...] += sg * Xj[a1:a2]
                if want('ring'):
                    with timer('reduce'):
                        for M, (row, sgn, xm, xx, s1, s2) in tabs.items():
                            _x13(lay, 1, lay.block(R4, M), M, (row, sgn, xx, s1, s2), Xb)
                del Mb, Xb, Xj
            del S

    # ---- hh ladder with F_oo: a matrix on the occupied quadruples of a sector
    if want('hh'):
        with timer('t4.hh'):
            nI = len(lay.o4)
            H = np.zeros((nI, nI), dtype=dtype)
            for s in range(6):
                p1, p2 = o22.idx1[:, s], o22.idx2[:, s]
                np.add.at(H, (np.arange(nI)[:, None], o22.pos[:, p2].T),
                          (o22.sgn[:, s][:, None] * o22.psg[:, p2].T) * Wh[:, p1].T)
            for M in lay.sectors:
                idx = lay.O4cls[M]
                gemm(c_(H[np.ix_(idx, idx)]), lay.block(T4, M), lay.block(R4, M), beta=1.0)

    # ---- the rows of a sector in batches: the products with the six-index intermediates
    # and T3 with T3, the splits of the quadruple summed by the products
    if want('b1', 'b2', 't3t3h', 't3t3r'):
        with timer('t4.rows'):
            bufs['t2A'] = c_(t2_xq.transpose(0, 2, 1))                  # [i, (ab), m]
            p22 = [_PIECE[x] for x in ('b1', 't3t3r') if want(x)]
            p13 = [_PIECE[x] for x in ('b2', 't3t3h') if want(x)]
            for M in lay.sectors:
                t22, t13 = lay.tasks_rows(M)
                R = lay.block(R4, M)
                for kind, tasks, pieces, n, fn in (('22', t22, p22, int(lay.n22[M]), _x22),
                                                   ('13', t13, p13, int(lay.n13[M]), _x13)):
                    if not pieces or not n:
                        continue
                    name = 'x' + kind
                    for r0, r1 in _batches(lay.nI[M], batch_bytes // (isz * n)):
                        X = P.zero(np.empty((r1 - r0) * n, dtype=dtype))
                        bufs[name] = X
                        _run(tasks, bufs, dtype, pieces, (r0, r1), -r0 * n)
                        with timer('reduce'):
                            fn(lay, 1, R, M, lay.ent_rows(M, r0, r1, kind), X)
                        del bufs[name], X
    out2 = np.empty_like(R2)
    out2[:, lay.qperm] = R2
    return R4, R3, out2
