'''EOM-DEA with complete 5p3h operators on the sector itself: the sigma
vector of R = 2p + 3p1h + 4p2h + 5p3h on the CCSDT ground state (T1-dressed
integrals, T2 and T3), in the blocks of the symmetry labels (docs/eom.md,
section 11.11).

The rows of 2p, 3p1h and 4p2h are those of zccsdt.cc.dea_sector (R to
4p2h) plus the four terms of r5 in them; the 5p3h row is the T5 equation of
CCSDTQP with R in the place of one amplitude (zccsdt.cc.dea_ref derives its
61 terms).  Every term of the 5p3h row is a product of a fixed factor F and
the product B of the others (the families of section 11.11):

    S[ F B' ],  S the sum over the distinct splits of the external indices,
                B' = |O_F|! |V_F|! B,  B antisymmetric in its external groups

r5[(i<j<k), (a<b<c<d<e)] (the ghosts implied):

  pp      sum_{f<g} Wq5(ab, fg) r5(ijk; fg cde)            Wq5 = Wpp + F_vv on pairs / 4
  hh      the occupied triples: - S(i|jk) F_oo(m,i) r5(mjk) + S(k|ij) sum_{m<n} Whh(mn,ij) r5(mnk)
  ring    S(i|jk) S(a|bcde) sum_{mf} Wr(m,a,f,i) r5(mjk; f bcde)
  D1      S(i|jk) S(ab|cde)  - sum_f Wv(ab,f,i) r4(jk; f cde)
  D2      S(i|jk) S(ab|cde)  sum_m t2(im; ab) XoR5(m; jk; cde)
  D3      S(k|ij) S(de|abc)  sum_g r3(k; g de) Xv2(g; ij; abc)
  D4      S(ij|k) S(ab|cde)  sum_{mf} t3(ijm; f ab) Z4(mf; k; cde)
  D5                S(ab|cde)  sum_f t3(ijk; f ab) XvR5(f; cde)
  D6                S(ab|cde)  sum_m Xo3(m; ijk; ab) r3(m; cde)
  D7      S(i|jk) S(de|abc)  sum_{m<n} t3(imn; abc) Y4(mn; jk; de)
  D8      S(i|jk) S(de|abc)  sum_m t3(jkm; abc) XoRp(m; i; de)
  E1      S(ij|k) S(a|bcde)  sum_m Wo(m,a,ij) r4(mk; bcde) + sum_f t2(ij; af) Xv5(f; k; bcde)
  E2                S(a|bcde)  sum_{m<n} W2(mn; ijk; a) r4(mn; bcde)

with the intermediates of the ground state (once): Wpp, Whh, F, Wr, Wv, Wo
of zccsdt.cc.dea_sector.Ground; Xv2 = its six-index Xv with the term of Wpp
counted twice; Xo3(m; ijk; ab), W2(mn; ijk; a), B4(ml g; jk; c); and those
of the sector (per sigma vector, linear in R):

  XoR5(m; jk; cde)   P(jk) sum_{lh} <ml||hj> r4(lk; h cde) + S(c|de) sum_{h<g} <cm||hg> r4(jk; hg de)
                     + 1/2 P(jk) S(c|de) sum_l G(ml; j; c) t2(kl; de) - A(cde) X(c, d, e)
                     + sum_l sum_{h<g} <ml||hg> r5(ljk; hg cde) - 1/2 P(jk) S(e|cd) sum_l t2(jl; cd) YR(ml; k; e)
                     + S(c|de) sum_{lg} [ G3(ml; c, g) t3(jkl; g de) + B4(ml g; jk; c) r3(l; g de) ]
  Xv5(f; k; bcde)    S(b|cde) [ sum_{lh} <bl||fh> r4(lk; h cde) + sum_{l<n} G5(ln; f, b) t3(lnk; cde) ]
                     + sum_{l<n} sum_h <ln||fh> r5(lnk; h bcde)
  Z4(mf; k; cde)     sum_{lh} <ml||fh> r4(lk; h cde)
  XvR5(f; cde)       XvR of dea_sector with the term of Wpp counted twice
  Y4(mn; jk; de)     sum_{h<g} <mn||hg> r4(jk; hg de)
  XoRp(m; i; de)     small (m, i, a pair), seven terms
  G = <ml||hj> r2(ch),  G3 = <ml||hg> r2(ch),  G5 = <ln||hf> r2(bh),  X = <cm||hg> r2(dh) t2(jk; eg)

and in the rows below: sigma3 += sum_{m<n} sum_{e<f} <mn||ef> r5(imn; ef abc),
sigma4 += sum_{me} f_me r5(mij; e abcd) - 1/2 P(ij) sum_{mne} <mn||ei> r5(mnj; e abcd)
- S(a|bcd) sum_m sum_{e<f} <am||ef> r5(mij; ef bcd).

How the products are done (the passes):

- pp: per block of r5 the rows in batches, expanded on (fg | cde)
  (csrc/t4sym.c, zc_t5s_x23), one product with Wq5 per class of (fg),
  reduced; from the same view the thin products (XoR5 from r5, sigma3,
  sigma4 on (a | bcd)).
- ring: per spectator pair (u, w) the rows (m, u, w) of r5 on (e | bcde)
  (zc_t5s_x14), grouped by d = k_m - k_e, times the stacked operator [Wr
  rows (i a) | f_ov | <mu||ei> | <mw||ei> | <mu||fe> | <mw||fe>]: the ring,
  the terms of sigma4 and Xv5 from r5.
- hh: per block one product with the matrix on the occupied triples.
- rows: per row (i<j<k) X[(ab), (cde)] per class of (ab) and X[a, (bcde)]
  per class of a, each one product of stacked operands, reduced once.

Storage (Sector5): r5 one block per class M of the virtual quintuples (a
list, zc_t5s_quints), rows the occupied triples of class M - shift; parity
a mask applied per block.
'''

import ctypes
import time
from math import comb

import numpy as np

from zccsdt.blas import gemm
from zccsdt.cc import dea_sector as D, t4engine_sym as S, t4pack as P
from zccsdt.timing import timer

_lib = S._lib
_I64, _INT = ctypes.c_int64, ctypes.c_int
if _lib is not None and hasattr(_lib, 'zc_t5s_quints'):
    _lib.zc_t5s_quints.restype = ctypes.c_int64
BATCH_BYTES = 1 << 30


def _p(a):
    return ctypes.c_void_p(a.ctypes.data)


def _c(x, dtype):
    return np.ascontiguousarray(x, dtype=dtype)


def _batches(n, step):
    step = int(max(1, step))
    return [(x, min(n, x + step)) for x in range(0, n, step)]


# ------------------------------------------------------------------ the blocks of the sector

class Sector5:
    '''The 5p3h blocks next to the blocks of dea_sector.Sector ``sec``: r5
    one block per class M of the virtual quintuples, rows the occupied
    triples of class M - shift; the quadruples of every class (lists and
    their positions) for the splits (a | bcde).'''

    def __init__(self, sec):
        self.sec, self.lay, self.s = sec, sec.lay, sec.s
        lay, s = sec.lay, sec.s
        no, nv, ncls = lay.no, lay.nv, lay.ncls
        self.no, self.nv, self.ncls = no, nv, ncls
        self.n5 = 5 * ncls - 4
        nQ, nK, nV = lay.nQ, lay.nK, lay.nV
        # the quadruples of every class and their positions in it
        self.quads, self.nA4 = [], np.zeros(lay.n4, dtype=np.int64)
        self.pos4 = np.full(comb(nv, 4), -1, dtype=np.int32)
        for A in range(lay.n4):
            n = int(_lib.zc_t4s_quads(_INT(nv), _p(lay.kv32), _INT(A), None))
            q = np.empty((n, 4), dtype=np.int16)
            if n:
                _lib.zc_t4s_quads(_INT(nv), _p(lay.kv32), _INT(A), _p(q))
                self.pos4[_cb4(q, nv)] = np.arange(n, dtype=np.int32)
            self.quads.append(q)
            self.nA4[A] = n
        # a row of (b | cde) layout for any class A of quadruples: [b of class p, triples of class A - p]
        self.off13 = np.full((lay.n4, ncls), -1, dtype=np.int64)
        self.n13 = np.zeros(lay.n4, dtype=np.int64)
        for A in range(lay.n4):
            n = 0
            for p in range(ncls):
                if nV[p] and 0 <= A - p < lay.n3 and nK[A - p]:
                    self.off13[A, p] = n
                    n += nV[p] * nK[A - p]
            self.n13[A] = n
        # 5p3h blocks
        nI3 = len(lay.o3)
        self.rows5, self.quints, self.nA5, self.off5 = [], [], np.zeros(self.n5, dtype=np.int64), [0]
        self.blk5, self.pos5 = np.full(nI3, -1, dtype=np.int64), np.full(nI3, -1, dtype=np.int64)
        for M in range(self.n5):
            c = M - s
            r = np.asarray(lay.O3cls[c], dtype=np.int64) if 0 <= c < lay.n3 else np.zeros(0, dtype=np.int64)
            n = int(_lib.zc_t5s_quints(_INT(nv), _p(lay.kv32), _INT(M), None)) if len(r) else 0
            q = np.empty((n, 5), dtype=np.int16)
            if n:
                _lib.zc_t5s_quints(_INT(nv), _p(lay.kv32), _INT(M), _p(q))
            else:
                r = np.zeros(0, dtype=np.int64)
            self.rows5.append(r)
            self.quints.append(q)
            self.nA5[M] = n
            self.blk5[r], self.pos5[r] = M, np.arange(len(r))
            self.off5.append(self.off5[-1] + len(r) * n)
        self.size5 = int(self.off5[-1])
        self.blocks5 = [M for M in range(self.n5) if len(self.rows5[M]) and self.nA5[M]]
        # one row of a block: X[(ab), (cde)] by the class of (ab), X[a, (bcde)] by the class of a
        self.off23 = np.full((self.n5, lay.n2), -1, dtype=np.int64)
        self.off14 = np.full((self.n5, ncls), -1, dtype=np.int64)
        self.n23, self.n14 = np.zeros(self.n5, dtype=np.int64), np.zeros(self.n5, dtype=np.int64)
        for M in self.blocks5:
            n = 0
            for q in range(lay.n2):
                if nQ[q] and 0 <= M - q < lay.n3 and nK[M - q]:
                    self.off23[M, q] = n
                    n += nQ[q] * nK[M - q]
            self.n23[M] = n
            n = 0
            for p in range(ncls):
                if nV[p] and 0 <= M - p < lay.n4 and self.nA4[M - p]:
                    self.off14[M, p] = n
                    n += nV[p] * self.nA4[M - p]
            self.n14[M] = n

    def block5(self, r5, M):
        return r5[self.off5[M]:self.off5[M + 1]].reshape(len(self.rows5[M]), self.nA5[M])

    def lex5(self, M):
        return _cb5(self.quints[M], self.nv)

    # ---------------------------------------------------------------- kernels
    def x23(self, mode, R, M, ent, X):
        row, sgn, xoff, s1, s2 = ent
        lay = self.lay
        getattr(_lib, 'zc_t5s_x23_' + P._SFX[R.dtype])(
            _INT(mode), _p(R), _I64(R.shape[1]), _p(self.quints[M]), _I64(self.nA5[M]), _I64(len(row)), _p(row),
            _p(sgn), _p(xoff), _p(s1), _p(s2), _INT(lay.n2), _INT(lay.nv), _p(lay.kv32), _p(lay.posQ),
            _p(lay.pos3), _p(X))

    def x14(self, mode, R, M, ent, X):
        row, sgn, xoff, s1, s2 = ent
        lay = self.lay
        getattr(_lib, 'zc_t5s_x14_' + P._SFX[R.dtype])(
            _INT(mode), _p(R), _I64(R.shape[1]), _p(self.quints[M]), _I64(self.nA5[M]), _I64(len(row)), _p(row),
            _p(sgn), _p(xoff), _p(s1), _p(s2), _INT(lay.ncls), _INT(lay.nv), _p(lay.kv32), _p(lay.posV),
            _p(self.pos4), _p(X))

    def x13q(self, mode, R, A, ent, X):
        '''(b | cde) of the quadruples of class A (any class).'''
        row, sgn, xoff, s1, s2 = ent
        lay = self.lay
        getattr(_lib, 'zc_t4s_x13_' + P._SFX[R.dtype])(
            _INT(mode), _p(R), _I64(R.shape[1]), _p(self.quads[A]), _I64(self.nA4[A]), _I64(len(row)), _p(row),
            _p(sgn), _p(xoff), _p(s1), _p(s2), _INT(lay.ncls), _INT(lay.nv), _p(lay.kv32), _p(lay.posV),
            _p(lay.pos3), _p(X))

    def ent_row(self, M, r, kind):
        '''One row of block M: '23' X[(ab), (cde)] or '14' X[a, (bcde)].'''
        lay = self.lay
        if kind == '23':
            n2c = lay.n2
            ok = self.off23[M] >= 0
            nc = np.where(ok, lay.nK[np.clip(M - np.arange(n2c), 0, lay.n3 - 1)], 0)
            xoff = np.where(ok, self.off23[M], -1)
        else:
            ok = self.off14[M] >= 0
            nc = np.where(ok, self.nA4[np.clip(M - np.arange(self.ncls), 0, lay.n4 - 1)], 0)
            xoff = np.where(ok, self.off14[M], -1)
        return (np.array([r], dtype=np.int64), np.ones(1), np.ascontiguousarray(xoff[None, :], dtype=np.int64),
                np.ascontiguousarray(nc[None, :], dtype=np.int64), np.ones((1, len(nc)), dtype=np.int64))

    def ent_pp(self, M, r0, r1):
        '''Rows r0:r1 of block M as [(fg), (row, (cde))] per class of (fg).'''
        nb, n2c, lay = r1 - r0, self.lay.n2, self.lay
        ok = self.off23[M] >= 0
        nc = np.where(ok, lay.nK[np.clip(M - np.arange(n2c), 0, lay.n3 - 1)], 0)
        xoff = np.where(ok[None, :], self.off23[M][None, :] * nb + np.arange(nb)[:, None] * nc[None, :], -1)
        return (np.arange(r0, r1, dtype=np.int64), np.ones(nb), np.ascontiguousarray(xoff, dtype=np.int64),
                np.ascontiguousarray(np.broadcast_to(nc * nb, xoff.shape), dtype=np.int64),
                np.ones(xoff.shape, dtype=np.int64))

    def ent_x13(self, A, nrow, base=0):
        '''Rows 0..nrow of an array on (b | cde) of class A, one after the other.'''
        lay = self.lay
        ok = self.off13[A] >= 0
        nc = np.where(ok, lay.nK[np.clip(A - np.arange(self.ncls), 0, lay.n3 - 1)], 0)
        xoff = np.where(ok[None, :], base + np.arange(nrow)[:, None] * self.n13[A] + self.off13[A][None, :], -1)
        return (np.arange(nrow, dtype=np.int64), np.ones(nrow), np.ascontiguousarray(xoff, dtype=np.int64),
                np.ascontiguousarray(np.broadcast_to(nc, xoff.shape), dtype=np.int64),
                np.ones(xoff.shape, dtype=np.int64))

    def dense_positions5(self):
        '''The places of the elements of r5 in the dense array [(i<j<k), (a<..<e)].'''
        n5 = comb(self.nv, 5)
        out = [np.zeros(0, dtype=np.int64)]
        for M in self.blocks5:
            out.append((self.rows5[M][:, None] * n5 + self.lex5(M)[None, :]).ravel())
        return np.concatenate(out)


def _cb4(q, n):
    y = n - 1 - np.asarray(q, dtype=np.int64)
    return (comb(n, 4) - 1 - y[:, 0] * (y[:, 0] - 1) * (y[:, 0] - 2) * (y[:, 0] - 3) // 24
            - y[:, 1] * (y[:, 1] - 1) * (y[:, 1] - 2) // 6 - y[:, 2] * (y[:, 2] - 1) // 2 - y[:, 3])


def _cb5(q, n):
    y = n - 1 - np.asarray(q, dtype=np.int64)
    return (comb(n, 5) - 1 - y[:, 0] * (y[:, 0] - 1) * (y[:, 0] - 2) * (y[:, 0] - 3) * (y[:, 0] - 4) // 120
            - y[:, 1] * (y[:, 1] - 1) * (y[:, 1] - 2) * (y[:, 1] - 3) // 24
            - y[:, 2] * (y[:, 2] - 1) * (y[:, 2] - 2) // 6 - y[:, 3] * (y[:, 3] - 1) // 2 - y[:, 4])


# ------------------------------------------------------------------ the ground state

class Ground5:
    '''The intermediates of the ground state for the 5p3h row besides those
    of dea_sector.Ground ``g`` (module docstring): Wq5, the matrices on the
    occupied triples, Xo3, W2, B4, Xv2.  ``T3p``: T3 packed [C(o,3), C(v,3)]
    (lexicographic).'''

    def __init__(self, sec5, g, T3p, ovvo, xv2=None):
        sec, lay = sec5.sec, sec5.lay
        self.ovvo = np.ascontiguousarray(ovvo, dtype=g.dtype)          # <md||hi>, dressed
        no, nv, ncls, ko = lay.no, lay.nv, lay.ncls, lay.ko
        dtype = self.dtype = g.dtype
        nP, nI3, nQt = len(lay.o2), len(lay.o3), len(lay.kQ)
        nO, nV, nQ, nK, qoff, o0, v0 = lay.nO, lay.nV, lay.nQ, lay.nK, lay.qoff, lay.o0, lay.v0
        aq, bq = lay.va, lay.vb
        i2, j2 = lay.o2[:, 0], lay.o2[:, 1]
        o12 = lay.o12

        def qs(q):
            return slice(qoff[q], qoff[q + 1])

        def cls(p):
            return slice(v0[p], v0[p + 1])
        E = D._E
        with timer('dea5.ground'):
            # pp: Wq5 = Wpp + F_vv on pairs / 4
            self.Wq5 = []
            for q in range(lay.n2):
                F = D._pair_one_body(g.Fv, aq[qs(q)], bq[qs(q)]) if nQ[q] else 0
                self.Wq5.append(_c(g.Wpp[q] + F / 4, dtype))
            # hh: the occupied triples of each class, sigma[x] = sum_y H[x, y] r5[y]
            W2p = g.Whh[i2, j2][:, i2, j2] - D.t4engine._one_body_pairs(g.Fo, no) / 2      # [(mn), (ij)]
            self.H = {}
            for c in range(lay.n3):
                rows = np.asarray(lay.O3cls[c], dtype=np.int64)
                if not len(rows):
                    continue
                Hc = np.zeros((len(rows), len(rows)), dtype=dtype)
                for xi, I in enumerate(rows):
                    t = lay.o3[I]
                    for (a, b), c3, sg in (((t[0], t[1]), t[2], 1.0), ((t[0], t[2]), t[1], -1.0),
                                           ((t[1], t[2]), t[0], 1.0)):
                        Pab = sec.pair[a, b]
                        Js, ss = o12.pos[c3], o12.psg[c3]               # sorted (c3, m, n) per pair (mn)
                        ok = (ss != 0) & (W2p[:, Pab] != 0)
                        for Pmn in np.flatnonzero(ok):
                            J = int(Js[Pmn])
                            if lay.kO3[J] != c:
                                continue
                            Hc[xi, lay.posO3[J]] += sg * ss[Pmn] * W2p[Pmn, Pab]
                self.H[c] = Hc
            # T3 on (single | pair) for every row: T3x[I, e, Q] (Q all pairs in the order of the classes)
            T3p = np.asarray(T3p).reshape(nI3, comb(nv, 3))
            T3x = np.zeros((nI3, nv, nQt), dtype=dtype)
            v3 = P.tuples(nv, 3)
            Q2 = np.full((nv, nv), -1, dtype=np.int64)
            Q2[aq, bq] = np.arange(nQt)
            Q2[bq, aq] = np.arange(nQt)
            for (x, y, z), sg in (((0, 1, 2), 1.0), ((1, 0, 2), -1.0), ((2, 0, 1), 1.0)):
                T3x[:, v3[:, x], Q2[v3[:, y], v3[:, z]]] = sg * T3p
            del v3
            fov, oovv, oovo = g.fov, g.oovv, g.oovo
            vovv_q = g.vovv_q                                   # [d, m, Q]
            t2_xq, t2_px = g.t2_xq, g.t2_px                     # [i, m, Q], [P, a, e]
            # Xo3(m; ijk; ab) = P(k/ij) sum_l Whh(lm, ij) t2(kl; ab) - P(i/jk) P(ab) sum_h Wr(m,a,h,i) t2(jk; bh)
            #   - sum_h f_mh t3(ijk; h ab) + P(i/jk) sum_{lh} <ml||hi> t3(jkl; h ab) + P(ab) sum_{h<g} <am||hg> t3(ijk; b hg)
            Xo3 = np.zeros((nI3, no, nQt), dtype=dtype)
            A1 = E('lmP,xlQ->PmxQ', g.Whh[:, :, i2, j2], t2_xq)                      # [(yz), m, x, Q]
            Wr = np.asarray(g.Wr)                                                     # [m, a, h, i]
            for I in range(nI3):
                t = lay.o3[I]
                for x, (y, z), sg in ((t[0], (t[1], t[2]), 1.0), (t[1], (t[0], t[2]), -1.0), (t[2], (t[0], t[1]), 1.0)):
                    Pyz = sec.pair[y, z]
                    Xo3[I] += sg * A1[Pyz, :, x, :]
                    A2 = E('mah,bh->mab', Wr[:, :, :, x], t2_px[Pyz])
                    Xo3[I] -= sg * (A2[:, aq, bq] - A2[:, bq, aq])
                    # <ml||hi> t3(yzl; h ab), i = x
                    for l in range(no):
                        J, sJ = o12.pos[l, Pyz], o12.psg[l, Pyz]
                        if sJ:
                            Xo3[I] += sg * sJ * (oovo[:, l, :, x] @ T3x[J])
                Xo3[I] -= fov @ T3x[I]
                Y = (vovv_q.reshape(nv * no, nQt) @ T3x[I].T).reshape(nv, no, nv)    # [a, m, b] = sum <am||Q> t3(I; b Q)
                Y = Y.transpose(1, 0, 2)
                Xo3[I] += Y[:, aq, bq] - Y[:, bq, aq]
            del A1
            self.Xo3 = Xo3
            # W2(mn; ijk; a) = P(i/jk) sum_h <mn||hi> t2(jk; ah) + sum_{h<g} <mn||hg> t3(ijk; a hg)   as [I, a, (mn)]
            oovv_pq = _c(oovv[i2, j2][:, aq, bq], dtype)                             # [(mn), Q]
            W2 = np.zeros((nI3, nv, nP), dtype=dtype)
            for I in range(nI3):
                t = lay.o3[I]
                W2[I] += T3x[I] @ oovv_pq.T
                for x, (y, z), sg in ((t[0], (t[1], t[2]), 1.0), (t[1], (t[0], t[2]), -1.0), (t[2], (t[0], t[1]), 1.0)):
                    W2[I] += sg * (t2_px[sec.pair[y, z]] @ oovo[i2, j2, :, x].T)
            self.W2 = W2
            # B4(ml g; jk; c) = sum_h <ml||hg> t2(jk; ch)  as [(jk), m, l, g, c]
            self.B4 = _c(E('mlhg,Pch->Pmlgc', oovv, t2_px), dtype)
            # Xv2 = Xv + S(a|bc) 1/2 sum_h Wpp(bc, hg) t2(ij; ah): the rows of Xv in R13 (below the rows of T3)
            self.Xv2 = np.zeros(max(int(lay.r13size), 1), dtype=dtype) if xv2 is None else xv2
            self.Xv2[:] = g.R13[:len(self.Xv2)]
            n12 = {}
            for K in range(lay.n3):
                off = np.full(ncls, -1, dtype=np.int64)
                n = 0
                for p in range(ncls):
                    if nV[p] and 0 <= K - p < lay.n2 and nQ[K - p]:
                        off[p] = n
                        n += nV[p] * nQ[K - p]
                n12[K] = (off, n)
            for Pq in range(nP):
                kP = lay.kO2[Pq]
                for phi in range(ncls):
                    o = lay.r13off[Pq, phi]
                    if o < 0 or not nV[phi]:
                        continue
                    K = kP + phi
                    off, n = n12[K]
                    if not n:
                        continue
                    U = np.zeros((nV[phi], n), dtype=dtype)              # [g, (a | bc)]
                    for p in range(ncls):
                        if off[p] < 0:
                            continue
                        qq = K - p                                      # class of (bc)
                        mu = qq - phi                                   # class of h
                        if not (0 <= mu < ncls and nV[mu]) or lay.wboff[qq, mu] < 0:
                            continue
                        wo = lay.wboff[qq, mu]
                        Wb = g.Wb[wo:wo + nQ[qq] * nV[mu] * nV[phi]].reshape(nQ[qq], nV[mu], nV[phi])
                        t2b = t2_px[Pq][cls(p), cls(mu)]               # t2(ij; a h)
                        x = E('Qhg,ah->gaQ', Wb, t2b)
                        U[:, off[p]:off[p] + nV[p] * nQ[qq]] += 0.5 * x.reshape(nV[phi], -1)
                    rows = o + nO[phi] * nK[K]
                    tgt = self.Xv2[rows:rows + nV[phi] * nK[K]].reshape(nV[phi], nK[K])
                    xoff = np.where(off[None, :] >= 0, np.arange(nV[phi])[:, None] * n + off[None, :], -1)
                    nc = np.where(off >= 0, nQ[np.clip(K - np.arange(ncls), 0, lay.n2 - 1)], 0)
                    ent = (np.arange(nV[phi], dtype=np.int64), np.ones(nV[phi]), _c(xoff, np.int64),
                           _c(np.broadcast_to(nc, xoff.shape), np.int64), np.ones(xoff.shape, dtype=np.int64))
                    S._x12(lay, 1, tgt, K, ent, _c(U, dtype))
            self.T3x = None

    def nbytes(self):
        return (sum(w.nbytes for w in self.Wq5) + sum(h.nbytes for h in self.H.values()) + self.Xo3.nbytes
                + self.W2.nbytes + self.B4.nbytes + self.Xv2.nbytes + self.ovvo.nbytes)


# ------------------------------------------------------------------ the sigma vector

PIECES5 = ('pp', 'hh', 'ring', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'E1', 'E1x', 'E2', 'low')

_SINGLES = ((0, (1, 2), 1.0), (1, (0, 2), -1.0), (2, (0, 1), 1.0))       # (i | jk) of a sorted triple and the sign


class Engine5:
    '''The sigma vector with R to 5p3h (module docstring): the rows to 4p2h
    by the engine of dea_sector ``eng4`` (R to 4p2h) plus the terms of r5,
    the 5p3h row from ``g`` (dea_sector.Ground) and ``g5`` (Ground5).'''

    def __init__(self, sec5, g, g5, eng4, batch_bytes=None):
        self.sec5, self.sec, self.lay, self.g, self.g5, self.eng4 = sec5, sec5.sec, sec5.lay, g, g5, eng4
        eng4.keep = True
        self.dtype = dtype = g.dtype
        self.batch_bytes = batch_bytes or BATCH_BYTES
        lay, sec, s = self.lay, self.sec, self.sec.s
        no, nv, ncls, ko = lay.no, lay.nv, lay.ncls, lay.ko
        nP = len(lay.o2)
        self.nP = nP
        # r4 on (h | cde), one row after the other
        self.x4off = np.full(nP + 1, 0, dtype=np.int64)
        for P_ in range(nP):
            M4 = sec.blk4[P_]
            self.x4off[P_ + 1] = self.x4off[P_] + (sec.n13[M4] if M4 >= 0 else 0)
        # XoR5(m; jk; cde): per class K of the triple, rows (jk, m) - the m of one class together
        self.xr5 = np.full((no, nP, 2), -1, dtype=np.int64)
        cnt = {}
        for P_ in range(nP):
            for m in range(no):
                K = s + lay.kO2[P_] - ko[m]
                if 0 <= K < lay.n3 and lay.nK[K]:
                    self.xr5[m, P_] = (K, cnt.get(K, 0))
                    cnt[K] = cnt.get(K, 0) + 1
        self.xr5n = cnt
        # Xv5(f; k; bcde): (k, class of f) -> [f, quadruples of class s + k_k + class of f]
        self.xv5keys = [(k, ph) for k in range(no) for ph in range(ncls)
                        if lay.nV[ph] and 0 <= s + ko[k] + ph < lay.n4 and sec5.nA4[s + ko[k] + ph]]
        i2, j2 = lay.o2[:, 0], lay.o2[:, 1]
        self.oovv_cc = {}
        for c in range(lay.n2):
            if lay.nO2[c] and lay.nQ[c]:
                oc = lay.O2cls[c]
                self.oovv_cc[c] = _c(g.oovv[i2[oc], j2[oc]][:, lay.va[self.qs(c)], lay.vb[self.qs(c)]], dtype)
        self._ring = {}
        self._ws = {}
        self.nsigma = 0

    def buf(self, name, n, zero=False):
        '''A work array of n elements kept from one sigma vector to the next
        (no fresh pages - and no page faults - for the large temporaries).'''
        a = self._ws.get(name)
        if a is None or a.size < n:
            self._ws.pop(name, None)
            a = np.empty(max(int(n), 1), dtype=self.dtype)
            self._ws[name] = a
        a = a[:n]
        return P.zero(a) if zero else a

    @staticmethod
    def _stack(parts, out):
        """The row blocks ``parts`` one below the other in ``out`` (large ones on all threads)."""
        r0 = 0
        for a in parts:
            n = len(a)
            if n * a.shape[1] > 1 << 16 and a.flags.c_contiguous:
                P.tn_rows(out[r0:r0 + n], a, np.arange(n), np.ones(n))
            else:
                out[r0:r0 + n] = a
            r0 += n
        return out

    # ---------------------------------------------------------------- views
    def qs(self, q):
        return slice(self.lay.qoff[q], self.lay.qoff[q + 1])

    def cls(self, p):
        return slice(self.lay.v0[p], self.lay.v0[p + 1])

    def ocls(self, p):
        return slice(self.lay.o0[p], self.lay.o0[p + 1])

    def okc(self, p):
        return 0 <= p < self.lay.ncls and self.lay.nO[p] > 0

    def vkc(self, p):
        return 0 <= p < self.lay.ncls and self.lay.nV[p] > 0

    def x4(self, X4r, P_, ph):
        '''r4 of the row P_ on (h of class ph | triples), or None.'''
        sec = self.sec
        M4 = sec.blk4[P_]
        if M4 < 0 or not 0 <= ph < self.lay.ncls:
            return None
        o = sec.off13[M4, ph]
        if o < 0:
            return None
        base = self.x4off[P_] + o
        return X4r[base:base + self.lay.nV[ph] * self.lay.nK[M4 - ph]].reshape(self.lay.nV[ph], self.lay.nK[M4 - ph])

    def xor5_rows(self, XoR5, m0, nm, P_):
        '''The rows (m0 .. m0 + nm, P_) of XoR5 (m of one class), or None.'''
        K, pos = self.xr5[m0, P_]
        if K < 0:
            return None
        return XoR5[K][pos:pos + nm]

    # ---------------------------------------------------------------- sigma
    def sigma(self, r2v, r3, r4, r5, only=None):
        '''(sigma2, sigma3, sigma4, sigma5) of the sector vector, flat arrays
        of the blocks of Sector and Sector5.  ``only``: the pieces of
        PIECES5 of the 5p3h row to add ('low': the rows to 4p2h; tests).'''
        def want(x):
            return only is None or x in only
        sec, sec5, lay, g, g5, dtype = self.sec, self.sec5, self.lay, self.g, self.g5, self.dtype
        r5 = _c(r5, dtype)
        r4 = _c(r4, dtype)
        s2, s3, s4 = self.eng4.sigma(r2v, r3, r4)
        if not want('low'):
            s2[:] = 0
            s3[:] = 0
            s4[:] = 0
        imds = self.eng4.imds
        s5 = self.buf('s5', sec5.size5, zero=True)
        with timer('dea5.small'):
            with timer('dea5.x4r'):
                X4r = self._x4r(r4)
            with timer('dea5.z4'):
                Z4 = self._z4(X4r)
            with timer('dea5.xvr5'):
                XvR5 = self._xvr5(imds['Z'], imds['r2'])
            with timer('dea5.pairs'):
                Y4, XoR5, XoRp = self._pair_imds(r4, X4r, imds)
            with timer('dea5.xv5'):
                Xv5 = self._xv5(X4r, imds['r2'])
        X4s = self.buf('x4s', int(self.x4off[-1]), zero=True)
        isz = np.dtype(dtype).itemsize
        with timer('dea5.pp'):
            for M in sec5.blocks5:
                R, out = sec5.block5(r5, M), sec5.block5(s5, M)
                n23 = int(sec5.n23[M])
                if not n23:
                    continue
                for r0, r1 in _batches(len(sec5.rows5[M]), self.batch_bytes // (isz * n23)):
                    nb = r1 - r0
                    ent = sec5.ent_pp(M, r0, r1)
                    Tv = self.buf('tv', nb * n23, zero=True)
                    with timer('dea5.expand'):
                        sec5.x23(0, R, M, ent, Tv)
                    with timer('dea5.thin'):
                        self._pp_thin(M, r0, r1, Tv, s3, X4s, XoR5)
                    if want('pp'):
                        X = self.buf('tx', nb * n23)
                        for q in range(lay.n2):
                            o = sec5.off23[M, q]
                            if o < 0:
                                continue
                            sl = slice(o * nb, (o + lay.nQ[q] * lay.nK[M - q]) * nb)
                            with timer('dea5.gemm'):
                                gemm(g5.Wq5[q], Tv[sl].reshape(lay.nQ[q], -1), X[sl].reshape(lay.nQ[q], -1))
                        with timer('dea5.reduce'):
                            sec5.x23(1, out, M, ent, X)
                        del X
                    del Tv
        with timer('dea5.ring'):
            self._ring_pass(r5, s5, s4, Xv5, want('ring'))
        if want('hh'):
            with timer('dea5.hh'):
                for M in sec5.blocks5:
                    gemm(g5.H[M - sec.s], sec5.block5(r5, M), sec5.block5(s5, M), beta=1.0)
        with timer('dea5.rows'):
            self._rows(r3, r4, imds['X3r'], X4r, Z4, XvR5, Y4, XoR5, XoRp, Xv5, s5, want)
        # sigma4 from r5 on (a | bcd), reduced
        if want('low'):
            for M4 in sec.blocks4:
                rows = sec.rows4[M4]
                ok = sec.off13[M4] >= 0
                nc = np.where(ok, lay.nK[np.clip(M4 - np.arange(lay.ncls), 0, lay.n3 - 1)], 0)
                xoff = np.where(ok[None, :], self.x4off[rows][:, None] + sec.off13[M4][None, :], -1)
                ent = (np.arange(len(rows), dtype=np.int64), -np.ones(len(rows)), _c(xoff, np.int64),
                       _c(np.broadcast_to(nc, xoff.shape), np.int64), np.ones(xoff.shape, dtype=np.int64))
                sec.x13(1, sec.block4(s4, M4), M4, ent, X4s)
        self.nsigma += 1
        return s2, s3, s4, s5

    # ---------------------------------------------------------------- intermediates of the sector
    def _x4r(self, r4):
        sec, lay, dtype = self.sec, self.lay, self.dtype
        X4r = self.buf('x4r', max(int(self.x4off[-1]), 1), zero=True)
        for M4 in sec.blocks4:
            rows = sec.rows4[M4]
            ok = sec.off13[M4] >= 0
            nc = np.where(ok, lay.nK[np.clip(M4 - np.arange(lay.ncls), 0, lay.n3 - 1)], 0)
            xoff = np.where(ok[None, :], self.x4off[rows][:, None] + sec.off13[M4][None, :], -1)
            ent = (np.arange(len(rows), dtype=np.int64), np.ones(len(rows)), _c(xoff, np.int64),
                   _c(np.broadcast_to(nc, xoff.shape), np.int64), np.ones(xoff.shape, dtype=np.int64))
            sec.x13(0, sec.block4(r4, M4), M4, ent, X4r)
        return X4r

    def _z4(self, X4r):
        '''Z4(mf; k; cde) = sum_{lh} <ml||fh> r4(lk; h cde): (k, class of m, class of f) -> [m, f, K].'''
        sec, lay, g, dtype = self.sec, self.lay, self.g, self.dtype
        no, ncls, ko, nO, nV = lay.no, lay.ncls, lay.ko, lay.nO, lay.nV
        if not hasattr(self, '_z4d'):
            self._z4d = {}
        Z4 = self._z4d
        for z in Z4.values():
            P.zero(z)
        for k in range(no):
            for l in range(no):
                if l == k:
                    continue
                Pl = sec.pair[l, k]
                sl = 1.0 if l < k else -1.0
                for ph in range(ncls):
                    G = self.x4(X4r, Pl, ph)
                    if G is None:
                        continue
                    for mu in range(ncls):
                        fe = mu + ko[l] - ph
                        if not (nO[mu] and self.vkc(fe)):
                            continue
                        A = _c(g.oovv[self.ocls(mu), l][:, self.cls(fe), self.cls(ph)].reshape(nO[mu] * nV[fe], nV[ph]), dtype)
                        key = (k, mu, fe)
                        if key not in Z4:
                            Z4[key] = np.zeros((nO[mu], nV[fe], G.shape[1]), dtype=dtype)
                        gemm(A, G, Z4[key].reshape(nO[mu] * nV[fe], G.shape[1]), alpha=sl, beta=1.0)
        return Z4

    def _xvr5(self, Z, r2):
        '''XvR5 = XvR of dea_sector (Z) with its term of Wpp counted twice.'''
        sec, lay, g, eng = self.sec, self.lay, self.g, self.eng4
        s, ncls, nV, nQ, nK = sec.s, lay.ncls, lay.nV, lay.nQ, lay.nK
        out = {K: z.copy() for K, z in Z.items()}
        if not sec.n2:
            return out
        for K, z in out.items():
            mu = K - s
            Uk = np.zeros((nV[mu], eng.n12_all(K)), dtype=self.dtype)
            for dl in range(ncls):
                q = K - dl
                if not (nV[dl] and 0 <= q < lay.n2 and nQ[q]) or lay.wboff[q, mu] < 0:
                    continue
                ph = q - mu
                o = lay.wboff[q, mu]
                W = g.Wb[o:o + nQ[q] * nV[mu] * nV[ph]].reshape(nQ[q] * nV[mu], nV[ph])
                x = (W @ r2[self.cls(ph), self.cls(dl)]).reshape(nQ[q], nV[mu], nV[dl])
                o2 = eng.off12_all(K, dl)
                Uk[:, o2:o2 + nV[dl] * nQ[q]] += 0.5 * x.transpose(1, 2, 0).reshape(nV[mu], -1)
            sec.x12(1, z, K, eng.ent_u(K), _c(Uk, self.dtype))
        return out

    def _pair_imds(self, r4, X4r, imds):
        '''Y4, XoR5 (without its term of r5) and XoRp.'''
        sec, lay, g, g5, dtype, eng = self.sec, self.lay, self.g, self.g5, self.dtype, self.eng4
        no, nv, ncls, ko, s = lay.no, lay.nv, lay.ncls, lay.ko, sec.s
        nO, nV, nQ, nK = lay.nO, lay.nV, lay.nQ, lay.nK
        aq, bq = lay.va, lay.vb
        nQt = len(lay.kQ)
        qs, cls, ocls = self.qs, self.cls, self.ocls
        o12 = lay.o12
        r2, X3r, WhR, YR = imds['r2'], imds['X3r'], imds['WhR'], imds['YR']
        E = D._E
        fov, oovv, oovo, vovv, vovv_q = g.fov, g.oovv, g.oovo, g.vovv, g.vovv_q
        t2_xq, t2_px = g.t2_xq, g.t2_px
        oovv_xq = g.oovv_xq
        # r3 dense on (h | pair): D3r[i, h, Q]
        D3r = np.zeros((no, nv, nQt), dtype=dtype)
        for i in range(no):
            K = sec.blk3[i]
            if K < 0:
                continue
            for ph in range(ncls):
                b = eng.x3(X3r, i, ph)
                if b is not None:
                    D3r[i, cls(ph), qs(K - ph)] = b
        # ---- XoRp(m; i; de)
        X = -E('mdhi,eh->mide', g5.ovvo, r2)
        X -= E('lmhg,dh,ileg->mide', oovv, r2, g.t2)
        X += E('dmQ,ieQ->mide', vovv_q, D3r)
        XoRp = X[:, :, aq, bq] - X[:, :, bq, aq]
        del X
        XoRp -= E('mh,ihQ->miQ', fov, D3r)
        XoRp += E('mlhi,lhQ->miQ', oovo, D3r)
        XoRp += E('lm,ilQ->miQ', WhR, t2_xq)
        # ---- the small dense pieces of XoR5
        G = E('mlhy,ch->mlyc', oovo, r2)                       # G(ml; y; c) = <ml||hy> r2(ch)
        G3 = E('mlhg,ch->mlcg', oovv, r2)                      # G3(ml; c, g)
        G2 = E('cmhg,dh->mcdg', vovv, r2)                      # G2(m; c, d, g) = <cm||hg> r2(dh)
        if not hasattr(self, '_xor5'):
            self._xor5 = {K: np.empty((n, nK[K]), dtype=dtype) for K, n in self.xr5n.items()}
        XoR5 = self._xor5
        for z in XoR5.values():
            P.zero(z)
        Y4 = {}
        nP = self.nP
        X22 = np.empty(max(int(sec.n22.max(initial=0)), 1), dtype=dtype)
        for P_ in range(nP):
            j, k = (int(x) for x in lay.o2[P_])
            M4 = sec.blk4[P_]
            kP = lay.kO2[P_]
            # r4 of the row on (hg | de)
            have = M4 >= 0 and sec.n22[M4] > 0
            if have:
                r = sec.pos4[P_]
                X22v = P.zero(X22[:sec.n22[M4]])
                sec.x22(0, sec.block4(r4, M4), M4, sec.ent_row(M4, r, '22'), X22v)

                def x22(q):
                    o = sec.off22[M4, q]
                    if o < 0:
                        return None
                    return X22v[o:o + nQ[q] * nQ[M4 - q]].reshape(nQ[q], nQ[M4 - q])
                for c, oc in self.oovv_cc.items():
                    b = x22(c)
                    if b is not None:
                        Y4[(P_, c)] = oc @ b
                # XoRp: - sum_l sum_{h<g} <ml||hg> r4(il; hg de), (i, l) = (j, k) and (k, j)
                for i, l, sg in ((j, k, 1.0), (k, j, -1.0)):
                    for q in range(lay.n2):
                        b = x22(q)
                        mu = q - ko[l]
                        if b is None or not self.okc(mu):
                            continue
                        XoRp[ocls(mu), i, qs(M4 - q)] -= sg * (oovv_xq[ocls(mu), l, qs(q)] @ b)
            # XoR5 rows (m of class mu, jk): the products on (c | de), reduced once per class of m
            for mu in range(ncls):
                if not nO[mu]:
                    continue
                K = s + kP - mu
                if not (0 <= K < lay.n3 and nK[K]):
                    continue
                off, n12 = eng._u_tables(K)
                if not n12:
                    continue
                ms = ocls(mu)
                nm = nO[mu]
                U = np.zeros((nm, n12), dtype=dtype)
                for p in range(ncls):
                    if off[p] < 0:
                        continue
                    Q = K - p
                    blk = U[:, off[p]:off[p] + nV[p] * nQ[Q]].reshape(nm, nV[p], nQ[Q])
                    # (b) sum_{h<g} <cm||hg> r4(jk; hg de)
                    if have:
                        qq = M4 - Q
                        b = x22(qq)
                        if b is not None:
                            A = vovv_q[cls(p), ms, qs(qq)].transpose(1, 0, 2).reshape(nm * nV[p], nQ[qq])
                            blk += (A @ b).reshape(nm, nV[p], nQ[Q])
                    # (c) 1/2 P(jk) sum_l G(ml; y; c) t2(zl; de);  (g) -1/2 P(jk) sum_l YR(ml; z; e) t2(yl; cd)
                    for y, z, sg in ((j, k, 0.5), (k, j, -0.5)):
                        A = G[ms, :, y, cls(p)].transpose(0, 2, 1).reshape(nm * nV[p], no)
                        blk += sg * (A @ t2_xq[z, :, qs(Q)]).reshape(nm, nV[p], nQ[Q])
                        A = YR[ms, :, z, cls(p)].transpose(0, 2, 1).reshape(nm * nV[p], no)
                        blk -= sg * (A @ t2_xq[y, :, qs(Q)]).reshape(nm, nV[p], nQ[Q])
                    # (h) sum_{lg} G3(ml; c, g) t3(jkl; g de);  (i) sum_{lg} B4(ml g; jk; c) r3(l; g de)
                    for l in range(no):
                        sJ = o12.psg[l, P_]
                        if sJ:
                            J = int(o12.pos[l, P_])
                            kJ = lay.kO3[J]
                            ph = kJ - Q
                            t0 = lay.t12off[kJ, ph] if 0 <= ph < ncls else -1
                            if t0 >= 0:
                                nJk = int(lay.nJ[kJ])
                                Tg = g.T12[t0:t0 + nV[ph] * nJk * nQ[Q]].reshape(nV[ph], nJk, nQ[Q])[:, lay.posO3[J]]
                                A = G3[ms, l][:, cls(p), cls(ph)].reshape(nm * nV[p], nV[ph])
                                blk += sJ * (A @ Tg).reshape(nm, nV[p], nQ[Q])
                        ph = s + ko[l] - Q
                        b = eng.x3(X3r, l, ph)
                        if b is not None and b.shape[1] == nQ[Q]:
                            A = g5.B4[P_, ms, l][:, cls(ph), cls(p)].transpose(0, 2, 1).reshape(nm * nV[p], nV[ph])
                            blk += (A @ b).reshape(nm, nV[p], nQ[Q])
                    # (d, e) - A(cde) sum_g G2(m; c, d, g) t2(jk; e g), as - [Y(c, d, e) - Y(c, e, d)] on (c | de)
                    Yd = (G2[ms, cls(p)].reshape(nm * nV[p] * nv, nv) @ t2_px[P_].T).reshape(nm, nV[p], nv, nv)
                    a_, b_ = aq[qs(Q)], bq[qs(Q)]
                    blk -= Yd[:, :, a_, b_] - Yd[:, :, b_, a_]
                rows = self.xr5[ms.start, P_]
                xoff = np.where(off[None, :] >= 0, np.arange(nm)[:, None] * n12 + off[None, :], -1)
                nc = np.where(off >= 0, nQ[np.clip(K - np.arange(ncls), 0, lay.n2 - 1)], 0)
                ent = (rows[1] + np.arange(nm, dtype=np.int64), np.ones(nm), _c(xoff, np.int64),
                       _c(np.broadcast_to(nc, xoff.shape), np.int64), np.ones(xoff.shape, dtype=np.int64))
                sec.x12(1, XoR5[K], K, ent, U)
            # (a) P(jk) sum_{lh} <ml||hy> r4(lz; h cde), (y, z) = (j, k) and (k, j)
            for y, z, sg in ((j, k, 1.0), (k, j, -1.0)):
                for l in range(no):
                    if l == z:
                        continue
                    Pl = sec.pair[l, z]
                    sl = 1.0 if l < z else -1.0
                    for ph in range(ncls):
                        Gx = self.x4(X4r, Pl, ph)
                        mu = ph + ko[y] - ko[l]
                        if Gx is None or not self.okc(mu):
                            continue
                        tgt = self.xor5_rows(XoR5, lay.o0[mu], nO[mu], P_)
                        if tgt is None or tgt.shape[1] != Gx.shape[1]:
                            continue
                        tgt += sg * sl * (oovo[ocls(mu), l, cls(ph), y] @ Gx)
        return Y4, XoR5, XoRp

    def _xv5(self, X4r, r2):
        '''Xv5 without its term of r5: S(b|cde) [ sum_{lh} <bl||fh> r4(lk; h cde)
        + sum_{l<n} G5(ln; f, b) t3(lnk; cde) ].'''
        sec, sec5, lay, g, dtype = self.sec, self.sec5, self.lay, self.g, self.dtype
        no, ncls, ko, s = lay.no, lay.ncls, lay.ko, sec.s
        nV, nK, nO2 = lay.nV, lay.nK, lay.nO2
        cls = self.cls
        i2, j2 = lay.o2[:, 0], lay.o2[:, 1]
        G5 = D._E('Phf,bh->Pfb', g.oovv[i2, j2], r2)          # G5(ln; f, b) = <ln||hf> r2(bh)
        if not hasattr(self, '_xv5d'):
            self._xv5d = {}
            for k, fe in self.xv5keys:
                self._xv5d[(k, fe)] = np.empty((nV[fe], sec5.nA4[s + ko[k] + fe]), dtype=dtype)
        Xv5 = self._xv5d
        for k, fe in self.xv5keys:
            A = s + ko[k] + fe
            n13 = int(sec5.n13[A])
            P.zero(Xv5[(k, fe)])
            if not n13:
                continue
            # U[(b of class p) block: f, b, (cde)] contiguous per class p
            U = self.buf('xv5u', nV[fe] * n13, zero=True)
            xoff = np.full((nV[fe], ncls), -1, dtype=np.int64)
            s1 = np.ones((nV[fe], ncls), dtype=np.int64)
            for p in range(ncls):
                o = sec5.off13[A, p]
                if o < 0:
                    continue
                nk = int(nK[A - p])
                base = nV[fe] * o
                xoff[:, p] = base + np.arange(nV[fe]) * nV[p] * nk
                s1[:, p] = nk
                blk = U[base:base + nV[fe] * nV[p] * nk].reshape(nV[fe] * nV[p], nk)
                for l in range(no):
                    if l == k:
                        continue
                    Pl = sec.pair[l, k]
                    sl = 1.0 if l < k else -1.0
                    ph = p - fe + ko[l]                     # class of h: k_b + k_l = k_f + k_h
                    Gx = self.x4(X4r, Pl, ph)
                    if Gx is None or Gx.shape[1] != nk:
                        continue
                    W = _c(g.vovv[cls(p), l][:, cls(fe), cls(ph)].transpose(1, 0, 2).reshape(nV[fe] * nV[p], nV[ph]), dtype)
                    gemm(W, Gx, blk, alpha=sl, beta=1.0)
                c = A - p - ko[k]
                if 0 <= c < lay.n2 and nO2[c] and lay.t3poff[k, c] >= 0:
                    o3 = lay.t3poff[k, c]
                    T = g.T3P[o3:o3 + nO2[c] * nk].reshape(nO2[c], nk)
                    W = _c(G5[lay.O2cls[c]][:, cls(fe), cls(p)].transpose(1, 2, 0).reshape(nV[fe] * nV[p], nO2[c]), dtype)
                    gemm(W, T, blk, beta=1.0)
            ent = (np.arange(nV[fe], dtype=np.int64), np.ones(nV[fe]), xoff, s1, np.ones(xoff.shape, dtype=np.int64))
            sec5.x13q(1, Xv5[(k, fe)], A, ent, U)
        return Xv5

    # ---------------------------------------------------------------- the passes over r5
    def _pp_thin(self, M, r0, r1, Tv, s3, X4s, XoR5):
        '''From r5 on [(fg), (row, (cde))]: XoR5 (term of r5), sigma3, sigma4 on (a | bcd).'''
        sec, sec5, lay, g = self.sec, self.sec5, self.lay, self.g
        ko, nQ, nK, nV, nO = lay.ko, lay.nQ, lay.nK, lay.nV, lay.nO
        nb = r1 - r0
        for q in range(lay.n2):
            o = sec5.off23[M, q]
            if o < 0:
                continue
            nc = int(nK[M - q])
            Tq = Tv[o * nb:(o + nQ[q] * nc) * nb].reshape(nQ[q], nb * nc)
            for r in range(r0, r1):
                t = lay.o3[sec5.rows5[M][r]]
                T = Tq[:, (r - r0) * nc:(r - r0 + 1) * nc]
                for x, (y, z), sg in _SINGLES:
                    n_, j, k = int(t[x]), int(t[y]), int(t[z])
                    Pjk = sec.pair[j, k]
                    mu = q - ko[n_]
                    if self.okc(mu):
                        tgt = self.xor5_rows(XoR5, lay.o0[mu], nO[mu], Pjk)
                        if tgt is not None and tgt.shape[1] == nc:
                            gemm(g.oovv_xq[self.ocls(mu), n_, self.qs(q)], T, tgt, alpha=sg, beta=1.0)
                    # sigma3(n_; abc) += sum_{e<f} <jk||ef> r5(n_ jk; ef abc)
                    if q == lay.kO2[Pjk] and sec.blk3[n_] == M - q:
                        xs = sec.block3(s3, M - q)[sec.pos3[n_]]
                        xs += sg * (g.oovv_xq[j, k, self.qs(q)] @ T)
                    # sigma4(jk; a bcd) -= S(a|bcd) sum_{e<f} <a n_||ef> r5(n_ jk; ef bcd), on (a | bcd)
                    p = q - ko[n_]
                    M4 = sec.blk4[Pjk]
                    if self.vkc(p) and M4 >= 0 and sec.off13[M4, p] >= 0:
                        base = self.x4off[Pjk] + sec.off13[M4, p]
                        tgt = X4s[base:base + nV[p] * nc].reshape(nV[p], nc)
                        gemm(g.vovv_q[self.cls(p), n_, self.qs(q)], T, tgt, alpha=sg, beta=1.0)

    def ring_tables(self, P_):
        if P_ in self._ring:
            return self._ring[P_]
        sec, sec5, lay = self.sec, self.sec5, self.lay
        no, nv, ncls, ko, s = lay.no, lay.nv, lay.ncls, lay.ko, sec.s
        nO, nV, o0, v0 = lay.nO, lay.nV, lay.o0, lay.v0
        u, w = (int(x) for x in lay.o2[P_])
        kP = lay.kO2[P_]
        nW = no * nv
        blocks = []
        for d in range(-(ncls - 1), ncls):
            A = s + kP + d
            if not (0 <= A < lay.n4 and sec5.nA4[A]):
                continue
            cols, base = [], {}
            for m in range(no):
                c = ko[m] - d
                if m not in (u, w) and 0 <= c < ncls and nV[c]:
                    base[m] = len(cols)
                    cols.extend(m * nv + np.arange(v0[c], v0[c + 1]))
            if not cols:
                continue
            rows, xbase = [], {}
            for i in range(no):
                c = ko[i] - d
                if i not in (u, w) and 0 <= c < ncls and nV[c]:
                    xbase[i] = len(rows)
                    rows.extend(i * nv + np.arange(v0[c], v0[c + 1]))
            fov = -1
            if d == 0:
                fov = len(rows)
                rows.append(nW)
            extra = {}
            cu, cw = ko[u] + d, ko[w] + d
            for name, c, base_, kind in (('pu', cu, nW + 1, 'o'), ('pw', cw, nW + 1 + no, 'o'),
                                         ('xu', cu, nW + 1 + 2 * no, 'v'), ('xw', cw, nW + 1 + 2 * no + nv, 'v')):
                if not 0 <= c < ncls:
                    continue
                idx = np.arange(o0[c], o0[c + 1]) if kind == 'o' else np.arange(v0[c], v0[c + 1])
                if len(idx):
                    extra[name] = (len(rows), idx)
                    rows.extend(base_ + idx)
            blocks.append(dict(d=d, A=A, nA=int(sec5.nA4[A]), cols=np.array(cols, dtype=np.int64),
                               rows=np.array(rows, dtype=np.int64), base=base, xbase=xbase, fov=fov, extra=extra))
        self._ring[P_] = blocks
        return blocks

    def ring_entries(self, P_, blocks, mo, xo):
        sec5, lay = self.sec5, self.lay
        ncls, ko = lay.ncls, lay.ko
        o12 = lay.o12
        out = []
        for offs, key in ((mo, 'base'), (xo, 'xbase')):
            ents = {}
            for m in range(lay.no):
                sg = o12.psg[m, P_]
                if not sg:
                    continue
                I = int(o12.pos[m, P_])
                M = sec5.blk5[I]
                if M < 0:
                    continue
                xoff = np.full(ncls, -1, dtype=np.int64)
                s1 = np.ones(ncls, dtype=np.int64)
                for b, o in zip(blocks, offs):
                    if m in b[key]:
                        l = ko[m] - b['d']
                        xoff[l] = o + b[key][m] * b['nA']
                        s1[l] = b['nA']
                if (xoff < 0).all():
                    continue
                ents.setdefault(M, []).append((sec5.pos5[I], float(sg), xoff, s1))
            tabs = {}
            for M, es in ents.items():
                tabs[M] = (np.array([e[0] for e in es], dtype=np.int64), np.array([e[1] for e in es]),
                           np.ascontiguousarray([e[2] for e in es], dtype=np.int64),
                           np.ascontiguousarray([e[3] for e in es], dtype=np.int64),
                           np.ones((len(es), ncls), dtype=np.int64))
            out.append(tabs)
        return out

    def _ring_pass(self, r5, s5, s4, Xv5, ring):
        sec, sec5, lay, g, dtype = self.sec, self.sec5, self.lay, self.g, self.dtype
        no, nv = lay.no, lay.nv
        isz = np.dtype(dtype).itemsize
        nW = no * nv
        Wr2 = self.eng4.Wr2
        for P_ in range(self.nP):
            blocks = self.ring_tables(P_)
            if not blocks:
                continue
            u, w = (int(x) for x in lay.o2[P_])
            Sj = np.empty((nW + 1 + 2 * no + 2 * nv, nW), dtype=dtype)
            Sj[:nW] = Wr2
            Sj[nW] = g.fov.ravel()
            Sj[nW + 1:nW + 1 + no] = g.oovo[:, u, :, :].transpose(2, 0, 1).reshape(no, nW)
            Sj[nW + 1 + no:nW + 1 + 2 * no] = -g.oovo[:, w, :, :].transpose(2, 0, 1).reshape(no, nW)
            # Xv5(f; k) = 1/2 sum_{lnh} <ln||fh> r5(lnk; h ..): the spectator n of the pair, l = m running
            Sj[nW + 1 + 2 * no:nW + 1 + 2 * no + nv] = 0.5 * g.oovv[:, u, :, :].transpose(1, 0, 2).reshape(nv, nW)
            Sj[nW + 1 + 2 * no + nv:] = -0.5 * g.oovv[:, w, :, :].transpose(1, 0, 2).reshape(nv, nW)
            batch, cur, tot = [], [], 0
            for b in blocks:
                sz = (len(b['cols']) + len(b['rows'])) * b['nA']
                if cur and (tot + sz) * isz > self.batch_bytes:
                    batch.append(cur)
                    cur, tot = [], 0
                cur.append(b)
                tot += sz
            batch.append(cur)
            for bl in batch:
                mo, xo, m0, x0 = [], [], 0, 0
                for b in bl:
                    mo.append(m0)
                    xo.append(x0)
                    m0 += len(b['cols']) * b['nA']
                    x0 += len(b['rows']) * b['nA']
                Mb = self.buf('mb', m0, zero=True)
                Xb = self.buf('xb', x0)
                em, ex = self.ring_entries(P_, bl, mo, xo)
                with timer('dea5.expand'):
                    for M, ent in em.items():
                        sec5.x14(0, sec5.block5(r5, M), M, ent, Mb)
                with timer('dea5.gemm'):
                    for b, mo_, xo_ in zip(bl, mo, xo):
                        nc, nr, nA = len(b['cols']), len(b['rows']), b['nA']
                        W = _c(Sj[np.ix_(b['rows'], b['cols'])], dtype)
                        gemm(W, Mb[mo_:mo_ + nc * nA].reshape(nc, nA), Xb[xo_:xo_ + nr * nA].reshape(nr, nA))
                if ring:
                    with timer('dea5.reduce'):
                        for M, ent in ex.items():
                            sec5.x14(1, sec5.block5(s5, M), M, ent, Xb)
                for b, xo_ in zip(bl, xo):
                    nA, A = b['nA'], b['A']
                    X = Xb[xo_:xo_ + len(b['rows']) * nA].reshape(len(b['rows']), nA)
                    if b['fov'] >= 0 and sec.blk4[P_] == A:
                        sec.block4(s4, A)[sec.pos4[P_]] += X[b['fov']]
                    for name, tgt_o in (('pu', w), ('pw', u)):
                        if name not in b['extra']:
                            continue
                        r0, idx = b['extra'][name]
                        for x, i in enumerate(idx):
                            if i == tgt_o:
                                continue
                            Pt = sec.pair[i, tgt_o]
                            if sec.blk4[Pt] != A:
                                continue
                            sec.block4(s4, A)[sec.pos4[Pt]] += (-0.5 if i < tgt_o else 0.5) * X[r0 + x]
                    for name, k in (('xu', w), ('xw', u)):
                        if name not in b['extra']:
                            continue
                        r0, idx = b['extra'][name]
                        key = (k, lay.kv[idx[0]])
                        if key in Xv5:
                            Xv5[key] += X[r0:r0 + len(idx)]
                del Mb, Xb

    # ---------------------------------------------------------------- the rows of sigma5
    def _rows(self, r3, r4, X3r, X4r, Z4, XvR5, Y4, XoR5, XoRp, Xv5, s5, want):
        sec, sec5, lay, g, g5, dtype, eng = self.sec, self.sec5, self.lay, self.g, self.g5, self.dtype, self.eng4
        no, ncls, ko, s = lay.no, lay.ncls, lay.ko, sec.s
        nO, nV, nQ, nK = lay.nO, lay.nV, lay.nQ, lay.nK
        qs, cls, ocls = self.qs, self.cls, self.ocls
        o12 = lay.o12
        t2_xq = g.t2_xq
        # E1: [r4(m x; .) of m of class mu; Xv5(f; x; .) of f of class mu] over quadruples of class s + k_x + mu,
        # the rows of r4 gathered once (Xv5 as it is)
        top14 = {}
        for x in range(no):
            for mu in range(ncls):
                A = s + ko[x] + mu
                if not (0 <= A < lay.n4 and sec5.nA4[A]) or not (nO[mu] or nV[mu]):
                    continue
                top = np.zeros((nO[mu], sec5.nA4[A]), dtype=dtype)
                for xm, m in enumerate(range(lay.o0[mu], lay.o0[mu + 1])):
                    if m == x:
                        continue
                    Pm = sec.pair[m, x]
                    if sec.blk4[Pm] == A:
                        top[xm] = (1.0 if m < x else -1.0) * sec.block4(r4, A)[sec.pos4[Pm]]
                bot = Xv5.get((x, mu))
                if bot is None:
                    bot = np.zeros((nV[mu], sec5.nA4[A]), dtype=dtype)
                top14[(x, mu)] = (top, bot)
        for M in sec5.blocks5:
            out = sec5.block5(s5, M)
            for r, I in enumerate(sec5.rows5[M]):
                t = [int(v) for v in lay.o3[I]]
                kI = lay.kO3[I]
                if sec5.n23[M]:
                    X23 = self.buf('x23', int(sec5.n23[M]))
                    for q in range(lay.n2):
                        o = sec5.off23[M, q]
                        if o < 0:
                            continue
                        nc, nq = int(nK[M - q]), int(nQ[q])
                        Xq = X23[o:o + nq * nc].reshape(nq, nc)
                        Ls, Rs = [], []          # [K, nq] and [K, nc] with the signs in L
                        for xi, (yi, zi), sg in _SINGLES:
                            x, y, z = t[xi], t[yi], t[zi]
                            Pyz = sec.pair[y, z]
                            # D1: - Wv(ab, f, x) r4(yz; f cde)
                            ph = q - ko[x]
                            if want('D1') and self.vkc(ph):
                                b = self.x4(X4r, Pyz, ph)
                                if b is not None and b.shape[1] == nc:
                                    Ls.append(-sg * eng.WvAt[x][cls(ph), qs(q)])
                                    Rs.append(b)
                            # D2: t2(xm; ab) XoR5(m; yz; cde)
                            mu = q - ko[x]
                            if want('D2') and self.okc(mu):
                                b = self.xor5_rows(XoR5, lay.o0[mu], nO[mu], Pyz)
                                if b is not None and b.shape[1] == nc:
                                    Ls.append(sg * t2_xq[x, ocls(mu), qs(q)])
                                    Rs.append(b)
                            # D3: r3(x; g de) Xv2(g; yz; abc), the pair (de) of class q
                            ph = s + ko[x] - q
                            if want('D3') and self.vkc(ph):
                                b = eng.x3(X3r, x, ph)
                                o2 = lay.r13off[Pyz, ph]
                                if b is not None and o2 >= 0 and lay.kO2[Pyz] + ph == M - q:
                                    o2 += nO[ph] * nc
                                    Ls.append(sg * b)
                                    Rs.append(g5.Xv2[o2:o2 + nV[ph] * nc].reshape(nV[ph], nc))
                            # D7: t3(x mn; abc) Y4(mn; yz; de), (mn) of class c
                            c = M - q - ko[x]
                            if want('D7') and (Pyz, c) in Y4 and lay.t3poff[x, c] >= 0:
                                o3 = lay.t3poff[x, c]
                                Ls.append(sg * Y4[(Pyz, c)])
                                Rs.append(g.T3P[o3:o3 + lay.nO2[c] * nc].reshape(lay.nO2[c], nc))
                            # D8: t3(yz m; abc) XoRp(m; x; de)
                            mu = M - q - lay.kO2[Pyz]
                            if want('D8') and self.okc(mu) and lay.r13off[Pyz, mu] >= 0:
                                o2 = lay.r13off[Pyz, mu]
                                Ls.append(sg * XoRp[ocls(mu), x, qs(q)])
                                Rs.append(g.R13[o2:o2 + nO[mu] * nc].reshape(nO[mu], nc))
                            # D4: t3(yz m; f ab) Z4(mf; x; cde), the pair (yz) on t3
                            if want('D4'):
                                for m in range(no):
                                    sJ = o12.psg[m, Pyz]
                                    if not sJ:
                                        continue
                                    J = int(o12.pos[m, Pyz])
                                    kJ = lay.kO3[J]
                                    ph = kJ - q
                                    if not self.vkc(ph) or lay.t12off[kJ, ph] < 0:
                                        continue
                                    key = (x, ko[m], ph)
                                    if key not in Z4:
                                        continue
                                    zb = Z4[key][m - lay.o0[ko[m]]]
                                    if zb.shape[1] != nc:
                                        continue
                                    t0 = lay.t12off[kJ, ph]
                                    nJk = int(lay.nJ[kJ])
                                    Tg = g.T12[t0:t0 + nV[ph] * nJk * nq].reshape(nV[ph], nJk, nq)[:, lay.posO3[J]]
                                    Ls.append(sg * sJ * Tg)
                                    Rs.append(zb)
                        # D5: t3(ijk; f ab) XvR5(f; cde)
                        ph = kI - q
                        if want('D5') and self.vkc(ph) and lay.t12off[kI, ph] >= 0 and (s + ph) in XvR5:
                            t0 = lay.t12off[kI, ph]
                            nJk = int(lay.nJ[kI])
                            zb = XvR5[s + ph]
                            if zb.shape[1] == nc:
                                Ls.append(g.T12[t0:t0 + nV[ph] * nJk * nq].reshape(nV[ph], nJk, nq)[:, lay.posO3[I]])
                                Rs.append(zb)
                        # D6: Xo3(m; ijk; ab) r3(m; cde)
                        K3 = M - q
                        mu = K3 - s
                        if want('D6') and self.okc(mu) and 0 <= K3 < lay.n3 and len(sec.rows3[K3]):
                            Ls.append(g5.Xo3[I, ocls(mu), qs(q)])
                            Rs.append(sec.block3(r3, K3))
                        if not Ls:
                            Xq[...] = 0
                            continue
                        with timer('dea5.rows.stack'):
                            k_ = sum(len(a) for a in Ls)
                            Lc = np.concatenate(Ls, axis=0, out=self.buf('lc', k_ * nq).reshape(k_, nq))
                            Rc = self._stack(Rs, self.buf('rc', k_ * nc).reshape(k_, nc))
                        with timer('dea5.rows.gemm'):
                            gemm(Lc.T, Rc, Xq)
                    with timer('dea5.rows.reduce'):
                        sec5.x23(1, out, M, sec5.ent_row(M, r, '23'), X23)
                if sec5.n14[M]:
                    X14 = self.buf('x14', int(sec5.n14[M]))
                    for p in range(ncls):
                        o = sec5.off14[M, p]
                        if o < 0:
                            continue
                        nA = int(sec5.nA4[M - p])
                        Xp = X14[o:o + nV[p] * nA].reshape(nV[p], nA)
                        Ls, Rs = [], []          # [nV[p], K] and [K, nA]
                        if want('E1') or want('E1x'):
                            for zi, (xi, yi), sg in _SINGLES:
                                z, Pxy = t[zi], sec.pair[t[xi], t[yi]]
                                mu = lay.kO2[Pxy] - p
                                if (z, mu) in top14 and top14[(z, mu)][0].shape[1] == nA:
                                    m0 = lay.mix0[mu]
                                    L = sg * g.la13[Pxy, cls(p), m0:m0 + nO[mu] + nV[mu]]
                                    if not want('E1'):
                                        L[:, :nO[mu]] = 0
                                    if not want('E1x'):
                                        L[:, nO[mu]:] = 0
                                    Ls.append(L)
                                    Rs.extend(top14[(z, mu)])
                        c = M - p - s
                        if want('E2') and 0 <= c < lay.n2 and lay.nO2[c] and sec.blk4[lay.O2cls[c][0]] == M - p:
                            Ls.append(g5.W2[I, cls(p)][:, lay.O2cls[c]])
                            Rs.append(sec.block4(r4, M - p))
                        if not Ls:
                            Xp[...] = 0
                            continue
                        with timer('dea5.rows.stack'):
                            k_ = sum(a.shape[1] for a in Ls)
                            Lc = np.concatenate(Ls, axis=1, out=self.buf('lc', nV[p] * k_).reshape(nV[p], k_))
                            Rc = self._stack(Rs, self.buf('rc', k_ * nA).reshape(k_, nA))
                        with timer('dea5.rows.gemm'):
                            gemm(Lc, Rc, Xp)
                    with timer('dea5.rows.reduce'):
                        sec5.x14(1, out, M, sec5.ent_row(M, r, '14'), X14)


# ------------------------------------------------------------------ the problem of the DEA driver

def memory5(sec5, isz, batch_bytes=None, ground=True, xv_disk=False):
    '''Bytes that ProblemSector5 holds besides the vectors of the Davidson
    subspace, and the most a sigma vector allocates on top: (fixed, per
    sigma), array by array (the rows to 4p2h: dea_sector.memory).'''
    bb = batch_bytes or BATCH_BYTES
    sec, lay, s = sec5.sec, sec5.lay, sec5.s
    no, nv, ncls, ko = lay.no, lay.nv, lay.ncls, lay.ko
    nO, nV, nQ, nK, nO2 = lay.nO, lay.nV, lay.nQ, lay.nK, lay.nO2
    nP, nI3, nQt = len(lay.o2), len(lay.o3), len(lay.kQ)
    nA4 = sec5.nA4
    fixed4, per4 = D.memory(sec, 4, isz, bb, ground=ground, xv_disk=xv_disk)
    fixed4, per4 = fixed4 // isz, per4 // isz
    # Ground5
    wq = int((nQ ** 2).sum())
    g5 = (wq + int((lay.nJ ** 2).sum()) + nI3 * no * nQt + nI3 * nv * nP + nP * no * no * nv * nv
          + (0 if xv_disk else int(lay.r13size)) + no * no * nv * nv)
    g5b = nI3 * nv * nQt + nP * no * no * nQt + 2 * no * nv * nv + nv * no * nv
    nA5 = int(sum(sec5.nA5[M] for M in sec5.blocks5))
    tables = (10 * nA5 + 4 * comb(nv, 4) + 8 * int(nA4.sum())) // isz + 2 * nA5 * 8 // isz + nA5
    fixed = fixed4 + g5 + tables + (g5b if ground else 0)
    # per sigma
    size5 = sec5.size5
    x4 = int(sum(sec.n13[sec.blk4[P_]] for P_ in range(nP) if sec.blk4[P_] >= 0))
    z4 = 0
    keys = set()
    for k in range(no):
        for ln in range(no):
            if ln == k:
                continue
            M4 = sec.blk4[sec.pair[ln, k]]
            if M4 < 0:
                continue
            for ph in range(ncls):
                if sec.off13[M4, ph] < 0:
                    continue
                for mu in range(ncls):
                    fe = mu + ko[ln] - ph
                    if nO[mu] and 0 <= fe < ncls and nV[fe]:
                        keys.add((k, mu, fe, M4 - ph))
    z4 = sum(int(nO[mu] * nV[fe] * nK[K]) for _, mu, fe, K in keys)
    xvr = sum(int(nV[K - s] * nK[K]) for K in range(lay.n3) if nK[K] and 0 <= K - s < ncls)
    n12 = []
    for K in range(lay.n3):
        n12.append(sum(int(nV[p] * nQ[K - p]) for p in range(ncls) if nV[p] and 0 <= K - p < lay.n2))
    xr = 0
    for P_ in range(nP):
        for m in range(no):
            K = s + lay.kO2[P_] - ko[m]
            if 0 <= K < lay.n3:
                xr += int(nK[K])
    y4 = sum(int(nO2[c] * nQ[sec.blk4[P_] - c]) for P_ in range(nP) for c in range(lay.n2)
             if sec.blk4[P_] >= 0 and nO2[c] and 0 <= sec.blk4[P_] - c < lay.n2)
    mO, mV = int(nO.max()), int(nV.max())
    pair = (no * nv * nQt + 3 * no * no * nv * nv + no * no * nQt + no ** 3 * nv + no * no * nv * nv + no * nv ** 3
            + xr + y4 + int(sec.n22.max(initial=0)) + mO * max(n12 + [0]) + mO * mV * nv * nv)
    xv5 = sum(int(nV[ph] * nA4[s + ko[k] + ph]) for k in range(no) for ph in range(ncls)
              if nV[ph] and 0 <= s + ko[k] + ph < lay.n4)
    xv5 += mV * int(sec5.n13.max(initial=0))
    pp = 0
    for M in sec5.blocks5:
        n23 = int(sec5.n23[M])
        if n23:
            nb = min(len(sec5.rows5[M]), max(1, bb // (isz * n23)))
            pp = max(pp, 2 * nb * n23)
    nW = no * nv
    ring = 0
    for P_ in range(nP):
        u, w = (int(x) for x in lay.o2[P_])
        sizes = []
        for d in range(-(ncls - 1), ncls):
            A = s + lay.kO2[P_] + d
            if not (0 <= A < lay.n4 and nA4[A]):
                continue
            nc = sum(int(nV[ko[m] - d]) for m in range(no) if m not in (u, w) and 0 <= ko[m] - d < ncls)
            if not nc:
                continue
            nr = sum(int(nV[ko[i] - d]) for i in range(no) if i not in (u, w) and 0 <= ko[i] - d < ncls) + (d == 0)
            for c in (ko[u] + d, ko[w] + d):
                if 0 <= c < ncls:
                    nr += int(nO[c] + nV[c])
            sizes.append(((nc + nr) * int(nA4[A]), nc * nr))
        tot = wmax = 0
        for sz, wz in sizes:
            if tot and (tot + sz) * isz > bb:
                ring = max(ring, tot + wmax)
                tot = 0
            tot, wmax = tot + sz, max(wmax, wz)
        ring = max(ring, tot + wmax)
    ring += (nW + 1 + 2 * no + 2 * nv) * nW
    top = sum(int(nO[mu] * nA4[s + ko[x] + mu]) for x in range(no) for mu in range(ncls)
              if 0 <= s + ko[x] + mu < lay.n4)
    kmax = 3 * (2 * mV + mO + int(nO2.max()) + no * mV) + mV + mO
    rows = (int(sec5.n23.max(initial=0)) + int(sec5.n14.max(initial=0)) + top
            + 2 * kmax * (int(nQ.max()) + int(nK.max())) + 2 * 3 * (mO + mV) * int(nA4.max()) * 2)
    per = per4 + 2 * size5 + 2 * x4 + z4 + 2 * xvr + pair + xv5 + max(pp, ring, rows)
    return isz * fixed, isz * per


class ProblemSector5:
    '''EOM-DEA with R to 5p3h on the sector (module docstring): parts r2,
    r3, r4, r5 (flat, in the blocks of Sector and Sector5), the parity
    applied block by block, the preconditioner and the guess - for kernel().
    ``eris``: integrals with labels in the order of the labels, ``t3`` flat
    in the blocks of the engine of ``eris`` (the T of CCSDT).'''

    def __init__(self, eris, t1, t2, t3, target=0, parity=None, target_parity=0, mem_mb=2000.0,
                 batch_bytes=None, verbose=False):
        from zccsdt.cc import ccsdt_opt, t3pack, t4engine
        t0 = time.perf_counter()
        if t3pack.classes_of(eris) is None or S.layout_for(eris) is None:
            raise ValueError('EOM-DEA on the sector: labels without a modulus in use, the spinors in their order')
        self.rank = 5
        no, n = eris.nocc, np.asarray(eris.fock).shape[0]
        nv = n - no
        self.no, self.nv = no, nv
        self.dtype = dtype = np.dtype(np.result_type(t1, t2, eris.oovv, eris.fock))
        self.lay = lay = S.layout_for(eris)
        sh = D.shift_of(eris, target)
        if sh is None or not 0 <= sh < lay.n2:
            raise ValueError('EOM-DEA: no pair of virtual spinors with the target %d' % target)
        self.sec = sec = D.Sector(lay, sh)
        self.sec5 = sec5 = Sector5(sec)
        isz = dtype.itemsize
        # the plan: the six-index Xv (and Xv2) in scratch files when the rest and four vectors do not fit
        # with them; the batches as large as the budget leaves them, at most BATCH_BYTES
        vec = (sec.n2 + sec.size3 + sec.size4 + sec5.size5 // 2) * isz
        fixed, per = memory5(sec5, isz, 1)
        self.xv_disk = fixed + per + 4 * vec > mem_mb * 1e6
        r13, self.scr = None, None
        if self.xv_disk:
            from zccsdt.scratch import Scratch
            self.scr = Scratch(prefix='zccsdt_dea5_', what='the six-index intermediates Xv, Xv2 of EOM-DEA')
            r13 = self.scr.create('xv', max(int(lay.r13size), 1), dtype)
            fixed, per = memory5(sec5, isz, 1, xv_disk=True)
        bb = batch_bytes
        if bb is None:
            bb = int(min(BATCH_BYTES, max(1 << 24, (mem_mb * 1e6 - fixed - per) / 3)))
        self.batch_bytes = bb
        T3p = ccsdt_opt.packed_t3(eris, t3)
        T3 = np.ascontiguousarray(lay.t3_from_packed(T3p), dtype=dtype)
        ints = t4engine.Dressed.from_eris(eris, np.asarray(t1, dtype=dtype))
        self.g = D.Ground(lay, np.asarray(t2), T3, ints, xv=True, r13=r13)
        xv2 = self.scr.create('xv2', max(int(lay.r13size), 1), dtype) if self.xv_disk else None
        self.g5 = Ground5(sec5, self.g, T3p, ints.ovvo, xv2=xv2)
        del ints, T3, T3p
        self.mem = memory5(sec5, isz, bb, ground=False, xv_disk=self.xv_disk)
        self.eng4 = D.Engine(sec, self.g, 4, bb)
        self.eng = Engine5(sec5, self.g, self.g5, self.eng4, bb)
        # the elements: denominators, parity
        ev = np.asarray(eris.mo_energy).real
        self.eo, self.ev = ev[:no], ev[no:]
        par = None if parity is None else np.asarray(parity, dtype=np.int64)
        self.par, self.tpar = par, int(target_parity) % 2
        aq, bq = lay.va, lay.vb
        qs = slice(lay.qoff[sh], lay.qoff[sh + 1])
        evv = self.ev
        self.cden = [evv[aq[qs]] + evv[bq[qs]], {}, {}, {}]
        m2 = np.ones(sec.n2, bool)
        if par is not None:
            m2 = (par[no:][aq[qs]] + par[no:][bq[qs]] - self.tpar) % 2 == 0
        masks = [m2, np.ones(sec.size3, bool), np.ones(sec.size4, bool)]
        pv, po = (par[no:], par[:no]) if par is not None else (None, None)
        for K in sec.blocks3:
            tri = lay.tri[K].astype(np.int64)
            self.cden[1][K] = evv[tri].sum(axis=1)
            if par is not None:
                mk = (pv[tri].sum(axis=1)[None, :] - po[sec.rows3[K]][:, None] - self.tpar) % 2 == 0
                masks[1][sec.off3[K]:sec.off3[K + 1]] = mk.ravel()
        for M in sec.blocks4:
            qd = sec.quads[M].astype(np.int64)
            self.cden[2][M] = evv[qd].sum(axis=1)
            if par is not None:
                pr = lay.o2[sec.rows4[M]]
                mk = (pv[qd].sum(axis=1)[None, :] - po[pr].sum(axis=1)[:, None] - self.tpar) % 2 == 0
                masks[2][sec.off4[M]:sec.off4[M + 1]] = mk.ravel()
        # r5: the sums of e over the quintuples, and its parity g only, stored: per block the rows of
        # parity 0 and then those of parity 1, each on the quintuples of the parity that goes with it
        self.r5cols, self.r5rows, self.r5off, off = {}, {}, {}, 0
        for M in sec5.blocks5:
            qn = sec5.quints[M]
            e = np.zeros(len(qn))
            pc = np.zeros(len(qn), dtype=np.int64)
            for x in range(5):
                e += evv[qn[:, x]]
                if par is not None:
                    pc += pv[qn[:, x]]
            self.cden[3][M] = e
            nr = len(sec5.rows5[M])
            if par is None:
                rows = [np.arange(nr), np.zeros(0, dtype=np.int64)]
                cols = [np.arange(len(qn)), np.zeros(0, dtype=np.int64)]
            else:
                pr = po[lay.o3[sec5.rows5[M]]].sum(axis=1) % 2
                rows = [np.flatnonzero(pr == b) for b in (0, 1)]
                cols = [np.flatnonzero((pc - b - self.tpar) % 2 == 0) for b in (0, 1)]
            self.r5rows[M] = rows
            self.r5cols[M] = [c.astype(np.int64) for c in cols]
            self.r5off[M] = off
            off += sum(len(r) * len(c) for r, c in zip(rows, cols))
        self.size5c = off
        self.masks = masks
        self.sizes = [sec.n2, sec.size3, sec.size4, self.size5c]
        self.shapes = [(n_,) for n_ in self.sizes]
        self.nsigma, self.seconds = 0, 0.0
        self.prepare_seconds = time.perf_counter() - t0
        if verbose:
            print('  sector: 2p %d, 3p1h %d, 4p2h %d, 5p3h %d (%d of parity) elements; memory %.2f GB fixed + '
                  '%.2f GB per sigma vector; batches of %.0f MB%s; prepared in %.1f s'
                  % (sec.n2, sec.size3, sec.size4, sec5.size5, self.size5c, self.mem[0] / 1e9, self.mem[1] / 1e9,
                     bb / 1e6, '; Xv, Xv2 in scratch files' if self.xv_disk else '', self.prepare_seconds),
                  flush=True)

    def counts(self):
        return tuple([int(m.sum()) for m in self.masks] + [self.size5c])

    def _parts5(self, chunk=1 << 22):
        """(block M, rows of the block, columns, slice of the stored 5p3h part) in pieces of
        at most about ``chunk`` elements."""
        sec5 = self.sec5
        for M in sec5.blocks5:
            off = self.r5off[M]
            for rows, cols in zip(self.r5rows[M], self.r5cols[M]):
                n = len(cols)
                if not n or not len(rows):
                    continue
                step = max(1, chunk // n)
                for r0 in range(0, len(rows), step):
                    r = rows[r0:r0 + step]
                    yield M, r, cols, slice(off + r0 * n, off + (r0 + len(r)) * n)
                off += len(rows) * n

    def unpack5(self, c, out=None):
        """The 5p3h part in the blocks of Sector5 (zeros where the parity does not go) of the stored one."""
        out = np.zeros(self.sec5.size5, dtype=self.dtype) if out is None else P.zero(out)
        for M, r, cols, sl in self._parts5():
            blk = self.sec5.block5(out, M)
            blk[np.ix_(r, cols)] = np.asarray(c[sl]).reshape(len(r), len(cols))
        return out

    def pack5(self, full, out):
        for M, r, cols, sl in self._parts5():
            out[sl] = self.sec5.block5(full, M)[np.ix_(r, cols)].ravel()
        return out

    def project(self, x):
        for p, m in zip(x[:3], self.masks):
            if p.size:
                p *= m
        return x

    def new(self):
        return [np.zeros(s, dtype=self.dtype) for s in self.sizes]

    def sigma(self, x, y, only=None):
        t0 = time.perf_counter()
        with timer('dea5.unpack'):
            x5 = self.unpack5(x[3], self.eng.buf('x5', self.sec5.size5))
        s2, s3, s4, s5 = self.eng.sigma(x[0], x[1], x[2], x5, only=only)
        del x5
        y[0][:] = s2
        y[1][:] = s3
        y[2][:] = s4
        with timer('dea5.pack'):
            self.pack5(s5, y[3])
        del s2, s3, s4, s5
        self.project(y)
        self.nsigma += 1
        self.seconds += time.perf_counter() - t0
        return y

    def _den_blocks(self, part, chunk=1 << 21):
        sec = self.sec
        if part == 0:
            yield slice(0, sec.n2), self.cden[0][None, :]
            return
        if part == 3:
            for M, r, cols, sl in self._parts5(chunk):
                rows = self.eo[self.lay.o3[self.sec5.rows5[M][r]]].sum(axis=1)
                yield sl, self.cden[3][M][cols][None, :] - rows[:, None]
            return
        if part == 1:
            blocks = [(sec.off3[K], self.cden[1][K], self.eo[sec.rows3[K]]) for K in sec.blocks3]
        else:
            blocks = [(sec.off4[M], self.cden[2][M], self.eo[self.lay.o2[sec.rows4[M]]].sum(axis=1))
                      for M in sec.blocks4]
        for off, cols, rows in blocks:
            n = len(cols)
            step = max(1, chunk // max(n, 1))
            for r0 in range(0, len(rows), step):
                r1 = min(len(rows), r0 + step)
                yield slice(off + r0 * n, off + r1 * n), cols[None, :] - rows[r0:r1, None]

    def precond(self, x, omega, floor=1e-3):
        for part, p in enumerate(x):
            for sl, d in self._den_blocks(part):
                den = omega - d
                den[abs(den) < floor] = floor
                p[sl] /= den.ravel()
        return x

    def guesses(self, n):
        d = self.cden[0].copy()
        d[~self.masks[0]] = np.inf
        out = []
        for k in np.argsort(d, kind='stable')[:n]:
            if np.isfinite(d[k]):
                v = self.new()
                v[0][k] = 1.0
                out.append(v)
        return out

    def dense_positions(self):
        """The places of the stored elements in the dense packed arrays (the 5p3h part as stored)."""
        from math import comb as _comb
        n5 = _comb(self.nv, 5)
        d5 = [np.zeros(0, dtype=np.int64)]
        for M, r, cols, sl in self._parts5():
            d5.append((self.sec5.rows5[M][r][:, None] * n5 + self.sec5.lex5(M)[cols][None, :]).ravel())
        return self.sec.dense_positions() + [np.concatenate(d5)]


def kernel(pb, nroots=1, nguess=None, conv_tol=1e-8, max_space=None, max_cycle=80, mem_mb=2000.0, verbose=False,
           stdout=None, mapped=None):
    '''The ``nroots`` lowest roots of a ProblemSector5: (omega = E(N+2) -
    E_CC(N), Solved of zccsdt.cc.eom.davidson).  The 5p3h part of the
    subspace in a scratch file when it does not fit in ``mem_mb`` (or
    ``mapped``).'''
    import sys
    from zccsdt.cc import eom
    out = stdout or sys.stdout
    gs_n = nguess or nroots + 2
    max_space = max_space or max(8 + 4 * nroots, gs_n + nroots)
    vec = sum(pb.sizes) * pb.dtype.itemsize
    if mapped is None:
        mapped = sum(pb.mem) + (2 * max_space + nroots + 2) * vec > mem_mb * 1e6
    store = eom.Store(pb.sizes, pb.dtype, mapped=mapped)
    gs = pb.guesses(gs_n)
    if verbose:
        print('EOM-DEA with 5p3h on the sector: %d roots, vectors of %s elements (%s in the masks)%s'
              % (nroots, ' + '.join(str(s) for s in pb.sizes), ' + '.join(str(c) for c in pb.counts()),
                 '; the 5p3h part of the subspace in a scratch file' if mapped else ''), file=out, flush=True)

    def log(it, n, w, res, conv, sec):
        if verbose:
            print('iter %3d  %3d vectors  %s  %.2f s' % (it, n, '  '.join(
                '%.11f %.1e%s' % (x.real, r, '*' if c else ' ') for x, r, c in zip(w, res, conv)), sec),
                file=out, flush=True)
    try:
        res = eom.davidson(pb.sigma, pb.precond, gs, store, nroots, conv_tol, max(max_space, len(gs) + nroots),
                           max_cycle, log=log)
        if mapped:
            res.vectors = [[np.array(p) for p in v] for v in res.vectors]
    finally:
        store.close()
    order = np.argsort(res.e.real, kind='stable')
    return res.e.real[order], res
