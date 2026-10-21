'''EOM-DEA-CCSDT on the sector itself: the sigma vector of R = 2p + 3p1h +
4p2h on the short arrays r2[(a<b)], r3[i, (a<b<c)], r4[(i<j), (a<b<c<d)]
in the blocks of the symmetry labels, without padding the ground state
(docs/eom.md, section 11.9).

The terms are those of the ghost route (zccsdt.cc.dea) that survive on the
rows with both ghosts (zccsdt.cc.dea_ref derives them from the table of the
amplitude equations); here they are factorized as the T4 equation of
zccsdt.cc.t4engine is, with T = T2 + T3 of CCSDT (T1-dressed integrals) and
the sector vector in the place of one amplitude:

  4p2h, sigma4[ij; abcd] (i < j real, the ghosts implied):
    pp      P(ab/cd) sum_{e<f} Wq(ab,ef) r4(ij; efcd)                Wq = Wpp + F_vv on pairs / 3
    hh      sum_{m<n} Wh(mn,ij) r4(mn; abcd)                          Wh = Whh - F_oo on pairs
    ring    P(ij) P(a/bcd) sum_{me} Wr(m,a,e,i) r4(mj; ebcd)
    22      P(ab/cd) { P(ij) [ - sum_e Wv(ab,e,i) r3(j; cde) + sum_m t(im,ab) XoR(m; j; cd) ]
                       + sum_{nf} ZR(nf; ab) t(nij; fcd) }
    13      P(a/bcd) { sum_m Wo(m,a,ij) r3(m; bcd) + sum_e t(ij,ae) XvR(e; bcd)
                       + sum_m WoR(m,a) t(mij; bcd) + sum_e r2(ae) Xv(e; ij; bcd)
                       - P(ij) sum_{m<n} YR(mn; j; a) t(mni; bcd) }
  with the intermediates of the sector (linear in R)
    XoR(m; j; cd) = sum_{ne} <mn||ej> r3(n; ecd) + 1/2 sum_n WhR(mn) t(nj, cd)
                    - P(cd) sum_{e<f} <dm||ef> r3(j; efc) + sum_n sum_{e<f} <mn||ef> r4(nj; efcd)
    XvR(e; bcd)   = P(d/bc) [ sum_{mf} <dm||ef> r3(m; fbc) + 1/2 sum_f Wpp(bc,ef) r2(fd) ]
                    + 1/2 sum_{mnf} <mn||ef> r4(mn; fbcd)
    WoR(m, d)     = - sum_{e<f} <dm||ef> r2(ef) - sum_e f_me r2(de) - sum_n YR(mn; n; d)
    YR(mn; j; d)  = sum_{e<f} <mn||ef> r3(j; efd),   WhR(mn) = sum_{e<f} <mn||ef> r2(ef)
    ZR(nf; ab)    = sum_{me} <mn||ef> r3(m; eab)
  and those of the ground state (Wpp, Whh, Wr, F; Wv, Wo with their T3
  terms; the six-index Xv(e; kl; bcd) without T4) - all of
  zccsdt.cc.t4engine.

  With the T of CCSDTQ two terms more:
    T4      sum_{m<n} WhR(mn) t4(mnij; abcd) + P(a/bcd) sum_{m<n} sum_f [sum_e r2(ae) <mn||ef>] t4(mnij; fbcd)

  The terms of the 4p2h row of dea_ref (T1-dressed f, v; coefficients of
  the table, A over the external indices) and where they are:
    1/4 v_cdej r3_iabe, 1/8 v_mnej t2_mnab r3_iecd, -1/2 v_dmef t2_imae r3_jfbc,
      -1/4 f_me t2_imab r3_jecd, 1/8 v_mnef r3_ieab t3_mnjfcd        22: Wv r3
    1/4 v_mnej t2_imab r3_necd, -1/4 v_dmef t2_imab r3_jefc, -1/16 v_mnef r2_ef t2_miab t2_njcd,
      1/8 v_mnef t2_imab r4_njefcd                                    22: t2 XoR
    1/16 v_mnef r3_meab t3_nijfcd, 1/16 v_mnef t3_mijeab r3_nfcd    22: ZR t3 (the two together)
    -1/12 v_mdij r3_mabc, -1/6 v_mnej t2_imae r3_nbcd, 1/24 v_dmef t2_ijef r3_mabc,
      -1/12 f_me t2_ijae r3_mbcd, 1/24 v_mnef r3_mabc t3_nijefd       13: Wo r3
    1/8 v_bcef t2_ijae r2_fd, -1/16 v_mnef t2_mnab t2_ijec r2_fd, 1/4 v_dmef t2_ijae r3_mfbc,
      1/24 v_mnef t2_ijae r4_mnfbcd                                   13: t2 XvR
    -1/12 f_me r2_ae t3_mijbcd, 1/24 v_dmef r2_ef t3_mijabc, 1/24 v_mnef t3_mijabc r3_nefd
                                                                      13: WoR t3
    1/8 v_bcef r2_ae t2_ijfd, 1/2 v_mbei r2_ae t2_mjcd, -1/12 v_mnej r2_ae t3_mnibcd,
      1/4 v_dmef r2_ae t3_mijfbc, -1/16 v_mnef t2_mnab r2_ec t2_ijfd,
      1/2 v_mnef t2_imae t2_njbc r2_fd                                13: r2 Xv
    1/24 v_mnef t3_mniabc r3_jefd                                     13: YR t3
    1/12 f_de r4, 1/16 v_cdef r4, 1/32 v_mnef t2_mnab r4_ijefcd, 1/24 v_mnef t2_mnae r4_ijfbcd
                                                                      pp
    -1/24 f_mj r4, 1/96 v_mnij r4, 1/192 v_mnef t2_ijef r4_mnabcd, 1/48 v_mnef t2_imef r4_njabcd
                                                                      hh
    1/6 v_mdej r4_imabce, 1/6 v_mnef t2_imae r4_njfbcd                ring

  3p1h, sigma3[i; abc]:
    P(a/bc) [ sum_{e<f} Wq3(bc,ef) r3(i; aef) + sum_e Wv(bc,e,i) r2(ae) ]  Wq3 = Wpp + F_vv on pairs / 2
    + P(c/ab) [ sum_{me} Wr(m,c,e,i) r3(m; abe) + sum_m t(im,ab) (sum_n YR(mn; n; c) + hR(c,m)) ]
    - sum_m Fo(m,i) r3(m; abc) + sum_{m<n} WhR(mn) t(mni; abc)
    + sum_{me} f_me r4(im; abce) + 1/2 sum_{mne} <mn||ei> r4(mn; abce) + P(c/ab) sum_m sum_{e<f} <cm||ef> r4(im; abef)
    with hR(c, m) = sum_{e<f} <cm||ef> r2(ef)
  2p, sigma2[ab]:
    sum_{e<f} Wq2(ab,ef) r2(ef) + sum_{me} f_me r3(m; eab) + P(ab) sum_m sum_{e<f} <bm||ef> r3(m; aef)
    + sum_{m<n, e<f} <mn||ef> r4(mn; efab)                              Wq2 = Wpp + F_vv on pairs

Storage (Sector): the labels of zccsdt.cc.t3pack.classes_of(); ``shift``
the class of the pairs of two attached electrons (the target 2 Omega in
classes).  r2: the pairs of class shift, in the order of the classes; r3:
one block per class K of the virtual triples, its rows the occupied i of
class K - shift; r4: one block per class M of the virtual quadruples (a
list, zc_t4s_quads), its rows the occupied pairs of class M - shift.  The
products go through the kernels of the label-block T4 engine
(csrc/t4sym.c: expansion of a block into the splits (ab|cd), (a|bcd),
(a|bc) and the antisymmetrizing reduction back).

The levels: R to 3p1h (rank 3) or 4p2h (rank 4); T of CCSD (t3 None), of
CCSDT, or of CCSDTQ (t4 in the blocks of zccsdt.cc.t4engine_sym).  The
lowest-order rows (CC3, CC4) and the folded 5p3h operators (dea5) are on
the ghost route.
'''

import ctypes
import time
from math import comb

import numpy as np

from zccsdt.blas import gemm
from zccsdt.cc import t4engine, t4engine_sym as S, t4pack as P
from zccsdt.timing import timer

_lib = S._lib
_I64, _INT = ctypes.c_int64, ctypes.c_int
BATCH_BYTES = 1 << 30


def _p(a):
    return ctypes.c_void_p(a.ctypes.data)


def _c(x, dtype):
    return np.ascontiguousarray(x, dtype=dtype)


def _E(sub, *ops):
    return np.einsum(sub, *ops, optimize=True)


def _batches(n, step):
    step = int(max(1, step))
    return [(x, min(n, x + step)) for x in range(0, n, step)]


# ------------------------------------------------------------------ the blocks of the sector

class Sector:
    '''The blocks of r2, r3, r4 of the class ``shift`` for the label-block
    Layout ``lay`` of the ground state (zccsdt.cc.t4engine_sym.Layout).'''

    def __init__(self, lay, shift):
        self.lay = lay
        self.s = s = int(shift)
        no, nv, ncls = lay.no, lay.nv, lay.ncls
        self.no, self.nv, self.ncls = no, nv, ncls
        ko, nQ, nK = lay.ko, lay.nQ, lay.nK
        # 2p
        self.n2 = int(nQ[s]) if 0 <= s < lay.n2 else 0
        # 3p1h: block K = class of the triple, rows i of class K - s
        self.rows3, self.off3 = [], [0]
        self.blk3, self.pos3 = np.full(no, -1, dtype=np.int64), np.full(no, -1, dtype=np.int64)
        for K in range(lay.n3):
            c = K - s
            r = np.flatnonzero(ko == c) if 0 <= c < ncls and nK[K] else np.zeros(0, dtype=np.int64)
            self.rows3.append(r)
            self.blk3[r], self.pos3[r] = K, np.arange(len(r))
            self.off3.append(self.off3[-1] + len(r) * int(nK[K]))
        self.size3 = self.off3[-1]
        # 4p2h: block M = class of the quadruple, rows the pairs of class M - s
        self.rows4, self.off4, self.quads, self.nA = [], [0], [], np.zeros(lay.n4, dtype=np.int64)
        nP = len(lay.o2)
        self.blk4, self.pos4 = np.full(nP, -1, dtype=np.int64), np.full(nP, -1, dtype=np.int64)
        self.pair = -np.ones((no, no), dtype=np.int64)          # (i, j) -> the pair index of {i, j}
        for P_, (i, j) in enumerate(lay.o2):
            self.pair[i, j] = self.pair[j, i] = P_
        for M in range(lay.n4):
            c = M - s
            r = lay.O2cls[c] if 0 <= c < lay.n2 else np.zeros(0, dtype=np.int64)
            n = int(_lib.zc_t4s_quads(_INT(nv), _p(lay.kv32), _INT(M), None)) if len(r) else 0
            q = np.empty((n, 4), dtype=np.int16)
            if n:
                _lib.zc_t4s_quads(_INT(nv), _p(lay.kv32), _INT(M), _p(q))
            else:
                r = np.zeros(0, dtype=np.int64)
            self.rows4.append(np.asarray(r, dtype=np.int64))
            self.quads.append(q)
            self.nA[M] = n
            self.blk4[r], self.pos4[r] = M, np.arange(len(r))
            self.off4.append(self.off4[-1] + len(r) * n)
        self.size4 = self.off4[-1]
        self.blocks4 = [M for M in range(lay.n4) if len(self.rows4[M]) and self.nA[M]]
        self.blocks3 = [K for K in range(lay.n3) if len(self.rows3[K])]
        # the blocks of one row: X[(ab), (cd)] by the class of (ab), X[a, (bcd)] by that of a
        n2c = lay.n2
        self.off22 = np.full((lay.n4, n2c), -1, dtype=np.int64)
        self.off13 = np.full((lay.n4, ncls), -1, dtype=np.int64)
        self.n22, self.n13 = np.zeros(lay.n4, dtype=np.int64), np.zeros(lay.n4, dtype=np.int64)
        for M in self.blocks4:
            n = 0
            for q in range(n2c):
                if nQ[q] and 0 <= M - q < n2c and nQ[M - q]:
                    self.off22[M, q] = n
                    n += nQ[q] * nQ[M - q]
            self.n22[M] = n
            n = 0
            for p in range(ncls):
                if lay.nV[p] and 0 <= M - p < lay.n3 and nK[M - p]:
                    self.off13[M, p] = n
                    n += lay.nV[p] * nK[M - p]
            self.n13[M] = n
        # a row of r3 on (a | bc): [a of class p, (bc) of class K - p]
        self.off12 = np.full((lay.n3, ncls), -1, dtype=np.int64)
        self.n12 = np.zeros(lay.n3, dtype=np.int64)
        for K in self.blocks3:
            n = 0
            for p in range(ncls):
                if lay.nV[p] and 0 <= K - p < n2c and nQ[K - p]:
                    self.off12[K, p] = n
                    n += lay.nV[p] * nQ[K - p]
            self.n12[K] = n
        # the expanded rows of r3, one after the other (occupied index order)
        self.x3off = np.zeros(no + 1, dtype=np.int64)
        for i in range(no):
            K = self.blk3[i]
            self.x3off[i + 1] = self.x3off[i] + (self.n12[K] if K >= 0 else 0)
        self.sizes = [self.n2, self.size3, self.size4]

    # ---------------------------------------------------------------- views
    def block3(self, r3, K):
        return r3[self.off3[K]:self.off3[K + 1]].reshape(len(self.rows3[K]), self.lay.nK[K])

    def block4(self, r4, M):
        return r4[self.off4[M]:self.off4[M + 1]].reshape(len(self.rows4[M]), self.nA[M])

    def lex4(self, M):
        '''Positions of the quadruples of block M among all sorted quadruples.'''
        n = self.nv
        y = n - 1 - self.quads[M].astype(np.int64)
        return (comb(n, 4) - 1 - y[:, 0] * (y[:, 0] - 1) * (y[:, 0] - 2) * (y[:, 0] - 3) // 24
                - y[:, 1] * (y[:, 1] - 1) * (y[:, 1] - 2) // 6 - y[:, 2] * (y[:, 2] - 1) // 2 - y[:, 3])

    def dense_positions(self):
        '''The places of the elements in the dense arrays [(a<b)], [i, (abc)],
        [(i<j), (abcd)] (lexicographic tuples).'''
        lay = self.lay
        d2 = np.asarray(lay.Q2cls[self.s], dtype=np.int64) if self.n2 else np.zeros(0, dtype=np.int64)
        n3 = comb(self.nv, 3)
        d3 = [np.zeros(0, dtype=np.int64)]
        for K in self.blocks3:
            d3.append((self.rows3[K][:, None] * n3 + np.asarray(lay.A3cls[K], dtype=np.int64)[None, :]).ravel())
        n4 = comb(self.nv, 4)
        d4 = [np.zeros(0, dtype=np.int64)]
        for M in self.blocks4:
            d4.append((self.rows4[M][:, None] * n4 + self.lex4(M)[None, :]).ravel())
        return [d2, np.concatenate(d3), np.concatenate(d4)]

    # ---------------------------------------------------------------- kernels
    def x22(self, mode, R, M, ent, X):
        row, sgn, xoff, s1, s2 = ent
        lay = self.lay
        getattr(_lib, 'zc_t4s_x22_' + P._SFX[R.dtype])(
            _INT(mode), _p(R), _I64(R.shape[1]), _p(self.quads[M]), _I64(self.nA[M]), _I64(len(row)), _p(row),
            _p(sgn), _p(xoff), _p(s1), _p(s2), _INT(lay.n2), _INT(lay.nv), _p(lay.kv32), _p(lay.posQ), _p(X))

    def x13(self, mode, R, M, ent, X):
        row, sgn, xoff, s1, s2 = ent
        lay = self.lay
        getattr(_lib, 'zc_t4s_x13_' + P._SFX[R.dtype])(
            _INT(mode), _p(R), _I64(R.shape[1]), _p(self.quads[M]), _I64(self.nA[M]), _I64(len(row)), _p(row),
            _p(sgn), _p(xoff), _p(s1), _p(s2), _INT(lay.ncls), _INT(lay.nv), _p(lay.kv32), _p(lay.posV),
            _p(lay.pos3), _p(X))

    def x12(self, mode, R, K, ent, X):
        row, sgn, xoff, s1, s2 = ent
        lay = self.lay
        getattr(_lib, 'zc_t4s_x12_' + P._SFX[R.dtype])(
            _INT(mode), _p(R), _I64(R.shape[1]), _p(lay.tri[K]), _I64(lay.nK[K]), _I64(len(row)), _p(row),
            _p(sgn), _p(xoff), _p(s1), _p(s2), _INT(lay.ncls), _INT(lay.nv), _p(lay.kv32), _p(lay.posV),
            _p(lay.posQ), _p(X))

    def ent_pp(self, M, r0, r1):
        '''Rows r0:r1 of block M as [(ab), (row, (cd))] per class of (ab).'''
        nb, n2c = r1 - r0, self.lay.n2
        ok = self.off22[M] >= 0
        nc = np.where(ok, self.lay.nQ[np.clip(M - np.arange(n2c), 0, n2c - 1)], 0)
        xoff = np.where(ok[None, :], self.off22[M][None, :] * nb + np.arange(nb)[:, None] * nc[None, :], -1)
        return (np.arange(r0, r1, dtype=np.int64), np.ones(nb), np.ascontiguousarray(xoff, dtype=np.int64),
                np.ascontiguousarray(np.broadcast_to(nc * nb, xoff.shape), dtype=np.int64),
                np.ones(xoff.shape, dtype=np.int64))

    def ent_row(self, M, r, kind):
        '''One row of block M on its own: '22' X[(ab), (cd)] or '13' X[a, (bcd)].'''
        lay = self.lay
        if kind == '22':
            n2c = lay.n2
            ok = self.off22[M] >= 0
            nc = np.where(ok, lay.nQ[np.clip(M - np.arange(n2c), 0, n2c - 1)], 0)
            xoff = np.where(ok, self.off22[M], -1)
        else:
            ok = self.off13[M] >= 0
            nc = np.where(ok, lay.nK[np.clip(M - np.arange(self.ncls), 0, lay.n3 - 1)], 0)
            xoff = np.where(ok, self.off13[M], -1)
        return (np.array([r], dtype=np.int64), np.ones(1), np.ascontiguousarray(xoff[None, :], dtype=np.int64),
                np.ascontiguousarray(nc[None, :], dtype=np.int64), np.ones((1, len(nc)), dtype=np.int64))

    def ent_x3(self, K):
        '''The rows of block K of r3 as [a, (bc)] at their places in the
        expanded rows (x3off).'''
        lay = self.lay
        rows = self.rows3[K]
        ok = self.off12[K] >= 0
        nc = np.where(ok, lay.nQ[np.clip(K - np.arange(self.ncls), 0, lay.n2 - 1)], 0)
        xoff = np.where(ok[None, :], self.x3off[rows][:, None] + self.off12[K][None, :], -1)
        return (np.arange(len(rows), dtype=np.int64), np.ones(len(rows)), np.ascontiguousarray(xoff, dtype=np.int64),
                np.ascontiguousarray(np.broadcast_to(nc, xoff.shape), dtype=np.int64),
                np.ones(xoff.shape, dtype=np.int64))


# ------------------------------------------------------------------ the ground state

class Ground:
    '''The intermediates of T (CCSDT) that the sigma vector uses, in the
    blocks of ``lay`` (the same as zccsdt.cc.t4engine_sym.residual makes
    them): F_vv, F_oo, Whh, Wr, Wpp by classes of pairs, Wv and Wo with their
    T3 terms, T3 on its splits, and the six-index Xv(e; kl; bcd) without T4.'''

    def __init__(self, lay, t2, T3, ints, xv=True, r13=None):
        no, nv, ncls, ko = lay.no, lay.nv, lay.ncls, lay.ko
        dtype = np.result_type(T3, t2, ints.oovv, ints.f)
        self.dtype = dtype
        nP, nQt, nI3 = len(lay.o2), len(lay.kQ), len(lay.o3)
        nO, nV, nQ, nK, nJ, o0, v0, qoff = lay.nO, lay.nV, lay.nQ, lay.nK, lay.nJ, lay.o0, lay.v0, lay.qoff
        o12 = lay.o12
        i2, j2 = lay.o2[:, 0], lay.o2[:, 1]
        aq, bq = lay.va, lay.vb

        def c_(x):
            return np.ascontiguousarray(x, dtype=dtype)

        def qs(q):
            return slice(qoff[q], qoff[q + 1])
        T3b = [c_(b) for b in lay.t3_blocks(T3)]
        bufs = {}
        f = np.asarray(ints.f)
        foo, fov, fvv = f[:no, :no], c_(f[:no, no:]), f[no:, no:]
        oovv = c_(ints.oovv)
        oovv_pq = c_(oovv[i2, j2][:, aq, bq])
        oovv_xq = c_(oovv[:, :, aq, bq])
        oovo = np.asarray(ints.oovo)
        oovo_pp = c_(oovo[i2, j2])
        vovv = np.asarray(ints.vovv)
        vovv_q = c_(vovv[:, :, aq, bq])
        t2 = c_(t2)
        t2_pq = c_(t2[i2, j2][:, aq, bq])
        t2_xq = c_(t2[:, :, aq, bq])
        t2_px = c_(t2[i2, j2])
        bufs.update(oovv=oovv, t2xq=t2_xq, t2px=t2_px)
        with timer('dea.ground'):
            self.Fv = Fv = fvv - 0.5 * _E('mnef,mndf->de', oovv, t2)
            self.Fo = Fo = foo + 0.5 * _E('mnef,lnef->ml', oovv, t2)
            self.Whh = Whh = np.asarray(ints.oooo) + 0.5 * _E('mnef,klef->mnkl', oovv, t2)
            self.Wr = Wr = c_(np.asarray(ints.ovvo) + _E('mnef,lndf->mdel', oovv, t2))      # [m, d, e, l]
            Wv = c_(np.asarray(ints.vvvo_p)[lay.qperm])
            gemm(t2_pq.T, oovo_pp.reshape(nP, nv * no), Wv.reshape(nQt, nv * no), beta=1.0)
            x = _E('dmef,lmce->cdfl', vovv, t2)
            Wv += x[aq, bq]
            Wv -= x[bq, aq]
            del x
            Wv += _E('me,lmQ->Qel', fov, t2_xq)
            bufs['wvA'] = c_(Wv.transpose(2, 0, 1))                     # [l, (cd), e]
            del Wv
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
            # T3 on (e | cd) and its terms of Wv and Wo
            T12 = np.zeros(max(lay.t12size, 1), dtype=dtype)
            for k in range(lay.n3):
                if nJ[k] and nK[k]:
                    S._x12(lay, 0, T3b[k], k, lay.ent_t12(k), T12)
            bufs['t12'] = T12
            _, _, twv, two = lay.tasks_t3()
            bufs['gB'] = c_(oovv_xq.transpose(1, 2, 0))
            S._run(twv, bufs, dtype)        # Wv += 1/2 <mn||ef> t(mnl, fcd)
            S._run(two, bufs, dtype)        # Wo -= 1/2 <mn||ef> t(nkl, efd)
            del bufs['gB']
            # t(i mn; K) as [(mn) of one class, K]; t(m kl; K) above Xv(e; kl; K)
            T3P = np.zeros(max(lay.t3psize, 1), dtype=dtype)
            for i in range(no):
                for c in range(lay.n2):
                    o = lay.t3poff[i, c]
                    if o >= 0 and nJ[ko[i] + c]:
                        k, mn = ko[i] + c, lay.O2cls[c]
                        T3P[o:o + len(mn) * nK[k]].reshape(len(mn), nK[k])[...] = \
                            T3b[k][lay.posO3[o12.pos[i, mn]]] * o12.psg[i, mn][:, None]
            # (``r13``: an array of a scratch file where memory is short)
            R13 = np.zeros(max(lay.r13size, 1), dtype=dtype) if r13 is None else P.zero(r13)
            for Pq in range(nP):
                for mu in range(ncls):
                    o = lay.r13off[Pq, mu]
                    if o >= 0 and nO[mu] and nJ[lay.kO2[Pq] + mu]:
                        k, ms = lay.kO2[Pq] + mu, np.arange(o0[mu], o0[mu + 1])
                        R13[o:o + nO[mu] * nK[k]].reshape(nO[mu], nK[k])[...] = \
                            T3b[k][lay.posO3[o12.pos[ms, Pq]]] * o12.psg[ms, Pq][:, None]
            bufs.update(t3p=T3P, r13=R13)
            # the ladder operator in blocks
            vpp = np.asarray(ints.vvvv_pp)
            Wpp = []
            for q in range(lay.n2):
                idx = lay.Q2cls[q]
                W = np.array(vpp[np.ix_(idx, idx)], dtype=dtype, order='C')
                if lay.nO2[q] and nQ[q]:
                    oc = lay.O2cls[q]
                    gemm(c_(t2_pq[oc][:, qs(q)].T), c_(oovv_pq[oc][:, qs(q)]), W, beta=1.0)
                Wpp.append(W)
            del vpp
            # <bc||ef> as [(bc) e, f] per (class of (bc), of e)
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
            # Xv without T4, below the rows of T3 in R13
            if xv:
                first, per = lay.tasks_xv()
                bufs['oovoA'] = c_(oovo_pp.transpose(2, 1, 0)[:, :, lay.o2perm])
                S._run(first, bufs, dtype)
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
                bufs.update(vo=VO, wrB=WrB, wblk=Wb)
                Mx = np.empty(max(2 * int(lay.mxsize.max()), 1), dtype=dtype)
                bufs['mx'] = Mx
                for Pq in range(nP):
                    kP = lay.kO2[Pq]
                    if not lay.mxsize[kP]:
                        continue
                    P.zero(Mx[:2 * lay.mxsize[kP]])
                    S._run(per[Pq], bufs, dtype)
                    for mu in range(ncls):
                        o = lay.r13off[Pq, mu]
                        if o < 0 or not nV[mu] or not np.any(lay.mxoff[kP, mu] >= 0):
                            continue
                        k = kP + mu
                        o += nO[mu] * nK[k]
                        S._x12(lay, 1, R13[o:o + nV[mu] * nK[k]].reshape(nV[mu], nK[k]), k, lay.ent_xv(kP, mu), Mx)
                del VO, WrB, Mx
        self.WvA, self.la13, self.T12, self.T3P, self.R13, self.Wpp, self.Wb = (
            bufs['wvA'], la13, T12, T3P, R13, Wpp, Wb)
        self.t2, self.t2_xq, self.t2_px, self.fov = t2, t2_xq, t2_px, fov
        self.oovv, self.oovv_xq, self.oovo, self.vovv, self.vovv_q = oovv, oovv_xq, oovo, vovv, vovv_q
        self.ovoo = np.asarray(ints.ovoo)
        self.vvvo_p = None

    def nbytes(self):
        return sum(a.nbytes for a in (self.WvA, self.la13, self.T12, self.T3P, self.R13, self.Wb)) + \
            sum(w.nbytes for w in self.Wpp)


def _pair_one_body(F, a, b):
    A, B, a_, b_ = a[:, None], b[:, None], a[None, :], b[None, :]
    return (F[A, a_] * (B == b_) - F[A, b_] * (B == a_) - F[B, a_] * (A == b_) + F[B, b_] * (A == a_))


PIECES = ('pp', 'hh', 'ring', 'x22', 'x13', 'row3', 'row2')


class Engine:
    '''The sigma vector of the sector (module docstring) for the ground
    state ``g`` (Ground) and the blocks ``sec`` (Sector).  ``rank`` 4: R to
    4p2h, 3: to 3p1h.  ``batch_bytes``: an operand or a product of one batch
    of the passes over r4, at most.'''

    def __init__(self, sec, g, rank=4, batch_bytes=None, T4=None):
        self.sec, self.g, self.rank = sec, g, rank
        self.T4 = None if T4 is None or rank != 4 else np.ascontiguousarray(T4, dtype=g.dtype)
        self.batch_bytes = batch_bytes or BATCH_BYTES
        lay = sec.lay
        self.lay = lay
        dtype = self.dtype = g.dtype
        no, nv, s = lay.no, lay.nv, sec.s
        aq, bq, qoff = lay.va, lay.vb, lay.qoff
        Fv = g.Fv
        self.Wq4, self.Wq3 = [], []
        for q in range(lay.n2):
            sl = slice(qoff[q], qoff[q + 1])
            F = _pair_one_body(Fv, aq[sl], bq[sl]) if lay.nQ[q] else 0
            self.Wq4.append(_c(g.Wpp[q] + F / 3, dtype))
            self.Wq3.append(_c(g.Wpp[q] + F / 2, dtype))
        sl = slice(qoff[s], qoff[s + 1]) if sec.n2 else slice(0, 0)
        self.Wq2 = _c(g.Wpp[s] + _pair_one_body(Fv, aq[sl], bq[sl]), dtype) if sec.n2 else np.zeros((0, 0), dtype)
        i2, j2 = lay.o2[:, 0], lay.o2[:, 1]
        self.Wh = _c(g.Whh[i2, j2][:, i2, j2] - t4engine._one_body_pairs(g.Fo, no), dtype)       # [(mn), (ij)]
        self.Wr2 = _c(g.Wr.transpose(3, 1, 0, 2).reshape(no * nv, no * nv), dtype)              # [(i a), (m e)]
        self.WvAt = _c(g.WvA.transpose(0, 2, 1), dtype)                                          # [x, e, (ab)]
        self.t2_2d = _c(g.t2_xq.transpose(1, 0, 2).reshape(no * no, -1), dtype)                  # [(m x), (ab)]
        self._ring = {}
        self._u = {}
        self.nsigma = 0
        self.keep, self.imds = False, None

    # ---------------------------------------------------------------- views
    def qs(self, q):
        return slice(self.lay.qoff[q], self.lay.qoff[q + 1])

    def cls(self, p):
        return slice(self.lay.v0[p], self.lay.v0[p + 1])

    def ocls(self, p):
        return slice(self.lay.o0[p], self.lay.o0[p + 1])

    def x3(self, X, i, p):
        '''The block (a of class p | (bc)) of the expanded row i of r3 (a view
        of X), or None.'''
        sec, lay = self.sec, self.lay
        K = sec.blk3[i]
        if K < 0 or not 0 <= p < lay.ncls:
            return None
        o = sec.off12[K, p]
        if o < 0:
            return None
        nq = lay.nQ[K - p]
        base = sec.x3off[i] + o
        return X[base:base + lay.nV[p] * nq].reshape(lay.nV[p], nq)

    # ---------------------------------------------------------------- the ring
    def ring_tables(self, j):
        '''For the spectator j: per d = k_m - k_e the rows and columns of the
        stacked operator, the places of the rows of r4 and of the product.'''
        if j in self._ring:
            return self._ring[j]
        sec, lay = self.sec, self.lay
        no, nv, ncls, ko, s = lay.no, lay.nv, lay.ncls, lay.ko, sec.s
        kj = int(ko[j])
        nW = no * nv
        blocks = []
        for d in range(-(ncls - 1), ncls):
            kK = kj + s + d
            if not (0 <= kK < lay.n3 and lay.nK[kK]):
                continue
            cols, base = [], {}
            for m in range(no):
                c = ko[m] - d
                if m != j and 0 <= c < ncls and lay.nV[c]:
                    base[m] = len(cols)
                    cols.extend(m * nv + np.arange(lay.v0[c], lay.v0[c + 1]))
            if not cols:
                continue
            rows, xbase = [], {}
            for i in range(no):
                c = ko[i] - d
                if i != j and 0 <= c < ncls and lay.nV[c]:
                    xbase[i] = len(rows)
                    rows.extend(i * nv + np.arange(lay.v0[c], lay.v0[c + 1]))
            fov = -1
            if d == 0:
                fov = len(rows)
                rows.append(nW)
            ci = kj + d
            r_o = r_v = None
            if 0 <= ci < ncls and lay.nO[ci]:
                r_o = (len(rows), np.arange(lay.o0[ci], lay.o0[ci + 1]))
                rows.extend(nW + 1 + np.arange(lay.o0[ci], lay.o0[ci + 1]))
            if 0 <= ci < ncls and lay.nV[ci]:
                r_v = (len(rows), np.arange(lay.v0[ci], lay.v0[ci + 1]))
                rows.extend(nW + 1 + no + np.arange(lay.v0[ci], lay.v0[ci + 1]))
            blocks.append(dict(d=d, kK=kK, nK=int(lay.nK[kK]), cols=np.array(cols, dtype=np.int64),
                               rows=np.array(rows, dtype=np.int64), base=base, xbase=xbase,
                               fov=fov, r_o=r_o, r_v=r_v))
        self._ring[j] = blocks
        return blocks

    def ring_entries(self, j, blocks, mo, xo):
        '''Entries {M: (row, sgn, xoff, s1, s2)} of the operand (rows (m, j),
        the blocks at the offsets mo) and of the product (rows (i, j), at xo).'''
        sec, lay = self.sec, self.lay
        ncls, ko = lay.ncls, lay.ko
        out = []
        for offs, key in ((mo, 'base'), (xo, 'xbase')):
            ents = {}
            for m in range(lay.no):
                if m == j:
                    continue
                P_ = sec.pair[m, j]
                M = sec.blk4[P_]
                if M < 0:
                    continue
                xoff = np.full(ncls, -1, dtype=np.int64)
                s1 = np.ones(ncls, dtype=np.int64)
                for b, o in zip(blocks, offs):
                    if m in b[key]:
                        l = ko[m] - b['d']
                        xoff[l] = o + b[key][m] * b['nK']
                        s1[l] = b['nK']
                if (xoff < 0).all():
                    continue
                ents.setdefault(M, []).append((sec.pos4[P_], 1.0 if m < j else -1.0, xoff, s1))
            tabs = {}
            for M, es in ents.items():
                tabs[M] = (np.array([e[0] for e in es], dtype=np.int64), np.array([e[1] for e in es]),
                           np.ascontiguousarray([e[2] for e in es], dtype=np.int64),
                           np.ascontiguousarray([e[3] for e in es], dtype=np.int64),
                           np.ones((len(es), ncls), dtype=np.int64))
            out.append(tabs)
        return out

    # ---------------------------------------------------------------- sigma
    def sigma(self, r2v, r3, r4, only=None):
        '''(sigma2, sigma3, sigma4) of the sector vector (r2, r3, r4) - flat
        arrays of the blocks of Sector.  ``only``: the pieces of PIECES to
        add (tests; the intermediates are made in full).'''
        def want(x):
            return only is None or x in only
        sec, lay, g, dtype = self.sec, self.lay, self.g, self.dtype
        no, nv, ncls, ko, s = lay.no, lay.nv, lay.ncls, lay.ko, sec.s
        nO, nV, nQ, nK = lay.nO, lay.nV, lay.nQ, lay.nK
        aq, bq = lay.va, lay.vb
        qs, cls, ocls = self.qs, self.cls, self.ocls
        nQt = len(lay.kQ)
        rank4 = self.rank == 4
        r2v = _c(r2v, dtype)
        r3 = _c(r3, dtype)
        r4 = _c(r4, dtype) if rank4 else None
        s2 = np.zeros(sec.n2, dtype=dtype)
        s3 = np.zeros(sec.size3, dtype=dtype)
        s4 = np.zeros(sec.size4, dtype=dtype) if rank4 else None
        fov, oovv_xq, vovv_q, oovo, vovv = g.fov, g.oovv_xq, g.vovv_q, g.oovo, g.vovv
        t2_xq = g.t2_xq

        with timer('dea.small'):
            r2 = np.zeros((nv, nv), dtype=dtype)
            if sec.n2:
                r2[aq[qs(s)], bq[qs(s)]] = r2v
                r2[bq[qs(s)], aq[qs(s)]] = -r2v
            X3r = np.zeros(max(int(sec.x3off[-1]), 1), dtype=dtype)       # r3 on (a | bc), row after row
            for K in sec.blocks3:
                sec.x12(0, sec.block3(r3, K), K, sec.ent_x3(K), X3r)
            X3s = np.zeros_like(X3r)                                        # sigma3 on (a | bc), before reduction
            h = vovv_q[:, :, qs(s)] @ r2v if sec.n2 else np.zeros((nv, no), dtype)     # h[c, m]
            WhR = oovv_xq[:, :, qs(s)] @ r2v if sec.n2 else np.zeros((no, no), dtype)  # [m, n]
            YR = np.zeros((no, no, no, nv), dtype=dtype)                    # YR(mn; y; d)
            Hm = np.zeros((no, nv, no, nv), dtype=dtype)                    # H[y][d, m, c]
            for y in range(no):
                K = sec.blk3[y]
                if K < 0:
                    continue
                for p in range(ncls):
                    b = self.x3(X3r, y, p)
                    if b is None:
                        continue
                    q = K - p
                    YR[:, :, y, cls(p)] = (oovv_xq[:, :, qs(q)].reshape(no * no, -1) @ b.T).reshape(no, no, -1)
                    Hm[y][:, :, cls(p)] = (vovv_q[:, :, qs(q)].reshape(nv * no, -1) @ b.T).reshape(nv, no, -1)
            Y3 = np.einsum('mnnc->mc', YR)
            WoR = -h.T - fov @ r2.T - Y3                                    # [m, d]
            Q3 = Y3 + h.T                                                   # [m, c]
            # XoR(m; j; cd) without its 4p2h term
            XoR = np.zeros((no, no, nQt), dtype=dtype)
            for n in range(no):
                K = sec.blk3[n]
                if K < 0:
                    continue
                for p in range(ncls):
                    b = self.x3(X3r, n, p)
                    if b is not None:
                        XoR[:, :, qs(K - p)] += np.einsum('mej,eQ->mjQ', oovo[:, n, cls(p), :], b)
            XoR += 0.5 * np.einsum('mn,njQ->mjQ', WhR, t2_xq)
            for y in range(no):
                XoR[:, y, :] -= (Hm[y][bq, :, aq] - Hm[y][aq, :, bq]).T
            # ZR(nf; ab) = sum_{me} <mn||ef> r3(m; eab): (n, class of f) -> [f, (ab)]
            ZR = {}
            for m in range(no):
                if sec.blk3[m] < 0:
                    continue
                for p in range(ncls):
                    b = self.x3(X3r, m, p)
                    if b is None:
                        continue
                    for n in range(no):
                        ph = ko[m] + ko[n] - p
                        if not (0 <= ph < ncls and nV[ph]) or n == m:
                            continue
                        z = g.oovv[m, n, cls(p), cls(ph)].T @ b
                        key = (n, ph)
                        ZR[key] = ZR[key] + z if key in ZR else z
            # XvR(e; bcd) without its 4p2h term, from U(e; d, (bc)) per class K of (bcd)
            Z = {}
            if rank4:
                U = {}
                for K in range(lay.n3):
                    mu = K - s
                    if nK[K] and 0 <= mu < ncls and nV[mu]:
                        U[K] = np.zeros((nV[mu], self.n12_all(K)), dtype=dtype)
                        Z[K] = np.zeros((nV[mu], nK[K]), dtype=dtype)
                for m in range(no):
                    Km = sec.blk3[m]
                    if Km < 0:
                        continue
                    for ph in range(ncls):
                        b = self.x3(X3r, m, ph)
                        if b is None:
                            continue
                        q = Km - ph
                        for mu in range(ncls):
                            dl = mu + ph - ko[m]
                            K = dl + q
                            if not (0 <= dl < ncls and nV[dl] and nV[mu]) or K not in U or K - s != mu:
                                continue
                            o = self.off12_all(K, dl)
                            A = vovv[cls(dl), m, cls(mu), cls(ph)].transpose(1, 0, 2).reshape(nV[mu] * nV[dl], nV[ph])
                            U[K][:, o:o + nV[dl] * nQ[q]] += (A @ b).reshape(nV[mu], nV[dl] * nQ[q])
                if sec.n2:
                    for K, Uk in U.items():
                        mu = K - s
                        for dl in range(ncls):
                            q = K - dl
                            if not (nV[dl] and 0 <= q < lay.n2 and nQ[q]) or lay.wboff[q, mu] < 0:
                                continue
                            ph = q - mu
                            o = lay.wboff[q, mu]
                            W = g.Wb[o:o + nQ[q] * nV[mu] * nV[ph]].reshape(nQ[q] * nV[mu], nV[ph])
                            x = (W @ r2[cls(ph), cls(dl)]).reshape(nQ[q], nV[mu], nV[dl])
                            o2 = self.off12_all(K, dl)
                            Uk[:, o2:o2 + nV[dl] * nQ[q]] += 0.5 * x.transpose(1, 2, 0).reshape(nV[mu], -1)
                for K, Uk in U.items():
                    sec.x12(1, Z[K], K, self.ent_u(K), _c(Uk, dtype))
                del U

        # ---- the passes over r4: pp ladder (+ its thin products), ring (+ thin), hh
        if rank4:
            isz = np.dtype(dtype).itemsize
            with timer('dea.pp'):
                for M in sec.blocks4:
                    R, out = sec.block4(r4, M), sec.block4(s4, M)
                    n22 = int(sec.n22[M])
                    if not n22:
                        continue
                    for r0, r1 in _batches(len(sec.rows4[M]), self.batch_bytes // (isz * n22)):
                        nb = r1 - r0
                        ent = sec.ent_pp(M, r0, r1)
                        Tv = P.zero(np.empty(nb * n22, dtype=dtype))
                        with timer('dea.expand'):
                            sec.x22(0, R, M, ent, Tv)
                        with timer('dea.thin'):
                            self._pp_thin(M, r0, r1, Tv, s2, X3s, XoR)
                        if want('pp'):
                            X = np.empty(nb * n22, dtype=dtype)
                            for q in range(lay.n2):
                                o = sec.off22[M, q]
                                if o < 0:
                                    continue
                                sl = slice(o * nb, (o + nQ[q] * nQ[M - q]) * nb)
                                with timer('dea.gemm'):
                                    gemm(self.Wq4[q], Tv[sl].reshape(nQ[q], -1), X[sl].reshape(nQ[q], -1))
                            with timer('dea.reduce'):
                                sec.x22(1, out, M, ent, X)
                            del X
                        del Tv
            with timer('dea.ring'):
                self._ring_pass(r4, s4, s3, Z, want('ring'))
            if want('hh'):
                with timer('dea.hh'):
                    for M in sec.blocks4:
                        rows = sec.rows4[M]
                        gemm(_c(self.Wh[np.ix_(rows, rows)].T, dtype), sec.block4(r4, M), sec.block4(s4, M), beta=1.0)
                    if self.T4 is not None and sec.n2:
                        self._hh_t4(WhR, s4)
            with timer('dea.rows'):
                self._rows(r2, r3, X3r, XoR, ZR, Z, WoR, YR, s4, want('x22'), want('x13'), want('hh'))

        # ---- sigma3
        if want('row3'):
            with timer('dea.row3'):
                for i in range(no):
                    K = sec.blk3[i]
                    if K < 0:
                        continue
                    for p in range(ncls):
                        xs = self.x3(X3s, i, p)
                        if xs is None:
                            continue
                        b = self.x3(X3r, i, p)
                        q = K - p
                        gemm(b, self.Wq3[q].T, xs, beta=1.0)                       # pp
                        e = s - p
                        if 0 <= e < ncls and nV[e]:
                            gemm(r2[cls(p), cls(e)], g.WvA[i, qs(q), cls(e)].T, xs, beta=1.0)     # Wv r2
                        if 0 <= e < ncls and nO[e]:
                            gemm(_c(Q3[ocls(e), cls(p)].T, dtype), t2_xq[i, ocls(e), qs(q)], xs, beta=1.0)
                        for m in range(no):                                         # ring
                            ep = ko[m] + p - ko[i]
                            bm = self.x3(X3r, m, ep)
                            if bm is None or bm.shape[1] != nQ[q]:
                                continue
                            gemm(_c(g.Wr[m, cls(p), cls(ep), i], dtype), bm, xs, beta=1.0)
                for K in sec.blocks3:
                    sec.x12(1, sec.block3(s3, K), K, sec.ent_x3(K), X3s)
                    rows = sec.rows3[K]
                    blk = sec.block3(s3, K)
                    blk -= _c(g.Fo[np.ix_(rows, rows)].T, dtype) @ sec.block3(r3, K)        # F_oo
                    if sec.n2:
                        mn = lay.O2cls[s]
                        wv = WhR[lay.o2[mn, 0], lay.o2[mn, 1]]
                        for x, i in enumerate(rows):
                            o = lay.t3poff[i, s]
                            if o >= 0:
                                blk[x] += wv @ g.T3P[o:o + len(mn) * nK[K]].reshape(len(mn), nK[K])
        else:
            s3[:] = 0
        # ---- sigma2
        if want('row2') and sec.n2:
            with timer('dea.row2'):
                s2 += self.Wq2 @ r2v
                G = np.zeros((nv, nv), dtype=dtype)
                for m in range(no):
                    Km = sec.blk3[m]
                    if Km < 0:
                        continue
                    b = self.x3(X3r, m, ko[m])
                    if b is not None and Km - ko[m] == s:
                        s2 += fov[m, cls(ko[m])] @ b
                    for p in range(ncls):
                        b = self.x3(X3r, m, p)
                        e = s - p
                        if b is None or not (0 <= e < ncls and nV[e]):
                            continue
                        G[cls(p), cls(e)] += b @ vovv_q[cls(e), m, qs(Km - p)].T
                s2 += G[aq[qs(s)], bq[qs(s)]] - G[bq[qs(s)], aq[qs(s)]]
        else:
            s2[:] = 0
        self.nsigma += 1
        if self.keep:
            # for the 5p3h row (zccsdt.cc.dea5_sector)
            self.imds = dict(r2=r2, X3r=X3r, WhR=WhR, YR=YR, Z=Z)
        return s2, s3, s4

    # ---------------------------------------------------------------- pieces
    def _u_tables(self, K):
        if K not in self._u:
            lay = self.lay
            off = np.full(lay.ncls, -1, dtype=np.int64)
            n = 0
            for p in range(lay.ncls):
                if lay.nV[p] and 0 <= K - p < lay.n2 and lay.nQ[K - p]:
                    off[p] = n
                    n += lay.nV[p] * lay.nQ[K - p]
            self._u[K] = (off, n)
        return self._u[K]

    def n12_all(self, K):
        return self._u_tables(K)[1]

    def off12_all(self, K, p):
        return self._u_tables(K)[0][p]

    def ent_u(self, K):
        '''The rows e of Z[K], U[e] on (d | bc).'''
        lay, s = self.lay, self.sec.s
        off, n = self._u_tables(K)
        ne = lay.nV[K - s]
        ok = off >= 0
        nc = np.where(ok, lay.nQ[np.clip(K - np.arange(lay.ncls), 0, lay.n2 - 1)], 0)
        xoff = np.where(ok[None, :], np.arange(ne)[:, None] * n + off[None, :], -1)
        return (np.arange(ne, dtype=np.int64), np.ones(ne), np.ascontiguousarray(xoff, dtype=np.int64),
                np.ascontiguousarray(np.broadcast_to(nc, xoff.shape), dtype=np.int64), np.ones(xoff.shape, dtype=np.int64))

    def _pp_thin(self, M, r0, r1, Tv, s2, X3s, XoR):
        '''From r4 on [(ef), (row, (cd))]: the 4p2h terms of sigma2, of sigma3
        (<cm||ef>) and of XoR.'''
        sec, lay, g = self.sec, self.lay, self.g
        ko, s, nQ, nV, ncls = lay.ko, sec.s, lay.nQ, lay.nV, lay.ncls
        nb = r1 - r0
        for q in range(lay.n2):
            o = sec.off22[M, q]
            if o < 0:
                continue
            nc = int(nQ[M - q])
            Tq = Tv[o * nb:(o + nQ[q] * nc) * nb].reshape(nQ[q], nb * nc)
            for r in range(r0, r1):
                u, w = (int(x) for x in lay.o2[sec.rows4[M][r]])
                T = Tq[:, (r - r0) * nc:(r - r0 + 1) * nc]
                if M - q == s:
                    s2 += g.oovv_xq[u, w, self.qs(q)] @ T
                for a, b, sg in ((u, w, 1.0), (w, u, -1.0)):
                    p = q - ko[b]
                    xs = self.x3(X3s, a, p) if 0 <= p < ncls and nV[p] else None
                    if xs is not None and xs.shape[1] == nc:
                        gemm(g.vovv_q[self.cls(p), b, self.qs(q)], T, xs, alpha=sg, beta=1.0)
                    mu = q - ko[a]
                    if 0 <= mu < ncls and lay.nO[mu]:
                        XoR[self.ocls(mu), b, self.qs(M - q)] += sg * (g.oovv_xq[self.ocls(mu), a, self.qs(q)] @ T)

    def _ring_pass(self, r4, s4, s3, Z, ring):
        sec, lay, g, dtype = self.sec, self.lay, self.g, self.dtype
        no, nv = lay.no, lay.nv
        isz = np.dtype(dtype).itemsize
        nW = no * nv
        for j in range(no):
            blocks = self.ring_tables(j)
            if not blocks:
                continue
            Sj = np.empty((nW + 1 + no + nv, nW), dtype=dtype)
            Sj[:nW] = self.Wr2
            Sj[nW] = g.fov.ravel()
            Sj[nW + 1:nW + 1 + no] = -0.5 * g.oovo[:, j, :, :].transpose(2, 0, 1).reshape(no, nW)
            Sj[nW + 1 + no:] = 0.5 * g.oovv[:, j, :, :].transpose(1, 0, 2).reshape(nv, nW)
            batch, cur, tot = [], [], 0
            for b in blocks:
                sz = (len(b['cols']) + len(b['rows'])) * b['nK']
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
                    m0 += len(b['cols']) * b['nK']
                    x0 += len(b['rows']) * b['nK']
                Mb = P.zero(np.empty(m0, dtype=dtype))
                Xb = np.empty(x0, dtype=dtype)
                em, ex = self.ring_entries(j, bl, mo, xo)
                with timer('dea.expand'):
                    for M, ent in em.items():
                        sec.x13(0, sec.block4(r4, M), M, ent, Mb)
                with timer('dea.gemm'):
                    for b, mo_, xo_ in zip(bl, mo, xo):
                        nc, nr, nK = len(b['cols']), len(b['rows']), b['nK']
                        W = _c(Sj[np.ix_(b['rows'], b['cols'])], dtype)
                        gemm(W, Mb[mo_:mo_ + nc * nK].reshape(nc, nK), Xb[xo_:xo_ + nr * nK].reshape(nr, nK))
                if ring:
                    with timer('dea.reduce'):
                        for M, ent in ex.items():
                            sec.x13(1, sec.block4(s4, M), M, ent, Xb)
                for b, xo_ in zip(bl, xo):
                    nK, kK = b['nK'], b['kK']
                    X = Xb[xo_:xo_ + len(b['rows']) * nK].reshape(len(b['rows']), nK)
                    if b['fov'] >= 0 and sec.blk3[j] == kK:
                        sec.block3(s3, kK)[sec.pos3[j]] += X[b['fov']]
                    if b['r_o'] is not None:
                        r0, idx = b['r_o']
                        for x, i in enumerate(idx):
                            if sec.blk3[i] == kK:
                                sec.block3(s3, kK)[sec.pos3[i]] += X[r0 + x]
                    if b['r_v'] is not None and kK in Z:
                        r0, idx = b['r_v']
                        Z[kK] += X[r0:r0 + len(idx)]
                del Mb, Xb

    def _x13_layout(self, MI):
        """(offsets by the class of the first index, size) of a row of the T4 block MI on (f | bcd)."""
        if not hasattr(self, '_l13'):
            self._l13 = {}
        if MI not in self._l13:
            lay = self.lay
            off = np.full(lay.ncls, -1, dtype=np.int64)
            n = 0
            for p in range(lay.ncls):
                if lay.nV[p] and 0 <= MI - p < lay.n3 and lay.nK[MI - p]:
                    off[p] = n
                    n += lay.nV[p] * lay.nK[MI - p]
            nc = np.where(off >= 0, lay.nK[np.clip(MI - np.arange(lay.ncls), 0, lay.n3 - 1)], 0)
            self._l13[MI] = (off, n, nc)
        return self._l13[MI]

    def _xv_t4(self, M, P_, r2, X13):
        """T of CCSDTQ: sum_{m<n} sum_f G(mn; a, f) t4(mnij; f, bcd) into X[a, (bcd)]
        of the row (ij), G(mn; a, f) = sum_e r2(a, e) <mn||ef>."""
        sec, lay, g, dtype = self.sec, self.lay, self.g, self.dtype
        s, ncls, nV, nK = sec.s, lay.ncls, lay.nV, lay.nK
        o22 = lay.o22
        for Pmn, (m, n) in enumerate(lay.o2):
            sg = int(o22.psg[Pmn, P_])
            if not sg:
                continue
            I = int(o22.pos[Pmn, P_])
            MI = int(lay.kO4[I])
            if MI not in lay.sectors:
                continue
            off, size, nc = self._x13_layout(MI)
            E = P.zero(np.empty(max(size, 1), dtype=dtype))
            ent = (np.array([lay.posO4[I]], dtype=np.int64), np.array([float(sg)]),
                   np.ascontiguousarray(off[None, :]), np.ascontiguousarray(nc[None, :]),
                   np.ones((1, ncls), dtype=np.int64))
            S._x13(lay, 0, lay.block(self.T4, MI), MI, ent, E)
            G = r2 @ g.oovv[m, n]                                   # [a, f]
            kmn = lay.ko[m] + lay.ko[n]
            for p in range(ncls):
                o = sec.off13[M, p]
                ph = kmn - s + p
                if o < 0 or not (0 <= ph < ncls) or off[ph] < 0:
                    continue
                nk = int(nK[M - p])
                assert MI - ph == M - p
                gemm(G[self.cls(p), self.cls(ph)], E[off[ph]:off[ph] + nV[ph] * nk].reshape(nV[ph], nk),
                     X13[o:o + nV[p] * nk].reshape(nV[p], nk), beta=1.0)

    def _hh_t4(self, WhR, s4):
        '''T of CCSDTQ: sum_{m<n} WhR(mn) t4(mnij; abcd).'''
        sec, lay = self.sec, self.lay
        mn = lay.O2cls[sec.s]
        w = WhR[lay.o2[mn, 0], lay.o2[mn, 1]]
        o22 = lay.o22
        for M in sec.blocks4:
            if M not in lay.sectors:
                continue
            T, out = lay.block(self.T4, M), sec.block4(s4, M)
            for r, P_ in enumerate(sec.rows4[M]):
                I, sg = o22.pos[mn, P_], o22.psg[mn, P_]
                ok = sg != 0
                if ok.any():
                    out[r] += (w[ok] * sg[ok]) @ T[lay.posO4[I[ok]]]

    def _rows(self, r2, r3, X3r, XoR, ZR, Z, WoR, YR, s4, w22, w13, whh=True):
        sec, lay, g, dtype = self.sec, self.lay, self.g, self.dtype
        no, ko, s, ncls = lay.no, lay.ko, sec.s, lay.ncls
        nO, nV, nQ, nK = lay.nO, lay.nV, lay.nQ, lay.nK
        qs, cls, ocls = self.qs, self.cls, self.ocls
        o12 = lay.o12
        # [r3 rows of class mu; Z rows of class mu] for K = mu + s
        S13 = {}
        for mu in range(ncls):
            K = mu + s
            if 0 <= K < lay.n3 and nK[K]:
                top = sec.block3(r3, K) if len(sec.rows3[K]) else np.zeros((nO[mu], nK[K]), dtype)
                bot = Z.get(K, np.zeros((nV[mu], nK[K]), dtype))
                S13[mu] = _c(np.vstack((top, bot)), dtype)
        isz = np.dtype(dtype).itemsize
        self.XoR_2d = XoR.reshape(no * no, -1)
        for M in sec.blocks4:
            out = sec.block4(s4, M)
            for r, P_ in enumerate(sec.rows4[M]):
                i, j = (int(x) for x in lay.o2[P_])
                if w22 and sec.n22[M]:
                    # per class q of (ab): one product of the stacked operands (gathered by rows,
                    # on all threads)  [-Wv(ab, e, x) | t(xm, ab) | ZR(nf; ab)]
                    #                  [r3(y; e, cd); XoR(m; y; cd); t(nij; f, cd)]
                    X22 = np.empty(int(sec.n22[M]), dtype=dtype)
                    for q in range(lay.n2):
                        o = sec.off22[M, q]
                        if o < 0:
                            continue
                        nc, nq = int(nQ[M - q]), int(nQ[q])
                        Xq = X22[o:o + nq * nc].reshape(nq, nc)
                        parts = []          # (L^T source [*, nQt or nq], rows, sign, cols | R source, rows, sign, cols)
                        for x, y, sg in ((i, j, 1), (j, i, -1)):
                            e = q - ko[x]
                            if not (0 <= e < ncls):
                                continue
                            b = self.x3(X3r, y, e) if nV[e] else None
                            if b is not None and b.shape[1] == nc:
                                parts.append((self.WvAt[x], np.arange(lay.v0[e], lay.v0[e + 1]), -sg, qs(q),
                                              b, np.arange(nV[e]), 1, None))
                            if nO[e]:
                                ms = np.arange(lay.o0[e], lay.o0[e + 1])
                                parts.append((self.t2_2d, ms * no + x, sg, qs(q),
                                              self.XoR_2d, ms * no + y, 1, qs(M - q)))
                        for n in range(no):
                            ph = q - s + ko[n]
                            z = ZR.get((n, ph))
                            sg3 = int(o12.psg[n, P_])
                            if z is None or not sg3:
                                continue
                            J = int(o12.pos[n, P_])
                            kJ = lay.kO3[J]
                            t0 = lay.t12off[kJ, ph]
                            if t0 < 0 or kJ - ph != M - q:
                                continue
                            nJk = int(lay.nJ[kJ])
                            Tsrc = g.T12[t0:t0 + nV[ph] * nJk * nc].reshape(nV[ph] * nJk, nc)
                            parts.append((z, np.arange(nV[ph]), sg3, None,
                                          Tsrc, np.arange(nV[ph]) * nJk + int(lay.posO3[J]), 1, None))
                        k = sum(len(pt[1]) for pt in parts)
                        if not k:
                            Xq[...] = 0
                            continue
                        with timer('dea.rows.stack'):
                            Lt = np.empty((k, nq), dtype=dtype)
                            Rc = np.empty((k, nc), dtype=dtype)
                            r0 = 0
                            for A, ra, sa, ca, B, rb, sb, cb in parts:
                                n_ = len(ra)
                                P.tn_rows(Lt[r0:r0 + n_], A, ra, np.full(n_, sa), None if ca is None else (ca.start, ca.stop))
                                P.tn_rows(Rc[r0:r0 + n_], B, rb, np.full(n_, sb), None if cb is None else (cb.start, cb.stop))
                                r0 += n_
                        with timer('dea.rows.gemm'):
                            gemm(Lt.T, Rc, Xq)
                    with timer('dea.rows.reduce'):
                        sec.x22(1, out, M, sec.ent_row(M, r, '22'), X22)
                if w13 and sec.n13[M]:
                    # per class p of a: [Wo(m, a, ij) | t(ij, ae) | WoR(m, a) | r2(a, e) | -YR(mn; y; a)]
                    # [r3(m; K); XvR(e; K); t(mij; K); Xv(e; ij; K); t(mnx; K)]
                    X13 = np.empty(int(sec.n13[M]), dtype=dtype)
                    kP = ko[i] + ko[j]
                    for p in range(ncls):
                        o = sec.off13[M, p]
                        if o < 0:
                            continue
                        nk = int(nK[M - p])
                        Xp = X13[o:o + nV[p] * nk].reshape(nV[p], nk)
                        Ls, Rs = [], []
                        mu = kP - p
                        if mu in S13 and S13[mu].shape[1] == nk:
                            m0 = lay.mix0[mu]
                            Ls.append(g.la13[P_, cls(p), m0:m0 + nO[mu] + nV[mu]])
                            Rs.append(S13[mu])
                        mu = s - p
                        if 0 <= mu < ncls and lay.r13off[P_, mu] >= 0:
                            o2 = lay.r13off[P_, mu]
                            nr = nO[mu] + nV[mu]
                            Ls.append(np.concatenate((WoR[ocls(mu), cls(p)].T, r2[cls(p), cls(mu)]), axis=1))
                            Rs.append(g.R13[o2:o2 + nr * nk].reshape(nr, nk))
                        for x, y, sg in ((i, j, 1.0), (j, i, -1.0)):
                            c = M - p - ko[x]
                            if not (0 <= c < lay.n2) or lay.t3poff[x, c] < 0:
                                continue
                            mn = lay.O2cls[c]
                            o3 = lay.t3poff[x, c]
                            Ls.append(-sg * YR[lay.o2[mn, 0], lay.o2[mn, 1], y, cls(p)].T)
                            Rs.append(g.T3P[o3:o3 + len(mn) * nk].reshape(len(mn), nk))
                        with timer('dea.rows.stack'):
                            if Ls:
                                Lc = _c(np.concatenate(Ls, axis=1), dtype)
                                Rc = np.empty((Lc.shape[1], nk), dtype=dtype)
                                r0 = 0
                                for B in Rs:
                                    P.tn_rows(Rc[r0:r0 + len(B)], _c(B, dtype), np.arange(len(B)), np.ones(len(B)))
                                    r0 += len(B)
                        with timer('dea.rows.gemm'):
                            if Ls:
                                gemm(Lc, Rc, Xp)
                            else:
                                Xp[...] = 0
                    if self.T4 is not None and whh and sec.n2:
                        self._xv_t4(M, P_, r2, X13)
                    with timer('dea.rows.reduce'):
                        sec.x13(1, out, M, sec.ent_row(M, r, '13'), X13)


# ------------------------------------------------------------------ the problem of the DEA driver

def shift_of(eris, target):
    '''The class of the pairs of two attached electrons for the target (the
    sum of their labels, 2 Omega), or None if no pair can have it.'''
    from zccsdt.cc import t3pack
    lab = np.asarray(eris.labels, dtype=np.int64)
    u = np.unique(lab)
    step = int(np.gcd.reduce(np.diff(u))) if len(u) > 1 else 1
    d = int(target) - 2 * int(lab.min())
    if d % step:
        return None
    del t3pack
    return d // step


def memory(sec, rank, isz, batch_bytes=None, ground=True, xv_disk=False):
    '''Bytes that ProblemSector holds besides the vectors of the Davidson
    subspace (the integrals of the ground state included), and the most a
    sigma vector allocates on top: (fixed, per sigma), each counted array
    by array.  ``xv_disk``: the six-index Xv (with the rows of T3) in a
    scratch file, not counted.'''
    batch_bytes = batch_bytes or BATCH_BYTES
    lay = sec.lay
    no, nv, ncls = lay.no, lay.nv, lay.ncls
    nP, nQ = len(lay.o2), len(lay.kQ)
    ov = no * nv
    wq = int((lay.nQ ** 2).sum())
    # the dressed integrals (zccsdt.cc.t4engine.Dressed): oovv, oooo, oovo, ovvo, vovv, ovoo, vvvo on pairs, vvvv on pairs
    ints = no * no * nv * nv * 2 + no ** 4 + 2 * no ** 3 * nv + no * nv ** 3 + nQ * nv * no + nQ * nQ
    # Ground: WvA, la13, T12, T3P, R13, Wpp, Wb; the small intermediates; its copies of the integrals
    gr = (no * nQ * nv + nP * nv * (no + nv) + int(lay.t12size) + int(lay.t3psize) + wq
          + (0 if xv_disk else int(lay.r13size))
          + int(lay.wbsize) + 2 * no ** 4 + 2 * ov * ov + no * nv * nQ + no * no * nQ * 2)
    # its construction on top: einsum of <dm||ef> t2 (v^3 o), the tables of Xv
    build = nv ** 3 * no + 2 * int(lay.mxsize.max(initial=0)) + int(lay.vosize) + int(lay.wrsize) + nv * nv * no * no
    # Engine: Wq4, Wq3, Wq2, Wh, Wr2
    en = 2 * wq + int(lay.nQ[sec.s] ** 2 if sec.n2 else 0) + nP * nP + ov * ov
    # Sector tables: the quadruples (int16) and the rest
    tables = 8 * int(sec.nA.sum()) // isz + (1 << 20) // isz
    # the elements of the sector: masks (1 byte) and the sums of the orbital energies per column
    elems = (sum(sec.sizes) // isz + 1) + int(sec.nA.sum()) + int(lay.nK.sum())
    fixed = ints + gr + en + tables + elems
    # per sigma: the result, r3 and sigma3 on (a | bc), the intermediates of the sector, the
    # largest of the passes (each batch as Engine makes it)
    s_, ko, nV, nO, nK, nQc = sec.s, lay.ko, lay.nV, lay.nO, lay.nK, lay.nQ
    x3 = int(sec.x3off[-1])
    zr = sum(int(nV[ph] * nQc[s_ + ph - ko[n]]) for n in range(no) for ph in range(ncls)
             if 0 <= s_ + ph - ko[n] < lay.n2)
    small = (2 * x3 + 2 * no ** 3 * nv + no * no * nv * nv + no * no * nQ + zr + 2 * nv * nv
             + nv * no + no * no + 3 * no * nv)
    uz = 0
    if rank == 4:
        for K in range(lay.n3):
            mu = K - s_
            if nK[K] and 0 <= mu < ncls and nV[mu]:
                n12 = sum(int(nV[p] * nQc[K - p]) for p in range(ncls) if nV[p] and 0 <= K - p < lay.n2)
                uz += int(nV[mu]) * (n12 + int(nK[K]))                      # U and Z
        uz += max([int(nQc[q] * nV[mu] * nV[dl]) for q in range(lay.n2) for mu in range(ncls)
                   for dl in range(ncls)] + [0])                             # a product of Wpp with r2
    passes = 0
    if rank == 4:
        for M in sec.blocks4:
            n22 = int(sec.n22[M])
            if n22:
                nb = min(len(sec.rows4[M]), max(1, batch_bytes // (isz * n22)))
                passes = max(passes, 2 * nb * n22)                           # pp: the operand and the product
        ring = 0
        for j in range(no):
            kj, sizes = int(ko[j]), []
            for d in range(-(ncls - 1), ncls):
                kK = kj + s_ + d
                if not (0 <= kK < lay.n3 and nK[kK]):
                    continue
                nc = sum(int(nV[ko[m] - d]) for m in range(no) if m != j and 0 <= ko[m] - d < ncls)
                if not nc:
                    continue
                nr = sum(int(nV[ko[i] - d]) for i in range(no) if i != j and 0 <= ko[i] - d < ncls)
                nr += (d == 0) + (int(nO[kj + d]) + int(nV[kj + d]) if 0 <= kj + d < ncls else 0)
                sizes.append(((nc + nr) * int(nK[kK]), nc * nr))
            cur = tot = w = 0
            for sz, wz in sizes:
                if cur and (tot + sz) * isz > batch_bytes:
                    ring = max(ring, tot)
                    tot = 0
                cur, tot, w = 1, tot + sz, max(w, wz)
            ring = max(ring, tot + w)
        passes = max(passes, ring + (ov + 1 + no + nv) * ov)                 # with the stacked operator
        s13 = sum(int((nO[mu] + nV[mu]) * nK[mu + s_]) for mu in range(ncls) if 0 <= mu + s_ < lay.n3)
        # the stacked operands of a row (and their copies in one array each)
        mV, mO, mQ, mK = int(nV.max()), int(nO.max()), int(nQc.max()), int(nK.max())
        k22, k13 = 2 * mV + 2 * mO + no * mV, 2 * (mO + mV) + 2 * int(lay.nO2.max())
        rows = (s13 + int(sec.n22.max(initial=0)) + int(sec.n13.max(initial=0)) + 2 * k22 * 2 * mQ
                + 2 * k13 * (mV + mK) + no * nv * 4)
        passes = max(passes, rows)
    s = sum(sec.sizes) + small + uz + passes
    return isz * (fixed + (build if ground else 0)), isz * s


class ProblemSector:
    '''The DEA problem of zccsdt.cc.dea on the sector itself (module
    docstring): the interface of zccsdt.cc.dea.ProblemSym - parts r2, r3,
    r4 (flat, in the blocks of Sector), masks of the parity, the
    preconditioner and the guess - for the driver dea.kernel.  ``eris``:
    integrals with labels in the order of the labels (the layout of the
    ground state in label blocks), ``t3`` flat in the blocks of the engine
    of ``eris`` (or None: the T of CCSD).  ``rank`` 3 or 4.'''

    def __init__(self, eris, t1, t2, t3=None, rank=4, target=0, parity=None, target_parity=0, mem_mb=2000.0,
                 batch_bytes=None, verbose=False, t4=None):
        from zccsdt.cc import ccsdt_opt, t3pack
        t0 = time.perf_counter()
        if t3pack.classes_of(eris) is None or S.layout_for(eris) is None:
            raise ValueError('EOM-DEA on the sector: labels without a modulus in use, the spinors in their order')
        if rank not in (3, 4):
            raise ValueError('rank: 3 or 4')
        self.rank, self.row3, self.row4 = rank, 'full', 'full'
        no, n = eris.nocc, np.asarray(eris.fock).shape[0]
        nv = n - no
        self.no, self.nv = no, nv
        self.dtype = dtype = np.dtype(np.result_type(t1, t2, eris.oovv, eris.fock))
        isz = dtype.itemsize
        self.lay = lay = S.layout_for(eris)
        sh = shift_of(eris, target)
        if sh is None or not 0 <= sh < lay.n2:
            raise ValueError('EOM-DEA: no pair of virtual spinors with the target %d' % target)
        self.sec = sec = Sector(lay, sh)
        # the six-index Xv of the ground state (the largest array) in a scratch file if the rest
        # and four vectors do not fit with it
        vec = sum(sec.sizes) * isz
        fixed, per = memory(sec, rank, isz, 1)
        self.xv_disk = rank == 4 and fixed + per + 4 * vec > mem_mb * 1e6
        r13, self.scr = None, None
        if self.xv_disk:
            from zccsdt.scratch import Scratch
            self.scr = Scratch(prefix='zccsdt_dea_', what='the six-index intermediate Xv of EOM-DEA')
            r13 = self.scr.create('xv', max(int(lay.r13size), 1), dtype)
            fixed, per = memory(sec, rank, isz, 1, xv_disk=True)
        bb = batch_bytes
        if bb is None:
            # the batches as large as the budget leaves them, at most BATCH_BYTES
            bb = int(min(BATCH_BYTES, max(1 << 24, (mem_mb * 1e6 - fixed - per) / 3)))
        self.batch_bytes = bb
        if t3 is None:
            T3 = np.zeros(int(lay.t3off[-1]), dtype=dtype)
        else:
            T3 = np.ascontiguousarray(lay.t3_from_packed(ccsdt_opt.packed_t3(eris, t3)), dtype=dtype)
        ints = t4engine.Dressed.from_eris(eris, np.asarray(t1, dtype=dtype))
        self.g = Ground(lay, np.asarray(t2), T3, ints, xv=(rank == 4), r13=r13)
        del ints, T3
        if t4 is not None and rank != 4:
            raise ValueError('T4 with rank 3: not a level of this code')
        self.eng = Engine(sec, self.g, rank, bb, None if t4 is None else np.asarray(t4).ravel())
        self.mem = memory(sec, rank, isz, bb, ground=False, xv_disk=self.xv_disk)
        # the elements: denominators e_a + .. - e_i - .. (per block, from the columns), parity masks
        ev = np.asarray(eris.mo_energy).real
        self.eo, self.ev = ev[:no], ev[no:]
        evv = self.ev
        par = None if parity is None else np.asarray(parity, dtype=np.int64)
        aq, bq = lay.va, lay.vb
        qs = slice(lay.qoff[sh], lay.qoff[sh + 1])
        self.cden = [None, {}, {}]           # sums of e over the columns per block
        self.cden[0] = evv[aq[qs]] + evv[bq[qs]]
        m2 = np.ones(sec.n2, bool)
        if par is not None:
            m2 = (par[no:][aq[qs]] + par[no:][bq[qs]] - target_parity) % 2 == 0
        masks = [m2, np.ones(sec.size3, bool), np.ones(sec.size4 if rank == 4 else 0, bool)]
        pv, po = (par[no:], par[:no]) if par is not None else (None, None)
        for K in sec.blocks3:
            tri = lay.tri[K].astype(np.int64)
            self.cden[1][K] = evv[tri].sum(axis=1)
            if par is not None:
                mk = (pv[tri].sum(axis=1)[None, :] - po[sec.rows3[K]][:, None] - target_parity) % 2 == 0
                masks[1][sec.off3[K]:sec.off3[K + 1]] = mk.ravel()
        if rank == 4:
            for M in sec.blocks4:
                qd = sec.quads[M].astype(np.int64)
                self.cden[2][M] = evv[qd].sum(axis=1)
                if par is not None:
                    pr = lay.o2[sec.rows4[M]]
                    mk = (pv[qd].sum(axis=1)[None, :] - po[pr].sum(axis=1)[:, None] - target_parity) % 2 == 0
                    masks[2][sec.off4[M]:sec.off4[M + 1]] = mk.ravel()
        self.masks = masks
        self.sizes = [sec.n2, sec.size3] + ([sec.size4] if rank == 4 else [])
        self.masks = self.masks[:len(self.sizes)]
        self.shapes = [(n_,) for n_ in self.sizes]
        self.block_sizes = (int(lay.t3off[-1]), sec.size4)
        self.nsigma, self.seconds = 0, 0.0
        self.prepare_seconds = time.perf_counter() - t0
        if verbose:
            print('  sector: 2p %d, 3p1h %d, 4p2h %d elements; memory %.2f GB fixed + %.2f GB per sigma vector; '
                  'batches of %.0f MB%s; prepared in %.1f s'
                  % (sec.n2, sec.size3, sec.size4 if rank == 4 else 0, self.mem[0] / 1e9, self.mem[1] / 1e9,
                     bb / 1e6, '; Xv in a scratch file (%.2f GB)' % (lay.r13size * isz / 1e9) if self.xv_disk else '',
                     self.prepare_seconds), flush=True)

    def project(self, x):
        for p, m in zip(x, self.masks):
            if p.size:
                p *= m
        return x

    def new(self):
        return [np.zeros(s, dtype=self.dtype) for s in self.sizes]

    def sigma(self, x, y):
        t0 = time.perf_counter()
        r4 = x[2] if self.rank == 4 else None
        s2, s3, s4 = self.eng.sigma(x[0], x[1], r4)
        y[0][:] = s2
        y[1][:] = s3
        if self.rank == 4:
            y[2][:] = s4
        del s2, s3, s4
        self.project(y)
        self.nsigma += 1
        self.seconds += time.perf_counter() - t0
        return y

    def _den_blocks(self, part, chunk=1 << 21):
        """(slice, den [rows, cols]) of the blocks of a part, a few rows at a time
        (at most about ``chunk`` elements: no array of the size of a block)."""
        sec = self.sec
        if part == 0:
            yield slice(0, sec.n2), self.cden[0][None, :]
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
        '''Unit vectors on the ``n`` lowest pairs (a<b) of the masks.'''
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
        return self.sec.dense_positions()[:len(self.sizes)]
