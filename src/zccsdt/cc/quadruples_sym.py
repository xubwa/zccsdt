'''(Q) in symmetry sectors: zccsdt.cc.quadruples for spinors that carry a
label of an Abelian group.

Every spinor has an integer label that adds - 2 m_j for a linear molecule,
+1 and -1 modulo 4 for the two fermion irreps of C2 (``eris.labels``,
``eris.modulus``; no symmetry is the case of one label) -, and an integral or
an amplitude vanishes unless the labels of its upper indices sum to those of
its lower ones.  For an occupied quadruple of total label M only the virtual
quadruples of total M are there, and each of the five products of
zccsdt.cc.quadruples falls into blocks:

    X[(ab), (cd)]    one block per label q of (ab): the pairs of label q
                     times those of label M - q; of the stacked rows of a
                     split only the summed indices with the label that q
                     leaves
    X[a, (bcd)]      one block per label p of a: the virtual spinors of label
                     p times the triples of label M - p

The blocks of one occupied quadruple are formed by matrix products on the
rows and columns picked from the operands of the dense code, and one kernel
(zc_q_sector_energy, csrc/tnpack.c) walks through the virtual quadruples of
the sector, sums the splits of each from the blocks and adds the energies:
nothing of the size of the virtual quadruples is stored, and no tables of
their size are made.

The intermediate of T2 with <vv||vv> is in blocks too: Xv(e; xl; bcd) for
the e of one label has the triples of one label only, and is formed from the
blocks <bc||ef> t(xl, fd) of the labels of (bc) (zc_xv_sector).  The rows of
T3 that a product needs are read where they are (zc_rows_gemm).
'''

import ctypes
from dataclasses import dataclass
from math import comb

import numpy as np

from zccsdt.blas import gemm
from zccsdt.cc import quadruples as Q, t4pack as P
from zccsdt.timing import timer

_I64, _INT, _DBL = ctypes.c_int64, ctypes.c_int, ctypes.c_double


def available(eris):
    '''Whether the container has labels and the kernel is built.'''
    return (getattr(eris, 'labels', None) is not None and P.USE_C
            and hasattr(P._lib, 'zc_q_sector_energy_d'))


def _classes(lab):
    '''{label: positions with it, ascending} and the position of every
    element inside its class.'''
    out, pos = {}, np.empty(len(lab), dtype=np.int32)
    for k in np.unique(lab):
        idx = np.where(lab == k)[0]
        out[int(k)] = idx
        pos[idx] = np.arange(len(idx))
    return out, pos


class Sectors:
    '''The labels of the active spinors of ``eris`` and the classes of the
    virtual spinors, pairs and triples by label.'''

    def __init__(self, eris):
        no = eris.nocc
        lab = np.asarray(eris.labels, dtype=np.int64)
        self.modulus = N = int(getattr(eris, 'modulus', 0) or 0)
        self.nocc, self.nvir = no, len(lab) - no
        self.ko, self.kv = self.red(lab[:no]), self.red(lab[no:])
        nv = self.nvir
        self.O, _ = _classes(self.ko)
        self.V, self.posV = _classes(self.kv)
        p2, p3 = P.tuples(nv, 2), P.tuples(nv, 3)
        self.Q2, self.posQ = _classes(self.red(self.kv[p2].sum(axis=1)))
        self.A3, self.pos3 = _classes(self.red(self.kv[p3].sum(axis=1)))
        self.pairs = p2
        # what the kernel needs of the classes of the virtual spinors
        self.vlab = sorted(self.V)
        self.kmin, self.nk = self.vlab[0], self.vlab[-1] - self.vlab[0] + 1
        self.lid_of = np.full(self.nk, -1, dtype=np.int32)
        self.lid_of[np.array(self.vlab) - self.kmin] = np.arange(len(self.vlab))
        self.lmem = np.concatenate([self.V[k] for k in self.vlab]).astype(np.int32)
        self.lstart = np.concatenate(([0], np.cumsum([len(self.V[k]) for k in self.vlab]))).astype(np.int32)
        self.qlab = sorted(self.Q2)
        self.qmin, self.nq = self.qlab[0], self.qlab[-1] - self.qlab[0] + 1
        self.tri = {k: np.ascontiguousarray(p3[idx], dtype=np.int32) for k, idx in self.A3.items()}
        self._layout, self._t3tab, self._w = {}, {}, {}

    def red(self, k):
        return k % self.modulus if self.modulus else k

    def layout(self, M):
        '''The blocks of the products of an occupied quadruple of total label
        M: ([(q, offset, rows, columns)], size) for X[(ab), (cd)],
        ([(p, offset, rows, columns)], size) for X[a, (bcd)], and the same as
        the arrays of the kernel.'''
        if M not in self._layout:
            b22, n22 = [], 0
            off22, nc22 = np.full(self.nq, -1, dtype=np.int64), np.zeros(self.nq, dtype=np.int64)
            for q in self.qlab:
                r = int(self.red(M - q))
                if r in self.Q2:
                    nr, nc = len(self.Q2[q]), len(self.Q2[r])
                    b22.append((q, n22, nr, nc))
                    off22[q - self.qmin], nc22[q - self.qmin] = n22, nc
                    n22 += nr * nc
            b13, n13 = [], 0
            off13, nc13 = np.full(len(self.vlab), -1, dtype=np.int64), np.zeros(len(self.vlab), dtype=np.int64)
            for l, p in enumerate(self.vlab):
                r = int(self.red(M - p))
                if r in self.A3:
                    nr, nc = len(self.V[p]), len(self.A3[r])
                    b13.append((p, n13, nr, nc))
                    off13[l], nc13[l] = n13, nc
                    n13 += nr * nc
            self._layout[M] = (b22, n22, b13, n13, off22, nc22, off13, nc13)
        return self._layout[M]

    def t3_rows(self, ke, qc):
        '''(columns, signs) of t(J; e, c<d) in the packed row of J, for the e
        of label ``ke`` and the pairs of label ``qc``: [e, (cd)] each (sign 0
        where e is one of c, d).'''
        if (ke, qc) not in self._t3tab:
            nv = self.nvir
            e = self.V[ke][:, None]
            c, d = (self.pairs[self.Q2[qc]][:, x][None, :] for x in (0, 1))
            lo, mid, hi = np.minimum(e, c), np.clip(e, c, d), np.maximum(e, d)
            sgn = np.where(e < c, 1, np.where(e < d, -1, 1)) * ((e != c) & (e != d))
            y1, y2 = nv - 1 - lo, nv - 1 - mid
            rank = comb(nv, 3) - 1 - y1 * (y1 - 1) * (y1 - 2) // 6 - y2 * (y2 - 1) // 2 - (nv - 1 - hi)
            self._t3tab[ke, qc] = (np.where(sgn != 0, rank, 0), sgn.astype(np.float64))
        return self._t3tab[ke, qc]

    def w_block(self, vpp, q2, ke):
        '''<bc||ef> for the pairs (bc) of label ``q2``, the e of label ``ke``
        and the f of the label that is left: [(bc) e, f], kept.'''
        if (q2, ke) not in self._w:
            nv = self.nvir
            e, f = self.V[ke][:, None], self.V[int(self.red(q2 - ke))][None, :]
            lo, hi = np.minimum(e, f), np.maximum(e, f)
            rank = comb(nv, 2) - 1 - (nv - 1 - lo) * (nv - 2 - lo) // 2 - (nv - 1 - hi)
            sgn = (np.sign(f - e) + 0 * rank).ravel()
            col = np.where(sgn != 0, rank.ravel(), 0)
            W = np.asarray(vpp)[np.ix_(self.Q2[q2], col)] * sgn
            self._w[q2, ke] = np.ascontiguousarray(W.reshape(-1, f.shape[1]))
        return self._w[q2, ke]

    def w_size(self):
        '''Elements of all blocks of w_block().'''
        return sum(len(self.Q2[q]) * len(self.V[k]) * len(self.V[int(self.red(q - k))])
                   for q in self.qlab for k in self.vlab if int(self.red(q - k)) in self.V)

    def xv_size(self):
        '''Elements of the blocks of Xv for one pair of occupied indices, at
        most, and of the largest set of products that one of them is made
        from.'''
        nv = self.nvir
        slab = work = 0
        for kxl in np.unique(self.red(self.ko[:, None] + self.ko[None, :])):
            n = 0
            for ke in self.vlab:
                if int(self.red(ke + kxl)) not in self.A3:
                    continue
                n += len(self.V[ke]) * len(self.A3[int(self.red(ke + kxl))])
                m = 0
                for q in self.qlab:
                    kf = int(self.red(q - ke))
                    kd = int(self.red(kxl - kf))
                    if kf in self.V and kd in self.V:
                        m += len(self.Q2[q]) * len(self.V[ke]) * len(self.V[kd])
                work = max(work, m)
            slab = max(slab, n)
        return slab, work

    def sizes(self, quads):
        '''(elements of the largest X[(ab), (cd)], of the largest X[a, (bcd)])
        over the occupied quadruples ``quads``.'''
        Ms = np.unique(self.red(self.ko[quads].sum(axis=1)))
        return (max(self.layout(int(M))[1] for M in Ms), max(self.layout(int(M))[3] for M in Ms))


@dataclass
class Plan:
    need: int               # bytes besides T3 and the integrals
    dense: object           # the plan of the dense code for its operands and the x of Xv at a time
    sectors: object         # Sectors
    n22: int                # elements of the largest set of blocks X[(ab), (cd)]
    n13: int                # and X[a, (bcd)]


def plan(eris, isz, budget=None):
    """The plan for ``budget`` bytes (None: no limit); MemoryError if the
    blocks and the operands of the dense code do not fit.  Counted: the
    blocks of one occupied quadruple (three products on two pairs, two on one
    index and a triple), <vv||vv> in blocks, the products that Xv is made
    from, the tables of the sectors and the rows picked for a product; Xv
    itself for each occupied index as the dense plan counts it (v C(v,3)
    elements, of which the blocks have a part)."""
    sec = Sectors(eris)
    no, nv = sec.nocc, sec.nvir
    n22, n13 = sec.sizes(P.tuples(no, 4))
    slab, work = sec.xv_size()
    nQ, nA3 = comb(nv, 2), comb(nv, 3)
    tables = 16 * nv * nQ + 32 * nA3 + 8 * nQ
    rows = 4 * (4 * nv + 4 * no + 6) * max(len(x) for x in sec.Q2.values()) * isz
    mine = (3 * n22 + 2 * n13 + sec.w_size() + work) * isz + tables + rows
    pl = Q.plan(no, nv, isz, None if budget is None else budget - mine)
    pl = Q.explicit_plan(no, nv, isz, 1, 1, pl.xv, pl.rows)
    return Plan(pl.need + mine, pl, sec, n22, n13)


def xv_blocks(sec, q, x, l):
    '''{label of e: Xv[e, (bcd)]} for the occupied pair (x, l): the e of
    that label and the triples of the label that goes with it,
    Xv(e; xl; bcd) = P(d/bc) 1/2 sum_f <bc||ef> t(xl, fd).'''
    dtype = q.dtype
    kxl = int(sec.red(sec.ko[x] + sec.ko[l]))
    t = np.asarray(q.t2[x, l])
    fn = getattr(P._lib, 'zc_xv_sector_' + P._SFX[np.dtype(dtype)])
    kv32 = np.ascontiguousarray(sec.kv, dtype=np.int32)
    out = {}
    for ke in sec.vlab:
        T = int(sec.red(ke + kxl))
        if T not in sec.A3:
            continue
        ne = len(sec.V[ke])
        todo, n = [], 0
        off, nd = np.full(sec.nq, -1, dtype=np.int64), np.zeros(sec.nq, dtype=np.int64)
        for q2 in sec.qlab:
            kf = int(sec.red(q2 - ke))
            kd = int(sec.red(kxl - kf))
            if kf in sec.V and kd in sec.V:
                todo.append((q2, kf, kd, n))
                off[q2 - sec.qmin], nd[q2 - sec.qmin] = n, len(sec.V[kd])
                n += len(sec.Q2[q2]) * ne * len(sec.V[kd])
        m3 = np.empty(max(n, 1), dtype=dtype)
        for q2, kf, kd, o in todo:
            W = sec.w_block(q.vpp, q2, ke)
            tb = np.ascontiguousarray(t[np.ix_(sec.V[kf], sec.V[kd])], dtype=dtype)
            gemm(W, tb, m3[o:o + W.shape[0] * tb.shape[1]].reshape(W.shape[0], tb.shape[1]))
        tri = sec.tri[T]
        blk = np.empty((ne, len(tri)), dtype=dtype)
        fn(_INT(sec.nvir), P._p(kv32), _INT(sec.modulus), P._p(tri), _I64(len(tri)), _INT(ne),
           _INT(sec.qmin), _INT(sec.nq), P._p(off), P._p(nd), P._p(sec.posV), P._p(sec.posQ), P._p(m3), P._p(blk))
        out[ke] = blk
    return out


def _stack(parts, ncol, dtype):
    if not parts:
        return np.zeros((0, ncol), dtype=dtype)
    return parts[0] if len(parts) == 1 else np.concatenate(parts)


def kernel(eris, t2, t3, max_memory=None, plan_=None, stride=1, progress=None):
    '''(E[Q], E(Q)) as zccsdt.cc.quadruples.kernel, in the symmetry sectors
    of ``eris.labels`` (``stride``, ``progress``: as there).'''
    from zccsdt import memory
    no = eris.nocc
    nv = np.asarray(eris.fock).shape[0] - no
    if no < 4 or nv < 4:
        return 0.0, 0.0
    dtype = np.result_type(t2, t3, eris.oovv)
    isz = np.dtype(dtype).itemsize
    if plan_ is None:
        plan_ = plan(eris, isz, memory.budget(max_memory))
    sec, n22, n13 = plan_.sectors, plan_.n22, plan_.n13
    quads = P.tuples(no, 4)
    q = Q.Quadruples(eris, t2, t3, plan_.dense)
    B1, B2, B3 = (np.empty(n22, dtype=dtype) for _ in range(3))
    C1, C2 = (np.empty(n13, dtype=dtype) for _ in range(2))
    ko = sec.ko
    kpair = sec.red(ko[P.tuples(no, 2)].sum(axis=1))
    xl = P.tuples(no, 2)
    ev = np.ascontiguousarray(np.asarray(eris.mo_energy)[no:], dtype=np.float64)
    fn = getattr(P._lib, 'zc_q_sector_energy_' + P._SFX[np.dtype(dtype)])
    p_ = P._p
    kv32 = np.ascontiguousarray(sec.kv, dtype=np.int32)
    so13, so22, so12, T3 = q.so13, q.so22, q.so12, q.T3
    rg = getattr(P._lib, 'zc_rows_gemm_' + P._SFX[np.dtype(dtype)])
    nA3 = T3.shape[1]
    e = np.zeros(2)
    XV, last = {}, None
    for l, xs, Is in Q.walk(q.schedule(), stride, progress):
        with timer('qs.xv'):
            if last != l:
                XV, last = {}, l
            for x in [x for x in XV if x not in xs]:
                del XV[x]
            for x in xs:
                if int(x) not in XV:
                    XV[int(x)] = xv_blocks(sec, q, int(x), l)
        for I in Is:
            M = int(sec.red(ko[quads[I]].sum()))
            b22, _, b13, _, off22, nc22, off13, nc13 = sec.layout(M)
            x13, J13, s13 = so13.idx1[I], so13.idx2[I], so13.sgn[I]
            p22, q22, s22 = so22.idx1[I], so22.idx2[I], so22.sgn[I]
            with timer('qs.22'):
                for qq, off, nr, nc in b22:
                    rq, cq = sec.Q2[qq], sec.Q2[int(sec.red(M - qq))]
                    LA, RA, LB, RB, LZ, RZ = [], [], [], [], [], []
                    for k in range(4):
                        x, J, s = int(x13[k]), int(J13[k]), float(s13[k])
                        ke = int(sec.red(qq - ko[x]))
                        if ke in sec.V:             # <ab||e x> t(J; e, cd)
                            E = sec.V[ke]
                            col, sg = sec.t3_rows(ke, int(sec.red(M - qq)))
                            LA.append(-s * q.vvvo_t[x][np.ix_(E, rq)])
                            RA.append(T3[J][col] * sg)
                        if ke in sec.O:             # t(x m, ab) Xo(m; J; cd)
                            ms = sec.O[ke]
                            LB.append(s * q.t2_xp[x][np.ix_(ms, rq)])
                            RB.append(q.Xo[J][np.ix_(ms, cq)])
                    for k in range(6):              # t(ij, ab) <cd||kl>
                        if kpair[p22[k]] == qq:
                            LZ.append(float(s22[k]) * q.t2_pp[p22[k]][rq][None, :])
                            RZ.append(q.vvoo_t[q22[k]][cq][None, :])
                    for Ls, Rs, B in ((LA, RA, B1), (LB, RB, B2), (LZ, RZ, B3)):
                        X = B[off:off + nr * nc].reshape(nr, nc)
                        if Ls:
                            gemm(np.ascontiguousarray(_stack(Ls, nr, dtype).T), _stack(Rs, nc, dtype), X)
                        else:
                            X[...] = 0
            with timer('qs.13'):
                for pp, off, nr, nc in b13:
                    va, a3 = sec.V[pp], sec.A3[int(sec.red(M - pp))]
                    LC, rows, sgs = [], [], []
                    for k in range(6):              # <m a||ij> t(m kl; bcd)
                        km = int(sec.red(kpair[p22[k]] - pp))
                        if km in sec.O:
                            ms = sec.O[km]
                            sg = so12.psg[ms, q22[k]]
                            ms = ms[sg != 0]
                            if len(ms):
                                LC.append(float(s22[k]) * q.ovoo_t[p22[k]][np.ix_(ms, va)])
                                rows.append(so12.pos[ms, q22[k]])
                                sgs.append(sg[sg != 0])
                    X = C1[off:off + nr * nc].reshape(nr, nc)
                    if LC:
                        L = np.ascontiguousarray(_stack(LC, nr, dtype))
                        rows = np.ascontiguousarray(np.concatenate(rows), dtype=np.int64)
                        sgs = np.ascontiguousarray(np.concatenate(sgs), dtype=np.float64)
                        cols = np.ascontiguousarray(a3, dtype=np.int64)
                        rg(p_(T3), _I64(nA3), p_(rows), p_(sgs), _I64(len(rows)), p_(cols), _I64(nc),
                           p_(L), _I64(nr), p_(X))
                    else:
                        X[...] = 0
                    X = C2[off:off + nr * nc].reshape(nr, nc)
                    first = True
                    for k in (0, 1, 3):             # 2 t(ij, a e) Xv(e; kl; bcd), l in the second pair
                        ke = int(sec.red(kpair[p22[k]] - pp))
                        if ke in sec.V:
                            L = np.ascontiguousarray(q.t2_px[p22[k]][np.ix_(sec.V[ke], va)].T)
                            gemm(L, XV[int(xl[q22[k], 0])][ke], X, alpha=2.0 * float(s22[k]),
                                 beta=0.0 if first else 1.0)
                            first = False
                    if first:
                        X[...] = 0
            with timer('qs.sum'):
                fn(_INT(nv), p_(kv32), _INT(sec.modulus), _INT(M), p_(ev), _DBL(float(q.eo4[I])),
                   _INT(sec.kmin), _INT(sec.nk), p_(sec.lid_of), p_(sec.lstart), p_(sec.lmem),
                   _INT(sec.qmin), _INT(sec.nq), p_(off22), p_(nc22), p_(off13), p_(nc13),
                   p_(sec.posV), p_(sec.posQ), p_(sec.pos3), p_(B1), p_(B2), p_(B3), p_(C1), p_(C2), p_(e))
    return float(e[0]), float(e[0] + e[1])
