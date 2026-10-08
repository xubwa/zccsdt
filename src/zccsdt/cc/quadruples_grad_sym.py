'''The first derivatives of (Q) in symmetry sectors: zccsdt.cc.quadruples_grad
for spinors that carry a label (zccsdt.cc.quadruples_sym: 2 m_j, or a label
modulo a number; one label is the case without symmetry).

The pass is that of quadruples_sym.kernel with every product followed by its
two adjoints.  For an occupied quadruple of total label M the five products
are formed in their blocks; one kernel (zc_qg_sector_vectors, csrc/tnpack.c)
walks through the virtual quadruples of the sector, sums the splits of each -
r and z -, and writes the left-hand quadruples conj(y), conj(w) and y back to
the places the splits were read from: the expanded left-hand quadruples in
the blocks of the products.  Then, block by block, X_ B^T and A^T X_ on the
rows and columns that were picked for the product X = A B:

    X[(ab), (cd)], label q of (ab)      s2 and the density of <cd||kl>; the
                                        density of <ab||ei> and s3; s2 and the
                                        adjoint of Xo
    X[a, (bcd)], label p of a           the density of <ma||ij> and s3; s2 and
                                        the adjoint of Xv; Gamma of the
                                        virtual spinors of label p

The adjoint of Xv is in the blocks of Xv (the e of one label, the triples of
the label that goes with it) and goes back through the blocks <bc||ef>
t(xl, fd) when Xv of the pair leaves (zc_xv_sector_adj): s2, and the density
of <bc||ef>, which has the label of (bc) on (ef) - it is kept in the diagonal
blocks of the labels of the pairs (quadruples_grad.PairBlocks), sum |Q2(q)|^2
elements instead of C(v,2)^2.  The adjoint of Xo is small and goes back
through the dense code.

What is returned is a quadruples_grad.Gradient with the arrays of the dense
pass (s3 on the packed block) but for the density of <bc||ef>.

Timers: 'qgs.xv', 'qgs.22', 'qgs.13' (the products), 'qgs.vectors' (the
kernel), 'qgs.back22', 'qgs.back13' (the adjoints), 'qgs.backxv', 'qg.xo'.
'''

import ctypes
from dataclasses import dataclass
from math import comb

import numpy as np

from zccsdt.blas import gemm
from zccsdt.cc import quadruples as Q, quadruples_grad as QG, quadruples_sym as S, t4pack as P
from zccsdt.timing import timer

_I64, _INT, _DBL = ctypes.c_int64, ctypes.c_int, ctypes.c_double


def available(eris):
    '''Whether the container has labels and the kernels are built.'''
    return S.available(eris) and hasattr(P._lib, 'zc_qg_sector_vectors_d')


@dataclass
class Plan:
    need: int               # bytes besides T3 and the integrals
    kept: int               # of which are the derivatives that are returned
    energy: object          # the quadruples_sym.Plan of the products
    sectors: object         # Sectors
    n22: int                # elements of the largest set of blocks X[(ab), (cd)]
    n13: int                # and X[a, (bcd)]


def pair_block_elems(sec):
    '''Elements of the diagonal blocks of the labels of the virtual pairs.'''
    return sum(len(x) ** 2 for x in sec.Q2.values())


def _own(sec, n22, n13, isz):
    '''Bytes (kept, accumulators, work of a quadruple, work of the adjoint of
    Xv for one pair, work at the end) of what the derivative has besides the
    arrays of (Q) in sectors.'''
    no, nv = sec.nocc, sec.nvir
    nP, nQ, nI3, nA3, _ = Q._sizes(no, nv)
    kept = (nI3 * nA3 + pair_block_elems(sec) + no * no * nv * nv + no + nv * nv + nP * nQ + no * nv * nQ
            + nP * no * nv + no * no * nP + no * nv * nv * no)
    acc = nP * nQ + no * no * nQ + nP * nv * nv + no * no * nv * nv + nI3 * no * nQ
    q2 = max(len(x) for x in sec.Q2.values())
    a3 = max(len(x) for x in sec.A3.values())
    # the stacked operands of all blocks of a quadruple, their adjoints for one block, the
    # rows of T3 gathered for a block and their adjoint
    stacks = (8 * (nv + no) + 12) * nQ + 2 * (4 * (nv + no) + 6) * q2 + 12 * no * (a3 + nv)
    work = 2 * n22 + 3 * n13 + stacks
    slab, m3 = sec.xv_size()
    wmax = max([len(sec.Q2[q]) * len(sec.V[k]) * len(sec.V[int(sec.red(q - k))])
                for q in sec.qlab for k in sec.vlab if int(sec.red(q - k)) in sec.V] or [0])
    xvb = m3 + wmax + nv * nv
    back = max(2 * no * nP * nQ + 3 * nP * nv * nv + no * nP * no, 2 * nI3 * nI3 + 2 * nv * nQ)
    return isz * kept, isz * acc, isz * work, isz * xvb, isz * back, isz * slab


def plan(eris, isz, budget=None):
    '''The Plan for ``budget`` bytes (None: no limit) besides T3 and the
    integrals; MemoryError if the blocks do not fit.  Counted besides the
    plan of (Q) in sectors: what is returned (s3 on the packed block, the
    density of <bc||ef> in the blocks of the pair labels, the small
    densities), the accumulators of s2 and the adjoint of Xo, five expanded
    blocks of a quadruple with the stacked operands of its products, and the
    adjoint of Xv next to Xv with the work of its way back.'''
    full = S.plan(eris, isz)
    sec = full.sectors
    kept, acc, work, xvb, back, slab = _own(sec, full.n22, full.n13, isz)
    xfull = max(3, sec.nocc - 1)
    mine = kept + acc + max(work + xvb, back) + xfull * slab
    pl, failed = full, False
    if budget is not None:
        try:
            pl = S.plan(eris, isz, budget - mine)
        except MemoryError:
            failed = True
    if failed:
        raise MemoryError('the derivatives of (Q) in symmetry sectors need %.1f MB besides T3 and the '
                          'integrals with everything at its largest (%.1f MB of them for the derivatives '
                          'themselves); %.1f MB are left'
                          % ((full.need + mine) / 1e6, kept / 1e6, budget / 1e6))
    return Plan(pl.need + mine, kept, pl, sec, pl.n22, pl.n13)


def _tables(sec, q2, ke):
    '''(places, signs) of the pairs (e, f) in the class of the label ``q2``,
    for the e of label ``ke`` and the f of the label that is left, [e f]
    flat: the adjoint of Sectors.w_block.'''
    key = ('d', q2, ke)
    if key not in sec._t3tab:
        nv = sec.nvir
        e, f = sec.V[ke][:, None], sec.V[int(sec.red(q2 - ke))][None, :]
        lo, hi = np.minimum(e, f), np.maximum(e, f)
        rank = comb(nv, 2) - 1 - (nv - 1 - lo) * (nv - 2 - lo) // 2 - (nv - 1 - hi)
        sgn = (np.sign(f - e) + 0 * rank).ravel().astype(np.float64)
        pos = np.where(sgn != 0, sec.posQ[np.where(sgn != 0, rank.ravel(), 0)], 0).astype(np.int64)
        sec._t3tab[key] = (np.ascontiguousarray(pos), np.ascontiguousarray(sgn))
    return sec._t3tab[key]


def _scatter(R, idx, sg, X):
    '''R[r, idx[j]] += sg[j] X[r, j] (R, X: one row or the same number of rows).'''
    R2 = R.reshape(1, -1) if R.ndim == 1 else R
    getattr(P._lib, 'zc_tn_scatter_cols_' + P._SFX[R.dtype])(
        P._p(R2), _I64(R2.shape[1]), _I64(R2.shape[0]), P._p(idx), P._p(sg), _I64(len(idx)), P._p(X))


def xv_back(sec, q, x, l, bars, G2, D):
    '''The adjoint of quadruples_sym.xv_blocks for the occupied pair (x, l):
    ``bars`` = {label of e: the adjoint of its block of Xv} goes to G2[x, l]
    (the coefficient of t(xl, fd)) and to the blocks ``D`` = {label: [(bc),
    (ef)]} of the density of <bc||ef>.'''
    dtype = q.dtype
    kxl = int(sec.red(sec.ko[x] + sec.ko[l]))
    t = np.asarray(q.t2[x, l])
    fn = getattr(P._lib, 'zc_xv_sector_adj_' + P._SFX[np.dtype(dtype)])
    kv32 = np.ascontiguousarray(sec.kv, dtype=np.int32)
    acc = np.zeros((sec.nvir, sec.nvir), dtype=dtype)
    for ke, bar in bars.items():
        T = int(sec.red(ke + kxl))
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
        m3 = np.zeros(max(n, 1), dtype=dtype)
        tri = sec.tri[T]
        fn(_INT(sec.nvir), P._p(kv32), _INT(sec.modulus), P._p(tri), _I64(len(tri)), _INT(ne),
           _INT(sec.qmin), _INT(sec.nq), P._p(off), P._p(nd), P._p(sec.posV), P._p(sec.posQ), P._p(m3), P._p(bar))
        for q2, kf, kd, o in todo:
            W = sec.w_block(q.vpp, q2, ke)                                      # [(bc) e, f]
            F, Dd = sec.V[kf], sec.V[kd]
            Mb = m3[o:o + W.shape[0] * len(Dd)].reshape(W.shape[0], len(Dd))    # [(bc) e, d]
            acc[np.ix_(F, Dd)] += gemm(W.T, Mb)
            tb = np.ascontiguousarray(t[np.ix_(F, Dd)], dtype=dtype)
            Wb = gemm(Mb, tb.T)                                                 # [(bc) e, f]
            pos, sgn = _tables(sec, q2, ke)
            nq2 = len(sec.Q2[q2])
            _scatter(D[q2], pos, sgn, Wb.reshape(nq2, ne * len(F)))
    G2[x, l] += acc


def kernel(eris, t2, t3, max_memory=None, plan_=None):
    '''The Gradient of (Q) as zccsdt.cc.quadruples_grad.kernel, in the
    symmetry sectors of ``eris.labels``; its density of <bc||ef> is a
    quadruples_grad.PairBlocks.'''
    from zccsdt import memory
    no = eris.nocc
    nv = np.asarray(eris.fock).shape[0] - no
    if no < 4 or nv < 4:
        raise ValueError('(Q) needs four occupied and four virtual spin orbitals')
    dtype = np.dtype(np.result_type(t2, t3, eris.oovv))
    isz = dtype.itemsize
    if plan_ is None:
        plan_ = plan(eris, isz, memory.budget(max_memory))
    sec, n22, n13 = plan_.sectors, plan_.n22, plan_.n13
    quads = P.tuples(no, 4)
    q = Q.Quadruples(eris, t2, t3, plan_.energy.dense)
    nP, nQ, nI3, nA3, _ = Q._sizes(no, nv)
    z = lambda *shape: np.zeros(shape, dtype=dtype)                 # noqa: E731
    # what is returned
    S3 = z(nI3, nA3)
    D = {k: z(len(idx), len(idx)) for k, idx in sec.Q2.items()}
    Dvvoo, Dvvvo, Dovoo = z(nP, nQ), z(no, nv, nQ), z(nP, no, nv)
    gamma, de_occ = z(nv, nv), np.zeros(no)
    # the coefficients of t2 in its layouts, and of Xo
    b_pp, b_xp, b_px, G2 = z(nP, nQ), z(no, no, nQ), z(nP, nv, nv), z(no, no, nv, nv)
    Xobar = z(nI3, no, nQ)
    B1, B2, B3, Ey, Ew = (np.empty(n22, dtype=dtype) for _ in range(5))
    C1, C2, Fw, Fy, Fs = (np.empty(n13, dtype=dtype) for _ in range(5))
    ko = sec.ko
    kpair = sec.red(ko[P.tuples(no, 2)].sum(axis=1))
    xl = P.tuples(no, 2)
    ev = np.ascontiguousarray(np.asarray(eris.mo_energy)[no:], dtype=np.float64)
    sfx = P._SFX[dtype]
    fn = getattr(P._lib, 'zc_qg_sector_vectors_' + sfx)
    rget = getattr(P._lib, 'zc_rows_cols_get_' + sfx)
    radd = getattr(P._lib, 'zc_rows_cols_add_' + sfx)
    p_ = P._p
    kv32 = np.ascontiguousarray(sec.kv, dtype=np.int32)
    so13, so22, so12, T3 = q.so13, q.so22, q.so12, q.T3
    stack = S._stack
    e = np.zeros(2)
    XV, XB, last = {}, {}, None

    def leave(xs):
        for x in xs:
            with timer('qgs.backxv'):
                xv_back(sec, q, x, last, XB.pop(x), G2, D)
            del XV[x]

    for l, xs, Is in q.schedule():
        if last != l:
            leave(list(XV))
            last = l
        leave([x for x in XV if x not in xs])
        with timer('qgs.xv'):
            for x in xs:
                if int(x) not in XV:
                    XV[int(x)] = S.xv_blocks(sec, q, int(x), l)
                    XB[int(x)] = {k: np.zeros_like(b) for k, b in XV[int(x)].items()}
        for I in Is:
            M = int(sec.red(ko[quads[I]].sum()))
            b22, m22, b13, m13, off22, nc22, off13, nc13 = sec.layout(M)
            x13, J13, s13 = so13.idx1[I], so13.idx2[I], so13.sgn[I]
            p22, q22, s22 = so22.idx1[I], so22.idx2[I], so22.sgn[I]
            ops22, ops13 = [], []
            with timer('qgs.22'):
                for qq, off, nr, nc in b22:
                    rq, cq = sec.Q2[qq], sec.Q2[int(sec.red(M - qq))]
                    LA, RA, LB, RB, LZ, RZ = [], [], [], [], [], []
                    tA, tB, tZ = [], [], []
                    for k in range(4):
                        x, J, s = int(x13[k]), int(J13[k]), float(s13[k])
                        ke = int(sec.red(qq - ko[x]))
                        if ke in sec.V:             # <ab||e x> t(J; e, cd)
                            E = sec.V[ke]
                            col, sg = sec.t3_rows(ke, int(sec.red(M - qq)))
                            LA.append(-s * q.vvvo_t[x][np.ix_(E, rq)])
                            RA.append(T3[J][col] * sg)
                            tA.append((x, J, s, E, col, sg))
                        if ke in sec.O:             # t(x m, ab) Xo(m; J; cd)
                            ms = sec.O[ke]
                            LB.append(s * q.t2_xp[x][np.ix_(ms, rq)])
                            RB.append(q.Xo[J][np.ix_(ms, cq)])
                            tB.append((x, J, s, ms))
                    for k in range(6):              # t(ij, ab) <cd||kl>
                        if kpair[p22[k]] == qq:
                            LZ.append(float(s22[k]) * q.t2_pp[p22[k]][rq][None, :])
                            RZ.append(q.vvoo_t[q22[k]][cq][None, :])
                            tZ.append((int(p22[k]), int(q22[k]), float(s22[k])))
                    ops = []
                    for Ls, Rs, B in ((LA, RA, B1), (LB, RB, B2), (LZ, RZ, B3)):
                        X = B[off:off + nr * nc].reshape(nr, nc)
                        if Ls:
                            L, R = np.ascontiguousarray(stack(Ls, nr, dtype)), np.ascontiguousarray(stack(Rs, nc, dtype))
                            gemm(L.T, R, X)
                            ops.append((L, R))
                        else:
                            X[...] = 0
                            ops.append(None)
                    ops22.append((ops, tA, tB, tZ))
            with timer('qgs.13'):
                for pp, off, nr, nc in b13:
                    va, a3 = sec.V[pp], sec.A3[int(sec.red(M - pp))]
                    LC, rows, sgs, tC = [], [], [], []
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
                                tC.append((int(p22[k]), float(s22[k]), ms))
                    X = C1[off:off + nr * nc].reshape(nr, nc)
                    opC = None
                    if LC:
                        L = np.ascontiguousarray(stack(LC, nr, dtype))
                        rows = np.ascontiguousarray(np.concatenate(rows), dtype=np.int64)
                        sgs = np.ascontiguousarray(np.concatenate(sgs), dtype=np.float64)
                        cols = np.ascontiguousarray(a3, dtype=np.int64)
                        G = np.empty((len(rows), nc), dtype=dtype)          # the rows of T3 with their signs
                        rget(p_(T3), _I64(nA3), p_(rows), p_(sgs), _I64(len(rows)), p_(cols), _I64(nc), p_(G))
                        gemm(L.T, G, X)
                        opC = (L, G, rows, sgs, cols)
                    else:
                        X[...] = 0
                    X = C2[off:off + nr * nc].reshape(nr, nc)
                    first, tV = True, []
                    for k in (0, 1, 3):             # 2 t(ij, a e) Xv(e; kl; bcd), l in the second pair
                        ke = int(sec.red(kpair[p22[k]] - pp))
                        if ke in sec.V:
                            T2 = np.ascontiguousarray(q.t2_px[p22[k]][np.ix_(sec.V[ke], va)])   # [e, a]
                            xx, alpha = int(xl[q22[k], 0]), 2.0 * float(s22[k])
                            gemm(T2.T, XV[xx][ke], X, alpha=alpha, beta=0.0 if first else 1.0)
                            tV.append((int(p22[k]), xx, ke, alpha, T2))
                            first = False
                    if first:
                        X[...] = 0
                    ops13.append((opC, tC, tV))
            with timer('qgs.vectors'):
                for a, m in ((Ey, m22), (Ew, m22), (Fw, m13), (Fy, m13), (Fs, m13)):
                    P._lib.zc_tn_zero(p_(a), _I64(m * isz))
                e1 = e[1]
                fn(_INT(nv), p_(kv32), _INT(sec.modulus), _INT(M), p_(ev), _DBL(float(q.eo4[I])),
                   _INT(sec.kmin), _INT(sec.nk), p_(sec.lid_of), p_(sec.lstart), p_(sec.lmem),
                   _INT(sec.qmin), _INT(sec.nq), p_(off22), p_(nc22), p_(off13), p_(nc13),
                   p_(sec.posV), p_(sec.posQ), p_(sec.pos3), p_(B1), p_(B2), p_(B3), p_(C1), p_(C2),
                   p_(Ey), p_(Ew), p_(Fw), p_(Fy), p_(Fs), p_(e))
                de_occ[quads[I]] -= e[1] - e1
            with timer('qgs.back22'):
                for (qq, off, nr, nc), (ops, tA, tB, tZ) in zip(b22, ops22):
                    rq, cq = sec.Q2[qq], sec.Q2[int(sec.red(M - qq))]
                    ey = Ey[off:off + nr * nc].reshape(nr, nc)
                    ew = Ew[off:off + nr * nc].reshape(nr, nc)
                    if ops[2] is not None:          # z, with conj(y): s2 and the density of <cd||kl>
                        L, R = ops[2]
                        Lb, Rb = gemm(R, ey.T), gemm(L, ey)
                        for k, (p, c, s) in enumerate(tZ):
                            b_pp[p, rq] += s * Lb[k]
                            Dvvoo[c, cq] += Rb[k]
                    if ops[1] is not None:          # t2 Xo, with conj(w): s2 and the adjoint of Xo
                        L, R = ops[1]
                        Lb, Rb = gemm(R, ew.T), gemm(L, ew)
                        n = 0
                        for x, J, s, ms in tB:
                            b_xp[x][np.ix_(ms, rq)] += s * Lb[n:n + len(ms)]
                            Xobar[J][np.ix_(ms, cq)] += Rb[n:n + len(ms)]
                            n += len(ms)
                    if ops[0] is not None:          # <ab||ei> t3, with both: the density of <ab||ei> and s3
                        L, R = ops[0]
                        ey += ew
                        Lb, Rb = gemm(R, ey.T), gemm(L, ey)
                        n = 0
                        for x, J, s, E, col, sg in tA:
                            Dvvvo[x][np.ix_(E, rq)] -= s * Lb[n:n + len(E)]
                            _scatter(S3[J], col.reshape(-1), sg.reshape(-1), Rb[n:n + len(E)])
                            n += len(E)
            with timer('qgs.back13'):
                for (pp, off, nr, nc), (opC, tC, tV) in zip(b13, ops13):
                    va = sec.V[pp]
                    fw = Fw[off:off + nr * nc].reshape(nr, nc)
                    fy = Fy[off:off + nr * nc].reshape(nr, nc)
                    fs = Fs[off:off + nr * nc].reshape(nr, nc)
                    gamma[np.ix_(va, va)] += gemm(fw, fy.T)
                    if opC is not None:             # <ma||ij> t3, with both: the density of <ma||ij> and s3
                        L, G, rows, sgs, cols = opC
                        Lb = gemm(G, fs.T)                                  # [(split, m), a]
                        gemm(L, fs, G)
                        radd(p_(S3), _I64(nA3), p_(rows), p_(sgs), _I64(len(rows)), p_(cols), _I64(nc), p_(G))
                        n = 0
                        for p, s, ms in tC:
                            Dovoo[p][np.ix_(ms, va)] += s * Lb[n:n + len(ms)]
                            n += len(ms)
                    for p, xx, ke, alpha, T2 in tV:  # t2 Xv, with conj(w): s2 and the adjoint of Xv
                        b_px[p][np.ix_(sec.V[ke], va)] += alpha * gemm(XV[xx][ke], fw.T)
                        gemm(T2, fw, XB[xx][ke], alpha=alpha, beta=1.0)
            del ops22, ops13
    leave(list(XV))
    with timer('qg.xo'):
        Doooo, Dovvo = QG.back_xo(q, eris, Xobar, b_xp, G2)
    s2 = QG.sources2(no, nv, G2, b_pp, b_xp, b_px)
    pairs = QG.PairBlocks(nQ, dtype, [(sec.Q2[k], D[k]) for k in sorted(D)])
    return QG.Gradient(float(e[0]), s2, S3, de_occ, gamma, Dvvoo, Dvvvo, Dovoo, Doooo, Dovvo, pairs, plan_,
                       QG.t3_terms(S3, T3, no, nv))
