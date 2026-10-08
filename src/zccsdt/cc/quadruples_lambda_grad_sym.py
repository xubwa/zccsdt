'''The first derivatives of (Q)_Lambda in symmetry sectors:
zccsdt.cc.quadruples_lambda_grad for spinors that carry a label
(zccsdt.cc.quadruples_sym: 2 m_j, or a label modulo a number; one label is
the case without symmetry), as zccsdt.cc.quadruples_grad_sym is that of
(Q).

The pass is that of quadruples_lambda_sym.kernel - the seven products of r
and z in their blocks - with every product followed by its two adjoints.
One kernel (zc_qlg_sector_vectors, csrc/qlambda.c) walks through the
virtual quadruples of the sector, sums the splits of each and writes y and
w back to the places the splits were read from: the expanded left-hand
quadruples in the blocks of the products, two on two pairs and two on an
index and a triple.  Then, block by block, X_ B^T and A^T X_ on the rows
and columns that were picked for the product X = A B - with w for the four
products of r, with y for the three of z:

    X[(ab), (cd)], label q of (ab)      w: the density of <ab||ei> and s3; s2
                                        and the adjoint of Xo
                                        y: q2 and the density of <kl||cd>; the
                                        density of <ie||ab> and q3
    X[a, (bcd)], label p of a           w: the density of <ma||ij> and s3; s2
                                        and the adjoint of Xv
                                        y: the density of <ij||ma> and q3
                                        w and y: Gamma of the virtual spinors
                                        of label p

The adjoint of Xv goes back through the blocks of <bc||ef> t(xl, fd) when Xv
of the pair leaves (quadruples_grad_sym.xv_back), the density of <bc||ef> is
kept in the diagonal blocks of the labels of the pairs, the adjoint of Xo
goes back through the dense code.

What is returned is a quadruples_lambda_grad.Gradient with the arrays of the
dense pass (s3 and q3 on the packed block) but for the density of <bc||ef>.

Timers: 'qgs.xv', 'qgs.22', 'qgs.13' (the products), 'qgs.vectors' (the
kernel), 'qgs.back22', 'qgs.back13' (the adjoints), 'qgs.backxv', 'qg.xo'.
'''

import ctypes

import numpy as np

from zccsdt.blas import gemm
from zccsdt.cc import (quadruples as Q, quadruples_grad as QG, quadruples_grad_sym as GS, quadruples_lambda as QL,
                       quadruples_lambda_grad as LG, quadruples_sym as S, t4pack as P)
from zccsdt.timing import timer

_I64, _INT, _DBL = ctypes.c_int64, ctypes.c_int, ctypes.c_double
Plan = GS.Plan


def available(eris):
    '''Whether the container has labels and the kernels are built.'''
    return GS.available(eris) and QL._lib is not None and hasattr(QL._lib, 'zc_qlg_sector_vectors_d')


def _own(sec, n22, n13, isz):
    '''Bytes (kept, everything else at its largest) of what the pass has
    besides the arrays of (Q) in sectors: what quadruples_grad_sym counts,
    and for the left-hand side q3, q2 and the coefficients of l2 on pairs,
    the arrays of the dense code (quadruples_lambda), one block more of
    each kind, and the stacked operands of the products of Lambda_3 with
    the rows of Lambda_3 gathered for a block.'''
    no, nv = sec.nocc, sec.nvir
    nP, nQ, nI3, nA3, _ = Q._sizes(no, nv)
    kept, acc, work, xvb, back, slab = GS._own(sec, n22, n13, isz)
    q2 = max(len(x) for x in sec.Q2.values())
    a3 = max(len(x) for x in sec.A3.values())
    kept += isz * (nI3 * nA3 + no * no * nv * nv)
    acc += isz * nP * nQ
    work += QL._extra(no, nv, isz) + isz * (n22 + 8 * nv * nQ + 8 * nv * q2 + 12 * no * (a3 + nv))
    xfull = max(3, no - 1)
    return kept, acc + max(work + xvb, back) + xfull * slab


def plan(eris, isz, budget=None):
    '''The Plan (quadruples_grad_sym.Plan) for ``budget`` bytes (None: no
    limit) besides T3, Lambda_3 and the integrals; MemoryError if the
    blocks do not fit.'''
    full = S.plan(eris, isz)
    sec = full.sectors
    kept, rest = _own(sec, full.n22, full.n13, isz)
    mine = kept + rest
    pl, failed = full, False
    if budget is not None:
        try:
            pl = S.plan(eris, isz, budget - mine)
        except MemoryError:
            failed = True
    if failed:
        raise MemoryError('the derivatives of (Q)_Lambda in symmetry sectors need %.1f MB besides T3, Lambda_3 '
                          'and the integrals with everything at its largest (%.1f MB of them for the '
                          'derivatives themselves); %.1f MB are left'
                          % ((full.need + mine) / 1e6, kept / 1e6, budget / 1e6))
    return Plan(pl.need + mine, kept, pl, sec, pl.n22, pl.n13)


def kernel(eris, t2, t3, l2, l3, max_memory=None, plan_=None):
    '''The Gradient of (Q)_Lambda as zccsdt.cc.quadruples_lambda_grad.kernel,
    in the symmetry sectors of ``eris.labels``; its density of <bc||ef> is a
    quadruples_grad.PairBlocks.'''
    from zccsdt import memory
    no = eris.nocc
    nv = np.asarray(eris.fock).shape[0] - no
    if no < 4 or nv < 4:
        raise ValueError('(Q)_Lambda needs four occupied and four virtual spin orbitals')
    dtype = np.dtype(np.result_type(t2, t3, l2, l3, eris.oovv))
    isz = dtype.itemsize
    if plan_ is None:
        plan_ = plan(eris, isz, memory.budget(max_memory))
    sec, n22, n13 = plan_.sectors, plan_.n22, plan_.n13
    quads = P.tuples(no, 4)
    q = QL.QuadruplesLambda(eris, t2, t3, l2, l3, plan_.energy.dense)
    nP, nQ, nI3, nA3, _ = Q._sizes(no, nv)
    z = lambda *shape: np.zeros(shape, dtype=dtype)                 # noqa: E731
    # what is returned
    S3, Q3 = z(nI3, nA3), z(nI3, nA3)
    D = {k: z(len(idx), len(idx)) for k, idx in sec.Q2.items()}
    Dvvoo, Dvvvo, Dovoo = z(nP, nQ), z(no, nv, nQ), z(nP, no, nv)
    gamma, de_occ = z(nv, nv), np.zeros(no)
    # the coefficients of t2 and l2 in their layouts, and of Xo
    bq_pp, b_xp, b_px, G2 = z(nP, nQ), z(no, no, nQ), z(nP, nv, nv), z(no, no, nv, nv)
    Xobar = z(nI3, no, nQ)
    B1, B2, B3, B4, Ey, Ew = (np.empty(n22, dtype=dtype) for _ in range(6))
    C1, C2, C3, Fw, Fy = (np.empty(n13, dtype=dtype) for _ in range(5))
    ko = sec.ko
    kpair = sec.red(ko[P.tuples(no, 2)].sum(axis=1))
    xl = P.tuples(no, 2)
    ev = np.ascontiguousarray(np.asarray(eris.mo_energy)[no:], dtype=np.float64)
    sfx = P._SFX[dtype]
    fn = getattr(QL._lib, 'zc_qlg_sector_vectors_' + sfx)
    rget = getattr(P._lib, 'zc_rows_cols_get_' + sfx)
    radd = getattr(P._lib, 'zc_rows_cols_add_' + sfx)
    p_ = P._p
    kv32 = np.ascontiguousarray(sec.kv, dtype=np.int32)
    so13, so22, so12, T3, L3 = q.so13, q.so22, q.so12, q.T3, q.L3
    cj = np.conj if q.cplx else (lambda x: x)
    stack = S._stack
    scatter = GS._scatter
    e = np.zeros(3)
    XV, XB, last = {}, {}, None

    def leave(xs):
        for x in xs:
            with timer('qgs.backxv'):
                GS.xv_back(sec, q, x, last, XB.pop(x), G2, D)
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
                    LA, RA, RL, LB, RB, LZ, RZ = [], [], [], [], [], [], []
                    tA, tB, tZ = [], [], []
                    for k in range(4):
                        x, J, s = int(x13[k]), int(J13[k]), float(s13[k])
                        ke = int(sec.red(qq - ko[x]))
                        if ke in sec.V:             # <ab||e x> t(J; e, cd) and <x e||ab> l(J; e, cd)
                            E = sec.V[ke]
                            col, sg = sec.t3_rows(ke, int(sec.red(M - qq)))
                            LA.append(-s * q.vvvo_t[x][np.ix_(E, rq)])
                            RA.append(T3[J][col] * sg)
                            RL.append(L3[J][col] * sg)
                            tA.append((x, J, s, E, col, sg))
                        if ke in sec.O:             # t(x m, ab) Xo(m; J; cd)
                            ms = sec.O[ke]
                            LB.append(s * q.t2_xp[x][np.ix_(ms, rq)])
                            RB.append(q.Xo[J][np.ix_(ms, cq)])
                            tB.append((x, J, s, ms))
                    for k in range(6):              # l(ij, ab) <kl||cd>
                        if kpair[p22[k]] == qq:
                            LZ.append(float(s22[k]) * q.l2_pp[p22[k]][rq][None, :])
                            RZ.append(q.oovv_t[q22[k]][cq][None, :])
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
                    X = B4[off:off + nr * nc].reshape(nr, nc)
                    if LA:                          # the operand of [V,T3], conjugated, with the rows of Lambda_3
                        L, R = cj(ops[0][0]), np.ascontiguousarray(stack(RL, nc, dtype))
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
                    for k in range(6):              # <m a||ij> t(m kl; bcd) and <ij||m a> l(m kl; bcd)
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
                    Z = C3[off:off + nr * nc].reshape(nr, nc)
                    opC = opZ = None
                    if LC:
                        L = np.ascontiguousarray(stack(LC, nr, dtype))
                        rows = np.ascontiguousarray(np.concatenate(rows), dtype=np.int64)
                        sgs = np.ascontiguousarray(np.concatenate(sgs), dtype=np.float64)
                        cols = np.ascontiguousarray(a3, dtype=np.int64)
                        G = np.empty((len(rows), nc), dtype=dtype)          # the rows of T3 with their signs
                        rget(p_(T3), _I64(nA3), p_(rows), p_(sgs), _I64(len(rows)), p_(cols), _I64(nc), p_(G))
                        gemm(L.T, G, X)
                        opC = (L, G, rows, sgs, cols)
                        GL = np.empty((len(rows), nc), dtype=dtype)         # those of Lambda_3
                        rget(p_(L3), _I64(nA3), p_(rows), p_(sgs), _I64(len(rows)), p_(cols), _I64(nc), p_(GL))
                        Lc = cj(L)
                        gemm(Lc.T, GL, Z)
                        opZ = (Lc, GL)
                    else:
                        X[...] = 0
                        Z[...] = 0
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
                    ops13.append((opC, opZ, tC, tV))
            with timer('qgs.vectors'):
                for a, m in ((Ey, m22), (Ew, m22), (Fw, m13), (Fy, m13)):
                    P._lib.zc_tn_zero(p_(a), _I64(m * isz))
                e1 = e[1]
                fn(_INT(nv), p_(kv32), _INT(sec.modulus), _INT(M), p_(ev), _DBL(float(q.eo4[I])),
                   _INT(sec.kmin), _INT(sec.nk), p_(sec.lid_of), p_(sec.lstart), p_(sec.lmem),
                   _INT(sec.qmin), _INT(sec.nq), p_(off22), p_(nc22), p_(off13), p_(nc13),
                   p_(sec.posV), p_(sec.posQ), p_(sec.pos3), p_(B1), p_(B2), p_(B3), p_(B4), p_(C1), p_(C2), p_(C3),
                   p_(Ey), p_(Ew), p_(Fw), p_(Fy), p_(e))
                de_occ[quads[I]] -= e[1] - e1
            with timer('qgs.back22'):
                for (qq, off, nr, nc), (ops, tA, tB, tZ) in zip(b22, ops22):
                    rq, cq = sec.Q2[qq], sec.Q2[int(sec.red(M - qq))]
                    ey = Ey[off:off + nr * nc].reshape(nr, nc)
                    ew = Ew[off:off + nr * nc].reshape(nr, nc)
                    if ops[2] is not None:          # l2 <kl||cd>, with y: q2 and the density of <kl||cd>
                        L, R = ops[2]
                        Lb, Rb = gemm(R, ey.T), gemm(L, ey)
                        for k, (p, c, s) in enumerate(tZ):
                            bq_pp[p, rq] += s * Lb[k]
                            Dvvoo[c, cq] += cj(Rb[k])
                    if ops[1] is not None:          # t2 Xo, with w: s2 and the adjoint of Xo
                        L, R = ops[1]
                        Lb, Rb = gemm(R, ew.T), gemm(L, ew)
                        n = 0
                        for x, J, s, ms in tB:
                            b_xp[x][np.ix_(ms, rq)] += s * Lb[n:n + len(ms)]
                            Xobar[J][np.ix_(ms, cq)] += Rb[n:n + len(ms)]
                            n += len(ms)
                    if ops[0] is not None:
                        # <ab||ei> t3, with w: the density of <ab||ei> and s3
                        L, R = ops[0]
                        Lb, Rb = gemm(R, ew.T), gemm(L, ew)
                        # <ie||ab> l3, with y: the density of <ie||ab> = -conj(<ab||ei>) and q3
                        L, R = ops[3]
                        Lz, Rz = gemm(R, ey.T), gemm(L, ey)
                        n = 0
                        for x, J, s, E, col, sg in tA:
                            Dvvvo[x][np.ix_(E, rq)] -= s * (Lb[n:n + len(E)] + cj(Lz[n:n + len(E)]))
                            scatter(S3[J], col.reshape(-1), sg.reshape(-1), Rb[n:n + len(E)])
                            scatter(Q3[J], col.reshape(-1), sg.reshape(-1), Rz[n:n + len(E)])
                            n += len(E)
            with timer('qgs.back13'):
                for (pp, off, nr, nc), (opC, opZ, tC, tV) in zip(b13, ops13):
                    va = sec.V[pp]
                    fw = Fw[off:off + nr * nc].reshape(nr, nc)
                    fy = Fy[off:off + nr * nc].reshape(nr, nc)
                    gamma[np.ix_(va, va)] += gemm(fw, fy.T)
                    if opC is not None:
                        # <ma||ij> t3, with w: the density of <ma||ij> and s3
                        L, G, rows, sgs, cols = opC
                        Lb = gemm(G, fw.T)                                  # [(split, m), a]
                        gemm(L, fw, G)
                        radd(p_(S3), _I64(nA3), p_(rows), p_(sgs), _I64(len(rows)), p_(cols), _I64(nc), p_(G))
                        # <ij||ma> l3, with y: the density of <ij||ma> = conj(<ma||ij>) and q3
                        Lc, GL = opZ
                        Lz = gemm(GL, fy.T)
                        gemm(Lc, fy, GL)
                        radd(p_(Q3), _I64(nA3), p_(rows), p_(sgs), _I64(len(rows)), p_(cols), _I64(nc), p_(GL))
                        n = 0
                        for p, s, ms in tC:
                            Dovoo[p][np.ix_(ms, va)] += s * (Lb[n:n + len(ms)] + cj(Lz[n:n + len(ms)]))
                            n += len(ms)
                    for p, xx, ke, alpha, T2 in tV:  # t2 Xv, with w: s2 and the adjoint of Xv
                        b_px[p][np.ix_(sec.V[ke], va)] += alpha * gemm(XV[xx][ke], fw.T)
                        gemm(T2, fw, XB[xx][ke], alpha=alpha, beta=1.0)
            del ops22, ops13
    leave(list(XV))
    with timer('qg.xo'):
        Doooo, Dovvo = QG.back_xo(q, eris, Xobar, b_xp, G2)
    s2 = QG.sources2(no, nv, G2, 0.0, b_xp, b_px)
    q2 = LG.pairs_to_dense(bq_pp, no, nv)
    pairs = QG.PairBlocks(nQ, dtype, [(sec.Q2[k], D[k]) for k in sorted(D)])
    return LG.Gradient(float(e[0]), s2, S3, de_occ, gamma, Dvvoo, Dvvvo, Dovoo, Doooo, Dovvo, pairs, plan_,
                       LG.t3_terms(S3, T3, Q3, L3, no, nv), None, q2, Q3, float(e[2]))
