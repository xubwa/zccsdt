'''(Q)_Lambda in symmetry sectors: zccsdt.cc.quadruples_lambda for spinors
that carry a label of an Abelian group, the way zccsdt.cc.quadruples_sym is
(Q) there.

l2 and Lambda_3 have the labels of t2 and T3 (those of the upper indices sum
to those of the lower ones), so the two products of Lambda_3 fall into the
blocks of the [V,T3] products, with the same rows and columns picked - the
integral operands conjugated - and the product of l2 with <oo||vv> into
those of t2 with <vv||oo>.  The kernel that sums the splits of the virtual
quadruples of a sector (zc_ql_sector_energy, csrc/qlambda.c) takes the two
blocks more:

    r1 = B1 (22) + C1 (13),  r2 = B2 (22) + C2 (13)       as in (Q)
    z2 = B3 (22),            z3 = B4 (22) + C3 (13)
'''

import ctypes
from dataclasses import replace

import numpy as np

from zccsdt.blas import gemm
from zccsdt.cc import quadruples_lambda as QL, quadruples_sym as S, t4pack as P
from zccsdt.timing import timer

_I64, _INT, _DBL = ctypes.c_int64, ctypes.c_int, ctypes.c_double


def available(eris):
    '''Whether the container has labels and the kernels are built.'''
    return S.available(eris) and QL._lib is not None


def _extra(sec, n22, n13, isz):
    '''Bytes besides those of the plan of (Q) in sectors: the arrays of the
    left-hand side of the dense code, one block more of each kind and the
    rows picked for the product of Lambda_3.'''
    rows = 4 * 4 * sec.nvir * max(len(x) for x in sec.Q2.values()) * isz
    return QL._extra(sec.nocc, sec.nvir, isz) + (n22 + n13) * isz + rows


def plan(eris, isz, budget=None):
    '''The plan (zccsdt.cc.quadruples_sym.Plan) for ``budget`` bytes (None:
    no limit) besides T3, Lambda_3 and the integrals; MemoryError if the
    blocks and the operands do not fit.'''
    full = S.plan(eris, isz)
    x = _extra(full.sectors, full.n22, full.n13, isz)
    if budget is None:
        return replace(full, need=full.need + x)
    failed = False
    try:
        pl = S.plan(eris, isz, budget - x)
    except MemoryError:
        failed = True
    if failed:
        raise MemoryError('(Q)_Lambda in symmetry sectors: the blocks do not fit into %.0f MB '
                          '(%.0f MB with everything at its largest)' % (budget / 1e6, (full.need + x) / 1e6))
    return replace(pl, need=pl.need + x)


def kernel(eris, t2, t3, l2, l3, max_memory=None, plan_=None):
    '''(E[Q]_L, E(Q)_L) as zccsdt.cc.quadruples_lambda.kernel, in the
    symmetry sectors of ``eris.labels``.'''
    from zccsdt import memory
    no = eris.nocc
    nv = np.asarray(eris.fock).shape[0] - no
    if no < 4 or nv < 4:
        return 0.0, 0.0
    dtype = np.result_type(t2, t3, l2, l3, eris.oovv)
    isz = np.dtype(dtype).itemsize
    if plan_ is None:
        plan_ = plan(eris, isz, memory.budget(max_memory))
    sec, n22, n13 = plan_.sectors, plan_.n22, plan_.n13
    quads = P.tuples(no, 4)
    q = QL.QuadruplesLambda(eris, t2, t3, l2, l3, plan_.dense)
    B1, B2, B3, B4 = (np.empty(n22, dtype=dtype) for _ in range(4))
    C1, C2, C3 = (np.empty(n13, dtype=dtype) for _ in range(3))
    ko = sec.ko
    kpair = sec.red(ko[P.tuples(no, 2)].sum(axis=1))
    xl = P.tuples(no, 2)
    ev = np.ascontiguousarray(np.asarray(eris.mo_energy)[no:], dtype=np.float64)
    fn = getattr(QL._lib, 'zc_ql_sector_energy_' + P._SFX[np.dtype(dtype)])
    p_ = P._p
    kv32 = np.ascontiguousarray(sec.kv, dtype=np.int32)
    so13, so22, so12, T3, L3 = q.so13, q.so22, q.so12, q.T3, q.L3
    rg = getattr(P._lib, 'zc_rows_gemm_' + P._SFX[np.dtype(dtype)])
    nA3 = T3.shape[1]
    cj = np.conj if q.cplx else (lambda x: x)
    stack = S._stack
    e = np.zeros(2)
    XV, last = {}, None
    for l, xs, Is in q.schedule():
        with timer('qls.xv'):
            if last != l:
                XV, last = {}, l
            for x in [x for x in XV if x not in xs]:
                del XV[x]
            for x in xs:
                if int(x) not in XV:
                    XV[int(x)] = S.xv_blocks(sec, q, int(x), l)
        for I in Is:
            M = int(sec.red(ko[quads[I]].sum()))
            b22, _, b13, _, off22, nc22, off13, nc13 = sec.layout(M)
            x13, J13, s13 = so13.idx1[I], so13.idx2[I], so13.sgn[I]
            p22, q22, s22 = so22.idx1[I], so22.idx2[I], so22.sgn[I]
            with timer('qls.22'):
                for qq, off, nr, nc in b22:
                    rq, cq = sec.Q2[qq], sec.Q2[int(sec.red(M - qq))]
                    LA, RA, LB, RB, LZ, RZ, RL = [], [], [], [], [], [], []
                    for k in range(4):
                        x, J, s = int(x13[k]), int(J13[k]), float(s13[k])
                        ke = int(sec.red(qq - ko[x]))
                        if ke in sec.V:             # <ab||e x> t(J; e, cd) and <x e||ab> l(J; e, cd)
                            E = sec.V[ke]
                            col, sg = sec.t3_rows(ke, int(sec.red(M - qq)))
                            LA.append(-s * q.vvvo_t[x][np.ix_(E, rq)])
                            RA.append(T3[J][col] * sg)
                            RL.append(L3[J][col] * sg)
                        if ke in sec.O:             # t(x m, ab) Xo(m; J; cd)
                            ms = sec.O[ke]
                            LB.append(s * q.t2_xp[x][np.ix_(ms, rq)])
                            RB.append(q.Xo[J][np.ix_(ms, cq)])
                    for k in range(6):              # l(ij, ab) <kl||cd>
                        if kpair[p22[k]] == qq:
                            LZ.append(float(s22[k]) * q.l2_pp[p22[k]][rq][None, :])
                            RZ.append(q.oovv_t[q22[k]][cq][None, :])
                    A = np.ascontiguousarray(stack(LA, nr, dtype).T) if LA else None
                    X = B1[off:off + nr * nc].reshape(nr, nc)
                    Z = B4[off:off + nr * nc].reshape(nr, nc)
                    if LA:
                        gemm(A, stack(RA, nc, dtype), X)
                        gemm(cj(A), stack(RL, nc, dtype), Z)
                    else:
                        X[...] = 0
                        Z[...] = 0
                    for Ls, Rs, B in ((LB, RB, B2), (LZ, RZ, B3)):
                        X = B[off:off + nr * nc].reshape(nr, nc)
                        if Ls:
                            gemm(np.ascontiguousarray(stack(Ls, nr, dtype).T), stack(Rs, nc, dtype), X)
                        else:
                            X[...] = 0
            with timer('qls.13'):
                for pp, off, nr, nc in b13:
                    va, a3 = sec.V[pp], sec.A3[int(sec.red(M - pp))]
                    LC, rows, sgs = [], [], []
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
                    X = C1[off:off + nr * nc].reshape(nr, nc)
                    Z = C3[off:off + nr * nc].reshape(nr, nc)
                    if LC:
                        L = np.ascontiguousarray(stack(LC, nr, dtype))
                        rows = np.ascontiguousarray(np.concatenate(rows), dtype=np.int64)
                        sgs = np.ascontiguousarray(np.concatenate(sgs), dtype=np.float64)
                        cols = np.ascontiguousarray(a3, dtype=np.int64)
                        rg(p_(T3), _I64(nA3), p_(rows), p_(sgs), _I64(len(rows)), p_(cols), _I64(nc),
                           p_(L), _I64(nr), p_(X))
                        L = cj(L)
                        rg(p_(L3), _I64(nA3), p_(rows), p_(sgs), _I64(len(rows)), p_(cols), _I64(nc),
                           p_(L), _I64(nr), p_(Z))
                    else:
                        X[...] = 0
                        Z[...] = 0
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
            with timer('qls.sum'):
                fn(_INT(nv), p_(kv32), _INT(sec.modulus), _INT(M), p_(ev), _DBL(float(q.eo4[I])),
                   _INT(sec.kmin), _INT(sec.nk), p_(sec.lid_of), p_(sec.lstart), p_(sec.lmem),
                   _INT(sec.qmin), _INT(sec.nq), p_(off22), p_(nc22), p_(off13), p_(nc13),
                   p_(sec.posV), p_(sec.posQ), p_(sec.pos3), p_(B1), p_(B2), p_(B3), p_(B4),
                   p_(C1), p_(C2), p_(C3), p_(e))
    return float(e[0]), float(e[0] + e[1])
