'''CCSDT(Q) on spin orbitals or spinors, real or complex: the (Q) correction
from the amplitudes of CCSDT, with T3 on its unique block (i<j<k, a<b<c).

    E[Q] = sum conj(z) r / D,    r = <Q| [V,T3] + 1/2 [[V,T2],T2] |0>,
                                 z = <Q| V T2 |0>   (the disconnected product),
    E(Q):                        z = <Q| V (T2 + T3) |0>

summed over the unique quadruples i<j<k<l, a<b<c<d, D the differences of the
orbital energies (Bomble, Stanton, Kallay and Gauss); bare integrals,
canonical orbitals, T1 does not enter.  The terms are those of
zccsdt.cc.ccsdtq_terms.

Nothing of the size of T4 is kept: for one occupied quadruple at a time, r
and z over all virtual quadruples are five matrix products, each followed by
the antisymmetrizing reduction onto a<b<c<d (zccsdt.cc.t4pack):

    X[(ab),(cd)]  - sum_e <ab||ei> t(jkl,cde)             (i|jkl), 4 splits     [V,T3]
                  + sum_m t(im,ab) Xo(m;jkl;cd)           (i|jkl)               T2 T2
                  + t(ij,ab) <cd||kl>                     (ij|kl), 6 splits     z
    X[a,(bcd)]    + sum_m <ma||ij> t(mkl,bcd)             (ij|kl), 6 splits     [V,T3]
                  + 2 sum_e t(ij,ae) Xv(e;kl;bcd)         (ij|kl) with l in     T2 T2
                                                          the second pair, 3

the splits of a quadruple stacked along the summed index.  The intermediates
of T2 with one integral:

    Xo(m;jkl;cd) = P(l/jk) [ 1/2 sum_n <mn||jk> t(nl,cd)
                             + sum_e ( <md||el> t(jk,ce) - <mc||el> t(jk,de) ) ]
    Xv(e;kl;bcd) = P(d/bc) 1/2 sum_f <bc||ef> t(kl,fd)

Xo is small (o^4 v^2 / 12).  Xv has v^4 / 6 elements per pair of occupied
indices: the product of two T2 through <vv||vv> is the same for (ij|kl) and
(kl|ij), so only the three splits with the largest index l in the second
pair are taken, twice - and Xv is needed for the pairs (x, l) of one l at a
time, each formed once.

Cost: o^4 v^4 (v + o) / 12 multiplications in the products, o^2 v^5 / 4 for Xv.

Memory (plan()): besides T3 and the packed <vv||vv>, which are given, the
products of one batch of columns (v^4 / 4 and v^4 / 6 elements at most) and
Xv for the x < l (up to (o - 1) v^4 / 6).  With less memory the products are
done in batches of columns, and Xv for a few x at a time: the quadruples of
one l are then taken in groups by the thirds of the x that their other
three indices fall in.
'''

from dataclasses import dataclass
from math import comb

import numpy as np

from zccsdt.blas import gemm
from zccsdt.cc import t3pack, t4pack as P
from zccsdt.cc.ccsdt_opt import vvvv_packed
from zccsdt.timing import timer

BATCH_BYTES = 1 << 30           # a product of one batch of columns, at most
PASSES = 16                     # batches of columns of a product, at most
SLACK = 1 << 20                 # Python objects and small arrays


def _E(sub, *ops):
    return np.einsum(sub, *ops, optimize=True)


def _pairs(n):
    t = P.tuples(n, 2)
    return t[:, 0], t[:, 1]


def _batches(n, step):
    step = int(max(1, step))
    return [(x, min(n, x + step)) for x in range(0, n, step)]


def check_canonical(eris, tol=1e-7):
    f = np.asarray(eris.fock)
    off = abs(f - np.diag(f.diagonal())).max() if f.size else 0.0
    if off > tol:
        raise NotImplementedError('perturbative corrections need canonical orbitals '
                                  '(largest off-diagonal element of the Fock matrix: %.1e)' % off)


# ---------------------------------------------------------------- memory

@dataclass
class Plan:
    cols22: int         # columns (cd) of one product X[(ab),(cd)]
    cols13: int         # columns (bcd) of one product X[a,(bcd)]
    xv: int             # occupied indices x whose Xv(x, l) are held at a time
    rows: int           # rows (bc) of <bc||ef> in one step of the formation of Xv
    need: int           # bytes of the arrays of the calculation (without T3 and <vv||vv>)
    fixed: int          # of which do not depend on the plan
    floor: int          # the least that a calculation takes


def _sizes(no, nv):
    return comb(no, 2), comb(nv, 2), comb(no, 3), comb(nv, 3), comb(nv, 4)


def _fixed(no, nv, isz):
    '''Bytes of the arrays whose size does not depend on the plan: (those
    kept from the start, the work arrays of a quadruple, the arrays that
    exist only while Xo is formed - before the work arrays are there).'''
    nP, nQ, nI3, nA3, nA = _sizes(no, nv)
    amps = 2 * nP * nQ + no * no * nQ + nP * nv * nv + nI3 * no * nQ      # T2 in three layouts, <vv||oo>, Xo
    ints = no * nv * nQ + nP * no * nv                                    # <vv||vo>, <ov||oo>
    # the tables of the occupied tuples and of their splits, the sums of orbital energies
    tables = comb(no, 4) * 210 + 9 * (no * nI3 + nP * nP + no * nP) + 51 * nI3 + 8 * nA
    work = 8 * (nv + no) * nQ + 6 * no * (nv + nA3) + 3 * nA              # stacked operands, r and z
    xo = no * nP * (nQ + no) + nP * nv * nv + 3 * nI3 * nQ + 2 * nP * nQ
    return isz * (amps + ints) + tables + SLACK, isz * work, isz * xo


def _variable(no, nv, isz, cols22, cols13, xv, rows):
    nP, nQ, nI3, nA3, nA = _sizes(no, nv)
    return isz * (nQ * cols22 + nv * cols13 + nv * xv * nA3 + nv * nv * (xv + rows * (1 + xv)))


def plan(no, nv, isz, budget=None, batch_bytes=None):
    '''The Plan for ``budget`` bytes (None: no limit).  The floor - below it
    MemoryError - is the products in PASSES batches of columns and Xv for
    three x; from there first Xv for more x, up to all, then the products in
    fewer batches (of ``batch_bytes`` at most), then more rows per step of
    the formation of Xv.'''
    batch_bytes = batch_bytes or BATCH_BYTES
    nP, nQ, nI3, nA3, nA = _sizes(no, nv)
    base, work, xo = _fixed(no, nv, isz)
    xfull = max(3, no - 1)
    top22 = int(max(1, min(nQ, batch_bytes // (isz * nQ))))
    top13 = int(max(1, min(nA3, batch_bytes // (isz * nv))))
    min22, min13 = -(-nQ // PASSES), -(-nA3 // PASSES)
    top22, top13 = max(top22, min22), max(top13, min13)
    top_rows = int(max(1, min(nQ, batch_bytes // (isz * nv * nv * (1 + xfull)))))
    least = work + _variable(no, nv, isz, min22, min13, 3, 1)
    floor = base + max(xo, least)
    if budget is None:
        cols22, cols13, xv, rows = top22, top13, xfull, top_rows
    else:
        if budget < floor:
            raise MemoryError('(Q) needs %.0f MB besides T3 and the integrals for %d occupied and %d '
                              'virtual spin orbitals; %.0f MB are left' % (floor / 1e6, no, nv, budget / 1e6))
        # one after the other, each to its top before the next starts (so that
        # no part of a plan shrinks when the budget grows)
        rest = budget - base - least
        per_x = isz * nv * (nA3 + 2 * nv)
        xv = int(min(xfull, 3 + rest // per_x))
        cols22, cols13, rows = min22, min13, 1
        if xv == xfull:
            rest -= (xv - 3) * per_x
            # the two products grow together, by the same fraction of what they lack
            lack = isz * (nQ * (top22 - min22) + nv * (top13 - min13))
            if rest >= lack:
                cols22, cols13 = top22, top13
                rows = int(min(top_rows, 1 + (rest - lack) // (isz * nv * nv * (1 + xv))))
            else:
                cols22 = int(min22 + rest / lack * (top22 - min22))
                cols13 = int(min13 + rest / lack * (top13 - min13))
    return explicit_plan(no, nv, isz, cols22, cols13, xv, rows)


def explicit_plan(no, nv, isz, cols22, cols13, xv, rows):
    '''The Plan with these batch sizes.'''
    nP, nQ, nI3, nA3, nA = _sizes(no, nv)
    base, work, xo = _fixed(no, nv, isz)
    min22, min13 = -(-nQ // PASSES), -(-nA3 // PASSES)
    floor = base + max(xo, work + _variable(no, nv, isz, min22, min13, 3, 1))
    need = base + max(xo, work + _variable(no, nv, isz, cols22, cols13, xv, rows))
    return Plan(cols22, cols13, xv, rows, need, base + work, floor)


# ------------------------------------------------------------- the correction

class Quadruples:
    '''r and z of (Q) for one occupied quadruple at a time.'''

    def __init__(self, eris, t2, t3, plan_=None, left_only=False):
        check_canonical(eris)
        self.nocc = no = eris.nocc
        self.nvir = nv = np.asarray(eris.fock).shape[0] - no
        self.dtype = dtype = np.result_type(t2, eris.oovv) if t3 is None else np.result_type(t2, t3, eris.oovv)
        self.isz = isz = np.dtype(dtype).itemsize
        self.plan = pl = plan_ or plan(no, nv, isz)
        i2, j2 = _pairs(no)
        a2, b2 = _pairs(nv)
        self.nQ, self.nA3, self.nA = nQ, nA3, nA = comb(nv, 2), comb(nv, 3), comb(nv, 4)
        mo_e = np.asarray(eris.mo_energy)
        self.quads = P.tuples(no, 4)
        self.eo4 = mo_e[:no][self.quads].sum(axis=1)
        self.ev4 = P.tuple_sums(mo_e[no:], 4)
        self.so13, self.so22 = P.split(no, 1, 3), P.split(no, 2, 2)
        self.t2_pp = np.ascontiguousarray(t2[i2, j2][:, a2, b2], dtype=dtype)           # [(ij), (ab)]
        x = np.asarray(eris.oovv)[i2, j2][:, a2, b2]                                    # <cd||kl> = <kl||cd>*
        self.vvoo_t = np.ascontiguousarray(np.conj(x, out=x), dtype=dtype)              # [(kl), (cd)]
        if left_only:
            self.w22 = np.empty(nQ * pl.cols22, dtype=dtype)
            self.z = np.empty(nA, dtype=dtype)
            return
        self.T3 = np.ascontiguousarray(np.asarray(t3).reshape(comb(no, 3), nA3), dtype=dtype)
        self.so12 = P.split(no, 1, 2)
        with timer('q.blocks'):
            # <ab||ei> = - <ie||ab>* as [i, e, (ab)],  <ma||ij> = <ij||ma>* as [(ij), m, a]
            x = np.asarray(eris.ovvv)[:, :, a2, b2]
            self.vvvo_t = np.ascontiguousarray(np.negative(np.conj(x, out=x), out=x), dtype=dtype)
            x = np.asarray(eris.ooov)[i2, j2]
            self.ovoo_t = np.ascontiguousarray(np.conj(x, out=x), dtype=dtype)
            self.vpp = vvvv_packed(eris, t3pack.triple_map(nv))[0]                      # [(bc), (ef)]
        self.t2 = t2
        self.t2_xp = np.ascontiguousarray(t2[:, :, a2, b2], dtype=dtype)                # [i, m, (ab)]
        self.t2_px = np.ascontiguousarray(t2[i2, j2].transpose(0, 2, 1), dtype=dtype)   # [(ij), e, a]
        with timer('q.xo'):
            # Xo[(jkl), m, (cd)], for one m at a time: G[l, (jk), (cd)] and P(l/jk) of it
            self.Xo = np.empty((comb(no, 3), no, nQ), dtype=dtype)
            oooo_t = np.ascontiguousarray(np.asarray(eris.oooo)[:, :, i2, j2].transpose(0, 2, 1), dtype=dtype)
            ovvo = np.asarray(eris.ovvo)
            t2_pc = np.ascontiguousarray(t2[i2, j2], dtype=dtype).reshape(len(i2) * nv, nv)     # [(jk) c, e]
            G = np.empty((no, len(i2), nQ), dtype=dtype)
            for m in range(no):
                for l in range(no):
                    # sum_e t(jk,ce) <md||el> as [(jk), c, d], then c < d of its antisymmetric part
                    K = gemm(t2_pc, np.ascontiguousarray(ovvo[m, :, :, l].T, dtype=dtype)).reshape(-1, nv, nv)
                    np.subtract(K[:, a2, b2], K[:, b2, a2], out=G[l])
                    # + 1/2 sum_n <mn||jk> t(nl,cd)
                    gemm(oooo_t[m], self.t2_xp[:, l], G[l], alpha=0.5, beta=1.0)
                self.Xo[:, m] = P.fold(G, self.so12, 0, 1)
            del G, K, oooo_t, t2_pc
        # the work arrays of one quadruple (after the arrays that Xo is formed with)
        self.w22 = np.empty(nQ * pl.cols22, dtype=dtype)
        self.z = np.empty(nA, dtype=dtype)
        self.w13 = np.empty(nv * pl.cols13, dtype=dtype)
        self.LA, self.RA = (np.empty((4 * nv, nQ), dtype=dtype) for _ in range(2))
        self.LB, self.RB = (np.empty((4 * no, nQ), dtype=dtype) for _ in range(2))
        self.LC, self.RC = np.empty((6 * no, nv), dtype=dtype), np.empty((6 * no, nA3), dtype=dtype)
        self.rt3, self.rtt = np.empty(nA, dtype=dtype), np.empty(nA, dtype=dtype)
        self._xv = (None, None, None)

    # ---- the intermediate of T2 with <vv||vv>, for pairs (x, l)

    def xi_v(self, l, xs):
        '''Xv[e, n, (bcd)] = P(d/bc) 1/2 sum_f <bc||ef> t(x l, fd) for
        x = xs[n], and {x: n}.'''
        key = (l, tuple(int(x) for x in xs))
        if self._xv[0] == key:
            return self._xv[1], self._xv[2]
        self._xv = (None, None, None)                   # (the old one goes before the new one comes)
        nv, nQ, nA3, dtype, nx = self.nvir, self.nQ, self.nA3, self.dtype, len(xs)
        with timer('q.xv'):
            out = np.empty((nv, nx, nA3), dtype=dtype)
            P.zero(out)
            tl = np.ascontiguousarray(self.t2[xs, l].transpose(1, 0, 2), dtype=dtype).reshape(nv, nx * nv)
            for b0, b1 in _batches(nQ, self.plan.rows):
                nb = b1 - b0
                W = np.empty((nb, nv, nv), dtype=dtype)                         # [(bc), e, f]
                P.tn_expand(W, np.ascontiguousarray(self.vpp[b0:b1], dtype=dtype), nv, 1, 1, outer=0)
                M = gemm(W.reshape(nb * nv, nv), tl)                            # [(bc), e, n, d]
                P.tn_reduce(out.reshape(nv * nx, nA3), M.reshape(nb, nv * nx, nv), nv, 2, 1,
                            rows=(b0, b1), scale=0.5, outer=1)
                del W, M
        self._xv = (key, out, {int(x): n for n, x in enumerate(xs)})
        return out, self._xv[2]

    def schedule(self):
        '''[(l, xs, quadruples)]: the occupied quadruples by their largest
        index l, and - where Xv cannot be held for all x < l - by the groups
        of x that their other indices fall in; xs: the x that Xv is formed
        for.'''
        out = []
        for l in range(3, self.nocc):
            Is = np.where(self.quads[:, 3] == l)[0]
            if self.plan.xv >= l:
                out.append((l, np.arange(l), Is))
                continue
            size = max(1, self.plan.xv // 3)
            grp = self.quads[Is, :3] // size
            for g in np.unique(grp, axis=0):
                xs = np.unique(np.concatenate([np.arange(x * size, min(l, (x + 1) * size)) for x in g]))
                out.append((l, xs, Is[(grp == g).all(axis=1)]))
        return out

    # ---- one occupied quadruple

    def _x22(self, c):
        return self.w22[:self.nQ * (c[1] - c[0])].reshape(self.nQ, c[1] - c[0])

    def left(self, I):
        '''z of [Q] for the occupied quadruple I: <Q| V T2 |0>, [C(nv,4)]
        (a work array: valid until the next call).'''
        so, nv, z = self.so22, self.nvir, self.z
        L = self.t2_pp[so.idx1[I]] * so.sgn[I][:, None]                         # [6, (ab)]
        Rz = self.vvoo_t[so.idx2[I]]                                            # [6, (cd)]
        P.zero(z)
        for c in _batches(self.nQ, self.plan.cols22):
            P.tn_reduce(z, gemm(L.T, Rz[:, c[0]:c[1]], self._x22(c)), nv, 2, 2, cols=c)
        return z

    def right(self, I, Xv, where):
        '''(r of [V,T3], r of 1/2 [[V,T2],T2]) for the occupied quadruple I
        (work arrays: valid until the next call); Xv, where: xi_v() for its
        largest index.'''
        no, nv, nQ, nA3, T3 = self.nocc, self.nvir, self.nQ, self.nA3, self.T3
        so13, so22, so12 = self.so13, self.so22, self.so12
        x13, J13, s13 = so13.idx1[I], so13.idx2[I], so13.sgn[I]
        p22, q22, s22 = so22.idx1[I], so22.idx2[I], so22.sgn[I]
        rt3, rtt = self.rt3, self.rtt
        P.zero(rt3)
        P.zero(rtt)
        with timer('q.operands'):
            # the splits (i|jkl) stacked along the summed index
            for k in range(4):
                x, J, s = x13[k], J13[k], int(s13[k])
                np.multiply(self.vvvo_t[x], -s, out=self.LA[k * nv:(k + 1) * nv])       # [e, (ab)]
                P.tn_expand(self.RA[k * nv:(k + 1) * nv], T3[J], nv, 1, 2)             # t(jkl; e, cd)
                np.multiply(self.t2_xp[x], s, out=self.LB[k * no:(k + 1) * no])         # [m, (ab)]
                self.RB[k * no:(k + 1) * no] = self.Xo[J]                               # [m, (cd)]
            # the splits (ij|kl)
            for k in range(6):
                np.multiply(self.ovoo_t[p22[k]], int(s22[k]), out=self.LC[k * no:(k + 1) * no])   # [m, a]
            P.tn_rows(self.RC, T3, so12.pos[:, q22].T.ravel(), so12.psg[:, q22].T.ravel())       # t(m kl; bcd)
        for c in _batches(nQ, self.plan.cols22):
            X = self._x22(c)
            with timer('q.gemm'):
                gemm(self.LA.T, self.RA[:, c[0]:c[1]], X)
            with timer('q.reduce'):
                P.tn_reduce(rt3, X, nv, 2, 2, cols=c)
            with timer('q.gemm'):
                gemm(self.LB.T, self.RB[:, c[0]:c[1]], X)
            with timer('q.reduce'):
                P.tn_reduce(rtt, X, nv, 2, 2, cols=c)
        # the splits whose second pair has the largest index: (ij|kl), (ik|jl), (jk|il)
        last = [(p22[k], where[int(P.tuples(no, 2)[q22[k], 0])], 2.0 * int(s22[k])) for k in (0, 1, 3)]
        for c in _batches(nA3, self.plan.cols13):
            X = self.w13[:nv * (c[1] - c[0])].reshape(nv, c[1] - c[0])
            with timer('q.gemm'):
                gemm(self.LC.T, self.RC[:, c[0]:c[1]], X)
            with timer('q.reduce'):
                P.tn_reduce(rt3, X, nv, 1, 3, cols=c)
            with timer('q.gemm'):
                for k, (p, n, alpha) in enumerate(last):
                    gemm(self.t2_px[p].T, Xv[:, n, c[0]:c[1]], X, alpha=alpha, beta=float(k > 0))
            with timer('q.reduce'):
                P.tn_reduce(rtt, X, nv, 1, 3, cols=c)
        return rt3, rtt


def kernel(eris, t2, t3, max_memory=None, plan_=None):
    '''(E[Q], E(Q)) from converged CCSDT amplitudes: t2[i,j,a,b] and T3
    packed [C(no,3), C(nv,3)] (or flat), on an integral container
    (zccsdt.eris.ERIs).  ``max_memory``: MB for the whole process (default:
    what the node and the job leave); ``plan_``: a Plan given explicitly.'''
    from zccsdt import memory
    no = eris.nocc
    nv = np.asarray(eris.fock).shape[0] - no
    if no < 4 or nv < 4:
        return 0.0, 0.0
    if plan_ is None:
        isz = np.dtype(np.result_type(t2, t3, eris.oovv)).itemsize
        plan_ = plan(no, nv, isz, memory.budget(max_memory))
    q = Quadruples(eris, t2, t3, plan_)
    e5 = e6 = 0.0
    for l, xs, Is in q.schedule():
        Xv, where = q.xi_v(l, xs)
        for I in Is:
            with timer('q.left'):
                z = q.left(I)
            rt3, rtt = q.right(I, Xv, where)
            with timer('q.energy'):
                e = P.energy2(z, rt3, rtt, q.ev4, q.eo4[I])
            e5, e6 = e5 + e[0], e6 + e[1]
        del Xv
    return float(e5), float(e5 + e6)


def rows(eris, t2, t3, plan_=None):
    '''(z, r of [V,T3], r of the T2 T2 terms) for all quadruples,
    [C(no,4), C(nv,4)] each: for tests and small cases.'''
    q = Quadruples(eris, t2, t3, plan_)
    out = [np.zeros((len(q.eo4), q.nA), dtype=q.dtype) for _ in range(3)]
    for l, xs, Is in q.schedule():
        Xv, where = q.xi_v(l, xs)
        for I in Is:
            out[0][I] = q.left(I)
            out[1][I], out[2][I] = q.right(I, Xv, where)
    return out


def second_order_quadruples(eris, t2, plan_=None):
    '''<Q| V T2 |0> / D on the unique block, [C(no,4), C(nv,4)]: the
    quadruples of the bra to second order (the left-hand side of (P)).'''
    q = Quadruples(eris, t2, None, plan_, left_only=True)
    out = np.empty((len(q.eo4), q.nA), dtype=q.dtype)
    for I in range(len(q.eo4)):
        out[I] = q.left(I) / (q.eo4[I] - q.ev4)
    return out
