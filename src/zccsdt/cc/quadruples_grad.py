'''The first derivatives of (Q) from packed amplitudes: the source terms of
the Lambda equations of CCSDT(Q) and the densities of (Q), for one occupied
quadruple at a time.

The functional and its derivatives are those of zccsdt.cc.qgrad_ref (the
dense reference; docs/lambda.md, section 11):

    E(Q) = Re sum conj(z) r / D,   y = r / D,   w = z / D,
    dE(Q) = Re sum [ conj(y) dz + conj(w) dr ] - sum Re(conj(w) y) dD.

z and r of a quadruple are the five matrix products of zccsdt.cc.quadruples,
each followed by an antisymmetrizing reduction onto a<b<c<d.  The derivative
runs through them backwards: the left-hand quadruples conj(y), conj(w) and
their sum are expanded to the layouts of the products (the adjoint of the
reduction), and every product X = A B gives two, X_ B^T for the operand A
and A^T X_ for the operand B:

    product of (Q)                             its two adjoints go to
    t(ij,ab) <cd||kl>                          s2;                the density of <cd||kl>
    - sum_e <ab||ei> t(jkl,cde)                the density of <ab||ei>;       s3
    sum_m <ma||ij> t(mkl,bcd)                  the density of <ma||ij>;       s3
    sum_m t(im,ab) Xo(m;jkl;cd)                s2;                the adjoint of Xo
    2 sum_e t(ij,ae) Xv(e;kl;bcd)              s2;                the adjoint of Xv

with conj(y) on the first, conj(y) + conj(w) on the two of [V,T3] and
conj(w) on the two of T2 T2.  The adjoints of the intermediates go back
through their definitions when they are complete: that of Xo (o^4 v^2 / 12,
kept for all quadruples) at the end - s2 and the densities of <mn||jk> and
<md||el> -, that of Xv for the pairs (x, l) when Xv of the pair leaves -
s2 and the density of <bc||ef>, with the packed <vv||vv> in slabs of rows as
Xv was made.

The orbital energies: -Re(conj(w) y) summed for every occupied index of the
quadruple; for the virtual ones the matrix

    Gamma[a,e] = sum conj(w[a,(bcd)]) y[e,(bcd)]

(one more product on the expanded w and y), whose diagonal is the derivative
with respect to e_a and whose Hermitian part is the rotation among the
virtual spin orbitals that follows from the Fock matrix staying diagonal
(qgrad_ref.fock_density).

Cost: every product of (Q) twice (o^4 v^4 (v + o) / 6 multiplications),
Gamma (o^4 v^5 / 144), Xv twice more (o^2 v^5 / 2).

What is kept besides the arrays of (Q): s3 (the size of T3), the density of
<bc||ef> on pairs (C(v,2)^2, the size of the packed <vv||vv>), the densities
of the other blocks (the largest: that of <ab||ei>, o v C(v,2)), the adjoint
of Xv next to Xv, and the work arrays of a quadruple three times.  plan()
plans all of it within a budget.

Timers (zccsdt.timing): those of (Q) for the forward products, 'qg.expand',
'qg.gemm', 'qg.reduce', 'qg.gamma', 'qg.scatter', 'qg.xv' (the adjoint of
Xv), 'qg.xo'.
'''

from dataclasses import dataclass
from math import comb

import numpy as np

from zccsdt.blas import gemm
from zccsdt.cc import quadruples as Q, t4pack as P
from zccsdt.timing import timer


@dataclass
class Gradient:
    '''E(Q) and its first derivatives.  The densities are the coefficients
    of the integrals in dE = Re sum (density x d integral), on the unique
    elements named.'''
    e_q: float
    s2: np.ndarray              # [i,j,a,b], antisymmetric: dE = Re sum_(i<j,a<b) s2 dt2
    s3: np.ndarray              # [C(o,3), C(v,3)]: dE = Re sum s3 dT3
    de_occ: np.ndarray          # [i]: dE / d e_i
    gamma_vv: np.ndarray        # [a,e] (see the module); its diagonal: dE / d e_a
    vvoo: np.ndarray            # [(k<l), (c<d)] of <cd||kl>
    vvvo: np.ndarray            # [i, e, (a<b)] of <ab||ei>
    ovoo: np.ndarray            # [(i<j), m, a] of <ma||ij>
    oooo: np.ndarray            # [m, n, (j<k)] of <mn||jk>
    ovvo: np.ndarray            # [m, d, e, l] of <md||el>
    vvvv: np.ndarray            # [(b<c), (e<f)] of <bc||ef>
    plan: object = None

    @property
    def de(self):
        '''dE / d e_p for all spin orbitals, occupied first.'''
        return np.concatenate((self.de_occ, self.gamma_vv.diagonal().real))


@dataclass
class Plan:
    q: Q.Plan                   # the batches of the products, as for (Q)
    need: int                   # bytes of all arrays (without T3 and the integrals)
    kept: int                   # of which are the derivatives that are returned
    floor: int


def _own(no, nv, isz):
    '''Bytes (kept, work) of what the derivative has besides the arrays of
    (Q) and what scales with its batches: the derivatives that are returned
    and the accumulators of s2; the vectors of a quadruple and the adjoints
    of the stacked operands.'''
    nP, nQ, nI3, nA3, nA = Q._sizes(no, nv)
    kept = (nI3 * nA3 + nQ * nQ + no * no * nv * nv + no + nv * nv + nP * nQ + no * nv * nQ
            + nP * no * nv + no * no * nP + no * nv * nv * no)
    acc = nP * nQ + no * no * nQ + nP * nv * nv + no * no * nv * nv + nI3 * no * nQ
    work = nA + (4 * nv + 8 * no) * nQ + 6 * no * nv + 24 * nQ
    return isz * kept, isz * (acc + work) + 8 * nA


def _need(no, nv, isz, qp):
    '''Bytes of everything with the batches of the Plan of (Q) ``qp``.'''
    nP, nQ, nI3, nA3, nA = Q._sizes(no, nv)
    base, work, xo = Q._fixed(no, nv, isz)
    kept, own = _own(no, nv, isz)
    nslot = max(3, min(qp.xv, no - 1))
    var = isz * (3 * nQ * qp.cols22 + 3 * nv * qp.cols13 + (4 * nv * qp.cols22 if 4 * nv > nQ else 0)
                 + 6 * no * qp.cols13 + 2 * nv * nslot * nA3 + nv * nv * (2 * nslot + 3 * qp.rows))
    # the adjoint of Xo at the end: its arrays for one m, with the accumulators still there
    back = isz * (2 * no * nP * nQ + 3 * nP * nv * nv + no * nP * no)
    return kept + base + max(xo, work + own + max(var, back)), kept


def plan(no, nv, isz, budget=None, batch_bytes=None):
    '''The Plan for ``budget`` bytes (None: no limit) besides T3 and the
    integrals.  The batches are those of a Plan of (Q): every work array of
    a batch is there three times (the product and two expanded left-hand
    quadruples) and Xv twice, so it is the Plan of (Q) for the largest
    budget of its own with which all of this fits (the plans of (Q) grow
    with their budget, and so does what is needed here: found by
    bisection).  MemoryError below the floor - the smallest batches of
    (Q).'''
    kept = _own(no, nv, isz)[0]
    top = Q.plan(no, nv, isz, None, batch_bytes)
    floor = _need(no, nv, isz, Q.explicit_plan(no, nv, isz, *Q._least(nv)[:2], 3, Q._least(nv)[2]))[0]
    qp = top
    if budget is not None and _need(no, nv, isz, top)[0] > budget:
        if budget < floor:
            raise MemoryError('the derivatives of (Q) need %.1f MB besides T3 and the integrals for %d '
                              'occupied and %d virtual spin orbitals (%.1f MB of them for the '
                              'derivatives themselves); %.1f MB are left'
                              % (floor / 1e6, no, nv, kept / 1e6, budget / 1e6))
        lo, hi = top.floor, top.need            # budgets of (Q): lo fits, hi does not
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if _need(no, nv, isz, Q.plan(no, nv, isz, mid, batch_bytes))[0] <= budget:
                lo = mid
            else:
                hi = mid
        qp = Q.plan(no, nv, isz, lo, batch_bytes)
    need, kept = _need(no, nv, isz, qp)
    return Plan(qp, need, kept, floor)


def explicit_plan(no, nv, isz, cols22, cols13, xv, rows):
    '''The Plan with these batch sizes.'''
    qp = Q.explicit_plan(no, nv, isz, cols22, cols13, xv, rows)
    need, kept = _need(no, nv, isz, qp)
    return Plan(qp, need, kept, _need(no, nv, isz, Q.explicit_plan(
        no, nv, isz, *Q._least(nv)[:2], 3, Q._least(nv)[2]))[0])


class QuadruplesGradient(Q.Quadruples):
    '''(Q) of one occupied quadruple and its derivative.'''

    def __init__(self, eris, t2, t3, plan_=None):
        no = eris.nocc
        nv = np.asarray(eris.fock).shape[0] - no
        isz = np.dtype(np.result_type(t2, t3, eris.oovv)).itemsize
        self.gplan = plan_ or plan(no, nv, isz)
        super().__init__(eris, t2, t3, self.gplan.q)
        dtype, nQ, nA3, nA, nP = self.dtype, self.nQ, self.nA3, self.nA, comb(no, 2)
        self.eris = eris
        z = lambda *shape: np.zeros(shape, dtype=dtype)             # noqa: E731
        # what is returned
        self.S3 = z(comb(no, 3), nA3)
        self.Dvvvv = z(nQ, nQ)
        self.Dvvoo, self.Dvvvo, self.Dovoo = z(nP, nQ), z(no, nv, nQ), z(nP, no, nv)
        self.gamma = z(nv, nv)
        self.de_occ = np.zeros(no)
        self.e_q = 0.0
        # the coefficients of t2 in its layouts, and of Xo
        self.b_pp, self.b_xp, self.b_px = z(nP, nQ), z(no, no, nQ), z(nP, nv, nv)
        self.G2 = z(no, no, nv, nv)
        self.Xobar = z(comb(no, 3), no, nQ)
        # one quadruple
        self.y = np.empty(nA, dtype=dtype)
        self.c = np.empty(nA)
        self.e22 = [np.empty(nQ * self.plan.cols22, dtype=dtype) for _ in range(2)]
        self.e13 = [np.empty(nv * self.plan.cols13, dtype=dtype) for _ in range(2)]
        self.LAb, self.LBb = z(4 * nv, nQ), z(4 * no, nQ)
        self.LCb = z(6 * no, nv)
        self.Lzb, self.Rzb = z(6, nQ), z(6, nQ)
        self.RBb = z(4 * no, nQ)
        self.Xvbar = None

    # ---- Xv and its adjoint

    def xi_v(self, l, xs):
        last, store, where = self._xv
        if store is not None and where:
            nslot = max(3, min(self.plan.xv, self.nocc - 1), len(xs))
            need = {int(x) for x in xs}
            if last != l or len(store) < nslot:
                gone = list(where.items())
            else:
                gone = [(x, n) for x, n in where.items() if x not in need]
            if gone:
                self._back_xv(last, gone)
            if len(store) < nslot:
                self.Xvbar = None
        out = super().xi_v(l, xs)
        if self.Xvbar is None:
            self.Xvbar = np.zeros(out[0].shape, dtype=self.dtype)
        return out

    def _back_xv(self, l, items):
        '''The adjoint of Xv of the pairs (x, l), ``items`` = [(x, its
        place)], through Xv = P(d/bc) 1/2 sum_f <bc||ef> t(x l, fd): s2 and
        the density of <bc||ef>; the adjoints are zero afterwards.'''
        nv, nQ, dtype = self.nvir, self.nQ, self.dtype
        with timer('qg.xv'):
            tl = [np.ascontiguousarray(self.t2[x, l], dtype=dtype) for x, _ in items]       # [f, d]
            acc = [np.zeros((nv, nv), dtype=dtype) for _ in items]
            for b0, b1 in Q._batches(nQ, self.plan.rows):
                nb = b1 - b0
                W = np.empty((nb, nv, nv), dtype=dtype)                         # [(bc), e, f]
                P.tn_expand(W, np.ascontiguousarray(self.vpp[b0:b1], dtype=dtype), nv, 1, 1, outer=0)
                M = np.empty((nb, nv, nv), dtype=dtype)                         # [(bc), e, d]
                Wb = np.empty((nb * nv, nv), dtype=dtype)
                for (x, n), t, a in zip(items, tl, acc):
                    P.tn_expand(M, self.Xvbar[n], nv, 2, 1, rows=(b0, b1), scale=0.5, outer=1)
                    gemm(W.reshape(nb * nv, nv).T, M.reshape(nb * nv, nv), a, beta=1.0)
                    gemm(M.reshape(nb * nv, nv), t.T, Wb)
                    P.tn_reduce(self.Dvvvv[b0:b1], Wb.reshape(nb, nv, nv), nv, 1, 1, outer=0)
                del W, M, Wb
            for (x, n), a in zip(items, acc):
                self.G2[x, l] += a
                P.zero(self.Xvbar[n])

    # ---- one occupied quadruple

    def back(self, I, Xv, where):
        '''The derivative of the terms of the occupied quadruple I, after
        left(I) and right(I, Xv, where): their results and the stacked
        operands are those of this quadruple.  Returns its energy.'''
        no, nv, nQ, nA3 = self.nocc, self.nvir, self.nQ, self.nA3
        so13, so22, so12 = self.so13, self.so22, self.so12
        x13, J13, s13 = so13.idx1[I], so13.idx2[I], so13.sgn[I]
        p22, q22, s22 = so22.idx1[I], so22.idx2[I], so22.sgn[I]
        z, rt3, rtt, y, c = self.z, self.rt3, self.rtt, self.y, self.c
        cplx = np.iscomplexobj(z)
        with timer('qg.vectors'):
            d = self.eo4[I] - self.ev4
            np.add(rt3, rtt, out=y)
            y /= d                                  # y = r / D
            rt3 += z
            rt3 /= d                                # w = z / D
            if cplx:
                np.conj(rt3, out=rt3)               # conj(w)
                np.conj(y, out=z)                   # conj(y)
            else:
                z[:] = y
            np.multiply(rt3, y, out=rtt)
            c[:] = rtt.real                         # Re(conj(w) y)
            self.de_occ[self.quads[I]] -= c.sum()
            e = float(c @ d)
            np.add(z, rt3, out=rtt)                 # conj(y) + conj(w)
        Ly, Lw, Lyw = z, rt3, rtt
        Lz = self.t2_pp[p22] * s22[:, None]                                     # [6, (ab)]
        Rz = self.vvoo_t[q22]                                                   # [6, (cd)]
        for a in (self.LAb, self.LBb, self.LCb, self.Lzb):
            P.zero(a)
        for cc in Q._batches(nQ, self.plan.cols22):
            nc = cc[1] - cc[0]
            Ey, Ew = (b[:nQ * nc].reshape(nQ, nc) for b in self.e22)
            X = self._x22(cc)
            with timer('qg.expand'):
                P.tn_expand(Ey, Ly, nv, 2, 2, cols=cc)
                P.tn_expand(Ew, Lw, nv, 2, 2, cols=cc)
            with timer('qg.gemm'):
                # z: t(ij,ab) <cd||kl>, with conj(y)
                gemm(Rz[:, cc[0]:cc[1]], Ey.T, self.Lzb, beta=1.0)
                gemm(Lz, Ey, self.Rzb[:, cc[0]:cc[1]])
                # t(im,ab) Xo(m;jkl;cd), with conj(w)
                gemm(self.RB[:, cc[0]:cc[1]], Ew.T, self.LBb, beta=1.0)
                gemm(self.LB, Ew, self.RBb[:, cc[0]:cc[1]])
                # <ab||ei> t(jkl,cde), with both
                Ey += Ew
                gemm(self.RA[:, cc[0]:cc[1]], Ey.T, self.LAb, beta=1.0)
                Xa = gemm(self.LA, Ey, X.reshape(-1)[:4 * nv * nc].reshape(4 * nv, nc)) \
                    if 4 * nv <= nQ else gemm(self.LA, Ey)
            with timer('qg.reduce'):
                for k in range(4):
                    P.tn_reduce(self.S3[J13[k]], np.ascontiguousarray(Xa[k * nv:(k + 1) * nv]), nv, 1, 2, cols=cc)
        with timer('qg.scatter'):
            for k in range(6):
                self.b_pp[p22[k]] += int(s22[k]) * self.Lzb[k]
                self.Dvvoo[q22[k]] += self.Rzb[k]
            for k in range(4):
                x, J, s = x13[k], J13[k], int(s13[k])
                self.b_xp[x] += s * self.LBb[k * no:(k + 1) * no]
                self.Xobar[J] += self.RBb[k * no:(k + 1) * no]
                self.Dvvvo[x] -= s * self.LAb[k * nv:(k + 1) * nv]
        rows = so12.pos[:, q22].T.ravel()
        sgn = so12.psg[:, q22].T.ravel()
        last = [(p22[k], where[int(P.tuples(no, 2)[q22[k], 0])], 2.0 * int(s22[k])) for k in (0, 1, 3)]
        for cc in Q._batches(nA3, self.plan.cols13):
            nc = cc[1] - cc[0]
            Ew, Eyw = (b[:nv * nc].reshape(nv, nc) for b in self.e13)
            X = self.w13[:nv * nc].reshape(nv, nc)
            with timer('qg.expand'):
                P.tn_expand(Ew, Lw, nv, 1, 3, cols=cc)
                P.tn_expand(X, y, nv, 1, 3, cols=cc)
                P.tn_expand(Eyw, Lyw, nv, 1, 3, cols=cc)
            with timer('qg.gamma'):
                gemm(Ew, X.T, self.gamma, beta=1.0)
            with timer('qg.gemm'):
                # <ma||ij> t(mkl,bcd), with both
                gemm(self.RC[:, cc[0]:cc[1]], Eyw.T, self.LCb, beta=1.0)
                Xc = gemm(self.LC, Eyw)                                         # [(split, m), (bcd)]
                # 2 t(ij,ae) Xv(e;kl;bcd), with conj(w)
                for p, n, alpha in last:
                    gemm(Xv[n][:, cc[0]:cc[1]], Ew.T, self.b_px[p], alpha=alpha, beta=1.0)
                    gemm(self.t2_px[p], Ew, self.Xvbar[n][:, cc[0]:cc[1]], alpha=alpha, beta=1.0)
            with timer('qg.reduce'):
                _axpy_rows(self.S3, Xc, rows, sgn, cc[0])
        with timer('qg.scatter'):
            for k in range(6):
                self.Dovoo[p22[k]] += int(s22[k]) * self.LCb[k * no:(k + 1) * no]
        self.e_q += e
        return e

    # ---- the end

    def _back_xo(self):
        '''The adjoint of Xo through its definition (Quadruples.__init__):
        s2 and the densities of <mn||jk> and <md||el>.'''
        no, nv, nQ, dtype = self.nocc, self.nvir, self.nQ, self.dtype
        i2, j2 = Q._pairs(no)
        a2, b2 = Q._pairs(nv)
        nP = len(i2)
        eris, so12 = self.eris, self.so12
        oooo_t = np.ascontiguousarray(np.asarray(eris.oooo)[:, :, i2, j2].transpose(0, 2, 1), dtype=dtype)
        ovvo = np.asarray(eris.ovvo)
        t2_pc = np.ascontiguousarray(self.t2[i2, j2], dtype=dtype).reshape(nP * nv, nv)     # [(jk) c, e]
        b_pc = np.zeros((nP * nv, nv), dtype=dtype)
        Doooo = np.zeros((no, no, nP), dtype=dtype)
        Dovvo = np.zeros((no, nv, nv, no), dtype=dtype)
        Kb = np.zeros((nP, nv, nv), dtype=dtype)
        for m in range(no):
            # the adjoint of the fold over (l | jk): [l, (jk), (cd)]
            Gb = self.Xobar[so12.pos, m] * so12.psg[:, :, None]
            for l in range(no):
                g = np.ascontiguousarray(Gb[l])
                # 1/2 sum_n <mn||jk> t(nl,cd)
                Doooo[m] += 0.5 * gemm(self.t2_xp[:, l], g.T)
                gemm(oooo_t[m].T, g, self.b_xp[:, l], alpha=0.5, beta=1.0)
                # sum_e t(jk,ce) <md||el>, antisymmetric in c, d
                Kb[:, a2, b2] = g
                Kb[:, b2, a2] = -g
                K2 = Kb.reshape(nP * nv, nv)
                o_ml = np.ascontiguousarray(ovvo[m, :, :, l], dtype=dtype)      # [d, e]
                gemm(K2, o_ml, b_pc, beta=1.0)
                Dovvo[m, :, :, l] += gemm(K2.T, t2_pc)
        self.G2[i2, j2] += b_pc.reshape(nP, nv, nv)
        return Doooo, Dovvo

    def finish(self):
        '''The Gradient, after all quadruples.'''
        no, nv = self.nocc, self.nvir
        last, store, where = self._xv
        if store is not None and where:
            self._back_xv(last, list(where.items()))
        self.Xvbar = None
        self._xv = (None, None, {})
        with timer('qg.xo'):
            Doooo, Dovvo = self._back_xo()
        i2, j2 = Q._pairs(no)
        a2, b2 = Q._pairs(nv)
        G2 = self.G2
        G2[i2[:, None], j2[:, None], a2[None, :], b2[None, :]] += self.b_pp
        G2[:, :, a2, b2] += self.b_xp
        G2[i2, j2] += self.b_px.transpose(0, 2, 1)
        s2 = G2 - G2.transpose(1, 0, 2, 3)
        s2 = s2 - s2.transpose(0, 1, 3, 2)
        return Gradient(self.e_q, s2, self.S3, self.de_occ, self.gamma, self.Dvvoo, self.Dvvvo,
                        self.Dovoo, Doooo, Dovvo, self.Dvvvv, self.gplan)


def _axpy_rows(R, X, rows, sgn, col0):
    '''R[rows[r], col0 + c] += sgn[r] X[r, c].'''
    if P.USE_C and X.flags.c_contiguous:
        P.tn_axpy_rows(R, X, rows, np.arange(len(rows)) * X.shape[1], sgn, col0)
        return
    for r in range(len(rows)):
        if sgn[r]:
            R[rows[r], col0:col0 + X.shape[1]] += int(sgn[r]) * X[r]


def kernel(eris, t2, t3, max_memory=None, plan_=None):
    '''The Gradient of (Q) for converged CCSDT amplitudes: t2[i,j,a,b] and T3
    packed [C(no,3), C(nv,3)] (or flat) on an integral container
    (zccsdt.eris.ERIs).  ``max_memory``: MB for the whole process (default:
    what the node and the job leave); ``plan_``: a Plan given explicitly.
    At least four occupied and four virtual spin orbitals.'''
    from zccsdt import memory
    no = eris.nocc
    nv = np.asarray(eris.fock).shape[0] - no
    if no < 4 or nv < 4:
        raise ValueError('(Q) needs four occupied and four virtual spin orbitals')
    if plan_ is None:
        isz = np.dtype(np.result_type(t2, t3, eris.oovv)).itemsize
        plan_ = plan(no, nv, isz, memory.budget(max_memory))
    q = QuadruplesGradient(eris, t2, t3, plan_)
    for l, xs, Is in q.schedule():
        Xv, where = q.xi_v(l, xs)
        for I in Is:
            with timer('q.left'):
                q.left(I)
            q.right(I, Xv, where)
            q.back(I, Xv, where)
        del Xv
    return q.finish()


def dense_d2(g, nocc):
    '''d2[p,q,r,s] over all spin orbitals of a Gradient (zero outside of the
    elements that the Gradient names): dE = Re sum d2 d<pq||rs>, as
    qgrad_ref.quadruples_gradient has it up to the antisymmetry of the
    integrals.  For small cases.'''
    o = nocc
    v = g.gamma_vv.shape[0]
    n = o + v
    i2, j2 = Q._pairs(o)
    a2, b2 = Q._pairs(v)
    d2 = np.zeros((n,) * 4, dtype=g.vvvv.dtype)
    A, B, I, J = o + a2, o + b2, i2, j2
    d2[A[None, :], B[None, :], I[:, None], J[:, None]] += g.vvoo
    d2[A[None, None, :], B[None, None, :], o + np.arange(v)[None, :, None], np.arange(o)[:, None, None]] += g.vvvo
    d2[np.arange(o)[None, :, None], o + np.arange(v)[None, None, :], I[:, None, None], J[:, None, None]] += g.ovoo
    d2[np.arange(o)[:, None, None], np.arange(o)[None, :, None], I[None, None, :], J[None, None, :]] += g.oooo
    d2[:o, o:, o:, :o] += g.ovvo
    d2[A[:, None], B[:, None], A[None, :], B[None, :]] += g.vvvv
    return d2


# ------------------------------------------------------------ orbital gradient

PATTERNS = ('vvoo', 'vvvo', 'ovoo', 'oooo', 'ovvo')


def dense_blocks(g, nocc):
    '''{block: D[p,q,r,s]} of a Gradient for the blocks of PATTERNS, dense
    over the four indices (zero outside of the unique elements): the
    coefficients of <pq||rs> in dE.'''
    o, v = nocc, g.gamma_vv.shape[0]
    i2, j2 = Q._pairs(o)
    a2, b2 = Q._pairs(v)
    dtype = g.vvvv.dtype
    out = {'vvoo': np.zeros((v, v, o, o), dtype=dtype), 'vvvo': np.zeros((v, v, v, o), dtype=dtype),
           'ovoo': np.zeros((o, v, o, o), dtype=dtype), 'oooo': np.zeros((o, o, o, o), dtype=dtype),
           'ovvo': np.asarray(g.ovvo)}
    out['vvoo'][a2[None, :], b2[None, :], i2[:, None], j2[:, None]] = g.vvoo
    out['vvvo'][a2, b2] = g.vvvo.transpose(2, 1, 0)
    out['ovoo'][:, :, i2, j2] = g.ovoo.transpose(1, 2, 0)
    out['oooo'][:, :, i2, j2] = g.oooo
    return out


def gblock(eris, pattern):
    '''The block <pq||rs> of the integrals ``eris`` for a pattern of 'o' and
    'v' with an occupied index, dense, from the blocks that are kept.'''
    def get(k):
        return np.asarray(getattr(eris, k))
    if pattern in ('oooo', 'ooov', 'oovv', 'ovov', 'ovvo', 'ovvv'):
        return get(pattern)
    if pattern == 'oovo':
        return -get('ooov').transpose(0, 1, 3, 2)
    if pattern == 'ovoo':                                   # <ia||jk> = <jk||ia>*
        return get('ooov').conj().transpose(2, 3, 0, 1)
    if pattern == 'vooo':
        return -get('ooov').conj().transpose(3, 2, 0, 1)
    if pattern == 'vvoo':
        return get('oovv').conj().transpose(2, 3, 0, 1)
    if pattern == 'voov':
        return get('ovvo').transpose(1, 0, 3, 2)
    if pattern == 'vovo':
        return get('ovov').transpose(1, 0, 3, 2)
    if pattern == 'vovv':
        return -get('ovvv').transpose(1, 0, 2, 3)
    if pattern == 'vvov':                                   # <ab||ic> = <ic||ab>*
        return get('ovvv').conj().transpose(2, 3, 0, 1)
    if pattern == 'vvvo':
        return -get('ovvv').conj().transpose(2, 3, 1, 0)
    raise KeyError(pattern)


def _contract(I, D, p):
    '''sum over the three indices other than p of I[..] D[..]: [index p of I,
    index p of D].'''
    dtype = np.result_type(I, D)
    A = np.ascontiguousarray(np.moveaxis(I, p, 0), dtype=dtype).reshape(I.shape[p], -1)
    B = np.ascontiguousarray(np.moveaxis(D, p, 0), dtype=dtype).reshape(D.shape[p], -1)
    return gemm(A, B.T)


def _vvvv_terms(Dpp, S, nv):
    '''The density of <bc||ef>, Dpp[(b<c), (e<f)], with integrals of one
    other index k and three virtual ones, S[k, c, (e<f)] = <kc||ef>:

        bra[k, x] = sum D[(bc),(ef)] (d_xb <kc||ef> + d_xc <bk||ef>)
        ket[k, y] = sum D[(bc),(ef)] (d_ye <bc||kf> + d_yf <bc||ek>),

    <bc||kf> = <kf||bc>*.'''
    a2, b2 = Q._pairs(nv)
    nQ = len(a2)
    nk = S.shape[0]
    dtype = np.result_type(Dpp, S)
    S2 = np.ascontiguousarray(S, dtype=dtype).reshape(nk * nv, nQ)
    Dpp = np.ascontiguousarray(Dpp, dtype=dtype)
    q = np.arange(nQ)
    Z = gemm(Dpp, S2.T).reshape(nQ, nk, nv)                         # [(bc), k, c']
    bra = np.zeros((nv, nk), dtype=dtype)
    np.add.at(bra, a2, Z[q, :, b2])
    np.subtract.at(bra, b2, Z[q, :, a2])
    del Z
    Z = gemm(np.conj(S2), Dpp).reshape(nk, nv, nQ)                  # [k, f', (ef)]
    ket = np.zeros((nv, nk), dtype=dtype)
    np.add.at(ket, a2, Z[:, b2, q].T)
    np.subtract.at(ket, b2, Z[:, a2, q].T)
    return bra.T, ket.T


def active_terms(eris, g, dens=None):
    '''Gnh[y,x] of E(Q) among the spin orbitals of ``eris`` without its
    block among the virtual ones: the first-order change of E(Q) with the
    integrals one-index transformed - a bra index y of an integral taking
    the values of x, a ket index x minus those of y -, the orbital energies
    as the diagonal of the Fock matrix of the rotated spin orbitals.  From
    the dense blocks of the densities (dense_blocks), every one with the
    blocks of integrals that one of its indices changed gives; the density
    of <ab||ei> meets the packed <vv||vv> in slabs of rows, that of <bc||ef>
    the integrals <ie||bc> on pairs.'''
    from zccsdt.cc import t3pack
    from zccsdt.cc.ccsdt_opt import vvvv_packed
    o = eris.nocc
    v = g.gamma_vv.shape[0]
    n = o + v
    dens = dense_blocks(g, o) if dens is None else dens
    dtype = np.result_type(g.vvvv.dtype, np.asarray(eris.oovv).dtype)
    G = np.zeros((n, n), dtype=dtype)
    sl = {'o': slice(0, o), 'v': slice(o, n)}
    for pat, D in dens.items():
        for p in range(4):
            kd = pat[p]
            for kn in 'ov':
                ky, kx = (kn, kd) if p < 2 else (kd, kn)
                ipat = pat[:p] + kn + pat[p + 1:]
                if ky == kx == 'v' or ipat == 'vvvv':
                    continue
                val = _contract(gblock(eris, ipat), D, p)           # [new index, index of D]
                if p < 2:
                    G[sl[ky], sl[kx]] += val
                else:
                    G[sl[ky], sl[kx]] -= val.T
    a2, b2 = Q._pairs(v)
    nQ = len(a2)
    # the density of <cd||el> with <cd||ea>: Gnh[l, a] -= sum D[(cd), e, l] <cd||ea>
    vpp = vvvv_packed(eris, t3pack.triple_map(v))[0]
    rows = max(1, min(nQ, (64 << 20) // max(1, 16 * v * v)))
    T = np.zeros((o, v), dtype=dtype)
    for b0, b1 in Q._batches(nQ, rows):
        nb = b1 - b0
        W = np.empty((nb, v, v), dtype=dtype)                                   # [(cd), e, a]
        P.tn_expand(W, np.ascontiguousarray(vpp[b0:b1], dtype=dtype), v, 1, 1, outer=0)
        A = np.ascontiguousarray(np.asarray(g.vvvo[:, :, b0:b1], dtype=dtype).transpose(0, 2, 1)).reshape(o, nb * v)
        gemm(A, W.reshape(nb * v, v), T, beta=1.0)
        del W, A
    G[:o, o:] -= T
    # the density of <bc||ef> with <ic||ef>
    bra, ket = _vvvv_terms(g.vvvv, np.asarray(eris.ovvv)[:, :, a2, b2], v)
    G[:o, o:] += bra
    G[o:, :o] -= ket.T
    # the orbital energies: e_p = f[p,p], and the Fock operator has the occupied spin orbitals
    de = g.de
    V = (np.einsum('j,jaji->ai', de[:o], gblock(eris, 'ovoo'))
         + np.einsum('b,babi->ai', de[o:], gblock(eris, 'vvvo')))
    G[o:, :o] += V
    G[:o, o:] -= V.conj().T
    return G


def source_terms(g, t2, T3):
    '''(Gnh[j,i], Gnh[b,a]) of the CCSDT Lagrangian among the occupied and
    among the virtual spin orbitals when lambda solves the equations with
    the sources s of a Gradient, dL/dt = -s: a rotation among them is a
    change of t (the Lagrangian does not change when t, lambda and the
    integrals are rotated together),

        Gnh[j,i] = -1/2 sum s2[j,k,a,b] t2[i,k,a,b] - 1/12 sum s3[j,k,l,a,b,c] t3[i,k,l,a,b,c]
        Gnh[b,a] =  1/2 sum s2[i,j,a,c] t2[i,j,b,c] + 1/12 sum s3[i,j,k,a,c,d] t3[i,j,k,b,c,d],

    in the T1-dressed basis and in the bare one alike (s1 = 0).'''
    o, v = t2.shape[0], t2.shape[2]
    s2 = g.s2
    dtype = np.result_type(s2, t2)
    S3 = np.ascontiguousarray(np.asarray(g.s3).reshape(comb(o, 3), -1), dtype=dtype)
    T3 = np.ascontiguousarray(np.asarray(T3).reshape(comb(o, 3), -1), dtype=dtype)
    nQ = comb(v, 2)
    oo = -0.5 * np.einsum('jkab,ikab->ji', s2, t2, optimize=True).astype(dtype)
    vv = 0.5 * np.einsum('ijac,ijbc->ba', s2, t2, optimize=True).astype(dtype)
    if S3.size:
        so = P.split(o, 1, 2)
        M = gemm(S3, T3.T)                                                      # [(jkl), (ikl)]
        oo -= np.einsum('jp,ip,jip->ji', so.psg.astype(float), so.psg.astype(float),
                        M[so.pos[:, None, :], so.pos[None, :, :]])
        del M
        Sx, Tx = (np.empty((v, nQ), dtype=dtype) for _ in range(2))
        acc = np.zeros((v, v), dtype=dtype)
        for I in range(S3.shape[0]):
            P.tn_expand(Sx, S3[I], v, 1, 2)
            P.tn_expand(Tx, T3[I], v, 1, 2)
            gemm(Tx, Sx.T, acc, beta=1.0)                                       # [b, a]
        vv += acc
    return oo, vv


def orbital_gradient(eris, g, t1, t2, T3, gnh, dens=None):
    '''G[y,x] of zccsd.zvector among the spin orbitals of ``eris`` for
    CCSDT(Q), anti-Hermitian: ``gnh`` [a,i] is the table GRADIENT of the
    CCSDT Lagrangian (density_opt.kernel) for lambda with the sources of
    the Gradient ``g``.

    - virtual-occupied: with the sources the table is not the gradient of
      the bare integrals - the rotations among occupied and among virtual
      spin orbitals, contracted with t1, no longer vanish:
          Gnh[a,i] = gnh[a,i] + sum_j t1[j,a] Gnh[j,i] - sum_b t1[i,b] Gnh[a,b]
      with the two blocks of source_terms -, and E(Q) has both halves
      (active_terms), Gnh[i,a] too;
    - among the occupied: source_terms and active_terms;
    - among the virtual: (e_q - e_p) times the Hermitian part of Gamma,
      transposed (qgrad_ref.fock_density) - that is the whole of it.'''
    o = eris.nocc
    oo, vv = source_terms(g, t2, T3)
    Gnh = active_terms(eris, g, dens)
    Gnh = Gnh.astype(np.result_type(Gnh, gnh, oo, t1))
    Gnh[:o, :o] += oo
    Gnh[o:, :o] += gnh + t1.T @ oo - vv @ t1.T
    G = 0.5 * (Gnh - Gnh.conj().T)
    ev = np.asarray(eris.mo_energy)[o:]
    gam = 0.5 * (g.gamma_vv + g.gamma_vv.conj().T)
    G[o:, o:] = (ev[None, :] - ev[:, None]) * gam.T
    return G


class FrozenTerms:
    '''The (Q) terms of bf and kf of density_opt.kernel(frozen=): for a
    batch of frozen spinors f with their integrals G[f,v,u,w], <fu||vw> =
    G[f,v,u,w] - G[f,w,u,v] (u, v, w the spin orbitals of the amplitudes),

        b[f,x] = sum over the bra indices x of the densities, f in its place in the integral
        k[f,x] = sum over the ket indices x of the densities, f in its place

    so that Gnh[f,x] = b[f,x], Gnh[x,f] = -k[f,x] for E(Q) and the gradient of
    zccsd.zvector is (b + conj(k)) / 2, as for the two-particle density of
    CCSDT.  One frozen spinor at a time (arrays of n^3 elements).'''

    def __init__(self, g, nocc, real=False, dens=None):
        self.g, self.o, self.real = g, nocc, real
        self.v = g.gamma_vv.shape[0]
        self.dens = dense_blocks(g, nocc) if dens is None else dens

    def work_bytes(self):
        n = self.o + self.v
        return 16 * 4 * n ** 3 + 16 * 2 * comb(self.v, 2) * self.v

    def __call__(self, G):
        o, v = self.o, self.v
        n = o + v
        a2, b2 = Q._pairs(v)
        sl = {'o': slice(0, o), 'v': slice(o, n)}
        nf = G.shape[0]
        dtype = self.g.vvvv.dtype if self.real else np.result_type(self.g.vvvv.dtype, np.complex128)
        b = np.zeros((nf, n), dtype=dtype)
        k = np.zeros((nf, n), dtype=dtype)
        for f in range(nf):
            Gf = G[f].real if self.real else G[f]
            S = Gf.transpose(1, 0, 2) - Gf.transpose(1, 2, 0)                   # [u, v, w] = <fu||vw>
            Sc = S.conj()
            for pat, D in self.dens.items():
                s = [sl[c] for c in pat]
                # bra: <f q||r s> and <q f||r s> = -<f q||r s>
                b[f, s[0]] += np.tensordot(D, S[s[1], s[2], s[3]], axes=3)
                b[f, s[1]] -= np.einsum('qxrs,qrs->x', D, S[s[0], s[2], s[3]], optimize=True)
                # ket: <p q||f s> = <f s||p q>*, <p q||r f> = -<f r||p q>*
                k[f, s[2]] += np.einsum('pqxs,spq->x', D, Sc[s[3], s[0], s[1]], optimize=True)
                k[f, s[3]] -= np.einsum('pqrx,rpq->x', D, Sc[s[2], s[0], s[1]], optimize=True)
            bra, ket = _vvvv_terms(self.g.vvvv, np.ascontiguousarray(S[o:][:, o + a2, o + b2])[None], v)
            b[f, o:] += bra[0]
            k[f, o:] += ket[0]
        return b, k
