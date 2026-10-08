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
from pyscf import lib

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
    vvvv: object                # [(b<c), (e<f)] of <bc||ef>, or its PairBlocks
    plan: object = None
    src3: tuple = None          # t3_terms(s3, T3): the parts of source_terms with t3
    scratch: object = None      # the file of the arrays that are mapped (Plan.mapped)

    @property
    def de(self):
        '''dE / d e_p for all spin orbitals, occupied first.'''
        return np.concatenate((self.de_occ, self.gamma_vv.diagonal().real))


@dataclass
class Plan:
    q: Q.Plan                   # the batches of the products, as for (Q)
    need: int                   # bytes of all arrays in memory (without T3 and the integrals)
    kept: int                   # of which are the derivatives that are returned
    floor: int                  # the least that a calculation takes
    mapped: tuple = ()          # of 'vvvv', 's3': the derivatives that go to a scratch file


MAPPED = (('vvvv', 's3'), ('vvvv',), ())        # from short memory to all of it


def _own(no, nv, isz, mapped=()):
    '''Bytes (kept, work) of what the derivative has in memory besides the
    arrays of (Q) and what scales with its batches: the derivatives that are
    returned (without those of ``mapped``) and the accumulators of s2; the
    vectors of a quadruple and the adjoints of the stacked operands.'''
    nP, nQ, nI3, nA3, nA = Q._sizes(no, nv)
    kept = (no * no * nv * nv + no + nv * nv + nP * nQ + no * nv * nQ
            + nP * no * nv + no * no * nP + no * nv * nv * no)
    kept += (0 if 's3' in mapped else nI3 * nA3) + (0 if 'vvvv' in mapped else nQ * nQ)
    acc = nP * nQ + no * no * nQ + nP * nv * nv + no * no * nv * nv + nI3 * no * nQ
    work = nA + (4 * nv + 8 * no) * nQ + 6 * no * nv + 24 * nQ
    return isz * kept, isz * (acc + work) + 8 * nA


def _need(no, nv, isz, qp, mapped=()):
    '''Bytes of everything with the batches of the Plan of (Q) ``qp``.'''
    nP, nQ, nI3, nA3, nA = Q._sizes(no, nv)
    base, work, xo = Q._fixed(no, nv, isz)
    kept, own = _own(no, nv, isz, mapped)
    nslot = max(3, min(qp.xv, no - 1))
    var = isz * (3 * nQ * qp.cols22 + 3 * nv * qp.cols13 + (4 * nv * qp.cols22 if 4 * nv > nQ else 0)
                 + 6 * no * qp.cols13 + 2 * nv * nslot * nA3 + nv * nv * (2 * nslot + 3 * qp.rows))
    # the adjoint of Xo at the end: its arrays for one m, with the accumulators still there;
    # the parts of the orbital gradient with s3 and t3
    back = isz * max(2 * no * nP * nQ + 3 * nP * nv * nv + no * nP * no, 2 * nI3 * nI3 + 2 * nv * nQ)
    return kept + base + max(xo, work + own + max(var, back)), kept


def _least_plan(no, nv, isz):
    return Q.explicit_plan(no, nv, isz, *Q._least(nv)[:2], 3, Q._least(nv)[2])


def plan(no, nv, isz, budget=None, batch_bytes=None):
    '''The Plan for ``budget`` bytes (None: no limit) besides T3 and the
    integrals.  The batches are those of a Plan of (Q): every work array of
    a batch is there three times (the product and two expanded left-hand
    quadruples) and Xv twice, so it is the Plan of (Q) for the largest
    budget of its own with which all of this fits (the plans of (Q) grow
    with their budget, and so does what is needed here: found by
    bisection).  Where the largest batches do not fit, s3 and the density of
    <bc||ef> - arrays of the sizes of T3 and of the packed <vv||vv> - go to
    a scratch file (Plan.mapped), and the batches take the memory; with the
    largest batches the density alone goes there if s3 fits, and nothing if
    both do.  MemoryError below the floor: the smallest batches of (Q) with
    both arrays mapped.'''
    top = Q.plan(no, nv, isz, None, batch_bytes)
    least = _least_plan(no, nv, isz)
    floor = _need(no, nv, isz, least, MAPPED[0])[0]
    qp, mapped = top, ()
    if budget is not None and _need(no, nv, isz, top)[0] > budget:
        if budget < floor:
            raise MemoryError('the derivatives of (Q) need %.1f MB besides T3 and the integrals for %d '
                              'occupied and %d virtual spin orbitals (with %.1f MB more in a scratch file '
                              'for those of the sizes of T3 and <vv||vv>); %.1f MB are left'
                              % (floor / 1e6, no, nv, (_own(no, nv, isz)[0] - _own(no, nv, isz, MAPPED[0])[0]) / 1e6,
                                 budget / 1e6))
        mapped = next(m for m in MAPPED[::-1] if m == MAPPED[0] or _need(no, nv, isz, top, m)[0] <= budget)
        if _need(no, nv, isz, top, mapped)[0] > budget:
            lo, hi = top.floor, top.need            # budgets of (Q): lo fits, hi does not
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if _need(no, nv, isz, Q.plan(no, nv, isz, mid, batch_bytes), mapped)[0] <= budget:
                    lo = mid
                else:
                    hi = mid
            qp = Q.plan(no, nv, isz, lo, batch_bytes)
    need, kept = _need(no, nv, isz, qp, mapped)
    return Plan(qp, need, kept, floor, mapped)


def explicit_plan(no, nv, isz, cols22, cols13, xv, rows, mapped=()):
    '''The Plan with these batch sizes.'''
    qp = Q.explicit_plan(no, nv, isz, cols22, cols13, xv, rows)
    need, kept = _need(no, nv, isz, qp, mapped)
    return Plan(qp, need, kept, _need(no, nv, isz, _least_plan(no, nv, isz), MAPPED[0])[0], mapped)


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
        # what is returned: in memory, or in a scratch file (new space reads as zeros)
        self.scratch = None
        if self.gplan.mapped:
            from zccsdt.scratch import Scratch
            self.scratch = Scratch(prefix='zccsdt_qgrad_', what='the derivatives of (Q)',
                                   hint=', or more memory (max_memory)')
        mapped = lambda key, *shape: (self.scratch.create(key, shape, dtype) if key in self.gplan.mapped  # noqa: E731
                                      else z(*shape))
        self.S3 = mapped('s3', comb(no, 3), nA3)
        self.Dvvvv = mapped('vvvv', nQ, nQ)
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
            eo, ev4 = self.eo4[I], self.ev4

            def part(lo, hi):
                # (slices on threads: numpy does these one core at a time)
                sl = slice(lo, hi)
                d = eo - ev4[sl]
                zz, r3, rr, yy, cc_ = z[sl], rt3[sl], rtt[sl], y[sl], c[sl]
                np.add(r3, rr, out=yy)
                yy /= d                             # y = r / D
                r3 += zz
                r3 /= d                             # w = z / D
                if cplx:
                    np.conj(r3, out=r3)             # conj(w)
                    np.conj(yy, out=zz)             # conj(y)
                else:
                    zz[:] = yy
                np.multiply(r3, yy, out=rr)
                cc_[:] = rr.real                    # Re(conj(w) y)
                np.add(zz, r3, out=rr)              # conj(y) + conj(w)
                return float(cc_.sum()), float(cc_ @ d)
            step = max(1 << 16, -(-len(z) // max(1, lib.num_threads())))
            sums = _threaded(lambda lh: part(*lh), Q._batches(len(z), step))
            self.de_occ[self.quads[I]] -= sum(x[0] for x in sums)
            e = sum(x[1] for x in sums)
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

    def finish(self):
        '''The Gradient, after all quadruples.'''
        no, nv = self.nocc, self.nvir
        last, store, where = self._xv
        if store is not None and where:
            self._back_xv(last, list(where.items()))
        self.Xvbar = None
        self._xv = (None, None, {})
        with timer('qg.xo'):
            Doooo, Dovvo = back_xo(self, self.eris, self.Xobar, self.b_xp, self.G2)
        s2 = sources2(no, nv, self.G2, self.b_pp, self.b_xp, self.b_px)
        self.b_pp = self.b_xp = self.b_px = self.G2 = self.Xobar = None
        return Gradient(self.e_q, s2, self.S3, self.de_occ, self.gamma, self.Dvvoo, self.Dvvvo,
                        self.Dovoo, Doooo, Dovvo, self.Dvvvv, self.gplan,
                        t3_terms(self.S3, self.T3, no, nv), self.scratch)


def back_xo(q, eris, Xobar, b_xp, G2):
    '''The adjoint of Xo, ``Xobar`` [(jkl), m, (cd)], through its definition
    (Quadruples.__init__) for the Quadruples ``q``: the coefficients of t2
    are added to ``b_xp`` and ``G2``; returns the densities of <mn||jk> and
    <md||el>.'''
    no, nv, nQ, dtype = q.nocc, q.nvir, q.nQ, q.dtype
    i2, j2 = Q._pairs(no)
    a2, b2 = Q._pairs(nv)
    nP = len(i2)
    so12 = q.so12
    oooo_t = np.ascontiguousarray(np.asarray(eris.oooo)[:, :, i2, j2].transpose(0, 2, 1), dtype=dtype)
    ovvo = np.asarray(eris.ovvo)
    t2_pc = np.ascontiguousarray(q.t2[i2, j2], dtype=dtype).reshape(nP * nv, nv)        # [(jk) c, e]
    b_pc = np.zeros((nP * nv, nv), dtype=dtype)
    Doooo = np.zeros((no, no, nP), dtype=dtype)
    Dovvo = np.zeros((no, nv, nv, no), dtype=dtype)
    Kb = np.zeros((nP, nv, nv), dtype=dtype)
    for m in range(no):
        # the adjoint of the fold over (l | jk): [l, (jk), (cd)]
        Gb = Xobar[so12.pos, m] * so12.psg[:, :, None]
        for l in range(no):
            g = np.ascontiguousarray(Gb[l])
            # 1/2 sum_n <mn||jk> t(nl,cd)
            Doooo[m] += 0.5 * gemm(q.t2_xp[:, l], g.T)
            gemm(oooo_t[m].T, g, b_xp[:, l], alpha=0.5, beta=1.0)
            # sum_e t(jk,ce) <md||el>, antisymmetric in c, d
            Kb[:, a2, b2] = g
            Kb[:, b2, a2] = -g
            K2 = Kb.reshape(nP * nv, nv)
            o_ml = np.ascontiguousarray(ovvo[m, :, :, l], dtype=dtype)      # [d, e]
            gemm(K2, o_ml, b_pc, beta=1.0)
            Dovvo[m, :, :, l] += gemm(K2.T, t2_pc)
    G2[i2, j2] += b_pc.reshape(nP, nv, nv)
    return Doooo, Dovvo


def sources2(no, nv, G2, b_pp, b_xp, b_px):
    '''s2[i,j,a,b] from the coefficients of t2 in its layouts: G2 [i,j,a,b]
    (changed), b_pp [(ij), (ab)], b_xp [i, m, (ab)], b_px [(ij), e, a].'''
    i2, j2 = Q._pairs(no)
    a2, b2 = Q._pairs(nv)
    G2[i2[:, None], j2[:, None], a2[None, :], b2[None, :]] += b_pp
    G2[:, :, a2, b2] += b_xp
    G2[i2, j2] += b_px.transpose(0, 2, 1)
    s2 = G2 - G2.transpose(1, 0, 2, 3)
    return s2 - s2.transpose(0, 1, 3, 2)


class PairBlocks:
    '''The density of <bc||ef> where it has a symmetry label: the diagonal
    blocks of the labels of the pairs, ``blocks`` = [(the pairs of a label
    in the packed order, D[these pairs, these pairs])]; zero elsewhere.'''

    def __init__(self, npair, dtype, blocks):
        self.npair, self.dtype, self.blocks = npair, np.dtype(dtype), blocks

    @property
    def nbytes(self):
        return sum(b.nbytes for _, b in self.blocks)

    def dense(self):
        out = np.zeros((self.npair, self.npair), dtype=self.dtype)
        for idx, b in self.blocks:
            out[np.ix_(idx, idx)] = b
        return out


def pair_blocks(D):
    '''[(pairs or None for all, block)] of a density of <bc||ef>: an array
    [(b<c), (e<f)] or a PairBlocks.'''
    return D.blocks if isinstance(D, PairBlocks) else [(None, D)]


def dense_pairs(D):
    return D.dense() if isinstance(D, PairBlocks) else np.asarray(D)


def _threaded(function, items):
    '''[function(x) for x in items] on threads.'''
    from concurrent.futures import ThreadPoolExecutor
    items = list(items)
    if len(items) < 2:
        return [function(x) for x in items]
    with ThreadPoolExecutor(min(len(items), max(1, lib.num_threads()))) as pool:
        return list(pool.map(function, items))


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
    d2[A[:, None], B[:, None], A[None, :], B[None, :]] += dense_pairs(g.vvvv)
    return d2


# ------------------------------------------------------------ orbital gradient

PATTERNS = ('vvoo', 'vvvo', 'ovoo', 'oooo', 'ovvo')
SMALL = ('vvoo', 'ovoo', 'oooo', 'ovvo')        # (at most two virtual indices)
SLAB_BYTES = 128 << 20                          # rows of an array of the size of <vv||vv> at a time


def dense_blocks(g, nocc, patterns=PATTERNS):
    '''{block: D[p,q,r,s]} of a Gradient for the blocks of ``patterns``,
    dense over the four indices (zero outside of the unique elements): the
    coefficients of <pq||rs> in dE.'''
    o, v = nocc, g.gamma_vv.shape[0]
    i2, j2 = Q._pairs(o)
    a2, b2 = Q._pairs(v)
    dtype = g.vvvv.dtype
    out = {}
    if 'vvoo' in patterns:
        out['vvoo'] = np.zeros((v, v, o, o), dtype=dtype)
        out['vvoo'][a2[None, :], b2[None, :], i2[:, None], j2[:, None]] = g.vvoo
    if 'vvvo' in patterns:
        out['vvvo'] = np.zeros((v, v, v, o), dtype=dtype)
        out['vvvo'][a2, b2] = g.vvvo.transpose(2, 1, 0)
    if 'ovoo' in patterns:
        out['ovoo'] = np.zeros((o, v, o, o), dtype=dtype)
        out['ovoo'][:, :, i2, j2] = g.ovoo.transpose(1, 2, 0)
    if 'oooo' in patterns:
        out['oooo'] = np.zeros((o, o, o, o), dtype=dtype)
        out['oooo'][:, :, i2, j2] = g.oooo
    if 'ovvo' in patterns:
        out['ovvo'] = np.asarray(g.ovvo)
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


def _unpack_pair(Dp, nv, out=None):
    '''U[x, y, e] of Dp[e, (a<b)]: Dp[e, (xy)] for x < y, minus Dp[e, (yx)]
    for x > y.'''
    a2, b2 = Q._pairs(nv)
    U = np.zeros((nv, nv, Dp.shape[0]), dtype=Dp.dtype) if out is None else out
    U[a2, b2] = Dp.T
    U[b2, a2] = -Dp.T
    return U


def _vvvv_terms(Dpp, S, nv, slab_bytes=None):
    '''The density of <bc||ef>, Dpp[(b<c), (e<f)] - an array, in memory or
    mapped, or a PairBlocks -, with integrals of one other index k and
    three virtual ones, S[k, c, (e<f)] = <kc||ef>:

        bra[k, x] = sum D[(bc),(ef)] (d_xb <kc||ef> + d_xc <bk||ef>)
        ket[k, y] = sum D[(bc),(ef)] (d_ye <bc||kf> + d_yf <bc||ek>),

    <bc||kf> = <kf||bc>*.  The density is read in slabs of rows of
    ``slab_bytes`` (default SLAB_BYTES); besides a slab: two arrays of the
    size of S for a block of the density, and two of a slab of S.'''
    a2, b2 = Q._pairs(nv)
    nk = S.shape[0]
    dtype = np.result_type(Dpp.dtype, S.dtype)
    S2 = np.ascontiguousarray(S, dtype=dtype).reshape(nk * nv, -1)
    bra = np.zeros((nv, nk), dtype=dtype)
    ket = np.zeros((nv, nk), dtype=dtype)
    cap = slab_bytes or SLAB_BYTES
    for idx, Dq in pair_blocks(Dpp):
        nq = Dq.shape[0]
        if not nq or not nk:
            continue
        Sq = S2 if idx is None else np.ascontiguousarray(S2[:, idx])             # [(k c), (ef) of the block]
        aq, bq = (a2, b2) if idx is None else (a2[idx], b2[idx])
        Z2 = np.zeros((nk * nv, nq), dtype=dtype)                                # [(k f'), (ef)]
        step = int(max(1, min(nq, cap // max(1, dtype.itemsize * max(nq, nk * nv)))))
        for r0, r1 in Q._batches(nq, step):
            Dr = np.ascontiguousarray(Dq[r0:r1], dtype=dtype)                   # (a slab of a mapped array: read)
            Z = gemm(Dr, Sq.T).reshape(r1 - r0, nk, nv)                         # [(bc), k, c']
            q = np.arange(r1 - r0)
            np.add.at(bra, aq[r0:r1], Z[q, :, bq[r0:r1]])
            np.subtract.at(bra, bq[r0:r1], Z[q, :, aq[r0:r1]])
            del Z
            gemm(np.conj(Sq[:, r0:r1]), Dr, Z2, beta=1.0)
            del Dr
        Z2 = Z2.reshape(nk, nv, nq)
        q = np.arange(nq)
        np.add.at(ket, aq, Z2[:, bq, q].T)
        np.subtract.at(ket, bq, Z2[:, aq, q].T)
        del Z2, Sq
    return bra.T, ket.T


def active_bytes(nocc, nvir, isz):
    '''Bytes of the work arrays of orbital_gradient() (active_terms,
    source_terms) besides the Gradient: two blocks of integrals with three
    virtual indices in the layout of a product (the blocks of the densities
    with at most two virtual indices meet them), <ab||ei> and <ie||bc> on
    pairs with the two arrays of their size of _vvvv_terms, the dense blocks
    of the small densities with their copies, a slab of <vv||vv>.'''
    o, v = nocc, nvir
    nQ = comb(v, 2)
    return isz * (2 * o * v ** 3 + 4 * o * v * nQ + 12 * o * o * v * v + 4 * o ** 3 * v + 2 * o ** 4
                  + 3 * v ** 3) + 4 * min(SLAB_BYTES, isz * nQ * max(nQ, o * v))


def active_terms(eris, g):
    '''Gnh[y,x] of E(Q) among the spin orbitals of ``eris`` without its
    block among the virtual ones: the first-order change of E(Q) with the
    integrals one-index transformed - a bra index y of an integral taking
    the values of x, a ket index x minus those of y -, the orbital energies
    as the diagonal of the Fock matrix of the rotated spin orbitals.  The
    densities with at most two virtual indices as dense blocks, every one
    with the blocks of integrals that one of its indices changed gives; the
    density of <ab||ei> on its pairs - with <kb||ei>, <ab||ki> and <ab||ek>,
    and with the packed <vv||vv> in slabs of rows -; that of <bc||ef> with
    the integrals <ie||bc> on pairs (_vvvv_terms).'''
    from zccsdt.cc import t3pack
    from zccsdt.cc.ccsdt_opt import vvvv_packed
    o = eris.nocc
    v = g.gamma_vv.shape[0]
    n = o + v
    dtype = np.result_type(g.vvvv.dtype, np.asarray(eris.oovv).dtype)
    G = np.zeros((n, n), dtype=dtype)
    sl = {'o': slice(0, o), 'v': slice(o, n)}
    for pat, D in dense_blocks(g, o, SMALL).items():
        for p in range(4):
            kd = pat[p]
            for kn in 'ov':
                ky, kx = (kn, kd) if p < 2 else (kd, kn)
                ipat = pat[:p] + kn + pat[p + 1:]
                if ky == kx == 'v':
                    continue
                val = _contract(gblock(eris, ipat), D, p)           # [new index, index of D]
                if p < 2:
                    G[sl[ky], sl[kx]] += val
                else:
                    G[sl[ky], sl[kx]] -= val.T
        del D
    a2, b2 = Q._pairs(v)
    i2, j2 = Q._pairs(o)
    nQ = len(a2)
    D3 = np.ascontiguousarray(g.vvvo, dtype=dtype)                              # [i, e, (a<b)]: of <ab||ei>
    # ---- the density of <ab||ei>
    # a bra index: Gnh[k, x] += sum <ky||ei> D[(xy), e, i], antisymmetric in (x, y)
    ovvo = np.asarray(eris.ovvo)
    U = np.zeros((v, v, v), dtype=dtype)
    T = np.zeros((o, v), dtype=dtype)
    for i in range(o if nQ else 0):
        _unpack_pair(D3[i], v, U)
        gemm(np.ascontiguousarray(ovvo[:, :, :, i], dtype=dtype).reshape(o, v * v), U.reshape(v, v * v).T, T, beta=1.0)
    del U
    G[:o, o:] += T
    # the ket index e: Gnh[e, k] -= sum <ab||ki> D[(ab), e, i]
    x = np.asarray(eris.oovv)[:, :, a2, b2]
    oovv_c = np.ascontiguousarray(np.conj(x), dtype=dtype)                      # [k, i, (ab)] = <ab||ki>
    T = np.zeros((o, v), dtype=dtype)
    for i in range(o):
        gemm(oovv_c[:, i], D3[i].T, T, beta=1.0)
    del oovv_c
    G[o:, :o] -= T.T
    # the ket index i: Gnh[i, k] -= sum <ab||ek> D[(ab), e, i], and with <cd||ea>: Gnh[l, a] -= ...
    x = np.asarray(eris.ovvv)[:, :, a2, b2]
    ovvv_p = np.ascontiguousarray(x, dtype=dtype)                               # [k, e, (ab)] = <ke||ab>
    del x
    G[:o, :o] += gemm(D3.reshape(o, v * nQ), np.conj(ovvv_p).reshape(o, v * nQ).T)
    vpp = vvvv_packed(eris, t3pack.triple_map(v))[0]
    rows = max(1, min(nQ, SLAB_BYTES // max(1, dtype.itemsize * (2 * v * v + o * v))))
    T = np.zeros((o, v), dtype=dtype)
    for b0, b1 in Q._batches(nQ, rows):
        nb = b1 - b0
        W = np.empty((nb, v, v), dtype=dtype)                                   # [(cd), e, a]
        P.tn_expand(W, np.ascontiguousarray(vpp[b0:b1], dtype=dtype), v, 1, 1, outer=0)
        A = np.ascontiguousarray(D3[:, :, b0:b1].transpose(0, 2, 1)).reshape(o, nb * v)
        gemm(A, W.reshape(nb * v, v), T, beta=1.0)
        del W, A
    G[:o, o:] -= T
    del D3
    # ---- the density of <bc||ef> with <ic||ef>
    bra, ket = _vvvv_terms(g.vvvv, ovvv_p, v)
    del ovvv_p
    G[:o, o:] += bra
    G[o:, :o] -= ket.T
    # the orbital energies: e_p = f[p,p], and the Fock operator has the occupied spin orbitals
    de = g.de
    V = (np.einsum('j,jaji->ai', de[:o], gblock(eris, 'ovoo'))
         - np.einsum('b,ibba->ai', de[o:], np.asarray(eris.ovvv)).conj())
    G[o:, :o] += V
    G[:o, o:] -= V.conj().T
    return G


def t3_terms(S3, T3, no, nv):
    '''The parts of source_terms() with s3 and t3, both on the packed block
    [C(no,3), C(nv,3)]: (-1/12 sum s3[j,k,l,a,b,c] t3[i,k,l,a,b,c] as [j,i],
    1/12 sum s3[i,j,k,a,c,d] t3[i,j,k,b,c,d] as [b,a]).'''
    dtype = np.result_type(S3, T3)
    nQ = comb(nv, 2)
    oo, vv = np.zeros((no, no), dtype=dtype), np.zeros((nv, nv), dtype=dtype)
    S3 = np.asarray(S3).reshape(comb(no, 3), -1)
    T3 = np.asarray(T3).reshape(comb(no, 3), -1)
    if S3.size:
        so = P.split(no, 1, 2)
        M = gemm(S3, T3.T)                                                      # [(jkl), (ikl)]
        oo -= np.einsum('jp,ip,jip->ji', so.psg.astype(float), so.psg.astype(float),
                        M[so.pos[:, None, :], so.pos[None, :, :]])
        del M
        Sx, Tx = (np.empty((nv, nQ), dtype=dtype) for _ in range(2))
        for I in range(S3.shape[0]):
            P.tn_expand(Sx, np.ascontiguousarray(S3[I], dtype=dtype), nv, 1, 2)
            P.tn_expand(Tx, np.ascontiguousarray(T3[I], dtype=dtype), nv, 1, 2)
            gemm(Tx, Sx.T, vv, beta=1.0)                                        # [b, a]
    return oo, vv


def source_terms(g, t2, T3=None):
    '''(Gnh[j,i], Gnh[b,a]) of the CCSDT Lagrangian among the occupied and
    among the virtual spin orbitals when lambda solves the equations with
    the sources s of a Gradient, dL/dt = -s: a rotation among them is a
    change of t (the Lagrangian does not change when t, lambda and the
    integrals are rotated together),

        Gnh[j,i] = -1/2 sum s2[j,k,a,b] t2[i,k,a,b] - 1/12 sum s3[j,k,l,a,b,c] t3[i,k,l,a,b,c]
        Gnh[b,a] =  1/2 sum s2[i,j,a,c] t2[i,j,b,c] + 1/12 sum s3[i,j,k,a,c,d] t3[i,j,k,b,c,d],

    in the T1-dressed basis and in the bare one alike (s1 = 0).  The parts
    with t3: those that the pass left in the Gradient (``src3``, t3_terms),
    or from ``T3`` on the packed block.'''
    o, v = t2.shape[0], t2.shape[2]
    s2 = g.s2
    dtype = np.result_type(s2, t2)
    oo = -0.5 * np.einsum('jkab,ikab->ji', s2, t2, optimize=True).astype(dtype)
    vv = 0.5 * np.einsum('ijac,ijbc->ba', s2, t2, optimize=True).astype(dtype)
    src3 = g.src3 if T3 is None else t3_terms(g.s3, T3, o, v)
    return oo + src3[0], vv + src3[1]


def orbital_gradient(eris, g, t1, t2, gnh, T3=None):
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
      transposed (qgrad_ref.fock_density) - that is the whole of it.

    Work arrays: active_bytes().'''
    o = eris.nocc
    oo, vv = source_terms(g, t2, T3)
    Gnh = active_terms(eris, g)
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
    CCSDT.  A batch at a time, as the frozen stage of density_opt has its
    own terms: every block of the densities with at most two virtual
    indices is laid out once for each of its four indices and meets the
    block of the integrals of the batch in one matrix product; the density
    of <ab||ei> stays on its pairs, and so do the integrals <f e||ab> that
    it and the density of <bc||ef> meet.  own(nk): the bytes of the work
    arrays for a batch of nk frozen spinors.'''

    def __init__(self, g, nocc, real=False):
        self.g, self.o, self.real = g, nocc, real
        self.v = v = g.gamma_vv.shape[0]
        o = nocc
        n = o + v
        self.dtype = g.vvvv.dtype if real else np.result_type(g.vvvv.dtype, np.complex128)
        sl = {'o': slice(0, o), 'v': slice(o, n)}
        # (the block of S, the index x, sign, conjugated, D laid out as [x, the indices of the block])
        self.lay = []
        for pat, D in dense_blocks(g, nocc, SMALL).items():
            s = [sl[c] for c in pat]
            for kind, x, cut, perm, sign in (('b', 0, (1, 2, 3), (0, 1, 2, 3), 1.0), ('b', 1, (0, 2, 3), (1, 0, 2, 3), -1.0),
                                             ('k', 2, (3, 0, 1), (2, 3, 0, 1), 1.0), ('k', 3, (2, 0, 1), (3, 2, 0, 1), -1.0)):
                M = np.ascontiguousarray(D.transpose(perm), dtype=self.dtype).reshape(D.shape[x], -1)
                self.lay.append((kind, s[x], tuple(s[c] for c in cut), sign, M))
        self.D3 = np.ascontiguousarray(g.vvvo, dtype=self.dtype)

    def kept_bytes(self):
        '''Bytes of what the object holds besides the Gradient.'''
        return sum(x[4].nbytes for x in self.lay) + (0 if self.D3 is self.g.vvvo else self.D3.nbytes)

    def own(self, nk):
        '''Bytes of the work arrays of a call for ``nk`` frozen spinors.'''
        o, v = self.o, self.v
        n, nQ = o + v, comb(v, 2)
        isz = self.dtype.itemsize
        return (isz * (nk * (n ** 3 + n * n * v + 4 * v * nQ + 2 * o * nQ) + 2 * v ** 3)
                + 4 * min(SLAB_BYTES, isz * nQ * max(nQ, nk * v)))

    def __call__(self, G):
        o, v, dtype = self.o, self.v, self.dtype
        n = o + v
        a2, b2 = Q._pairs(v)
        nQ = len(a2)
        nf = G.shape[0]
        b = np.zeros((nf, n), dtype=dtype)
        k = np.zeros((nf, n), dtype=dtype)
        Gf = G.real if self.real else G
        S = np.empty((nf, n, n, n), dtype=dtype)                                # [f, u, v, w] = <fu||vw>
        np.subtract(Gf.transpose(0, 2, 1, 3), Gf.transpose(0, 2, 3, 1), out=S)
        cj = (lambda x: x) if self.real else (lambda x: np.conj(x, out=x))
        for kind, sx, cut, sign, M in self.lay:
            X = np.ascontiguousarray(S[:, cut[0], cut[1], cut[2]]).reshape(nf, -1)
            if kind == 'k':
                X = cj(X)
            # bra: <f q||r s> and <q f||r s> = -<f q||r s>;  ket: <p q||f s> = <f s||p q>*, <p q||r f> = -<f r||p q>*
            gemm(X, M.T, (b if kind == 'b' else k)[:, sx], alpha=sign, beta=1.0)
            del X
        if nQ:
            D3 = self.D3
            # the density of <ab||ei>: its bra indices with <f y||e i>
            U = np.zeros((v, v, v), dtype=dtype)
            T = np.zeros((nf, v), dtype=dtype)
            for i in range(o):
                _unpack_pair(D3[i], v, U)
                gemm(np.ascontiguousarray(S[:, o:, o:, i]).reshape(nf, v * v), U.reshape(v, v * v).T, T, beta=1.0)
            del U
            b[:, o:] += T
            # its ket indices with <f i||ab>* and <f e||ab>*, on pairs
            SQ = cj(np.ascontiguousarray(S[:, :o, o + a2, o + b2]))             # [f, i, (ab)]
            T = np.zeros((nf, v), dtype=dtype)
            for i in range(o):
                gemm(SQ[:, i], D3[i].T, T, beta=1.0)
            del SQ
            k[:, o:] += T
            SQ = np.ascontiguousarray(S[:, o:, o + a2, o + b2])                 # [f, e, (ab)] = <fe||ab>
            del S
            bra, ket = _vvvv_terms(self.g.vvvv, SQ, v)
            b[:, o:] += bra
            k[:, o:] += ket
            SQ = cj(SQ)
            k[:, :o] -= gemm(SQ.reshape(nf, v * nQ), D3.reshape(o, v * nQ).T)
        return b, k
