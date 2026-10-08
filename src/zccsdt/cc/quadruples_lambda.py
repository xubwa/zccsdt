'''CCSDT(Q)_Lambda on spin orbitals or spinors, real or complex: the (Q)
correction with Lambda_2 and Lambda_3 on the left (docs/lambda.md, section
5; zccsdt.cc.lambda_ref.quadruples_lambda is the definition).

    E[Q]_L = sum z2 r / D,       r  = <Q| [V,T3] + 1/2 [[V,T2],T2] |0>,
    E(Q)_L = sum (z2 + z3) r / D,
                                 z2 = <0| Lambda_2 V |Q>   (disconnected),
                                 z3 = <0| Lambda_3 V |Q>   (connected)

(the real part), summed over the unique quadruples; bare integrals, canonical
orbitals, t1 and l1 do not enter.  r is that of zccsdt.cc.quadruples, formed
by the same code.  In (Q) the left-hand side is the conjugate of
<Q| V (T2 + T3) |0>, and its T3 part is the [V,T3] part of r: no product of
its own.  Here it has two more products per occupied quadruple, those of
[V,T3] with l3 in the place of T3 and the integrals the other way round:

    X[(ab),(cd)]  + l(ij,ab) <kl||cd>                     (ij|kl), 6 splits     z2
                  + sum_e <ie||ab> l(jkl,cde)             (i|jkl), 4 splits     z3
    X[a,(bcd)]    + sum_m <ij||ma> l(mkl,bcd)             (ij|kl), 6 splits     z3

The integrals of the right-hand side are <ab||ei> = -<ie||ab>* and
<ma||ij> = <ij||ma>*: the stacked operands of the left-hand side are the
complex conjugates of those of [V,T3], and nothing else is conjugated - with
l2 = t2*, l3 = T3* the two energies are those of (Q).

Cost: the two products of [V,T3] once more, o^4 v^4 (v + o) / 24 of the
o^4 v^4 (v + o) / 12 of (Q) (and of its o^2 v^5 / 4 for the intermediate of
T2 with <vv||vv>, nothing).

Memory (plan()): the plan of (Q) with, besides Lambda_3, which is given like
T3: l2 and <oo||vv> on pairs, the stacked operands of the two products of
Lambda_3 and z3 - no more products at a time than in (Q).
'''

import ctypes
import os
from dataclasses import replace
from math import comb

import numpy as np

from zccsdt.blas import gemm
from zccsdt.cc import quadruples as Q, t4pack as P
from zccsdt.timing import timer

_I64, _DBL = ctypes.c_int64, ctypes.c_double
_p = P._p


def _load_lib():
    path = os.environ.get('ZCCSDT_QLAMBDA_LIB') or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lib', 'libzc_qlambda.so')
    try:
        return ctypes.CDLL(path)
    except OSError:
        return None


_lib = _load_lib()
USE_C = _lib is not None        # set False to force the numpy implementation


def energy2(z2, z3, r1, r2, ev, eo):
    '''(sum Re(z2 (r1 + r2)) / (eo - ev), sum Re(z3 (r1 + r2)) / (eo - ev)).'''
    if USE_C:
        e = np.zeros(2)
        getattr(_lib, 'zc_ql_energy_' + P._SFX[z2.dtype])(_p(z2), _p(z3), _p(r1), _p(r2), _p(ev), _DBL(eo),
                                                         _I64(len(z2)), _p(e))
        return float(e[0]), float(e[1])
    r = (r1 + r2) / (eo - ev)
    return float(np.dot(z2, r).real), float(np.dot(z3, r).real)


# ---------------------------------------------------------------- memory

def _extra(no, nv, isz):
    '''Bytes of the arrays of the left-hand side: l2 and <oo||vv> on pairs
    (the latter is <vv||oo> of (Q) where the integrals are real), the
    stacked operands of the two products of Lambda_3, z3.'''
    nP, nQ, nI3, nA3, nA = Q._sizes(no, nv)
    return isz * (nP * nQ * (2 if isz == 16 else 1) + 8 * nv * nQ + 6 * no * (nv + nA3) + nA)


def _with_extra(pl, no, nv, isz):
    x = _extra(no, nv, isz)
    return replace(pl, need=pl.need + x, fixed=pl.fixed + x, floor=pl.floor + x)


def plan(no, nv, isz, budget=None, batch_bytes=None):
    '''The Plan (zccsdt.cc.quadruples.Plan) for ``budget`` bytes (None: no
    limit) besides T3, Lambda_3 and the integrals: that of (Q) for what the
    arrays of the left-hand side leave.'''
    x = _extra(no, nv, isz)
    if budget is not None:
        floor = Q.plan(no, nv, isz, None, batch_bytes).floor + x
        if budget < floor:
            raise MemoryError('(Q)_Lambda needs %.0f MB besides T3, Lambda_3 and the integrals for %d '
                              'occupied and %d virtual spin orbitals; %.0f MB are left'
                              % (floor / 1e6, no, nv, budget / 1e6))
        budget -= x
    return _with_extra(Q.plan(no, nv, isz, budget, batch_bytes), no, nv, isz)


def explicit_plan(no, nv, isz, cols22, cols13, xv, rows):
    '''The Plan with these batch sizes.'''
    return _with_extra(Q.explicit_plan(no, nv, isz, cols22, cols13, xv, rows), no, nv, isz)


# ------------------------------------------------------------- the correction

class QuadruplesLambda(Q.Quadruples):
    '''r of (Q) and the left-hand side of (Q)_Lambda for one occupied
    quadruple at a time.'''

    def __init__(self, eris, t2, t3, l2, l3, plan_=None):
        no = eris.nocc
        nv = np.asarray(eris.fock).shape[0] - no
        dtype = np.result_type(t2, t3, l2, l3, eris.oovv)
        i2, j2 = Q._pairs(no)
        a2, b2 = Q._pairs(nv)
        # (first: nothing else is there yet while l2 on (ij) with all a, b is)
        self.l2_pp = np.ascontiguousarray(np.asarray(l2)[i2, j2][:, a2, b2], dtype=dtype)   # [(ij), (ab)]
        if np.dtype(dtype) != np.result_type(t2, t3, eris.oovv):
            t2 = np.asarray(t2, dtype=dtype)
        super().__init__(eris, t2, t3, plan_ or plan(no, nv, np.dtype(dtype).itemsize))
        nQ, nA3 = self.nQ, self.nA3
        self.cplx = np.iscomplexobj(self.vvoo_t)
        self.oovv_t = np.conj(self.vvoo_t) if self.cplx else self.vvoo_t        # <kl||cd> as [(kl), (cd)]
        self.L3 = np.ascontiguousarray(np.asarray(l3).reshape(comb(no, 3), nA3), dtype=dtype)
        self.ZA, self.ZRA = (np.empty((4 * nv, nQ), dtype=dtype) for _ in range(2))
        self.ZC, self.ZRC = np.empty((6 * no, nv), dtype=dtype), np.empty((6 * no, nA3), dtype=dtype)
        self.z3 = np.empty(self.nA, dtype=dtype)

    def left_lambda(self, I):
        '''(z2, z3) = <0| Lambda_2 V |Q>, <0| Lambda_3 V |Q> for the occupied
        quadruple I, [C(nv,4)] each (work arrays: valid until the next
        call).'''
        no, nv, nQ, nA3, L3 = self.nocc, self.nvir, self.nQ, self.nA3, self.L3
        so13, so22, so12 = self.so13, self.so22, self.so12
        x13, J13, s13 = so13.idx1[I], so13.idx2[I], so13.sgn[I]
        p22, q22, s22 = so22.idx1[I], so22.idx2[I], so22.sgn[I]
        z2, z3 = self.z, self.z3
        P.zero(z2)
        P.zero(z3)
        with timer('ql.operands'):
            L = self.l2_pp[p22] * s22[:, None]                                  # [6, (ab)]
            Rz = self.oovv_t[q22]                                               # [6, (cd)]
            # the operands of [V,T3] of the right-hand side, conjugated: <ie||ab> as [e, (ab)]
            # and <ij||ma> as [m, a], with their signs
            for k in range(4):
                x, J, s = x13[k], J13[k], int(s13[k])
                A = self.ZA[k * nv:(k + 1) * nv]
                if self.cplx:
                    np.conj(self.vvvo_t[x], out=A)
                    A *= -s
                else:
                    np.multiply(self.vvvo_t[x], -s, out=A)
                P.tn_expand(self.ZRA[k * nv:(k + 1) * nv], L3[J], nv, 1, 2)             # l(jkl; e, cd)
            for k in range(6):
                C = self.ZC[k * no:(k + 1) * no]
                if self.cplx:
                    np.conj(self.ovoo_t[p22[k]], out=C)
                    C *= int(s22[k])
                else:
                    np.multiply(self.ovoo_t[p22[k]], int(s22[k]), out=C)
            P.tn_rows(self.ZRC, L3, so12.pos[:, q22].T.ravel(), so12.psg[:, q22].T.ravel())      # l(m kl; bcd)
        for c in Q._batches(nQ, self.plan.cols22):
            X = self._x22(c)
            with timer('ql.gemm'):
                gemm(L.T, Rz[:, c[0]:c[1]], X)
            with timer('ql.reduce'):
                P.tn_reduce(z2, X, nv, 2, 2, cols=c)
            with timer('ql.gemm'):
                gemm(self.ZA.T, self.ZRA[:, c[0]:c[1]], X)
            with timer('ql.reduce'):
                P.tn_reduce(z3, X, nv, 2, 2, cols=c)
        for c in Q._batches(nA3, self.plan.cols13):
            X = self.w13[:nv * (c[1] - c[0])].reshape(nv, c[1] - c[0])
            with timer('ql.gemm'):
                gemm(self.ZC.T, self.ZRC[:, c[0]:c[1]], X)
            with timer('ql.reduce'):
                P.tn_reduce(z3, X, nv, 1, 3, cols=c)
        return z2, z3


def kernel(eris, t2, t3, l2, l3, max_memory=None, plan_=None):
    '''(E[Q]_L, E(Q)_L) from converged CCSDT amplitudes and Lambda:
    t2[i,j,a,b], l2[i,j,a,b], T3 and Lambda_3 packed [C(no,3), C(nv,3)] (or
    flat, the tuples in lexicographic order), on an integral container
    (zccsdt.eris.ERIs).  ``max_memory``: MB for the whole process (default:
    what the node and the job leave); ``plan_``: a Plan given explicitly.'''
    from zccsdt import memory
    no = eris.nocc
    nv = np.asarray(eris.fock).shape[0] - no
    if no < 4 or nv < 4:
        return 0.0, 0.0
    if plan_ is None:
        isz = np.dtype(np.result_type(t2, t3, l2, l3, eris.oovv)).itemsize
        plan_ = plan(no, nv, isz, memory.budget(max_memory))
    q = QuadruplesLambda(eris, t2, t3, l2, l3, plan_)
    e5 = e6 = 0.0
    for l, xs, Is in q.schedule():
        Xv, where = q.xi_v(l, xs)
        for I in Is:
            with timer('ql.left'):
                z2, z3 = q.left_lambda(I)
            rt3, rtt = q.right(I, Xv, where)
            with timer('ql.energy'):
                e = energy2(z2, z3, rt3, rtt, q.ev4, q.eo4[I])
            e5, e6 = e5 + e[0], e6 + e[1]
        del Xv
    return float(e5), float(e5 + e6)


def rows(eris, t2, t3, l2, l3, plan_=None):
    '''(z2, z3, r of [V,T3], r of the T2 T2 terms) for all quadruples,
    [C(no,4), C(nv,4)] each: for tests and small cases.'''
    q = QuadruplesLambda(eris, t2, t3, l2, l3, plan_)
    out = [np.zeros((len(q.eo4), q.nA), dtype=q.dtype) for _ in range(4)]
    for l, xs, Is in q.schedule():
        Xv, where = q.xi_v(l, xs)
        for I in Is:
            out[0][I], out[1][I] = q.left_lambda(I)
            out[2][I], out[3][I] = q.right(I, Xv, where)
    return out
