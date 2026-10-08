'''The first derivatives of (Q)_Lambda from packed amplitudes: for one
occupied quadruple at a time, the sources of the two linear equations of
CCSDT(Q)_Lambda and the densities of the correction.

The functional and its derivatives are those of zccsdt.cc.qlgrad_ref (the
dense reference; docs/lambda.md, section 12):

    e = sum z r / D,   E(Q)_L = Re e,   y = r / D,   w = z / D,
    de = sum [ y dz + w dr ] - sum w y dD,

holomorphic in t2, t3, l2, l3 and the integrals: nothing is conjugated.  r
is the right-hand side of (Q) (zccsdt.cc.quadruples: four matrix products),
z = <0| (Lambda_2 + Lambda_3) V |Q> the three products of
zccsdt.cc.quadruples_lambda.  The derivative runs through the seven
backwards, as zccsdt.cc.quadruples_grad does for (Q): the left-hand
quadruples are expanded to the layouts of the products (the adjoint of the
reduction), and every product X = A B gives X_ B^T for the operand A and
A^T X_ for B:

    product                                    with   its two adjoints go to
    - sum_e <ab||ei> t(jkl,cde)                 w     the density of <ab||ei>;   s3
    sum_m <ma||ij> t(mkl,bcd)                   w     the density of <ma||ij>;   s3
    sum_m t(im,ab) Xo(m;jkl;cd)                 w     s2;          the adjoint of Xo
    2 sum_e t(ij,ae) Xv(e;kl;bcd)               w     s2;          the adjoint of Xv
    l(ij,ab) <kl||cd>                           y     q2;          the density of <kl||cd>
    sum_e <ie||ab> l(jkl,cde)                   y     the density of <ie||ab>;   q3
    sum_m <ij||ma> l(mkl,bcd)                   y     the density of <ij||ma>;   q3

s = de/dt is the part of the sources of the Lambda equations for lambda'
that comes from e, q = de/dlambda the source of the equation of zeta.  In
(Q) the two products of [V,T3] serve r and z at once and take the sum of
two left-hand vectors; here r and z have their own products, and one
left-hand vector each: four expansions of a quadruple (w and y in the two
layouts) where (Q) has five, seven products with two adjoints each where
(Q) has five.

The densities are returned in the convention of quadruples_grad.Gradient,
dE = Re sum (density x d integral) on the blocks <cd||kl>, <ab||ei>,
<ma||ij>, <mn||jk>, <md||el>, <bc||ef>: E(Q)_L is the real part of e, and
for the Hermitian integrals of a rotation of the spin orbitals the
coefficients of <kl||cd>, <ie||ab> and <ij||ma> are the conjugates of
coefficients of <cd||kl>, -<ab||ei> and <ma||ij>.  So the orbital gradient
of e and its terms with frozen spinors are those of quadruples_grad
(active_terms, FrozenTerms) on this Gradient.

The orbital energies: -Re(w y) summed for every occupied index of the
quadruple, and Gamma[a,e] = sum w[a,(bcd)] y[e,(bcd)] for the virtual ones
(qlgrad_ref.fock_density: its diagonal is the derivative with respect to
e_a, its Hermitian part the rotation among the virtual spin orbitals).

Memory (plan()): that of quadruples_grad with the arrays of the left-hand
side (quadruples_lambda), q3 - one more array of the size of T3 -, q2 and
the adjoints of the operands of z.  Where the largest batches do not fit,
s3, q3 and the density of <bc||ef> go to a scratch file (Plan.mapped).

The same pass in the sectors of symmetry labels: quadruples_lambda_grad_sym.

Timers (zccsdt.timing): those of (Q) and (Q)_Lambda for the forward
products, 'qg.vectors', 'qg.expand', 'qg.gemm', 'qg.reduce', 'qg.gamma',
'qg.scatter', 'qg.xv', 'qg.xo'.
'''

from dataclasses import dataclass
from math import comb

import numpy as np
from pyscf import lib

from zccsdt.blas import gemm
from zccsdt.cc import quadruples as Q, quadruples_grad as QG, quadruples_lambda as QL, t4pack as P
from zccsdt.timing import timer


@dataclass
class Gradient(QG.Gradient):
    '''E(Q)_L and the first derivatives of e = sum z r / D (see the module):
    a quadruples_grad.Gradient - e_q is E(Q)_L = Re e, s2 and s3 are de/dt,
    the densities those of dE = Re sum (density x d integral) - with'''
    q2: np.ndarray = None       # [i,j,a,b], antisymmetric: de = sum_(i<j,a<b) q2 dl2
    q3: np.ndarray = None       # [C(o,3), C(v,3)]: de = sum q3 dL3
    e_imag: float = 0.0         # Im e (zero for a Hamiltonian that makes the energy real)
    # src3: the parts of source_terms with t3 and with l3 (those of q3 included)


@dataclass
class Plan:
    q: Q.Plan                   # the batches of the products, as for (Q)
    need: int                   # bytes of all arrays in memory (without T3, Lambda_3 and the integrals)
    kept: int                   # of which are the derivatives that are returned
    floor: int                  # the least that a calculation takes
    mapped: tuple = ()          # of 'vvvv', 's3', 'q3': the derivatives that go to a scratch file


MAPPED = (('vvvv', 's3', 'q3'), ('vvvv',), ())      # from short memory to all of it


def _extra(no, nv, isz, mapped=()):
    '''Bytes (kept, work) of what the pass has besides the arrays of
    quadruples_grad: q2 and q3 (unless mapped); the arrays of the left-hand
    side (quadruples_lambda), the coefficients of l2 on pairs and the
    adjoints of the stacked operands of z.'''
    nP, nQ, nI3, nA3, nA = Q._sizes(no, nv)
    kept = isz * (no * no * nv * nv + (0 if 'q3' in mapped else nI3 * nA3))
    work = QL._extra(no, nv, isz) + isz * (nP * nQ + (4 * nv + 6) * nQ + 6 * no * nv)
    return kept, work


def _need(no, nv, isz, qp, mapped=()):
    '''Bytes of everything with the batches of the Plan of (Q) ``qp``:
    (all, of which are returned).'''
    need, kept = QG._need(no, nv, isz, qp, tuple(m for m in mapped if m != 'q3'))
    xk, xw = _extra(no, nv, isz, mapped)
    return need + xk + xw, kept + xk


def plan(no, nv, isz, budget=None, batch_bytes=None):
    '''The Plan for ``budget`` bytes (None: no limit) besides T3, Lambda_3
    and the integrals, as quadruples_grad.plan: the batches are those of the
    Plan of (Q) for the largest budget of its own with which everything
    fits (bisection: the plans of (Q) grow with their budget).  Where the
    largest batches do not fit, s3, q3 and the density of <bc||ef> go to a
    scratch file, and the batches take the memory; with the largest batches
    the density alone goes there if s3 and q3 fit, and nothing if all do.
    MemoryError below the floor: the smallest batches of (Q) with the three
    arrays mapped.'''
    top = Q.plan(no, nv, isz, None, batch_bytes)
    least = QG._least_plan(no, nv, isz)
    floor = _need(no, nv, isz, least, MAPPED[0])[0]
    qp, mapped = top, ()
    if budget is not None and _need(no, nv, isz, top)[0] > budget:
        if budget < floor:
            on_file = _need(no, nv, isz, least)[1] - _need(no, nv, isz, least, MAPPED[0])[1]
            raise MemoryError('the derivatives of (Q)_Lambda need %.1f MB besides T3, Lambda_3 and the '
                              'integrals for %d occupied and %d virtual spin orbitals (with %.1f MB more in '
                              'a scratch file for those of the sizes of T3 and <vv||vv>); %.1f MB are left'
                              % (floor / 1e6, no, nv, on_file / 1e6, budget / 1e6))
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
    return Plan(qp, need, kept, _need(no, nv, isz, QG._least_plan(no, nv, isz), MAPPED[0])[0], mapped)


class QuadruplesLambdaGradient(QG.QuadruplesGradient):
    '''r of (Q), z of (Q)_Lambda and the derivative of e for one occupied
    quadruple at a time.'''

    left_lambda = QL.QuadruplesLambda.left_lambda

    def __init__(self, eris, t2, t3, l2, l3, plan_=None):
        no = eris.nocc
        nv = np.asarray(eris.fock).shape[0] - no
        dtype = np.result_type(t2, t3, l2, l3, eris.oovv)
        isz = np.dtype(dtype).itemsize
        i2, j2 = Q._pairs(no)
        a2, b2 = Q._pairs(nv)
        self.l2_pp = np.ascontiguousarray(np.asarray(l2)[i2, j2][:, a2, b2], dtype=dtype)   # [(ij), (ab)]
        if np.dtype(dtype) != np.result_type(t2, t3, eris.oovv):
            t2 = np.asarray(t2, dtype=dtype)
        super().__init__(eris, t2, t3, plan_ or plan(no, nv, isz))
        nQ, nA3, nP = self.nQ, self.nA3, comb(no, 2)
        self.cplx = np.iscomplexobj(self.vvoo_t)
        self.oovv_t = np.conj(self.vvoo_t) if self.cplx else self.vvoo_t        # <kl||cd> as [(kl), (cd)]
        self.L3 = np.ascontiguousarray(np.asarray(l3).reshape(comb(no, 3), nA3), dtype=dtype)
        self.ZA, self.ZRA = (np.empty((4 * nv, nQ), dtype=dtype) for _ in range(2))
        self.ZC, self.ZRC = np.empty((6 * no, nv), dtype=dtype), np.empty((6 * no, nA3), dtype=dtype)
        self.z3 = np.empty(self.nA, dtype=dtype)
        # what is returned besides the arrays of (Q): q3 like s3, the coefficients of l2 on pairs
        if 'q3' in self.gplan.mapped:
            self.Q3 = self.scratch.create('q3', (comb(no, 3), nA3), dtype)
        else:
            self.Q3 = np.zeros((comb(no, 3), nA3), dtype=dtype)
        self.bq_pp = np.zeros((nP, nQ), dtype=dtype)
        self.e_imag = 0.0
        # the adjoints of the stacked operands of z
        self.ZAb = np.zeros((4 * nv + 6, nQ), dtype=dtype)      # (six rows more: of l2 in its product)
        self.ZCb = np.zeros((6 * no, nv), dtype=dtype)

    # ---- one occupied quadruple

    def back(self, I, Xv, where):
        '''The derivative of the terms of the occupied quadruple I, after
        left_lambda(I) and right(I, Xv, where): their results and the
        stacked operands are those of this quadruple.  Returns its e.'''
        no, nv, nQ, nA3 = self.nocc, self.nvir, self.nQ, self.nA3
        so13, so22, so12 = self.so13, self.so22, self.so12
        x13, J13, s13 = so13.idx1[I], so13.idx2[I], so13.sgn[I]
        p22, q22, s22 = so22.idx1[I], so22.idx2[I], so22.sgn[I]
        z2, z3, rt3, rtt, y, c = self.z, self.z3, self.rt3, self.rtt, self.y, self.c
        cplx = np.iscomplexobj(z2)
        with timer('qg.vectors'):
            eo, ev4 = self.eo4[I], self.ev4

            def part(lo, hi):
                # (slices on threads: numpy does these one core at a time)
                sl = slice(lo, hi)
                d = eo - ev4[sl]
                zz, r3, rr, yy, cc_ = z2[sl], rt3[sl], rtt[sl], y[sl], c[sl]
                np.add(r3, rr, out=yy)
                yy /= d                             # y = r / D
                zz += z3[sl]
                zz /= d                             # w = z / D
                np.multiply(zz, yy, out=rr)         # w y
                cc_[:] = rr.real
                out = [float(cc_.sum()), float(cc_ @ d), 0.0]
                if cplx:
                    cc_[:] = rr.imag
                    out[2] = float(cc_ @ d)
                return out
            step = max(1 << 16, -(-len(y) // max(1, lib.num_threads())))
            sums = QG._threaded(lambda lh: part(*lh), Q._batches(len(y), step))
            self.de_occ[self.quads[I]] -= sum(x[0] for x in sums)
            e = sum(x[1] for x in sums)
            self.e_imag += sum(x[2] for x in sums)
        w = z2
        # the operands of the product of l2, as left_lambda had them
        Lz = self.l2_pp[p22] * s22[:, None]                                     # [6, (ab)]
        Rz = self.oovv_t[q22]                                                   # [6, (cd)]
        for a in (self.LAb, self.LBb, self.LCb, self.ZAb, self.ZCb):
            P.zero(a)
        Lzb = self.ZAb[4 * nv:]
        na = 4 * nv
        for cc in Q._batches(nQ, self.plan.cols22):
            nc = cc[1] - cc[0]
            Ew, Ey = (b[:nQ * nc].reshape(nQ, nc) for b in self.e22)
            X = self._x22(cc)
            Xa = X.reshape(-1)[:na * nc].reshape(na, nc) if na <= nQ else np.empty((na, nc), dtype=self.dtype)
            with timer('qg.expand'):
                P.tn_expand(Ew, w, nv, 2, 2, cols=cc)
                P.tn_expand(Ey, y, nv, 2, 2, cols=cc)
            with timer('qg.gemm'):
                # t(im,ab) Xo(m;jkl;cd), with w
                gemm(self.RB[:, cc[0]:cc[1]], Ew.T, self.LBb[:4 * no], beta=1.0)
                gemm(self.LB, Ew, self.RBb[:4 * no, cc[0]:cc[1]])
                # l(ij,ab) <kl||cd>, with y
                gemm(Rz[:, cc[0]:cc[1]], Ey.T, Lzb, beta=1.0)
                gemm(Lz, Ey, self.Rzb[:, cc[0]:cc[1]])
                # <ab||ei> t(jkl,cde), with w
                gemm(self.RA[:, cc[0]:cc[1]], Ew.T, self.LAb[:na], beta=1.0)
                gemm(self.LA, Ew, Xa)
            with timer('qg.reduce'):
                for k in range(4):
                    P.tn_reduce(self.S3[J13[k]], np.ascontiguousarray(Xa[k * nv:(k + 1) * nv]), nv, 1, 2, cols=cc)
            with timer('qg.gemm'):
                # <ie||ab> l(jkl,cde), with y
                gemm(self.ZRA[:, cc[0]:cc[1]], Ey.T, self.ZAb[:na], beta=1.0)
                gemm(self.ZA, Ey, Xa)
            with timer('qg.reduce'):
                for k in range(4):
                    P.tn_reduce(self.Q3[J13[k]], np.ascontiguousarray(Xa[k * nv:(k + 1) * nv]), nv, 1, 2, cols=cc)
            del Xa
        cj = np.conj if cplx else (lambda x: x)
        with timer('qg.scatter'):
            for k in range(6):
                self.bq_pp[p22[k]] += int(s22[k]) * Lzb[k]
                self.Dvvoo[q22[k]] += cj(self.Rzb[k])           # of <kl||cd>, as that of <cd||kl>
            for k in range(4):
                x, J, s = x13[k], J13[k], int(s13[k])
                self.b_xp[x] += s * self.LBb[k * no:(k + 1) * no]
                self.Xobar[J] += self.RBb[k * no:(k + 1) * no]
                # LA = -s <ab||ei>, ZA = s <ie||ab> = -s conj(<ab||ei>)
                self.Dvvvo[x] -= s * (self.LAb[k * nv:(k + 1) * nv] + cj(self.ZAb[k * nv:(k + 1) * nv]))
        rows = so12.pos[:, q22].T.ravel()
        sgn = so12.psg[:, q22].T.ravel()
        last = [(p22[k], where[int(P.tuples(no, 2)[q22[k], 0])], 2.0 * int(s22[k])) for k in (0, 1, 3)]
        for cc in Q._batches(nA3, self.plan.cols13):
            nc = cc[1] - cc[0]
            Ew, Ey = (b[:nv * nc].reshape(nv, nc) for b in self.e13)
            Xc = self.xc[:6 * no * nc].reshape(6 * no, nc)                      # [(split, m), (bcd)]
            with timer('qg.expand'):
                P.tn_expand(Ew, w, nv, 1, 3, cols=cc)
                P.tn_expand(Ey, y, nv, 1, 3, cols=cc)
            with timer('qg.gamma'):
                gemm(Ew, Ey.T, self.gamma, beta=1.0)
            with timer('qg.gemm'):
                # <ma||ij> t(mkl,bcd), with w
                gemm(self.RC[:, cc[0]:cc[1]], Ew.T, self.LCb, beta=1.0)
                gemm(self.LC, Ew, Xc)
            with timer('qg.reduce'):
                QG._axpy_rows(self.S3, Xc, rows, sgn, cc[0])
            with timer('qg.gemm'):
                # <ij||ma> l(mkl,bcd), with y
                gemm(self.ZRC[:, cc[0]:cc[1]], Ey.T, self.ZCb, beta=1.0)
                gemm(self.ZC, Ey, Xc)
            with timer('qg.reduce'):
                QG._axpy_rows(self.Q3, Xc, rows, sgn, cc[0])
            with timer('qg.gemm'):
                # 2 t(ij,ae) Xv(e;kl;bcd), with w
                for p, n, alpha in last:
                    gemm(Xv[n][:, cc[0]:cc[1]], Ew.T, self.b_px[p], alpha=alpha, beta=1.0)
                    gemm(self.t2_px[p], Ew, self.Xvbar[n][:, cc[0]:cc[1]], alpha=alpha, beta=1.0)
        with timer('qg.scatter'):
            for k in range(6):
                # LC = s <ma||ij>, ZC = s <ij||ma> = s conj(<ma||ij>)
                self.Dovoo[p22[k]] += int(s22[k]) * (self.LCb[k * no:(k + 1) * no] + cj(self.ZCb[k * no:(k + 1) * no]))
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
            Doooo, Dovvo = QG.back_xo(self, self.eris, self.Xobar, self.b_xp, self.G2)
        s2 = QG.sources2(no, nv, self.G2, 0.0, self.b_xp, self.b_px)
        self.b_pp = self.b_xp = self.b_px = self.G2 = self.Xobar = None
        q2 = pairs_to_dense(self.bq_pp, no, nv)
        self.bq_pp = None
        return Gradient(self.e_q, s2, self.S3, self.de_occ, self.gamma, self.Dvvoo, self.Dvvvo,
                        self.Dovoo, Doooo, Dovvo, self.Dvvvv, self.gplan,
                        t3_terms(self.S3, self.T3, self.Q3, self.L3, no, nv), self.scratch,
                        q2, self.Q3, self.e_imag)


def pairs_to_dense(b_pp, no, nv):
    '''x[i,j,a,b], antisymmetric, of its unique elements b_pp[(i<j), (a<b)].'''
    i2, j2 = Q._pairs(no)
    a2, b2 = Q._pairs(nv)
    G = np.zeros((no, no, nv, nv), dtype=b_pp.dtype)
    G[i2[:, None], j2[:, None], a2[None, :], b2[None, :]] = b_pp
    G -= G.transpose(1, 0, 2, 3)
    return G - G.transpose(0, 1, 3, 2)


def t3_terms(S3, T3, Q3, L3, no, nv):
    '''The parts of source_terms() with the packed arrays: those of s3 with
    t3 (quadruples_grad.t3_terms) and of q3 with l3,

        [j,i]: -1/12 sum s3[j,k,l,a,b,c] t3[i,k,l,a,b,c] + 1/12 sum q3[i,k,l,a,b,c] l3[j,k,l,a,b,c]
        [b,a]:  1/12 sum s3[i,j,k,a,c,d] t3[i,j,k,b,c,d] - 1/12 sum q3[i,j,k,b,c,d] l3[i,j,k,a,c,d].'''
    a, b = QG.t3_terms(S3, T3, no, nv)
    c, d = QG.t3_terms(L3, Q3, no, nv)
    return a - c, b - d


def source_terms(g, t2, l2):
    '''(Gnh[j,i], Gnh[b,a]) of the Lagrangian of CCSDT(Q)_Lambda without the
    densities of e, among the occupied and among the virtual spin orbitals:
    L and zeta . W do not change when t, lambda, zeta, lambda' and the
    integrals are rotated together, and at the solution of the two linear
    equations what is left of dF/dt and dF/dlambda are the derivatives of e
    (docs/lambda.md 12.5, identity 1),

        Gnh[j,i] = -1/2 sum s2[j,k,a,b] t2[i,k,a,b] - 1/12 sum s3[j,k,l,a,b,c] t3[i,k,l,a,b,c]
                   +1/2 sum q2[i,k,a,b] l2[j,k,a,b] + 1/12 sum q3[i,k,l,a,b,c] l3[j,k,l,a,b,c]
        Gnh[b,a] =  1/2 sum s2[i,j,a,c] t2[i,j,b,c] + 1/12 sum s3[i,j,k,a,c,d] t3[i,j,k,b,c,d]
                   -1/2 sum q2[i,j,b,c] l2[i,j,a,c] - 1/12 sum q3[i,j,k,b,c,d] l3[i,j,k,a,c,d]

    with lambda of CCSDT (it rotates the other way round).  The parts with
    the packed arrays are those that the pass left in the Gradient (src3).'''
    s2, q2 = g.s2, g.q2
    dtype = np.result_type(s2, q2, t2, l2)
    es = lambda *a: np.einsum(*a, optimize=True).astype(dtype)                  # noqa: E731
    oo = -0.5 * es('jkab,ikab->ji', s2, t2) + 0.5 * es('ikab,jkab->ji', q2, l2)
    vv = 0.5 * es('ijac,ijbc->ba', s2, t2) - 0.5 * es('ijbc,ijac->ba', q2, l2)
    return oo + g.src3[0], vv + g.src3[1]


def orbital_gradient(eris, g, t1, t2, l2, gnh):
    '''G[y,x] of zccsd.zvector among the spin orbitals of ``eris`` for
    CCSDT(Q)_Lambda, anti-Hermitian, as quadruples_grad.orbital_gradient is
    that of CCSDT(Q).  ``gnh`` [a,i]: the table GRADIENT of the CCSDT
    Lagrangian (density_opt.kernel) for (t, lambda') plus its derivative
    along zeta in t at (t, lambda); ``l2``: of lambda of CCSDT.

    - virtual-occupied: gnh with the two blocks of source_terms and t1
      (identity 2 of docs/lambda.md 12.5), and the densities of e with the
      integrals (quadruples_grad.active_terms);
    - among the occupied: source_terms and active_terms;
    - among the virtual: (e_q - e_p) times the Hermitian part of Gamma,
      transposed.

    Work arrays: quadruples_grad.active_bytes().'''
    o = eris.nocc
    oo, vv = source_terms(g, t2, l2)
    Gnh = QG.active_terms(eris, g)
    Gnh = Gnh.astype(np.result_type(Gnh, gnh, oo, t1))
    Gnh[:o, :o] += oo
    Gnh[o:, :o] += gnh + t1.T @ oo - vv @ t1.T
    G = 0.5 * (Gnh - Gnh.conj().T)
    ev = np.asarray(eris.mo_energy)[o:]
    gam = 0.5 * (g.gamma_vv + g.gamma_vv.conj().T)
    G[o:, o:] = (ev[None, :] - ev[:, None]) * gam.T
    return G


def kernel(eris, t2, t3, l2, l3, max_memory=None, plan_=None):
    '''The Gradient of (Q)_Lambda for converged CCSDT amplitudes and Lambda:
    t2[i,j,a,b], l2[i,j,a,b], T3 and Lambda_3 packed [C(no,3), C(nv,3)] (or
    flat, the tuples in lexicographic order), on an integral container
    (zccsdt.eris.ERIs).  ``max_memory``: MB for the whole process (default:
    what the node and the job leave); ``plan_``: a Plan given explicitly.
    At least four occupied and four virtual spin orbitals.'''
    from zccsdt import memory
    no = eris.nocc
    nv = np.asarray(eris.fock).shape[0] - no
    if no < 4 or nv < 4:
        raise ValueError('(Q)_Lambda needs four occupied and four virtual spin orbitals')
    if plan_ is None:
        isz = np.dtype(np.result_type(t2, t3, l2, l3, eris.oovv)).itemsize
        plan_ = plan(no, nv, isz, memory.budget(max_memory))
    q = QuadruplesLambdaGradient(eris, t2, t3, l2, l3, plan_)
    for l, xs, Is in q.schedule():
        Xv, where = q.xi_v(l, xs)
        for I in Is:
            with timer('ql.left'):
                q.left_lambda(I)
            q.right(I, Xv, where)
            q.back(I, Xv, where)
        del Xv
    return q.finish()
