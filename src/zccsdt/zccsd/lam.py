'''The Lambda equations of the AO-driven spinor CCSD on packed amplitudes.

The equations are those of pyscf.cc.gccsd_lambda (lambda1[i,a], lambda2
[i,j,a,b]: the de-excitation amplitudes of the left ground state, conj(t) in
first order), evaluated in the layouts of zccsdt.zccsd.amps and with what its
Residual holds: the integrals, the memory plan, the kernels.

What depends on t alone is made once (Lambda.__init__), by the first pass of
the amplitude step with the full product <mn||ef> t2: elements of the
similarity-transformed Hamiltonian Hbar = exp(-T) H exp(T),

    Fov, Hoo, Hvv       its one-particle blocks,
    Whh[(i>j),(k>l)]    <ij|Hbar|kl> on packed pairs,
    W[(m e),(j b)]      <mb|Hbar|ej> in the ring layout (o^2 v^2 elements: in
                        memory or a mapped scratch array, as in a step),
    Hq[i,c,(j>k)]       <ic|Hbar|jk>-like, with three occupied indices,
    res1[i,a]           <ia|Hbar|0>, the T1 equation: zero for converged t.

No intermediate with three virtual indices is made.  The terms of the
lambda1 equation that contract lambda2 with <ab|Hbar|ci> are evaluated from
its parts instead:

- the integral itself: the rows of lambda2 of the pairs that contain k with
  <ka||bc> packed over b > c;
- its ring part, sum <kb||ad> t2[j,k,c,d]: the product Q = L T of the ring
  layouts of lambda2 and t2 (o^3 v^3, the second product of this size next
  to the ring term L W^T of the lambda2 equation), in blocks of its columns,
  each contracted with the rows of <ma||ef> of its occupied index;
- the t1 <mb|Hbar|ej> part, with the rows of W as they are read for the ring
  term;
- the ladder part as lambda2 <ab||cd> t1: the pp ladder of the lambda2
  equation (AO-driven: amps.Residual.ladder with the two sets of
  coefficients exchanged) contracted with t1.

Per iteration: two o^3 v^3 products, the AO ladder, one pass over <ma||ef>
(one GEMM with lambda2 t1 for a slab) and one over W.

Kramers pairs.  With t and lambda symmetric under time reversal, the terms
that are sums over the first index of <ma||ef> run over the unbarred m and
are completed by their image (kernels.ksym_packed for the packed array, in
place: these terms therefore go first into the new array), of the ring
term the rows with unbarred i are computed, in blocks of unbarred j and
their partners, and the ladder is done for one pair of each two partners.

What the lambda1 equation contracts with t1 - the hh ladder, the tau term,
the pp ladder - is taken from the new lambda2 array once these are in it.
The array then holds four terms more (the two of the pass over <ma||ef>,
two one-particle terms that ride on the matrices of the hh ladder and of
the tau term): their contractions with t1 are taken back term by term.

Memory.  The iterations are planned by amps.plan_step for the budget less
reserve_bytes, the arrays of this module that the plan does not know;
working_set is what a step takes (tests/test_zccsd_lambda.py holds the
allocations against it).  kernel() releases the unpacked t2 of the CCSD
object while it runs.

Timers (zccsdt.timing): 'lambda.imds' (the intermediates, once); per
iteration 'lambda.pairs', 'lambda.ladder', 'lambda.ovvv' (with 'ovvv.gemm',
'ovvv.rows', 'ovvv.x'), 'lambda.ring', 'lambda.t2', 'lambda.t1',
'lambda.kramers', and 'ring.gemm', 'ring.layout', 'ring.pack' inside them.
'''

from types import SimpleNamespace

import numpy as np
from pyscf.lib import logger

from zccsdt import blas, scratch
from zccsdt.cc.diis import DIIS
from zccsdt.timing import timer
from zccsdt.zccsd import amps
from zccsdt.zccsd import kernels as K

UNPACK_BYTES = 64 << 20     # rows of <ma||ef> unpacked at a time, at most


def reserve_bytes(nocc, nvir):
    '''Bytes of the arrays of the Lambda equations that the memory plan of
    the amplitude step (amps.plan_step) does not know: Hq and lambda2 t1
    with three occupied indices (while Hq is made: Hq and an array of twice
    its size), four matrices on the pairs, what the terms on the rows of
    <ma||ef> take for one occupied index (three arrays of o v^2 elements,
    the rows that are unpacked, two arrays of o v(v-1)/2 elements), and
    the third array of the product L T for one occupied index (_q_block).'''
    o, v = int(nocc), int(nvir)
    npo, npv = K.npair(o), K.npair(v)
    kept = 16 * (3 * o * v * npo + 4 * npo * npo + 2 * o ** 4)
    rows = 64 * o * v * v + 32 * o * npv
    return kept + rows + 16 * o * v * v


def _q_block(plan, o, v):
    '''Occupied indices in a block of columns of the product L T.  The
    blocks of the plan of the amplitude step hold two arrays of the size of
    a block of columns (four with the real parts of three real GEMMs, which
    is what this product takes as well); with ZGEMM this product has three:
    its blocks are two thirds of those, more where the plan leaves memory.'''
    col = 16 * o * v * v
    size = max(j1 - j0 for j0, j1 in plan.jblocks)
    msize = min(size, max(m1 - m0 for m0, m1 in plan.mblocks))
    if plan.three_m or plan.budget is None:
        return msize
    slack = max(0, plan.budget - plan.work) + col       # one block is in reserve_bytes
    return int(min(msize, max(1, (2 * size * col + slack) // (3 * col))))


def working_set(plan, nocc, nvir):
    '''Bytes that a Lambda step takes under the Plan ``plan`` of the
    amplitude step: its working set, reserve_bytes, and what the blocks of
    the product L T take of the memory that the plan leaves.'''
    o, v = int(nocc), int(nvir)
    size = max(j1 - j0 for j0, j1 in plan.jblocks)
    more = 0 if plan.three_m else max(0, 3 * _q_block(plan, o, v) - 2 * size - 1)
    return plan.work + reserve_bytes(o, v) + more * 16 * o * v * v


class Lambda:
    '''The Lambda equations for the amplitudes t1[i,a], t2p[(i>j),(a>b)] and
    the integrals of the amps.Residual ``res``.

    update_packed(l1, l2p) is one Jacobi step on lambda1[i,a] and lambda2
    packed as [(i>j),(a>b)].'''

    def __init__(self, res, t1, t2p):
        o, v = res.nocc, res.nvir
        n1 = o * v
        io, jo = K.tril(o)
        self.res = res
        t1 = self.t1 = np.ascontiguousarray(t1, dtype=np.complex128)
        t2p = self.t2p = np.ascontiguousarray(t2p, dtype=np.complex128)
        s = SimpleNamespace(t1=t1, t1T=np.ascontiguousarray(t1.T), t2p=t2p, R=None,
                            KW=np.empty((o, o, n1), dtype=np.complex128))
        self.t1T = s.t1T
        paired = res.kramers
        # t with the time-reversal symmetry of Kramers pairs: so are the intermediates
        self.paired = res.kramers = bool(paired and res.symmetric(t1, t2p))
        try:
            with timer('lambda.imds'):
                foo_x, self.Whh = res._imds_pairs(s)
                self.W = res._w_array()
                res._pass_rows(s, self.W, ring=1.0)
        finally:
            res.kramers = paired
        with timer('lambda.imds'):
            del s.taup
            self.fvv_a = s.fvv_a                # [a, e] = sum_mf <ma||ef> t1[m,f]
            Fov, Ft, tF, Foo, Fvv = res._fock(s, foo_x)
            self.Fov = Fov
            self.Hoo = Foo + 0.5 * Ft           # [i, j]
            self.Hvv = Fvv - 0.5 * tF           # [b, a]
            # the T1 equation <ia|Hbar|0>
            r1 = s.r1 + s.X1 + res.fov.conj() + blas.gemm(t1, Fvv.T) - blas.gemm(Foo.T, t1)
            K.t2p_matvec(t2p, Fov, r1)
            for p0, p1 in res.plan.pblocks:
                n = p1 - p0
                t2u = K.unpack_tril(t2p[p0:p1], v, 'apb').reshape(v * n, v)
                r1 += blas.gemm(res.ooov_en[:, :, p0:p1].reshape(o, v * n), t2u)
                del t2u
            self.res1 = r1
            # Hq[i, c, (j>k)] = <jk||ic>* + sum_{d>b} <ic||db> tau[(jk),(db)]
            #   + P(jk) sum_b t1[j,b] (<ic||bk> + sum <il||bd> t2[l,k,d,c])
            #   - P(jk) sum_lb <li||jb> t2[k,l,c,b]
            Hq = s.Nq
            KW = s.KW.reshape(o, o, o, v)                                   # [i, j, k, c]
            for i in range(o):
                Hq[i] += (KW[i][io, jo] - KW[i][jo, io]).T
            del KW, s.KW
            # A[(p i), (q c)] = sum_lb <li||pb> t2[q,l,c,b] for a block of q: - A into
            # the pairs (p > q), + A into the pairs (q > p)
            Or = res.ooov_r.reshape(n1, o * o).T                            # [(p i), (l b)]
            pairs = np.arange(o)
            for q0, q1 in res.plan.jblocks:
                A = blas.gemm(Or, K.ring_cols(t2p, o, v, q0, q1)).reshape(o, o, q1 - q0, v)
                for q in range(q0, q1):
                    hi, lo = pairs[q + 1:], pairs[:q]
                    Hq[:, :, hi * (hi - 1) // 2 + q] -= A[hi, :, q - q0].transpose(1, 2, 0)
                    Hq[:, :, q * (q - 1) // 2 + lo] += A[lo, :, q - q0].transpose(1, 2, 0)
                del A
            self.Hq = Hq
        # the pairs that contain an occupied spinor: packed rows and signs, [i, k]
        self._prow = np.array([r for r, _ in res.pair_rows])
        self._psgn = np.array([g for _, g in res.pair_rows])

    def close(self):
        '''Release W.'''
        self.W = None
        self.res.close()

    # -- products with the ring layout of lambda2 ------------------------------
    def _left(self, l2p, B, C, blocks, m3):
        '''C[(i a), :] = sum_(k c) lambda2[i,k,a,c] B[(k c), :] for the rows of
        the ``blocks`` of i (the ring layout of lambda2 is generated block by
        block).  With ``m3`` B is given as its three real parts and each
        product is three real GEMMs.'''
        o, v = self.res.nocc, self.res.nvir
        for i0, i1 in blocks:
            rows = slice(i0 * v, i1 * v)
            with timer('ring.layout'):
                Li = (K.ring_cols3 if m3 else K.ring_cols)(l2p, o, v, i0, i1)
            with timer('ring.gemm'):
                if m3:
                    amps._gemm3([x.T for x in Li], B, C=C[rows])
                else:
                    blas.gemm(Li.T, B, C=C[rows])
            del Li
        return C

    def _ovvv_x(self, Om, X, out):
        '''out[i,a] += sum_cb <mc||ab> X[i,c,b] from the packed rows
        Om[c,(a>b)], a few of them unpacked at a time.'''
        o, v = X.shape[:2]
        step = max(1, min(o, UNPACK_BYTES // (16 * v * v)))     # no more than X itself
        Xm = X.reshape(o, v * v)
        for c0 in range(0, v, step):
            c1 = min(v, c0 + step)
            U = K.unpack_tril(Om[c0:c1], v, 'pab')                      # [c, b, a] = <mc||ba>
            blas.gemm(Xm[:, c0 * v:c1 * v], U.reshape((c1 - c0) * v, v), C=out, alpha=-1.0,
                      beta=1.0)
            del U
        return out

    # -- one step ---------------------------------------------------------------
    def update_packed(self, l1, l2p, level_shift=0.0):
        '''(l1new, l2pnew) from lambda1[i,a] and lambda2 packed as
        [(i>j),(a>b)].  With Kramers pairs, and t and lambda with their
        symmetry, half of each expensive term is computed (below).'''
        res = self.res
        o, v = res.nocc, res.nvir
        n1, npo, npv = o * v, K.npair(o), K.npair(v)
        io, jo = K.tril(o)
        plan = res.plan
        t1, t1T, t2p, Fov, W = self.t1, self.t1T, self.t2p, self.Fov, self.W
        eo, ev = res.mo_e[:o], res.mo_e[o:] + level_shift
        l1 = np.ascontiguousarray(l1, dtype=np.complex128)
        l2p = np.ascontiguousarray(l2p, dtype=np.complex128)
        Hvv, Hoo = self.Hvv.copy(), self.Hoo.copy()
        Hvv[np.diag_indices(v)] -= ev
        Hoo[np.diag_indices(o)] -= eo
        prow, psgn = self._prow, self._psgn
        m3 = plan.three_m
        # Kramers pairs: sums over an occupied index run over its unbarred half
        # and are completed by their time-reversed image; of the ring term the
        # rows with unbarred i are computed
        kr = bool(self.paired and res.symmetric(l1, l2p))
        ho = o // 2
        size = max(j1 - j0 for j0, j1 in plan.jblocks)
        msize = _q_block(plan, o, v)
        qblocks = amps._blocks(o, msize)        # rows of lambda2 for the product with t2
        if kr:
            mblocks = amps._blocks(ho, msize)
            jblocks, iblocks = amps._blocks(ho, max(1, size // 2)), amps._blocks(ho, size)
        else:
            mblocks, jblocks, iblocks = amps._blocks(o, msize), plan.jblocks, plan.jblocks

        L2 = np.zeros((npo, npv), dtype=np.complex128)
        with timer('lambda.pairs'):
            # M[(ij),(kl)] = sum_{c>d} lambda2[(ij),(cd)] t2[(kl),(cd)]
            M = blas.gemm(l2p, t2p.T)
            # mij[i,j] = 1/2 sum_kcd lambda2[k,i,c,d] t2[k,j,c,d]
            mij = np.einsum('ik,jk,ijk->ij', psgn, psgn, M[prow[:, None, :], prow[None, :, :]])
            r1 = Fov + blas.gemm(l1, Hvv) - blas.gemm(Hoo, l1)
            # mba[b,a] = 1/2 sum_klc lambda2[k,l,c,a] t2[k,l,c,b];
            # lt[p, k, c] = sum_d lambda2[p,c,d] t1[k,d]
            mba = np.zeros((v, v), dtype=np.complex128)
            lt = np.empty((npo, o, v), dtype=np.complex128)
            for p0, p1 in plan.pblocks:
                n = p1 - p0
                l2u = K.unpack_tril(l2p[p0:p1], v, 'apb').reshape(v * n, v)     # [(c p), a]
                t2u = K.unpack_tril(t2p[p0:p1], v, 'apb').reshape(v * n, v)
                blas.gemm(t2u.T, l2u, C=mba, beta=1.0)
                del t2u
                lt[p0:p1] = blas.gemm(l2u, t1T).reshape(v, n, o).transpose(1, 2, 0)
                # sum_{(j>k), c} Hq[i,c,(jk)] lambda2[(jk),c,a]
                r1 += blas.gemm(self.Hq[:, :, p0:p1].reshape(o, v * n), l2u)
                del l2u
            # X[(ij),(kl)] = sum_{c>d} lambda2[(ij),(cd)] tau[(kl),(cd)]
            X = M
            X += blas.gemm(lt.reshape(npo * o, v), t1.T).reshape(npo, o, o)[:, jo, io]
            lvv = blas.gemm(t1T, l1)                    # [f, b] = sum_j t1[j,f] lambda1[j,b]
            loo = blas.gemm(l1, t1.T)                   # [j, n] = sum_b lambda1[j,b] t1[n,b]
            tmp1vv, tmp1oo = mba + lvv, mij + loo
            # - sum_k sum_(j>l) <jl||ka> X[(i k),(jl)]
            for k in range(o):
                r1 += psgn[k][:, None] * blas.gemm(np.ascontiguousarray(X[prow[k]]),
                                                   res.ooov_en[k].T)
            # X with the occupied one-particle term - P(ij) tmp1oo[i,k] <kj||ab>, as a
            # matrix on the pairs like the Fock term of the amplitude step
            for rows, cols, f0, f1, sign in res._g:
                X[rows, cols] += sign * tmp1oo[f0, f1]

        def Yrows(j):
            '''Y[i, j, k, c] = sum_b lambda2[i,k,b,c] t1[j,b] for one j, as [i, k, c].'''
            return -psgn[:, :, None] * lt[prow, j]

        with timer('lambda.ovvv'):
            # the terms on the rows of <ma||ef>, summed over m, with the product
            # Q[(i b), (m d)] = sum_kc lambda2[i,k,b,c] t2[k,m,c,d] for the same m
            rm = np.zeros((o, v), dtype=np.complex128)
            dm = np.ascontiguousarray((mba - lvv).T)
            lt2 = lt.reshape(npo, n1)
            for m0, m1 in mblocks:
                nm = m1 - m0
                O = res.ovvv_p[m0:m1]                                   # [m, c, (a>b)]
                with timer('ovvv.gemm'):
                    # - sum_kc <kc||ba> lt[(ij), k, c]
                    blas.gemm(lt2[:, m0 * v:m1 * v], O.reshape(nm * v, npv), C=L2, beta=1.0)
                with timer('ring.layout'):
                    Tc = (K.ring_cols3 if m3 else K.ring_cols)(t2p, o, v, m0, m1)
                Q = self._left(l2p, Tc, np.empty((n1, nm * v), dtype=np.complex128), qblocks, m3)
                del Tc
                Q4 = Q.reshape(o, v, nm, v)
                for m in range(m0, m1):
                    Om = O[m - m0]
                    with timer('ovvv.rows'):
                        # P(ij) sum_c lambda1[j,c] <ic||ab>
                        K.add_rows(L2, prow[m], psgn[m], blas.gemm(l1, Om))
                        # 1/2 sum_kbc lambda2[i,k,b,c] <ka||cb>*
                        Lk = np.conj(l2p[np.maximum(prow[m], 0)])
                        rm += psgn[m][:, None] * blas.gemm(Lk, Om.T).conj()
                        del Lk
                    with timer('ovvv.x'):
                        # sum_cb <mc||ab> X[i,c,b]: the ring part - Q, the t1 parts of
                        # the terms that are in L2 when it is contracted with t1
                        Xm = blas.gemm(np.ascontiguousarray(Yrows(m).transpose(0, 2, 1))
                                       .reshape(n1, o), t1).reshape(o, v, v)
                        Xm -= Q4[:, :, m - m0, :]
                        Xm[m] += dm
                        self._ovvv_x(Om, Xm, rm)
                        del Xm
                scratch.unmap(O)
                del Q, Q4, O
            if kr:
                with timer('lambda.kramers'):
                    from zccsdt.zccsd.kramers import image
                    K.ksym_packed(L2, res.kpo, res.sgo, res.kpv, res.sgv)
                    rm += image(rm)
            r1 += rm

        with timer('lambda.pairs'):
            # the hh ladder and the tau <kl||ab> term, each with a one-particle term
            # on the occupied side (their contractions with t1 are taken back below)
            Gl = self.Whh.copy()
            for rows, cols, f0, f1, sign in res._g:
                Gl[rows, cols] += sign * Hoo[f0, f1]
            blas.gemm(Gl, l2p, C=L2, beta=1.0)
            blas.gemm(X, res.oovv_p, C=L2, beta=1.0)
            del Gl, X

        with timer('lambda.ladder'):
            saved = res.kramers
            res.kramers = kr
            try:
                res.ladder(l2p, L2, left=True)
            finally:
                res.kramers = saved

        with timer('lambda.t1'):
            # what the lambda1 equation contracts with t1 is in L2, with four terms
            # too many: the t1 parts of two are in the contraction with <ma||ef>
            # above and in the first line here, those of the two one-particle
            # terms follow
            K.t2p_matvec(L2, t1, r1)
            r1 += blas.gemm(l1, self.fvv_a)
            lt1 = K.t2p_matvec(l2p, t1, np.zeros((o, v), dtype=np.complex128))
            r1 += blas.gemm(Hoo, lt1) + blas.gemm(tmp1oo, Fov - res.fov)
            K.t2p_matvec(l2p, self.res1 + blas.gemm(Hoo.T, t1), r1)
            K.t2p_matvec(res.oovv_p, blas.gemm(tmp1oo.T, t1) - blas.gemm(t1, mba.T)
                         - blas.gemm(mij.T, t1), r1)

        with timer('lambda.ring'):
            l1v = np.ascontiguousarray(l1.reshape(n1, 1))
            for j0, j1 in jblocks:
                nj = j1 - j0
                # the rows (j b) of W: of the block, and of the partners of its j
                parts = [slice(j0 * v, j1 * v)]
                if kr:
                    parts.append(slice((j0 + ho) * v, (j1 + ho) * v))
                # P(ij) P(ab) sum_kc lambda2[k,i,c,a] <jc|Hbar|bk>
                r4 = np.empty((n1, len(parts) * nj * v), dtype=np.complex128)
                if m3:
                    with timer('ring.layout'):
                        if len(parts) == 1:
                            WJ = W[parts[0]]
                        else:
                            WJ = np.empty((2 * nj * v, n1), dtype=np.complex128)
                            WJ[:nj * v], WJ[nj * v:] = W[parts[0]], W[parts[1]]
                        W3 = [x.T for x in K.split3(WJ)]
                        del WJ
                    self._left(l2p, W3, r4, iblocks, True)
                    del W3
                else:
                    for k, c in enumerate(parts):
                        self._left(l2p, W[c].T, r4[:, k * nj * v:(k + 1) * nj * v], iblocks, False)
                with timer('ring.pack'):
                    if kr:
                        K.kfill_ring(r4, o, v, 0, ho, nj)
                        K.asym_pack_cols(r4, o, v, j0, j1, 1.0, L2)
                        K.asym_pack_cols(r4, o, v, j0 + ho, j1 + ho, 1.0, L2, offset=nj * v)
                    else:
                        K.asym_pack_cols(r4, o, v, j0, j1, 1.0, L2)
                del r4
                for c in parts:
                    Wr, jf = W[c], c.start // v
                    # sum_me lambda1[m,e] <ie|Hbar|am>
                    r1[jf:jf + nj] += blas.gemm(Wr, l1v).reshape(nj, v)
                    # - sum_j sum_kc <jc|Hbar|ak> Y[i,j,k,c]
                    for j in range(nj):
                        blas.gemm(Yrows(jf + j).reshape(o, n1), Wr[j * v:(j + 1) * v].T, C=r1,
                                  alpha=-1.0, beta=1.0)
                    scratch.unmap(Wr)

        with timer('lambda.t2'):
            # the t1 parts of <jc|Hbar|ak> that the lambda2 t1 terms do not take:
            # Yc[i,j,n,k] = sum_c Y[i,j,k,c] t1[n,c], Ykc[i,j,n,f] = sum_k Yc[i,j,n,k] t1[k,f]
            ltt = blas.gemm(lt.reshape(npo * o, v), t1.T).reshape(npo, o, o)    # [p, j, n]
            Yc = np.ascontiguousarray(-(psgn[:, :, None, None] * ltt[prow]).transpose(0, 2, 3, 1))
            del ltt
            r1 += blas.gemm(Yc.reshape(o, o ** 3), res.ooov.reshape(o ** 3, v))
            Ykc = blas.gemm(Yc.reshape(o ** 3, o), t1).reshape(o, o, o, v)
            Zp = Ykc[:, io, jo] - Ykc[:, jo, io]                        # [i, (j>n), f]
            del Yc, Ykc
            # virtual side, in blocks of pairs
            for p0, p1 in plan.pblocks:
                n = p1 - p0
                l2u = K.unpack_tril(l2p[p0:p1], v, 'apb').reshape(v * n, v)     # [(a p), c]
                Yv = blas.gemm(l2u, Hvv)                # sum_c lambda2[p,a,c] Hvv[c,b]
                del l2u
                Nb = res.ooov_en[:, :, p0:p1].reshape(o, v * n)
                blas.gemm(Nb.T, l1, C=Yv, beta=1.0)     # sum_k <p||kb> lambda1[k,a] as [(b p), a]
                vu = K.unpack_tril(res.oovv_p[p0:p1], v, 'apb')                 # [b, p, c]
                blas.gemm(vu.reshape(v * n, v), tmp1vv, C=Yv, alpha=-1.0, beta=1.0)
                K.asym_pack2(Yv.reshape(v, n, v), 1.0, L2[p0:p1], 'apb')
                # - sum_{(j>n), f} <jn||af> Zp[i,(jn),f]
                blas.gemm(np.ascontiguousarray(Zp[:, p0:p1]).reshape(o, n * v),
                          vu.reshape(v, n * v).T, C=r1, alpha=-1.0, beta=1.0)
                del Yv, Nb, vu
            del Zp
            L2 += res.oovv_p
            K.outer_pp(l1, Fov, 1.0, L2)

        with timer('lambda.t1'):
            r1 -= np.einsum('jika,kj->ia', res.ooov, mij)
            r1 -= blas.gemm(mij, Fov) + blas.gemm(Fov, mba)
            K.div_packed(L2, eo, ev)
            r1 /= eo[:, None] - ev
        return r1, L2

    def update(self, l1, l2, level_shift=0.0):
        '''(l1new, l2new) on lambda2[i,j,a,b]: update_packed between a pack
        and an unpack.'''
        o, v = self.res.nocc, self.res.nvir
        l2p = K.pack_t2(np.ascontiguousarray(l2, dtype=np.complex128))
        l1new, l2pnew = self.update_packed(l1, l2p, level_shift)
        return l1new, K.unpack_t2(l2pnew, o, v)


def kernel(mycc, eris, t1=None, t2=None, l1=None, l2=None, max_cycle=50, tol=1e-8,
           verbose=None, sources=None):
    '''The iterations of pyscf.cc.ccsd_lambda.kernel on packed amplitudes:
    (converged, l1, l2) with lambda2[i,j,a,b] unpacked at the end.  The first
    guess is conj(t); the DIIS vector is [lambda1, packed lambda2].
    ``sources``: (s1[i,a], s2[i,j,a,b]) that are added to lambda1 and lambda2
    after every step - the (T) terms of zccsd.lam_t.'''
    log = logger.new_logger(mycc, verbose)
    settings = (mycc.max_memory, mycc.diis_space, True if mycc.incore_complete else None,
                getattr(mycc, 'three_m', None))
    if getattr(eris, 'residual', None) is None:
        eris.residual = amps.Residual(eris, plan=False)
    res = eris.residual
    o, v = res.nocc, res.nvir
    t1 = np.ascontiguousarray(mycc.t1 if t1 is None else t1, dtype=np.complex128)
    # t2 of the CCSD object is packed and released while lambda is solved for
    # (the unpacked array is four times the packed one), and unpacked again
    # at the end
    own = t2 is None
    t2p = K.pack_t2(np.ascontiguousarray(mycc.t2 if own else t2, dtype=np.complex128))
    if own:
        t2 = mycc.t2 = None
    try:
        return _iterate(mycc, res, t1, t2p, l1, l2, max_cycle, tol, log, sources, settings)
    finally:
        if own:
            mycc.t2 = K.unpack_t2(t2p, o, v)


def _iterate(mycc, res, t1, t2p, l1, l2, max_cycle, tol, log, sources, settings):
    '''The iterations of kernel() for the packed t2.'''
    o, v = res.nocc, res.nvir
    try:
        res.replan(*settings, reserve=reserve_bytes(o, v))
    except MemoryError as err:
        raise MemoryError('%s (the Lambda equations: the plan of the amplitude step, with %s '
                          'set aside for their own arrays)'
                          % (err, amps._size(reserve_bytes(o, v)))) from None
    log.info('Lambda equations: %s', res.describe())
    cput1 = cput0 = (logger.process_clock(), logger.perf_counter())
    lam = Lambda(res, t1, t2p)
    cput1 = log.timer('Lambda intermediates', *cput1)
    l1 = t1.conj() if l1 is None else np.ascontiguousarray(l1, dtype=np.complex128)
    l2p = (K.copy(t2p, conj=True) if l2 is None
           else K.pack_t2(np.ascontiguousarray(l2, dtype=np.complex128)))
    if sources is not None:
        s1 = np.ascontiguousarray(sources[0], dtype=np.complex128)
        s2p = K.pack_t2(np.ascontiguousarray(sources[1], dtype=np.complex128))
    adiis = None
    if mycc.diis:
        adiis = DIIS(space=mycc.diis_space, filename=mycc.diis_file, incore=res.plan.diis_incore)
    conv = False
    try:
        for istep in range(max_cycle):
            l1new, l2pnew = lam.update_packed(l1, l2p, mycc.level_shift)
            if sources is not None:
                l1new += s1
                l2pnew += s2p
            with timer('lambda.diis'):
                norm = np.sqrt(K.diff_norm2(l1new, l1) + K.diff_norm2(l2pnew, l2p))
                l1, l2p = l1new, l2pnew
                l1new = l2pnew = None
                if adiis is not None and istep >= mycc.diis_start_cycle:
                    x1, x2 = adiis.update([l1, l2p])
                    l1, l2p = x1.reshape(o, v), x2.reshape(l2p.shape)
            log.info('cycle = %d  norm(lambda1,lambda2) = %.6g', istep + 1, norm)
            cput1 = log.timer('Lambda iter', *cput1)
            if norm < tol:
                conv = True
                break
    finally:
        if adiis is not None:
            adiis.close()
        lam.close()
    log.timer('Lambda', *cput0)
    return conv, l1, K.unpack_t2(np.ascontiguousarray(l2p), o, v)
