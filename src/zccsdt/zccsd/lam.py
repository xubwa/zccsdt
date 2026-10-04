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

Timers (zccsdt.timing): 'lambda.imds' (the intermediates, once); per
iteration 'lambda.pairs', 'lambda.ladder', 'lambda.ovvv', 'lambda.ring',
'lambda.t2', 'lambda.t1', 'lambda.diis'.
'''

from types import SimpleNamespace

import numpy as np
from pyscf.lib import logger

from zccsdt import blas, scratch
from zccsdt.cc.diis import DIIS
from zccsdt.timing import timer
from zccsdt.zccsd import kernels as K

UNPACK_BYTES = 64 << 20     # rows of <ma||ef> unpacked at a time


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
        res.kramers = bool(paired and res.symmetric(t1, t2p))
        try:
            with timer('lambda.imds'):
                foo_x, self.Whh = res._imds_pairs(s)
                self.W = res._w_array()
                res._pass_rows(s, self.W, ring=1.0)
        finally:
            res.kramers = paired
        with timer('lambda.imds'):
            self.taup = s.taup
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
            Hq += (KW[:, io, jo] - KW[:, jo, io]).transpose(0, 2, 1)
            del KW, s.KW
            A = np.empty((o * o, n1), dtype=np.complex128)                  # [(j i), (k c)]
            Or = np.ascontiguousarray(res.ooov_r.reshape(n1, o * o).T)      # [(j i), (l b)]
            for k0, k1 in res.plan.jblocks:
                blas.gemm(Or, K.ring_cols(t2p, o, v, k0, k1), C=A[:, k0 * v:k1 * v])
            A = A.reshape(o, o, o, v)                                       # [j, i, k, c]
            Hq += (A[jo, :, io] - A[io, :, jo]).transpose(1, 2, 0)
            del A, Or
            self.Hq = Hq
        # the pairs that contain an occupied spinor: packed rows and signs, [i, k]
        self._prow = np.array([r for r, _ in res.pair_rows])
        self._psgn = np.array([g for _, g in res.pair_rows])

    def close(self):
        '''Release W.'''
        self.W = None
        self.res.close()

    # -- products with the ring layout of lambda2 ------------------------------
    def _left(self, l2p, B, C):
        '''C[(i a), :] = sum_(k c) lambda2[i,k,a,c] B[(k c), :], the rows of
        the ring layout of lambda2 in blocks of i.'''
        o, v = self.res.nocc, self.res.nvir
        for i0, i1 in self.res.plan.jblocks:
            with timer('ring.layout'):
                Li = K.ring_cols(l2p, o, v, i0, i1)
            with timer('ring.gemm'):
                blas.gemm(Li.T, B, C=C[i0 * v:i1 * v])
            del Li
        return C

    def _ovvv_x(self, Om, X, out):
        '''out[i,a] += sum_cb <mc||ab> X[i,c,b] from the packed rows
        Om[c,(a>b)], a few of them unpacked at a time.'''
        o, v = X.shape[:2]
        step = max(1, UNPACK_BYTES // (16 * v * v))
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
        [(i>j),(a>b)].'''
        res = self.res
        o, v = res.nocc, res.nvir
        n1, npo, npv = o * v, K.npair(o), K.npair(v)
        io, jo = K.tril(o)
        plan = res.plan
        t1, t1T, t2p, taup, Fov, W = self.t1, self.t1T, self.t2p, self.taup, self.Fov, self.W
        eo, ev = res.mo_e[:o], res.mo_e[o:] + level_shift
        l1 = np.ascontiguousarray(l1, dtype=np.complex128)
        l2p = np.ascontiguousarray(l2p, dtype=np.complex128)
        Hvv, Hoo = self.Hvv.copy(), self.Hoo.copy()
        Hvv[np.diag_indices(v)] -= ev
        Hoo[np.diag_indices(o)] -= eo
        prow, psgn = self._prow, self._psgn

        with timer('lambda.pairs'):
            # X[(ij),(kl)] = sum_{c>d} lambda2[(ij),(cd)] tau[(kl),(cd)], M the same with t2
            X = blas.gemm(l2p, taup.T)
            M = blas.gemm(l2p, t2p.T)
            # mij[i,j] = 1/2 sum_kcd lambda2[k,i,c,d] t2[k,j,c,d]
            mij = np.einsum('ik,jk,ijk->ij', psgn, psgn, M[prow[:, None, :], prow[None, :, :]])
            del M
            # mba[b,a] = 1/2 sum_klc lambda2[k,l,c,a] t2[k,l,c,b];
            # lt[p, k, c] = sum_d lambda2[p,c,d] t1[k,d]
            mba = np.zeros((v, v), dtype=np.complex128)
            lt = np.empty((npo, o, v), dtype=np.complex128)
            for p0, p1 in plan.pblocks:
                n = p1 - p0
                l2u = K.unpack_tril(l2p[p0:p1], v, 'apb').reshape(v * n, v)     # [(c p), a]
                t2u = K.unpack_tril(t2p[p0:p1], v, 'apb').reshape(v * n, v)
                blas.gemm(t2u.T, l2u, C=mba, beta=1.0)
                lt[p0:p1] = blas.gemm(l2u, t1T).reshape(v, n, o).transpose(1, 2, 0)
                del l2u, t2u
            lvv = blas.gemm(t1T, l1)                    # [f, b] = sum_j t1[j,f] lambda1[j,b]
            loo = blas.gemm(l1, t1.T)                   # [j, n] = sum_b lambda1[j,b] t1[n,b]
            tmp1vv, tmp1oo = mba + lvv, mij + loo
            # the terms that the lambda1 equation contracts with t1: the hh ladder,
            # the tau <kl||ab> term
            L2 = blas.gemm(self.Whh, l2p)
            blas.gemm(X, res.oovv_p, C=L2, beta=1.0)

        with timer('lambda.ladder'):
            saved = res.kramers
            res.kramers = bool(res.paired and res.symmetric(l1, l2p))
            try:
                res.ladder(l2p, L2, left=True)
            finally:
                res.kramers = saved

        with timer('lambda.t1'):
            r1 = Fov + blas.gemm(l1, Hvv) - blas.gemm(Hoo, l1)
            K.t2p_matvec(L2, t1, r1)

        with timer('lambda.ovvv'):
            for m0, m1 in plan.jblocks:
                nm = m1 - m0
                O = res.ovvv_p[m0:m1]                                   # [m, c, (a>b)]
                # - sum_kc <kc||ba> lt[(ij), k, c]
                blas.gemm(lt.reshape(npo, n1)[:, m0 * v:m1 * v], O.reshape(nm * v, npv), C=L2,
                          beta=1.0)
                # Q[(i b), (m d)] = sum_kc lambda2[i,k,b,c] t2[k,m,c,d]
                with timer('ring.layout'):
                    Tc = K.ring_cols(t2p, o, v, m0, m1)
                Q = self._left(l2p, Tc, np.empty((n1, nm * v), dtype=np.complex128))
                del Tc
                Q4 = Q.reshape(o, v, nm, v)
                for m in range(m0, m1):
                    Om = O[m - m0]
                    # P(ij) sum_c lambda1[j,c] <ic||ab>
                    K.add_rows(L2, prow[m], psgn[m], blas.gemm(l1, Om))
                    # 1/2 sum_kbc lambda2[i,k,b,c] <ka||cb>*
                    Lk = np.conj(l2p[np.maximum(prow[m], 0)])
                    r1 += psgn[m][:, None] * blas.gemm(Lk, Om.T).conj()
                    del Lk
                    # - sum_bd <mb||ad> Q[(i b),(m d)] + sum_bc <mc||ab> mba[b,c] for i = m
                    Xm = -Q4[:, :, m - m0, :]
                    Xm[m] += mba.T
                    self._ovvv_x(Om, np.ascontiguousarray(Xm), r1)
                    del Xm
                scratch.unmap(O)
                del Q, Q4, O

        with timer('lambda.ring'):
            # Y[i, j, k, c] = sum_b lambda2[i,k,b,c] t1[j,b]
            Y = -(psgn[:, :, None, None] * lt[prow]).transpose(0, 2, 1, 3)
            Y = np.ascontiguousarray(Y)
            l1v = np.ascontiguousarray(l1.reshape(n1, 1))
            for j0, j1 in plan.jblocks:
                nj = j1 - j0
                rows = slice(j0 * v, j1 * v)
                Wr = W[rows]                                            # [(j b), (k c)]
                # P(ij) P(ab) sum_kc lambda2[k,i,c,a] <jc|Hbar|bk>
                r4 = self._left(l2p, Wr.T, np.empty((n1, nj * v), dtype=np.complex128))
                with timer('ring.pack'):
                    K.asym_pack_cols(r4, o, v, j0, j1, 1.0, L2)
                del r4
                # sum_me lambda1[m,e] <ie|Hbar|am>
                r1[j0:j1] += blas.gemm(Wr, l1v).reshape(nj, v)
                # - sum_j sum_kc <jc|Hbar|ak> Y[i,j,k,c]
                for j in range(j0, j1):
                    blas.gemm(Y[:, j].reshape(o, n1), Wr[(j - j0) * v:(j - j0 + 1) * v].T, C=r1,
                              alpha=-1.0, beta=1.0)
                scratch.unmap(Wr)

        with timer('lambda.t2'):
            # the t1 parts of <jc|Hbar|ak> that the lambda2 t1 terms do not take:
            # Yc[i,j,k,n] = sum_c Y[i,j,k,c] t1[n,c], Ykc[i,j,n,f] = sum_k Yc[i,j,k,n] t1[k,f]
            Yc = blas.gemm(Y.reshape(o ** 3, v), t1.T).reshape(o, o, o, o)
            del Y
            r1 += blas.gemm(np.ascontiguousarray(Yc.transpose(0, 1, 3, 2)).reshape(o, o ** 3),
                            res.ooov.reshape(o ** 3, v))
            Ykc = blas.gemm(np.ascontiguousarray(Yc.transpose(0, 1, 3, 2)).reshape(o ** 3, o),
                            t1).reshape(o, o, o, v)
            Zp = Ykc[:, io, jo] - Ykc[:, jo, io]                        # [i, (j>n), f]
            del Yc, Ykc
            # virtual side, in blocks of pairs
            for p0, p1 in plan.pblocks:
                n = p1 - p0
                l2u = K.unpack_tril(l2p[p0:p1], v, 'apb').reshape(v * n, v)     # [(a p), c]
                Yv = blas.gemm(l2u, Hvv)                # sum_c lambda2[p,a,c] Hvv[c,b]
                Nb = res.ooov_en[:, :, p0:p1].reshape(o, v * n)
                blas.gemm(Nb.T, l1, C=Yv, beta=1.0)     # sum_k <p||kb> lambda1[k,a] as [(b p), a]
                vu = K.unpack_tril(res.oovv_p[p0:p1], v, 'apb')                 # [b, p, c]
                blas.gemm(vu.reshape(v * n, v), tmp1vv, C=Yv, alpha=-1.0, beta=1.0)
                K.asym_pack2(Yv.reshape(v, n, v), 1.0, L2[p0:p1], 'apb')
                # sum_{(j>k), c} Hq[i,c,(jk)] lambda2[(jk),c,a]
                r1 += blas.gemm(self.Hq[:, :, p0:p1].reshape(o, v * n), l2u)
                # - sum_{(j>n), f} <jn||af> Zp[i,(jn),f]
                blas.gemm(np.ascontiguousarray(Zp[:, p0:p1]).reshape(o, n * v),
                          vu.reshape(v, n * v).T, C=r1, alpha=-1.0, beta=1.0)
                del l2u, Yv, Nb, vu
            del Zp
            # occupied side: the one-particle terms as matrices on the pairs
            Gl = np.zeros((npo, npo), dtype=np.complex128)
            Gv = np.zeros((npo, npo), dtype=np.complex128)
            for rows, cols, f0, f1, sign in res._g:
                Gl[rows, cols] += sign * Hoo[f0, f1]
                Gv[rows, cols] += sign * tmp1oo[f0, f1]
            blas.gemm(Gl, l2p, C=L2, beta=1.0)
            blas.gemm(Gv, res.oovv_p, C=L2, beta=1.0)
            del Gl, Gv
            L2 += res.oovv_p
            K.outer_pp(l1, Fov, 1.0, L2)

        with timer('lambda.t1'):
            # - sum_k sum_(j>l) <jl||ka> X[(i k),(jl)]
            for k in range(o):
                r1 += psgn[k][:, None] * blas.gemm(np.ascontiguousarray(X[prow[k]]),
                                                   res.ooov_en[k].T)
            del X
            K.t2p_matvec(l2p, np.ascontiguousarray(self.res1), r1)
            K.t2p_matvec(res.oovv_p, -blas.gemm(t1, mba.T) - blas.gemm(mij.T, t1), r1)
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
           verbose=None):
    '''The iterations of pyscf.cc.ccsd_lambda.kernel on packed amplitudes:
    (converged, l1, l2) with lambda2[i,j,a,b] unpacked at the end.  The first
    guess is conj(t); the DIIS vector is [lambda1, packed lambda2].'''
    from zccsdt.zccsd import amps
    log = logger.new_logger(mycc, verbose)
    settings = (mycc.max_memory, mycc.diis_space, True if mycc.incore_complete else None,
                getattr(mycc, 'three_m', None))
    if getattr(eris, 'residual', None) is None:
        eris.residual = amps.Residual(eris, *settings[:3], three_m=settings[3])
    else:
        eris.residual.replan(*settings)
    res = eris.residual
    o, v = res.nocc, res.nvir
    t1 = np.ascontiguousarray(mycc.t1 if t1 is None else t1, dtype=np.complex128)
    t2 = mycc.t2 if t2 is None else t2
    t2p = K.pack_t2(np.ascontiguousarray(t2, dtype=np.complex128))
    cput1 = cput0 = (logger.process_clock(), logger.perf_counter())
    lam = Lambda(res, t1, t2p)
    cput1 = log.timer('Lambda intermediates', *cput1)
    l1 = t1.conj() if l1 is None else np.ascontiguousarray(l1, dtype=np.complex128)
    l2p = (K.copy(t2p, conj=True) if l2 is None
           else K.pack_t2(np.ascontiguousarray(l2, dtype=np.complex128)))
    adiis = None
    if mycc.diis:
        adiis = DIIS(space=mycc.diis_space, filename=mycc.diis_file, incore=res.plan.diis_incore)
    conv = False
    try:
        for istep in range(max_cycle):
            l1new, l2pnew = lam.update_packed(l1, l2p, mycc.level_shift)
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
