'''The orbital gradient between frozen and active spinors, from the integrals
with one frozen index.

The Lagrangian of a CCSD with frozen spinors has integrals over the active
spinors only (and the Fock matrix, which has all occupied spinors in its
density).  A rotation between a frozen spinor f and an active one x puts f
into one index of every integral: with the two-particle density of the
active space in the convention of pyscf.cc.gccsd_rdm, D[x,q,r,s] (without
its one-particle terms), the gradient of zccsd.zvector is

    G[f, x] = X[f, x] + (n_x - n_f) V[f, x],
    X[f, x] = sum_qrs (f q|r s) D[x, q, r, s],

with the occupation numbers n and V the Fock matrix (Coulomb minus exchange)
of the correlation part of the one-particle density: zvector.kernel adds it.
X is what this module makes, for frozen occupied and frozen virtual spinors
alike.

The integrals (f q|r s) of one frozen spinor with three active ones are
(o + v)^3 numbers; they come from the integral transformation slab by slab
(zccsd.ao2mo.slabs with the frozen spinors as the first index) and are
contracted as they come.  The density is never formed.  Its blocks with at
most two virtual indices are arrays of the active space (the largest of them
of the size of t2: the product m of lambda2 and t2 in the ring layout, and
the density of <ij||ab> packed like t2); the one with three virtual indices
is a sum of products of these with t1 and lambda1, and the one with four is
lambda2 times tau.  Both meet the slab through

    A[b, (a>c)] = (f a|b c) - (f c|b a),   A tau^T,   A conj(lambda2)^T

(v^3 o^2 / 4 multiply-adds per frozen spinor, the largest term) and through
slabs whose virtual index is contracted with t1 first.

CCSD(T).  E(T) is that of the active spinors; its densities (zccsd.triples:
g1 of <bc||ek>, g2 of <ma||jk>, g3 of <bc||jk>) meet the same slabs: with
Gnh of zvector.gradient_nh, a frozen index is a bra index or a ket index of
the integrals, and G[f, x] gets (B[f, x] + conj(K[f, x])) / 2 with the two
contractions.

Cost: the integral transformation of the frozen spinors - what a CCSD of
that many more occupied spinors would add to its own -, and for each of them
a pass over arrays of the size of t2.  The code that this replaces
(zvector.gradient_frozen) treated the frozen spinors as occupied ones with
zero amplitudes and did a step of the amplitude equations, one of the Lambda
equations and the orbital gradient for all of them: o_all^3 v^3.

Kramers pairs.  With amplitudes that are symmetric under time reversal and
frozen spinors that are closed under it level by level, the frozen spinors
are rotated into pairs (zccsd.kramers.pair), X is made for the unbarred ones,
the rows of their partners are the time-reversed image, and the result is
rotated back.

Timers (zccsdt.timing): 'ogf.setup' (the arrays of the active space),
'ogf.small', 'ogf.vvv' (A and its products), 'ogf.pairs', 'ogf.ring' (the
pass over m), 'ogf.t'.
'''

import numpy as np
from pyscf.lib import logger

from zccsdt import blas, scratch
from zccsdt.timing import timer
from zccsdt.zccsd import amps, orbgrad
from zccsdt.zccsd import kernels as K

PARTS = ('oooo', 'ooov', 'oovv', 'ovvo', 'ovvv', 'vvvv', 'g1', 'g2', 'g3')
# The weights of the packed densities of (T) (zccsd.triples.Densities) against
# the full arrays of lam_t.densities: g1[k,e,(b>c)], g2[m,a,(j>k)], g3[(j>k),(b>c)]
T_WEIGHTS = (2.0, 2.0, 4.0)
G1_BYTES = 2 << 30      # rows of the density of <bc||ek>, and of m, in one product


class Active:
    '''The arrays of the active space that the slabs are contracted with, for
    t1[i,a], t2 packed [(i>j), (a>b)] and lambda in the same layouts; the
    integrals and the memory plan are those of the amps.Residual ``res``.
    In the names of pyscf.cc.gccsd_rdm._gamma2_intermediates:

      tau[(i>j), (a>b)]   t2 + t1 t1 - t1 t1 (antisymmetric);
      m[(k b), (i a)]     miajb[k,b,i,a] = sum_lc lambda2[k,l,b,c] t2[l,i,c,a]
                          (o^2 v^2: in memory or a mapped scratch array);
      gc[(i>j), (a>b)]    the conjugate of dovov[i,a,j,b];
      Ap[(j>k), i, a]     dooov[j,i,k,a];
      D1[i, k, j, l]      doooo[k,i,l,j];
      u[i, a, j, k]       what dovvo has besides m:
                          govvo[i,b,a,j] = m[(i a),(j b)] + sum_k u[i,a,j,k] t1[k,b];
      lt[(k>l), i, b]     sum_d lambda2[k,l,b,d] t1[i,d];
      tv[b, c]            1/4 sum_k miajb[k,b,k,c];
      l2c                 the conjugate of the packed lambda2.'''

    def __init__(self, res, t1, t2p, l1, l2p):
        o, v = self.o, self.v = res.nocc, res.nvir
        n1, npo = o * v, K.npair(o)
        io, jo = K.tril(o)
        plan = res.plan
        self.res, self.t1, self.l1, self.t2p, self.l2p = res, t1, l1, t2p, l2p
        prow = self.prow = np.array([r for r, _ in res.pair_rows])
        psgn = self.psgn = np.array([g for _, g in res.pair_rows])
        self.pblocks = amps._blocks(npo, max(1, orbgrad.UNPACK_BYTES // (16 * v * v)))
        t1T, l1T = np.ascontiguousarray(t1.T), np.ascontiguousarray(l1.T)
        kr = bool(res.kramers and res.symmetric(t1, t2p) and res.symmetric(l1, l2p))
        self.kramers = kr
        ho = o // 2 if kr else o
        size = max(j1 - j0 for j0, j1 in plan.jblocks)
        if kr:      # blocks of unbarred columns (each with its partners) and of unbarred rows
            cblocks, rblocks = amps._blocks(ho, max(1, size // 2)), amps._blocks(ho, size)
        else:
            cblocks, rblocks = plan.jblocks, plan.jblocks

        def ring(j0, j1, t1t1=False):
            '''The columns (j b) of t2 in the ring layout for the block of j
            (with Kramers pairs: followed by those of the partners of its j);
            with ``t1t1``: of t2[n,j,f,b] - 2 t1[n,b] t1[j,f].'''
            parts = [(j0, j1)] + ([(j0 + ho, j1 + ho)] if kr else [])
            nc = (j1 - j0) * v
            Tc = np.empty((n1, len(parts) * nc), dtype=np.complex128)
            for k, (a0, a1) in enumerate(parts):
                Tk = K.ring_cols(t2p, o, v, a0, a1).reshape(o, v, a1 - a0, v)
                if t1t1:
                    Tk -= 2.0 * t1[:, None, None, :] * t1[a0:a1].T[None, :, :, None]
                Tc[:, k * nc:(k + 1) * nc] = Tk.reshape(n1, nc)
                del Tk
            return Tc

        with timer('ogf.pairs'):
            tau = self.tau = K.outer_pp(t1, t1, 0.5, t2p.copy())
            GP = blas.gemm(l2p, tau.T, alpha=0.25)              # goooo[(kl), (ij)]
            lt = self.lt = np.empty((npo, o, v), dtype=np.complex128)
            ltau = np.empty((npo, o, v), dtype=np.complex128)   # sum_b tau[(jk),a,b] l1[i,b]
            for p0, p1 in self.pblocks:
                n = p1 - p0
                xu = K.unpack_tril(l2p[p0:p1], v, 'apb').reshape(v * n, v)
                lt[p0:p1] = blas.gemm(xu, t1T).reshape(v, n, o).transpose(1, 2, 0)
                xu = K.unpack_tril(tau[p0:p1], v, 'apb').reshape(v * n, v)
                ltau[p0:p1] = blas.gemm(xu, l1T).reshape(v, n, o).transpose(1, 2, 0)
                del xu

        # ---- m = L T in the ring layout, by blocks of its columns
        m = self.m = res._w_array()
        with timer('ogf.ring'):
            for m0, m1 in cblocks:
                Tc = ring(m0, m1)
                nc = (m1 - m0) * v
                Q = np.empty((n1, Tc.shape[1]), dtype=np.complex128)
                for i0, i1 in rblocks:
                    Li = K.ring_cols(l2p, o, v, i0, i1)
                    blas.gemm(Li.T, Tc, C=Q[i0 * v:i1 * v])
                    del Li
                if kr:
                    K.kfill_ring(Q, o, v, 0, ho, m1 - m0)
                    m[:, (m0 + ho) * v:(m1 + ho) * v] = Q[:, nc:]
                m[:, m0 * v:m1 * v] = Q[:, :nc]
                del Q, Tc

        with timer('ogf.small'):
            too = np.empty((o, o), dtype=np.complex128)         # sum_d miajb[l,d,j,d]
            tvv = np.zeros((v, v), dtype=np.complex128)         # sum_l miajb[l,d,l,b]
            Z = np.empty((o, o, o, v), dtype=np.complex128)     # sum_c m[(i c),(j a)] t1[k,c]
            for i in range(o):
                mi = np.asarray(m[i * v:(i + 1) * v])
                blk = mi.reshape(v, o, v)
                too[i] = np.einsum('djd->j', blk)
                tvv += blk[:, i, :]
                Z[i] = blas.gemm(t1, mi).reshape(o, o, v)
                del mi, blk
            self.tv = 0.25 * tvv

        # ---- gc[(i>j), (a>b)]: the antisymmetrized goovv of PySCF before its conjugation
        with timer('ogf.pairs'):
            l2c = self.l2c = K.copy(l2p, conj=True)
            gc = self.gc = l2c + tau
            gc *= 0.5
            K.outer_pp(K.t2p_matvec(t2p, l1, np.zeros((o, v), dtype=np.complex128)), t1, 0.5, gc)
            tvv1 = blas.gemm(l1T, t1)                           # sum_k l1[k,c] t1[k,b]
            Foo = 0.5 * blas.gemm(l1, t1T) + 0.25 * too
            for p0, p1 in self.pblocks:
                n = p1 - p0
                xu = K.unpack_tril(t2p[p0:p1], v, 'apb').reshape(v * n, v)      # [(a p), c]
                K.asym_pack2(blas.gemm(xu, tvv1).reshape(v, n, v), -0.5, gc[p0:p1], 'apb')
                xu = K.unpack_tril(tau[p0:p1], v, 'apb').reshape(v * n, v)
                K.asym_pack2(blas.gemm(xu, tvv).reshape(v, n, v), -0.25, gc[p0:p1], 'apb')
                del xu
            Gm = np.zeros((npo, npo), dtype=np.complex128)
            for rows, cols, f0, f1, sign in res._g:
                Gm[rows, cols] += sign * Foo[f0, f1]
            blas.gemm(Gm.T, tau, C=gc, beta=1.0)
            del Gm
            blas.gemm(GP.T, tau, C=gc, alpha=2.0, beta=1.0)
        with timer('ogf.ring'):
            # 1/4 P(ij) P(ab) sum_ld m[(l d),(i a)] (t2[l,j,d,b] - 2 t1[l,b] t1[j,d])
            for n0, n1_ in cblocks:
                Tc = ring(n0, n1_, t1t1=True)
                R = np.empty((n1, Tc.shape[1]), dtype=np.complex128)
                blas.gemm(m[:, :ho * v].T, Tc, C=R[:ho * v])
                if kr:
                    K.kfill_ring(R, o, v, 0, ho, n1_ - n0)
                    K.asym_pack_cols(R, o, v, n0 + ho, n1_ + ho, 0.25, gc, offset=(n1_ - n0) * v)
                K.asym_pack_cols(R, o, v, n0, n1_, 0.25, gc)
                del R, Tc

        with timer('ogf.small'):
            # ---- dooov[j,i,k,a] = Ap[(j>k), i, a]
            Ac = 0.5 * ltau
            del ltau
            for i in range(o):
                Gi = np.ascontiguousarray((psgn[i][:, None] * GP[prow[i]]).T)   # [(jk), l]
                Ac[:, i, :] += 2.0 * blas.gemm(Gi, t1)
            Ac -= 0.25 * (too[:, io].T[:, :, None] * t1[jo][:, None, :]
                          - too[:, jo].T[:, :, None] * t1[io][:, None, :])
            Ac += 0.5 * (Z[:, jo, io, :] - Z[:, io, jo, :]).transpose(1, 0, 2)
            del Z
            Ap = self.Ap = Ac.conj() + 0.5 * lt
            del Ac
            self.A2T = np.ascontiguousarray(Ap.transpose(0, 2, 1)).reshape(npo * v, o)
            Af = psgn[:, :, None, None] * Ap[prow]                              # [i, l, k, a]
            self.AfT = np.ascontiguousarray(Af.conj().reshape(o, o * o * v).T)
            del Af
            # ---- doooo
            g4 = K.unpack_t2(GP, o, o)
            d = 2.0 * (g4.transpose(0, 2, 1, 3) + g4.conj().transpose(2, 0, 3, 1))     # [k,i,l,j]
            self.D1T = np.ascontiguousarray(d.transpose(1, 0, 3, 2).reshape(o, o ** 3).T)
            del g4, d, GP
            # ---- u[i,a,j,k] = l1[i,a] d(j,k) - sum_c lambda2[i,k,a,c] t1[j,c]
            u = -(psgn[:, :, None, None] * lt[prow]).transpose(0, 3, 2, 1)
            k = np.arange(o)
            u[:, :, k, k] += l1[:, :, None]
            u = np.ascontiguousarray(u)
            self.uc_ajk = np.ascontiguousarray(u.conj().reshape(o, v * o * o).T)    # [(a j k), i]
            self.u_kjb = np.ascontiguousarray(u.transpose(3, 0, 1, 2)).reshape(o * o * v, o)
            self.u_ijk = np.ascontiguousarray(u.transpose(0, 2, 3, 1)).reshape(o ** 3, v)
            self.uc_ijb = np.ascontiguousarray(u.conj().transpose(2, 0, 1, 3)).reshape(o * o * v, o)
            del u
            # [(b (kl)), i]
            self.ltB = np.ascontiguousarray(lt.transpose(2, 0, 1)).reshape(v * npo, o)
            self.ltc = lt.conj().reshape(npo, o * v)

    def gather(self, M):
        '''out[f, i] = sum_j F[i,j,f,j] from M[(i>j), f, j] = F[(ij), f, j],
        antisymmetric in the pair.'''
        o = self.o
        x = M[self.prow, :, np.arange(o)[None, :]]              # [i, j, f]
        return np.einsum('ij,ijf->fi', self.psgn, x)

    def pairs_matvec(self, Pp, Q):
        '''out[f, a] = sum_(kl) sum_c P[(kl), a, c] Q[f, (kl), c] for the
        packed Pp[(k>l), (a>c)].'''
        out = np.zeros((Q.shape[0], self.v), dtype=np.complex128)
        return K.tril_matvec(Pp, Q, out)

    def g2_pa(self, g2):
        '''The density g2[m, a, (j>k)] of (T) as [(m (jk)), a].'''
        o, v = self.o, self.v
        return np.ascontiguousarray(np.asarray(g2).transpose(0, 2, 1)).reshape(o * K.npair(o), v)

    def g2_mka(self, g2):
        '''The density g2[m, a, (i>k)] of (T) as [(m k a), i], the pair
        unpacked.'''
        o, v = self.o, self.v
        x = self.psgn[None, None] * np.asarray(g2)[:, :, self.prow]             # [m, a, i, k]
        return np.ascontiguousarray(x.transpose(0, 3, 1, 2)).reshape(o * o * v, o)

    def close(self):
        self.m = None
        self.res.close()


def contract(act, S, X, dens=None, work=None):
    '''The slab S[f, q, r, s] = (f q|r s) of a batch of frozen spinors f with
    the active ones [occupied, virtual] in its other three places: X[part][f,
    x] += its contraction with the densities of the Active ``act`` and, for
    CCSD(T), with those of zccsd.triples (``dens``).  What it allocates next
    to the slab is counted by batch_bytes; ``work``: a dict that keeps the
    largest of these arrays from one batch to the next (pages that have been
    touched once).'''
    o, v, t1, l1 = act.o, act.v, act.t1, act.l1
    nb, n = S.shape[:2]
    n1, npo, npv = o * v, K.npair(o), K.npair(v)
    io, jo = K.tril(o)
    oc, vi = slice(0, o), slice(o, n)
    t1c, l1c = t1.conj(), l1.conj()
    t1T = np.ascontiguousarray(t1.T)
    E = np.einsum

    def add(part, x, occupied, fac=1.0):
        X[part][:, oc if occupied else vi] += fac * x

    with timer('ogf.small'):
        Sooo = np.ascontiguousarray(S[:, oc, oc, oc])
        # Q46[f,a,j,b] = (f a|b j) - (f j|b a); R10[f,i,j,b] = (f i|j b) - (f b|j i)
        Q46 = np.ascontiguousarray(S[:, vi, vi, oc].transpose(0, 1, 3, 2))
        Q46 -= S[:, oc, vi, vi].transpose(0, 3, 1, 2)
        R10 = np.ascontiguousarray(S[:, oc, oc, vi])
        R10 -= S[:, vi, oc, oc].transpose(0, 3, 2, 1)
        Y = S[:, oc, vi, oc].transpose(0, 1, 3, 2)              # [f, i, j, b] = (f i|b j)
        R9 = np.ascontiguousarray(Y[:, io, jo] - Y[:, jo, io])  # [f, (i>j), b]
        T12 = np.ascontiguousarray((Sooo.transpose(0, 3, 1, 2)
                                    - Sooo.transpose(0, 1, 3, 2))[:, io, jo])  # [f, (x>y), k]
        del Y
        # Q5[f, j, (a>b)] = (f a|j b) - (f b|j a)
        Q5 = np.empty((nb, o, npv), dtype=np.complex128)
        P16 = np.empty((nb, o, o, v), dtype=np.complex128)      # 1/2 sum_c Q5[i,(c x)] t1[k,c]
        q16 = np.zeros((nb, v), dtype=np.complex128)            # sum_ib Q5[i,(c b)] t1[i,b]
        for k in range(nb):
            K.asym_pack_view(S[k, vi, oc, vi], Q5[k])
            K.tril_matvec(Q5[k], t1[None], q16[k:k + 1])
            U = K.unpack_tril(Q5[k], v, 'pab')                  # [i, c, x]
            P16[k] = blas.gemm(U.reshape(n1, v), t1T,
                               alpha=-0.5).reshape(o, v, o).transpose(2, 0, 1)
            del U

        # ---- doooo and dooov
        add('oooo', blas.gemm(Sooo.reshape(nb, o ** 3), act.D1T), True)
        del Sooo
        add('ooov', blas.gemm(R9.reshape(nb, npo * v), act.A2T), True)
        add('ooov', blas.gemm(np.ascontiguousarray(R10.transpose(0, 2, 1, 3)).reshape(nb, o * n1),
                              act.AfT), True)
        add('ooov', blas.gemm(T12.reshape(nb, npo * o), act.Ap.reshape(npo * o, v)), False)

        # ---- what dovvo has besides m
        Q46tc = blas.gemm(Q46.reshape(nb * n1, v), t1c.T).reshape(nb, v, o, o)     # [f, a, j, k]
        add('ovvo', blas.gemm(Q46tc.reshape(nb, v * o * o), act.uc_ajk), True, 0.5)
        x = np.empty((nb, o, n1), dtype=np.complex128)                              # [f, k, (j b)]
        for k in range(nb):
            blas.gemm(t1, Q46[k].reshape(v, n1), C=x[k])
        add('ovvo', blas.gemm(x.reshape(nb, o * n1), act.u_kjb), True, 0.5)
        x = blas.gemm(R10.reshape(nb * o * o, v), t1T).reshape(nb, o ** 3)          # [f, (i j k)]
        add('ovvo', blas.gemm(x, act.u_ijk), False, 0.5)
        x = blas.gemm(R10.reshape(nb, o * n1), act.uc_ijb)                          # [f, k]
        add('ovvo', blas.gemm(x, t1c), False, 0.5)

        # ---- the density of three virtual indices, its part tv x t1
        tvc = act.tv.conj()
        x = np.matmul(Q46, tvc[:, :, None])[..., 0].sum(axis=1)                     # [f, i]
        add('ovvv', -blas.gemm(x, t1c), False)
        x = blas.gemm(Q46.reshape(nb * v, n1), t1c.reshape(n1, 1)).reshape(nb, v)
        add('ovvv', blas.gemm(x, tvc), False)
        add('ovvv', blas.gemm(q16, act.tv.T), False)
        del x, q16

    # ---- A[b, (a>c)] = (f a|b c) - (f c|b a) and its products
    with timer('ogf.vvv'):
        work = {} if work is None else work
        if work.get('A3p') is None or work['A3p'].shape[0] < nb:
            work['A3p'] = np.empty((nb, v, npv), dtype=np.complex128)
            work['A3t'] = np.empty((nb, o, v, v), dtype=np.complex128)
        A3p, A3t = work['A3p'][:nb], work['A3t'][:nb]
        # A3t[f, k, b, a] = sum_c A[b,a,c] t1[k,c]; FA[f, a] = sum_bc A[b,a,c] tv[b,c]
        FA = np.zeros((nb, v), dtype=np.complex128)
        for k in range(nb):
            Sk = S[k]
            K.asym_pack_view(Sk[o:, o:, o:], A3p[k])
            K.tril_matvec(A3p[k], act.tv[None], FA[k:k + 1])
            # the virtual index of t1 in the last place of the slab and in the
            # first: products with whole rows of the slab, nothing unpacked
            x = blas.gemm(t1, Sk.reshape(n * n, n)[:, o:].T).reshape(o, n, n)   # [k, q, r]
            y = blas.gemm(t1, Sk[o:].reshape(v, n * n)).reshape(o, n, n)        # [k, r, s]
            np.subtract(x[:, o:, o:].transpose(0, 2, 1), y[:, o:, o:], out=A3t[k])
            del x, y, Sk
        A2 = A3p.reshape(nb * v, npv)
        LT = blas.gemm(A2, act.tau.T).reshape(nb, v, npo)       # sum_(ac) A[b,(ac)] tau[(kl),(ac)]
        LL = blas.gemm(A2, act.l2c.T).reshape(nb, v, npo)       # the same with conj(lambda2)
    with timer('ogf.small'):
        add('ovvv', blas.gemm(FA, t1T), True)
        add('ovvv', blas.gemm(LT.reshape(nb, v * npo), act.ltB), True, 0.5)
        x = LT[:, :, act.prow] * act.psgn                       # [f, b, i, l]
        add('ovvv', E('fbil,lb->fi', x, l1), True, 0.5)
        x = LL[:, :, act.prow] * act.psgn
        add('ovvv', E('fbil,lb->fi', x, t1c), True, 0.5)
        # Qt[f,(kl),c], Ql[f,(kl),c]: what tau (conjugated) and lambda2 of the densities
        # of three and of four virtual indices are contracted with
        x = np.ascontiguousarray(Q46.transpose(2, 1, 0, 3)).reshape(n1, nb * v)     # [(i b), (f c)]
        Qt = np.empty((2 * nb, npo, v), dtype=np.complex128)
        Ql = np.empty((2 * nb, npo, v), dtype=np.complex128)
        Qt[:nb] = blas.gemm(act.ltc, x, alpha=-0.5).reshape(npo, nb, v).transpose(1, 0, 2)
        x = np.ascontiguousarray(Q46.transpose(0, 2, 3, 1)).reshape(nb * n1, v)     # [(f k c), b]
        W = blas.gemm(x, l1c.T).reshape(nb, o, v, o).transpose(0, 1, 3, 2)          # [f, k, l, c]
        Qt[:nb] -= 0.5 * (W[:, io, jo] - W[:, jo, io])
        W = blas.gemm(x, t1T).reshape(nb, o, v, o).transpose(0, 1, 3, 2)
        Ql[:nb] = -0.5 * (W[:, io, jo] - W[:, jo, io])
        del W, x
        Qt[nb:] = 0.5 * LL.transpose(0, 2, 1)
        Ql[nb:] = 0.5 * LT.transpose(0, 2, 1)
        del LT, LL
    with timer('ogf.pairs'):
        # tau, lambda2 and the density of <ij||ab> with the rows of the slab
        np.conj(Qt, out=Qt)
        out = act.pairs_matvec(act.tau, Qt).conj()
        add('ovvv', out[:nb], False)
        add('vvvv', out[nb:], False)
        out = act.pairs_matvec(act.l2p, Ql)
        add('ovvv', out[:nb], False)
        add('vvvv', out[nb:], False)
        del Qt, Ql, out
        np.conj(R9, out=R9)
        add('oovv', act.pairs_matvec(act.gc, R9).conj(), False)
        np.conj(R9, out=R9)
        Q5T = Q5.reshape(nb * o, npv).T
        add('oovv', act.gather(blas.gemm(act.gc, Q5T).reshape(npo, nb, o)), True)
        x = blas.gemm(act.tau, Q5T).reshape(npo, nb, o)
        add('ovvv', blas.gemm(np.ascontiguousarray(x.transpose(1, 0, 2)).reshape(nb, npo * o),
                              act.lt.reshape(npo * o, v)), False, -0.5)
        add('ovvv', blas.gemm(act.gather(x), l1), False, 0.5)
        x = blas.gemm(act.l2c, Q5T).reshape(npo, nb, o)
        add('ovvv', blas.gemm(act.gather(x), t1c), False, 0.5)
        del x

    # ---- the pass over m[(k b), (i a)], by its rows of one k
    with timer('ogf.ring'):
        K.copy(Q46, Q46, conj=True)
        Qf = Q46.reshape(nb, v * n1)                            # conj(Q46)[f, (b i c)]
        ob = np.empty((nb, o), dtype=np.complex128)
        og = np.zeros((2, nb, o), dtype=np.complex128)
        oa = np.zeros((2, nb, v), dtype=np.complex128)
        od = np.zeros((2, nb, v), dtype=np.complex128)

        def rows(x):
            return np.ascontiguousarray(x).reshape(nb, -1)
        # (a few occupied k at a time: the rows of a mapped m are let go again)
        kstep = max(1, min(o, G1_BYTES // (16 * v * n1)))
        for k0 in range(0, o, kstep):
            k1 = min(o, k0 + kstep)
            mb = np.asarray(act.m[k0 * v:k1 * v])
            blas.gemm(Qf, mb.reshape(k1 - k0, v * n1).T, C=ob[:, k0:k1])
            scratch.unmap(mb)
            del mb
        for k in range(o):
            mk = np.asarray(act.m[k * v:(k + 1) * v])           # [b, (i a)]
            blas.gemm(rows(R10[:, k]), mk.T, C=od[0], beta=1.0)
            blas.gemm(rows(P16[:, k]), mk.T, C=od[1], beta=1.0)
            mr = mk.reshape(n1, v)                              # [(b i), a]
            blas.gemm(rows(R10[:, :, k, :].transpose(0, 2, 1)).conj(), mr, C=oa[0], beta=1.0)
            blas.gemm(rows(Q46tc[:, :, :, k]).conj(), mr, C=oa[1], beta=1.0)
            mp = np.ascontiguousarray(mk.reshape(v, o, v).transpose(1, 0, 2)).reshape(o, v * v)
            blas.gemm(rows(Q46[:, :, k, :].transpose(0, 2, 1)).conj(), mp.T, C=og[0], beta=1.0)
            blas.gemm(rows(A3t[:, k]), mp.T, C=og[1], beta=1.0)
            scratch.unmap(mk)
            del mk, mr, mp
        K.copy(Q46, Q46, conj=True)
        ob = ob.conj()
        add('ovvo', ob, True, 0.5)
        add('ovvv', blas.gemm(ob, t1c), False, 0.5)
        add('ovvo', og[0], True, 0.5)
        add('ovvv', og[1], True, 0.5)
        add('ovvo', oa[0].conj(), False, 0.5)
        add('ovvv', oa[1].conj(), False, -0.5)
        add('ovvo', od[0], False, 0.5)
        add('ovvv', od[1], False)
        del A3t, P16, Q46tc, Qf

    if dens is None:
        return
    # ---- the densities of (T): G[f, x] = (B[f, x] + conj(K[f, x])) / 2
    w1, w2, w3 = T_WEIGHTS
    with timer('ogf.t'):
        g1, g2, g3 = dens.g1, dens.g2, dens.g3
        K.copy(Q5, Q5, conj=True)
        K.copy(A3p, A3p, conj=True)
        Af = A3p.reshape(nb, v * npv)
        Bv = np.zeros((nb, v), dtype=np.complex128)
        Kv = np.zeros((nb, v), dtype=np.complex128)
        Ko = np.empty((nb, o), dtype=np.complex128)
        # (a few occupied k at a time: the rows of a mapped g1 are let go again)
        kstep = max(1, min(o, G1_BYTES // (16 * v * npv)))
        for k0 in range(0, o, kstep):
            k1 = min(o, k0 + kstep)
            Gb = np.asarray(g1[k0:k1])                          # [k, e, (b>c)]
            blas.gemm(Af, Gb.reshape(k1 - k0, v * npv).T, C=Ko[:, k0:k1])
            for k in range(k0, k1):
                blas.gemm(Q5[:, k], Gb[k - k0].T, C=Kv, beta=1.0)
                K.tril_matvec(Gb[k - k0], Q46[:, :, k, :], Bv)
            scratch.unmap(Gb)
            del Gb
        add('g1', Bv + Kv.conj(), False, 1.0 / w1)
        add('g1', Ko.conj(), True, -1.0 / w1)
        del A3p, Af
        # g3[(j>k), (b>c)]
        add('g3', act.pairs_matvec(g3, R9), False, 2.0 / w3)
        add('g3', act.gather(blas.gemm(g3, Q5.reshape(nb * o, npv).T).reshape(npo, nb, o)).conj(),
            True, 2.0 / w3)
        # g2[m, a, (j>k)]
        add('g2', blas.gemm(np.ascontiguousarray(T12.transpose(0, 2, 1)).reshape(nb, o * npo),
                            act.g2_pa(g2)), False, 1.0 / w2)
        Bo = blas.gemm(np.ascontiguousarray(R9.transpose(0, 2, 1)).reshape(nb, v * npo),
                       np.asarray(g2).reshape(o, v * npo).T)
        K.copy(R10, R10, conj=True)
        Bo += blas.gemm(R10.reshape(nb, o * n1), act.g2_mka(g2)).conj()
        add('g2', Bo, True, 1.0 / w2)


def own_bytes(nocc, nvir):
    '''(kept, most): bytes of the arrays of Active besides its three packed
    ones and m - those that stay for the contractions, and the most that is
    there while they are made.'''
    o, v = int(nocc), int(nvir)
    npo = K.npair(o)
    kept = 16 * (5 * npo * o * v + 5 * o ** 3 * v + o ** 4 + v * v)
    most = kept + 16 * (3 * npo * o * v + 3 * o ** 3 * v + 3 * o ** 4 + 2 * npo * npo)
    return kept, most + 2 * orbgrad.UNPACK_BYTES


def reserve_bytes(nocc, nvir):
    '''Bytes to set aside when the amplitude step is planned for the arrays
    of Active: what own_bytes has more than the intermediates of a step
    (amps.step_arrays).  The three packed arrays of the plan are tau, the
    density of <ij||ab> and the conjugate of lambda2; its W is m.'''
    return max(0, own_bytes(nocc, nvir)[1] - amps.step_arrays(nocc, nvir))


def batch_bytes(nocc, nvir, nb, with_t=False):
    '''Bytes of the arrays of contract() for a batch of ``nb`` frozen
    spinors, next to their slab.'''
    o, v = int(nocc), int(nvir)
    npo, npv = K.npair(o), K.npair(v)
    per = 16 * (v * npv + o * npv + 5 * o * v * v + 8 * o * o * v + 8 * npo * v + 2 * o ** 3
                + 4 * v * v)
    fixed = 16 * (2 * o * v * v + 2 * o * o * v + 2 * o * (o + v) ** 2
                  + (4 * o ** 3 * v if with_t else 0))
    return int(nb) * per + fixed


def slab_plan(mol, left, nocc, nvir, nfirst, with_t=False):
    '''The ao2mo.Plan of the slabs (f q|r s) of ``nfirst`` frozen spinors
    within ``left`` bytes (None: no limit known): the most of them at a time
    that fit with the arrays of contract() for them.  MemoryError if one does
    not.'''
    from zccsdt.zccsd import ao2mo
    nq, nfirst = int(nocc) + int(nvir), max(1, int(nfirst))

    def plan_of(nk):
        '''The plan with slabs of ``nk`` frozen spinors; MemoryError where
        the passes for them do not fit next to the arrays of contract().'''
        own = batch_bytes(nocc, nvir, nk, with_t)
        plan = ao2mo.plan_slabs(mol, nfirst, nq, None if left is None else max(0, left - own),
                                sort_arrays=0)
        if plan.nket < nk:
            raise MemoryError
        plan.work += own - 16 * nq ** 3 * (plan.nket - nk)
        plan.nket = nk
        return plan

    try:
        best = plan_of(1)
    except MemoryError as err:
        raise MemoryError('%s (the integrals with a frozen index of the orbital gradient, '
                          'next to %s for what one frozen spinor is contracted with)'
                          % (err, amps._size(batch_bytes(nocc, nvir, 1, with_t)))) from None
    # A larger slab leaves less for the passes: the largest that fits, by
    # bisection up to what a slab holds at most.  (Halving from all frozen
    # spinors down to what the rest of the budget allowed ended at one or two
    # where ten fit: 143 slabs for AuH with 30 GB left.)
    lo, hi = 1, ao2mo.plan_slabs(mol, nfirst, nq, None, sort_arrays=0).nket
    while lo < hi:
        mid = (lo + hi + 1) // 2
        try:
            best, lo = plan_of(mid), mid
        except MemoryError:
            hi = mid - 1
    return best


def gradient(mycc, eris, t1, t2p, l1, l2p, Cf, dens=None, parts=None, slabs=None):
    '''X[f, x] (see the module) for the frozen spinors Cf[(spin, AO), f] and
    the active spinors x of ``eris`` ([occupied, virtual]), for t1[i,a], t2
    packed [(i>j), (a>b)] and lambda in the same layouts; ``dens``: the
    Densities of zccsd.triples for CCSD(T).  The memory plan is that of
    eris.residual; ``slabs``: the ao2mo.Plan of the slabs (slab_plan).
    Returns a dict: X by the block of the density (PARTS).'''
    from zccsdt.zccsd import ao2mo
    res = eris.residual
    o, v = res.nocc, res.nvir
    n, nf = o + v, Cf.shape[1]
    nao = eris.nao_sph
    X = {k: np.zeros((nf, n), dtype=np.complex128) for k in PARTS}
    with timer('ogf.setup'):
        act = Active(res, t1, t2p, l1, l2p)
    Cd = np.hstack((Cf, eris.mo_ao_spin))
    S, work = None, {}
    for f0, f1, S in ao2mo.slabs(eris.mol, Cd[:nao], Cd[nao:], (0, nf), (nf, nf + n), slabs):
        contract(act, S, {k: x[f0:f1] for k, x in X.items()}, dens, work)
    del S, work
    act.close()
    return X


def kernel(mycc, eris, t1, t2, l1, l2, Cf, dens=None, mo_energy=None, verbose=None,
           parts=None):
    '''X[f, x] for the frozen spinors Cf[(spin, AO), f] - occupied or virtual
    - and the amplitudes and lambda given (t2 and lambda2 as [i,j,a,b] or
    packed), within max_memory of ``mycc``: the plan of the amplitude step
    for the arrays of the active space (reserve_bytes set aside), and what
    it leaves for the slabs of the integral transformation.  ``mo_energy``:
    the orbital energies of the frozen spinors, for their Kramers pairs (see
    the module); without them every frozen spinor is done.  ``parts``: a dict
    that gets X by the block of the density (PARTS).'''
    log = logger.new_logger(mycc, verbose)
    cput0 = (logger.process_clock(), logger.perf_counter())
    if getattr(eris, 'residual', None) is None:
        eris.residual = amps.Residual(eris, plan=False)
    res = eris.residual
    o, v = res.nocc, res.nvir
    n1, nf = o * v, Cf.shape[1]

    def packed(x):
        x = np.ascontiguousarray(x, dtype=np.complex128)
        return x if x.ndim == 2 else K.pack_t2(x)
    t1 = np.ascontiguousarray(t1, dtype=np.complex128)
    l1 = np.ascontiguousarray(l1, dtype=np.complex128)
    t2p, l2p = packed(t2), packed(l2)
    del t2, l2
    kept, reserve = own_bytes(o, v)[0], reserve_bytes(o, v)
    try:
        res.replan(mycc.max_memory, 0, False, False, reserve=reserve)
    except MemoryError as err:
        raise MemoryError('%s (the orbital gradient between frozen and active spinors: the '
                          'plan of the amplitude step, with %s set aside for its own arrays)'
                          % (err, amps._size(reserve))) from None
    plan = res.plan
    # Kramers pairs: the unbarred half of the frozen spinors
    U = None
    if (res.kramers and mo_energy is not None and nf % 2 == 0
            and res.symmetric(t1, t2p) and res.symmetric(l1, l2p)):
        from zccsdt.zccsd import kramers
        s2 = np.kron(np.eye(2), eris.mol.intor('int1e_ovlp'))
        order = np.argsort(mo_energy, kind='stable')
        pair = kramers.pair(Cf[:, order], s2, np.asarray(mo_energy)[order])
        if pair is not None:
            U = Cf.conj().T @ s2 @ pair[0]
            Cf = pair[0][:, :nf // 2]

    def slabs_within(mapped):
        left = None
        if plan.budget is not None:
            P = 16 * K.npair(o) * K.npair(v)
            left = plan.budget + reserve - 3 * P - kept - (0 if mapped else 16 * n1 * n1)
        return slab_plan(eris.mol, left, o, v, Cf.shape[1], dens is not None)
    try:
        try:
            slabs = slabs_within(plan.w_mapped)
        except MemoryError:
            if plan.w_mapped:
                raise
            plan.w_mapped = True        # m in a scratch file leaves room for the slabs
            slabs = slabs_within(True)
        log.info('Orbital gradient between %d frozen and the active spinors: %s; the integrals '
                 'of %d frozen spinors at a time (%s)%s', nf, res.describe(), slabs.nket,
                 amps._size(slabs.work), '' if U is None else
                 '; Kramers pairs: half of the frozen spinors')
        X = gradient(mycc, eris, t1, t2p, l1, l2p, Cf, dens, slabs=slabs)
    finally:
        res.close()
    if U is not None:
        from zccsdt.zccsd import kramers
        for k, x in X.items():
            full = np.zeros((nf, o + v), dtype=np.complex128)
            full[:nf // 2] = x
            full[:, :o] += kramers.image(full[:, :o])
            full[:, o:] += kramers.image(full[:, o:])
            X[k] = U @ full
    if parts is not None:
        parts.update(X)
    log.timer('Orbital gradient between frozen and active spinors', *cput0)
    return sum(X.values())
