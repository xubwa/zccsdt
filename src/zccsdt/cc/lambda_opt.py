'''The Lambda equations of spin-orbital CCSDT on packed Lambda_3.

Same equations as ``zccsdt.cc.lambda_ref`` (the tables of
``zccsdt.cc.lambda_terms``: T1-dressed integrals, nothing conjugated),
organized like the amplitude equations of ``zccsdt.cc.ccsdt_opt``: l1 and l2
are dense, l3 and its residual live on the unique block [i<j<k, a<b<c] like
T3, and no array of the size o^3 v^3 or v^4 is formed.

What depends on t only is made once (make_intermediates); it is what one
amplitude iteration makes, plus two contractions of T3:

  Fvv, Foo, Wov, Woooo   the effective one-body operators, the ring operator
                         and the hh operator of the T3 equation
  Wpp                    its pp ladder operator ~<ab||de> + t2 <mn||de> with
                         Fvv folded in, packed (v^4 / 4)
  W_vvvo, W_vooo         the matrix elements of Hbar <ab||ei>, <ej||mn> with
                         their T3 parts (W_vvvo is the drive intermediate of
                         the T3 equation; W_vooo has f t2 besides)
  Xq, X2                 sum_{f<g} <ij||fg> t3[I,e,fg] and
                         A(ef) sum_{g<h} <ei||gh> t3[I,f,gh]  (I: a triple)

A Lambda iteration (residuals) passes over the packed l3 three times, as an
amplitude iteration passes over T3, with the transposed operators:

  pass 1   half-expanded batches L[P, d, m, Q] of l3, and of T3 next to them:
           two stacked GEMMs give the four contractions of l3 with t2 and
           with the W's (Y1, Y2 and the parts of the Lambda_2 residual); one
           GEMM per occupied pair gives the density sum L[P] T[P]^T over
           (m<n, e<f) with one occupied and one virtual index free on either
           side (o^4 v^4 / 4, as the ring)
  pass 2   ring with the transposed ring operator, the terms of l1 and l2
           (two outer products and two drive products) and the two drive
           products of Y1, Y2 with <mn||ef>, reduced once
  ladder   pp ladder with the transposed operator (Fvv folded in), and the
           contractions of l3 with Xq and X2 on the same gathered batches
  hh       the transposed occupied operator on the packed array

and the densities of l3 with T3 over all virtual indices come from one
product of the packed arrays, L3p T3p^T (C(o,3)^2 C(v,3)).

Which term of the tables is where: docs/lambda.md, section 9.

Everything that touches the packed arrays goes through an engine: DenseL3
for the layout above, and lambda_blk.BlockL3 for l3 in the blocks of the
symmetry labels of the integrals (kernel(..., blocks=True),
make_intermediates(..., blocks=True)), with the same methods on the layout
of the block engine of the T3 equation; the operands are the same dense
arrays, pairs and triples in the order of their groups, and the pp operator
its diagonal pair blocks.  The memory is planned (plan_memory,
plan_memory_blocks); T3, l3 and the residual are in ordinary memory.
'''

import ctypes
import os
import time
from dataclasses import dataclass

import numpy as np
from pyscf import lib

from zccsdt.blas import gemm
from zccsdt.cc import ccsdt_opt, lambda_blk, solver, t3engine, t3kern, t3pack
from zccsdt.cc.ccsdt_opt import (add_one_body_pairs, bra_dressed_vvvo_pp, dress_block,
                                 dress_fock, dressed_vvvv_pp, wvvvo_packed)
from zccsdt.cc.diis import DIIS
from zccsdt.timing import timer

einsum = lib.einsum

# the groups of terms of the Lambda_3 residual (residuals(..., terms=))
GROUPS = ('outer', 'drive2', 'drive3', 'ring', 'pp', 'hh')



def _load_lib():
    path = os.environ.get('ZCCSDT_LAMBDA_LIB') or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lib', 'libzc_lambda.so')
    try:
        return ctypes.CDLL(path)
    except OSError:
        return None


_lib = _load_lib()
USE_C = _lib is not None        # csrc/lambda_t3.c; set False to force the numpy forms


def pair_expand(Y, vm):
    '''out[y, e, x, a] = Y[y, x, (a<e)] for a < e, -Y[y, x, (e<a)] for a > e:
    an array packed over a virtual pair on both orders of the pair.'''
    ny, nx, nq = Y.shape
    n = vm.n
    out = np.empty((ny, n, nx, n), dtype=Y.dtype)
    if USE_C and vm.spins is None and Y.flags.c_contiguous \
            and Y.dtype in (np.float64, np.complex128):
        getattr(_lib, 'zc_lam_pair_expand_' + t3pack._sfx(Y.dtype))(
            t3pack._p(Y), t3pack._p(out), t3pack._I64(ny), t3pack._I64(nx), t3pack._I64(n))
        return out
    if USE_C and hasattr(_lib, 'zc_lam_pair_expand_tab_d') and Y.flags.c_contiguous \
            and Y.dtype in (np.float64, np.complex128):     # (pairs in the order of their groups)
        getattr(_lib, 'zc_lam_pair_expand_tab_' + t3pack._sfx(Y.dtype))(
            t3pack._p(Y), t3pack._p(out), t3pack._I64(ny), t3pack._I64(nx), t3pack._I64(n),
            t3pack._I64(nq), t3pack._p(vm.pidx_c))
        return out
    md, ne = vm.pairs[:, 0], vm.pairs[:, 1]
    out[...] = 0
    out[:, ne, :, md] = Y.transpose(2, 0, 1)
    out[:, md, :, ne] = -Y.transpose(2, 0, 1)
    return out


def pair_matvec(Y, W, vm):
    '''K[x, a] = sum_{Q,h} Y[h, x, Q] w[Q, (a,h)] with the packed columns of
    W (npair, npair) on both orders of the pair: w[Q, (a,h)] = W[Q, (a<h)]
    for a < h and -W[Q, (h<a)] for a > h.  W may be the list of its diagonal
    pair blocks (vm.blocks; nothing between the blocks).'''
    n, nx, nq = Y.shape
    out = np.zeros((nx, n), dtype=Y.dtype)
    blocked = isinstance(W, (list, tuple))
    if USE_C and not blocked and vm.spins is None and Y.flags.c_contiguous \
            and W.flags.c_contiguous and Y.dtype == W.dtype \
            and Y.dtype in (np.float64, np.complex128):
        getattr(_lib, 'zc_lam_pair_matvec_' + t3pack._sfx(Y.dtype))(
            t3pack._p(Y), t3pack._p(W), t3pack._p(out), t3pack._I64(nx), t3pack._I64(n))
        return out
    if not blocked:
        W = [W]
        assert len(vm.blocks) == 1
    tab = USE_C and hasattr(_lib, 'zc_lam_pair_matvec_tab_d') and Y.flags.c_contiguous \
        and Y.dtype in (np.float64, np.complex128)
    r = np.arange(n)
    for (lo, hi), Wb in zip(vm.blocks, W):
        if hi == lo:
            continue
        if tab:
            Wb = np.ascontiguousarray(Wb, dtype=Y.dtype)
            getattr(_lib, 'zc_lam_pair_matvec_tab_' + t3pack._sfx(Y.dtype))(
                t3pack._p(Y), t3pack._p(Wb), t3pack._p(out), t3pack._I64(nx), t3pack._I64(n),
                t3pack._I64(nq), t3pack._I64(lo), t3pack._I64(hi - lo),
                t3pack._p(np.ascontiguousarray(vm.pa_c[lo:hi])),
                t3pack._p(np.ascontiguousarray(vm.pb_c[lo:hi])))
            continue
        for h in range(n):
            a = np.delete(r, h)
            cols = vm.pidx[np.minimum(a, h), np.maximum(a, h)]
            a = a[(cols >= lo) & (cols < hi)]
            cols = vm.pidx[np.minimum(a, h), np.maximum(a, h)] - lo
            if len(a):
                out[:, a] += np.where(a < h, 1.0, -1.0) * gemm(
                    np.ascontiguousarray(Y[h, :, lo:hi]), np.asarray(Wb[:, cols], dtype=Y.dtype))
    return out


# ------------------------------------------------------------------- engine

class DenseL3:
    '''The steps of a Lambda iteration that touch the packed arrays, for the
    dense layout [I, A] (I: i<j<k, A: a<b<c, lexicographic) of
    t3engine.DenseT3.  Buffers: two half-expanded batches of occupied pairs
    in a pass (pair_batch_size), a gathered batch of occupied triples and its
    product in the ladder.'''

    def __init__(self, om, vm, dtype, mem_mb):
        self.om, self.vm, self.dtype = om, vm, np.dtype(dtype)
        self.t3 = t3engine.DenseT3(om, vm, dtype, mem_mb)
        self.size = self.t3.size
        self.set_memory(mem_mb)

    def set_memory(self, mem_mb):
        self.mem_mb = mem_mb
        self.t3.set_memory(mem_mb)
        self.nb = min(self.t3.nb, max(self.om.npair, 1))

    def view(self, x):
        return self.t3.view(x)

    def buffer_bytes(self, mem_mb=None):
        '''Bytes of the batch buffers of a pass.'''
        return self.t3.buffer_bytes(self.mem_mb if mem_mb is None else mem_mb)

    def _half(self, X, p0, p1, out):
        '''out[P - p0, d, m, Q] = x[m, P_0, P_1, d, Q_0, Q_1] for the
        occupied pairs p0 <= P < p1 (t3pack.gather_half into a buffer).'''
        om, vm = self.om, self.vm
        nb = p1 - p0
        if t3pack.USE_C:
            rows = np.ascontiguousarray(om.tri[:, p0:p1].T)
            sgo = np.ascontiguousarray(om.sgn[:, p0:p1].T)
            I64, p_ = t3pack._I64, t3pack._p
            getattr(t3pack._lib, 'zc_gather_half_' + t3pack._sfx(X.dtype))(
                p_(X), I64(X.shape[1]), p_(rows), p_(sgo), p_(vm.tri_c), p_(vm.sgn_c),
                I64(nb), I64(om.n), I64(vm.n), I64(vm.npair), p_(out))
        else:
            out[:nb] = t3pack.gather_half(X, om, vm, np.arange(p0, p1))
        return out[:nb]

    def pass1(self, L3, T3, GPd, GmQ, l2pp, Xacc):
        '''With L[P, d, m, Q] and T[P, d, m, Q], the half-expanded l3 and T3:

          Xacc[x, (i, Q)]  = sum_{P,d} GPd[x, P, d] L[P, d, i, Q]
          Yall[P, d, y]    = sum_{m,Q} L[P, d, m, Q] GmQ[(m, Q), y]
          Dr[(a,i), (g,o)] = sum_{P,Q} L[P, a, i, Q] T[P, g, o, Q]
          D1[(g, o)]       = sum_{P,Q} T[P, g, o, Q] l2pp[P, Q]

        GPd (nx, npo, nv), GmQ (no nvp, ny), l2pp (npo, nvp); Xacc
        (nx, no nvp) is written; returns (Yall, Dr, D1).'''
        om, vm, dtype = self.om, self.vm, self.dtype
        no, nv, npo, nvp = om.n, vm.n, om.npair, vm.npair
        dm, nx, ny = nv * no, GPd.shape[0], GmQ.shape[1]
        L3, T3 = self.view(L3), self.view(T3)
        Yall = np.empty((npo, nv, ny), dtype=dtype)
        Dr = np.zeros((dm, dm), dtype=dtype)
        D1 = np.zeros(dm, dtype=dtype)
        Xacc[...] = 0
        if not self.size:           # no triples
            Yall[...] = 0
            return Yall, Dr, D1
        nb = self.nb
        bufL = np.empty((nb, nv, no, nvp), dtype=dtype)
        bufT = np.empty((nb, nv, no, nvp), dtype=dtype)
        GmQ = np.ascontiguousarray(GmQ, dtype=dtype)
        for p0 in range(0, npo, nb):
            p1 = min(p0 + nb, npo)
            n = p1 - p0
            with timer('gather'):
                L = self._half(L3, p0, p1, bufL)
                T = self._half(T3, p0, p1, bufT)
            with timer('gemm'):
                Lf = L.reshape(n * nv, no * nvp)
                gemm(np.ascontiguousarray(GPd[:, p0:p1], dtype=dtype).reshape(nx, n * nv), Lf,
                     Xacc, beta=1.0)
                gemm(Lf, GmQ, Yall[p0:p1].reshape(n * nv, ny))
            with timer('density'):
                for k in range(n):
                    Tk = T[k].reshape(dm, nvp)
                    gemm(L[k].reshape(dm, nvp), Tk.T, Dr, beta=1.0)
                    D1 += Tk @ l2pp[p0 + k]
        return Yall, Dr, D1

    def gram(self, L3, T3):
        '''G[I, J] = sum_A l3[I, A] t3[J, A] over the unique virtual triples.'''
        L3, T3 = self.view(L3), self.view(T3)
        if not self.size:
            return np.zeros((self.om.ntri,) * 2, dtype=self.dtype)
        return gemm(L3, T3.T)

    def pass2(self, L3, WL, Adrv, Bdrv, U, Vp, R):
        '''R += P(i/P) P(a/Q) Z[P, a, i, Q],

          Z[P, (a,i), Q]    = sum_(d,m) WL[(a,i), (d,m)] L[P, (d,m), Q]
                              + sum_x U[(a,i), x] Vp[P, x, Q]
          Z[(P,c), (i,Q)]  -= sum_k Adrv[P, c, k] Bdrv[k, (i,Q)]

        WL (nv no, nv no) or None, U (nv no, nu), Vp (npo, nu, nvp), Adrv
        (npo, nv, K), Bdrv (K, no nvp).'''
        om, vm, dtype = self.om, self.vm, self.dtype
        no, nv, npo, nvp = om.n, vm.n, om.npair, vm.npair
        dm, K = nv * no, Bdrv.shape[0]
        L3 = self.view(L3)
        nb, nu = self.nb, U.shape[1]
        # the outer products ride on the ring GEMM: [WL | U] [L[P]; Vp[P]]
        WLU = np.zeros((dm, dm + nu), dtype=dtype)
        if WL is not None:
            WLU[:, :dm] = WL
        WLU[:, dm:] = U
        bufL = np.empty((nb, dm + nu, nvp), dtype=dtype)
        Z = np.empty((nb, dm, nvp), dtype=dtype)
        for p0 in range(0, npo, nb):
            p1 = min(p0 + nb, npo)
            n = p1 - p0
            with timer('gather'):
                for k in range(n):
                    self._half(L3, p0 + k, p0 + k + 1, bufL[k, :dm].reshape(1, nv, no, nvp))
                bufL[:n, dm:] = Vp[p0:p1]
            with timer('gemm'):
                for k in range(n):
                    gemm(WLU, bufL[k], Z[k])
            with timer('drive'):
                gemm(np.ascontiguousarray(Adrv[p0:p1], dtype=dtype).reshape(n * nv, K), Bdrv,
                     Z[:n].reshape(n * nv, no * nvp), alpha=-1.0, beta=1.0)
            with timer('reduce'):
                t3pack.add_reduce_split(R, Z[:n].reshape(n, nv, no, nvp), om, vm,
                                        np.arange(p0, p1))

    def ladder(self, L3, Wpp, R, Xq, X2):
        '''With Lv[Q, I, c] = l3[I, Q_0, Q_1, c]:

          R[I, A]  += P(c/Q') sum_Q Wpp[Q, Q'] Lv[Q, I, c]     (Wpp = None: not)
          r2[P, Q]  = sum_{I,e} Xq[P, I, e] Lv[Q, I, e]
          r1[a, i]  = sum_{I,Q} Lv[Q, I, a] X2[Q, I, i]

        Returns (r2, r1).'''
        om, vm, dtype = self.om, self.vm, self.dtype
        no, nv, npo, nvp = om.n, vm.n, om.npair, vm.npair
        L3 = self.view(L3)
        r2 = np.zeros((npo, nvp), dtype=dtype)
        r1 = np.zeros((nv, no), dtype=dtype)
        if not self.size:
            return r2, r1
        nbt = self.triple_batch()
        # the gathered batch and its product (a short last batch in the same storage)
        ws = np.empty((2, nvp * nbt * nv), dtype=dtype)
        for i0 in range(0, om.ntri, nbt):
            rows = np.arange(i0, min(i0 + nbt, om.ntri))
            n = len(rows)
            with timer('gather'):
                Lv = t3pack.gather_vir_half(L3, vm, rows,
                                            out=ws[0, :nvp * n * nv].reshape(nvp, n, nv))
                Lf = Lv.reshape(nvp, n * nv)
            if Wpp is not None:
                X = ws[1, :nvp * n * nv].reshape(nvp, n * nv)
                with timer('gemm'):
                    gemm(Wpp.T, Lf, X)
                with timer('reduce'):
                    t3pack.add_reduce_pair_first(R, X.reshape(nvp, n, nv), vm, rows)
            with timer('small'):
                gemm(np.ascontiguousarray(Xq[:, i0:i0 + n]).reshape(npo, n * nv), Lf.T, r2,
                     beta=1.0)
                gemm(Lv.reshape(nvp * n, nv).T,
                     np.ascontiguousarray(X2[:, i0:i0 + n]).reshape(nvp * n, no), r1, beta=1.0)
        return r2, r1

    def triple_batch(self):
        '''Occupied triples per batch of the ladder: the gathered batch and
        its product, two thirds of the buffer budget.'''
        per = 2 * self.vm.npair * self.vm.n * self.dtype.itemsize
        return int(min(max(1, self.mem_mb * 1e6 * 2 / 3 // max(per, 1)), max(self.om.ntri, 1)))

    def vir_batches(self, T3, extra=0):
        '''(rows, Tv[Q, I, c]) for batches of occupied triples; ``extra``:
        elements per triple that the caller allocates next to a batch.'''
        T3 = self.view(T3)
        per = (self.vm.npair * self.vm.n + extra) * self.dtype.itemsize
        nbt = int(min(max(1, self.mem_mb * 1e6 * 2 / 3 // max(per, 1)), max(self.om.ntri, 1)))
        for i0 in range(0, self.om.ntri, nbt):
            rows = np.arange(i0, min(i0 + nbt, self.om.ntri))
            yield rows, t3pack.gather_vir_half(T3, self.vm, rows)

    def occ(self, O6, L3, R):
        '''R += O6 @ l3 on the packed arrays.'''
        L3 = self.view(L3)
        if not self.size:
            return
        if t3kern.USE_C:
            t3kern.occ_operator(O6, L3, R, t3engine.occ_cols(self.om.ntri, self.dtype.itemsize,
                                                             self.mem_mb))
        else:
            R += O6.astype(self.dtype) @ L3

    def target(self, out=None):
        return self.t3.target(out)

    def update(self, L3, R, eo, ev):
        '''l3 + R / D written over R.'''
        return self.t3.update(L3, R, eo, ev)


# ------------------------------------------------------------ intermediates

class Intermediates:
    '''What a Lambda iteration needs of the integrals and of t (see the
    module).  Index letters as in the tables.'''


def _maps(eris, t1):
    nocc, nvir = t1.shape
    return t3pack.triple_map(nocc), t3pack.triple_map(nvir)


def block_engine(eris, dtype, mem_mb=None):
    '''The engine on the blocks of the symmetry labels of these integrals
    (lambda_blk.BlockL3 on the T3 engine of the amplitude equations), or
    None: no labels in use, integrals in the U/R form, or no C kernels.'''
    if ccsdt_opt._is_spin_eris(eris) or t3pack.maps_for(eris)[0].spins is None:
        return None
    t3eng = ccsdt_opt.engine_for(eris, dtype, mem_mb)
    if not lambda_blk.available(t3eng):
        return None
    t3eng.io = None             # (T3, l3 and the residual are in ordinary memory here)
    return lambda_blk.BlockL3(t3eng)


def make_intermediates(eris, t1, t2, T3p, mem_mb=2000.0, blocks=False):
    '''Intermediates for the amplitudes t1, t2 and the packed T3
    (C(no,3), C(nv,3)), lexicographic: ccsdt_opt.packed_t3) - or, with
    ``blocks``, T3 in the blocks of the symmetry labels as the amplitude
    equations store it (the engine is then lambda_blk.BlockL3, and pairs and
    triples are in the order of their groups in everything below).'''
    if ccsdt_opt._is_spin_eris(eris):
        raise NotImplementedError('the Lambda equations take spin-orbital integrals (zccsdt.eris)')
    nocc, nvir = t1.shape
    dtype = np.result_type(t1, t2, T3p, eris.oovv, eris.fock)
    if blocks:
        eng = block_engine(eris, dtype, mem_mb)
        if eng is None:
            raise ValueError('no blocks of symmetry labels for these integrals (labels in use, '
                             'C kernels)')
        om, vm = eng.om, eng.vm
    else:
        om, vm = _maps(eris, t1)
        eng = DenseL3(om, vm, dtype, mem_mb)
    T3p = eng.view(T3p)
    mm, nn = om.pairs[:, 0], om.pairs[:, 1]
    md, ne = vm.pairs[:, 0], vm.pairs[:, 1]
    npo, nvp = om.npair, vm.npair
    t1 = np.asarray(t1, dtype=dtype)
    t2 = np.asarray(t2, dtype=dtype)
    im = Intermediates()
    im.eng, im.om, im.vm, im.dtype = eng, om, vm, dtype
    im.nocc, im.nvir = nocc, nvir

    with timer('dress'):
        fdr = dress_fock(eris, t1)
        ge = {k: np.asarray(dress_block(eris, t1, k), dtype=dtype)
              for k in ('ovoo', 'oooo', 'ovvo', 'oovv', 'ovvv', 'ooov', 'oovo')}
        vvvv_pp = dressed_vvvv_pp(eris, t1, om, vm)
    foo, fov, fvv = fdr[:nocc, :nocc], fdr[:nocc, nocc:], fdr[nocc:, nocc:]
    goovv = ge['oovv']
    t2mn = np.ascontiguousarray(t2[mm, nn])                     # (P, a, b)
    t2pp = np.ascontiguousarray(t2mn[:, md, ne])                # (P, Q)
    t2P = np.ascontiguousarray(t2[:, :, md, ne])                # (i, j, Q)
    goovv_mn = np.ascontiguousarray(goovv[mm, nn])              # (P, e, f)
    goovv_pp = np.ascontiguousarray(goovv_mn[:, md, ne])        # (P, Q)
    goovv_P = np.ascontiguousarray(goovv[:, :, md, ne])         # (m, n, Q)
    goovo_mn = np.ascontiguousarray(ge['oovo'][mm, nn])         # (P, e, j)

    # ---- the drive operands: the rows and columns that do not change, with
    # <ei||Q> = -<ie||Q> packed over the ket pair
    K = 2 * (nvir + nocc)
    with timer('operands'):
        im.Bdrv = Bdrv = np.zeros((K, nocc, nvp), dtype=dtype)
        vovvQ = Bdrv[nvir + nocc:2 * nvir + nocc]               # [e, i, Q]
        t3pack.axpby(vovvQ, ge['ovvv'][:, :, md, ne].transpose(1, 0, 2), -1.0, 0.0)
        Bdrv[2 * nvir + nocc:] = goovv_P.transpose(1, 0, 2)     # [n, k, Q] = <kn||Q>
        im.Adrv = Adrv = np.zeros((npo, nvir, K), dtype=dtype)
        Adrv[:, :, :nvir] = goovv_mn                            # [P, c, f]
        Adrv[:, :, nvir:nvir + nocc] = goovo_mn                 # [P, c, m]

    # ---- pass 1 over T3, as in the amplitude equations: the T3 parts of
    # W_vvvo and W_vooo
    with timer('pass1'):
        G_Pd = np.concatenate((goovv_mn.transpose(1, 0, 2), goovo_mn.transpose(2, 0, 1)), axis=0)
        G_mQ = np.empty((nocc, nvp, nocc + nvir), dtype=dtype)
        t3pack.axpby(G_mQ[:, :, :nocc], goovv_P.transpose(1, 2, 0), 1.0, 0.0)
        t3pack.axpby(G_mQ[:, :, nocc:], vovvQ.transpose(1, 2, 0), 1.0, 0.0)
        Xacc, Yall, _, _ = eng.t3.pass1(T3p, G_Pd, G_mQ.reshape(nocc * nvp, nocc + nvir),
                                        goovv_pp, fov)
        del G_mQ, G_Pd

    # ---- W_vvvo[Q(ab), e, i]: the drive intermediate of the T3 equation
    with timer('Wvvvo'):
        Bv = bra_dressed_vvvo_pp(eris, t1, om, vm)
        t3pack.add_transposed(Bv.reshape(nvp, -1), Xacc[:nvir].reshape(nvir * nocc, nvp))
        del Xacc
        Wv = wvvvo_packed(Bv, vvvv_pp, t1, t2, fov, ge, om, vm, inplace=True)
        # operand of pass 1: [(m, Q), n] = t2[m,n,Q];  [(m, Q), no + b] = W_vvvo[Q, b, m]
        GmQ = np.empty((nocc, nvp, nocc + nvir), dtype=dtype)
        t3pack.axpby(GmQ[:, :, :nocc], t2P.transpose(0, 2, 1), 1.0, 0.0)
        t3pack.axpby(GmQ[:, :, nocc:], Wv.transpose(2, 0, 1), 1.0, 0.0)
        im.GmQ = GmQ.reshape(nocc * nvp, nocc + nvir)
        del Wv, Bv
        # <ej||bg> = -<je||bg> as [g, e, j, b]; the dressed ovvv block goes
        im.Vx = np.empty((nvir, nvir, nocc, nvir), dtype=dtype)
        t3pack.axpby(im.Vx, ge['ovvv'].transpose(3, 1, 0, 2), -1.0, 0.0)
        del ge['ovvv']

    # ---- W_vooo[j, P(mn), e] = <ej||mn>-bar
    with timer('Wvooo'):
        Wo = einsum('if,Pef->iPe', fov, t2mn)
        Wo += gemm(vovvQ.reshape(nvir * nocc, nvp), t2pp.T).reshape(nvir, nocc, npo).transpose(1, 2, 0)
        Wo -= ge['ovoo'][:, :, mm, nn].transpose(0, 2, 1)
        Wo += Yall[:, :, :nocc].transpose(2, 0, 1)
        X = einsum('iofm,noef->eimn', ge['oovo'], t2)
        X = X - X.transpose(0, 1, 3, 2)
        Wo += X[:, :, mm, nn].transpose(1, 2, 0)
        im.GPd = np.empty((nvir + nocc, npo, nvir), dtype=dtype)
        im.GPd[:nvir] = t2mn.transpose(2, 0, 1)                 # [f, P, e] = t2[P, e, f]
        im.GPd[nvir:] = Wo
        del Wo, X, Yall

    with timer('small'):
        im.fov = np.ascontiguousarray(fov)
        im.Wov = ge['ovvo'] + einsum('mnde,inae->madi', goovv, t2)      # [k, e, c, m]
        im.Fvv = fvv - einsum('Paf,Pdf->ad', t2mn, goovv_mn)
        im.Foo = foo + einsum('mnP,inP->mi', goovv_P, t2P)
        Woooo = ge['oooo'] + einsum('mnP,ijP->mnij', goovv_P, t2P)
        im.Woooo_P = np.ascontiguousarray(Woooo[:, :, mm, nn])          # [i, j, P(mn)]
        im.Woooo_pp = np.ascontiguousarray(im.Woooo_P[mm, nn])          # [P(ij), P(mn)]
        O6 = t3pack.occ_operator(om, lambda t: ccsdt_opt._occ_apply(t, Woooo, im.Foo))
        im.O6T = np.ascontiguousarray(O6.T, dtype=dtype)
        im.WL = np.ascontiguousarray(im.Wov.transpose(2, 0, 1, 3)).reshape(nvir * nocc, -1)
        # [(m, e), (j, b)] and [i, (g, f, m)] of Wov[k, e, c, m]
        im.Wov_mejb = np.ascontiguousarray(im.Wov.transpose(3, 1, 0, 2)).reshape(nvir * nocc, -1)
        im.Wov_igfm = np.ascontiguousarray(im.Wov.transpose(0, 2, 1, 3)).reshape(nocc, -1)
        im.oovv, im.oovo = goovv, ge['oovo']
        im.goovv_pp, im.goovv_P, im.goovv_mn, im.goovo_mn = goovv_pp, goovv_P, goovv_mn, goovo_mn
        im.t2, im.t2P = t2, t2P
    with timer('Wpp'):          # the T3 ladder operator, in the dressed <ab||de>
        Wpp = vvvv_pp           # (its diagonal pair blocks: one without labels)
        del vvvv_pp
        for h, (lo, hi) in enumerate(vm.blocks):
            if hi > lo:
                if Wpp[h].dtype != dtype:
                    Wpp[h] = Wpp[h].astype(dtype)
                if npo:
                    gemm(np.asfortranarray(t2pp[:, lo:hi].T),
                         np.ascontiguousarray(goovv_pp[:, lo:hi]), Wpp[h], beta=1.0)
                add_one_body_pairs(Wpp[h], im.Fvv, vm, 0.5, block=h)
        im.Wpp = Wpp if blocks else Wpp[0]
    del ge

    # ---- contractions of T3 with <ij||fg> and <ei||gh>
    with timer('X'):
        im.Xq = np.zeros((npo, om.ntri, nvir), dtype=dtype)
        im.X2 = np.zeros((nvp, om.ntri, nocc), dtype=dtype)
        if eng.size and blocks:
            eng.contract_x(T3p, goovv_pp, vovvQ, im.Xq, im.X2)
        elif eng.size:
            vq = vovvQ.reshape(nvir * nocc, nvp)
            for rows, Tv in eng.vir_batches(T3p, nocc * nvir * nvir + 2 * nvp * nocc):
                n = len(rows)
                Tf = Tv.reshape(nvp, n * nvir)
                im.Xq[:, rows] = gemm(goovv_pp, Tf).reshape(npo, n, nvir)
                Z = gemm(vq, Tf).reshape(nvir, nocc, n, nvir)           # [e, i, I, f]
                del Tv, Tf
                A = Z[md, :, :, ne]
                A -= Z[ne, :, :, md]
                del Z
                im.X2[:, rows] = A.transpose(0, 2, 1)
                del A
    mo_e = np.asarray(eris.mo_energy)
    im.eia = mo_e[:nocc, None] - mo_e[None, nocc:]
    im.eo = mo_e[:nocc][om.triples].sum(axis=1) if om.ntri else np.zeros(0)
    im.ev = mo_e[nocc:][vm.triples].sum(axis=1) if vm.ntri else np.zeros(0)
    return im


# ---------------------------------------------------------------- residuals

def _pack_ij(X, om):
    '''P(ij) of X[i, j, ...] on the pairs i<j.'''
    mm, nn = om.pairs[:, 0], om.pairs[:, 1]
    return X[mm, nn] - X[nn, mm]


def _unpack_occ(X, om):
    '''x[i, j, ...] antisymmetric from X[P(i<j), ...].'''
    mm, nn = om.pairs[:, 0], om.pairs[:, 1]
    out = np.zeros((om.n, om.n) + X.shape[1:], dtype=X.dtype)
    out[mm, nn] = X
    out[nn, mm] = -X
    return out


def residuals(im, l1, l2, L3p, T3p, out=None, terms=None):
    '''The residuals dL/dt_k of the three Lambda equations: (r1[i,a],
    r2[(i<j), (a<b)], R3[I, A]) for dense l1, l2 and the packed l3.  R3 is
    new, or ``out`` (overwritten).  ``terms``: the groups of terms of R3 to
    evaluate (a subset of GROUPS; default all - for the tests).'''
    eng, om, vm, dtype = im.eng, im.om, im.vm, im.dtype
    no, nv, npo, nvp = im.nocc, im.nvir, om.npair, vm.npair
    dm = nv * no
    mm, nn = om.pairs[:, 0], om.pairs[:, 1]
    md, ne = vm.pairs[:, 0], vm.pairs[:, 1]
    on = set(GROUPS if terms is None else terms)
    l1 = np.asarray(l1, dtype=dtype)
    l2 = np.asarray(l2, dtype=dtype)
    L3p, T3p = eng.view(L3p), eng.view(T3p)
    l2mn = np.ascontiguousarray(l2[mm, nn])                     # (P, a, b)
    l2pp = np.ascontiguousarray(l2mn[:, md, ne])                # (P, Q)
    l2P = np.ascontiguousarray(l2[:, :, md, ne])                # (i, j, Q)
    Adrv, Bdrv = im.Adrv, im.Bdrv
    K = Bdrv.shape[0]

    # ---- pass 1: l3 with t2, with the W's and with T3
    with timer('pass1'):
        Xacc = Bdrv[:nv + no].reshape(nv + no, no * nvp)
        Yall, Dr, D1 = eng.pass1(L3p, T3p, im.GPd, im.GmQ, l2pp, Xacc)
        Y2 = Bdrv[:nv]                                          # [f, i, Q(ab)]: sum_{m<n,e} t2 l3
        Y1 = Yall[:, :, :no]                                    # [P(ij), a, n]: sum_{m,e<f} l3 t2
    with timer('gram'):
        G = eng.gram(L3p, T3p)
        sg = om.sgn.astype(np.float64)
        Do4 = np.zeros((npo, npo), dtype=dtype)                 # l3[ijm,efg] t3[nom,efg]
        Goo3 = np.zeros((no, no), dtype=dtype)                  # l3[imn,efg] t3[omn,efg]
        if om.ntri:
            for m in range(no):
                Do4 += 6.0 * (sg[m][:, None] * sg[m][None, :]) \
                    * G[om.tri[m][:, None], om.tri[m][None, :]]
            Goo3 = 12.0 * np.einsum('iP,oP,ioP->io', sg, sg,
                                    G[om.tri[:, None, :], om.tri[None, :, :]])
        del G
    with timer('r12'):
        Dr4 = Dr.reshape(nv, no, nv, no)                        # [a, i, g, o]
        Gvv3 = 4.0 * np.einsum('aigi->ag', Dr4)                 # l3[mno,aef] t3[mno,gef]
        r2 = np.array(im.goovv_pp, dtype=dtype)
        # the parts of the Lambda_2 residual that pass 1 has made
        Xij = np.array(Xacc[nv:].reshape(no, no, nvp).transpose(1, 0, 2))     # [i, j, Q]
        Yab = np.array(Yall[:, :, no:])                                       # [P, a, b]
        # drive operands of this iteration (Xacc[nv:] is given up for l2)
        Bdrv[nv:nv + no] = l2P                                  # [m, i, Q]
        Adrv[:, :, nv + no:2 * nv + no] = l2mn.transpose(0, 2, 1)             # [P, a, e] = l2[P, e, a]
        Adrv[:, :, 2 * nv + no:] = Y1

    # ---- the Lambda_3 residual
    R = eng.target(out)
    if eng.size:
        with timer('pass2'):
            U = np.zeros((dm, 2), dtype=dtype)
            Vp = np.zeros((npo, 2, nvp), dtype=dtype)
            if 'outer' in on:
                U[:, 0], U[:, 1] = l1.T.ravel(), im.fov.T.ravel()
                Vp[:, 0], Vp[:, 1] = im.goovv_pp, l2pp
            A, B = Adrv, Bdrv.reshape(K, no * nvp)
            if not {'drive2', 'drive3'} <= on:
                A = Adrv.copy()
                if 'drive2' not in on:
                    A[:, :, nv:2 * nv + no] = 0
                if 'drive3' not in on:
                    A[:, :, :nv] = 0
                    A[:, :, 2 * nv + no:] = 0
            eng.pass2(L3p, im.WL if 'ring' in on else None, A, B, U, Vp, R)
    with timer('ladder'):
        r2q, r1x = eng.ladder(L3p, im.Wpp if 'pp' in on else None, R, im.Xq, im.X2)
    if 'hh' in on:
        with timer('hh'):
            eng.occ(im.O6T, L3p, R)

    # ---- the Lambda_1 and Lambda_2 residuals
    with timer('r12'):
        t2, t2P, Wov, Fvv, Foo, Vx = im.t2, im.t2P, im.Wov, im.Fvv, im.Foo, im.Vx
        fov, oovv, oovo = im.fov, im.oovv, im.oovo
        vovvQ = Bdrv[nv + no:2 * nv + no]                       # [e, i, Q]
        Gvv = 0.5 * einsum('mnef,mnfg->eg', l2, t2) - Gvv3 / 12.0
        Goo = 0.5 * einsum('mnef,noef->mo', l2, t2) - Goo3 / 12.0
        # [j, h]: l2 with T3, and l3 with t2 twice
        Dov = D1.reshape(nv, no).T - gemm(Y2.reshape(nv, no * nvp), t2P.reshape(no, no * nvp).T).T
        Y1u = _unpack_occ(Y1, om)                               # [i, m, a, n]
        # Y2 on both orders of its pair: [g, e, i, a] = Y2[g, i, (a e)]
        Y2x = pair_expand(Y2, vm)

        r1 = np.array(fov, dtype=dtype)
        r1 += l1 @ Fvv - Foo @ l1 + einsum('me,ieam->ia', l1, Wov)
        r1 += gemm(l2P.reshape(no, no * nvp), im.GmQ[:, no:])
        r1 += einsum('Pae,iPe->ia', l2mn, im.GPd[nv:])
        # (the products with o v^3-sized arrays as they are stored: no transposed copies)
        r1 += (Gvv.T.ravel() @ Vx.reshape(nv * nv, dm)).reshape(no, nv)
        r1 += einsum('jh,ijah->ia', Dov, oovv) + einsum('mj,ijam->ia', Goo, oovo)
        r1 -= (Goo3 @ fov + fov @ Gvv3.T) / 12.0
        # - sum_{Q,h} Y2[h,i,Q] (~<Q||ah> + t2 <mn||ah>): with the ladder
        # operator, whose folded Fvv is taken out again
        r1 -= pair_matvec(Y2, im.Wpp, vm)
        r1 += 0.5 * (np.einsum('hhie->ie', Y2x) @ Fvv
                     + (Fvv.T.ravel() @ Y2x.reshape(nv * nv, dm)).reshape(no, nv))
        r1 += gemm(np.ascontiguousarray(Dr4.transpose(1, 2, 0, 3)).reshape(no, nv * nv * no),
                   Vx.reshape(nv * nv * no, nv))
        r1 += einsum('amgj,ijgm->ia', Dr4, oovo)
        r1 -= einsum('imeo,oeam->ia', Y1u, Wov)
        X = gemm(Do4, im.goovo_mn.reshape(npo, dm)).reshape(npo, nv, no) / 6.0
        P = np.arange(npo)
        np.add.at(r1, mm, X[P, :, nn])
        np.subtract.at(r1, nn, X[P, :, mm])
        r1 += r1x.T
        r1 -= einsum('Paj,ijP->ia', Y1, im.Woooo_P)
        r1 -= gemm(im.Wov_igfm, Y2x.reshape(nv * nv * no, nv))       # Wov[i,f,g,m] Y2x[g,f,m,a]

        Xij += einsum('ie,ejQ->ijQ', l1, vovvQ) - einsum('imQ,jm->ijQ', l2P, Foo) \
            - einsum('io,joQ->ijQ', Goo, im.goovv_P)
        r2 += _pack_ij(Xij, om)
        Yab += einsum('ma,Pbm->Pab', l1, im.goovo_mn) + 0.5 * einsum('Pae,eb->Pab', l2mn, Fvv) \
            - einsum('ag,Pbg->Pab', Gvv, im.goovv_mn)
        r2 += Yab[:, md, ne] - Yab[:, ne, md]
        if nvp and npo:
            if isinstance(im.Wpp, list):    # (the diagonal pair blocks)
                for (plo, phi), (lo, hi), Wb in zip(om.blocks, vm.blocks, im.Wpp):
                    if phi > plo and hi > lo:
                        r2[plo:phi, lo:hi] += gemm(np.ascontiguousarray(l2pp[plo:phi, lo:hi]), Wb)
            else:
                gemm(l2pp, im.Wpp, r2, beta=1.0)
            gemm(im.Woooo_pp, l2pp, r2, beta=1.0)
            gemm(np.ascontiguousarray(Y1).reshape(npo, dm), vovvQ.reshape(dm, nvp), r2, beta=1.0)
            gemm(Do4, im.goovv_pp, r2, alpha=1.0 / 6.0, beta=1.0)
            gemm(im.goovo_mn.reshape(npo, dm), Y2.reshape(dm, nvp), r2, beta=1.0)
        r2 += r2q
        # P(ij) P(ab) M[(i,a), (j,b)]
        M = np.multiply.outer(l1.ravel(), fov.ravel()).astype(dtype)
        gemm(np.ascontiguousarray(l2.transpose(0, 2, 1, 3)).reshape(dm, dm), im.Wov_mejb, M,
             beta=1.0)
        gemm(Y2x.reshape(nv * nv, dm).T, Vx.reshape(nv * nv, dm), M, beta=1.0)
        M += einsum('aigo,jobg->iajb', Dr4, oovv).reshape(dm, dm)
        M += einsum('imao,jobm->iajb', Y1u, oovo).reshape(dm, dm)
        ia = mm[:, None] * nv + md[None, :]
        ib = mm[:, None] * nv + ne[None, :]
        ja = nn[:, None] * nv + md[None, :]
        jb = nn[:, None] * nv + ne[None, :]
        r2 += M[ia, jb] - M[ja, ib] - M[ib, ja] + M[jb, ia]
    return r1, r2, R


def update_lams(im, l1, l2, L3p, T3p, out=None, level_shift=0.0):
    '''One Jacobi step l += (dL/dt) / D.  Returns (l1, l2, l3) with the new
    l3 packed and flat, in ``out`` if given.'''
    om, vm = im.om, im.vm
    r1, r2, R = residuals(im, l1, l2, L3p, T3p, out=out)
    eia = im.eia - level_shift
    eijab = eia[:, None, :, None] + eia[None, :, None, :]
    l1n = l1 + r1 / eia
    l2n = l2 + ccsdt_opt._unpack_pairs(r2, om, vm) / eijab
    with timer('update'):
        L3n = im.eng.update(L3p, R, im.eo, im.ev + 3 * level_shift) if im.eng.size else R.ravel()
    return l1n, l2n, L3n.ravel()


# ------------------------------------------------------------------- memory

@dataclass
class LambdaPlan:
    '''Memory of the Lambda equations (plan_memory), in bytes of ordinary
    memory.'''
    mem_mb: float = 2000.0      # batch buffers of the passes over the packed arrays
    need: int = 0               # all of it
    setup: int = 0              # the o v^3-sized and smaller arrays while the intermediates are made
    iteration: int = 0          # ... and in an iteration (the larger of the two is in ``need``)
    kept: int = 0               # of which the operands that live through the iterations
    diis: int = 0               # DIIS vectors that are not in the scratch file
    blocks: bool = False        # l3 in the blocks of the symmetry labels (plan_memory_blocks)
    buffers: int = 0            # ... then: the work array of the block engine


def plan_memory(nocc, nvir, itemsize=16, budget=None, mem_mb=2000.0, diis_space=0):
    '''LambdaPlan for ``budget`` bytes of ordinary memory beyond what the
    integral object holds (None: no limit).  With f = o v^3 elements of
    ``itemsize`` bytes, s = o^2 v^2 elements and r = 1 + o / v:

      T3-sized  3 C(o,3) C(v,3): T3, l3 and the residual; the DIIS vectors
                of this size are in the scratch file unless they are small
                (diis.INCORE_SIZE): then 2 ``diis_space`` more (``diis``)
      pp        v^4 / 4: the ladder operator, made in the dressed <ab||de>
      kept      the operands of the passes: r f / 2 for pass 1 (t2 and
                W_vvvo), r f for the drive products, f for <ej||bg>, 9 s,
                and the contractions of T3 with <ei||gh> and <ij||fg>,
                o C(o,3) v^2 / 2 + C(o,3) o^2 v / 2 elements
      setup     the largest of: the dressed <ie||bc> (f) next to the
                transient of dressing the pp operator (as in an amplitude
                iteration: at most 2 GB); f (2 + 1.5 r) when <ej||bg> is
                made; the kept operands and the batch buffers (or the
                occupied operator while it is made, 6 C(o,3) o^3 elements)
      iteration the kept operands and 8 s next to the batch buffers (two
                thirds of mem_mb, at least one occupied pair: f) or, after
                the passes, next to 1.5 f for the residuals of rank 1 and 2
      overhead  of the threaded BLAS and the allocator (ccsdt_opt.overhead),
                and three C(o,3)^2 arrays

    If that exceeds the budget the buffers shrink, down to one occupied
    pair; MemoryError if it still does not fit (nothing of the Lambda
    equations is put into a scratch file except the DIIS vectors).'''
    from zccsdt.cc import diis
    o, v = nocc, nvir
    npo, nvp = o * (o - 1) // 2, v * (v - 1) // 2
    nto, nvt = o * (o - 1) * (o - 2) // 6, v * (v - 1) * (v - 2) // 6
    isz = itemsize
    f = o * v ** 3 * isz
    s = o * o * v * v * isz
    t3 = nto * nvt * isz
    pp = nvp * nvp * isz
    gmq = o * nvp * (o + v) * isz
    bdrv = 2 * (o + v) * o * nvp * isz
    kept = gmq + bdrv + f + (nvp * nto * o + npo * nto * v) * isz + 9 * s
    fixed = 3 * nto * nto * isz + ccsdt_opt.overhead()
    dress = min(2_000_000_000, 2 * pp)
    occ = 6 * nto * o ** 3 * isz            # the occupied operator from the packed basis
    plan = LambdaPlan(mem_mb=mem_mb, kept=kept)
    plan.diis = 2 * diis_space * t3 if nto * nvt < diis.INCORE_SIZE else 0

    def bufs(mem):
        return int(min(max(mem * 1e6 * 2 / 3, f), max(npo, 1) * f))

    def fill(mem):
        plan.mem_mb = mem
        plan.setup = max(f + dress, gmq + bdrv + 2 * f, kept - 9 * s + max(bufs(mem), occ)) + 9 * s
        plan.iteration = kept + max(bufs(mem), 3 * f // 2) + 8 * s
        plan.need = fixed + 3 * t3 + plan.diis + pp + max(plan.setup, plan.iteration)
        return budget is None or plan.need <= budget

    mem_min = 1.5 * f / 1e6
    if fill(mem_mb):
        return plan
    lo, hi = min(mem_mb, mem_min), mem_mb
    if fill(lo):
        # the largest buffers that fit (the need grows with them)
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if fill(mid) else (lo, mid)
        fill(lo)
        return plan
    raise MemoryError('the Lambda equations of CCSDT with %d occupied and %d virtual spin '
                      'orbitals need about %.1f GB of memory (T3, Lambda_3 and its residual: '
                      '%.1f GB; the pp ladder operator: %.1f GB; operands and buffers: %.1f GB); '
                      '%.1f GB are available (max_memory, or what the node and the memory '
                      'limit of the job leave)'
                      % (o, v, plan.need / 1e9, 3 * t3 / 1e9, pp / 1e9,
                         max(plan.setup, plan.iteration) / 1e9, budget / 1e9))


def plan_memory_blocks(eng, budget=None, mem_mb=2000.0, diis_space=0):
    '''LambdaPlan of the equations on the blocks of the symmetry labels
    (``eng``: a lambda_blk.BlockL3) for ``budget`` bytes, as plan_memory.
    The operands are the dense ones of plan_memory (f = o v^3, s = o^2 v^2
    elements, r = 1 + o / v); what the labels change:

      T3-sized  three times the blocks of T3
      pp        the diagonal pair blocks of the ladder operator
      buffers   the work array of the block engine (kept; its
                buffer_bytes(mem_mb): the products of a batch of occupied
                pairs for all groups of virtual pairs and one gathered block)
      kept      as in plan_memory, and the blocks of the ring operator
      setup     as in plan_memory with the dressing of one pair block, the
                copies in blocks of the operand of pass 1 over T3, and the
                transients of the contractions of T3 with <ei||gh>
      iteration the kept operands and 8 s next to the copies in blocks of
                the drive operand [x, (i, Q)] (pass 2: the elements of
                r f whose labels add up; half of that in pass 1) or, after
                the passes, next to 1.5 f

    If that exceeds the budget the buffers shrink, to one occupied pair and
    one occupied triple per batch; MemoryError if it still does not fit.'''
    from zccsdt.cc import diis
    om, vm = eng.om, eng.vm
    o, v, isz = om.n, vm.n, eng.dtype.itemsize
    npo, nvp, nto = om.npair, vm.npair, om.ntri
    f = o * v ** 3 * isz
    s = o * o * v * v * isz
    t3 = eng.size * isz
    nq = [hi - lo for lo, hi in vm.blocks]
    pp = sum(n * n for n in nq) * isz
    gmq = o * nvp * (o + v) * isz
    bdrv = 2 * (o + v) * o * nvp * isz
    ring = eng.ring_bytes()
    drive = eng.drive_bytes()
    kept = gmq + bdrv + f + (nvp * nto * o + npo * nto * v) * isz + 9 * s + ring
    fixed = 3 * nto * nto * isz + ccsdt_opt.overhead()
    # dressing a pair block: its rows in batches of at most 2 GB next to o v
    # of its columns
    dress = max([min(2_000_000_000, v * v * n * isz) + o * v * n * isz for n in nq] + [0])
    occ = 6 * nto * o ** 3 * isz            # the occupied operator from the packed basis
    plan = LambdaPlan(mem_mb=mem_mb, kept=kept, blocks=True)
    plan.diis = 2 * diis_space * t3 if eng.size < diis.INCORE_SIZE else 0

    def fill(mem):
        plan.mem_mb = mem
        plan.buffers = int(eng.buffer_bytes(mem))
        plan.setup = max(f + dress, gmq + bdrv + 2 * f + drive,
                         kept - 9 * s + max(eng.x_bytes(mem), occ)) + 9 * s
        plan.iteration = kept + max(2 * drive + s, 3 * f // 2) + 8 * s + eng.occ_bytes()
        plan.need = fixed + 3 * t3 + plan.diis + pp + plan.buffers + max(plan.setup, plan.iteration)
        return budget is None or plan.need <= budget

    if fill(mem_mb):
        return plan
    lo, hi = min(mem_mb, 1e-3), mem_mb
    if fill(lo):
        # the largest buffers that fit (the need grows with them)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if fill(mid) else (lo, mid)
        fill(lo)
        return plan
    raise MemoryError('the Lambda equations of CCSDT in the blocks of the symmetry labels, %d '
                      'occupied and %d virtual spinors, need about %.1f GB of memory (T3, '
                      'Lambda_3 and its residual: %.1f GB; the pp ladder operator: %.1f GB; '
                      'operands and buffers: %.1f GB); %.1f GB are available (max_memory, or what '
                      'the node and the memory limit of the job leave)'
                      % (o, v, plan.need / 1e9, 3 * t3 / 1e9, pp / 1e9,
                         (plan.buffers + max(plan.setup, plan.iteration)) / 1e9, budget / 1e9))


def plan_for(eris, t1, dtype, opts, blocks=False):
    '''LambdaPlan of a run with the solver options ``opts`` (max_memory in
    MB for the whole process; default: 80% of what is available).  T3 is
    taken to be in memory already, as the caller passes it.  ``blocks``: of
    the equations on the blocks of the symmetry labels.'''
    from zccsdt import memory
    nocc, nvir = np.shape(t1)
    isz = np.dtype(dtype).itemsize
    space = 0 if opts.incore_complete else opts.diis_space
    eng = None
    if blocks:
        eng = block_engine(eris, dtype)
        if eng is None:
            raise ValueError('no blocks of symmetry labels for these integrals (labels in use, '
                             'C kernels)')
    if opts.incore_complete:
        if blocks:
            return plan_memory_blocks(eng, None, opts.mem_mb, space)
        return plan_memory(nocc, nvir, isz, None, opts.mem_mb, space)
    cached = ccsdt_opt.cached_bytes(eris)
    om, vm = t3pack.triple_map(nocc), t3pack.triple_map(nvir)
    budget = memory.budget(opts.max_memory)
    # what the dressing puts on the integral object if it is not there for
    # these maps: the bare <my||Q> and <mn||Q> on pairs and, for the blocks
    # of the labels, the pair blocks of the bare <ab||de>
    bare = 0
    pm = eng.vm if blocks else vm
    isz_b = np.dtype(np.result_type(eris.oovv)).itemsize
    held = eris.__dict__.get('_bare_Q')
    if held is None or held[0] is not pm:
        bare += (nocc * nvir + om.npair) * pm.npair * isz_b
    held = eris.__dict__.get('_vvvv_blocks')
    if blocks and (held is None or held[0] is not pm):
        bare += sum((hi - lo) ** 2 for lo, hi in pm.blocks) * isz_b
    if budget is not None:      # (what is there already: T3, kept arrays of the amplitude step)
        budget += cached + (eng.size if blocks else om.ntri * vm.ntri) * isz - bare
    if blocks:
        plan = plan_memory_blocks(eng, budget, opts.mem_mb, space)
    else:
        plan = plan_memory(nocc, nvir, isz, budget, opts.mem_mb, space)
    plan.need += bare
    return plan


# ------------------------------------------------------------------- solver

@dataclass
class Result:
    converged: bool
    ncycle: int
    l1: np.ndarray
    l2: np.ndarray
    l3: np.ndarray              # packed (i<j<k, a<b<c), flat; or its blocks (``blocks``)
    timings: list = None        # wall time per iteration
    plan: object = None         # the LambdaPlan
    setup: float = 0.0          # wall time of the intermediates
    blocks: bool = False        # l3 in the blocks of the symmetry labels, as T3 was given


def kernel(eris, t1, t2, t3, opts=None, l1=None, l2=None, l3=None, blocks=False):
    '''Solve the Lambda equations of CCSDT for converged amplitudes: t1, t2
    dense, t3 packed as solver.kernel returns it.  Jacobi steps with DIIS
    from conj(t) (or from l1, l2, l3), converged when the norm of the change
    of the full tensors is below ``opts.conv_tol_normt``.  ``opts``:
    solver.Options (level_shift, max_cycle, diis_space, diis_file,
    incore_complete, mem_mb, max_memory, verbose, callback are used; ``plan``
    takes a LambdaPlan).  Returns a Result with l3 packed.

    Integrals with symmetry labels in use, whose T3 solver.kernel returns in
    the blocks of the labels: by default the blocks are put on the unique
    block [i<j<k, a<b<c] in lexicographic order and the equations are solved
    there, l3 returned on it; with ``blocks`` they are solved in the blocks
    (lambda_blk) and l3 (given and returned) has the layout of t3 -
    packed_l3() puts it on the unique block.

    T3, l3 and the residual are held in memory (plan_memory,
    plan_memory_blocks raise MemoryError if they do not fit in
    max_memory).'''
    opts = opts or solver.Options()
    nocc, nvir = np.shape(t1)
    T3p = np.asarray(t3).ravel() if blocks else ccsdt_opt.packed_t3(eris, t3)
    dtype = np.result_type(t1, t2, T3p, eris.oovv, eris.fock)
    if isinstance(opts.plan, LambdaPlan) and opts.plan.blocks == bool(blocks):
        plan = opts.plan
    else:
        plan = plan_for(eris, t1, dtype, opts, blocks)
    T3p = np.ascontiguousarray(T3p, dtype=dtype)
    t0 = time.perf_counter()
    with timer('lam.setup'):
        im = make_intermediates(eris, t1, t2, T3p, mem_mb=plan.mem_mb, blocks=blocks)
    setup = time.perf_counter() - t0
    size = im.eng.size
    l1 = np.conj(t1).astype(dtype) if l1 is None else np.asarray(l1, dtype=dtype)
    l2 = np.conj(t2).astype(dtype) if l2 is None else np.asarray(l2, dtype=dtype)
    # (overwritten by the DIIS extrapolation: not the caller's array)
    l3 = np.conj(T3p).ravel() if l3 is None else np.array(l3, dtype=dtype).ravel()
    if not l3.flags.writeable or l3.base is T3p or np.shares_memory(l3, T3p):
        l3 = l3.copy()
    adiis = DIIS(space=opts.diis_space, filename=opts.diis_file, incore=opts.incore_complete)
    timings, conv = [], False
    rbuf = None
    try:
        for icycle in range(opts.max_cycle):
            t0, c0 = time.perf_counter(), time.process_time()
            if rbuf is None and size:
                rbuf = np.empty(size, dtype=dtype)
            with timer('lam.iter'):
                l1n, l2n, l3n = update_lams(im, l1, l2, l3, T3p, out=rbuf if size else None,
                                            level_shift=opts.level_shift)
            with timer('lam.diis'):
                first = adiis.xprev is None
                if first:
                    norms = [np.linalg.norm(l1n - l1), np.linalg.norm(l2n - l2),
                             6.0 * solver._diff_norm(l3n, l3)]
                parts = adiis.update(solver._parts(l1n, l2n, l3n), out=[None, None, l3])
                if not first:
                    en = adiis.err_norms
                    norms = [en[0], 2.0 * en[1], 6.0 * en[2]]
                norms = [float(x) for x in norms]
                normt = sum(norms)
                l1, l2, l3 = solver._from_parts(parts, nocc, nvir, True)
                # the array of the new l3 is free again unless DIIS keeps it
                if rbuf is not None and any(np.shares_memory(rbuf, x[2]) for x in adiis.xs):
                    rbuf = None
            timings.append(time.perf_counter() - t0)
            if opts.verbose:
                print('lambda cycle %3d  |dl| = %.3e  %.2fs' % (icycle + 1, normt, timings[-1]))
            if opts.callback is not None:
                opts.callback(dict(cycle=icycle + 1, norms=tuple(norms), normt=normt,
                                   diis=len(adiis.xs), wall=timings[-1],
                                   cpu=time.process_time() - c0))
            if normt < opts.conv_tol_normt:
                conv = True
                break
    finally:
        adiis.close()
    return Result(converged=conv, ncycle=icycle + 1, l1=l1, l2=l2, l3=l3, timings=timings,
                  plan=plan, setup=setup, blocks=bool(blocks))


def packed_l3(eris, l3):
    '''l3 of kernel(..., blocks=True) on the unique block [C(no,3),
    C(nv,3)] in lexicographic order of the triples (zeros outside of the
    blocks of the labels): the layout of kernel() without ``blocks``.'''
    return ccsdt_opt.packed_t3(eris, l3)


def blocked_l3(eris, l3):
    '''The inverse of packed_l3: the blocks of the symmetry labels of these
    integrals of an l3 on the unique block.'''
    return ccsdt_opt.blocked_t3(eris, l3)


def dense_l3(l3, nocc, nvir):
    '''The full antisymmetric l3[i,j,k,a,b,c] of a packed one (small cases).'''
    from zccsdt.cc import ccsdtq_terms as Q
    from math import comb
    return Q.expand(np.asarray(l3).reshape(comb(nocc, 3), comb(nvir, 3)), nocc, nvir, 3)
