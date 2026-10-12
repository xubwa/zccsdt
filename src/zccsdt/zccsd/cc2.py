'''CC2 for spinors on the Cholesky vectors of the scalar AO integrals: ground
state and excitation energies without any four-index integral in the spinor
basis and without stored doubles (docs/ccn.md, section 8).

    (mu nu | la si) = sum_c L[c, mu, nu] L[c, la, si]       (real, symmetric: ladder='chol')

T1 is in the coefficients.  With x = 1 - t1^T (bra / creator indices) and
y = 1 + t1 (ket / annihilator indices) the T1-dressed integrals are
(pq|rs)~ = sum_c L~[c,p,q] L~[c,r,s], L~ = x L y^T, and CC2 needs of them

    B[c, i, a] = L~[c, a, i]                    the dressed virtual-occupied block
    t2[ij, ab] = ((ai|bj)~ - (aj|bi)~) / (e_i + e_j - e_a - e_b)
               = antisym[sum_c B[c,i,a] B[c,j,b]] / D            (canonical spinors)

- the doubles equation of CC2 in closed form.  The amplitudes are made per
occupied index i for all j > i (one GEMM, v x n_chol x (o - i) v), used and
discarded; what the singles equation needs of them is

    Y[c, i, a] = sum_jb t2[ij, ab] L[c, j, b]                    (two GEMMs per i)

and the same with the dressed f~[j, b] in the place of L (an extra row of
the operand).  The singles residual, a function of t1 alone:

    R1[i, a] = f~[a, i] + sum_jb t2[ij, ab] f~[j, b]
             + sum_ce L~[c, a, e] Y[c, i, e] - sum_cn L~[c, n, i] Y[c, n, a]

The third term goes through the AO basis (Y back-transformed in e, one
product with L): no L[c, a, b] over two virtual indices is ever made.  The
three-index quantities of an iteration - L~ with a dressed occupied ket
index, and from it B, L~[c, n, i] and the exchange part of f~ - come from
one half-transformation of L with the T1 part of the ket coefficients,
o N^2 n_chol.  An iteration is 1.5 o^2 v^2 n_chol complex multiply-adds in
3 o large ZGEMMs plus work of lower order; the memory is 16 o n_chol
(n + 3 v) bytes (n spinors) next to L.

Excitation energies (linear response, the eigenvalues of the CC2
Jacobian): the doubles are folded,

    r2[ij, ab] = antisym[sum_c B'[c,i,a] B[c,j,b] + B[c,i,a] B'[c,j,b]] / (omega + D),

B' the derivative of B along r1 (one-index transformed coefficients), and
the eigenproblem in the singles space depends on omega (sigma(); solved
root by root to self-consistency, zccsdt.cc.ccn.folded_roots).

The class of the integrals is that of zccsdt.zccsd.DirectZCCSD(ladder =
'chol'); DirectZCCSD.cc2() and .cc2_ee() are the front end.  Timers
(zccsdt.timing): 'cc2.half' (three-index quantities from L), 'cc2.pairs'
(with .m, .amps, .y: the GEMM of the integrals, the amplitudes, the two
GEMMs of Y), 'cc2.r1' (the terms from Y), 'cc2.fock'.
'''

import ctypes
import os
import time

import numpy as np
from pyscf.lib import logger

from zccsdt import memory
from zccsdt.blas import gemm
from zccsdt.cc.ccsdt_opt import transposed
from zccsdt.timing import timer
from zccsdt.zccsd import kernels


def _load():
    path = os.environ.get('ZCCSDT_ZCCSD_LIB') or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lib', 'libzc_zccsd.so')
    try:
        lib = ctypes.CDLL(path)
        P, I, D = ctypes.c_void_p, ctypes.c_int64, ctypes.c_double
        lib.zc_cc2_amps.argtypes, lib.zc_cc2_amps.restype = [P, I, I, D, P, P, D], None
        lib.zc_cc2_vec.argtypes, lib.zc_cc2_vec.restype = [P, I, I, P, P, P, P], None
    except (OSError, AttributeError):
        return None
    return lib


_lib = _load()
Z = np.complex128


def _amps(M, ei, eoj, ev, omega=0.0):
    '''M[a, j, b] <- (M[a,j,b] - M[b,j,a]) / (ei + eoj[j] - ev[a] - ev[b] + omega), in place.'''
    nv, nj = M.shape[0], M.shape[1]
    if _lib is not None:
        eoj, ev = np.ascontiguousarray(eoj, dtype=np.float64), np.ascontiguousarray(ev, dtype=np.float64)
        _lib.zc_cc2_amps(M.ctypes.data, nv, nj, float(ei), eoj.ctypes.data, ev.ctypes.data, float(omega))
        return M
    M -= M.transpose(2, 1, 0).copy()
    M /= (ei + omega + eoj[None, :, None] - ev[:, None, None] - ev[None, None, :])
    return M


def _vec(t, f, g, out, outj):
    '''out[a] += sum_jb t[a,j,b] f[j,b];  outj[j,b] += sum_a t[a,j,b] g[a].'''
    nv, nj = t.shape[0], t.shape[1]
    if _lib is not None:
        _lib.zc_cc2_vec(t.ctypes.data, nv, nj, f.ctypes.data, g.ctypes.data, out.ctypes.data, outj.ctypes.data)
        return
    t2 = t.reshape(nv, nj * nv)
    out += t2 @ f.ravel()
    outj += (g @ t2).reshape(outj.shape)


def cholesky_exact(mol, tol=1e-12):
    '''Cholesky vectors L[c, mu, nu] that reproduce the scalar AO integrals
    to rounding, from the eigenvectors of the integral matrix: for checks
    on small molecules (nao^6 work).'''
    from pyscf import ao2mo
    nao = mol.nao_nr()
    eri = ao2mo.restore(1, mol.intor('int2e', aosym='s8'), nao).reshape(nao * nao, nao * nao)
    w, u = np.linalg.eigh(eri)
    keep = w > tol * w[-1]
    return np.ascontiguousarray((u[:, keep] * np.sqrt(w[keep])).T).reshape(-1, nao, nao)


class Engine:
    '''CC2 on Cholesky vectors for the active spinors ``mog`` (2 nao, n) in
    the (spin, scalar AO) basis, the first ``nocc`` occupied; ``fock``: their
    Fock matrix (canonical: diagonal blocks).  L: (n_chol, nao, nao) real.'''

    def __init__(self, mog, nocc, fock, L, e_hf=0.0, max_memory=None, verbose=0, stdout=None):
        self.log = logger.Logger(stdout, verbose) if stdout is not None else None
        mog = np.ascontiguousarray(mog, dtype=Z)
        self.nao = nao = mog.shape[0] // 2
        self.n, self.o = mog.shape[1], nocc
        self.v = self.n - nocc
        self.L = np.ascontiguousarray(np.asarray(L).reshape(-1, nao, nao), dtype=np.float64)
        self.nchol = self.L.shape[0]
        self.C = mog
        self.fock = np.asarray(fock, dtype=Z)
        e = self.fock.diagonal().real.copy()
        self.eo, self.ev = e[:nocc].copy(), e[nocc:].copy()
        o = nocc
        off = max(abs(self.fock[:o, :o] - np.diag(e[:o])).max(), abs(self.fock[o:, o:] - np.diag(e[o:])).max())
        if off > 1e-7:
            raise NotImplementedError('CC2 on Cholesky vectors: canonical spinors (the Fock matrix has off-diagonal '
                                      'elements of %.1e in its occupied or virtual block)' % off)
        self.e_hf = e_hf
        self.max_memory = max_memory
        self.plan = self._plan()
        self._static()
        self.t1 = None
        self.e_corr = None
        self.Y0 = self.Q0 = None

    # ------------------------------------------------------------- memory

    def stored_bytes(self, eom=False):
        '''The three-index arrays kept over an iteration: L[c, i, p], L[c, i, a] with
        the row of f~, B, Y, L~[c, i, n] (and for a sigma vector B', the Y of
        the ground state and L~'[c, i, n]).'''
        o, v, n, nc = self.o, self.v, self.n, self.nchol
        b = 16 * nc * o * (n + 3 * v + o) + 16 * 3 * o * v
        if eom:
            b += 16 * nc * o * (2 * v + o)
        return b

    def _plan(self, eom=False):
        '''(columns of L per chunk of the half-transformation, occupied j per
        batch of the pair loop) within max_memory; MemoryError with the
        numbers if the stored arrays do not fit.'''
        o, v, n, nao, nc = self.o, self.v, self.n, self.nao, self.nchol
        budget = memory.budget(self.max_memory)
        stored = self.stored_bytes(eom)
        # one column of L in a chunk: the half-transformed columns (nao x 2 o), their
        # transposed copy, L~[i, p] three times (TA, S, conj)
        per_c = 16 * o * (4 * nao + 3 * n) + 16 * o * 2 * nao
        per_j = 16 * v * v
        if budget is None:
            return dict(chunk=max(1, min(nc, int(2e9 // per_c))), nj=max(1, o - 1), stored=stored, budget=None,
                        work=per_c * min(nc, int(2e9 // per_c)) + per_j * max(1, o - 1))
        left = budget - stored
        floor = per_c + per_j
        if left < floor:
            raise MemoryError('CC2 on Cholesky vectors: %.0f MB for the three-index arrays (16 o n_chol (n + 3 v + o) '
                              'bytes, o = %d, v = %d, n_chol = %d) and %.1f MB of work do not fit in the %.0f MB that '
                              'max_memory leaves' % (stored / 1e6, o, v, nc, floor / 1e6, budget / 1e6))
        nj = max(1, min(o - 1, int(left / 2 // per_j)))
        chunk = max(1, min(nc, int((left - nj * per_j) // per_c), int(4e9 // per_c) or 1))
        return dict(chunk=chunk, nj=nj, stored=stored, budget=budget, work=per_c * chunk + per_j * nj)

    def describe(self):
        p = self.plan
        return ('%d + %d spinors, %d Cholesky vectors of %d AO functions; three-index arrays %.0f MB, work %.0f MB '
                '(chunks of %d vectors, batches of %d occupied indices)%s'
                % (self.o, self.v, self.nchol, self.nao, p['stored'] / 1e6, p['work'] / 1e6, p['chunk'], p['nj'],
                   '' if p['budget'] is None else '; %.0f MB allowed' % (p['budget'] / 1e6)))

    # ------------------------------------------- three-index quantities from L

    def _half(self, c0, c1, ket):
        '''X[c, mu, (s, k)] = sum_nu L[c, mu, nu] ket[(s, nu), k] for the columns
        c0:c1 of L: (nc, nao, 2 nk) complex, one real GEMM.'''
        nao = self.nao
        nk = ket.shape[1]
        K = np.ascontiguousarray(np.concatenate((ket[:nao], ket[nao:]), axis=1))        # (nao, (s, k))
        Kr = K.view(np.float64).reshape(nao, 4 * nk)
        X = gemm(self.L[c0:c1].reshape(-1, nao), Kr)
        return X.view(Z).reshape(c1 - c0, nao, 2 * nk)

    def _full(self, X, bra):
        '''out[c, k, p] = sum_(s, mu) conj(bra[(s, mu), p]) X[c, mu, (s, k)].'''
        nc, nao = X.shape[0], self.nao
        nk = X.shape[2] // 2
        Xt = transposed(X.reshape(nc, nao, 2, nk), (0, 3, 2, 1))                         # [c, k, s, mu]
        return gemm(Xt.reshape(nc * nk, 2 * nao), self._conj(bra)).reshape(nc, nk, bra.shape[1])

    @staticmethod
    def _conj(a):
        a = np.ascontiguousarray(a, dtype=Z)
        return kernels.copy(a, conj=True) if kernels.USE_C else a.conj()

    def chunks(self):
        step = self.plan['chunk']
        return [(c0, min(c0 + step, self.nchol)) for c0 in range(0, self.nchol, step)]

    def _static(self):
        '''L[c, i, p] = L[c, p, i] of the spinors (bra p, ket i occupied) and
        V[c, (i a)] = L[c, i, a], with one more row for f~[i, a].'''
        o, v, n, nc = self.o, self.v, self.n, self.nchol
        self.Lpo = np.empty((nc, o, n), dtype=Z)
        self.V = np.empty((nc + 1, o * v), dtype=Z)
        with timer('cc2.half'):
            for c0, c1 in self.chunks():
                T = self._full(self._half(c0, c1, self.C[:, :o]), self.C)
                self.Lpo[c0:c1] = T
                self.V[c0:c1] = self._conj(T[:, :, o:]).reshape(c1 - c0, o * v)
        self.V[nc] = 0

    def _dressing(self, t1, lpo_out, b_out=None, prev=None):
        '''One pass over L for the ket coefficients C_v t1^T.  Returns
        (GK, w) - the exchange part sum_ci TA[c,i,r] conj(L[c,i,s]) of the
        Fock dressing and w[c] = sum_ia L[c,i,a] t1[i,a] of its Coulomb
        part - and fills

            lpo_out[c, i, n]   TA[c, i, n] (+ L[c, i, n] if b_out is given: L~[c, n, i])
            b_out[c, i, a]     B = L~[c, a, i], dressed in bra and ket;

        ``prev`` = (t1 of the ground state, its L~[c, i, n], r1) instead:
        b_out is the derivative of B along r1 = ``t1``.'''
        o, v, n = self.o, self.v, self.n
        ket = gemm(self.C[:, o:], np.ascontiguousarray(t1.T, dtype=Z))                  # (2 nao, o)
        GK = np.zeros((n, n), dtype=Z)
        for c0, c1 in self.chunks():
            nc = c1 - c0
            TA = self._full(self._half(c0, c1, ket), self.C)                            # [c, i, p]
            TB = self.Lpo[c0:c1]
            gemm(TA.reshape(nc * o, n).T, self._conj(TB.reshape(nc * o, n)), GK, beta=1.0)
            if prev is None:
                TA += TB                                                                # L~ with the dressed ket
                lpo_out[c0:c1] = TA[:, :, :o]
                B = b_out[c0:c1].reshape(nc * o, v)
                B[:] = TA[:, :, o:].reshape(nc * o, v)
                gemm(np.ascontiguousarray(TA[:, :, :o]).reshape(nc * o, o), t1, B, alpha=-1.0, beta=1.0)
            else:
                t0, lt_oo, r1 = prev
                lpo_out[c0:c1] = TA[:, :, :o]
                B = b_out[c0:c1].reshape(nc * o, v)
                B[:] = TA[:, :, o:].reshape(nc * o, v)
                gemm(np.ascontiguousarray(TA[:, :, :o]).reshape(nc * o, o), t0, B, alpha=-1.0, beta=1.0)
                gemm(np.ascontiguousarray(lt_oo[c0:c1]).reshape(nc * o, o), r1, B, alpha=-1.0, beta=1.0)
        w = self.V[:self.nchol] @ np.ascontiguousarray(t1, dtype=Z).ravel()
        return GK, w

    def _g(self, GK, w):
        '''G[r, s] = sum_ia <ri||sa> t1[i, a] = J - K from w and GK.'''
        nao = self.nao
        J = (self.L.reshape(self.nchol, nao * nao).T @ w.view(np.float64).reshape(self.nchol, 2)).view(Z).reshape(nao, nao)
        Ca, Cb = self.C[:nao], self.C[nao:]
        return Ca.conj().T @ (J @ Ca) + Cb.conj().T @ (J @ Cb) - GK

    def _xy(self, t1):
        n, o = self.n, self.o
        x, y = np.eye(n, dtype=Z), np.eye(n, dtype=Z)
        x[o:, :o] -= t1.T
        y[:o, o:] += t1
        return x, y

    # ----------------------------------------------------------- pair loop

    def _pairs(self, B, Y, V, dB=None, omega=0.0, fvec=None):
        '''The loop over occupied i and batches of j > i.

        Ground state (dB None): t[a, j, b] = t2[ij, ab] from B, and
        Y[c, (i a)] += sum_jb t2[ij, ab] V[c, (j b)] for all rows of V.
        With ``dB``: r2 from B, dB at the frequency ``omega`` in the place
        of t2; and ``fvec`` = (f'[i,a], out[i,a]): out += sum_jb t2[ij,ab]
        f'[j,b] with the amplitudes t2 of the ground state.'''
        o, v = self.o, self.v
        nrow = V.shape[0]
        njmax = self.plan['nj']
        M = np.empty(v * njmax * v, dtype=Z)
        Bf = B.reshape(self.nchol, o * v)
        dBf = None if dB is None else dB.reshape(self.nchol, o * v)
        if fvec is not None:
            fp, outf = np.ascontiguousarray(fvec[0], dtype=Z), fvec[1]
        for i in range(o - 1):
            ci = slice(i * v, (i + 1) * v)
            for j0 in range(i + 1, o, njmax):
                j1 = min(j0 + njmax, o)
                nj = j1 - j0
                cj = slice(j0 * v, j1 * v)
                Mb = M[:v * nj * v].reshape(v, nj * v)
                if fvec is not None:
                    with timer('cc2.pairs.m'):
                        gemm(Bf[:, ci].T, Bf[:, cj], Mb)
                    with timer('cc2.pairs.amps'):
                        t = _amps(Mb.reshape(v, nj, v), self.eo[i], self.eo[j0:j1], self.ev)
                        _vec(t, fp[j0:j1], fp[i], outf[i], outf[j0:j1])
                with timer('cc2.pairs.m'):
                    if dB is None:
                        gemm(Bf[:, ci].T, Bf[:, cj], Mb)
                    else:
                        gemm(dBf[:, ci].T, Bf[:, cj], Mb)
                        gemm(Bf[:, ci].T, dBf[:, cj], Mb, beta=1.0)
                with timer('cc2.pairs.amps'):
                    _amps(Mb.reshape(v, nj, v), self.eo[i], self.eo[j0:j1], self.ev, omega)
                with timer('cc2.pairs.y'):
                    # Y[c, (i a)] += sum_(j b) V[c, (j b)] t[a, (j b)];  Y[c, (j b)] += sum_a V[c, (i a)] t[a, (j b)]
                    gemm(V[:, cj], Mb.T, Y[:, ci], beta=1.0)
                    gemm(V[:, ci], Mb, Y[:, cj], beta=1.0)
        return Y

    def _from_y(self, Y, lt_oo, t1):
        '''sum_ce L~[c,a,e] Y[c,i,e] - sum_cn L~[c,n,i] Y[c,n,a] for the rows
        c < n_chol of Y[c, (i a)], L~[c, n, i] = lt_oo[c, i, n].'''
        o, v, nao, nchol = self.o, self.v, self.nao, self.nchol
        Cv = self.C[:, o:]
        Zt = np.zeros((nao, 4 * o))                             # [mu, (s, i)] complex, as real
        CvT = np.ascontiguousarray(Cv.T)                        # [e, (s, nu)]
        for c0, c1 in self.chunks():
            nc = c1 - c0
            Yb = gemm(Y[c0:c1].reshape(nc * o, v), CvT)         # [(c i), (s nu)]
            Yt = transposed(Yb.reshape(nc, o, 2, nao), (0, 3, 2, 1))                 # [c, nu, s, i]
            gemm(self.L[c0:c1].reshape(nc * nao, nao).T, Yt.view(np.float64).reshape(nc * nao, 4 * o), Zt, beta=1.0)
        Zc = Zt.view(Z).reshape(nao, 2, o)                      # [mu, s, i]
        Z2 = np.ascontiguousarray(Zc.transpose(1, 0, 2)).reshape(2 * nao, o)
        bra = Cv - self.C[:, :o] @ t1.conj()                    # the dressed bra coefficients of the virtual spinors
        out = np.ascontiguousarray((bra.conj().T @ Z2).T)       # [i, a]
        Lt = transposed(lt_oo, (1, 0, 2)).reshape(o, nchol * o)                      # [i, (c n)]
        gemm(Lt, Y[:nchol].reshape(nchol * o, v), out, alpha=-1.0, beta=1.0)
        return out

    # --------------------------------------------------------- ground state

    def residual(self, t1, keep=False):
        '''(R1[i, a], E_corr) of CC2 at t1 (the doubles in closed form).
        ``keep``: the quantities of a sigma vector are kept (the converged
        ground state).'''
        o, v, n, nchol = self.o, self.v, self.n, self.nchol
        t1 = np.ascontiguousarray(t1, dtype=Z)
        B = np.empty((nchol, o, v), dtype=Z)
        lt_oo = np.empty((nchol, o, o), dtype=Z)
        with timer('cc2.half'):
            GK, w = self._dressing(t1, lt_oo, B)
        with timer('cc2.fock'):
            G = self._g(GK, w)
            F0 = self.fock + G
            x, y = self._xy(t1)
            fd = x @ F0 @ y.T
            self.V[nchol] = fd[:o, o:].ravel()
        Y = np.zeros((nchol + 1, o * v), dtype=Z)
        with timer('cc2.pairs'):
            self._pairs(B, Y, self.V)
        with timer('cc2.r1'):
            R1 = fd[o:, :o].T + Y[nchol].reshape(o, v) + self._from_y(Y, lt_oo, t1)
            e = np.vdot(self.fock[:o, o:].conj(), t1) + 0.5 * np.vdot(G[:o, o:].conj(), t1)
            e += 0.5 * kernels.dotu(self.V[:nchol].ravel(), Y[:nchol].ravel()) if kernels.USE_C \
                else 0.5 * np.dot(self.V[:nchol].ravel(), Y[:nchol].ravel())
        if keep:
            self.B, self.lt_oo, self.F0, self.fd = B, lt_oo, F0, fd
            self.Y0 = np.ascontiguousarray(Y[:nchol])
            # Q0[m, i] = sum_ce L[c, m, e] Y0[c, i, e]
            self.Q0 = gemm(transposed(self.V[:nchol].reshape(nchol, o, v), (1, 0, 2)).reshape(o, nchol * v),
                           transposed(self.Y0.reshape(nchol, o, v), (0, 2, 1)).reshape(nchol * v, o))
        return R1, complex(e)

    def kernel(self, t1=None, conv_tol=1e-9, conv_tol_normt=1e-7, max_cycle=100, diis_space=8, verbose=None):
        '''The ground state: t1 by Jacobi steps with DIIS (the doubles are a
        function of t1).  Returns (E_corr, t1); self.converged.'''
        from zccsdt.cc.diis import DIIS
        o, v = self.o, self.v
        log = self.log
        eia = self.eo[:, None] - self.ev[None, :]
        t1 = np.zeros((o, v), dtype=Z) if t1 is None else np.asarray(t1, dtype=Z)
        adiis = DIIS(space=diis_space, incore=True)
        e_old, self.converged = 0.0, False
        self.timings = []
        for cycle in range(max_cycle):
            t0 = time.perf_counter()
            R1, e = self.residual(t1)
            new = t1 + R1 / eia
            normt = float(np.linalg.norm(new - t1))
            t1 = adiis.update([new.ravel()])[0].reshape(o, v)
            self.timings.append(time.perf_counter() - t0)
            if log is not None:
                log.info('cycle = %d  E_corr(CC2) = %.15g  dE = %.9g  norm(t1) = %.6g  %.2f s',
                         cycle + 1, e.real, e.real - e_old, normt, self.timings[-1])
            done = abs(e.real - e_old) < conv_tol and normt < conv_tol_normt
            e_old = e.real
            if done:
                self.converged = True
                break
        adiis.close()
        # the energy and the kept quantities at the final t1
        R1, e = self.residual(t1, keep=True)
        self.t1, self.e_corr, self.cycles = t1, float(e.real), cycle + 1
        self.r1_norm = float(np.linalg.norm(R1))
        return self.e_corr, t1

    def t2(self):
        '''t2[i, j, a, b] of the converged t1, dense: for checks.'''
        o, v = self.o, self.v
        Bf = self.B.reshape(self.nchol, o * v)
        M = (Bf.T @ Bf).reshape(o, v, o, v).transpose(0, 2, 1, 3)       # (ai|bj) as [i, j, a, b]
        M = M - M.transpose(0, 1, 3, 2)
        return M / (self.eo[:, None, None, None] + self.eo[None, :, None, None]
                    - self.ev[None, None, :, None] - self.ev[None, None, None, :])

    # ------------------------------------------------------ excited states

    def eom_ee(self, nroots=3, conv_tol=1e-6, omega_tol=None, nguess=None, max_cycle=60, max_outer=30):
        '''The ``nroots`` lowest excitation energies of CC2 linear response
        (Hartree) and the zccsdt.cc.eom.Solved of the solver: vectors r1[i, a]
        (flat, normalized in the singles space), per root the frequencies
        of its outer steps in ``history``.  ``conv_tol``: the norm of the
        residual of the folded problem; ``omega_tol`` (default conv_tol /
        100, at least 1e-10): the change of omega at self-consistency.'''
        from zccsdt.cc import ccn, eom
        o, v = self.o, self.v
        d1 = (self.ev[None, :] - self.eo[:, None]).ravel()
        store = eom.Store([o * v], Z)
        want = min(o * v, max(nguess or 0, nroots + 2))
        guesses = []
        for k in np.argsort(d1, kind='stable')[:want]:
            x = store.new()
            x[0][k] = 1.0
            guesses.append(x)

        def precond(x, omega):
            den = omega - d1
            den[abs(den) < 1e-3] = 1e-3
            x[0] /= den
            return x

        def apply_at(omega):
            def apply(x, y):
                y[0][:] = self.sigma(x[0].reshape(o, v), omega).ravel()
            return apply
        log = None
        if self.log is not None:
            def log(k, outer, omega, change, res, nsig):
                self.log.info('EOM-CC2 root %d  step %d  omega = %.12f  change %.1e  |residual| %.1e  %d sigma vectors',
                              k + 1, outer, omega, change, res, nsig)
        tol_w = max(1e-10, 1e-2 * conv_tol) if omega_tol is None else omega_tol
        sol = ccn.folded_roots(apply_at, precond, guesses, store, nroots, conv_tol, tol_w, max_cycle=max_cycle,
                               max_outer=max_outer, log=log)
        self.e_ee, self.eom = sol.e.real, sol
        return sol.e.real, sol

    def sigma(self, r1, omega):
        '''The folded Jacobian of CC2 at the frequency omega on r1[i, a]:
        A_11 r1 + A_12 (omega - A_22)^-1 A_21 r1 (docs/ccn.md, section 4).'''
        o, v, n, nchol = self.o, self.v, self.n, self.nchol
        if self.Y0 is None:
            raise RuntimeError('sigma(): the ground state first (kernel())')
        t1 = self.t1
        r1 = np.ascontiguousarray(r1, dtype=Z).reshape(o, v)
        dB = np.empty((nchol, o, v), dtype=Z)
        ta_oo = np.empty((nchol, o, o), dtype=Z)
        with timer('cc2.half'):
            GK, w = self._dressing(r1, ta_oo, dB, prev=(t1, self.lt_oo, r1))
        with timer('cc2.fock'):
            dG = self._g(GK, w)
            x, y = self._xy(t1)
            dx, dy = np.zeros((n, n), dtype=Z), np.zeros((n, n), dtype=Z)
            dx[o:, :o] = -r1.T
            dy[:o, o:] = r1
            df = dx @ self.F0 @ y.T + x @ self.F0 @ dy.T + x @ dG @ y.T
        Y = np.zeros((nchol + 1, o * v), dtype=Z)
        s = np.ascontiguousarray(df[o:, :o].T)
        with timer('cc2.pairs'):
            self.V[nchol] = self.fd[:o, o:].ravel()
            self._pairs(self.B, Y, self.V, dB=dB, omega=omega, fvec=(df[:o, o:], s))
        with timer('cc2.r1'):
            s += Y[nchol].reshape(o, v) + self._from_y(Y, self.lt_oo, t1)
            # the terms of H' = [H~, R1] with the Y of the ground state:
            # L'[c,a,e] = -r1[m,a] L[c,m,e] and L~'[c,n,i] = TA'[c,i,n]
            s -= self.Q0.T @ r1
            Lt = transposed(ta_oo, (1, 0, 2)).reshape(o, nchol * o)
            gemm(Lt, self.Y0.reshape(nchol * o, v), s, alpha=-1.0, beta=1.0)
        return s
