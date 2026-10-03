'''The amplitude equations of the AO-driven spinor CCSD, term by term on
fixed layouts.

The equations are those of zccsd_direct.update_amps (PySCF's GCCSD with the
intermediates of Stanton, Gauss et al., bare integrals); only the way each
term is evaluated differs:

- pairs are packed, i > j and a > b (numpy.tril_indices(n, -1) order), and
  tau = t2 + t1 t1 is formed once;
- the two o^3 v^3 terms, W(mbej) += <mn||ef> t2 and the ring t2 W, are one
  ZGEMM each on matrices [(i a), (j b)] (t2, <mn||ef> and W kept in that
  layout);
- the five terms with <ma||ef> read the integrals once, one occupied index
  at a time, from the array packed over e > f (half of ovvv);
- the pp ladder runs in the scalar AO basis over the pairs i > j.  The
  alpha-beta block of the back-transformed tau is split into its symmetric
  and antisymmetric parts in the two AO indices: the antisymmetric part
  joins alpha-alpha and beta-beta on the packed antisymmetrized integrals
  (mu si|nu rho) - (mu rho|nu si), the symmetric part takes the packed
  symmetrized integrals.  Two operators of 2 nao^4 bytes replace the
  nao^2 x nao^2 matrix, and the real GEMMs do o^2 nao^4 multiply-adds.

Timers (zccsdt.timing): 'amps.layouts', 'amps.imds', 'amps.ovvv',
'amps.ring', 'amps.ladder', 'amps.t2', 'amps.unpack'.
'''

import ctypes
import os

import numpy as np

from zccsdt.timing import timer


def _load():
    '''libzc_zccsd.so (csrc/zccsd_kernels.c); None if it has not been built:
    numpy then does the same, slower.'''
    path = os.environ.get('ZCCSDT_ZCCSD_LIB') or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lib', 'libzc_zccsd.so')
    try:
        lib = ctypes.CDLL(path)
    except OSError:
        return None
    lib.zc_ovvv_t1.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                               ctypes.c_int64, ctypes.c_void_p]
    lib.zc_ovvv_t1.restype = None
    return lib


_lib = _load()


class Residual:
    '''One Jacobi step of the CCSD equations for the integrals ``eris`` (the
    container of DirectZCCSD with the stored <ma||ef>): update(t1, t2)
    returns (t1new, t2new) as zccsd_direct.update_amps does.'''

    def __init__(self, eris):
        o = self.nocc = eris.nocc
        fock = np.asarray(eris.fock)
        v = self.nvir = fock.shape[0] - o
        self.foo, self.fov, self.fvv = fock[:o, :o], fock[:o, o:], fock[o:, o:]
        self.mo_e = np.asarray(eris.mo_energy)
        io, jo = self.io, self.jo = np.tril_indices(o, -1)
        av, bv = self.av, self.bv = np.tril_indices(v, -1)

        oovv = np.asarray(eris.oovv)                                    # <mn||ef>
        self.V_ring = np.ascontiguousarray(oovv.transpose(0, 2, 1, 3)).reshape(o * v, o * v)
        self.oovv_pf = oovv[io, jo]                                     # [(m>n), e, f]
        self.oovv_p = np.ascontiguousarray(self.oovv_pf[:, av, bv])     # [(m>n), (e>f)]
        self.oovv_ov = np.ascontiguousarray(oovv[:, :, av, bv])         # [m, n, (e>f)]
        ovov = np.ascontiguousarray(np.asarray(eris.ovov))              # <mb||je> as [m, b, j, e]
        self.ovov = ovov
        self.E_ring = -np.ascontiguousarray(ovov.transpose(0, 3, 2, 1)).reshape(o * v, o * v)
        self.oooo_p = np.ascontiguousarray(np.asarray(eris.oooo)[io, jo][:, io, jo])
        ooov = self.ooov = np.ascontiguousarray(np.asarray(eris.ooov))  # <mn||ie> as [m, n, i, e]
        self.ooov_r = np.ascontiguousarray(ooov.transpose(0, 3, 2, 1))  # [m, e, j, n]
        self.ooov_x = np.ascontiguousarray(ooov.transpose(0, 2, 1, 3))  # [m, i, n, e]
        ooov_p = ooov[io, jo]                                           # [(m>n), i, e]
        self.ooov_pe = np.ascontiguousarray(ooov_p.transpose(0, 2, 1))  # [(m>n), e, i]
        self.ooov_c2 = np.ascontiguousarray(ooov_p.transpose(1, 0, 2).conj())   # [m, (i>j), b]

        # <ma||ef> packed over e > f: [m, a, (e>f)]
        ovvv_p = getattr(eris, 'ovvv_p', None)
        self.ovvv_p = np.asarray(eris.ovvv)[:, :, av, bv] if ovvv_p is None else ovvv_p

        # AO side of the ladder
        nao = self.nao = eris.nao_sph
        self.C2 = np.ascontiguousarray(eris.mo_ao_spin[:, o:])           # (2 nao, nvir)
        self.W_same, self.W_plus = eris.W_same, eris.W_plus
        self.pi, self.pj = np.tril_indices(nao, -1)
        self.si, self.sj = np.tril_indices(nao)

    # -- the pp ladder in the AO basis --------------------------------------
    def ladder(self, taup):
        '''L[(i>j), (a>b)] = sum_ef (ae|bf) tau_ij^ef = 1/2 sum_ef <ab||ef> tau_ij^ef.'''
        o, v, nao = self.nocc, self.nvir, self.nao
        av, bv, pi, pj, si, sj = self.av, self.bv, self.pi, self.pj, self.si, self.sj
        npo, C2 = len(taup), self.C2
        tf = np.zeros((npo, v, v), dtype=taup.dtype)
        tf[:, av, bv] = taup
        tf[:, bv, av] = -taup
        # X[P, x, y] = sum_ef C2[x, e] tau[P, e, f] C2[y, f], x and y over (spin, AO)
        X = np.matmul(C2, (tf.reshape(npo * v, v) @ C2.T).reshape(npo, v, 2 * nao))
        ab = X[:, :nao, nao:]
        anti = np.concatenate((X[:, :nao, :nao][:, pi, pj], X[:, nao:, nao:][:, pi, pj],
                               0.5 * (ab[:, pi, pj] - ab[:, pj, pi])))
        sym = 0.5 * (ab[:, si, sj] + ab[:, sj, si])
        del X
        # the real AO operators on real and imaginary parts at once
        Ga = np.vstack((anti.real, anti.imag)) @ self.W_same
        Ga = Ga[:3 * npo] + 1j * Ga[3 * npo:]
        Gs = np.vstack((sym.real, sym.imag)) @ self.W_plus
        Gs = Gs[:npo] + 1j * Gs[npo:]
        G = np.zeros((npo, 2 * nao, 2 * nao), dtype=np.complex128)
        aa, bb, ab, ba = G[:, :nao, :nao], G[:, nao:, nao:], G[:, :nao, nao:], G[:, nao:, :nao]
        aa[:, pi, pj] = Ga[:npo]
        aa[:, pj, pi] = -Ga[:npo]
        bb[:, pi, pj] = Ga[npo:2 * npo]
        bb[:, pj, pi] = -Ga[npo:2 * npo]
        ab[:, si, sj] = Gs                          # G_S, symmetric
        ab[:, sj, si] = Gs
        ab[:, pi, pj] += Ga[2 * npo:]               # G_A, antisymmetric
        ab[:, pj, pi] -= Ga[2 * npo:]
        ba[:] = -ab.transpose(0, 2, 1)
        # L[P] = C2^+ G[P] C2^*
        C2c = C2.conj()
        L = np.matmul(C2c.T, (G.reshape(npo * 2 * nao, 2 * nao) @ C2c).reshape(npo, 2 * nao, v))
        return L[:, av, bv]

    # -- one step -------------------------------------------------------------
    def update(self, t1, t2, level_shift=0.0):
        o, v = self.nocc, self.nvir
        io, jo, av, bv = self.io, self.jo, self.av, self.bv
        npo, npv = io.size, av.size
        fov = self.fov
        mo_e_o = self.mo_e[:o]
        mo_e_v = self.mo_e[o:] + level_shift
        t1T = np.ascontiguousarray(t1.T)

        with timer('amps.layouts'):
            t2pv = t2[io, jo]                                   # [(i>j), a, b]
            t2p = np.ascontiguousarray(t2[:, :, av, bv])        # [i, j, (a>b)]
            Tr = np.ascontiguousarray(t2.transpose(0, 2, 1, 3)).reshape(o * v, o * v)
            # t1 t1 on packed pairs: t1[i,a] t1[j,b] - t1[i,b] t1[j,a]
            x_pp = t1[io][:, av] * t1[jo][:, bv] - t1[io][:, bv] * t1[jo][:, av]
            taup = t2pv[:, av, bv] + x_pp                       # [(i>j), (a>b)]
            x_pf = t1[io][:, :, None] * t1[jo][:, None, :]
            tautpv = t2pv + 0.5 * (x_pf - x_pf.transpose(0, 2, 1))        # tau~ [(m>n), f, a]
            x_op = t1[:, None, av] * t1[None, :, bv] - t1[:, None, bv] * t1[None, :, av]
            tautp = t2p + 0.5 * x_op                            # tau~ [i, n, (e>f)]
            del x_pf, x_op

        with timer('amps.imds'):
            Fov = fov + (self.V_ring @ t1.ravel()).reshape(o, v)
            Foo = self.foo + 0.5 * fov @ t1.T
            Foo += (self.ooov_x.reshape(o * o, o * v) @ t1.ravel()).reshape(o, o)
            Foo += (tautp.reshape(o, o * npv) @ self.oovv_ov.reshape(o, o * npv).T).T
            Fvv = self.fvv - 0.5 * t1T @ fov
            Fvv -= tautpv.reshape(npo * v, v).T @ self.oovv_pf.reshape(npo * v, v)
            # W(mnij) on packed pairs, with the tau tau <mn||ef> term of the ladder
            tmp = (self.ooov.reshape(o ** 3, v) @ t1T).reshape(o, o, o, o)
            Wp = self.oooo_p + (tmp[io, jo][:, io, jo] - tmp[io, jo][:, jo, io])
            Wp += self.oovv_p @ taup.T
            del tautp, tautpv

        with timer('amps.ring'):
            # W(mbej) as [(m e), (j b)]; its o^3 v^3 term is V (t2/2 - t1 t1)
            Tm = (0.5 * Tr).reshape(o, v, o, v)
            Tm -= np.einsum('nb,jf->nfjb', t1, t1)
            Wr = self.V_ring @ Tm.reshape(o * v, o * v)
            del Tm
            Wr += self.E_ring
            Wr += (self.ooov_r.reshape(o * v * o, o) @ t1).reshape(o * v, o * v)
            Wr4 = Wr.reshape(o, v, o, v)

        with timer('amps.ovvv'):
            # the five terms with <ma||ef>, one m at a time
            M = np.empty((o, v, npo), dtype=np.complex128)       # sum_ef <ma||ef> tau_ij^ef
            X1 = np.empty((o, v), dtype=np.complex128)
            Yall = np.empty((o, o, npv), dtype=np.complex128)
            A = np.empty((v, v, o), dtype=np.complex128)
            U = None if _lib is not None else np.zeros((v, v, v), dtype=np.complex128)
            # one pass over <ma||ef> for its products with tau and t2[m]
            B = np.empty((npv, npo + o), dtype=np.complex128)
            B[:, :npo] = taup.T
            t1c = t1.conj()
            t1C = np.ascontiguousarray(t1)
            for m in range(o):
                Om = np.ascontiguousarray(self.ovvv_p[m])        # [a, (e>f)]
                with timer('ovvv.gemm'):
                    B[:, npo:] = t2p[m].T
                    OB = Om @ B
                    M[m] = 2.0 * OB[:, :npo]
                    X1[...] = OB[:, npo:].T if m == 0 else X1 + OB[:, npo:].T
                    Yall[m] = (t1c @ Om).conj()
                with timer('ovvv.t1'):
                    # A[b, e, j] = sum_f <mb||ef> t1[j, f]
                    if _lib is not None:
                        _lib.zc_ovvv_t1(Om.ctypes.data, t1C.ctypes.data, v, o, A.ctypes.data)
                    else:
                        U[:, av, bv] = Om
                        U[:, bv, av] = -Om
                        A = (U.reshape(v * v, v) @ t1T).reshape(v, v, o)
                    Fvv -= A[:, :, m]
                    Wr4[m] += A.transpose(1, 2, 0)

        with timer('amps.ring'):
            r4 = (Tr @ Wr).reshape(o, v, o, v)                   # [i, a, j, b]
            del Wr, Wr4
            # + t1[i,e] t1[m,a] <mb||je> as [a, b, j, i]
            K = (self.ovov.reshape(o * v * o, v) @ t1T).reshape(o, v * o * o)
            s4 = (t1T @ K).reshape(v, v, o, o)

        Foo[np.diag_indices(o)] -= mo_e_o
        Fvv[np.diag_indices(v)] -= mo_e_v

        with timer('amps.t2'):
            I, J, A, B = io[:, None], jo[:, None], av[None, :], bv[None, :]
            R = self.oovv_p.conj()
            R += r4[I, A, J, B] - r4[J, A, I, B] - r4[I, B, J, A] + r4[J, B, I, A]
            R += s4[A, B, J, I] - s4[A, B, I, J] - s4[B, A, J, I] + s4[B, A, I, J]
            del r4, s4
            Y = (t2pv.reshape(npo * v, v) @ (Fvv - 0.5 * t1T @ Fov).T).reshape(npo, v, v)
            R += Y[:, av, bv] - Y[:, bv, av]
            Z = ((Foo + 0.5 * Fov @ t1.T).T @ t2p.reshape(o, o * npv)).reshape(o, o, npv)
            R += Z[jo, io] - Z[io, jo]
            R += Wp.T @ taup
            Q = (M.reshape(o, v * npo).T @ t1).reshape(v, npo, v)        # [a, (i>j), b]
            R += 0.5 * (Q[av, :, bv] - Q[bv, :, av]).T
            R += Yall[io, jo] - Yall[jo, io]
            Zc = (t1T @ self.ooov_c2.reshape(o, npo * v)).reshape(v, npo, v)
            R -= (Zc[av, :, bv] - Zc[bv, :, av]).T
        with timer('amps.ladder'):
            R += self.ladder(taup)

        with timer('amps.t2'):
            r1 = t1 @ Fvv.T - Foo.T @ t1 + (Tr @ Fov.ravel()).reshape(o, v)
            r1 += (t1.ravel() @ self.E_ring).reshape(o, v) + X1
            r1 += self.ooov_pe.reshape(npo * v, o).T @ t2pv.reshape(npo * v, v)
            r1 += fov.conj()

        with timer('amps.unpack'):
            eia = mo_e_o[:, None] - mo_e_v
            R /= eia[io][:, av] + eia[jo][:, bv]
            t2new = np.zeros((o, o, v, v), dtype=np.complex128)
            t2new[I, J, A, B] = R
            t2new[J, I, A, B] = -R
            t2new[I, J, B, A] = -R
            t2new[J, I, B, A] = R
        return r1 / eia, t2new
