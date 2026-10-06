'''The (T) terms of the orbital gradient of CCSD(T), in the layouts of
zccsdt.zccsd.amps and of the densities of zccsd.triples.

With E(T) in the energy the orbital gradient of zccsd.zvector changes in
three ways (zvector.gradient_t has all of it from full arrays, the reference
of this module):

- lambda has the sources of (T): zccsd.orbgrad with that lambda is
  Gnh[a, i] of the CCSD Lagrangian as before;
- the derivative of the Lagrangian with respect to t no longer vanishes,
  dL/dt = -s with the sources s1[i,a], s2[i,j,a,b]: the excitation half of
  the gradient and the rotations among occupied and among virtual spinors,
  which are changes of t, come back,

      Gnh[i, a] += s1[i,a]
      Gnh[j, i] -= sum_a s1[j,a] t1[i,a] + 1/2 sum_kab s2[j,k,a,b] t2[i,k,a,b]
      Gnh[b, a] += sum_i s1[i,a] t1[i,b] + 1/2 sum_ijc s2[i,j,a,c] t2[i,j,b,c];

- E(T) itself depends on the spinors.  Its real part changes in first
  order by Re sum (density x change of an integral), with the densities

      g1 of <bc||ek>,  g2 of <ma||jk>,  g3 of <bc||jk>,
      conj(dvo) of f[a,i],  -goo[i] of e_i,  gvv[a] of e_a

  (zccsd.triples.kernel with terms='gradient'; lam_t.densities), and a
  rotation takes a bra index y of an integral to x, a ket index x to minus y:

      Gnh[y, x] = sum D[x ...] <y ...||...> - sum D[... y ...] <...||... x ...>

  over the indices of every density, in all four blocks of (y, x), and the
  Fock matrix elements with their own rule (zvector.gradient_nh).

Every contraction is with integrals that the amplitude step keeps, but one:
g1[(bc), e, i] with <bc||ea>, in the occupied-virtual block.  <ab||cd> is not
kept, and it is not made here either.  The sum over b > c and e of
g1 <bc||ea> is sum_bce g1[b,c;e,i] (be|ca) with the antisymmetric g1, and
with the spinors written out in the scalar AOs (spin components s, s')

    D_i[(mu nu), s' la] = sum_s sum_bce C*[s mu,b] C[s nu,e] C*[s' la,c] g1[b,c;e,i]
    X[i, a] = sum_{s' si} C[s' si,a] sum_{(mu nu), la} (mu nu|la si) D_i[(mu nu), s' la]:

three virtual indices of the density of one occupied i are transformed to
the AO basis (2 N v^3 and 2 N^2 v^2 multiply-adds, the first on the packed
pairs), and all i are contracted with the AO integrals at once - one pass
over the integrals, computed for ranges of the shells of la and not kept, 2
o N^4 multiply-adds.  The cost goes with the number of occupied spinors, not
with that of the virtual ones as an integral transformation would (H2O/
cc-pVQZ, 10 + 220 spinors: 18.5 s with slabs (ea|..) of zccsd.ao2mo, the
time of all of (T) with its densities).  With Kramers pairs the unbarred i
are done and the rows of the barred ones are their time-reversed image.

Frozen spinors are not known here: E(T) is that of the active spinors, and
this is its gradient among them.  The blocks between a frozen and an active
spinor are in zccsd.orbgrad_frozen, from the integrals with one frozen index
and the same densities.

The result is Gnh[y, x] for all spinors; with Gnh[a, i] of zccsd.orbgrad
added, the orbital gradient is (Gnh - Gnh^+) / 2.  Its occupied-occupied and
virtual-virtual blocks are (e_q - e_p) y[p, q] exactly - y the off-diagonal
one-particle density of (T), whatever the convergence of the amplitudes -,
and vanish between degenerate spinors (zvector.kernel divides).  For that
the Fock matrix is taken as E(T) has it - the orbital energies on its
diagonal and f[i,a] -, not with what the SCF left elsewhere off the diagonal.

Timers (zccsdt.timing): 'ogt.pairs', 'ogt.small', 'ogt.ovvv' (the pass over
g1 and <ke||bc> by k), 'ogt.rows' (the one by e), 'ogt.vvvv' (g1 to the AO
basis: 'ogt.vvvv.ao', the AO integrals: 'ogt.vvvv.eri', their contraction),
'ogt.fock'.
'''

import numpy as np
from pyscf.lib import logger

from zccsdt import blas, memory, scratch
from zccsdt.timing import timer
from zccsdt.zccsd import amps, orbgrad
from zccsdt.zccsd import kernels as K

PARTS = ('sources', 'g1', 'g2', 'g3', 'fock')


def own_bytes(nocc, nvir, nao=0):
    '''Bytes of the arrays of gradient() besides the packed ones it is given
    and those of the contraction with <bc||ea> (vvvv_plan): a matrix on the
    occupied pairs, the matrices on the spinors, what the passes over g1 and
    <ke||bc> take for one occupied and one virtual index (rows that are
    unpacked: orbgrad.UNPACK_BYTES at a time), and the density of one
    occupied spinor on its way to the AO basis (``nao`` scalar AOs).'''
    o, v, nao = int(nocc), int(nvir), int(nao)
    npo, npv, n = K.npair(o), K.npair(v), o + v
    ao = 16 * (nao * npv + 2 * nao * nao * v + nao * (nao + 1) * v)
    return (16 * (npo * npo + 2 * o ** 3 + 12 * n * n + 4 * o * v * v + (3 * o + v) * npv)
            + 4 * orbgrad.UNPACK_BYTES + ao)


def vvvv_plan(mol, max_memory, nocc, nvir, nrows):
    '''(occupied spinors per pass over the AO integrals, AOs of the third
    index per block of integrals): the contraction of g1 with <bc||ea>
    within what max_memory leaves now next to own_bytes.  The density of an
    occupied spinor in the AO basis takes 32 N (N + 1) / 2 x N bytes (and
    once more while it is made), a block of nl AOs of the integrals
    8 N (N + 1) / 2 x N nl and as much of the densities again.  MemoryError
    if one spinor and the largest shell do not fit.'''
    nao = mol.nao_nr()
    npair = nao * (nao + 1) // 2
    ao_loc = mol.ao_loc_nr()
    lmax = int((ao_loc[1:] - ao_loc[:-1]).max())
    left = memory.budget(max_memory)
    row = 32 * npair * nao
    if left is None:
        return int(nrows), nao
    left -= own_bytes(nocc, nvir, nao)
    need = 2 * row + npair * lmax * (8 * nao + 32)
    if left < need:
        raise MemoryError(
            'the orbital gradient of (T) needs %s for the density of one occupied spinor in '
            'the AO basis and the integrals of one shell (%d AO functions) next to %s of its '
            'other arrays, %s are left of max_memory'
            % (amps._size(need), nao, amps._size(own_bytes(nocc, nvir, nao)),
               amps._size(max(left, 0))))
    nk = int(max(1, min(nrows, (left - npair * lmax * (8 * nao + 32)) // row - 1,
                        max(1, left // 2 // row))))
    nl = int(max(lmax, min(nao, (left - (nk + 1) * row) // (npair * (8 * nao + 32 * nk)))))
    return nk, nl


def g1_vvvv(mol, C2, g1, rows, plan):
    '''X[k, a] = sum_{b>c} sum_e g1[r, e, (bc)] <bc||ea> for the rows r =
    rows[k] of g1 (the density of <bc||er>), AO direct (see the module);
    C2[(spin, AO), a]: the virtual spinors.  ``plan`` = (rows per pass over
    the AO integrals, AOs per block of them), as vvvv_plan gives it.'''
    nao, v = mol.nao_nr(), C2.shape[1]
    npair = nao * (nao + 1) // 2
    ti, tj = np.tril_indices(nao)
    diag = ti * (ti + 1) // 2 + ti == np.arange(npair)
    Cs = [np.ascontiguousarray(C2[:nao]), np.ascontiguousarray(C2[nao:])]
    Cc = [np.conj(x) for x in Cs]
    C2h = np.ascontiguousarray(C2.conj().T)                     # [c, (s' la)]
    ao_loc = mol.ao_loc_nr()
    nk, nl = plan
    blocks, k0 = [], 0                      # ranges of shells of at most nl AOs
    while k0 < mol.nbas:
        k1 = k0 + 1
        while k1 < mol.nbas and ao_loc[k1 + 1] - ao_loc[k0] <= nl:
            k1 += 1
        blocks.append((k0, k1))
        k0 = k1
    X = np.empty((len(rows), v), dtype=np.complex128)
    for r0 in range(0, len(rows), nk):
        part = rows[r0:r0 + nk]
        n = len(part)
        # D[k, s', (mu nu), la], real and imaginary part
        D = [np.empty((n, 2, npair, nao)) for _ in range(2)]
        with timer('ogt.vvvv.ao'):
            for k, r in enumerate(part):
                Gk = np.asarray(g1[r])                          # [e, (b>c)]
                P = np.zeros((nao, nao * v), dtype=np.complex128)       # [mu, (nu, c)]
                for s in (0, 1):
                    R = blas.gemm(Cs[s], Gk)                    # [nu, (b>c)]
                    Ru = K.unpack_tril(R, v, 'apb').reshape(v, nao * v)         # [b, (nu, c)]
                    del R
                    blas.gemm(Cc[s], Ru, C=P, beta=1.0)
                    del Ru
                scratch.unmap(Gk)
                P = P.reshape(nao, nao, v)
                Ps = P[ti, tj]
                Ps += P[tj, ti]
                Ps[diag] *= 0.5
                del P
                Dk = blas.gemm(Ps, C2h).reshape(npair, 2, nao)          # [(mu nu), s', la]
                del Ps
                for s in (0, 1):
                    D[0][k, s], D[1][k, s] = Dk[:, s].real, Dk[:, s].imag
                del Dk
        Z = [np.zeros((2 * n, nao)) for _ in range(2)]          # [(k, s'), si]
        for k0, k1 in blocks:
            l0, l1 = int(ao_loc[k0]), int(ao_loc[k1])
            with timer('ogt.vvvv.eri'):
                M = mol.intor('int2e', aosym='s2ij',
                              shls_slice=(0, mol.nbas, 0, mol.nbas, k0, k1, 0, mol.nbas))
            M = M.reshape(npair * (l1 - l0), nao)               # [((mu nu), la), si]
            for x in (0, 1):
                A = np.ascontiguousarray(D[x][:, :, :, l0:l1]).reshape(2 * n, npair * (l1 - l0))
                blas.gemm(A, M, C=Z[x], beta=1.0)
                del A
            del M
        del D
        Zc = (Z[0] + 1j * Z[1]).reshape(n, 2 * nao)             # [k, (s' si)]
        X[r0:r0 + n] = blas.gemm(Zc, np.ascontiguousarray(C2))
    return X


def kernel(mycc, eris, t1, t2, terms, dens, verbose=None, parts=None, coulomb=None):
    '''Gnh[y, x], all spinors of ``eris`` ([occupied, virtual]): the (T)
    terms of the orbital gradient (see the module) for the amplitudes t1 and
    t2 (packed or not) and what zccsd.triples.kernel(terms='gradient')
    returned for them - ``terms`` = (s1, s2p, goo, gvv, dvo) and the
    Densities ``dens``.  Within max_memory of ``mycc``: own_bytes and the
    arrays of vvvv_plan; nothing of the amplitude step is planned or kept
    (its integrals are read as they are stored).'''
    log = logger.new_logger(mycc, verbose)
    cput0 = (logger.process_clock(), logger.perf_counter())
    if getattr(eris, 'residual', None) is None:
        eris.residual = amps.Residual(eris, plan=False)
    res = eris.residual
    o, v = res.nocc, res.nvir
    t1 = np.ascontiguousarray(t1, dtype=np.complex128)
    t2 = np.ascontiguousarray(t2, dtype=np.complex128)
    t2p = t2 if t2.ndim == 2 else K.pack_t2(t2)
    del t2
    kr = bool(res.kramers and res.symmetric(t1, t2p))
    plan = vvvv_plan(eris.mol, mycc.max_memory, o, v, o // 2 if kr else o)
    log.info('Orbital gradient of (T): %d occupied and %d virtual spinors; the density of '
             '<bc||ek> with <bc||ea> in the AO basis, %d occupied spinors and %d of %d AOs at '
             'a time', o, v, plan[0], plan[1], eris.mol.nao_nr())
    out = gradient(mycc, eris, t1, t2p, tuple(terms), dens.g1, dens.g2, dens.g3, plan, parts,
                   coulomb)
    log.timer('Orbital gradient of (T)', *cput0)
    return out


def gradient(mycc, eris, t1, t2p, terms, g1, g2, g3, plan, parts=None, coulomb=None):
    '''kernel() for t2 packed, the terms and the densities g1[k, e, (b>c)],
    g2[m, a, (j>k)], g3[(j>k), (b>c)]; the integrals are those of
    eris.residual, ``plan`` that of vvvv_plan.  ``parts``: a dict that gets
    Gnh by origin (PARTS).'''
    res = eris.residual
    o, v = res.nocc, res.nvir
    n1, npo, npv, n = o * v, K.npair(o), K.npair(v), o + v
    # blocks of occupied pairs whose rows are unpacked over the virtual pair
    pblocks = amps._blocks(npo, max(1, orbgrad.UNPACK_BYTES // (16 * v * v)))
    prow = np.array([r for r, _ in res.pair_rows])
    psgn = np.array([g for _, g in res.pair_rows])
    s1, s2p, goo, gvv, dvo = terms
    G = {k: np.zeros((n, n), dtype=np.complex128) for k in PARTS}
    oc, vi = slice(0, o), slice(o, n)

    def pair_sum(M):
        '''sum_k X[j,k] Y[i,k] as [j, i] from M[(jk), (ik)] on the packed pairs.'''
        return np.einsum('jk,ik,jik->ji', psgn, psgn, M[prow[:, None, :], prow[None, :, :]])

    with timer('ogt.pairs'):
        # ---- the sources with t
        X = G['sources']
        X[oc, vi] += s1
        M = blas.gemm(s2p, t2p.T)
        X[oc, oc] -= blas.gemm(s1, t1.T) + pair_sum(M)
        Lvv = blas.gemm(t1.T, s1)
        # ---- g3 with <bc||jk> = conj(<jk||bc>)
        for p0, p1 in pblocks:
            blas.gemm(g3, res.oovv_p[p0:p1].conj().T, M[:, p0:p1])
        G['g3'][oc, oc] -= pair_sum(M)
        del M
        T3 = np.zeros((v, v), dtype=np.complex128)
        Tvo2 = np.zeros((v, o), dtype=np.complex128)
        Tov5 = np.zeros((o, v), dtype=np.complex128)
        for p0, p1 in pblocks:
            nb = p1 - p0
            tu = K.unpack_tril(t2p[p0:p1], v, 'apb').reshape(v, nb * v)        # [b, (p c)]
            su = K.unpack_tril(s2p[p0:p1], v, 'apb').reshape(v, nb * v)        # [a, (p c)]
            blas.gemm(tu, su.T, C=Lvv, beta=1.0)
            del tu, su
            vu = K.unpack_tril(res.oovv_p[p0:p1], v, 'apb')                     # [b, p, c]
            np.conj(vu, out=vu)
            vu = vu.reshape(v, nb * v)                                          # <bc||p>
            gu = K.unpack_tril(g3[p0:p1], v, 'apb')                             # [a, p, c]
            blas.gemm(vu, gu.reshape(v, nb * v).T, C=T3, beta=1.0)
            # g3[c, a; p] with <ic||p> = conj(<p||ic>)
            blas.gemm(np.conj(res.ooov_en[:, :, p0:p1]).reshape(o, v * nb),
                      gu.reshape(v * nb, v), C=Tov5, alpha=-1.0, beta=1.0)
            # g2[i, c, p] with <ac||p>
            blas.gemm(vu, np.ascontiguousarray(g2[:, :, p0:p1].transpose(2, 1, 0))
                      .reshape(nb * v, o), C=Tvo2, beta=1.0)
            del vu, gu
        X[vi, vi] += Lvv
        G['g3'][vi, vi] += T3
        G['g3'][oc, vi] += Tov5
        G['g2'][vi, oc] += Tvo2

    with timer('ogt.small'):
        # ---- g2[m, a, (jk)], one m at a time, with <mi||jk>, <ja||kl> =
        # conj(<kl||ja>), <ma||ik> = conj(<ik||ma>), <mb||jk> = conj(<jk||mb>)
        X = G['g2']
        en = res.ooov_en                                                # [i, e, (m>n)] = <mn||ie>
        T2oo = np.zeros((o, o), dtype=np.complex128)
        for m in range(o):
            gm = g2[m]                                                  # [a, (jk)]
            Om = -psgn[:, m][:, None] * res.oooo_p[prow[:, m]]          # [i, (jk)] = <mi||jk>
            X[oc, vi] += blas.gemm(Om, gm.T)
            X[vi, vi] += blas.gemm(np.conj(en[m]), gm.T)
            # the conjugate of sum_a conj(<kl||ja>) g2[m, a, (kl)], all j
            blas.gemm(en.reshape(o, v * npo), np.conj(gm).reshape(v * npo, 1), C=T2oo[:, m:m + 1])
            gu = K.unpack_tril(np.ascontiguousarray(gm), o, 'pab')      # [a, j, k]
            X[oc, oc] -= blas.gemm(np.ascontiguousarray(gu.transpose(1, 2, 0)).reshape(o, n1),
                                   np.conj(res.ooov[:, :, m, :]).reshape(o, n1).T)
            del gu, Om
        X[oc, oc] += T2oo.conj()

    # ---- the pass over <ke||bc> and g1 by k (Kramers pairs: the unbarred k;
    # every term is a sum over k, completed by its time-reversed image)
    kr = bool(res.kramers and res.symmetric(t1, t2p))
    c6 = np.zeros((o, v), dtype=np.complex128)          # conjugates
    cvv = np.zeros((v, v), dtype=np.complex128)
    T1vo = np.zeros((v, o), dtype=np.complex128)
    T1vv = np.zeros((v, v), dtype=np.complex128)
    T1ov = np.zeros((o, v), dtype=np.complex128)
    T2ov = np.zeros((v, o), dtype=np.complex128)
    fstep = max(1, min(v, orbgrad.UNPACK_BYTES // (16 * npv)))
    ustep = max(1, min(v, orbgrad.UNPACK_BYTES // (16 * v * v)))
    with timer('ogt.ovvv'):
        for k in range(o // 2 if kr else o):
            Om = res.ovvv_p[k]                                          # [e, (b>c)]
            # g3[(bc); i, k] with <bc||ak> = -conj(<ka||bc>)
            Gk = np.conj(psgn[k][:, None] * g3[prow[k]])                # [i, (bc)]
            blas.gemm(Gk, Om.T, C=c6, beta=1.0)
            del Gk
            Ek = np.asarray(res.E[k * v:(k + 1) * v])                   # [a, (k' c)]: <kc||a k'>
            # g2[k, c; i, k'] with <kc||a k'>
            gu = K.unpack_tril(np.ascontiguousarray(g2[k]), o, 'pab')   # [c, i, k']
            blas.gemm(Ek, np.ascontiguousarray(gu.transpose(2, 0, 1)).reshape(n1, o),
                      C=T2ov, beta=1.0)
            del gu
            Gm = g1[k]                                                  # [e, (b>c)]
            # g1[(bc), a, k] with <bc||ik> = -conj(<ik||bc>)
            Ok = np.conj(psgn[k][:, None] * res.oovv_p[prow[k]])        # [i, (bc)]
            blas.gemm(Gm, Ok.T, C=T1vo, beta=1.0)
            del Ok
            # g1[(cd), b, k] with <cd||ak> = -conj(<ka||cd>), a few rows a at a time
            for a0 in range(0, v, fstep):
                a1 = min(v, a0 + fstep)
                blas.gemm(Gm, np.conj(Om[a0:a1]).T, C=T1vv[:, a0:a1], beta=1.0)
            # the rows e unpacked: g1[c, a; e, k] with <ic||ek> and with <bc||ek>
            Ec = np.conj(Ek).reshape(v, o, v)                           # [c, i, e] = <ic||ek>
            for e0 in range(0, v, ustep):
                e1 = min(v, e0 + ustep)
                ne = e1 - e0
                U1 = K.unpack_tril(np.ascontiguousarray(Gm[e0:e1]), v, 'pab').reshape(ne * v, v)
                W = np.ascontiguousarray(Ec[:, :, e0:e1].transpose(2, 0, 1)).reshape(ne * v, o)
                blas.gemm(W.T, U1, C=T1ov, alpha=-1.0, beta=1.0)
                UO = K.unpack_tril(np.ascontiguousarray(Om[e0:e1]), v, 'pab').reshape(ne * v, v)
                np.conj(U1, out=U1)
                blas.gemm(U1.T, UO, C=cvv, beta=1.0)
                del U1, W, UO
            del Ec, Ek
            scratch.unmap(Om)
            scratch.unmap(Gm)
    c6, T2ov, T1vv = -c6.conj(), -T2ov.T, T1vv - cvv.conj().T
    if kr:
        from zccsdt.zccsd.kramers import image
        c6, T2ov, T1vo, T1vv, T1ov = (x + image(x) for x in (c6, T2ov, T1vo, T1vv, T1ov))
    G['g3'][oc, vi] += c6
    G['g2'][oc, vi] += T2ov
    X = G['g1']
    X[vi, oc] += T1vo
    X[vi, vi] += T1vv
    X[oc, vi] += T1ov

    # ---- g1 by its rows e, with <bc||ei> = -conj(<ie||bc>) for all occupied i
    hv = v // 2 if kr else v
    Too = np.zeros((o, o), dtype=np.complex128)
    Ge = np.empty((o, npv), dtype=np.complex128)
    Oe = np.empty((o, npv), dtype=np.complex128)
    with timer('ogt.rows'):
        for e in range(hv):
            Ge[:] = g1[:, e, :]
            np.conj(res.ovvv_p[:, e, :], out=Oe)
            blas.gemm(Ge, Oe.T, C=Too, beta=1.0)
    del Ge, Oe
    # ---- g1 with <bc||ea>, in the AO basis
    with timer('ogt.vvvv'):
        rows = np.arange(o // 2 if kr else o)
        Tov = np.zeros((o, v), dtype=np.complex128)
        Tov[:rows.size] = -g1_vvvv(eris.mol, eris.mo_ao_spin[:, o:], g1, rows, plan)
    if kr:
        Tov += image(Tov)
        Too += image(Too)
    X[oc, vi] += Tov
    X[oc, oc] += Too

    # ---- the Fock matrix elements: f[a,i] with conj(dvo), e_i with -goo, e_a with gvv
    with timer('ogt.fock'):
        D = np.zeros((n, n), dtype=np.complex128)
        D[vi, oc] = dvo.conj()
        D[np.arange(o), np.arange(o)] = -goo.real
        D[o + np.arange(v), o + np.arange(v)] = gvv.real
        if coulomb is None:
            coulomb = orbgrad.Coulomb(eris.mol, mycc.max_memory)
        V = orbgrad.fock_matrices(coulomb, eris.mo_ao_spin, (D,))[0]
        # the Fock matrix as E(T) has it: the orbital energies and f[i,a] (what
        # the SCF left off the diagonal elsewhere is not in E(T); with it the
        # two blocks below would not be the multiples of e_q - e_p that they are)
        fock = np.diag(np.asarray(res.mo_e, dtype=np.complex128))
        fock[oc, vi], fock[vi, oc] = res.fov, res.fov.conj().T
        occ = (np.arange(n) < o).astype(np.float64)
        G['fock'] += fock @ D.T - D.T @ fock + V * (occ[None, :] - occ[:, None])
    if parts is not None:
        parts.update(G)
    return sum(G.values())
