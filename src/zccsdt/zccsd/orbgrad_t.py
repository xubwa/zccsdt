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
kept: the integrals (ea|..) come from the integral transformation slab by
slab with the virtual e as the first index (zccsd.ao2mo.slabs), are
antisymmetrized and packed for one e, and meet the row e of every g1[i] - one
pass over g1, and the (vv|vv) part of an integral transformation: o v^4 / 2
multiply-adds and v N^4 for the half transformation.  With Kramers pairs the
sum over e runs over the unbarred half and is completed by the time-reversed
image.

The occupied spinors of the integrals may be more than those of the
amplitudes (``act``: frozen spinors, as occupied ones without amplitudes in
the code for all spinors, zvector.gradient_frozen): the densities are zero
where an index is frozen, and the integrals carry it.

The result is Gnh[y, x] for all spinors; with Gnh[a, i] of zccsd.orbgrad
added, the orbital gradient is (Gnh - Gnh^+) / 2.  Its occupied-occupied and
virtual-virtual blocks are (e_q - e_p) y[p, q] exactly - y the off-diagonal
one-particle density of (T), whatever the convergence of the amplitudes -,
and vanish between degenerate spinors (zvector.kernel divides).  For that
the Fock matrix is taken as E(T) has it - the orbital energies on its
diagonal and f[i,a] -, not with what the SCF left elsewhere off the diagonal.

Timers (zccsdt.timing): 'ogt.pairs', 'ogt.small', 'ogt.ovvv' (the pass over
g1 and <ke||bc> by k), 'ogt.vvvv' (the slabs and the pass by e), 'ogt.fock'.
'''

import numpy as np
from pyscf.lib import logger

from zccsdt import blas, memory, scratch
from zccsdt.timing import timer
from zccsdt.zccsd import amps, orbgrad
from zccsdt.zccsd import kernels as K

PARTS = ('sources', 'g1', 'g2', 'g3', 'fock')


def pad(nocc, act, terms, dens):
    '''(terms, g2, g3) for ``nocc`` occupied spinors of which those at
    ``act`` (ascending) have amplitudes: the sources and the one-particle
    terms of zccsd.triples.kernel - (s1, s2p, goo, gvv, dvo) - and its
    densities g2 and g3, zero where an occupied index is frozen.  (g1 is
    read through ``act`` where it is used.)'''
    from zccsdt.zccsd.zvector import active_pairs
    s1, s2p, goo, gvv, dvo = terms
    act = np.asarray(act)
    if act.size == nocc:
        return tuple(terms), dens.g2, dens.g3
    v = gvv.size
    rows = active_pairs(act)
    S1 = np.zeros((nocc, v), dtype=np.complex128)
    S1[act] = s1
    S2 = np.zeros((K.npair(nocc), s2p.shape[1]), dtype=np.complex128)
    S2[rows] = s2p
    Goo = np.zeros(nocc, dtype=np.complex128)
    Goo[act] = goo
    Dvo = np.zeros((v, nocc), dtype=np.complex128)
    Dvo[:, act] = dvo
    G2 = np.zeros((nocc, v, K.npair(nocc)), dtype=np.complex128)
    G2[np.ix_(act, np.arange(v), rows)] = dens.g2
    G3 = np.zeros((K.npair(nocc), dens.g3.shape[1]), dtype=np.complex128)
    G3[rows] = dens.g3
    return (S1, S2, Goo, gvv, Dvo), G2, G3


def own_bytes(nocc, nvir):
    '''Bytes of the arrays of gradient() besides the packed ones it is given
    and the slabs of the integral transformation: a matrix on the occupied
    pairs, the matrices on the spinors, and what the passes over g1 and
    <ke||bc> take for one occupied and one virtual index (rows that are
    unpacked: orbgrad.UNPACK_BYTES at a time).'''
    o, v = int(nocc), int(nvir)
    npo, npv, n = K.npair(o), K.npair(v), o + v
    return (16 * (npo * npo + 2 * o ** 3 + 12 * n * n + 4 * o * v * v + (3 * o + v) * npv)
            + 4 * orbgrad.UNPACK_BYTES)


def slab_plan(mol, max_memory, nocc, nvir):
    '''The ao2mo.Plan of the integrals (ea|..) with a virtual first index:
    slabs within what max_memory leaves now next to the arrays of gradient().
    MemoryError if one virtual spinor does not fit (16 v^3 bytes for its
    slab).'''
    from zccsdt.zccsd import ao2mo
    left = memory.budget(max_memory)
    if left is not None:
        left -= own_bytes(nocc, nvir)
    try:
        return ao2mo.plan_slabs(mol, nvir, nvir, left, sort_arrays=0)
    except MemoryError as err:
        raise MemoryError('%s (the integrals <bc||ea> of the orbital gradient of (T))'
                          % err) from None


def kernel(mycc, eris, t1, t2, terms, dens, act=None, verbose=None, parts=None, coulomb=None):
    '''Gnh[y, x], all spinors of ``eris`` ([occupied, virtual]): the (T)
    terms of the orbital gradient (see the module) for the amplitudes t1 and
    t2 (packed or not) and what zccsd.triples.kernel(terms='gradient')
    returned for them - ``terms`` = (s1, s2p, goo, gvv, dvo) and the
    Densities ``dens``.  ``act``: the positions of the occupied spinors of
    the amplitudes among those of ``eris`` (default: all; t1, t2 are then
    given for the occupied spinors of ``eris``, zero where one is frozen,
    while ``terms`` and ``dens`` are those of the active ones).  Within
    max_memory of ``mycc``: own_bytes and the slabs of the integral
    transformation; nothing of the amplitude step is planned or kept (its
    integrals are read as they are stored).'''
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
    act = np.arange(o) if act is None else np.asarray(act)
    terms, g2, g3 = pad(o, act, terms, dens)
    slabs = slab_plan(eris.mol, mycc.max_memory, o, v)
    log.info('Orbital gradient of (T): %d occupied (%d with amplitudes) and %d virtual '
             'spinors; the integrals <bc||ea> in slabs of %d virtual spinors', o, act.size, v,
             slabs.nket)
    out = gradient(mycc, eris, t1, t2p, terms, dens.g1, g2, g3, act, slabs, parts, coulomb)
    log.timer('Orbital gradient of (T)', *cput0)
    return out


def gradient(mycc, eris, t1, t2p, terms, g1, g2, g3, act, slabs, parts=None, coulomb=None):
    '''kernel() for t2 packed, the terms and the densities g2[m, a, (j>k)],
    g3[(j>k), (b>c)] of all occupied spinors of ``eris`` and g1[k, e, (b>c)]
    of those at ``act``; the integrals are those of eris.residual, ``slabs``
    the ao2mo.Plan of slab_plan.  ``parts``: a dict that gets Gnh by origin
    (PARTS).'''
    from zccsdt.zccsd import ao2mo
    res = eris.residual
    o, v = res.nocc, res.nvir
    n1, npo, npv, n = o * v, K.npair(o), K.npair(v), o + v
    # blocks of occupied pairs whose rows are unpacked over the virtual pair
    pblocks = amps._blocks(npo, max(1, orbgrad.UNPACK_BYTES // (16 * v * v)))
    prow = np.array([r for r, _ in res.pair_rows])
    psgn = np.array([g for _, g in res.pair_rows])
    s1, s2p, goo, gvv, dvo = terms
    pos = np.full(o, -1)
    pos[act] = np.arange(act.size)
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
        for m in act:
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

    # ---- the pass over <ke||bc> and g1 by k
    c6 = np.zeros((o, v), dtype=np.complex128)          # conjugates
    cvv = np.zeros((v, v), dtype=np.complex128)
    T1vo = np.zeros((v, o), dtype=np.complex128)
    T1vv = np.zeros((v, v), dtype=np.complex128)
    T1ov = np.zeros((o, v), dtype=np.complex128)
    T2ov = np.zeros((v, o), dtype=np.complex128)
    fstep = max(1, min(v, orbgrad.UNPACK_BYTES // (16 * npv)))
    ustep = max(1, min(v, orbgrad.UNPACK_BYTES // (16 * v * v)))
    with timer('ogt.ovvv'):
        for k in range(o):
            if pos[k] < 0:              # no density has a frozen k
                continue
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
            Gm = g1[pos[k]]                                             # [e, (b>c)]
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
    G['g3'][oc, vi] -= c6.conj()
    G['g2'][oc, vi] -= T2ov.T
    X = G['g1']
    X[vi, oc] += T1vo
    X[vi, vi] += T1vv - cvv.conj().T
    X[oc, vi] += T1ov

    # ---- g1 by its rows e: with <bc||ea> from the slabs of the integral
    # transformation, and with <bc||ei> = -conj(<ie||bc>) (all occupied i)
    kr = bool(res.kramers and res.symmetric(t1, t2p))
    hv = v // 2 if kr else v
    nact = act.size
    Tov = np.zeros((nact, v), dtype=np.complex128)
    Too = np.zeros((nact, o), dtype=np.complex128)
    nao = eris.nao_sph
    Cv = eris.mo_ao_spin[:, o:]
    H = np.empty((v, npv), dtype=np.complex128)
    Ge = np.empty((nact, npv), dtype=np.complex128)
    Oe = np.empty((o, npv), dtype=np.complex128)
    with timer('ogt.vvvv'):
        for e0, e1, g in ao2mo.slabs(eris.mol, Cv[:nao], Cv[nao:], (0, hv), (0, v), slabs):
            for e in range(e0, e1):
                # H[a, (b>c)] = (eb|ac) - (ec|ab) = conj(<bc||ea>)
                H[:] = 0.0
                K.asym_pack2(g[e - e0], 1.0, H, 'apb')
                np.conj(H, out=H)
                Ge[:] = g1[:, e, :]
                blas.gemm(Ge, H.T, C=Tov, alpha=-1.0, beta=1.0)
                np.conj(res.ovvv_p[:, e, :], out=Oe)
                blas.gemm(Ge, Oe.T, C=Too, beta=1.0)
        del g
    if kr:
        from zccsdt.zccsd.kramers import image
        Tov += image(Tov)
        Too += image(Too)
    X[act, o:] += Tov
    X[act, :o] += Too
    del H, Ge, Oe

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
