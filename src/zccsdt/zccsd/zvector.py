'''Orbital relaxation for the spinor CCSD: the Z-vector equations.

The CC energy with its Lagrangian is a function E(C) of the spinors through
the integrals, at fixed t and lambda:

    E = sum_pq h_pq D_qp + 1/2 sum_pqrs (pq|rs) d_pqrs,

D and d the one- and two-particle densities of t and lambda (reference
included, in the convention of pyscf.cc.gccsd_rdm: D[p,q] = <q+ p>, d[p,q,r,s]
= <p+ r+ s q>, both Hermitized).  The spinors are those of an SCF, so a
perturbation changes them; the first derivative of the energy needs that
change only through one linear equation (Handy and Schaefer's Z-vector).

Conventions.  A rotation C -> C exp(-kappa), kappa anti-Hermitian, changes
the energy by sum_xy kappa_xy G_yx in first order, with the orbital gradient

    G = h D - D h + 1/2 [ (yq|rs) d_xqrs - (px|rs) d_pyrs
                        + (pq|ys) d_pqxs - (pq|rx) d_pqry ]      as G[y, x],

anti-Hermitian for Hermitian densities.  Its occupied-occupied and
virtual-virtual blocks vanish for CCSD without frozen spinors (the energy
does not depend on rotations within the two spaces); kappa_ai = x_ai and
kappa_ia = -conj(x_ai) remain, and dE / d conj(x_ai) = -G_ai.  For the SCF
energy the same derivative is -f_ai, and the Fock matrix of the rotated
spinors is f_ai - R(x)_ai with the orbital Hessian

    R(x)_ai = (e_a - e_i) x_ai + [Cv^+ veff(Cv x Co^+ + Co x^+ Cv^+) Co]_ai,

real-linear in x (veff: Coulomb minus exchange of a density change, AO
direct).  With R(z) = -G_vo, a one-particle perturbation V gives

    dE/d eps = sum_pq V_pq Drel_qp,   Drel = D,  Drel[a,i] += z_ai,
                                                 Drel[i,a] += conj(z_ai):

the relaxed density (tests/test_zccsd_zvector.py: against the derivative
of E(SCF) + E_corr with the SCF repeated in the field).

Half of it vanishes.  Without Hermitizing the densities, the Lagrangian
L = E + sum l1 R1 + 1/4 sum l2 R2 (R the residuals of the amplitude
equations: linear in the integrals) changes by sum_xy kappa_xy Gnh[y, x], and

    G[a, i] = 1/2 (Gnh[a, i] - conj(Gnh[i, a])).

kappa_ai multiplies the excitation a+_a a_i, which commutes with T: Gnh[i, a]
is minus the derivative of L with respect to t1[i,a], zero for lambda that
solve their equations.  What is left, Gnh[a, i], is L with the integrals
one-index transformed by the de-excitation a+_i a_a: a bra index i takes
the values of a, a ket index a minus those of i.  With D_B the coefficient
of the block B of integrals in L (the two-particle density, not
Hermitized), it is a sum over the blocks with an occupied bra or a virtual
ket index (gradient_terms: from full arrays, the reference of the code for
the layouts of zccsd.amps).  Only the block <ia||bc> meets <ab||cd>.

What is here: the orbital gradient from the full two-particle density and
all MO integrals (orbital_gradient: a reference for small molecules) and
its de-excitation half term by term (gradient_terms: the same), the
response equation with AO-direct Fock builds (solve: for any size), the
relaxed density (kernel: with the orbital gradient of zccsd.orbgrad, in the
layouts of zccsd.amps and within max_memory; relaxed_rdm1_dense: from the
full arrays, also with frozen occupied spinors).  Not here yet: frozen
spinors without the full arrays, and the (T) terms, for which the
occupied-occupied and virtual-virtual rotations do not drop out.
'''

import numpy as np
from pyscf.lib import logger

from zccsdt import blas, memory
from zccsdt.cc.diis import DIIS


def orbital_gradient(h, eri, dm1, dm2):
    '''G[y, x] (see the module) from the MO integrals h[p,q], eri[p,q,r,s] =
    (pq|rs) and the densities dm1[p,q] = <q+ p>, dm2[p,q,r,s] = <p+ r+ s q>:
    full arrays of nmo^4 elements, a reference for small molecules.'''
    G = h @ dm1 - dm1 @ h
    G += 0.5 * (np.einsum('yqrs,xqrs->yx', eri, dm2) - np.einsum('pxrs,pyrs->yx', eri, dm2)
                + np.einsum('pqys,pqxs->yx', eri, dm2) - np.einsum('pqrx,pqry->yx', eri, dm2))
    return G


def gradient_terms(t1, t2, l1, l2, fock, g, nocc):
    '''Gnh[a, i] (see the module) of the CCSD Lagrangian by blocks of
    integrals, a dict with the keys 'fock', 'oooo', 'ooov', 'oovv', 'ovov',
    'ovvv', 'vvvv': the block B of the one-index transformed integrals with
    its density D_B, the coefficient of B in L as pyscf.cc.gccsd.update_amps
    writes the equations.  ``fock`` [p,q] and g[p,q,r,s] = <pq||rs> are
    full arrays over the spinors [occupied, virtual]: a reference for small
    molecules.  The sum of the terms is twice G[virtual, occupied] of
    orbital_gradient for converged t and lambda.'''
    o, n = nocc, fock.shape[0]
    oc, vi = slice(0, o), slice(o, None)
    E = np.einsum
    t1t1 = E('ia,jb->ijab', t1, t1)
    tau = t2 + t1t1 - t1t1.transpose(0, 1, 3, 2)
    taut = t2 + 0.5 * (t1t1 - t1t1.transpose(0, 1, 3, 2))
    m = E('ikac,kjcb->iajb', l2, t2)
    Y = E('klab,ijab->klij', tau, l2)
    N = E('ijca,ijce->ae', l2, t2)
    O = E('ijab,imab->jm', l2, t2)
    X = {}
    # <mn||ij>: D = Y / 8
    X['oooo'] = 0.25 * E('anij,knij->ak', g[vi, oc, oc, oc], Y)
    # <mn||ie> and <mb||ij>
    D = (-E('ia,ma,ne->mnie', l1, t1, t1) - 0.5 * E('ia,mnae->mnie', l1, t2)
         - 0.5 * E('im,ne->mnie', O, t1) + 0.25 * E('mnij,je->mnie', Y, t1)
         + E('ibme,nb->mnie', m, t1))
    Dovoo = -0.5 * E('ijab,ma->mbij', l2, t1)
    X['ooov'] = (E('anie,knie->ak', g[vi, oc, oc, vi], D) + E('maie,mkie->ak', g[oc, vi, oc, vi], D)
                 - E('mnik,mnia->ak', g[oc, oc, oc, oc], D)
                 + E('abij,kbij->ak', g[vi, vi, oc, oc], Dovoo))
    # <mn||ef>
    D = (0.25 * tau
         - 0.5 * E('ia,ie,mnaf->mnef', l1, t1, taut) - 0.5 * E('ia,ma,inef->mnef', l1, t1, taut)
         + E('ia,imae,nf->mnef', l1, t2, t1)
         - 0.25 * E('be,mnbf->mnef', N, taut) - 0.25 * E('be,mb,nf->mnef', N, t1, t1)
         - 0.25 * E('jm,jnef->mnef', O, taut) - 0.25 * E('jm,je,nf->mnef', O, t1, t1)
         + E('mnij,ijef->mnef', Y, tau) / 16
         - 0.5 * E('jbme,jnfb->mnef', m, t2) - E('jbme,jf,nb->mnef', m, t1, t1))
    X['oovv'] = (E('anef,knef->ak', g[vi, oc, vi, vi], D) + E('maef,mkef->ak', g[oc, vi, vi, vi], D)
                 - E('mnkf,mnaf->ak', g[oc, oc, oc, vi], D)
                 - E('mnek,mnea->ak', g[oc, oc, vi, oc], D))
    # <mb||je>
    D = -E('jb,me->mbje', l1, t1) - m.transpose(2, 1, 0, 3) + E('ijab,ie,ma->mbje', l2, t1, t1)
    X['ovov'] = (E('abje,kbje->ak', g[vi, vi, oc, vi], D)
                 - E('mbjk,mbja->ak', g[oc, vi, oc, oc], D))
    # <ma||ef> (the only block that meets <ab||cd>) and <ba||je>
    D = (-E('ia,ie,mf->maef', l1, t1, t1) - 0.5 * E('ia,imef->maef', l1, t2)
         - 0.5 * E('ae,mf->maef', N, t1) + 0.25 * E('ijab,ijef,mb->maef', l2, tau, t1)
         + E('jame,jf->maef', m, t1))
    Dvvov = 0.5 * E('ijab,ie->baje', l2, t1)
    X['ovvv'] = (E('abef,kbef->ak', g[vi, vi, vi, vi], D) - E('mbkf,mbaf->ak', g[oc, vi, oc, vi], D)
                 - E('mbek,mbea->ak', g[oc, vi, vi, oc], D)
                 - E('bcjk,bcja->ak', g[vi, vi, oc, oc], Dvvov))
    # <ab||ef>: D = sum_ij l2[ij,ab] tau[ij,ef] / 8
    X['vvvv'] = -0.25 * E('bckf,ijbc,ijaf->ak', g[vi, vi, oc, vi], l2, tau)
    # the Fock matrix, f = h + sum_k <pk||qk>: D[p,q] the coefficient of f[p,q]
    Df = np.zeros((n, n), dtype=np.complex128)
    Df[oc, vi] = (t1 + E('ia,imae->me', l1, t2) - E('ia,ie,ma->me', l1, t1, t1)
                  - 0.5 * E('be,mb->me', N, t1) - 0.5 * E('jm,je->me', O, t1))
    Df[vi, oc] = l1.T
    Df[vi, vi] = E('ia,ie->ae', l1, t1) + 0.5 * N
    Df[oc, oc] = -E('ja,ma->mj', l1, t1) - 0.5 * O.T
    X['fock'] = (fock[vi, :] @ Df[oc, :].T - (fock[:, oc].T @ Df[:, vi]).T
                 + E('paqk,pq->ak', g[:, vi, :, oc], Df))
    return X


def scf_veff(mf, max_memory=None, coulomb=None):
    '''A function dm -> J - K for a density change dm in the 2c AO basis of
    the SCF ``mf``, for the orbital Hessian: the bare Coulomb operator on the
    scalar AO integrals (orbgrad.Coulomb on the spin blocks: many times
    faster than the spinor integrals of a 2c SCF) if that is the
    two-electron part of the Fock operator of ``mf`` - it is compared with
    mf.get_veff on the density of the SCF -, else mf.get_veff itself.'''
    from zccsdt.zccsd import orbgrad
    mol = mf.mol
    slow = lambda dm: mf.get_veff(mol, dm)                          # noqa: E731
    dm = mf.make_rdm1()
    nao = mol.nao_nr()
    if getattr(dm, 'ndim', 0) != 2 or dm.shape[0] not in (2 * nao, mol.nao_2c()):
        return slow
    if coulomb is None:
        coulomb = orbgrad.Coulomb(mol, max_memory)
    if dm.shape[0] == 2 * nao and not hasattr(mf, 'with_x2c'):      # GHF: the (spin, AO) basis
        fast = coulomb.veff
    else:                                                           # j-adapted spinors
        c2 = np.vstack(mol.sph2spinor_coeff())
        fast = lambda d: c2.conj().T @ coulomb.veff(c2 @ d @ c2.conj().T) @ c2     # noqa: E731
    ref = slow(dm)
    return fast if abs(fast(dm) - ref).max() < 1e-8 * max(1.0, abs(ref).max()) else slow


def response(mf, mo_coeff, mo_energy, nocc, veff=None):
    '''The orbital Hessian R of the SCF ``mf`` as a function x[a,i] ->
    R(x)[a,i] for the spinors ``mo_coeff`` (2c AO basis, [occupied,
    virtual], canonical) with the orbital energies ``mo_energy``.  ``veff``:
    J - K of a density change (default: mf.get_veff; see scf_veff).'''
    Co, Cv = mo_coeff[:, :nocc], mo_coeff[:, nocc:]
    eai = mo_energy[nocc:, None] - mo_energy[:nocc]
    if veff is None:
        veff = lambda dm: mf.get_veff(mf.mol, dm)                   # noqa: E731

    def R(x):
        dP = Cv @ x @ Co.conj().T
        dP = dP + dP.conj().T
        return eai * x + Cv.conj().T @ veff(dP) @ Co

    return R, eai


def solve(mf, mo_coeff, mo_energy, nocc, gvo, tol=1e-9, max_cycle=50, verbose=None, veff=None):
    '''z[a,i] with R(z) = gvo (gvo = -G[virtual, occupied]): Jacobi steps
    with the orbital energy differences, DIIS; one Fock build per step
    (``veff``: see response).  Returns (converged, z).'''
    log = logger.new_logger(mf, verbose)
    R, eai = response(mf, mo_coeff, mo_energy, nocc, veff)
    z = gvo / eai
    adiis = DIIS(space=8, incore=True)
    conv = False
    try:
        for it in range(max_cycle):
            r = gvo - R(z)
            err = abs(r).max()
            log.debug('Z-vector cycle %d  max residual %.3g', it + 1, err)
            if err < tol:
                conv = True
                break
            z = adiis.update([z + r / eai])[0].reshape(gvo.shape)
    finally:
        adiis.close()
    log.info('Z-vector equations: %d Fock builds, max residual %.3g', it + 1, err)
    return conv, z


def relaxed_rdm1(dm1, z, nocc):
    '''The relaxed one-particle density: dm1 with z in its virtual-occupied
    block and conj(z)^T in its occupied-virtual block.'''
    out = np.array(dm1, dtype=np.complex128)
    out[nocc:, :nocc] += z
    out[:nocc, nocc:] += z.conj().T
    return out


def full_space(mycc):
    '''(C, mo_energy, nocc, frozen) for the Z-vector equations of ``mycc``:
    all spinors of its SCF in the 2c AO basis - the active ones as the
    amplitudes have them (eris.mo_coeff: Kramers pairs), the frozen ones as
    the SCF has them, in the order of the SCF - their energies, the number
    of occupied ones, and the indices of the frozen occupied ones.'''
    from pyscf.cc import ccsd
    eris = mycc.eris
    active = ccsd.get_frozen_mask(mycc)
    C = np.array(mycc.mo_coeff, dtype=np.complex128)
    C[:, active] = eris.mo_coeff
    mo_e = np.array(mycc._scf.mo_energy, dtype=np.float64)
    mo_e[active] = eris.mo_energy
    occupied = mycc.mo_occ > 0
    if (~active & ~occupied).any() or not occupied[:occupied.sum()].all():
        raise NotImplementedError('Z-vector equations with frozen virtual spinors')
    return C, mo_e, int(occupied.sum()), np.flatnonzero(~active & occupied)


def pad_amplitudes(nocc, nvir, act, x1, x2):
    '''(y1[i,a], y2[(i>j), (a>b)]) for all ``nocc`` occupied spinors from the
    amplitudes x1[i,a] and x2 ([i,j,a,b] or packed) of the active ones at the
    positions ``act`` (ascending), zero where an index is frozen.'''
    from zccsdt.zccsd import kernels as K
    y1 = np.zeros((nocc, nvir), dtype=np.complex128)
    y1[act] = x1
    x2 = np.ascontiguousarray(x2, dtype=np.complex128)
    y2 = np.zeros((K.npair(nocc), K.npair(nvir)), dtype=np.complex128)
    y2[active_pairs(act)] = x2 if x2.ndim == 2 else K.pack_t2(x2)
    return y1, y2


def active_pairs(act):
    '''The packed rows (i > j) of the pairs of active occupied spinors among
    those of all occupied spinors.'''
    from zccsdt.zccsd import kernels as K
    ia, ja = K.tril(act.size)
    return act[ia] * (act[ia] - 1) // 2 + act[ja]


def gradient_frozen(mycc, t1=None, t2=None, l1=None, l2=None, verbose=None, coulomb=None):
    '''(G[a, x], G[i, c]): the blocks of the orbital gradient that the
    Z-vector equations of a CCSD with frozen occupied spinors need - the
    virtual spinors a with all occupied x, frozen and active, in the order of
    full_space, and the active occupied i with the frozen c - for the
    amplitudes t1[i,a], t2[i,j,a,b] and lambda of the active spinors, without
    full arrays.

    A frozen spinor is an occupied one whose amplitudes are zero: with t and
    lambda padded by zeros, the Lagrangian of the frozen-core CCSD is that of
    the CCSD of all spinors, and its orbital gradient comes from the code for
    all spinors (the integrals are made once more, with the frozen spinors
    among the occupied ones):

      Gnh[a, x]   zccsd.orbgrad, for all occupied x;
      Gnh[c, a]   = -rho1[c, a]: the lambda1 equation of a frozen c, which no
                  amplitude makes vanish;
      Gnh[c, i]   = sum rho[c ...] t[i ...],   Gnh[i, c] = -sum R[c ...] lambda[i ...]:
                  a rotation among occupied spinors is a change of the
                  amplitudes, and only those with a frozen index are not
                  stationary,

    with R and rho the residuals of the amplitude equations and of the Lambda
    equations with one frozen index: one step of each on the padded
    amplitudes (zccsd.amps, zccsd.lam).  G = (Gnh - Gnh^+) / 2.

    Memory: the three phases are planned one by one within max_memory, for
    all occupied spinors - at least what a step of the CCSD without frozen
    spinors takes.  t2 and lambda2 of the CCSD object (the default) are
    packed and released while this runs, lambda2 is padded only when the
    Lambda step needs it, and if a plan does not fit next to the integrals
    of the active space, those are released (mycc.eris = None: mycc.ao2mo()
    makes them again when they are needed).'''
    from zccsdt.zccsd import kernels as K
    log = logger.new_logger(mycc, verbose)
    C, mo_e, o, frozen = full_space(mycc)
    act = np.setdiff1d(np.arange(o), frozen)
    v = C.shape[1] - o
    oa, rows_act = act.size, active_pairs(act)

    def packed(x):
        x = np.ascontiguousarray(x, dtype=np.complex128)
        return x if x.ndim == 2 else K.pack_t2(x)
    # as lam.kernel: the arrays of the CCSD object are packed and released while
    # this runs, and unpacked again at the end
    own_t, own_l = t2 is None, l2 is None
    T1, T2p = pad_amplitudes(o, v, act, mycc.t1 if t1 is None else t1,
                             mycc.t2 if own_t else t2)
    l1 = np.ascontiguousarray(mycc.l1 if l1 is None else l1, dtype=np.complex128)
    l2a = packed(mycc.l2 if own_l else l2)          # of the active spinors, until it is needed
    t2 = l2 = None
    if own_t:
        mycc.t2 = None
    if own_l:
        mycc.l2 = None
    try:
        return _gradient_frozen(mycc, C, mo_e, o, frozen, act, T1, T2p, l1, l2a, log,
                                coulomb)
    finally:
        if own_t:
            mycc.t2 = K.unpack_t2(np.ascontiguousarray(T2p[rows_act]), oa, v)
        if own_l:
            mycc.l2 = K.unpack_t2(l2a, oa, v)


def _gradient_frozen(mycc, C, mo_e, o, frozen, act, T1, T2p, l1, l2a, log, coulomb):
    '''gradient_frozen for the padded t and the packed lambda2 of the active
    spinors, l2a.'''
    import gc
    from zccsdt.zccsd import amps, lam, orbgrad
    from zccsdt.zccsd import kernels as K
    v = C.shape[1] - o
    eo, ev = mo_e[:o], mo_e[o:]
    # the CCSD of all spinors, as they are: no frozen ones, no pairing of its own
    full = mycc.__class__(mycc._scf, frozen=0, mo_coeff=C, mo_occ=mycc.mo_occ)
    full.verbose, full.max_memory, full.kramers = mycc.verbose, mycc.max_memory, False
    full.three_m, full.ao_ladder = mycc.three_m, mycc.ao_ladder
    # what the CCSD keeps for its steps is made again when it is needed
    if getattr(mycc.eris, 'residual', None) is not None:
        mycc.eris.residual.close()
        mycc.eris.residual = None
    lad = getattr(mycc.eris, 'aoladder', None)          # the AO operators are the same

    def guarded(f):
        '''f(), once more without the integrals of the active space if it
        does not fit next to them.'''
        try:
            return f()
        except MemoryError:
            if mycc.eris is None:
                raise
        # (outside the except block: what the failed attempt made is released
        # with its traceback)
        before = memory.resident()
        mycc.eris = None
        gc.collect()
        log.info('Orbital gradient with frozen spinors: the integrals of the active space '
                 'are released (max_memory): %.2f GB of %.2f GB in memory',
                 (before - memory.resident()) / 1e9, before / 1e9)
        return f()

    def setup():
        eris = full.ao2mo(C)
        if lad is not None:
            eris.aoladder.close()
            eris.aoladder = lad
        res = eris.residual = amps.Residual(eris, plan=False)
        # one step of the amplitude equations: nothing of its own to set aside
        res.replan(full.max_memory, 0, False, full.three_m)
        return eris, res

    try:
        eris, res = guarded(setup)
    except MemoryError as err:
        raise MemoryError('%s (the orbital gradient with frozen spinors is that of all %d '
                          'occupied spinors)' % (err, o)) from None
    log.info('Orbital gradient with frozen spinors: all %d occupied spinors; %s', o,
             res.describe())
    prow = np.array([r for r, _ in res.pair_rows])
    psgn = np.array([g for _, g in res.pair_rows])
    av, bv = K.tril(v)

    def with_frozen(x1new, x2new, x1, x2p, y2p):
        '''For the residuals r = D (new - old) of a Jacobi step, with a
        frozen first index c: r1[c, a], and sum_j sum_(a>b) r2[c,j,(ab)]
        y2[i,j,(ab)] as [c, i].'''
        r1 = (eo[frozen][:, None] - ev) * (x1new - x1)[frozen]
        out = np.zeros((frozen.size, o), dtype=np.complex128)
        cols = np.maximum(prow, 0).T                                    # [j, i]: the pair (i, j)
        for k, c in enumerate(frozen):
            rows = np.maximum(prow[c], 0)
            d = (eo[c] + eo)[:, None] - (ev[av] + ev[bv])[None, :]
            rc = psgn[c][:, None] * d * (x2new[rows] - x2p[rows])       # [j, (ab)]
            m = blas.gemm(np.ascontiguousarray(rc), y2p.T)              # [j, (pairs)]
            out[k] = np.einsum('ij,ji->i', psgn, m[np.arange(o)[:, None], cols])
        return r1, out

    t1n, t2n = res.update_packed(T1, T2p)
    L1, L2p = pad_amplitudes(o, v, act, l1, l2a)
    R1, RL = with_frozen(t1n, t2n, T1, T2p, L2p)
    del t1n, t2n
    # one step of the Lambda equations: its arrays, less the lambda2 that is
    # here already (the plan counts three packed arrays of its own)
    guarded(lambda: res.replan(full.max_memory, 0, False, full.three_m,
                               reserve=max(0, lam.reserve_bytes(o, v) - L2p.nbytes)))
    L = lam.Lambda(res, T1, T2p)
    try:
        l1n, l2n = L.update_packed(L1, L2p)
    finally:
        L.close()
    P1, PT = with_frozen(l1n, l2n, L1, L2p, T2p)
    del l1n, l2n, L
    Gci = blas.gemm(P1, T1.T) + PT                      # [c, i], all occupied i
    Gic = -(blas.gemm(R1, L1.T) + RL)                   # [c, i] holds Gnh[i, c]
    Gnh = guarded(lambda: orbgrad.kernel(full, eris, T1, T2p, L1, L2p, log,
                                         coulomb=coulomb))
    res.close()
    Gvo = 0.5 * Gnh
    Gvo[:, frozen] += 0.5 * P1.conj().T
    return Gvo, 0.5 * (Gic.T - Gci.conj().T)[act]


def kernel(mycc, t1=None, t2=None, l1=None, l2=None, eris=None, mf=None, tol=1e-9,
           verbose=None):
    '''(dm1 relaxed, dm1 unrelaxed, z[a,i]) for the CCSD of ``mycc`` in the
    basis of all spinors (full_space): the density of t and lambda from the
    packed amplitudes (zccsd.rdm), the orbital gradient of zccsd.orbgrad and
    the Z-vector equations (solve).  lambda: the arguments, else mycc.l1 and
    mycc.l2 for the amplitudes of mycc, else it is solved for.  Frozen
    occupied spinors: gradient_frozen for the gradient; the rotations between
    them and the active occupied ones as in relaxed_rdm1_dense.'''
    from pyscf.cc import gccsd_rdm
    from zccsdt.zccsd import orbgrad, rdm
    from zccsdt.zccsd import kernels as K
    mf = mycc._scf if mf is None else mf
    if eris is None and mycc.eris is None:
        mycc.ao2mo(mycc.mo_coeff)
    if l1 is None or l2 is None:
        # those of the CCSD object (as PySCF's make_rdm1 takes them), solved for
        # if there are none, if they are not those of the amplitudes given, or
        # if they are those of CCSD(T)
        if (t1 is not None or t2 is not None or getattr(mycc, 'l1', None) is None
                or getattr(mycc, 'l2', None) is None or getattr(mycc, '_l_with_t', False)):
            mycc.solve_lambda(t1, t2, eris=eris)
        l1 = l2 = None
    C, mo_e, o, frozen = full_space(mycc)
    oa, v = o - frozen.size, C.shape[1] - o         # active occupied spinors
    act = np.setdiff1d(np.arange(o), frozen)

    def packed(x):
        x = np.ascontiguousarray(x, dtype=np.complex128)
        return x if x.ndim == 2 else K.pack_t2(x)
    # t2 and lambda2 packed; those of the CCSD object are released while this
    # runs (lam.kernel does the same) and unpacked again at the end
    own_t, own_l = t2 is None, l2 is None
    t1 = np.ascontiguousarray(mycc.t1 if t1 is None else t1, dtype=np.complex128)
    l1 = np.ascontiguousarray(mycc.l1 if l1 is None else l1, dtype=np.complex128)
    t2p, l2p = packed(mycc.t2 if own_t else t2), packed(mycc.l2 if own_l else l2)
    t2 = l2 = None
    if own_t:
        mycc.t2 = None
    if own_l:
        mycc.l2 = None
    try:
        budget = memory.budget(mycc.max_memory)
        d1 = rdm.gamma1(t1, t2p, l1, l2p, **({} if budget is None else
                                             {'pair_bytes': max(1, min(1 << 30, budget // 4))}))
        out = dm1 = gccsd_rdm._make_rdm1(mycc, d1, with_frozen=bool(frozen.size))
        if not frozen.size:
            gvo = -0.5 * orbgrad.kernel(mycc, mycc.eris if eris is None else eris, t1, t2p,
                                        l1, l2p, verbose)
        else:
            Gvo, Gic = gradient_frozen(mycc, t1, t2p, l1, l2p, verbose)
    finally:
        if own_t:
            mycc.t2 = K.unpack_t2(t2p, oa, v)
        if own_l:
            mycc.l2 = K.unpack_t2(l2p, oa, v)
    t2p = l2p = None
    # the Fock operator of the orbital Hessian: after the gradient, which has
    # taken what memory there is
    veff = scf_veff(mf, mycc.max_memory)
    if frozen.size:
        y = Gic / (mo_e[frozen][None, :] - mo_e[act][:, None])
        Y = C[:, act] @ y @ C[:, frozen].conj().T
        gvo = -Gvo - C[:, o:].conj().T @ veff(Y + Y.conj().T) @ C[:, :o]
        out = np.array(dm1, dtype=np.complex128)
        out[np.ix_(act, frozen)] += y
        out[np.ix_(frozen, act)] += y.conj().T
    conv, z = solve(mf, C, mo_e, o, gvo, tol=tol, verbose=verbose, veff=veff)
    if not conv:
        logger.warn(mycc, 'Z-vector equations not converged')
    return relaxed_rdm1(out, z, o), dm1, z


def relaxed_rdm1_dense(mycc, mf=None, l1=None, l2=None, tol=1e-9, verbose=None):
    '''The relaxed one-particle density of the CCSD of ``mycc`` (DirectZCCSD
    after kernel()) in the basis of all spinors (full_space), with the
    orbital gradient from the full two-particle density of pyscf.cc
    .gccsd_rdm and all MO integrals: the reference for small molecules.
    Returns (dm1 relaxed, dm1 unrelaxed, z[a,i] for all occupied i).

    Frozen occupied spinors c.  The energy depends on the rotations between
    them and the active occupied i; in a perturbation these follow from the
    Fock matrix staying diagonal, f_ic = 0, not from an equation of their
    own: with y_ic = G_ic / (e_c - e_i) the density gets y in its (i, c) and
    conj(y) in its (c, i) elements, and the right-hand side of the Z-vector
    equations the Fock matrix of the density Ci y Cc^+ + Cc y^+ Ci^+.'''
    from pyscf.cc import gccsd_rdm
    mf = mycc._scf if mf is None else mf
    C, mo_e, o, frozen = full_space(mycc)
    if l1 is None or l2 is None:
        l1, l2 = mycc.solve_lambda()
    dm1 = gccsd_rdm.make_rdm1(mycc, mycc.t1, mycc.t2, l1, l2)
    dm2 = gccsd_rdm.make_rdm2(mycc, mycc.t1, mycc.t2, l1, l2)
    h = C.conj().T @ mf.get_hcore() @ C
    eri = mf.mol.intor('int2e_spinor')
    eri = np.einsum('mp,nq,mnls,lr,st->pqrt', C.conj(), C, eri, C.conj(), C, optimize=True)
    G = orbital_gradient(h, eri, dm1, dm2)
    gvo = -G[o:, :o]
    out = np.array(dm1, dtype=np.complex128)
    if frozen.size:
        act = np.setdiff1d(np.arange(o), frozen)
        y = G[np.ix_(act, frozen)] / (mo_e[frozen][None, :] - mo_e[act][:, None])
        Y = C[:, act] @ y @ C[:, frozen].conj().T
        gvo = gvo - C[:, o:].conj().T @ mf.get_veff(mf.mol, Y + Y.conj().T) @ C[:, :o]
        out[np.ix_(act, frozen)] += y
        out[np.ix_(frozen, act)] += y.conj().T
    conv, z = solve(mf, C, mo_e, o, gvo, tol=tol, verbose=verbose)
    if not conv:
        logger.warn(mycc, 'Z-vector equations not converged')
    return relaxed_rdm1(out, z, o), dm1, z
