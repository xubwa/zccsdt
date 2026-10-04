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
relaxed density, frozen occupied spinors (relaxed_rdm1_dense).  Not here
yet: the orbital gradient from intermediates in the layouts of zccsd.amps
(no full two-particle density, the vvvv part through the AO ladder), and
the (T) terms, for which the occupied-occupied and virtual-virtual
rotations do not drop out.
'''

import numpy as np
from pyscf.lib import logger

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


def response(mf, mo_coeff, mo_energy, nocc):
    '''The orbital Hessian R of the SCF ``mf`` as a function x[a,i] ->
    R(x)[a,i] for the spinors ``mo_coeff`` (2c AO basis, [occupied,
    virtual], canonical) with the orbital energies ``mo_energy``.'''
    Co, Cv = mo_coeff[:, :nocc], mo_coeff[:, nocc:]
    eai = mo_energy[nocc:, None] - mo_energy[:nocc]

    def R(x):
        dP = Cv @ x @ Co.conj().T
        dP = dP + dP.conj().T
        return eai * x + Cv.conj().T @ mf.get_veff(mf.mol, dP) @ Co

    return R, eai


def solve(mf, mo_coeff, mo_energy, nocc, gvo, tol=1e-9, max_cycle=50, verbose=None):
    '''z[a,i] with R(z) = gvo (gvo = -G[virtual, occupied]): Jacobi steps
    with the orbital energy differences, DIIS; one Fock build per step.
    Returns (converged, z).'''
    log = logger.new_logger(mf, verbose)
    R, eai = response(mf, mo_coeff, mo_energy, nocc)
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
