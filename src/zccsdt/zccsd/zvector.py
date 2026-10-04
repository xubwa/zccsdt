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

What is here: the orbital gradient from the full two-particle density and
all MO integrals (orbital_gradient: a reference for small molecules), the
response equation with AO-direct Fock builds (solve: for any size), the
relaxed density.  Not here yet: the orbital gradient from intermediates in
the layouts of zccsd.amps (no full two-particle density, the vvvv part
through the AO ladder), frozen spinors, and the (T) terms, for which the
occupied-occupied and virtual-virtual rotations do not drop out.
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


def relaxed_rdm1_dense(mycc, mf=None, l1=None, l2=None, tol=1e-9, verbose=None):
    '''The relaxed one-particle density of the CCSD of ``mycc`` (DirectZCCSD
    after kernel(); no frozen spinors) in the basis of the spinors
    mycc.eris.mo_coeff, with the orbital gradient from the full
    two-particle density of pyscf.cc.gccsd_rdm and all MO integrals: the
    reference for small molecules.  Returns (dm1 relaxed, dm1 unrelaxed, z).'''
    from pyscf.cc import gccsd_rdm
    mf = mycc._scf if mf is None else mf
    eris = mycc.eris
    C, o = eris.mo_coeff, eris.nocc
    if C.shape[1] != mf.mo_coeff.shape[1]:
        raise NotImplementedError('Z-vector equations with frozen spinors')
    if l1 is None or l2 is None:
        l1, l2 = mycc.solve_lambda()
    dm1 = gccsd_rdm.make_rdm1(mycc, mycc.t1, mycc.t2, l1, l2)
    dm2 = gccsd_rdm.make_rdm2(mycc, mycc.t1, mycc.t2, l1, l2)
    h = C.conj().T @ mf.get_hcore() @ C
    eri = mf.mol.intor('int2e_spinor')
    eri = np.einsum('mp,nq,mnls,lr,st->pqrt', C.conj(), C, eri, C.conj(), C, optimize=True)
    G = orbital_gradient(h, eri, dm1, dm2)
    conv, z = solve(mf, C, eris.mo_energy, o, -G[o:, :o], tol=tol, verbose=verbose)
    if not conv:
        logger.warn(mycc, 'Z-vector equations not converged')
    return relaxed_rdm1(dm1, z, o), dm1, z
