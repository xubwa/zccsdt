'''Spin-orbital CCSD amplitude equations (T1/T2), complex-capable.

Ported from socutils (cc/zccsd.py, commit ffbe691) by the same author; only the
residual function is kept here, the PySCF SCF plumbing is not.  Intermediates
follow Gauss and Stanton, J. Chem. Phys. 103, 3561 (1995), Table III, and use
<ab||ij> = <ij||ab>^* (Hermitian Hamiltonian), no real 8-fold symmetry.
'''

import numpy as np
from pyscf import lib
from zccsdt.cc import gintermediates as imd

einsum = lib.einsum


def update_amps(cc, t1, t2, eris, alg='sym'):

    nocc, nvir = t1.shape
    fock = eris.fock

    fov = fock[:nocc,nocc:]
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + cc.level_shift

    tau = imd.make_tau(t2, t1, t1)

    Fvv = imd.cc_Fvv(t1, t2, eris)
    Foo = imd.cc_Foo(t1, t2, eris)
    Fov = imd.cc_Fov(t1, t2, eris)
    Woooo = imd.cc_Woooo(t1, t2, eris)
    
    Wovvo = imd.cc_Wovvo(t1, t2, eris)

    # Move energy terms to the other side
    Fvv[np.diag_indices(nvir)] -= mo_e_v
    Foo[np.diag_indices(nocc)] -= mo_e_o

    # T1 equation
    t1new  =  einsum('ie,ae->ia', t1, Fvv)
    t1new += -einsum('ma,mi->ia', t1, Foo)
    t1new +=  einsum('imae,me->ia', t2, Fov)
    t1new += -einsum('nf,naif->ia', t1, eris.ovov)
    t1new += -0.5*einsum('imef,maef->ia', t2, eris.ovvv)
    t1new += -0.5*einsum('mnae,mnie->ia', t2, eris.ooov)
    t1new += fov.conj()

    # T2 equation
    Ftmp = Fvv - 0.5*einsum('mb,me->be', t1, Fov)
    tmp = einsum('ijae,be->ijab', t2, Ftmp)
    t2new = tmp - tmp.transpose(0,1,3,2)
    Ftmp = Foo + 0.5*einsum('je,me->mj', t1, Fov)
    tmp = einsum('imab,mj->ijab', t2, Ftmp)
    t2new -= tmp - tmp.transpose(1,0,2,3)
    t2new += np.asarray(eris.oovv).conj()
    t2new += 0.5*einsum('mnab,mnij->ijab', tau, Woooo)
    if (alg == 'old'):
        Wvvvv = imd.cc_Wvvvv(t1, t2, eris)
        t2new += 0.5*einsum('ijef,abef->ijab', tau, Wvvvv)
    elif (alg == 'new'):
        t2new += imd.update_t2_vvvv(t1, t2, tau, eris)
    elif (alg == 'sym'):
        t2new += imd.update_t2_vvvv_sym(t1, t2, tau, eris)
    tmp = einsum('imae,mbej->ijab', t2, Wovvo)
    tmp -= -einsum('ie,ma,mbje->ijab', t1, t1, eris.ovov)
    tmp = tmp - tmp.transpose(1,0,2,3)
    tmp = tmp - tmp.transpose(0,1,3,2)
    t2new += tmp
    tmp = einsum('ie,jeba->ijab', t1, np.array(eris.ovvv).conj())
    t2new += (tmp - tmp.transpose(1,0,2,3))
    tmp = einsum('ma,ijmb->ijab', t1, np.asarray(eris.ooov).conj())
    t2new -= (tmp - tmp.transpose(0,1,3,2))

    eia = mo_e_o[:,None] - mo_e_v
    eijab = lib.direct_sum('ia,jb->ijab', eia, eia)
    t1new /= eia
    t2new /= eijab

    return t1new, t2new


def energy(t1, t2, eris):
    '''CC correlation energy from T1, T2 (T3 does not contribute directly).'''
    nocc = t1.shape[0]
    fov = eris.fock[:nocc, nocc:]
    oovv = np.asarray(eris.oovv)
    e = einsum('ia,ia', fov, t1)
    e += 0.25 * einsum('ijab,ijab', t2, oovv)
    e += 0.5 * einsum('ia,jb,ijab', t1, t1, oovv)
    return e
