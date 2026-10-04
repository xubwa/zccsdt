'''The (T) terms of the Lambda equations and of the one-particle density: a
reference on full arrays, for small molecules.

With E(T) in the energy, the Lambda equations get the derivative of E(T) with
respect to t1 and t2 as source terms: constants l1_t, l2_t that are added to
lambda1 and lambda2 after every step of the CCSD Lambda equations (zccsd.lam).
The expressions are those of the user's socutils (cc/lambda_t_zccsd.py on the
branch claude/ccsd-lambda-socutils, 8965b9f, copied here): pyscf.cc
.gccsd_t_lambda with the bare integrals of the source terms conjugated.
Complex spinors need that - lambda is conj(t) in first order, and PySCF's
expressions put the terms on the phase of t; for real integrals the two are
the same.  The one-particle density has (T) terms as well (pyscf.cc
.gccsd_t_rdm): the diagonals of its occupied and virtual blocks, from the
orbital energies in the denominators of E(T), and a virtual-occupied block
from the Fock matrix in the disconnected triples.

The connected and the disconnected triples are formed here as full o^3 v^3
arrays: this is the reference that an evaluation in the batches of
zccsd.triples has to reproduce.
'''

import numpy as np

from zccsdt.zccsd import kernels as K


def _blocks(eris):
    '''(ovvv, ooov, oovv, fock, o, v): <ia||bc>, <ij||ka>, <ij||ab> as full
    arrays from the integrals of DirectZCCSD.'''
    from zccsdt.zccsd.zccsd_direct import get_ovvv
    o = eris.nocc
    v = eris.fock.shape[0] - o
    oovv = getattr(eris, 'oovv', None)
    if oovv is None:
        oovv = K.unpack_t2(np.ascontiguousarray(eris.oovv_p), o, v)
    return get_ovvv(eris), np.asarray(eris.ooov), np.asarray(oovv), eris.fock, o, v


def _antisymmetrize(x):
    '''P(a/bc) P(i/jk) of x[i,j,k,a,b,c].'''
    x = x - x.transpose(0, 1, 2, 4, 3, 5) - x.transpose(0, 1, 2, 5, 4, 3)
    return x - x.transpose(1, 0, 2, 3, 4, 5) - x.transpose(2, 1, 0, 3, 4, 5)


def triples(t1, t2, eris):
    '''(t3c, t3d, eia): the connected and the disconnected triples
    [i,j,k,a,b,c] of (T), divided by their denominators.'''
    ovvv, ooov, oovv, fock, o, v = _blocks(eris)
    bcei = ovvv.conj().transpose(3, 2, 1, 0)            # <bc||ei>
    majk = ooov.conj().transpose(2, 3, 0, 1)            # <ma||jk>
    bcjk = oovv.conj().transpose(2, 3, 0, 1)            # <bc||jk>
    mo_e = eris.mo_energy
    eia = mo_e[:o, None] - mo_e[o:]
    d3 = (eia[:, None, None, :, None, None] + eia[None, :, None, None, :, None]
          + eia[None, None, :, None, None, :])
    t3c = _antisymmetrize(np.einsum('jkae,bcei->ijkabc', t2, bcei)
                          - np.einsum('imbc,majk->ijkabc', t2, majk)) / d3
    t3d = _antisymmetrize(np.einsum('ia,bcjk->ijkabc', t1, bcjk)
                          + np.einsum('ai,jkbc->ijkabc', fock[o:, :o], t2)) / d3
    return t3c, t3d, eia


def energy(t1, t2, eris):
    '''E(T) from the full triples (the check of this module against
    zccsd.triples).'''
    t3c, t3d, eia = triples(t1, t2, eris)
    d3 = (eia[:, None, None, :, None, None] + eia[None, :, None, None, :, None]
          + eia[None, None, :, None, None, :])
    return (np.einsum('ijkabc,ijkabc,ijkabc', (t3c + t3d).conj(), d3, t3c) / 36).real


def sources(t1, t2, eris):
    '''(l1_t[i,a], l2_t[i,j,a,b]): the (T) terms of the Lambda equations,
    divided by the denominators of lambda1 and lambda2.'''
    ovvv, ooov, oovv, fock, o, v = _blocks(eris)
    t3c, t3d, eia = triples(t1, t2, eris)
    l1_t = 0.25 * np.einsum('ijkabc,jkbc->ia', t3c.conj(), oovv.conj()) / eia
    m3 = (2 * t3c + t3d).conj()
    tmp = 0.5 * np.einsum('ijkaef,kbfe->ijab', m3, ovvv.conj())
    l2_t = tmp - tmp.transpose(0, 1, 3, 2)
    tmp = 0.5 * np.einsum('imnabc,mnjc->ijab', m3, ooov.conj())
    l2_t -= tmp - tmp.transpose(1, 0, 2, 3)
    l2_t += np.einsum('kc,ijkabc->ijab', fock[:o, o:].conj(), t3c.conj())
    l2_t /= eia[:, None, :, None] + eia[None, :, None, :]
    return l1_t, l2_t


def gamma1(t1, t2, eris):
    '''(goo, gvv, dvo): the (T) terms of the one-particle density in the
    convention of pyscf.cc.gccsd_rdm._gamma1_intermediates - the diagonals
    of doo (subtracted) and dvv (added), and what is added to dvo[a,i].'''
    t3c, t3d, _ = triples(t1, t2, eris)
    goo = np.einsum('iklabc,iklabc->i', (t3c + t3d).conj(), t3c) / 12
    gvv = np.einsum('ijkacd,ijkacd->a', t3c + t3d, t3c.conj()) / 12
    dvo = 0.25 * np.einsum('ijab,ijkabc->ck', t2.conj(), t3c)
    return goo, gvv, dvo
