'''One-particle density and dipole moment of the spinor CCSD and CCSD(T).

make_rdm1 and dip_moment as in the user's socutils code (cc/rdm_zccsd.py on
the branch claude/ccsd-lambda-socutils, 8965b9f, copied and changed): the
density of t and lambda with the expressions of pyscf.cc.gccsd_rdm, D[p,q] =
<q+ p>, Hermitized, in the basis of the spinors that the amplitudes refer
to (eris.mo_coeff: the Kramers pairs if the CCSD uses them) or in the 2c AO
basis.  It is the response density at fixed spinors; ``relaxed`` adds the
orbital relaxation of zccsd.zvector (CCSD without frozen spinors, small
molecules so far).  ``with_t``: lambda with the (T) terms and the (T) terms of
the density (zccsd.lam_t: full o^3 v^3 arrays, small molecules so far).

The dipole operator is that of the 2c AO basis, without a picture-change
transformation.
'''

import numpy as np
from pyscf.cc import ccsd, gccsd_rdm
from pyscf.data import nist


def make_rdm1(mycc, t1=None, t2=None, l1=None, l2=None, eris=None, ao_repr=False,
              with_t=False, relaxed=False):
    '''The one-particle density of the CCSD (``with_t``: CCSD(T)) of
    ``mycc``, in the basis of the active spinors eris.mo_coeff, or in the 2c
    AO basis with the frozen spinors (``ao_repr``).  lambda is solved for if
    it is not given.'''
    if eris is None:
        eris = mycc.eris if mycc.eris is not None else mycc.ao2mo(mycc.mo_coeff)
    t1 = mycc.t1 if t1 is None else t1
    t2 = mycc.t2 if t2 is None else t2
    if l1 is None or l2 is None:
        l1, l2 = mycc.solve_lambda(t1, t2, eris=eris, with_t=with_t)
    o = eris.nocc
    if relaxed:
        if with_t:
            raise NotImplementedError('orbital relaxation with the (T) terms')
        from zccsdt.zccsd import zvector
        dm1 = zvector.relaxed_rdm1_dense(mycc, l1=l1, l2=l2)[0]
    else:
        doo, dov, dvo, dvv = gccsd_rdm._gamma1_intermediates(mycc, t1, t2, l1, l2)
        if with_t:
            from zccsdt.zccsd import lam_t
            goo, gvv, gvo = lam_t.gamma1(t1, t2, eris)
            doo[np.diag_indices(o)] -= goo
            dvv[np.diag_indices(dvv.shape[0])] += gvv
            dvo += gvo
        dm1 = gccsd_rdm._make_rdm1(mycc, (doo, dov, dvo, dvv), with_frozen=False)
    if not ao_repr:
        return dm1
    C = eris.mo_coeff
    dm = C @ dm1 @ C.conj().T
    frozen = ~ccsd.get_frozen_mask(mycc) & (mycc.mo_occ > 0)
    Cf = mycc.mo_coeff[:, frozen]
    return dm + Cf @ Cf.conj().T


def dip_moment(mycc, t1=None, t2=None, l1=None, l2=None, eris=None, with_t=False,
               relaxed=False, unit='Debye'):
    '''The electric dipole moment from make_rdm1 and the dipole integrals of
    the 2c AO basis (origin at zero); ``unit``: 'Debye' or 'AU'.'''
    dm = make_rdm1(mycc, t1, t2, l1, l2, eris, ao_repr=True, with_t=with_t, relaxed=relaxed)
    mol = mycc.mol
    with mol.with_common_orig((0, 0, 0)):
        r = mol.intor('int1e_r_spinor')
    dip = (np.einsum('i,ix->x', mol.atom_charges(), mol.atom_coords())
           - np.einsum('xpq,qp->x', r, dm).real)
    if unit.upper() == 'DEBYE':
        dip = dip * nist.AU2DEBYE
    return dip
