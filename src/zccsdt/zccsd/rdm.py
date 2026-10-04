'''One-particle density and dipole moment of the spinor CCSD and CCSD(T).

make_rdm1 and dip_moment as in the user's socutils code (cc/rdm_zccsd.py on
the branch claude/ccsd-lambda-socutils, 8965b9f, copied and changed): the
density of t and lambda with the expressions of pyscf.cc.gccsd_rdm, D[p,q] =
<q+ p>, Hermitized, evaluated from the packed amplitudes (gamma1), in the
basis of the spinors that the amplitudes refer to (eris.mo_coeff: the
Kramers pairs if the CCSD uses them) or in the 2c AO basis.  It is the
response density at fixed spinors; ``relaxed`` adds the
orbital relaxation of zccsd.zvector (CCSD without frozen spinors, small
molecules so far).  ``with_t``: lambda with the (T) terms and the (T) terms of
the density (zccsd.lam_t: full o^3 v^3 arrays, small molecules so far).

The dipole operator is that of the 2c AO basis with the picture-change
transformation of the X2C Hamiltonian (dipole_integrals; the socutils code
used the bare operator: H2O/cc-pVDZ 1.6e-6 a.u. in the dipole).
'''

import numpy as np
from pyscf.cc import ccsd, gccsd_rdm
from pyscf.data import nist


def gamma1(t1, t2, l1, l2, pair_bytes=1 << 30):
    '''(doo, dov, dvo, dvv) of pyscf.cc.gccsd_rdm._gamma1_intermediates from
    t2 and lambda2 packed over i > j and a > b: the two contractions of
    lambda2 with t2 as products of the packed arrays (the occupied one) and
    of blocks of pairs unpacked over a and b (the virtual one), on all
    threads; nothing of o^2 v^2 elements besides the two packed arrays.'''
    from zccsdt import blas
    from zccsdt.zccsd import kernels as K
    o, v = t1.shape
    t1 = np.ascontiguousarray(t1, dtype=np.complex128)
    l1 = np.ascontiguousarray(l1, dtype=np.complex128)
    t2p = K.pack_t2(np.ascontiguousarray(t2, dtype=np.complex128))
    l2p = K.pack_t2(np.ascontiguousarray(l2, dtype=np.complex128))
    npo = K.npair(o)
    # mij[i,j] = 1/2 sum_kcd lambda2[k,i,c,d] t2[k,j,c,d]: from the pairs with k
    M = blas.gemm(l2p, t2p.T)
    k = np.arange(o)
    prow, psgn = np.zeros((o, o), dtype=np.int64), np.zeros((o, o))
    for m in range(o):
        hi, lo = np.maximum(m, k), np.minimum(m, k)
        prow[m], psgn[m] = hi * (hi - 1) // 2 + lo, np.where(k < m, 1.0, -1.0) * (k != m)
    prow[np.arange(o), np.arange(o)] = 0
    mij = np.einsum('ik,jk,ijk->ij', psgn, psgn, M[prow[:, None, :], prow[None, :, :]])
    del M
    # mba[b,a] = 1/2 sum_klc lambda2[k,l,c,a] t2[k,l,c,b], in blocks of pairs
    mba = np.zeros((v, v), dtype=np.complex128)
    step = max(1, int(pair_bytes) // (32 * v * v))
    for p0 in range(0, npo, step):
        n = min(npo, p0 + step) - p0
        l2u = K.unpack_tril(l2p[p0:p0 + n], v, 'apb').reshape(v * n, v)
        t2u = K.unpack_tril(t2p[p0:p0 + n], v, 'apb').reshape(v * n, v)
        blas.gemm(t2u.T, l2u, C=mba, beta=1.0)
        del l2u, t2u
    lvv = blas.gemm(np.ascontiguousarray(t1.T), l1)     # [a, b] = sum_m t1[m,a] lambda1[m,b]
    doo = -blas.gemm(l1, t1.T) - mij
    dvv = lvv + mba
    dvo = K.t2p_matvec(t2p, l1, t1 - blas.gemm(mij.T, t1) - blas.gemm(t1, dvv.T))
    return doo, l1.copy(), np.ascontiguousarray(dvo.T), dvv


def make_rdm1(mycc, t1=None, t2=None, l1=None, l2=None, eris=None, ao_repr=False,
              with_t=False, relaxed=False):
    '''The one-particle density of the CCSD (``with_t``: CCSD(T)) of
    ``mycc``, in the basis of the active spinors eris.mo_coeff (``relaxed``:
    of all spinors, zvector.full_space), or in the 2c AO basis with the
    frozen spinors (``ao_repr``).  lambda is solved for if it is not given.'''
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
        dm1 = zvector.relaxed_rdm1_dense(mycc, l1=l1, l2=l2)[0]     # all spinors
        if not ao_repr:
            return dm1
        C = zvector.full_space(mycc)[0]
        return C @ dm1 @ C.conj().T
    else:
        doo, dov, dvo, dvv = gamma1(t1, t2, l1, l2)
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


def dipole_integrals(mycc, picture_change=True):
    '''The dipole operator r[x, p, q] in the 2c AO basis (origin at zero):
    with the picture-change transformation of the X2C Hamiltonian of the
    SCF if it has one (``with_x2c``), else the bare operator.'''
    mol = mycc.mol
    x2c = getattr(mycc._scf, 'with_x2c', None)
    with mol.with_common_orig((0, 0, 0)):
        if picture_change and x2c is not None:
            return np.asarray(x2c.picture_change(('int1e_r_spinor', 'int1e_sprsp_spinor')))
        return mol.intor('int1e_r_spinor')


def dip_moment(mycc, t1=None, t2=None, l1=None, l2=None, eris=None, with_t=False,
               relaxed=False, unit='Debye', picture_change=True):
    '''The electric dipole moment from make_rdm1 and dipole_integrals;
    ``unit``: 'Debye' or 'AU'.'''
    dm = make_rdm1(mycc, t1, t2, l1, l2, eris, ao_repr=True, with_t=with_t, relaxed=relaxed)
    mol = mycc.mol
    r = dipole_integrals(mycc, picture_change)
    dip = (np.einsum('i,ix->x', mol.atom_charges(), mol.atom_coords())
           - np.einsum('xpq,qp->x', r, dm).real)
    if unit.upper() == 'DEBYE':
        dip = dip * nist.AU2DEBYE
    return dip
