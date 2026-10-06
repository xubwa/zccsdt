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
layouts of zccsd.amps and within max_memory, also with frozen occupied and
frozen virtual spinors - zccsd.orbgrad_frozen -; relaxed_rdm1_dense: from the
full arrays).

CCSD(T) (with_t).  E(T) is added to the Lagrangian, a function of t, of the
integrals and of the orbital energies; lambda then solves the equations with
the derivative of E(T) with respect to t as source terms (zccsd.lam_t).  The
gradient has three more parts (gradient_t and gradient_nh from full arrays,
zccsd.orbgrad_t in the layouts of the amplitude step):

- E(T) with its densities, the coefficients of <bc||ei>, <ma||jk>, <bc||jk>,
  f[a,i] and the orbital energies in its first-order change (one pass over
  the triples makes them with the sources: zccsd.triples.kernel);
- the excitation half Gnh[i, a], which is the source of lambda1 now;
- the rotations among occupied and among virtual spinors.  They change t in
  the Lagrangian (contractions of the sources with t) and the integrals in
  E(T), and they do not drop out: E(T) has the orbital energies in its
  denominators.  As for frozen spinors, they follow from the Fock matrix
  staying diagonal: y_pq = G_pq / (e_q - e_p) goes into the density and its
  Fock matrix to the right-hand side of the Z-vector equations.  G_pq is
  (e_q - e_p) times the off-diagonal one-particle density of (T) - the
  products of the triples, the derivative of E(T) with respect to an
  off-diagonal element of the Fock matrix - exactly, for any amplitudes: the
  division is safe down to the rounding of G, and between degenerate
  spinors, where G vanishes, there is no rotation (DEGENERATE).
'''

import numpy as np
from pyscf.lib import logger

from zccsdt import memory
from zccsdt.cc.diis import DIIS
from zccsdt.timing import timer

# Spinors closer in energy than this are degenerate: no rotation between them
# (orbital gradients from full arrays, which carry the convergence of t and
# lambda; and from the layouts of zccsd.orbgrad_t, exact up to rounding)
DEGENERATE_DENSE = 1e-6
DEGENERATE = 1e-9
# kernel() warns if the SCF left more than this in the occupied-virtual block
# of its Fock matrix (H2O/cc-pVDZ: 3e-8 there moved the dipole by 5e-8)
SCF_GRADIENT = 1e-7


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


def gradient_nh(fock, g, d2, d1, nocc):
    '''Gnh[y, x] (all blocks) of a real function of the integrals that changes
    in first order by Re sum (d2[p,q,r,s] d<pq||rs> + d1[p,q] df[p,q]): full
    arrays over the spinors [occupied, virtual], zero where the function does
    not depend on an integral; f = h + sum_k <pk||qk> over the ``nocc``
    occupied spinors.  A rotation takes a bra index y of an integral to x and
    a ket index x to minus y, and it rotates the occupied spinors of the Fock
    operator (the term V).  The orbital gradient is (Gnh - Gnh^+) / 2.  (For
    E(T), ``fock`` is t_fock: the Fock matrix as E(T) has it.)'''
    E = np.einsum
    n = fock.shape[0]
    bra = E('xqrs,yqrs->xy', d2, g) + E('pxrs,pyrs->xy', d2, g)
    ket = E('pqys,pqxs->xy', d2, g) + E('pqry,pqrx->xy', d2, g)
    occ = (np.arange(n) < nocc).astype(np.float64)
    V = E('pq,pyqx->yx', d1, g)
    return (bra - ket).T + fock @ d1.T - d1.T @ fock + V * (occ[None, :] - occ[:, None])


def t_fock(fock, nocc):
    '''The Fock matrix as E(T) has it: the orbital energies on the diagonal
    and the occupied-virtual blocks.  What an SCF leaves off the diagonal of
    the other two blocks (1e-8 at its convergence) is not in E(T), and with
    it the occupied-occupied and virtual-virtual blocks of the gradient of
    (T) would not be exact multiples of e_q - e_p.'''
    out = np.diag(np.asarray(fock, dtype=np.complex128).diagonal().real).astype(np.complex128)
    out[:nocc, nocc:], out[nocc:, :nocc] = fock[:nocc, nocc:], fock[nocc:, :nocc]
    return out


def t_densities(t1, t2, eris, nmo=None, nocc=None, act=None, vir=None):
    '''(d2[p,q,r,s], d1[p,q]): the densities of (T) (lam_t.densities and the
    diagonals of lam_t.gamma1) as gradient_nh takes them, full arrays over
    ``nmo`` spinors of which the first ``nocc`` are occupied and those at
    ``act`` and ``vir`` are the occupied and the virtual spinors of the
    amplitudes (default: the spinors of ``eris``, none frozen).  The
    reference for small molecules.'''
    from zccsdt.zccsd import lam_t
    o, v = t1.shape
    nocc = o if nocc is None else nocc
    nmo = nocc + v if nmo is None else nmo
    act = np.arange(o) if act is None else np.asarray(act)
    vir = np.arange(nocc, nmo) if vir is None else np.asarray(vir)
    g1, g2, g3, df = lam_t.densities(t1, t2, eris)
    goo, gvv, _ = lam_t.gamma1(t1, t2, eris)
    d2 = np.zeros((nmo,) * 4, dtype=np.complex128)
    d2[np.ix_(vir, vir, vir, act)] = g1
    d2[np.ix_(act, vir, act, act)] = g2
    d2[np.ix_(vir, vir, act, act)] = g3
    d1 = np.zeros((nmo, nmo), dtype=np.complex128)
    d1[np.ix_(vir, act)] = df
    d1[act, act] = -goo.real
    d1[vir, vir] = gvv.real
    return d2, d1


def gradient_t(t1, t2, fock, g, nocc):
    '''What (T) adds to Gnh, all blocks [y, x], from full arrays (the
    reference of zccsd.orbgrad_t; no frozen spinors):

    - E(T) itself, from its densities (gradient_nh);
    - the CCSD Lagrangian, whose derivative with respect to t no longer
      vanishes: lambda of CCSD(T) solves dL/dt = -s with the sources s1[i,a],
      s2[i,j,a,b] of (T) (lam_t.sources before the division).  A rotation
      among occupied or among virtual spinors is a change of t (and of
      lambda, which multiplies the residuals), and the excitation a+_a a_i is
      the derivative with respect to t1:

        Gnh[i, a] += s1[i,a]
        Gnh[j, i] -= sum_a s1[j,a] t1[i,a] + 1/2 sum_kab s2[j,k,a,b] t2[i,k,a,b]
        Gnh[b, a] += sum_i s1[i,a] t1[i,b] + 1/2 sum_ijc s2[i,j,a,c] t2[i,j,b,c].

    With gradient_terms for lambda of CCSD(T) in Gnh[virtual, occupied], the
    sum is Gnh of CCSD(T): its occupied-occupied and virtual-virtual blocks
    do not vanish, but G[p, q] = (e_q - e_p) y[p, q] exactly, with y the
    off-diagonal one-particle density of (T).'''
    import types
    from zccsdt.zccsd import lam_t
    o = nocc
    oc, vi = slice(0, o), slice(o, None)
    mo_e = fock.diagonal().real
    eris = types.SimpleNamespace(nocc=o, fock=fock, mo_energy=mo_e, ovvv=g[oc, vi, vi, vi],
                                 ooov=g[oc, oc, oc, vi], oovv=g[oc, oc, vi, vi])
    Gnh = gradient_nh(t_fock(fock, o), g, *t_densities(t1, t2, eris), o)
    s1, s2 = lam_t.sources(t1, t2, eris)
    eia = mo_e[:o, None] - mo_e[o:]
    s1 = s1 * eia
    s2 = s2 * (eia[:, None, :, None] + eia[None, :, None, :])
    Gnh[oc, vi] += s1
    Gnh[oc, oc] -= s1 @ t1.T + 0.5 * np.einsum('jkab,ikab->ji', s2, t2)
    Gnh[vi, vi] += np.einsum('ia,ib->ba', s1, t1) + 0.5 * np.einsum('ijac,ijbc->ba', s2, t2)
    return Gnh


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
    of occupied ones, and the indices of the frozen occupied ones.  (The
    frozen virtual ones: frozen_virtuals.)'''
    from pyscf.cc import ccsd
    eris = mycc.eris
    active = ccsd.get_frozen_mask(mycc)
    C = np.array(mycc.mo_coeff, dtype=np.complex128)
    C[:, active] = eris.mo_coeff
    mo_e = np.array(mycc._scf.mo_energy, dtype=np.float64)
    mo_e[active] = eris.mo_energy
    occupied = mycc.mo_occ > 0
    if not occupied[:occupied.sum()].all():
        raise NotImplementedError('occupied spinors after virtual ones')
    return C, mo_e, int(occupied.sum()), np.flatnonzero(~active & occupied)


def frozen_virtuals(mycc):
    '''The indices of the frozen virtual spinors among all spinors (the
    order of full_space).'''
    from pyscf.cc import ccsd
    return np.flatnonzero(~ccsd.get_frozen_mask(mycc) & ~(mycc.mo_occ > 0))


def kernel(mycc, t1=None, t2=None, l1=None, l2=None, eris=None, mf=None, tol=1e-9,
           verbose=None, with_t=False):
    '''(dm1 relaxed, dm1 unrelaxed, z[a,i]) for the CCSD of ``mycc`` in the
    basis of all spinors (full_space): the density of t and lambda from the
    packed amplitudes (zccsd.rdm), the orbital gradient of zccsd.orbgrad and
    the Z-vector equations (solve).  lambda: the arguments, else mycc.l1 and
    mycc.l2 for the amplitudes of mycc, else it is solved for.

    Frozen spinors, occupied or virtual: zccsd.orbgrad is the gradient among
    the active spinors; the blocks between a frozen and an active spinor are
    those of zccsd.orbgrad_frozen (the integrals with one frozen index) and
    the Fock matrix of the correlation part of the density, and between a
    frozen virtual and a frozen occupied one that Fock matrix alone.  The
    rotations between frozen and active spinors of the same kind follow from
    the Fock matrix staying diagonal, as in relaxed_rdm1_dense, and the
    Z-vector equations are those of all virtual with all occupied spinors.

    ``with_t``: for CCSD(T).  lambda is that of CCSD(T) - of the CCSD object
    if it has it, else solved for; l1 and l2 are not taken -, the density has
    its (T) terms, and the gradient those of zccsd.orbgrad_t (and of
    zccsd.orbgrad_frozen with the densities of (T)), from the
    densities of (T): one pass over the triples gives them together with the
    sources of lambda (solve_lambda with with_t='gradient'; if lambda is
    there without them, the pass is done here).  The occupied-occupied and
    virtual-virtual blocks of the gradient do not vanish then: the rotations
    among occupied and among virtual spinors follow from the Fock matrix
    staying diagonal, y_pq = G_pq / (e_q - e_p), as those between frozen and
    active spinors do; none between spinors closer than DEGENERATE.'''
    from pyscf.cc import gccsd_rdm
    from zccsdt.zccsd import orbgrad, rdm
    from zccsdt.zccsd import kernels as K
    mf = mycc._scf if mf is None else mf
    if eris is None and mycc.eris is None:
        mycc.ao2mo(mycc.mo_coeff)
    C, mo_e, o, frozen = full_space(mycc)
    fvir = frozen_virtuals(mycc)
    # the equations assume the spinors of a converged SCF: what is left of its
    # orbital gradient is in the relaxed density in first order
    fock = np.asarray((mycc.eris if eris is None else eris).fock)
    nact = o - frozen.size
    fov = abs(fock[:nact, nact:]).max() if nact and fock.shape[0] > nact else 0.0
    if fov > SCF_GRADIENT:
        logger.warn(mycc, 'The Fock matrix of the SCF has occupied-virtual elements of %.1e: '
                    'the relaxed density carries an error of that size. Converge the SCF '
                    'further (PySCF: conv_tol_grad; its default is sqrt(conv_tol)).', fov)
    tg = None
    if with_t:
        own = (t1 is None and t2 is None and getattr(mycc, 'l1', None) is not None
               and getattr(mycc, 'l2', None) is not None and getattr(mycc, '_l_with_t', False))
        if not own:
            mycc.solve_lambda(t1, t2, eris=eris, with_t='gradient')
        tg = getattr(mycc, '_t_gradient', None)
        l1 = l2 = None
    elif l1 is None or l2 is None:
        # those of the CCSD object (as PySCF's make_rdm1 takes them), solved for
        # if there are none, if they are not those of the amplitudes given, or
        # if they are those of CCSD(T)
        if (t1 is not None or t2 is not None or getattr(mycc, 'l1', None) is None
                or getattr(mycc, 'l2', None) is None or getattr(mycc, '_l_with_t', False)):
            mycc.solve_lambda(t1, t2, eris=eris)
        l1 = l2 = None
    n = C.shape[1]
    act = np.setdiff1d(np.arange(o), frozen)        # the active spinors among all
    vir = np.setdiff1d(np.arange(o, n), fvir)
    oa, v = act.size, vir.size
    both = np.concatenate((act, vir))
    held_out = np.concatenate((frozen, fvir))

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
    Gt = X = None
    try:
        budget = memory.budget(mycc.max_memory)
        with timer('zv.density'):
            d1 = rdm.gamma1(t1, t2p, l1, l2p, **({} if budget is None else {
                'pair_bytes': max(1, min(1 << 30, budget // 4))}))
        held = []
        if with_t:
            with timer('zv.triples'):
                terms, dens = _t_terms(mycc, mycc.eris if eris is None else eris, t1, t2p, tg)
            doo, dov, dvo, dvv = d1
            doo[np.diag_indices(oa)] -= terms[2]
            dvv[np.diag_indices(v)] += terms[3]
            d1 = (doo, dov, dvo + terms[4], dvv)
            # the sources and the densities go when the (T) terms of the gradient
            # are made; the terms of the one-particle density stay with the CCSD
            # object
            held, small = [terms, dens], terms[2:]
            del terms, dens, tg
        dm1 = gccsd_rdm._make_rdm1(mycc, d1, with_frozen=bool(held_out.size))

        def release():
            if held:
                held.pop().close()
                held.clear()
                mycc._drop_t_terms()
                mycc._t_density = small
        try:
            if with_t:
                # first: nothing of it is needed for the Lagrangian's own part,
                # which takes the memory of an amplitude step
                from zccsdt.zccsd import orbgrad_t
                with timer('zv.orbgrad_t'):
                    Gt = orbgrad_t.kernel(mycc, mycc.eris if eris is None else eris, t1,
                                          t2p, *held, verbose=verbose)
            if held_out.size:
                # the integrals with a frozen index meet the densities of (T) too
                from zccsdt.zccsd import orbgrad_frozen
                with timer('zv.orbgrad_frozen'):
                    X = orbgrad_frozen.kernel(
                        mycc, mycc.eris if eris is None else eris, t1, t2p, l1, l2p,
                        np.vstack(mf.mol.sph2spinor_coeff()) @ C[:, held_out],
                        held[1] if with_t else None, mo_e[held_out], verbose)
            release()
            with timer('zv.orbgrad'):
                Gnh = orbgrad.kernel(mycc, mycc.eris if eris is None else eris, t1, t2p,
                                     l1, l2p, verbose)
        finally:
            release()
    finally:
        if own_t:
            mycc.t2 = K.unpack_t2(t2p, oa, v)
        if own_l:
            mycc.l2 = K.unpack_t2(l2p, oa, v)
    t2p = l2p = None
    # the gradient among the active spinors, [occupied, virtual]
    if with_t:
        Gt[oa:, :oa] += Gnh
        Ga = 0.5 * (Gt - Gt.conj().T)
        del Gt
    else:
        Ga = np.zeros((oa + v,) * 2, dtype=np.complex128)
        Ga[oa:, :oa], Ga[:oa, oa:] = 0.5 * Gnh, -0.5 * Gnh.conj().T
    G = np.zeros((n, n), dtype=np.complex128)
    G[np.ix_(both, both)] = Ga
    # the Coulomb operator: after the gradient, which has taken what memory
    # there is
    with timer('zv.veff'):
        coulomb = orbgrad.Coulomb(mf.mol, mycc.max_memory)
        veff = scf_veff(mf, mycc.max_memory, coulomb)
    if held_out.size:
        # between frozen and active spinors: the integrals with a frozen index
        # and the Fock matrix of the correlation part of the density
        with timer('zv.veff'):
            occ = (np.arange(n) < o).astype(np.float64)
            c2 = np.vstack(mf.mol.sph2spinor_coeff()) @ C
            V = orbgrad.fock_matrices(coulomb, c2, ((dm1 - np.diag(occ)).T,))[0]
        Gf = X + (occ[both][None, :] - occ[held_out][:, None]) * V[np.ix_(held_out, both)]
        G[np.ix_(held_out, both)], G[np.ix_(both, held_out)] = Gf, -Gf.conj().T
        Gf = V[np.ix_(fvir, frozen)]
        G[np.ix_(fvir, frozen)], G[np.ix_(frozen, fvir)] = Gf, -Gf.conj().T
    del coulomb
    with timer('zv.solve'):
        relaxed, z = _relax(mycc, mf, C, mo_e, o, G, dm1, (act, frozen, vir, fvir), with_t,
                            DEGENERATE, tol, verbose, veff)
    return relaxed, dm1, z


def _relax(mycc, mf, C, mo_e, o, G, dm1, spaces, with_t, degenerate, tol, verbose, veff=None):
    '''(relaxed density, z) from the orbital gradient G[y, x] of all spinors
    and the density dm1: the rotations that follow from the Fock matrix
    staying diagonal - between frozen and active spinors of the same kind
    and, with (T), among the active occupied and among the active virtual
    ones, none between spinors closer than ``degenerate`` -, y_pq = G_pq /
    (e_q - e_p), go into the density and their Fock matrix to the right-hand
    side of the Z-vector equations of all virtual with all occupied spinors.
    ``spaces``: the indices of the active occupied, the frozen occupied, the
    active virtual and the frozen virtual spinors.'''
    act, frozen, vir, fvir = spaces
    if veff is None:
        veff = lambda dm: mf.get_veff(mf.mol, dm)                   # noqa: E731
    y = np.zeros_like(G)
    pairs = [(a, f) for a, f in ((act, frozen), (vir, fvir)) if f.size]
    if with_t:
        pairs += [(act, act), (vir, vir)]
    for p, q in pairs:
        de = mo_e[q][None, :] - mo_e[p][:, None]                    # e_q - e_p
        ok = abs(de) > degenerate
        Gpq = G[np.ix_(p, q)]
        if p is not q and not ok.all() and abs(Gpq[~ok]).max() > 1e-8:
            logger.warn(mycc, 'A frozen spinor is degenerate with an active one of its kind and '
                        'the energy changes with their rotation (orbital gradient %.1e): the '
                        'relaxed density is not defined. Freeze whole levels.',
                        abs(Gpq[~ok]).max())
        yb = np.where(ok, Gpq / np.where(ok, de, 1.0), 0.0)
        y[np.ix_(p, q)] = yb
        if p is not q:
            y[np.ix_(q, p)] = yb.conj().T
    gvo = -G[o:, :o]
    out = np.array(dm1, dtype=np.complex128)
    if pairs:
        gvo = gvo - C[:, o:].conj().T @ veff(C @ y @ C.conj().T) @ C[:, :o]
        out += y
    conv, z = solve(mf, C, mo_e, o, gvo, tol=tol, verbose=verbose, veff=veff)
    if not conv:
        logger.warn(mycc, 'Z-vector equations not converged')
    return relaxed_rdm1(out, z, o), z


def _t_terms(mycc, eris, t1, t2p, kept):
    '''((s1, s2p, goo, gvv, dvo), Densities) of (T) for the orbital gradient:
    what solve_lambda(with_t='gradient') kept (``kept``: the sources over
    their denominators, which are multiplied back in place - the arrays are
    not used again), else one pass over the triples.'''
    from zccsdt.zccsd import kernels as K
    if kept is None:
        return mycc.t_terms(eris, t1, t2p, gradient=True)
    l1t, l2t, dens = kept
    eo, ev = (np.ascontiguousarray(x, dtype=np.float64)
              for x in (eris.mo_energy[:eris.nocc], eris.mo_energy[eris.nocc:]))
    return (l1t * (eo[:, None] - ev), K.mul_packed(l2t, eo, ev)) + tuple(mycc._t_density), dens


def relaxed_rdm1_dense(mycc, mf=None, l1=None, l2=None, tol=1e-9, verbose=None, with_t=False):
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
    equations the Fock matrix of the density Ci y Cc^+ + Cc y^+ Ci^+.
    Frozen virtual spinors: the same between them and the active virtual
    ones; the Z-vector equations are those of all virtual with all occupied
    spinors.'''
    from pyscf.cc import gccsd_rdm
    mf = mycc._scf if mf is None else mf
    C, mo_e, o, frozen = full_space(mycc)
    fvir = frozen_virtuals(mycc)
    act = np.setdiff1d(np.arange(o), frozen)
    vir = np.setdiff1d(np.arange(o, C.shape[1]), fvir)
    if l1 is None or l2 is None:
        l1, l2 = mycc.solve_lambda(with_t=with_t)
    dm1 = gccsd_rdm.make_rdm1(mycc, mycc.t1, mycc.t2, l1, l2)
    dm2 = gccsd_rdm.make_rdm2(mycc, mycc.t1, mycc.t2, l1, l2)
    h = C.conj().T @ mf.get_hcore() @ C
    eri = mf.mol.intor('int2e_spinor')
    eri = np.einsum('mp,nq,mnls,lr,st->pqrt', C.conj(), C, eri, C.conj(), C, optimize=True)
    G = orbital_gradient(h, eri, dm1, dm2)
    if with_t:
        # E(T) of the active spinors: its densities with the integrals of all
        # spinors, and its terms of the one-particle density
        from zccsdt.zccsd import lam_t
        n = C.shape[1]
        g = eri.transpose(0, 2, 1, 3) - eri.transpose(0, 2, 3, 1)
        fock = h + np.einsum('pkqk->pq', g[:, :o, :, :o])
        Gt = gradient_nh(t_fock(fock, o), g,
                         *t_densities(mycc.t1, mycc.t2, mycc.eris, n, o, act, vir), o)
        G = G + 0.5 * (Gt - Gt.conj().T)
        doo, dov, dvo, dvv = gccsd_rdm._gamma1_intermediates(mycc, mycc.t1, mycc.t2, l1, l2)
        goo, gvv, gvo = lam_t.gamma1(mycc.t1, mycc.t2, mycc.eris)
        doo[np.diag_indices(act.size)] -= goo
        dvv[np.diag_indices(vir.size)] += gvv
        dm1 = gccsd_rdm._make_rdm1(mycc, (doo, dov, dvo + gvo, dvv), with_frozen=True)
    relaxed, z = _relax(mycc, mf, C, mo_e, o, G, dm1, (act, frozen, vir, fvir), with_t,
                        DEGENERATE_DENSE, tol, verbose)
    return relaxed, dm1, z
