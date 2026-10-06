'''Orbital relaxation of the spinor CCSD (zccsdt.zccsd.zvector): the relaxed
one-particle density against the derivative of E(SCF) + E_corr with respect to
a one-particle perturbation, the SCF repeated in the field.'''

import numpy as np
import pytest
from pyscf import gto, x2c
from pyscf.cc import gccsd_rdm

from zccsdt.zccsd import DirectZCCSD, orbgrad, zvector


@pytest.fixture(scope='module')
def mol():
    return gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis='6-31g', verbose=0)


def scf(mol, eps=0.0, V=None):
    mf = x2c.UHF(mol)
    if V is not None:
        h0 = mf.get_hcore()
        mf.get_hcore = lambda *args: h0 + eps * V
    mf.conv_tol = 1e-12
    mf.kernel()
    assert mf.converged
    return mf


def ccsd(mf, kr, frozen=0):
    mycc = DirectZCCSD(mf, frozen=frozen)
    mycc.verbose, mycc.conv_tol, mycc.conv_tol_normt, mycc.kramers = 0, 1e-12, 1e-9, kr
    mycc.max_cycle = 200
    mycc.kernel()
    assert mycc.converged
    return mycc


@pytest.mark.parametrize('kr', [False, None])
def test_relaxed_density_is_energy_derivative(mol, kr):
    '''The energy from the densities of t and lambda; the orbital gradient
    of the Lagrangian (anti-Hermitian, no occupied-occupied and no
    virtual-virtual block); the relaxed density with the dipole operator of
    the 2c AO basis against the central difference of E(SCF) + E_corr in the
    field, which the unrelaxed density misses by 2e-3.'''
    mf = scf(mol)
    mycc = ccsd(mf, kr)
    eris = mycc.eris
    assert bool(eris.kramers) == (kr is None)
    C, o = eris.mo_coeff, eris.nocc
    l1, l2 = mycc.solve_lambda()
    dm1 = gccsd_rdm.make_rdm1(mycc, mycc.t1, mycc.t2, l1, l2)
    dm2 = gccsd_rdm.make_rdm2(mycc, mycc.t1, mycc.t2, l1, l2)
    h = C.conj().T @ mf.get_hcore() @ C
    eri = mol.intor('int2e_spinor')
    eri = np.einsum('mp,nq,mnls,lr,st->pqrt', C.conj(), C, eri, C.conj(), C, optimize=True)
    e = np.einsum('pq,qp', h, dm1) + 0.5 * np.einsum('pqrs,pqrs', eri, dm2) + mol.energy_nuc()
    assert abs(e - mf.e_tot - mycc.e_corr) < 1e-9
    G = zvector.orbital_gradient(h, eri, dm1, dm2)
    assert abs(G + G.conj().T).max() < 1e-10
    assert abs(G[:o, :o]).max() < 1e-8 and abs(G[o:, o:]).max() < 1e-8
    # half of the de-excitation part of the gradient without Hermitized densities
    # (the excitation part is the lambda1 equation: zero), from its terms
    g = eri.transpose(0, 2, 1, 3) - eri.transpose(0, 2, 3, 1)
    fock = h + np.einsum('pkqk->pq', g[:, :o, :, :o])
    terms = zvector.gradient_terms(mycc.t1, mycc.t2, l1, l2, fock, g, o)
    assert abs(G[o:, :o]).max() > 1e-4 and abs(0.5 * sum(terms.values()) - G[o:, :o]).max() < 1e-7
    assert min(abs(x).max() for x in terms.values()) > 1e-4
    # the same terms in the layouts of the amplitude step (zccsd.orbgrad)
    parts = {}
    Gnh = orbgrad.kernel(mycc, l1=l1, l2=l2, parts=parts)
    for k, x in terms.items():
        assert abs(parts[k] - x).max() < 1e-11, k
    assert abs(Gnh - sum(terms.values())).max() < 1e-11
    assert abs(G[o:, :o]).max() > 1e-4

    relaxed, unrelaxed, z = zvector.relaxed_rdm1_dense(mycc, mf, l1, l2, tol=1e-10)
    assert abs(unrelaxed - dm1).max() < 1e-12 and abs(relaxed - relaxed.conj().T).max() < 1e-12
    # with the orbital gradient of zccsd.orbgrad: the two differ as lambda is converged
    rel2, unrel2, z2 = zvector.kernel(mycc, l1=l1, l2=l2, mf=mf, tol=1e-10)
    print('relaxed density, orbgrad against the full arrays: %.2e (z: %.2e)'
          % (abs(rel2 - relaxed).max(), abs(z2 - z).max()))
    assert abs(unrel2 - dm1).max() < 1e-12 and abs(rel2 - relaxed).max() < 1e-7
    R, eai = zvector.response(mf, C, eris.mo_energy, o)
    assert abs(R(z) + G[o:, :o]).max() < 1e-9
    V = mol.intor('int1e_r_spinor')[2]
    Vmo = C.conj().T @ V @ C
    eps = 1e-4
    ep, em = [(m.e_tot + ccsd(m, kr).e_corr) for m in (scf(mol, eps, V), scf(mol, -eps, V))]
    derivative = (ep - em) / (2 * eps)
    assert abs(derivative - np.einsum('pq,qp', Vmo, relaxed).real) < 2e-7
    assert abs(derivative - np.einsum('pq,qp', Vmo, unrelaxed).real) > 1e-3
    # the same through DirectZCCSD.make_rdm1 and dip_moment: the field r_z times
    # eps lowers the energy by eps times the electronic part of the dipole
    assert abs(mycc.make_rdm1(l1=l1, l2=l2) - dm1).max() < 1e-12
    assert abs(mycc.make_rdm1(l1=l1, l2=l2, relaxed=True) - rel2).max() < 1e-9
    nuclear = np.einsum('i,ix->x', mol.atom_charges(), mol.atom_coords())[2]
    dip = mycc.dip_moment(l1=l1, l2=l2, relaxed=True, unit='AU', picture_change=False)
    assert abs(dip[2] - (nuclear - derivative)) < 2e-7 and abs(dip[:2]).max() < 1e-7
    plain = mycc.dip_moment(l1=l1, l2=l2, unit='AU', picture_change=False)
    assert 1e-3 < abs(plain[2] - dip[2]) < 1e-2
    # the operator with the picture-change transformation of X2C: the default
    changed = mycc.dip_moment(l1=l1, l2=l2, unit='AU')
    assert 1e-10 < abs(changed[2] - plain[2]) < 1e-5     # 9e-9 in this basis
    assert abs(mycc.dip_moment(l1=l1, l2=l2)[2] - changed[2] * 2.541746) < 1e-5


FROZEN = {'core': [0, 1], 'virtual': [22, 23, 24, 25], 'both': [0, 1, 22, 23, 24, 25]}


@pytest.mark.parametrize('kr,which', [(False, 'core'), (None, 'core'), (False, 'virtual'),
                                      (None, 'both')])
def test_relaxed_density_with_frozen_spinors(mol, kr, which):
    '''Frozen occupied spinors, frozen virtual ones, and both: the rotations
    between them and the active spinors of their kind follow from the Fock
    matrix staying diagonal.  The relaxed density, in the basis of all
    spinors, against the central difference of E(SCF) + E_corr with the SCF
    repeated in the field; from the full arrays and from the integrals with
    one frozen index (zccsd.orbgrad_frozen), with and without Kramers pairs.'''
    mf = scf(mol)
    frozen = FROZEN[which]
    mycc = ccsd(mf, kr, frozen=frozen)
    relaxed, unrelaxed, z = zvector.relaxed_rdm1_dense(mycc, mf, tol=1e-10)
    C, mo_e, o, fc = zvector.full_space(mycc)
    fv = zvector.frozen_virtuals(mycc)
    n = C.shape[1]
    assert o == 10 and list(fc) + list(fv) == frozen and z.shape == (n - o, o)
    rel2, unrel2, z2 = zvector.kernel(mycc, mf=mf, tol=1e-10)
    assert abs(unrel2 - unrelaxed).max() < 1e-12 and abs(rel2 - relaxed).max() < 2e-7
    assert abs(C.conj().T @ mf.get_ovlp() @ C - np.eye(n)).max() < 1e-9
    act = np.setdiff1d(np.arange(o), fc)
    vir = np.setdiff1d(np.arange(o, n), fv)
    for a, f in ((act, fc), (vir, fv)):         # between frozen and active spinors of a kind
        if f.size:
            assert abs(relaxed[np.ix_(a, f)]).max() > 1e-5
            assert abs(unrelaxed[np.ix_(a, f)]).max() < 1e-12
    V = mol.intor('int1e_r_spinor')[2]
    Vmo = C.conj().T @ V @ C
    eps = 1e-4
    ep, em = [(m.e_tot + ccsd(m, kr, frozen).e_corr) for m in (scf(mol, eps, V), scf(mol, -eps, V))]
    derivative = (ep - em) / (2 * eps)
    assert abs(derivative - np.einsum('pq,qp', Vmo, relaxed).real) < 2e-7
    assert abs(derivative - np.einsum('pq,qp', Vmo, rel2).real) < 2e-7
    assert abs(derivative - np.einsum('pq,qp', Vmo, unrelaxed).real) > 1e-3
    nuclear = np.einsum('i,ix->x', mol.atom_charges(), mol.atom_coords())[2]
    dip = mycc.dip_moment(relaxed=True, unit='AU', picture_change=False)
    assert abs(dip[2] - (nuclear - derivative)) < 2e-7


@pytest.mark.parametrize('kr', [False, None])
def test_orbital_gradient_with_a_frozen_index(mol, kr):
    '''zccsd.orbgrad_frozen - the integrals with one frozen index, occupied
    or virtual, contracted slab by slab with the densities of the active
    space that are never formed - against the full two-particle density of
    PySCF, block by block, and for the densities of (T) against
    zvector.gradient_nh by origin; then the whole block of the orbital
    gradient.  With Kramers pairs the frozen spinors are paired too and half
    of them are done: equal as far as the amplitudes are symmetric.'''
    from pyscf.cc import ccsd as pccsd
    from zccsdt.zccsd import orbgrad_frozen, triples
    mf = scf(mol)
    mycc = ccsd(mf, kr, frozen=FROZEN['both'])
    l1, l2 = mycc.solve_lambda(with_t=True)
    t1, t2 = mycc.t1, mycc.t2
    oa = t1.shape[0]
    C, mo_e, o, fc = zvector.full_space(mycc)
    fv = zvector.frozen_virtuals(mycc)
    P = np.concatenate((fc, fv))
    both = np.flatnonzero(pccsd.get_frozen_mask(mycc))
    n = C.shape[1]
    eri = mol.intor('int2e_spinor')
    eri = np.einsum('mp,nq,mnls,lr,st->pqrt', C.conj(), C, eri, C.conj(), C, optimize=True)
    S = eri[np.ix_(P, both, both, both)]
    d1 = gccsd_rdm._gamma1_intermediates(mycc, t1, t2, l1, l2)
    d2 = gccsd_rdm._gamma2_intermediates(mycc, t1, t2, l1, l2)
    names = ('dovov', 'dvvvv', 'doooo', 'doovv', 'dovvo', 'dvvov', 'dovvv', 'dooov')
    ref = {}
    for part, name in zip(orbgrad_frozen.PARTS[:6], ('doooo', 'dooov', 'dovov', 'dovvo', 'dovvv',
                                                     'dvvvv')):
        blocks = [b if (b is None or nm == name) else np.zeros_like(b) for nm, b in zip(names, d2)]
        D = gccsd_rdm._make_rdm2(mycc, d1, blocks, with_dm1=False, with_frozen=False)
        ref[part] = np.einsum('pqrs,xqrs->px', S, D)
    # (T): the densities of the active spinors with the integrals of all
    h = C.conj().T @ mf.get_hcore() @ C
    g = eri.transpose(0, 2, 1, 3) - eri.transpose(0, 2, 3, 1)
    fock = h + np.einsum('pkqk->pq', g[:, :o, :, :o])
    act, vir = both[:oa], both[oa:]
    dT2, dT1 = zvector.t_densities(t1, t2, mycc.eris, n, o, act, vir)
    for part, sl in (('g1', np.ix_(vir, vir, vir, act)), ('g2', np.ix_(act, vir, act, act)),
                     ('g3', np.ix_(vir, vir, act, act))):
        x = np.zeros_like(dT2)
        x[sl] = dT2[sl]
        Gt = zvector.gradient_nh(zvector.t_fock(fock, o), g, x, np.zeros_like(dT1), o)
        ref[part] = (0.5 * (Gt - Gt.conj().T))[np.ix_(P, both)]
    e_t, terms, dens = triples.kernel(mycc, mycc.eris, t1, t2, terms='gradient')
    parts = {}
    Cf = np.vstack(mol.sph2spinor_coeff()) @ C[:, P]
    X = orbgrad_frozen.kernel(mycc, mycc.eris, t1, t2, l1, l2, Cf, dens, mo_energy=mo_e[P],
                              parts=parts)
    dens.close()
    tol = 1e-13 if kr is False else 1e-10
    for part in orbgrad_frozen.PARTS:
        for sl in (slice(0, oa), slice(oa, None)):
            size = abs(ref[part][:, sl]).max()
            assert abs(parts[part][:, sl] - ref[part][:, sl]).max() < tol, part
            # every block of the density meets the slabs, with an occupied and
            # with a virtual x where it has one
            assert size > 1e-5 or (part, sl.start) in (('oooo', oa), ('vvvv', 0)), part
    assert abs(X - sum(ref.values())).max() < tol
    # the gradient itself: with the Fock matrix of the correlation part of the density
    # (to what the SCF left of its Fock matrix between frozen and active spinors)
    dm1 = gccsd_rdm.make_rdm1(mycc, t1, t2, l1, l2)
    G = zvector.orbital_gradient(h, eri, dm1, gccsd_rdm.make_rdm2(mycc, t1, t2, l1, l2))
    occ = (np.arange(n) < o).astype(float)
    dm1c = dm1 - np.diag(occ)
    Vf = np.einsum('pqrs,sr->pq', eri, dm1c) - np.einsum('psrq,sr->pq', eri, dm1c)
    Xc = sum(parts[k] for k in orbgrad_frozen.PARTS[:6])
    Gf = Xc + (occ[both][None, :] - occ[P][:, None]) * Vf[np.ix_(P, both)]
    assert abs(Gf - G[np.ix_(P, both)]).max() < 1e-7 and abs(Gf).max() > 1e-3
    assert abs(Vf[np.ix_(fv, fc)] - G[np.ix_(fv, fc)]).max() < 1e-7
    assert abs(Vf[np.ix_(fv, fc)]).max() > 1e-4
    assert abs(G[np.ix_(fc, fc)]).max() < 1e-7 and abs(G[np.ix_(fv, fv)]).max() < 1e-7


def ccsd_t_blocks(mol, mf, mycc):
    '''(h, eri, g, fock) on the active spinors of the CCSD: h[p,q], (pq|rs),
    <pq||rs> and the Fock matrix.'''
    C, o = mycc.eris.mo_coeff, mycc.eris.nocc
    h = C.conj().T @ mf.get_hcore() @ C
    eri = mol.intor('int2e_spinor')
    eri = np.einsum('mp,nq,mnls,lr,st->pqrt', C.conj(), C, eri, C.conj(), C, optimize=True)
    g = eri.transpose(0, 2, 1, 3) - eri.transpose(0, 2, 3, 1)
    return h, eri, g, h + np.einsum('pkqk->pq', g[:, :o, :, :o])


@pytest.mark.parametrize('kr', [False, None])
def test_orbital_gradient_of_ccsd_t(mol, kr):
    '''The (T) terms of the orbital gradient.  The densities of (T) are the
    derivatives of Re E(T) with respect to the integrals (central
    differences in random directions).  zccsd.orbgrad_t, from the densities
    of triples.kernel in the layouts of the amplitude step, against the full
    arrays (zvector.gradient_nh, gradient_t), by origin and in all four
    blocks.  The occupied-occupied and virtual-virtual blocks of the gradient
    vanish between degenerate spinors and are (e_q - e_p) times the
    off-diagonal one-particle density of (T), the products of the triples.'''
    import types
    from zccsdt.zccsd import lam_t, orbgrad_t, triples
    mf = scf(mol)
    mycc = ccsd(mf, kr)
    eris, t1, t2 = mycc.eris, mycc.t1, mycc.t2
    o, n = eris.nocc, eris.fock.shape[0]
    oc, vi = slice(0, o), slice(o, n)
    h, eri, g, fock = ccsd_t_blocks(mol, mf, mycc)
    mo_e = eris.mo_energy

    def blocks(g):
        return types.SimpleNamespace(nocc=o, fock=fock, mo_energy=mo_e, ovvv=g[oc, vi, vi, vi],
                                     ooov=g[oc, oc, oc, vi], oovv=g[oc, oc, vi, vi])
    assert abs(lam_t.energy(t1, t2, blocks(g)) - mycc.ccsd_t()) < 1e-10
    d2, d1 = zvector.t_densities(t1, t2, eris)
    rng = np.random.default_rng(5)
    for sl in ((vi, vi, vi, oc), (oc, vi, oc, oc), (vi, vi, oc, oc)):
        dg = np.zeros_like(g)
        dg[sl] = rng.standard_normal(d2[sl].shape) + 1j * rng.standard_normal(d2[sl].shape)
        dg = dg - dg.transpose(1, 0, 2, 3)
        dg = dg - dg.transpose(0, 1, 3, 2)
        dg = dg + dg.transpose(2, 3, 0, 1).conj()           # an integral: <pq||rs> = <rs||pq>*
        ep, em = [lam_t.energy(t1, t2, blocks(g + x * dg)) for x in (1e-5, -1e-5)]
        analytic = np.einsum('pqrs,pqrs', d2[sl], dg[sl]).real
        assert abs(analytic) > 1e-6 and abs((ep - em) / 2e-5 - analytic) < 1e-11
    # by origin: one density at a time, the Fock matrix, the sources with t
    ref, zero2, zero1 = {}, np.zeros_like(d2), np.zeros_like(d1)
    for name, sl in (('g1', (vi, vi, vi, oc)), ('g2', (oc, vi, oc, oc)), ('g3', (vi, vi, oc, oc))):
        x = np.zeros_like(d2)
        x[sl] = d2[sl]
        ref[name] = zvector.gradient_nh(fock, g, x, zero1, o)
    ref['fock'] = zvector.gradient_nh(zvector.t_fock(fock, o), g, zero2, d1, o)
    total = zvector.gradient_t(t1, t2, fock, g, o)
    ref['sources'] = total - zvector.gradient_nh(zvector.t_fock(fock, o), g, d2, d1, o)
    e_t, terms, dens = triples.kernel(mycc, eris, t1, t2, terms='gradient')
    parts = {}
    Gt = orbgrad_t.kernel(mycc, eris, t1, t2, terms, dens, parts=parts)
    dens.close()
    for name in orbgrad_t.PARTS:
        assert abs(parts[name] - ref[name]).max() < 1e-11, name
        # every origin has an occupied-virtual block; <bc||jk> and the sources
        # (whose part is in zccsd.orbgrad) have no virtual-occupied one
        assert abs(ref[name][oc, vi]).max() > 1e-5, name
        assert (abs(ref[name][vi, oc]).max() > 1e-5) == (name in ('g1', 'g2', 'fock')), name
    assert min(abs(ref['g1'][a, a]).max() for a in (oc, vi)) > 1e-4
    assert abs(Gt - total).max() < 1e-11 and abs(total).max() > 1e-3
    # the blocks that vanish without (T)
    G = 0.5 * (total - total.conj().T)
    t3c, t3d, _ = lam_t.triples(t1, t2, eris)
    for blk, D in ((oc, -np.einsum('ijkabc,ljkabc->il', (t3c + t3d).conj(), t3c) / 12),
                   (vi, np.einsum('ijkabc,ijkdbc->da', (t3c + t3d).conj(), t3c) / 12)):
        de = mo_e[blk][None, :] - mo_e[blk][:, None]
        same = abs(de) < 1e-8
        assert abs(G[blk, blk][same]).max() < 1e-15 and abs(G[blk, blk]).max() > 5e-6
        D = 0.5 * (D + D.conj().T)
        assert abs(D[same & ~np.eye(same.shape[0], dtype=bool)]).max() < 1e-12
        # exactly: whatever the SCF left off the diagonal of its Fock matrix
        assert abs(G[blk, blk] - de * D).max() < 1e-15


@pytest.mark.parametrize('kr,frozen', [(False, 0), (None, 0), (False, 2), (None, 2)])
def test_relaxed_ccsd_t_density_is_energy_derivative(mol, kr, frozen):
    '''CCSD(T): the relaxed density - lambda with the sources of (T), the
    (T) terms of the density and of the orbital gradient, the rotations
    among occupied and among virtual spinors, the Z-vector equations -
    against the central difference of E(SCF) + E(CCSD) + E(T) with the SCF
    repeated in the field; from the full arrays and from the layouts of the
    amplitude step (zvector.kernel), in one pass over the triples.  Also
    with two frozen occupied spinors, whose E(T) is that of the active ones.'''
    mf = scf(mol)
    mycc = ccsd(mf, kr, frozen)
    C, o = zvector.full_space(mycc)[0], 10
    dense, unrelaxed, z = zvector.relaxed_rdm1_dense(mycc, mf, tol=1e-10, with_t=True)
    assert mycc._l_with_t and mycc._t_gradient is None
    relaxed, unrel2, z2 = zvector.kernel(mycc, mf=mf, tol=1e-10, with_t=True)
    assert abs(unrel2 - unrelaxed).max() < 1e-12 and abs(relaxed - dense).max() < 2e-7
    assert abs(relaxed - relaxed.conj().T).max() < 1e-12
    # the rotations among active occupied and among virtual spinors
    assert abs((relaxed - unrelaxed)[frozen:o, frozen:o]).max() > 5e-6
    assert abs((relaxed - unrelaxed)[o:, o:]).max() > 5e-6
    assert not frozen or abs((relaxed - unrelaxed)[:frozen, :frozen]).max() < 1e-12
    V = mol.intor('int1e_r_spinor')[2]
    Vmo = C.conj().T @ V @ C
    eps = 1e-4
    e = []
    for m in (scf(mol, eps, V), scf(mol, -eps, V)):
        cc = ccsd(m, kr, frozen)
        e.append(m.e_tot + cc.e_corr + cc.ccsd_t())
    derivative = (e[0] - e[1]) / (2 * eps)
    assert abs(derivative - np.einsum('pq,qp', Vmo, dense).real) < 2e-7
    assert abs(derivative - np.einsum('pq,qp', Vmo, relaxed).real) < 2e-7
    assert abs(derivative - np.einsum('pq,qp', Vmo, unrelaxed).real) > 3e-4
    # all of it from the CCSD object: lambda, the sources and the densities of
    # (T) in one pass over the triples
    fresh = ccsd(mf, kr, frozen)
    nuclear = np.einsum('i,ix->x', mol.atom_charges(), mol.atom_coords())[2]
    dip = fresh.dip_moment(with_t=True, relaxed=True, unit='AU', picture_change=False)
    assert abs(dip[2] - (nuclear - derivative)) < 2e-7 and abs(dip[:2]).max() < 1e-7
    assert fresh._l_with_t and fresh._t_gradient is None and abs(fresh.e_t - mycc.e_t) < 1e-10
    assert abs(fresh.make_rdm1(with_t=True) - unrelaxed[frozen:, frozen:]).max() < 1e-8


@pytest.mark.parametrize('kr,which', [(False, 'virtual'), (None, 'both')])
def test_relaxed_ccsd_t_density_with_frozen_virtual_spinors(mol, kr, which):
    '''CCSD(T) with frozen virtual spinors (and with frozen occupied ones
    too): E(T) is that of the active spinors, and its densities meet the
    integrals with a frozen index.  The relaxed density from the full arrays
    and from zvector.kernel against the central difference of the energy.'''
    mf = scf(mol)
    frozen = FROZEN[which]
    mycc = ccsd(mf, kr, frozen)
    C, mo_e, o, fc = zvector.full_space(mycc)
    fv = zvector.frozen_virtuals(mycc)
    dense, unrelaxed, z = zvector.relaxed_rdm1_dense(mycc, mf, tol=1e-10, with_t=True)
    relaxed, unrel2, z2 = zvector.kernel(mycc, mf=mf, tol=1e-10, with_t=True)
    assert abs(unrel2 - unrelaxed).max() < 1e-12 and abs(relaxed - dense).max() < 2e-7
    assert abs(relaxed - relaxed.conj().T).max() < 1e-12
    vir = np.setdiff1d(np.arange(o, C.shape[1]), fv)
    assert abs((relaxed - unrelaxed)[np.ix_(vir, fv)]).max() > 1e-5
    assert abs((relaxed - unrelaxed)[np.ix_(fv, fv)]).max() < 1e-12
    V = mol.intor('int1e_r_spinor')[2]
    Vmo = C.conj().T @ V @ C
    eps = 1e-4
    e = []
    for m in (scf(mol, eps, V), scf(mol, -eps, V)):
        cc = ccsd(m, kr, frozen)
        e.append(m.e_tot + cc.e_corr + cc.ccsd_t())
    derivative = (e[0] - e[1]) / (2 * eps)
    assert abs(derivative - np.einsum('pq,qp', Vmo, dense).real) < 2e-7
    assert abs(derivative - np.einsum('pq,qp', Vmo, relaxed).real) < 2e-7
    assert abs(derivative - np.einsum('pq,qp', Vmo, unrelaxed).real) > 3e-4


def test_gamma1_equals_pyscf():
    '''The one-particle density intermediates from packed amplitudes against
    PySCF's on the full ones, random complex t and lambda.'''
    from zccsdt.zccsd import rdm
    rng = np.random.default_rng(3)
    o, v = 6, 9

    def amplitudes():
        x1 = rng.standard_normal((o, v)) + 1j * rng.standard_normal((o, v))
        x2 = rng.standard_normal((o, o, v, v)) + 1j * rng.standard_normal((o, o, v, v))
        x2 = x2 - x2.transpose(1, 0, 2, 3)
        return 0.1 * x1, 0.1 * np.ascontiguousarray(x2 - x2.transpose(0, 1, 3, 2))

    (t1, t2), (l1, l2) = amplitudes(), amplitudes()
    ref = gccsd_rdm._gamma1_intermediates(None, t1, t2, l1, l2)
    for a, b in zip(rdm.gamma1(t1, t2, l1, l2, pair_bytes=5000), ref):
        assert abs(a - b).max() < 1e-12


def test_density_with_frozen_spinors_in_the_ao_basis(mol):
    '''make_rdm1(ao_repr=True) with frozen spinors and Kramers pairs: the
    number of electrons, and the expectation value of a one-particle
    operator as in the basis of the active spinors plus the frozen ones.'''
    mf = scf(mol)
    mycc = DirectZCCSD(mf, frozen=2)
    mycc.verbose, mycc.conv_tol, mycc.conv_tol_normt = 0, 1e-10, 1e-8
    mycc.kernel()
    dm1 = mycc.make_rdm1()
    dm_ao = mycc.make_rdm1(l1=mycc.l1, l2=mycc.l2, ao_repr=True)
    S = mf.get_ovlp()
    assert abs(np.trace(dm_ao @ S) - mol.nelectron) < 1e-9
    C = mycc.eris.mo_coeff
    V = mol.intor('int1e_r_spinor')[2]
    frozen = mf.mo_coeff[:, :2]
    expect = np.einsum('pq,qp', C.conj().T @ V @ C, dm1) + np.trace(frozen.conj().T @ V @ frozen)
    assert abs(np.einsum('pq,qp', V, dm_ao) - expect) < 1e-10


def test_orbital_gradient_within_plan(monkeypatch):
    '''orbgrad.gradient under memory plans of several sizes - m in memory
    and in a scratch file, the products and the pairs in one block and in
    several - with and without Kramers pairs, for random t and lambda: the
    same gradient, and what it allocates (tracemalloc follows numpy's
    allocations) next to the packed t2 and lambda2 that it is given stays
    within the budget of the plan and orbgrad.reserve_bytes, which is how
    orbgrad.kernel plans it.'''
    import tracemalloc
    from zccsdt import memory
    from zccsdt.zccsd import amps, ao2mo, kramers
    from zccsdt.zccsd import kernels as K
    monkeypatch.setattr(memory, 'resident', lambda: 0)
    monkeypatch.setattr(orbgrad, 'UNPACK_BYTES', 1 << 20)   # of the size of the arrays here
    mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis='cc-pvtz', verbose=0)
    mf = x2c.UHF(mol).run(conv_tol=1e-10)
    rng = np.random.default_rng(1)
    blocked = mapped = refused = 0
    for kr in (None, False):
        cc = DirectZCCSD(mf)
        cc.verbose, cc.kramers, cc.max_memory = 0, kr, 100000
        eris = cc.ao2mo()
        o, v, nao = eris.nocc, eris.fock.shape[0] - eris.nocc, mol.nao_nr()
        npo, npv = K.npair(o), K.npair(v)

        def packed():
            x1 = 0.05 * (rng.standard_normal((o, v)) + 1j * rng.standard_normal((o, v)))
            x2 = 0.02 * (rng.standard_normal((npo, npv)) + 1j * rng.standard_normal((npo, npv)))
            if eris.kramers:
                x1 = 0.5 * (x1 + kramers.image(x1))
                x2 = np.ascontiguousarray(0.5 * (x2 + kramers.image(x2, pairs=(0, 1))))
            return x1, x2

        (t1, t2p), (l1, l2p) = packed(), packed()
        need = amps.plan_step(o, v, nao, None, (), kramers=eris.kramers).need
        ref = None
        for factor in (1.0, 1.3, 2.0, 4.0, 40.0):
            plan = amps.plan_step(o, v, nao, int(factor * need), (), diis_incore=False,
                                  three_m=False, kramers=eris.kramers)
            # the least of the step is not that of the integrals <j~ e||ab>: a slab of
            # 16 v^3 bytes with what makes it, and the arrays of one occupied spinor
            room = plan.budget + orbgrad.reserve_bytes(o, v) - 64 * o * v * v
            if ao2mo.plan_slabs(mol, o, v, None, sort_arrays=0).need > room:
                with pytest.raises(MemoryError, match='of the orbital gradient'):
                    orbgrad.slab_plan(mol, plan, o, v)
                refused += 1
                continue
            eris.residual = amps.Residual(eris, plan=plan)
            tracemalloc.start()
            base = tracemalloc.get_traced_memory()[0]
            G = orbgrad.gradient(cc, eris, t1, t2p, l1, l2p)
            peak = tracemalloc.get_traced_memory()[1] - base
            tracemalloc.stop()
            # the integrals <j~ e||ab> are made in slabs as large as the budget
            # allows: beyond the working set of the plan where there is room
            work = max(orbgrad.working_set(plan, o, v), plan.budget + orbgrad.reserve_bytes(o, v))
            print('Kramers %-5s x%-4.1f budget %7.1f MB  working set %7.1f MB  peak %7.1f MB  '
                  'W mapped %-5s blocks %d %d' % (bool(eris.kramers), factor, plan.budget / 1e6,
                                                 work / 1e6, peak / 1e6, plan.w_mapped,
                                                 len(plan.jblocks), len(plan.pblocks)))
            assert peak <= work, (kr, factor, peak, work)
            blocked += len(plan.jblocks) > 1
            mapped += plan.w_mapped
            if ref is None:
                ref = G
            assert abs(G - ref).max() < 1e-11 and abs(ref).max() > 1e-3
    assert blocked > 3 and mapped > 1 and refused == 1
    # Kramers pairs (the last case was without): what is done for half of the
    # rows and sums and completed by time reversal against all of it, and
    # amplitudes without the symmetry, for which all of it is done
    cc = DirectZCCSD(mf)
    cc.verbose, cc.max_memory = 0, 100000
    eris = cc.ao2mo()
    assert eris.kramers
    plan = amps.plan_step(o, v, nao, None, (), diis_incore=False, three_m=False, kramers=True)
    for paired in (True, False):
        (t1, t2p), (l1, l2p) = packed(), packed()
        if not paired:
            t2p = t2p + 0.01 * (rng.standard_normal(t2p.shape)
                                + 1j * rng.standard_normal(t2p.shape))
        res = eris.residual = amps.Residual(eris, plan=plan)
        assert res.kramers and res.symmetric(t1, t2p) == paired
        G = orbgrad.gradient(cc, eris, t1, t2p, l1, l2p)
        res.kramers = False
        assert abs(G - orbgrad.gradient(cc, eris, t1, t2p, l1, l2p)).max() < 1e-10


def test_orbital_gradient_of_t_within_budget(monkeypatch, tmp_path):
    '''zccsd.orbgrad_t: what it allocates (tracemalloc follows numpy's
    allocations) next to the packed arrays and the densities it is given
    stays within max_memory, as its kernel plans the contraction with
    <bc||ea> in the AO basis - one occupied spinor and one shell of AO
    integrals at a time, a few, all at once -, with the density of <bc||ek>
    in memory and in a scratch file; MemoryError where one spinor and the
    largest shell do not fit.  Random amplitudes and densities with the
    symmetry of the Kramers pairs: the same result for every budget, and
    without the reduction of the sums by time reversal.'''
    import tracemalloc
    from pyscf import lib
    from zccsdt import memory, scratch
    from zccsdt.zccsd import kramers, orbgrad_t, triples
    from zccsdt.zccsd import kernels as K
    monkeypatch.setattr(lib.param, 'TMPDIR', str(tmp_path))
    monkeypatch.setenv('ZCCSDT_TMPDIR', str(tmp_path))
    monkeypatch.setattr(memory, 'resident', lambda: 0)
    monkeypatch.setattr(orbgrad, 'UNPACK_BYTES', 1 << 20)   # of the size of the arrays here
    mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis='cc-pvtz', verbose=0)
    mf = x2c.UHF(mol).run(conv_tol=1e-10)
    cc = DirectZCCSD(mf)
    cc.verbose = 0
    eris = cc.ao2mo()
    assert eris.kramers
    o, v, nao = eris.nocc, eris.fock.shape[0] - eris.nocc, mol.nao_nr()
    npo, npv, npair = K.npair(o), K.npair(v), nao * (nao + 1) // 2
    rng = np.random.default_rng(7)

    def random(shape, pairs=()):
        x = 0.02 * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
        return np.ascontiguousarray(0.5 * (x + kramers.image(x, pairs=pairs)))
    t1, t2p = random((o, v)), random((npo, npv), (0, 1))
    terms = (random((o, v)), random((npo, npv), (0, 1)), rng.standard_normal(o) + 0j,
             rng.standard_normal(v) + 0j, random((v, o)))
    for k in (2, 3):                    # the diagonals of a symmetric density
        terms[k][terms[k].size // 2:] = terms[k][:terms[k].size // 2]
    g1 = random((o, v, npv), (2,))
    g2, g3 = random((o, v, npo), (2,)), random((npo, npv), (0, 1))
    scr = scratch.Scratch()
    g1m = scr.create('g1', g1.shape, np.complex128)
    g1m[:] = g1
    ao_loc = mol.ao_loc_nr()
    lmax = int((ao_loc[1:] - ao_loc[:-1]).max())
    own = orbgrad_t.own_bytes(o, v, nao)
    row, block = 32 * npair * nao, npair * lmax * (8 * nao + 32)
    assert own > 4e6 and row > 1e6
    cc.max_memory = (own + 0.95 * (2 * row + block)) / 1e6
    with pytest.raises(MemoryError, match='of one occupied spinor in the AO basis'):
        orbgrad_t.kernel(cc, eris, t1, t2p, terms, triples.Densities(g1, g2, g3))
    ref, plans = None, set()
    for extra, dens in ((2 * row + block, g1m), (2 * row + block, g1),
                        (4 * row + 2 * block, g1), (9 * row, g1m), (40 * row, g1)):
        budget = own + extra + 1
        cc.max_memory = budget / 1e6
        plan = orbgrad_t.vvvv_plan(mol, cc.max_memory, o, v, o // 2)
        plans.add(plan)
        coulomb = orbgrad.Coulomb(mol, 0)       # AO direct: the same for every budget
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        G = orbgrad_t.kernel(cc, eris, t1, t2p, terms, triples.Densities(dens, g2, g3),
                             coulomb=coulomb)
        peak = tracemalloc.get_traced_memory()[1] - base
        tracemalloc.stop()
        print('budget %6.1f MB  own %5.1f MB  %d occupied spinors, %3d AOs at a time  '
              'peak %6.1f MB' % (budget / 1e6, own / 1e6, plan[0], plan[1], peak / 1e6))
        assert peak <= budget, (extra, peak, budget)
        ref = G if ref is None else ref
        assert abs(G - ref).max() < 1e-11 and abs(ref).max() > 1e-3
    assert (1, lmax) in plans and (o // 2, nao) in plans and len(plans) >= 3
    scr.close()
    # all rows and all occupied spinors, without their time-reversed image
    eris.residual.kramers = False
    cc.max_memory = 100000
    G = orbgrad_t.kernel(cc, eris, t1, t2p, terms, triples.Densities(g1, g2, g3))
    assert abs(G - ref).max() < 1e-10


def test_orbital_gradient_with_a_frozen_index_within_budget(monkeypatch, tmp_path):
    '''zccsd.orbgrad_frozen: what its kernel allocates (tracemalloc follows
    numpy's allocations) next to the packed amplitudes and the densities of
    (T) it is given stays within max_memory - the product of lambda2 and t2
    in memory and in a scratch file, the integrals of one frozen spinor at a
    time, of a few, of all -, and it raises MemoryError where one frozen
    spinor does not fit.  Random amplitudes and densities with the symmetry
    of the Kramers pairs: the same result for every budget, and without the
    reduction to half of the frozen spinors.'''
    import tracemalloc
    from pyscf import lib
    from zccsdt import memory, scratch
    from zccsdt.zccsd import amps, ao2mo, kramers, orbgrad_frozen, triples
    from zccsdt.zccsd import kernels as K
    monkeypatch.setattr(lib.param, 'TMPDIR', str(tmp_path))
    monkeypatch.setenv('ZCCSDT_TMPDIR', str(tmp_path))
    monkeypatch.setattr(memory, 'resident', lambda: 0)
    monkeypatch.setattr(orbgrad, 'UNPACK_BYTES', 1 << 20)   # of the size of the arrays here
    mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis='cc-pvtz', verbose=0)
    mf = x2c.UHF(mol).run(conv_tol=1e-10)
    nmo = mf.mo_coeff.shape[1]
    cc = DirectZCCSD(mf, frozen=[0, 1, 2, 3] + list(range(nmo - 8, nmo)))
    cc.verbose, cc.max_memory = 0, 100000
    eris = cc.ao2mo()
    assert eris.kramers
    o, v, nao = eris.nocc, eris.fock.shape[0] - eris.nocc, mol.nao_nr()
    npo, npv, n1 = K.npair(o), K.npair(v), o * v
    C, mo_e, nocc, fc = zvector.full_space(cc)
    P = np.concatenate((fc, zvector.frozen_virtuals(cc)))
    Cf = np.vstack(mol.sph2spinor_coeff()) @ C[:, P]
    rng = np.random.default_rng(11)

    def random(shape, pairs=()):
        x = 0.02 * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
        return np.ascontiguousarray(0.5 * (x + kramers.image(x, pairs=pairs)))
    t1, t2p, l1, l2p = (random((o, v)), random((npo, npv), (0, 1)), random((o, v)),
                        random((npo, npv), (0, 1)))
    dens = triples.Densities(random((o, v, npv), (2,)), random((o, v, npo), (2,)),
                             random((npo, npv), (0, 1)))
    # the density of <bc||ek> in a scratch file, as the pass over the triples
    # leaves it when it does not fit
    scr = scratch.Scratch()
    g1m = scr.create('g1', dens.g1.shape, np.complex128)
    g1m[:] = dens.g1
    mapped = triples.Densities(g1m, dens.g2, dens.g3)
    # the least: the amplitude step with its W in a scratch file, and one
    # frozen spinor of the half that is done
    kept, reserve = orbgrad_frozen.own_bytes(o, v)[0], orbgrad_frozen.reserve_bytes(o, v)
    step = amps.plan_step(o, v, nao, None, (), kramers=True).need + reserve
    one = (3 * 16 * npo * npv + kept + orbgrad_frozen.batch_bytes(o, v, 1, True)
           + ao2mo.plan_slabs(mol, 1, o + v, None, sort_arrays=0).need)
    floor = max(step, one)
    cc.max_memory = 0.97 * floor / 1e6
    with pytest.raises(MemoryError, match='frozen'):
        orbgrad_frozen.kernel(cc, eris, t1, t2p, l1, l2p, Cf, dens, mo_energy=mo_e[P])
    ref, seen = None, set()
    for factor in (1.0, 1.2, 1.6, 2.5, 5.0, 40.0):
        budget = int(factor * floor) + 1
        cc.max_memory = budget / 1e6
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        X = orbgrad_frozen.kernel(cc, eris, t1, t2p, l1, l2p, Cf,
                                  mapped if factor in (1.0, 1.6, 5.0) else dens,
                                  mo_energy=mo_e[P])
        peak = tracemalloc.get_traced_memory()[1] - base
        tracemalloc.stop()
        plan = eris.residual.plan
        left = budget - 3 * 16 * npo * npv - kept - (0 if plan.w_mapped else 16 * n1 * n1)
        slabs = orbgrad_frozen.slab_plan(mol, left, o, v, P.size // 2, True)
        seen.add((plan.w_mapped, slabs.nket))
        print('x%-4.1f budget %7.1f MB  peak %7.1f MB  m mapped %-5s  %d of %d frozen spinors '
              'at a time' % (factor, budget / 1e6, peak / 1e6, plan.w_mapped, slabs.nket,
                             P.size // 2))
        assert peak <= budget, (factor, peak, budget)
        ref = X if ref is None else ref
        assert abs(X - ref).max() < 1e-11 and abs(ref).max() > 1e-3
    assert (True, 1) in seen and (False, P.size // 2) in seen and len(seen) >= 3
    # every frozen spinor, without the time-reversed image of half of them
    cc.max_memory = 100000
    X = orbgrad_frozen.kernel(cc, eris, t1, t2p, l1, l2p, Cf, dens)
    assert abs(X - ref).max() < 1e-10
    scr.close()


def test_slabs_of_a_frozen_index_as_large_as_fit(monkeypatch):
    '''zccsd.orbgrad_frozen.slab_plan: the most frozen spinors in a slab that
    fit with the arrays of their contraction, within the bytes that are left
    - more of them with more bytes, and never fewer than fit (a search by
    halves from all frozen spinors ended at one where ten fit, and the
    contractions took three to five times as long).'''
    from zccsdt.zccsd import ao2mo, orbgrad_frozen
    mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis='cc-pvdz', verbose=0)
    o, v, nf = 8, 40, 14
    nq = o + v
    monkeypatch.setattr(ao2mo, 'SLAB_BYTES', 9 * 16 * nq ** 3)      # a slab holds 9 at most
    least = (ao2mo.plan_slabs(mol, nf, nq, None, sort_arrays=0).need
             + orbgrad_frozen.batch_bytes(o, v, 1, True))
    with pytest.raises(MemoryError, match='one frozen spinor is contracted with'):
        orbgrad_frozen.slab_plan(mol, int(0.99 * least), o, v, nf, True)
    seen, last = set(), 0
    for factor in np.geomspace(1.0, 60.0, 80):
        left = int(factor * least)
        plan = orbgrad_frozen.slab_plan(mol, left, o, v, nf, True)
        nk = plan.nket
        assert plan.work <= left and last <= nk <= 9, (factor, nk, last)
        if nk < 9:      # one more does not fit
            more = left - orbgrad_frozen.batch_bytes(o, v, nk + 1, True)
            try:
                assert ao2mo.plan_slabs(mol, nf, nq, max(0, more), sort_arrays=0).nket <= nk
            except MemoryError:
                pass
        seen.add(nk)
        last = nk
    assert seen == set(range(1, 10))
    assert orbgrad_frozen.slab_plan(mol, None, o, v, nf, True).nket == 9
