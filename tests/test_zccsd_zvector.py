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


@pytest.mark.parametrize('kr', [False, None])
def test_relaxed_density_with_frozen_spinors(mol, kr):
    '''Two frozen occupied spinors: the rotations between them and the active
    occupied ones follow from the Fock matrix staying diagonal.  The relaxed
    density, in the basis of all spinors, against the central difference of
    E(SCF) + E_corr with the SCF repeated in the field; the same with and
    without Kramers pairs.'''
    mf = scf(mol)
    mycc = ccsd(mf, kr, frozen=2)
    relaxed, unrelaxed, z = zvector.relaxed_rdm1_dense(mycc, mf, tol=1e-10)
    C, mo_e, o, frozen = zvector.full_space(mycc)
    assert o == 10 and list(frozen) == [0, 1] and z.shape == (C.shape[1] - o, o)
    # without the full arrays: the frozen spinors as occupied ones with zero
    # amplitudes in the code for all spinors (zvector.gradient_frozen)
    h = C.conj().T @ mf.get_hcore() @ C
    eri = mol.intor('int2e_spinor')
    eri = np.einsum('mp,nq,mnls,lr,st->pqrt', C.conj(), C, eri, C.conj(), C, optimize=True)
    amplitudes = (mycc.t1, mycc.t2, mycc.l1, mycc.l2)
    G = zvector.orbital_gradient(h, eri, gccsd_rdm.make_rdm1(mycc, *amplitudes),
                                 gccsd_rdm.make_rdm2(mycc, *amplitudes))
    Gvo, Gic = zvector.gradient_frozen(mycc, *amplitudes)
    assert abs(Gvo - G[o:, :o]).max() < 1e-7 and abs(Gvo[:, :2]).max() > 1e-4
    assert abs(Gic - G[2:o, :2]).max() < 1e-9 and abs(Gic).max() > 1e-4
    rel2, unrel2, z2 = zvector.kernel(mycc, mf=mf, tol=1e-10)
    assert abs(unrel2 - unrelaxed).max() < 1e-12 and abs(rel2 - relaxed).max() < 2e-7
    assert abs(C.conj().T @ mf.get_ovlp() @ C - np.eye(C.shape[1])).max() < 1e-9
    assert abs(relaxed[2:o, :2]).max() > 1e-5           # the core-valence elements
    assert abs(unrelaxed[2:o, :2]).max() < 1e-12
    V = mol.intor('int1e_r_spinor')[2]
    Vmo = C.conj().T @ V @ C
    eps = 1e-4
    ep, em = [(m.e_tot + ccsd(m, kr, 2).e_corr) for m in (scf(mol, eps, V), scf(mol, -eps, V))]
    derivative = (ep - em) / (2 * eps)
    assert abs(derivative - np.einsum('pq,qp', Vmo, relaxed).real) < 2e-7
    assert abs(derivative - np.einsum('pq,qp', Vmo, unrelaxed).real) > 1e-3
    nuclear = np.einsum('i,ix->x', mol.atom_charges(), mol.atom_coords())[2]
    dip = mycc.dip_moment(relaxed=True, unit='AU', picture_change=False)
    assert abs(dip[2] - (nuclear - derivative)) < 2e-7


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
    ref['fock'] = zvector.gradient_nh(fock, g, zero2, d1, o)
    total = zvector.gradient_t(t1, t2, fock, g, o)
    ref['sources'] = total - zvector.gradient_nh(fock, g, d2, d1, o)
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
        # the Fock matrix of the SCF is diagonal to 1e-8 only
        assert abs(G[blk, blk] - de * D)[~same].max() < 1e-10


@pytest.mark.parametrize('kr', [False, None])
def test_relaxed_ccsd_t_density_is_energy_derivative(mol, kr):
    '''CCSD(T): the relaxed density - lambda with the sources of (T), the
    (T) terms of the density and of the orbital gradient, the rotations
    among occupied and among virtual spinors, the Z-vector equations -
    against the central difference of E(SCF) + E(CCSD) + E(T) with the SCF
    repeated in the field; from the full arrays and from the layouts of the
    amplitude step (zvector.kernel), in one pass over the triples.'''
    mf = scf(mol)
    mycc = ccsd(mf, kr)
    C, o = mycc.eris.mo_coeff, mycc.eris.nocc
    dense, unrelaxed, z = zvector.relaxed_rdm1_dense(mycc, mf, tol=1e-10, with_t=True)
    assert mycc._l_with_t and mycc._t_gradient is None
    relaxed, unrel2, z2 = zvector.kernel(mycc, mf=mf, tol=1e-10, with_t=True)
    assert abs(unrel2 - unrelaxed).max() < 1e-12 and abs(relaxed - dense).max() < 2e-7
    assert abs(relaxed - relaxed.conj().T).max() < 1e-12
    # the rotations among occupied and among virtual spinors
    assert abs((relaxed - unrelaxed)[:o, :o]).max() > 5e-6
    assert abs((relaxed - unrelaxed)[o:, o:]).max() > 5e-6
    V = mol.intor('int1e_r_spinor')[2]
    Vmo = C.conj().T @ V @ C
    eps = 1e-4
    e = []
    for m in (scf(mol, eps, V), scf(mol, -eps, V)):
        cc = ccsd(m, kr)
        e.append(m.e_tot + cc.e_corr + cc.ccsd_t())
    derivative = (e[0] - e[1]) / (2 * eps)
    assert abs(derivative - np.einsum('pq,qp', Vmo, dense).real) < 2e-7
    assert abs(derivative - np.einsum('pq,qp', Vmo, relaxed).real) < 2e-7
    assert abs(derivative - np.einsum('pq,qp', Vmo, unrelaxed).real) > 3e-4
    # all of it from the CCSD object: lambda, the sources and the densities of
    # (T) in one pass over the triples
    fresh = ccsd(mf, kr)
    nuclear = np.einsum('i,ix->x', mol.atom_charges(), mol.atom_coords())[2]
    dip = fresh.dip_moment(with_t=True, relaxed=True, unit='AU', picture_change=False)
    assert abs(dip[2] - (nuclear - derivative)) < 2e-7 and abs(dip[:2]).max() < 1e-7
    assert fresh._l_with_t and fresh._t_gradient is None and abs(fresh.e_t - mycc.e_t) < 1e-10
    assert abs(fresh.make_rdm1(with_t=True) - unrelaxed).max() < 1e-8


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
