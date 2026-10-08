'''The one-particle density, the orbital gradient and the relaxed density
of CCSDT from packed amplitudes (zccsdt.cc.density_opt):

- against the dense reference (zccsdt.cc.lambda_ref, the tables of
  lambda_terms) for complex amplitudes that solve nothing, in every plan of
  the memory, and for converged amplitudes of molecules;
- the CCSD limit against the spinor CCSD code (zccsdt.zccsd.zvector);
- the relaxed dipole derivative against E(SCF) + E_corr(CCSDT) with the SCF
  repeated in the field, for spin orbitals and for X2C spinors;
- the memory plan against what is allocated (tracemalloc).
Lambda is that of the dense reference (lambda_ref.kernel), packed.'''
import numpy as np
import pytest
from pyscf import gto, scf, x2c

import fockspace as F
from test_ccsdtq_ref import _spin_mixed
from zccsdt import eris as E
from zccsdt.cc import ccsdtq_ref as R, ccsdtq_terms as Q, density_opt as D, lambda_ref as LR, \
    lambda_terms as L, solver, t3pack

WATER = 'O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587'


def container(f, v, no):
    '''An integral container of full arrays f[p,q] and <pq||rs>.'''
    o, vi = slice(0, no), slice(no, None)
    blocks = {k: np.ascontiguousarray(v[tuple(o if c == 'o' else vi for c in k)]) for k in E.BLOCKS}
    return E.ERIs(nocc=no, fock=f, mo_energy=f.diagonal().real.copy(), e_hf=0.0, **blocks)


def random_case(no, nv, seed, real=False):
    '''(f, <pq||rs>, [t1, t2, t3], [l1, l2, l3]): Hermitian integrals with a
    Fock matrix that is not diagonal, amplitudes that solve nothing.'''
    rng = np.random.default_rng(seed)
    n = no + nv
    v = F.hermitian_integrals(rng, n)
    f = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    f = f + f.conj().T
    amps = [F.complex_antisym(rng, no, nv, k, s) for k, s in ((1, 0.3), (2, 0.2), (3, 0.1))]
    lams = [F.complex_antisym(rng, no, nv, k, s) for k, s in ((1, 0.3), (2, 0.2), (3, 0.1))]
    if real:
        v = v.real.copy()
        v = 0.5 * (v + v.transpose(2, 3, 0, 1))
        f, amps, lams = f.real.copy(), [x.real.copy() for x in amps], [x.real.copy() for x in lams]
    return f, v, amps, lams


def packed(x):
    return Q.unique(x, 3)


def reference(f, v, no, amps, lams):
    '''(g1, d1, Gnh) of the tables, dense.'''
    arr = LR._arrays(f, v, no, amps, lams)
    return L.density(arr, no)[0], LR.densities(amps, lams, no)[0], L.gradient(arr, no)


@pytest.mark.parametrize('no,nv,seed', [(4, 5, 3), (5, 4, 1), (3, 6, 2), (6, 3, 7), (2, 4, 5), (4, 2, 6)])
def test_density_and_gradient_equal_the_tables(no, nv, seed):
    '''g1 (the coefficient of the dressed Fock matrix), the response density
    d1 and the table GRADIENT, to 1e-13 of their size; also without triples
    (two occupied or two virtual spin orbitals).'''
    f, v, amps, lams = random_case(no, nv, seed)
    g1, d1, gnh = reference(f, v, no, amps, lams)
    r = D.kernel(container(f, v, no), amps[0], amps[1], packed(amps[2]), lams[0], lams[1],
                 packed(lams[2]))
    assert abs(r.g1 - g1).max() < 1e-13 * abs(g1).max()
    assert abs(r.d1 - d1).max() < 1e-13 * abs(d1).max()
    assert abs(r.gnh - gnh).max() < 1e-13 * abs(gnh).max()
    assert abs(np.trace(r.d1) - no) < 1e-12 and abs(r.d1 - r.d1.conj().T).max() > 1e-3
    # the density alone
    r1 = D.kernel(container(f, v, no), amps[0], amps[1], packed(amps[2]), lams[0], lams[1],
                  packed(lams[2]), gradient=False)
    assert r1.gnh is None and abs(r1.d1 - d1).max() < 1e-13 * abs(d1).max()


@pytest.mark.parametrize('plan', [D.Plan(nb=1, slab_bytes=1), D.Plan(nb=2, slab_bytes=16 * 15 * 7),
                                  D.Plan(nb=100, slab_bytes=None)])
def test_any_plan(plan):
    '''One occupied pair per batch and one row of <ab||de> per slab, batches
    and slabs that do not divide the ranges, everything at once: the same
    numbers.  Flat packed arrays and the full six-index arrays are taken as
    well.'''
    no, nv = 5, 6
    f, v, amps, lams = random_case(no, nv, 11)
    g1, d1, gnh = reference(f, v, no, amps, lams)
    r = D.kernel(container(f, v, no), amps[0], amps[1], packed(amps[2]).ravel(), lams[0], lams[1],
                 lams[2], plan=plan)
    assert r.plan is plan
    assert abs(r.d1 - d1).max() < 1e-13 * abs(d1).max()
    assert abs(r.gnh - gnh).max() < 1e-13 * abs(gnh).max()


def test_real_and_without_the_kernels(monkeypatch):
    '''Real integrals and amplitudes stay real; the numpy forms of the
    gathers give the same as the C kernels.'''
    no, nv = 4, 5
    f, v, amps, lams = random_case(no, nv, 12, real=True)
    assert not np.iscomplexobj(v) and abs(v - v.transpose(2, 3, 0, 1)).max() < 1e-14
    g1, d1, gnh = reference(f, v, no, amps, lams)
    args = (container(f, v, no), amps[0], amps[1], packed(amps[2]), lams[0], lams[1], packed(lams[2]))
    r = D.kernel(*args)
    assert r.d1.dtype == np.float64 and r.gnh.dtype == np.float64
    assert abs(r.d1 - d1).max() < 1e-13 * abs(d1).max() and abs(r.gnh - gnh).max() < 1e-13 * abs(gnh).max()
    monkeypatch.setattr(t3pack, 'USE_C', False)
    r2 = D.kernel(*args)
    assert abs(r2.d1 - r.d1).max() < 1e-13 * abs(d1).max() and abs(r2.gnh - r.gnh).max() < 1e-13 * abs(gnh).max()


def test_without_triples_it_is_ccsd():
    '''t3 = lambda3 = None: the tables without their terms of T3 and
    Lambda3.'''
    no, nv = 4, 5
    f, v, amps, lams = random_case(no, nv, 13)
    zero = np.zeros_like(amps[2])
    g1, d1, gnh = reference(f, v, no, amps[:2] + [zero], lams[:2] + [zero])
    r = D.kernel(container(f, v, no), amps[0], amps[1], None, lams[0], lams[1], None)
    assert abs(r.d1 - d1).max() < 1e-13 * abs(d1).max() and abs(r.gnh - gnh).max() < 1e-13 * abs(gnh).max()
    full = reference(f, v, no, amps, lams)
    assert abs(full[2] - gnh).max() > 1e-2 * abs(gnh).max()


# ------------------------------------------------------------------ molecules

def _ccsdt(eris, tol=1e-12):
    r = solver.kernel(eris, 'ccsdt', solver.Options(conv_tol=tol, conv_tol_normt=1e-10, incore_complete=True))
    assert r.converged
    return r


def _lambda(eris, r, tol=1e-11):
    '''(l1, l2, l3 packed) of the dense reference for the amplitudes of the
    packed solver.'''
    fock, g, mo_e, nocc = R.integrals(eris)
    nvir = len(mo_e) - nocc
    t3 = Q.expand(np.asarray(r.t3).reshape(len(Q.tuples(nocc, 3)), -1), nocc, nvir, 3)
    rl = LR.kernel(fock, g, mo_e, nocc, [r.t1, r.t2, t3], conv_tol_normt=tol)
    assert rl.converged
    return rl.lams[0], rl.lams[1], packed(rl.lams[2]), (fock, g, nocc, [r.t1, r.t2, t3], rl.lams)


@pytest.mark.parametrize('cplx', [False, True])
def test_converged_amplitudes_of_water(cplx):
    '''H2O / STO-3G, real and complex spin orbitals, converged CCSDT (the
    packed solver) and its Lambda: the density and the gradient of the dense
    reference, the gradient from the full bare densities as well (which is
    the table only for converged amplitudes).'''
    mol = gto.M(atom=WATER, basis='sto-3g', verbose=0)
    ghf = scf.addons.convert_to_ghf(scf.RHF(mol).run(conv_tol=1e-13))
    C = _spin_mixed(ghf)[0] if cplx else ghf.mo_coeff
    eris = E.from_scf(ghf, mo_coeff=C)
    r = _ccsdt(eris)
    l1, l2, l3, (fock, g, nocc, amps, lams) = _lambda(eris, r)
    res = D.kernel(eris, r.t1, r.t2, r.t3, l1, l2, l3)
    assert (abs(np.asarray(eris.oovv).imag).max() > 0.01) == cplx
    d1 = LR.densities(amps, lams, nocc)[0]
    gnh = LR.orbital_gradient(fock, g, nocc, amps, lams, dressed=True)
    scale = abs(gnh).max()
    bare = abs(res.gnh - LR.orbital_gradient(fock, g, nocc, amps, lams)).max()
    print('H2O / STO-3G %s: |Gnh| %.2e, off the dense table %.1e, off the bare densities %.1e, d1 %.1e'
          % ('complex' if cplx else 'real', scale, abs(res.gnh - gnh).max(), bare, abs(res.d1 - d1).max()))
    assert scale > 1e-5                             # (1.8e-5 in this basis)
    assert abs(res.d1 - d1).max() < 1e-13
    assert abs(res.gnh - gnh).max() < 1e-12 * scale
    assert bare < 1e-6 * scale


def _field(mf, V, eps):
    h0 = mf.get_hcore()
    mf.get_hcore = lambda *args: h0 + eps * V
    return mf


def _relaxed_and_finite_field(make_scf, V, mix=None, eps=1e-3, nfrozen=0, ndrop=0):
    '''(relaxed, unrelaxed, four-point difference) of the dipole derivative:
    make_scf(eps) is the SCF in the field eps V (AO basis of the SCF); the
    ``nfrozen`` lowest occupied spinors frozen and the ``ndrop`` highest
    virtual ones left out.'''
    def run(e):
        mf = make_scf(e)
        C = mf.mo_coeff if mix is None else mix(mf)
        n = C.shape[1]
        eris = E.from_scf(mf, mo_coeff=C[:, :n - ndrop], mo_occ=mf.mo_occ[:n - ndrop], frozen=nfrozen)
        return mf, C, eris, _ccsdt(eris, 1e-13)
    mf, C, eris, r = run(0.0)
    n = C.shape[1]
    assert (mf.mo_occ[:-1] >= mf.mo_occ[1:]).all()
    l1, l2, l3, _ = _lambda(eris, r)
    rel = D.relaxed_rdm1(mf, eris, r.t1, r.t2, r.t3, l1, l2, l3, C[:, nfrozen:n - ndrop],
                         frozen_occ=C[:, :nfrozen] if nfrozen else None,
                         dropped_vir=C[:, n - ndrop:] if ndrop else None, tol=1e-11)
    assert rel.converged and abs(rel.relaxed - rel.relaxed.conj().T).max() < 1e-13
    assert abs(np.trace(rel.relaxed) - eris.nocc - nfrozen) < 1e-11
    assert abs(rel.mo_coeff - C).max() == 0 and rel.nocc == eris.nocc + nfrozen
    if nfrozen or ndrop:
        # the rotations between frozen and active spinors of a kind are in the relaxed density only
        fr, ac = (slice(0, nfrozen), slice(nfrozen, rel.nocc)) if nfrozen else (
            slice(n - ndrop, n), slice(rel.nocc, n - ndrop))
        assert abs(rel.unrelaxed[ac, fr]).max() == 0 and abs(rel.relaxed[ac, fr]).max() > 1e-6
    Vmo = C.conj().T @ V @ C
    e = {}
    for k in (-2, -1, 1, 2):
        m, _, _, rk = run(k * eps)
        e[k] = m.e_tot + float(np.real(rk.e_corr))
    fd = (8 * (e[1] - e[-1]) - (e[2] - e[-2])) / (12 * eps)
    return (np.einsum('pq,qp', Vmo, rel.relaxed).real, np.einsum('pq,qp', Vmo, rel.unrelaxed).real, fd)


@pytest.mark.parametrize('cplx', [False, True])
def test_relaxed_dipole_spin_orbitals(cplx):
    '''H2O / STO-3G as spin orbitals (real, and every pair mixed by its own
    SU(2) matrix): the relaxed density with z against the derivative of
    E(SCF) + E_corr(CCSDT), the SCF repeated in the field.'''
    mol = gto.M(atom=WATER, basis='sto-3g', verbose=0)
    with mol.with_common_orig((0, 0, 0)):
        z = mol.intor('int1e_r')[2]
    z2 = np.kron(np.eye(2), z)

    def make_scf(eps):
        mf = _field(scf.RHF(mol), z, eps)
        mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-10, 200
        mf.kernel()
        ghf = scf.addons.convert_to_ghf(mf)
        h2 = np.kron(np.eye(2), mf.get_hcore())
        ghf.get_hcore = lambda *args: h2
        return ghf

    relaxed, unrelaxed, fd = _relaxed_and_finite_field(
        make_scf, z2, (lambda m: _spin_mixed(m)[0]) if cplx else None)
    print('H2O / STO-3G %s: relaxed %.12f unrelaxed %.12f finite field %.12f, difference %.1e'
          % ('complex' if cplx else 'real', relaxed, unrelaxed, fd, relaxed - fd))
    assert abs(relaxed - fd) < 2e-10
    assert abs(unrelaxed - relaxed) > 5e-6


def test_relaxed_dipole_x2c_spinors():
    '''H2O / STO-3G with the X2C Hamiltonian (pyscf.x2c.UHF: spinors, complex
    integrals), the field z of the spinor AO basis added to the one-electron
    Hamiltonian of the SCF and of the CC alike.'''
    mol = gto.M(atom=WATER, basis='sto-3g', verbose=0)
    with mol.with_common_orig((0, 0, 0)):
        z = mol.intor('int1e_r_spinor')[2]

    def make_scf(eps):
        mf = _field(x2c.UHF(mol), z, eps)
        mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-10, 200
        mf.kernel()
        assert mf.converged
        return mf

    relaxed, unrelaxed, fd = _relaxed_and_finite_field(make_scf, z)
    print('H2O / STO-3G X2C: relaxed %.12f unrelaxed %.12f finite field %.12f, difference %.1e'
          % (relaxed, unrelaxed, fd, relaxed - fd))
    assert abs(relaxed - fd) < 2e-10
    assert abs(unrelaxed - relaxed) > 5e-6


def test_ccsd_limit_is_the_spinor_ccsd_code():
    '''Without T3 and Lambda3, with the amplitudes and lambda of the spinor
    CCSD (zccsdt.zccsd.DirectZCCSD) in its own spinors: the unrelaxed and
    the relaxed density of zccsd.zvector.kernel, which is validated against
    CFOUR.  H2O / STO-3G, X2C.'''
    from zccsdt.zccsd import DirectZCCSD, zvector
    mol = gto.M(atom=WATER, basis='sto-3g', verbose=0)
    mf = x2c.UHF(mol)
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-10, 200
    mf.kernel()
    mycc = DirectZCCSD(mf)
    mycc.verbose, mycc.conv_tol, mycc.conv_tol_normt, mycc.max_cycle = 0, 1e-13, 1e-10, 200
    mycc.kernel()
    l1, l2 = mycc.solve_lambda()
    want, plain, z = zvector.kernel(mycc, l1=l1, l2=l2, mf=mf, tol=1e-11)
    C = zvector.full_space(mycc)[0]
    eris = E.from_scf(mf, mo_coeff=C)
    assert abs(eris.mo_energy - mycc.eris.mo_energy).max() < 1e-9
    rel = D.relaxed_rdm1(mf, eris, mycc.t1, mycc.t2, None, l1, l2, None, C, tol=1e-11)
    print('CCSD limit: unrelaxed %.1e, relaxed %.1e, z %.1e off zccsd.zvector (relaxation %.1e)'
          % (abs(rel.unrelaxed - plain).max(), abs(rel.relaxed - want).max(), abs(rel.z - z).max(),
             abs(want - plain).max()))
    assert abs(rel.unrelaxed - plain).max() < 1e-10
    assert abs(rel.relaxed - want).max() < 1e-9
    assert abs(want - plain).max() > 1e-4


# --------------------------------------------------------------------- memory

def test_plan_memory():
    '''No limit: everything at once.  A budget: the batches of the passes
    and the slab of dressed <ab||de> take what is left, more with more
    (never less), and MemoryError below one occupied pair and the rows of
    one first index.'''
    o, v = 10, 106
    nvp, npo = v * (v - 1) // 2, o * (o - 1) // 2
    free = D.plan_memory(o, v)
    assert free.slab_bytes is None and free.nb == min(npo, D.BATCH_BYTES // D.pair_bytes(o, v))
    fixed, unit, pp = D.fixed_bytes(o, v), D.pair_bytes(o, v), 16 * nvp * nvp
    assert D.plan_memory(o, v, budget=free.need) == free
    with pytest.raises(MemoryError, match='densities of CCSDT'):
        D.plan_memory(o, v, budget=fixed + unit - 1)
    least = D.plan_memory(o, v, budget=fixed + unit)
    assert least.nb == 1 and least.slab_bytes == unit // (16 * nvp) * 16 * nvp >= 16 * v * nvp
    last = least
    for gb in np.arange(1.0, 12.0, 0.25):
        budget = int(gb * 1e9)
        if budget < fixed + unit:
            continue
        p = D.plan_memory(o, v, budget=budget)
        assert p.need <= budget and p.nb >= last.nb
        assert (p.slab_bytes or pp) >= (last.slab_bytes or pp) or last.slab_bytes is None
        if p.slab_bytes is not None:        # nothing more would fit
            assert p.nb == free.nb or fixed + (p.nb + 1) * unit > budget
            assert p.slab_bytes == pp or fixed + p.slab_bytes + 16 * nvp > budget
        last = p
    assert last == free


@pytest.mark.parametrize('tight', [True, False])
def test_allocations_stay_within_the_plan(tight):
    '''What kernel() allocates (tracemalloc follows numpy's allocations) is
    within the plan, for the smallest plan - one occupied pair per batch,
    the rows of one first index of <ab||de> per slab - and for everything
    at once; and the plan is not far above it.  5 + 40 spin orbitals.'''
    import tracemalloc
    from zccsdt.cc import ccsdt_opt
    no, nv = 5, 40
    nvp = nv * (nv - 1) // 2
    f, v, amps, lams = random_case(no, nv, 21)
    er = container(f, v, no)
    args = (er, amps[0], amps[1], packed(amps[2]), lams[0], lams[1], packed(lams[2]))
    del amps, lams, f, v
    ccsdt_opt.vvvv_packed(er, t3pack.triple_map(nv))        # (the packed integrals: of the container)
    fixed, unit = D.fixed_bytes(no, nv), D.pair_bytes(no, nv)
    budget = fixed + max(unit, 16 * nv * nvp) if tight else None
    plan = D.plan_memory(no, nv, budget=budget)
    assert (plan.nb, plan.slab_bytes is None) == ((1, False) if tight else (10, True))
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    r = D.kernel(*args, plan=plan)
    peak = tracemalloc.get_traced_memory()[1] - base
    tracemalloc.stop()
    need = plan.need - ccsdt_opt.overhead()
    print('%s: allocated %.1f MB, planned %.1f MB (%.2f)' % ('one pair, one slab' if tight else 'everything at once',
                                                             peak / 1e6, need / 1e6, peak / need))
    assert 0.6 * need < peak < need
    assert abs(np.trace(r.d1) - no) < 1e-10


# ------------------------------------------------------------- frozen spinors

def _gamma2(amps, lams, no):
    '''(gamma1, gamma2) of the bare integrals from the dense reference: d1
    and d2 without the reference and the products with it.'''
    d1, d2 = LR.densities(amps, lams, no)
    occ = np.zeros(d1.shape[0])
    occ[:no] = 1.0
    g1 = d1 - np.diag(occ)
    x = np.einsum('pr,qs->pqrs', g1 + 0.5 * np.diag(occ), np.diag(occ))
    x = x - x.transpose(1, 0, 2, 3)
    return g1, d2 - (x - x.transpose(0, 1, 3, 2))


@pytest.mark.parametrize('no,nv,seed', [(4, 5, 31), (5, 4, 32), (3, 6, 33), (2, 4, 34)])
def test_integrals_with_a_frozen_index(no, nv, seed, monkeypatch):
    '''The contractions of the two-particle density with integrals <fq||rs>
    that have one index f outside the active space (random numbers,
    antisymmetric in r and s; three "frozen spinors", one at a time and all
    at once) against the dense densities: B[f,x] = 1/2 sum <fq||rs>
    gamma2[x,q,r,s] and K[f,x] = 1/2 sum <pq||fs> gamma2[p,q,x,s] with
    <pq||fs> = conj(<fs||pq>).'''
    f, v, amps, lams = random_case(no, nv, seed)
    n, nf = no + nv, 3
    rng = np.random.default_rng(seed)
    S0 = rng.standard_normal((nf, n, n, n)) + 1j * rng.standard_normal((nf, n, n, n))
    S0 = S0 - S0.transpose(0, 1, 3, 2)
    g1, g2 = _gamma2(amps, lams, no)
    B = 0.5 * np.einsum('fqrs,xqrs->fx', S0, g2)
    K = 0.5 * np.einsum('fspq,pqxs->fx', S0.conj(), g2)

    def fock_of(M):
        return np.einsum('fqrs,qs->fr', S0, M), np.einsum('fwtu,uw->tf', S0.conj(), M)
    args = (container(f, v, no), amps[0], amps[1], packed(amps[2]), lams[0], lams[1], packed(lams[2]))
    calls = []

    def ints(f0, f1):
        calls.append((f0, f1))
        return S0[f0:f1]
    for per_batch in (3, 1):
        monkeypatch.setattr(D, 'frozen_batch', lambda *a: per_batch)
        r = D.kernel(*args, frozen=(nf, ints), plan=D.Plan(nb=2, slab_bytes=None))
        Bk, Kk = D.frozen_terms(r, amps[0], fock_of)
        assert abs(Bk - B).max() < 1e-12 * abs(B).max(), abs(Bk - B).max() / abs(B).max()
        assert abs(Kk - K).max() < 1e-12 * abs(K).max(), abs(Kk - K).max() / abs(K).max()
    assert calls == [(0, 3), (0, 3), (0, 1), (1, 2), (2, 3), (0, 1), (1, 2), (2, 3)]
    # (the terms of gamma1 and t1 are part of it)
    assert abs(r.bf - B).max() > 1e-3 * abs(B).max()


def _x2c_in_field(mol):
    with mol.with_common_orig((0, 0, 0)):
        z = mol.intor('int1e_r_spinor')[2]

    def make_scf(eps):
        mf = _field(x2c.UHF(mol), z, eps)
        mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-10, 200
        mf.kernel()
        assert mf.converged
        return mf
    return make_scf, z


@pytest.mark.parametrize('basis,nfrozen,ndrop', [('sto-3g', 2, 0), ('6-31g', 2, 4)])
def test_relaxed_dipole_with_frozen_spinors(basis, nfrozen, ndrop):
    '''CCSDT of H2O with X2C spinors, the two lowest occupied spinors
    frozen and (6-31G) the four highest virtual ones left out: the relaxed
    density in the basis of all spinors against the derivative of E(SCF) +
    E_corr(CCSDT) with the SCF repeated in the field - the gradient between
    frozen and active spinors from the integrals with one frozen index
    (density_opt.frozen_integrals), the rotations between frozen and active
    spinors of a kind from the Fock matrix staying diagonal.'''
    make_scf, z = _x2c_in_field(gto.M(atom=WATER, basis=basis, verbose=0))
    relaxed, unrelaxed, fd = _relaxed_and_finite_field(make_scf, z, nfrozen=nfrozen, ndrop=ndrop)
    print('H2O / %s X2C, %d frozen, %d dropped: relaxed %.12f unrelaxed %.12f finite field %.12f, difference %.1e'
          % (basis, nfrozen, ndrop, relaxed, unrelaxed, fd, relaxed - fd))
    assert abs(relaxed - fd) < 2e-10
    assert abs(unrelaxed - relaxed) > 5e-6


def test_frozen_spin_orbitals_real_and_complex():
    '''The same with RHF orbitals as spin orbitals (real integrals: the
    transformation of pyscf.ao2mo) and with every pair mixed by its own
    SU(2) matrix (complex): the two relaxed densities give the same dipole,
    that of the finite field.  H2O / STO-3G, the 1s pair frozen.'''
    mol = gto.M(atom=WATER, basis='sto-3g', verbose=0)
    with mol.with_common_orig((0, 0, 0)):
        z = mol.intor('int1e_r')[2]
    z2 = np.kron(np.eye(2), z)

    def make_scf(eps):
        mf = _field(scf.RHF(mol), z, eps)
        mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-10, 200
        mf.kernel()
        ghf = scf.addons.convert_to_ghf(mf)
        h2 = np.kron(np.eye(2), mf.get_hcore())
        ghf.get_hcore = lambda *args: h2
        return ghf
    real = _relaxed_and_finite_field(make_scf, z2, nfrozen=2)
    cplx = _relaxed_and_finite_field(make_scf, z2, lambda m: _spin_mixed(m)[0], nfrozen=2)
    assert abs(real[0] - real[2]) < 2e-10 and abs(cplx[0] - cplx[2]) < 2e-10
    assert abs(real[0] - cplx[0]) < 1e-11 and abs(real[0] - real[1]) > 5e-6


@pytest.mark.parametrize('frozen', [[0, 1], [22, 23, 24, 25], [0, 1, 22, 23, 24, 25]])
def test_ccsd_limit_with_frozen_spinors(frozen):
    '''Frozen occupied spinors, dropped virtual ones and both, without T3
    and Lambda3 and with the amplitudes of the spinor CCSD: the relaxed
    density of zccsd.zvector.kernel (zccsd.orbgrad_frozen), which is
    validated against CFOUR.  H2O / 6-31G, X2C.'''
    from zccsdt.zccsd import DirectZCCSD, zvector
    mol = gto.M(atom=WATER, basis='6-31g', verbose=0)
    mf = _x2c_in_field(mol)[0](0.0)
    mycc = DirectZCCSD(mf, frozen=frozen)
    mycc.verbose, mycc.conv_tol, mycc.conv_tol_normt, mycc.max_cycle = 0, 1e-13, 1e-10, 200
    mycc.kernel()
    l1, l2 = mycc.solve_lambda()
    want, plain, z = zvector.kernel(mycc, l1=l1, l2=l2, mf=mf, tol=1e-11)
    C = zvector.full_space(mycc)[0]
    n, nocc = C.shape[1], 10
    fo, fv = [f for f in frozen if f < nocc], [f for f in frozen if f >= nocc]
    act = [p for p in range(n) if p not in frozen]
    eris = E.from_scf(mf, mo_coeff=C[:, fo + act], mo_occ=mf.mo_occ[fo + act], frozen=len(fo))
    rel = D.relaxed_rdm1(mf, eris, mycc.t1, mycc.t2, None, l1, l2, None, C[:, act],
                         frozen_occ=C[:, fo] if fo else None, dropped_vir=C[:, fv] if fv else None, tol=1e-11)
    print('CCSD limit, frozen %s: unrelaxed %.1e, relaxed %.1e off zccsd.zvector (relaxation %.1e)'
          % (frozen, abs(rel.unrelaxed - plain).max(), abs(rel.relaxed - want).max(), abs(want - plain).max()))
    assert abs(rel.mo_coeff - C).max() == 0
    assert abs(rel.unrelaxed - plain).max() < 1e-10
    assert abs(rel.relaxed - want).max() < 1e-9


def test_front_end_layout_of_a_linear_molecule():
    '''What zccsdt.GCCSDT hands over for a linear molecule: real spinors
    adapted to m_j, in the order of their labels, and T3 in the symmetry
    blocks of its engine - and lambda3 in the same blocks.  HF / 6-31G with
    X2C, the 1s pair of fluorine frozen: the relaxed dipole derivative
    against that of the total energy of the front end with the SCF repeated
    in the field.'''
    from math import comb
    import zccsdt
    from zccsdt.cc import ccsdt_opt
    mol = gto.M(atom='H 0 0 0; F 0 0 0.9168', basis='6-31g', verbose=0)
    make_scf, z = _x2c_in_field(mol)

    def run(eps):
        mf = make_scf(eps)
        mycc = zccsdt.GCCSDT(mf, frozen=2)
        mycc.verbose, mycc.conv_tol, mycc.conv_tol_normt = 0, 1e-13, 1e-10
        mycc.kernel()
        assert mycc.converged
        return mf, mycc
    mf, mycc = run(0.0)
    eris = mycc.ao2mo()
    om, vm = t3pack.maps_for(eris)
    o, v = mycc.t1.shape
    assert om.spins is not None and eris.dtype == np.float64 and np.asarray(mycc.t3).size < comb(o, 3) * comb(v, 3)
    # Lambda of the dense reference, lambda3 put into the blocks of the engine
    lex = t3pack.triple_map(v)
    vvvv = ccsdt_opt._unpack_pairs(np.asarray(eris._vvvv_pp).T, lex, lex).transpose(2, 3, 0, 1)
    full = E.ERIs(vvvv=vvvv, **{k: getattr(eris, k) for k in (
        'nocc', 'fock', 'mo_energy', 'e_hf', 'oooo', 'ooov', 'oovv', 'ovov', 'ovvo', 'ovvv')})
    fock, g, mo_e, _ = R.integrals(full)
    t3 = Q.expand(ccsdt_opt.packed_t3(eris, mycc.t3), o, v, 3)
    rl = LR.kernel(fock, g, mo_e, o, [mycc.t1, mycc.t2, t3], conv_tol_normt=1e-11)
    assert rl.converged

    def order(m):
        y = m.n - 1 - m.triples
        return comb(m.n, 3) - 1 - y[:, 0] * (y[:, 0] - 1) * (y[:, 0] - 2) // 6 - y[:, 1] * (y[:, 1] - 1) // 2 - y[:, 2]
    eng = ccsdt_opt.engine_for(eris, np.asarray(mycc.t3).dtype)
    l3 = eng.L.from_dense(np.ascontiguousarray(packed(rl.lams[2])[np.ix_(order(om), order(vm))]))
    assert l3.shape == np.asarray(mycc.t3).shape
    # the spinors of the integrals, as the front end has them
    fo, _, _, _ = mycc._active()
    oc, vi = mycc._sorted
    Cs = mycc._axial()
    rel = D.relaxed_rdm1(mf, eris, mycc.t1, mycc.t2, mycc.t3, rl.lams[0], rl.lams[1], l3,
                         Cs[:, np.concatenate((oc, vi))], frozen_occ=Cs[:, fo], tol=1e-11)
    gnh = LR.orbital_gradient(fock, g, o, [mycc.t1, mycc.t2, t3], rl.lams, dressed=True)
    assert abs(rel.result.gnh - gnh).max() < 1e-9 * abs(gnh).max()
    Vmo = rel.mo_coeff.conj().T @ z @ rel.mo_coeff
    relaxed = np.einsum('pq,qp', Vmo, rel.relaxed).real
    eps = 1e-3
    e = {k: run(k * eps)[1].e_tot for k in (-2, -1, 1, 2)}
    fd = (8 * (e[1] - e[-1]) - (e[2] - e[-2])) / (12 * eps)
    print('HF / 6-31G X2C through the front end: relaxed %.12f finite field %.12f, difference %.1e'
          % (relaxed, fd, relaxed - fd))
    assert abs(relaxed - fd) < 2e-10
