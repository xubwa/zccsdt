'''The Lambda equations of CCSDT on packed Lambda_3 (zccsdt.cc.lambda_opt)
against the dense reference (zccsdt.cc.lambda_ref, the tables of
zccsdt.cc.lambda_terms):

- every group of terms of the Lambda_3 residual and the residuals of rank 1
  and 2, for random complex and real amplitudes that solve nothing and
  random Hermitian integrals;
- converged Lambda for a model Hamiltonian, for H2O / STO-3G with real and
  with complex spin orbitals and for X2C spinors of HBr;
- the dipole derivative of the CCSDT energy at fixed orbitals from the
  densities of the converged amplitudes;
- the memory plan against what a run allocates.'''
import gc
import os
import tracemalloc

import numpy as np
import pytest
from pyscf import gto, scf, x2c

import fockspace as F
from test_ccsdtq_ref import _spin_mixed
from zccsdt import eris as E
from zccsdt.cc import (ccsdt_opt, ccsdtq_ref as R, ccsdtq_terms as Q, lambda_opt as LO,
                       lambda_ref as LR, lambda_terms as LT, solver, t3kern, t3pack)
from zccsdt.genbas import read_genbas

ROOT = os.path.join(os.path.dirname(__file__), '..', 'bench', 'refs')

# the terms of LAMBDA[3] by the step of the iteration that evaluates them
GROUPS = {'outer': (0, 1), 'drive2': (2, 3), 'pp': (4, 5, 6, 7), 'hh': (8, 9, 13, 14),
          'ring': (10, 11), 'drive3': (12, 15)}


def _model_eris(no, nv, seed=1, cplx=True, scale=0.3):
    '''An integral object with random Hermitian <pq||rs> and a Fock matrix
    that is not diagonal.'''
    n = no + nv
    rng = np.random.default_rng(seed)
    v = scale * F.hermitian_integrals(rng, n)
    e = np.concatenate([-1.0 - 0.3 * rng.random(no), 1.0 + 0.3 * rng.random(nv)])
    x = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    f = np.diag(e) + scale * 0.5 * (x + x.conj().T)
    if not cplx:
        v, f = v.real.copy(), f.real.copy()
    o, vir = slice(0, no), slice(no, None)
    blocks = {k: np.ascontiguousarray(v[tuple(o if c == 'o' else vir for c in k)])
              for k in E.BLOCKS}
    return E.ERIs(nocc=no, fock=f, mo_energy=f.diagonal().real.copy(), e_hf=0.0, **blocks)


def _amps(rng, no, nv, cplx, scale=0.3):
    out = []
    for k in (1, 2, 3):
        x = F.random_antisym(rng, no, nv, k)
        if cplx:
            x = x + 1j * F.random_antisym(rng, no, nv, k)
        out.append(scale * x)
    return out


def _rel(a, b):
    return abs(a - b).max() / abs(b).max()


@pytest.mark.parametrize('no,nv,cplx,mem_mb,use_c', [
    (4, 5, True, 2000.0, True), (5, 6, False, 2000.0, True),
    (5, 6, True, 1e-3, True),           # one occupied pair and one triple per batch
    (4, 6, True, 2000.0, False), (5, 5, False, 1e-3, False)])
def test_residuals_against_the_tables(no, nv, cplx, mem_mb, use_c, monkeypatch):
    '''Each group of terms of the Lambda_3 residual, and the three
    residuals, for amplitudes and Lambda that solve nothing: equal to the
    dense evaluation of the tables to 1e-12 of their size.  With the C
    kernels and with the numpy forms of the gathers and reductions.'''
    if use_c and not (t3kern.USE_C and t3pack.USE_C):
        pytest.skip('C libraries not built')
    monkeypatch.setattr(t3kern, 'USE_C', use_c)
    monkeypatch.setattr(t3pack, 'USE_C', use_c)
    monkeypatch.setattr(LO, 'USE_C', use_c and LO.USE_C)
    eris = _model_eris(no, nv, seed=no + nv, cplx=cplx)
    rng = np.random.default_rng(7)
    t, l = _amps(rng, no, nv, cplx), _amps(rng, no, nv, cplx)
    fock, g, mo_e, nocc = R.integrals(eris)
    T3p, L3p = Q.unique(t[2], 3), Q.unique(l[2], 3)
    im = LO.make_intermediates(eris, t[0], t[1], T3p, mem_mb=mem_mb)
    assert im.eng.nb == (1 if mem_mb < 1 else no * (no - 1) // 2)
    arrays = LR._arrays(fock, g, nocc, t, l)
    for name, idx in GROUPS.items():
        ref = Q.asym(LT.total([LT.LAMBDA[3][i] for i in idx], arrays, nocc, 'ijkabc'), 3)
        got = LO.residuals(im, l[0], l[1], L3p, T3p, terms=[name])[2]
        assert abs(ref).max() > 1.0 and _rel(got, Q.unique(ref, 3)) < 1e-12, name
    assert sorted(i for idx in GROUPS.values() for i in idx) == list(range(len(LT.LAMBDA[3])))
    ref = LR.residuals(l, t, fock, g, nocc)
    r1, r2, r3 = LO.residuals(im, l[0], l[1], L3p, T3p)
    assert _rel(r1, ref[0]) < 1e-12
    assert _rel(r2, Q.unique(ref[1], 2)) < 1e-12
    assert _rel(r3, Q.unique(ref[2], 3)) < 1e-12
    assert all(np.iscomplexobj(x) == cplx for x in (r1, r2, r3))
    # the parts of the residuals of rank 1 and 2 by the rank of Lambda in them
    for sel in range(3):
        ll = [x if k == sel else 0 * x for k, x in enumerate(l)]
        ref = LR.residuals(ll, t, fock, g, nocc)
        r1, r2, r3 = LO.residuals(im, ll[0], ll[1], Q.unique(ll[2], 3), T3p)
        assert _rel(r1, ref[0]) < 1e-12 and _rel(r2, Q.unique(ref[1], 2)) < 1e-12, sel
    # a Jacobi step, into a given array
    out = np.empty(L3p.size, dtype=r3.dtype)
    new = LO.update_lams(im, l[0], l[1], L3p, T3p, out=out)
    ref = LR.residuals(l, t, fock, g, nocc)
    den = [R.denominator(mo_e, nocc, k) for k in (1, 2, 3)]
    want = [x + r / d for x, r, d in zip(l, ref, den)]
    assert new[2] is out or np.shares_memory(new[2], out)
    assert _rel(new[0], want[0]) < 1e-12 and _rel(new[1], want[1]) < 1e-12
    assert _rel(new[2].reshape(L3p.shape), Q.unique(want[2], 3)) < 1e-12


@pytest.mark.parametrize('cplx', [False, True])
def test_pair_kernels(cplx, monkeypatch):
    '''csrc/lambda_t3.c against the numpy forms and against the definitions:
    an array packed over a virtual pair on both orders of the pair, and its
    contraction with the packed columns of an operator on pairs (more pairs
    than one block of the kernel).'''
    if not LO.USE_C:
        pytest.skip('libzc_lambda not built')
    n, nx, ny = 13, 3, 4
    vm = t3pack.triple_map(n)
    md, ne = vm.pairs[:, 0], vm.pairs[:, 1]
    rng = np.random.default_rng(0)

    def rand(*sh):
        return rng.standard_normal(sh) + (1j * rng.standard_normal(sh) if cplx else 0)
    Y = rand(ny, nx, vm.npair)
    full = np.zeros((ny, nx, n, n), dtype=Y.dtype)
    full[:, :, md, ne] = Y
    full[:, :, ne, md] = -Y
    got = LO.pair_expand(Y, vm)
    assert got.dtype == Y.dtype and abs(got - full.transpose(0, 3, 1, 2)).max() == 0
    Y = rand(n, nx, vm.npair)
    W = rand(vm.npair, vm.npair)
    w = np.zeros((vm.npair, n, n), dtype=W.dtype)
    w[:, md, ne] = W
    w[:, ne, md] = -W
    ref = np.einsum('hxQ,Qah->xa', Y, w)
    got = LO.pair_matvec(Y, W, vm)
    assert abs(got - ref).max() < 1e-12 * abs(ref).max()
    monkeypatch.setattr(LO, 'USE_C', False)
    assert abs(LO.pair_matvec(Y, W, vm) - ref).max() < 1e-12 * abs(ref).max()
    Y = rand(ny, nx, vm.npair)
    full[:, :, md, ne] = Y
    full[:, :, ne, md] = -Y
    assert abs(LO.pair_expand(Y, vm) - full.transpose(0, 3, 1, 2)).max() == 0


def test_integrals_and_amplitudes_are_not_modified():
    no, nv = 4, 5
    eris = _model_eris(no, nv)
    rng = np.random.default_rng(3)
    t, l = _amps(rng, no, nv, True), _amps(rng, no, nv, True)
    T3p, L3p = Q.unique(t[2], 3), Q.unique(l[2], 3)
    before = {k: np.array(getattr(eris, k)) for k in E.BLOCKS + ('fock',)}
    keep = [x.copy() for x in (t[0], t[1], T3p, l[0], l[1], L3p)]
    im = LO.make_intermediates(eris, t[0], t[1], T3p)
    LO.update_lams(im, l[0], l[1], L3p, T3p)
    LO.update_lams(im, l[0], l[1], L3p, T3p)        # (the operands of an iteration are reused)
    for k, v in before.items():
        np.testing.assert_array_equal(getattr(eris, k), v, err_msg=k)
    for a, b in zip(keep, (t[0], t[1], T3p, l[0], l[1], L3p)):
        np.testing.assert_array_equal(a, b)


def test_model_hamiltonian():
    '''Converged Lambda of a complex Hermitian model (t1 up to 0.08): that
    of the dense reference, which is the left eigenvector of the matrix of
    exp(-T) H exp(T) (tests/test_lambda_ref.py); it is not conj(t).'''
    from test_lambda_ref import _model
    no, nv = 4, 5
    f, v = _model(no, nv)
    o, vir = slice(0, no), slice(no, None)
    eris = E.ERIs(nocc=no, fock=f, mo_energy=f.diagonal().real.copy(), e_hf=0.0,
                  **{k: np.ascontiguousarray(v[tuple(o if c == 'o' else vir for c in k)])
                     for k in E.BLOCKS})
    rt = R.kernel(f, v, f.diagonal().real, no, 'ccsdt', conv_tol=1e-13, conv_tol_normt=1e-11)
    ref = LR.kernel(f, v, f.diagonal().real, no, rt.amps, conv_tol_normt=1e-12)
    assert rt.converged and ref.converged
    r = LO.kernel(eris, rt.amps[0], rt.amps[1], Q.unique(rt.amps[2], 3),
                  solver.Options(conv_tol_normt=1e-12))
    assert r.converged and r.ncycle <= ref.ncycle + 2
    got = [r.l1, r.l2, LO.dense_l3(r.l3, no, nv)]
    for k in range(3):
        assert abs(got[k] - ref.lams[k]).max() < 1e-10
        assert abs(got[k] - rt.amps[k].conj()).max() > 1e-3
    # started from the solution: converged in one step, DIIS vectors in memory or not
    r = LO.kernel(eris, rt.amps[0], rt.amps[1], Q.unique(rt.amps[2], 3),
                  solver.Options(conv_tol_normt=1e-9, incore_complete=True), *got[:2], r.l3)
    assert r.converged and r.ncycle == 1


def _h2o(cplx):
    basis = {el: read_genbas(os.path.join(ROOT, 'GENBAS'), '%s:STO-3G' % el) for el in 'OH'}
    mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis=basis, verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-13)
    ghf = scf.addons.convert_to_ghf(mf)
    C = _spin_mixed(ghf)[0] if cplx else ghf.mo_coeff
    nao = mol.nao_nr()
    with mol.with_common_orig((0, 0, 0)):
        z = mol.intor('int1e_r')[2]
    zmo = C[:nao].conj().T @ z @ C[:nao] + C[nao:].conj().T @ z @ C[nao:]
    return mf, zmo, E.from_scf(ghf, mo_coeff=C)


@pytest.mark.parametrize('cplx', [False, True])
def test_h2o_and_the_dipole_derivative(cplx):
    '''H2O / STO-3G as spin orbitals, real and with every pair mixed by its
    own SU(2) matrix: amplitudes of the packed-T3 solver, Lambda of
    lambda_opt against the dense reference; real orbitals give real Lambda.
    The unrelaxed dipole from the densities of the dense code with this
    Lambda is the derivative of the CCSDT energy with respect to a field in
    the Hamiltonian at fixed orbitals (central differences of fourth
    order).'''
    mf, zmo, eris = _h2o(cplx)
    fock, g, mo_e, nocc = R.integrals(eris)
    assert (abs(g.imag).max() > 0.01) == cplx
    opts = solver.Options(conv_tol=1e-13, conv_tol_normt=1e-11)
    rt = solver.kernel(eris, 'ccsdt', opts)
    assert rt.converged
    nvir = len(mo_e) - nocc
    r = LO.kernel(eris, rt.t1, rt.t2, rt.t3, opts)
    assert r.converged and np.iscomplexobj(r.l3) == cplx
    amps = [rt.t1, rt.t2, LO.dense_l3(rt.t3, nocc, nvir)]
    ref = LR.kernel(fock, g, mo_e, nocc, amps, conv_tol_normt=1e-11)
    lams = [r.l1, r.l2, LO.dense_l3(r.l3, nocc, nvir)]
    for k in range(3):
        assert abs(lams[k] - ref.lams[k]).max() < 1e-9
        assert abs(lams[k] - np.conj(amps[k])).max() > 1e-4
    d1, d2 = LR.densities(amps, lams, nocc)
    dz = np.einsum('pq,pq', zmo, d1)
    assert abs(dz.imag) < 1e-12
    assert abs(LR.energy(fock, g, nocc, d1, d2) + mf.mol.energy_nuc() - mf.e_tot - rt.e_corr) < 1e-10

    def energy(eps):
        f = fock + eps * zmo
        x = R.kernel(f, g, mo_e, nocc, 'ccsdt', conv_tol=1e-13, conv_tol_normt=1e-11)
        assert x.converged
        return np.trace(f[:nocc, :nocc]).real + x.e_corr
    eps = 1e-3
    e = {k: energy(k * eps) for k in (-2, -1, 1, 2)}
    fd = (8 * (e[1] - e[-1]) - (e[2] - e[-2])) / (12 * eps)
    assert abs(fd - dz.real) < 2e-10
    assert abs(dz.real - np.trace(zmo[:nocc, :nocc]).real) > 1e-2


@pytest.fixture(scope='module')
def hbr():
    '''X2C spinors of HBr / cc-pVDZ-DK (off the z axis: complex integrals), 28 frozen, the 12 lowest virtual
    spinors: integrals without symmetry labels in use, converged CCSDT.'''
    import zccsdt
    mol = gto.M(atom='Br 0 0 0; H 0.50922 0.67896 1.1316', basis='cc-pvdz-dk', verbose=0)
    mf = x2c.UHF(mol)
    mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-8
    mf.kernel()
    cc = zccsdt.GCCSDT(mf, frozen=28)
    cc.verbose = 0
    cc.symmetry = cc.real_spinors = False     # the spinors of the SCF as they are: complex
    eris = E.drop_virtuals(cc.ao2mo(), 12)
    assert t3pack.maps_for(eris)[0].spins is None and np.iscomplexobj(eris.oovv)
    # (the dense reference takes the full <ab||cd>)
    a, b = np.triu_indices(12, 1)
    full = np.zeros((12,) * 4, dtype=complex)
    full[a[:, None], b[:, None], a, b] = eris._vvvv_pp
    full -= full.transpose(1, 0, 2, 3)
    eris.vvvv = full - full.transpose(0, 1, 3, 2)
    rt = solver.kernel(eris, 'ccsdt', solver.Options(conv_tol=1e-12, conv_tol_normt=1e-10))
    assert rt.converged
    return eris, rt


def test_hbr_x2c(hbr):
    '''Spinors: 8 occupied and 12 virtual, complex integrals.'''
    eris, rt = hbr
    nocc, nvir = rt.t1.shape
    assert (nocc, nvir) == (8, 12)
    r = LO.kernel(eris, rt.t1, rt.t2, rt.t3, solver.Options(conv_tol_normt=1e-10))
    assert r.converged
    t3 = LO.dense_l3(rt.t3, nocc, nvir)
    ref = LR.lambda_ccsdt(eris, rt.t1, rt.t2, t3, conv_tol_normt=1e-10)
    lams = [r.l1, r.l2, LO.dense_l3(r.l3, nocc, nvir)]
    for k in range(3):
        assert abs(lams[k] - ref[k]).max() < 1e-9
    assert abs(lams[2] - t3.conj()).max() > 1e-5 and abs(ref[2].imag).max() > 1e-5


@pytest.mark.parametrize('no,nv', [(4, 36), (6, 30)])
def test_memory_plan(no, nv):
    '''What the intermediates and an iteration allocate (tracemalloc follows
    numpy's allocations) stays within the plan, stage by stage, with all
    occupied pairs in one batch and with the smallest buffers, and the plan
    is not far above it; the buffers shrink to fit a budget; MemoryError
    below the floor.'''
    eris = _model_eris(no, nv, seed=5, cplx=True, scale=0.05)
    rng = np.random.default_rng(1)
    t, l = _amps(rng, no, nv, True, 0.05), _amps(rng, no, nv, True, 0.05)
    T3p, L3p = Q.unique(t[2], 3), Q.unique(l[2], 3)
    del t[2], l[2]
    om, vm = t3pack.triple_map(no), t3pack.triple_map(nv)
    ccsdt_opt.vvvv_packed(eris, vm)             # (the packed bare <ab||de>: the integral object's)
    ccsdt_opt._bare_Q(eris, om, vm)
    f, pp = no * nv ** 3 * 16, vm.npair ** 2 * 16
    for mem_mb in (2000.0, 1e-3):
        plan = LO.plan_memory(no, nv, 16, None, mem_mb)
        gc.collect()
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        im = LO.make_intermediates(eris, t[0], t[1], T3p, mem_mb=plan.mem_mb)
        setup = tracemalloc.get_traced_memory()[1] - base
        kept = tracemalloc.get_traced_memory()[0] - base
        tracemalloc.reset_peak()
        out = np.empty(L3p.size, dtype=complex)         # (the residual: one T3 of the plan)
        LO.update_lams(im, l[0], l[1], L3p, T3p, out=out)
        LO.update_lams(im, l[0], l[1], L3p, T3p, out=out)
        peak = tracemalloc.get_traced_memory()[1] - base - out.nbytes
        tracemalloc.stop()
        where = (mem_mb, setup / f, kept / f, peak / f, plan.setup / f, plan.kept / f, plan.iteration / f)
        assert setup <= plan.setup + pp, where
        assert kept <= plan.kept + pp, where
        assert peak <= plan.iteration + pp, where
        assert peak > 0.8 * (plan.iteration + pp) and kept > 0.9 * (plan.kept + pp), where
        if mem_mb < 1:
            assert setup > 0.8 * (plan.setup + pp), where
        assert plan.need >= max(plan.setup, plan.iteration) + pp + 3 * L3p.nbytes
        del im, out
    full = LO.plan_memory(no, nv, 16, None, 2000.0)
    assert plan.mem_mb == 1e-3 and full.need > plan.need and full.diis == 0
    assert LO.plan_memory(no, nv, 16, None, 2000.0, diis_space=6).diis == 12 * L3p.nbytes
    # a budget between the two: the buffers take what is left
    mid = LO.plan_memory(no, nv, 16, (full.need + plan.need) // 2, 2000.0)
    assert plan.need < mid.need <= (full.need + plan.need) // 2 and 1e-3 < mid.mem_mb < 2000.0
    assert mid.need > 0.9 * (full.need + plan.need) // 2
    low = LO.plan_memory(no, nv, 16, plan.need, 2000.0)
    assert low.need == plan.need
    with pytest.raises(MemoryError):
        LO.plan_memory(no, nv, 16, plan.need - 1, 2000.0)
