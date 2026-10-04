'''The Lambda equations of DirectZCCSD (zccsdt.zccsd.lam): one step and the
converged amplitudes against PySCF's GCCSD Lambda on integrals made by brute
force from the AO integrals; the density they give against the derivative of
the energy with respect to a one-particle perturbation.'''

from types import SimpleNamespace

import numpy as np
import pytest
from pyscf import gto, x2c
from pyscf.cc import gccsd_lambda, gccsd_rdm

from zccsdt.zccsd import DirectZCCSD, amps, kramers, lam
from zccsdt.zccsd import kernels as K

HAVE_C = K.USE_C


@pytest.fixture(scope='module')
def x2c_h2o():
    mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis='6-31g', verbose=0)
    return x2c.UHF(mol).run(conv_tol=1e-11)


def reference_eris(eris, mol):
    '''The antisymmetrized integrals <pq||rs> of the spinors of ``eris`` by
    brute force from the AO integrals, as the blocks of PySCF's GCCSD.'''
    C, o = eris.mo_ao_spin, eris.nocc
    nao = mol.nao_nr()
    Ca, Cb = C[:nao], C[nao:]
    D = np.einsum('mp,nq->pqmn', Ca.conj(), Ca) + np.einsum('mp,nq->pqmn', Cb.conj(), Cb)
    g = np.einsum('pqmn,mnlk,rslk->pqrs', D, mol.intor('int2e'), D, optimize=True)  # (pq|rs)
    g = g.transpose(0, 2, 1, 3)
    g = g - g.transpose(0, 1, 3, 2)
    oc, vi = slice(0, o), slice(o, None)
    return SimpleNamespace(
        fock=eris.fock, mo_energy=eris.mo_energy, nocc=o,
        oooo=g[oc, oc, oc, oc], ooov=g[oc, oc, oc, vi], oovv=g[oc, oc, vi, vi],
        ovov=g[oc, vi, oc, vi], ovvo=g[oc, vi, vi, oc], ovvv=g[oc, vi, vi, vi],
        vvvv=g[vi, vi, vi, vi])


def amplitudes(o, v, rng, paired):
    '''Random complex t1-like and t2-like arrays, with the time-reversal
    symmetry of Kramers pairs if ``paired``.'''
    x1 = 0.05 * (rng.standard_normal((o, v)) + 1j * rng.standard_normal((o, v)))
    x2 = rng.standard_normal((o, o, v, v)) + 1j * rng.standard_normal((o, o, v, v))
    x2 = 0.02 * (x2 - x2.transpose(1, 0, 2, 3))
    x2 = x2 - x2.transpose(0, 1, 3, 2)
    if paired:
        x1, x2 = 0.5 * (x1 + kramers.image(x1)), 0.5 * (x2 + kramers.image(x2))
    return x1, np.ascontiguousarray(x2)


@pytest.mark.parametrize('kr', [False, None])
def test_step_equals_pyscf(x2c_h2o, monkeypatch, tmp_path, kr):
    '''The intermediates and one step for random complex t and lambda,
    without and with Kramers pairs (there: amplitudes without and with their
    symmetry); in one block and in slabs and blocks, with W in a scratch
    file, with the numpy kernels, with a level shift.'''
    monkeypatch.setenv('ZCCSDT_TMPDIR', str(tmp_path))
    mol = x2c_h2o.mol
    mycc = DirectZCCSD(x2c_h2o, frozen=2)
    mycc.verbose, mycc.kramers = 0, kr
    eris = mycc.ao2mo()
    assert bool(eris.kramers) == (kr is None)
    res = amps.Residual(eris, mycc.max_memory)
    o, v = res.nocc, res.nvir
    npo = K.npair(o)
    (io, jo), (av, bv) = K.tril(o), K.tril(v)
    ref = reference_eris(eris, mol)
    assert abs(res.oovv_p - ref.oovv[io, jo][:, av, bv]).max() < 1e-12
    assert abs(np.asarray(res.ovvv_p) - ref.ovvv[:, :, av, bv]).max() < 1e-12
    rng = np.random.default_rng(11)
    one = res.plan
    for paired in ((False, True) if eris.kramers else (False,)):
        t1, t2 = amplitudes(o, v, rng, paired)
        l1, l2 = amplitudes(o, v, rng, paired)
        t2p = K.pack_t2(t2)
        assert res.symmetric(t1, t2p) == paired
        imds = gccsd_lambda.make_intermediates(mycc, t1, t2, ref)
        res.plan = one
        L = lam.Lambda(res, t1, t2p)
        wovvo = np.asarray(imds.wovvo).transpose(0, 2, 3, 1)        # [j, b, k, c]
        woooo, wovoo = np.asarray(imds.woooo), np.asarray(imds.wovoo)
        woooo = (woooo - woooo.transpose(0, 1, 3, 2))[io, jo][:, io, jo]
        wovoo = (wovoo - wovoo.transpose(0, 1, 3, 2))[:, :, io, jo]

        def same_intermediates(L):
            assert abs(L.Hvv - imds.v1).max() < 1e-12 and abs(L.Hoo - imds.v2).max() < 1e-12
            assert abs(L.res1 - imds.w3.T).max() < 1e-12
            assert abs(np.asarray(L.W).reshape(o, v, o, v) - wovvo).max() < 1e-12
            assert abs(L.Whh - woooo).max() < 1e-12 and abs(L.Hq - wovoo).max() < 1e-12

        same_intermediates(L)
        for shift in (0.0, 0.3):
            monkeypatch.setattr(mycc, 'level_shift', shift)
            l1r, l2r = gccsd_lambda.update_lambda(mycc, t1, t2, l1, l2, ref, imds)
            l1n, l2n = L.update(l1, l2, shift)
            assert abs(l1n - l1r).max() < 1e-12 and abs(l2n - l2r).max() < 1e-12
        # slabs and blocks, W in a scratch file; the numpy kernels
        res.plan = amps.plan_step(o, v, mol.nao_nr(), None, kramers=bool(eris.kramers))
        res.plan.jblocks = [(0, 2), (2, 4), (4, o)]
        res.plan.mblocks = [(0, 2), (2, 3), (3, o)]
        res.plan.pblocks = [(0, 3), (3, npo)]
        res.plan.lblocks = [(0, 5), (5, 6), (6, npo)]
        res.plan.w_mapped = True
        for use_c, three_m in ((HAVE_C, False), (HAVE_C, True), (False, False), (False, True)):
            monkeypatch.setattr(K, 'USE_C', use_c)
            res.plan.three_m = three_m      # three real GEMMs for a complex product
            L = lam.Lambda(res, t1, t2p)
            assert L.paired == paired
            same_intermediates(L)
            l1n, l2n = L.update(l1, l2, shift)
            assert abs(l1n - l1r).max() < 1e-12 and abs(l2n - l2r).max() < 1e-12
            L.close()
        monkeypatch.setattr(K, 'USE_C', HAVE_C)
        # in one block with three real GEMMs
        res.plan = amps.plan_step(o, v, mol.nao_nr(), None, three_m=True,
                                  kramers=bool(eris.kramers))
        L = lam.Lambda(res, t1, t2p)
        l1n, l2n = L.update(l1, l2, shift)
        assert abs(l1n - l1r).max() < 1e-12 and abs(l2n - l2r).max() < 1e-12
    res.close()


@pytest.mark.parametrize('kr', [False, None])
def test_lambda_converged(x2c_h2o, kr):
    '''solve_lambda against PySCF's iterations on the brute-force integrals;
    after (T), which drops what the CCSD step keeps.'''
    mycc = DirectZCCSD(x2c_h2o, frozen=2)
    mycc.verbose, mycc.conv_tol, mycc.conv_tol_normt, mycc.kramers = 0, 1e-11, 1e-9, kr
    mycc.kernel()
    mycc.ccsd_t()
    l1, l2 = mycc.solve_lambda()
    assert mycc.converged_lambda
    ref = reference_eris(mycc.eris, x2c_h2o.mol)
    conv, l1r, l2r = gccsd_lambda.kernel(mycc, ref, mycc.t1, mycc.t2, max_cycle=100, tol=1e-9,
                                         verbose=0)
    assert conv and abs(l1 - l1r).max() < 1e-7 and abs(l2 - l2r).max() < 1e-7
    assert 1e-4 < abs(l2 - mycc.t2.conj()).max() < 1e-2      # conj(t) in first order only
    if mycc.eris.kramers:
        assert kramers.symmetry_error(l1) < 1e-8 and kramers.symmetry_error(l2) < 1e-8


@pytest.mark.parametrize('kr', [False, None])
def test_density_is_energy_derivative(x2c_h2o, kr):
    '''The one-particle density of t and lambda (the expressions of PySCF's
    gccsd_rdm) is the derivative of the CCSD energy with respect to a
    Hermitian one-particle perturbation at fixed spinors: an independent
    check of lambda for complex amplitudes.  With Kramers pairs the
    perturbation is time-reversal symmetric.'''
    def run(eps=0.0, V=None):
        mycc = DirectZCCSD(x2c_h2o, frozen=2)
        mycc.verbose, mycc.conv_tol, mycc.conv_tol_normt, mycc.kramers = 0, 1e-12, 1e-9, kr
        mycc.max_cycle = 200
        eris = mycc.ao2mo()
        if V is not None:
            eris.fock = eris.fock + eps * V
        mycc.kernel(eris=eris)
        assert mycc.converged
        return mycc

    mycc = run()
    o, n = mycc.eris.nocc, mycc.eris.fock.shape[0]
    rng = np.random.default_rng(5)
    V = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    V = 0.1 * (V + V.conj().T)
    if mycc.eris.kramers:           # block by block: each space has its own pairs
        for a in (slice(0, o), slice(o, n)):
            for b in (slice(0, o), slice(o, n)):
                V[a, b] = 0.5 * (V[a, b] + kramers.image(V[a, b]))
        assert abs(V - V.conj().T).max() < 1e-14
    l1, l2 = mycc.solve_lambda()
    d1 = gccsd_rdm._gamma1_intermediates(mycc, mycc.t1, mycc.t2, l1, l2)
    dm1 = gccsd_rdm._make_rdm1(mycc, d1, with_frozen=False)
    expected = (np.einsum('pq,qp', V, dm1) - np.trace(V[:o, :o])).real
    # the error of the central difference is 1.5e-9 here (it scales as eps^2: 1.5e-7
    # with eps = 1e-3)
    eps = 1e-4
    derivative = (run(eps, V).e_corr - run(-eps, V).e_corr) / (2 * eps)
    assert abs(expected) > 1e-5 and abs(derivative - expected) < 2e-8


def test_lambda_peak_within_plan(monkeypatch):
    '''The arrays that a Lambda step allocates stay within lam.working_set of
    the plan (tracemalloc follows numpy's allocations): slabs and blocks of
    several sizes, ZGEMM and three real GEMMs, with and without Kramers
    pairs; and the working set within the budget and lam.reserve_bytes, which
    is what the iterations are planned with.'''
    import tracemalloc
    from zccsdt import memory
    monkeypatch.setattr(memory, 'resident', lambda: 0)
    mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis='cc-pvtz', verbose=0)
    mf = x2c.UHF(mol).run(conv_tol=1e-10)
    rng = np.random.default_rng(1)
    blocked = 0
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
        reserve = lam.reserve_bytes(o, v)
        ref = None
        for factor in (1.0, 1.3, 2.0, 4.0):
            for m3 in (False, True):
                plan = amps.plan_step(o, v, nao, int(factor * need), (), diis_incore=False,
                                      three_m=m3, kramers=eris.kramers)
                res = amps.Residual(eris, plan=plan)
                L = lam.Lambda(res, t1, t2p)        # W, Hq and the rest that is made once
                L.update_packed(l1, l2p)
                tracemalloc.start()
                base = tracemalloc.get_traced_memory()[0]
                new = L.update_packed(l1, l2p)
                peak = tracemalloc.get_traced_memory()[1] - base
                tracemalloc.stop()
                L.close()
                # outside the step: W if it is in memory, t2 and lambda2
                held = (0 if plan.w_mapped else 16 * (o * v) ** 2) + 32 * npo * npv
                work = lam.working_set(plan, o, v)
                assert peak + held <= work, (kr, factor, m3, peak, held, work)
                assert work <= max(plan.budget, plan.work) + reserve
                blocked += len(plan.jblocks) > 1
                if ref is None:
                    ref = new
                assert abs(new[0] - ref[0]).max() < 1e-12 and abs(new[1] - ref[1]).max() < 1e-12
    assert blocked > 6


@pytest.mark.parametrize('kr', [False, None])
def test_t_density_is_energy_derivative(x2c_h2o, kr):
    '''CCSD(T): lambda with the (T) terms (zccsd.lam_t, the expressions of
    the user's socutils code on full arrays) and the (T) terms of the
    one-particle density give the derivative of E(CCSD) + E(T) with respect
    to a perturbation of the Fock matrix in its occupied-virtual blocks and
    on its diagonal (the orbital energies of the denominators of (T)), at
    fixed spinors; without the (T) terms they give that of E(CCSD).'''
    from zccsdt.zccsd import lam_t

    def run(eps=0.0, V=None):
        mycc = DirectZCCSD(x2c_h2o, frozen=2)
        mycc.verbose, mycc.conv_tol, mycc.conv_tol_normt, mycc.kramers = 0, 1e-12, 1e-9, kr
        mycc.max_cycle = 200
        eris = mycc.ao2mo()
        if V is not None:
            eris.fock = eris.fock + eps * V
            eris.mo_energy = eris.fock.diagonal().real.copy()
        mycc.kernel(eris=eris)
        assert mycc.converged
        return mycc, mycc.e_corr, mycc.ccsd_t(eris=eris)

    mycc, _, e_t = run()
    eris = mycc.eris
    o, n = eris.nocc, eris.fock.shape[0]
    assert abs(lam_t.energy(mycc.t1, mycc.t2, eris) - e_t) < 1e-10
    rng = np.random.default_rng(5)
    V0 = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    V0 = 0.1 * (V0 + V0.conj().T)
    if eris.kramers:
        for a in (slice(0, o), slice(o, n)):
            for b in (slice(0, o), slice(o, n)):
                V0[a, b] = 0.5 * (V0[a, b] + kramers.image(V0[a, b]))
    V = np.diag(V0.diagonal())
    V[:o, o:], V[o:, :o] = V0[:o, o:], V0[o:, :o]
    eps = 1e-4
    (_, ep, tp), (_, em, tm) = run(eps, V), run(-eps, V)
    d_ccsd, d_t = (ep - em) / (2 * eps), (tp - tm) / (2 * eps)
    assert abs(d_t) > 1e-5              # what the (T) terms have to account for
    for with_t in (False, True):
        l1, l2 = mycc.solve_lambda(with_t=with_t)
        doo, dov, dvo, dvv = gccsd_rdm._gamma1_intermediates(mycc, mycc.t1, mycc.t2, l1, l2)
        if with_t:
            goo, gvv, gvo = lam_t.gamma1(mycc.t1, mycc.t2, eris)
            doo[np.diag_indices(o)] -= goo
            dvv[np.diag_indices(n - o)] += gvv
            dvo += gvo
        dm1 = gccsd_rdm._make_rdm1(mycc, (doo, dov, dvo, dvv), with_frozen=False)
        assert abs(mycc.make_rdm1(l1=l1, l2=l2, with_t=with_t) - dm1).max() < 1e-12
        expected = (np.einsum('pq,qp', V, dm1) - np.trace(V[:o, :o])).real
        # the error of the central difference is 1e-8 here
        assert abs(d_ccsd + with_t * d_t - expected) < 5e-8
