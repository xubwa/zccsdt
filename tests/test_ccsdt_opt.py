'''The packed-T3 CCSDT step must reproduce the ported reference step.'''
import numpy as np
import pytest
from pyscf import gto, scf

from zccsdt import eris as E
from zccsdt.cc import ccsdt, ccsdt_opt, solver
from zccsdt.cc import t3pack
from zccsdt.cc.ccsdt_opt import (get_block, dress_block, dress_fock,
                                 dressed_vvvv_pp, add_one_body_pairs)


@pytest.fixture(scope='module', params=['real', 'complex'])
def fh(request):
    mol = gto.M(atom='F 0 0 0; H 0 0 0.917', basis='6-31g', verbose=0)
    ghf = scf.addons.convert_to_ghf(scf.RHF(mol).run(conv_tol=1e-12))
    C = None
    if request.param == 'complex':
        C = E.rotate_occ_vir(ghf.mo_coeff, ghf.mo_occ, seed=4, frozen=4)
    return E.from_scf(ghf, mo_coeff=C, frozen=4)          # nocc 6, nvir 12


def _rand_amps(e, seed):
    rng = np.random.default_rng(seed)
    nocc, nvir = e.nocc, e.nmo - e.nocc
    cplx = np.iscomplexobj(e.oovv)

    def rand(*sh):
        x = rng.random(sh) - 0.5
        if cplx:
            x = x + 1j * (rng.random(sh) - 0.5)
        return 0.05 * x
    t1 = rand(nocc, nvir)
    t2 = rand(nocc, nocc, nvir, nvir)
    t2 = t2 - t2.transpose(1, 0, 2, 3)
    t2 = t2 - t2.transpose(0, 1, 3, 2)
    t3 = ccsdt.fullasym(rand(nocc, nocc, nocc, nvir, nvir, nvir))
    t3p = ccsdt.pack_t3(t3)
    if not cplx:
        t3p = t3p.real.copy()
    return t1, t2, t3p


def test_get_block_all_keys(fh):
    from zccsdt.cc.bruteforce import build_g
    g = build_g(fh, fh.nocc, fh.nmo)
    o, v = slice(0, fh.nocc), slice(fh.nocc, fh.nmo)
    for key in ['vovv', 'vvvo', 'ovoo', 'oovo', 'vvoo', 'voov', 'vovo', 'vvov',
                'ovvv', 'vooo']:
        ref = g[tuple(o if c == 'o' else v for c in key)]
        np.testing.assert_allclose(get_block(fh, key), ref, atol=1e-12, err_msg=key)


def test_dressing_matches_reference(fh):
    from zccsdt.cc.bruteforce import build_g
    t1, _, _ = _rand_amps(fh, 5)
    g = build_g(fh, fh.nocc, fh.nmo)
    ge, fdr = ccsdt._t1_dress(g, np.asarray(fh.fock), t1, fh.nocc, fh.nmo)
    o, v = slice(0, fh.nocc), slice(fh.nocc, fh.nmo)
    for key in ['vvvo', 'ovoo', 'vvvv', 'oooo', 'ovvo', 'oovv', 'ovvv', 'ooov',
                'oovo', 'vovv']:
        ref = ge[tuple(o if c == 'o' else v for c in key)]
        np.testing.assert_allclose(dress_block(fh, t1, key), ref, atol=1e-12,
                                   err_msg=key)
    np.testing.assert_allclose(dress_fock(fh, t1), fdr, atol=1e-12)
    vm = t3pack.triple_map(fh.nmo - fh.nocc)
    md, ne = vm.pairs[:, 0], vm.pairs[:, 1]
    ref = ge[v, v, v, v][md[:, None], ne[:, None], md, ne]
    np.testing.assert_allclose(dressed_vvvv_pp(fh, t1, vm), ref, atol=1e-12)


def test_add_one_body_pairs():
    rng = np.random.default_rng(0)
    n = 6
    vm = t3pack.triple_map(n)
    F = rng.standard_normal((n, n))
    I = np.eye(n)
    full = (np.einsum('ad,be->abde', F, I) - np.einsum('bd,ae->abde', F, I)
            - np.einsum('ae,bd->abde', F, I) + np.einsum('be,ad->abde', F, I))
    md, ne = vm.pairs[:, 0], vm.pairs[:, 1]
    W = np.zeros((vm.npair, vm.npair))
    add_one_body_pairs(W, F, vm, 0.5)
    np.testing.assert_allclose(W, 0.5 * full[md[:, None], ne[:, None], md, ne], atol=1e-14)


@pytest.mark.parametrize('use_c', [True, False])
@pytest.mark.parametrize('seed', [1, 2])
@pytest.mark.parametrize('mem_mb', [2000, 0.5])      # one batch / many batches
def test_step_matches_reference(fh, seed, mem_mb, use_c, monkeypatch):
    from zccsdt.cc import t3kern
    if use_c and not t3kern.USE_C:
        pytest.skip('C library not built')
    monkeypatch.setattr(t3kern, 'USE_C', use_c)
    t1, t2, t3p = _rand_amps(fh, seed)
    opts = solver.Options()
    ref = ccsdt.update_amps(opts, t1, t2, t3p.astype(complex), fh)
    new = ccsdt_opt.update_amps(opts, t1, t2, t3p, fh, mem_mb=mem_mb)
    for a, b, name in zip(new, ref, ('t1', 't2', 't3')):
        scale = max(1.0, abs(b).max())
        assert abs(a - b).max() < 1e-12 * scale, name
    if not np.iscomplexobj(fh.oovv):
        assert all(not np.iscomplexobj(x) for x in new)
