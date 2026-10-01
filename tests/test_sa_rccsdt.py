'''Block-stored spin-adapted RCCSDT (zccsdt.sa.rccsdt) against the reference
implementation (zccsdt.sa.rccsdt_ref).'''

import numpy as np
import pytest
from pyscf import gto, scf

from zccsdt.sa import rccsdt as E, rccsdt_ref as R, somap
from zccsdt.sa.spinfree import evaluate
from zccsdt.sa.t3blocks import Blocks


@pytest.fixture(scope='module')
def setup():
    mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis='6-31g', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    ints = R.from_scf(mf)
    no, nv = ints.nocc, ints.nvir
    rng = np.random.default_rng(0)
    t1 = 0.05 * rng.standard_normal((no, nv))
    t2 = 0.05 * rng.standard_normal((no, no, nv, nv))
    t2 = t2 + t2.transpose(1, 0, 3, 2)
    t3 = R.project_null(0.02 * somap.pair_symmetrize(rng.standard_normal((no,) * 3 + (nv,) * 3)))
    return ints, t1, t2, t3


def test_blocks_roundtrip(setup):
    ints, t1, t2, t3 = setup
    B = Blocks(ints.nocc)
    assert abs(B.to_dense(B.from_dense(t3)) - t3).max() < 1e-15


def test_residuals_equal_reference(setup):
    ints, t1, t2, t3 = setup
    B = Blocks(ints.nocc)
    a1, a2, a3 = R.residuals(t1, t2, t3, ints)
    for fn in (E.residuals_terms, E.residuals):
        r1, r2, r3b = fn(ints, B, t1, t2, B.from_dense(t3))
        assert abs(r1 - a1).max() < 1e-12
        assert abs(r2 - a2).max() < 1e-12
        assert abs(B.to_dense(r3b) - R.project_null(a3)).max() < 1e-12
    assert abs(r1 - a1).max() < 1e-12
    assert abs(r2 - a2).max() < 1e-12
    # equal up to the redundant component
    assert abs(B.to_dense(r3b) - R.project_null(a3)).max() < 1e-12


def test_energy_equals_reference():
    mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis='6-31g', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    ints = R.from_scf(mf, ncore=1)
    a = R.kernel(ints, 'ccsdt', conv_tol=1e-11, conv_tol_normt=1e-8)
    b = E.kernel(ints, conv_tol=1e-11, conv_tol_normt=1e-8)
    assert a.converged and b.converged and abs(a.e_tot - b.e_tot) < 1e-10


def test_vvvv_t1_packed_c_equals_numpy(monkeypatch):
    from zccsdt.cc import t3pack
    rng = np.random.default_rng(1)
    v, no = 6, 3
    W = rng.standard_normal((v, v, v, v))
    W = W + W.transpose(1, 0, 3, 2)                 # W_ab^ef = W_ba^fe
    s0, s1 = np.triu_indices(v)
    a0, a1 = np.triu_indices(v, 1)
    Ws = np.ascontiguousarray(0.5 * (W + W.transpose(0, 1, 3, 2))[s0, s1][:, s0, s1])
    Wa = np.ascontiguousarray(0.5 * (W - W.transpose(0, 1, 3, 2))[a0, a1][:, a0, a1])
    t1 = rng.standard_normal((no, v))
    ref = np.einsum('abef,if->abei', W, t1)
    assert abs(E.vvvv_t1_packed(Ws, Wa, t1) - ref).max() < 1e-12
    monkeypatch.setattr(t3pack, 'USE_C', False)
    assert abs(E.vvvv_t1_packed(Ws, Wa, t1) - ref).max() < 1e-12


def test_pair_gather_c_equals_numpy(monkeypatch):
    '''C pair gather against the numpy path (v not a multiple of the block
    size) and a pair with a zero cube.'''
    from zccsdt.cc import t3pack
    if not t3pack.USE_C:
        pytest.skip('no C library')
    no, v = 4, 37
    B = Blocks(no)
    t3b = np.random.default_rng(5).standard_normal((B.nblk, v, v, v))
    for p, q in ((0, 2), (1, 1), (3, 3)):
        Wc = E.pair_gather(B, t3b, p, q, np.full((5, v, no, v, v), np.nan))
        monkeypatch.setattr(t3pack, 'USE_C', False)
        Wn = E.pair_gather(B, t3b, p, q, np.full((5, v, no, v, v), np.nan))
        monkeypatch.setattr(t3pack, 'USE_C', True)
        assert np.isfinite(Wc).all()
        assert abs(Wc - Wn).max() < 1e-13


def test_drive_hh_equals_separate_terms(monkeypatch):
    '''Merged hh ladder + drive (attached form, virtual slabs) against the
    separate ordered drive and hh ladder; C kernels and numpy fallbacks.'''
    from zccsdt.cc import t3pack
    no, v = 3, 19
    rng = np.random.default_rng(3)
    B = Blocks(no)
    t3b = rng.standard_normal((B.nblk, v, v, v))
    t2 = rng.standard_normal((no, no, v, v))
    t2 = t2 + t2.transpose(1, 0, 3, 2)
    Wv = rng.standard_normal((v, v, v, no))
    Wo = rng.standard_normal((no, v, no, no))
    Wh = rng.standard_normal((no, no, no, no))
    Wh = Wh + Wh.transpose(1, 0, 3, 2)
    ref = np.zeros_like(t3b)
    E.drive(B, t2, Wv, Wo, ref)
    E.ladder_hh(B, t3b, Wh, ref)
    for use_c in (True, False):
        monkeypatch.setattr(t3pack, 'USE_C', use_c and t3pack._lib is not None)
        for mb in (1 << 29, 8 * no * no * v * v * 4):          # one slab / slabs of 4
            r3b = np.zeros_like(t3b)
            E.drive_hh(B, t3b, t2, Wv, Wo, Wh, r3b, max_bytes=mb)
            assert abs(r3b - ref).max() < 1e-10 * abs(ref).max()


def test_blocks_from_ao_integrals():
    '''Integral blocks transformed on demand from the AO integrals (no full V,
    batched vvvv) reproduce the residuals built from the full V.'''
    mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis='6-31g', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    full = R.from_scf(mf, ncore=1)
    lazy = R.from_scf(mf, ncore=1, full=False)
    assert lazy.V is None
    no, nv = full.nocc, full.nvir
    Vs, Va = E.vvvv_packed(full)
    for mb in (1 << 30, 8 * 3 * nv ** 3):            # 4-fold repack / batches of 3
        Ls, La = E.vvvv_packed(lazy, max_bytes=mb)
        assert abs(Vs - Ls).max() < 1e-12 and abs(Va - La).max() < 1e-12
    for key in ('ovvv', 'vvvo', 'oovv', 'ovov'):
        assert abs(full.block(key) - lazy.block(key)).max() < 1e-12, key
    rng = np.random.default_rng(1)
    B = Blocks(no)
    t1 = 0.05 * rng.standard_normal((no, nv))
    t2 = 0.05 * rng.standard_normal((no, no, nv, nv))
    t2 = t2 + t2.transpose(1, 0, 3, 2)
    t3b = 0.02 * rng.standard_normal((B.nblk, nv, nv, nv))
    a = E.residuals(full, B, t1, t2, t3b)
    b = E.residuals(lazy, B, t1, t2, t3b)
    for x, y in zip(a, b):
        assert abs(x - y).max() < 1e-11 * max(1.0, abs(x).max())

