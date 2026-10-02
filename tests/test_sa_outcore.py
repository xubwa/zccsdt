'''Spin-adapted RCCSDT when the large arrays do not fit in memory: T3 and its
residual in the scratch file, the packed <vv|vv> operator there and its
dressed rows in slabs (zccsdt.sa.rccsdt.MemoryPlan).  The sliced paths must
reproduce the run in memory.'''

import numpy as np
import pytest
from pyscf import gto, scf

from zccsdt.cc import t3pack
from zccsdt.sa import rccsdt as E, rccsdt_ref as R
from zccsdt.sa.t3blocks import Blocks

needs_c = pytest.mark.skipif(not t3pack.USE_C, reason='no C library')


@pytest.fixture(scope='module')
def mf():
    mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis='6-31g*', verbose=0)
    return scf.RHF(mol).run(conv_tol=1e-12)


def _amplitudes(no, nv, seed=1):
    rng = np.random.default_rng(seed)
    t1 = 0.05 * rng.standard_normal((no, nv))
    t2 = 0.05 * rng.standard_normal((no, no, nv, nv))
    t2 = t2 + t2.transpose(1, 0, 3, 2)
    t3b = 0.02 * rng.standard_normal((Blocks.count(no), nv, nv, nv))
    return t1, t2, t3b


def test_pp_slabs_cover_all_rows():
    for v in (1, 2, 7, 20):
        ps, pa = v * (v + 1) // 2, v * (v - 1) // 2
        assert E.pp_slabs(v) == [(0, v)]
        assert E.pair_rows(v, 0, v) == (0, ps, 0, pa)
        for mb in (1, 8 * 2 * v ** 3, 8 * (ps * ps + pa * pa)):
            slabs = E.pp_slabs(v, mb)
            assert slabs[0][0] == 0 and slabs[-1][1] == v
            assert all(x[1] == y[0] for x, y in zip(slabs, slabs[1:]))
            for a0, a1 in slabs:
                rs0, rs1, ra0, ra1 = E.pair_rows(v, a0, a1)
                assert a1 > a0 and (a1 == a0 + 1 or 8 * ((rs1 - rs0) * ps + (ra1 - ra0) * pa) <= mb)
        assert len(E.pp_slabs(v, 1)) == v and len(E.pp_slabs(v, 8 * (ps * ps + pa * pa))) == 1


def test_vvvv_t1_rows(monkeypatch):
    '''The vvvo intermediate accumulated over row slabs of W, C and numpy.'''
    rng = np.random.default_rng(1)
    v, no = 7, 3
    W = rng.standard_normal((v, v, v, v))
    W = W + W.transpose(1, 0, 3, 2)
    s0, s1 = np.triu_indices(v)
    a0, a1 = np.triu_indices(v, 1)
    Ws = np.ascontiguousarray(0.5 * (W + W.transpose(0, 1, 3, 2))[s0, s1][:, s0, s1])
    Wa = np.ascontiguousarray(0.5 * (W - W.transpose(0, 1, 3, 2))[a0, a1][:, a0, a1])
    t1 = rng.standard_normal((no, v))
    ref = np.einsum('abef,if->abei', W, t1)
    for use_c in (True, False):
        monkeypatch.setattr(t3pack, 'USE_C', use_c and t3pack._lib is not None)
        out = np.zeros((v, v, v, no))
        for r0, r1 in ((0, 2), (2, 3), (3, 7)):
            rs0, rs1, ra0, ra1 = E.pair_rows(v, r0, r1)
            E.vvvv_t1_packed(np.ascontiguousarray(Ws[rs0:rs1]), np.ascontiguousarray(Wa[ra0:ra1]),
                             t1, out=out, rows=(r0, r1))
        assert abs(out - ref).max() < 1e-12


def test_vvvv_packed_out_of_core(mf, monkeypatch, tmp_path):
    '''The packed bare <vv|vv> from PySCF's out-of-core transformation (rows
    of one a at a time) equals the one cut from the full V; with and without
    the C kernel, into ordinary and mapped arrays; nothing is left behind.'''
    from zccsdt.scratch import Scratch
    monkeypatch.setenv('ZCCSDT_TMPDIR', str(tmp_path))
    Vs, Va = E.vvvv_packed(R.from_scf(mf, ncore=1))
    lazy = R.from_scf(mf, ncore=1, full=False)
    for use_c in (True, False):
        monkeypatch.setattr(t3pack, 'USE_C', use_c and t3pack._lib is not None)
        Ls, La = E.vvvv_packed(lazy, max_bytes=1)
        assert abs(Vs - Ls).max() < 1e-12 and abs(Va - La).max() < 1e-12
    scr = Scratch()
    out = (scr.create('Vs', Vs.shape), scr.create('Va', Va.shape))
    Ls, La = E.vvvv_packed(lazy, max_bytes=1, out=out)
    assert Ls is out[0] and abs(Vs - Ls).max() < 1e-12 and abs(Va - La).max() < 1e-12
    scr.close()
    assert list(tmp_path.iterdir()) == []


@needs_c
def test_sliced_residuals_equal_in_memory(mf, monkeypatch, tmp_path):
    '''Residuals with the pp operator in row slabs (both loop orders), in the
    scratch file, with small work buffers and with mapped T3 arrays.'''
    from zccsdt.scratch import Scratch
    monkeypatch.setenv('ZCCSDT_TMPDIR', str(tmp_path))
    ints = R.from_scf(mf, ncore=1, full=False)
    no, nv = ints.nocc, ints.nvir
    B = Blocks(no)
    t1, t2, t3b = _amplitudes(no, nv)
    ref = E.residuals(ints, B, t1, t2, t3b)
    scr = Scratch()
    t3m = scr.create('t3', t3b.shape)
    t3m[...] = t3b
    plans = (E.MemoryPlan(pp_bytes=8 * 3 * nv ** 3),
             E.MemoryPlan(pp_bytes=1, batch=1, slab_bytes=1),
             E.MemoryPlan(pp_bytes=8 * 3 * nv ** 3, pp_order='batch', batch=3),
             E.MemoryPlan(pp_file=True, pp_bytes=8 * 5 * nv ** 3, build_bytes=1, batch=3),
             E.MemoryPlan(pp_file=True, pp_bytes=1, build_bytes=1, pp_order='batch', batch=2),
             E.MemoryPlan(pp_file=True))
    for n, plan in enumerate(plans):
        fresh = R.from_scf(mf, ncore=1, full=False)
        mapped = n % 2 == 1
        r3m = scr.create('r3.%d' % n, t3b.shape) if mapped else None
        got = E.residuals(fresh, B, t1, t2, t3m if mapped else t3b, r3m, plan)
        assert isinstance(E.bare_packed(fresh)['Vs'], np.memmap) == plan.pp_file
        for x, y in zip(got, ref):
            assert abs(x - y).max() < 1e-12, plan
    scr.close()
    assert list(tmp_path.iterdir()) == []


@needs_c
def test_kernel_out_of_core_equals_in_memory(mf, monkeypatch, tmp_path):
    '''A whole calculation with T3, its residual and the pp operator in the
    scratch directory: same iterations and energy, amplitudes readable after
    the run, (Q) from the mapped amplitudes, no file left.'''
    from zccsdt.cc import diis
    monkeypatch.setenv('ZCCSDT_TMPDIR', str(tmp_path))
    monkeypatch.setattr(diis, 'INCORE_SIZE', 1000)
    ref = E.kernel(R.from_scf(mf, ncore=1, full=False), conv_tol=1e-10, conv_tol_normt=1e-7,
                   incore_complete=True)
    assert not ref.plan.t3_file and not isinstance(ref.t3, np.memmap)
    ints = R.from_scf(mf, ncore=1, full=False)
    nv = ints.nvir
    plan = E.MemoryPlan(t3_file=True, pp_file=True, pp_bytes=8 * 4 * nv ** 3, pp_order='batch',
                        build_bytes=1, batch=2, slab_bytes=1 << 16)
    r = E.kernel(ints, conv_tol=1e-10, conv_tol_normt=1e-7, plan=plan)
    # (two converged runs agree to the convergence threshold)
    assert r.converged and abs(r.ncycle - ref.ncycle) <= 1 and abs(r.e_tot - ref.e_tot) < 1e-9
    assert isinstance(r.t3, np.memmap) and abs(np.asarray(r.t3) - ref.t3).max() < 1e-6
    assert r.plan is plan and r.diis_on_disk
    a = E.add_q(R.from_scf(mf, ncore=1, full=False), ref)
    b = E.add_q(ints, r)
    assert abs(a.e_q - b.e_q) < 1e-9 and abs(a.e_q5 - b.e_q5) < 1e-9
    del r, b
    assert list(tmp_path.iterdir()) == []


def test_plan_memory():
    G = 10 ** 9
    o, v = 8, 108
    full = E.plan_memory(o, v)
    assert not (full.t3_file or full.pp_file) and full.pp_bytes is None and full.need > 7 * G
    assert E.plan_memory(o, v, full.need + 1).need == full.need
    # a little short: the bare pp operator goes to the scratch file
    p = E.plan_memory(o, v, 6 * G)
    assert not p.t3_file and p.pp_file and p.pp_bytes is None and p.need <= 6 * G
    # shorter: its dressed rows come in slabs (a small operator: batch-wise)
    p = E.plan_memory(o, v, 56 * G // 10)
    assert not p.t3_file and p.pp_file and p.pp_order == 'batch' and p.need <= 56 * G // 10
    assert len(E.pp_slabs(v, p.pp_bytes)) > 1
    # shorter: T3 goes to the scratch file before the buffers get tiny
    p = E.plan_memory(o, v, 5 * G)
    assert p.t3_file and not p.pp_file and p.batch == 8 and p.need <= 5 * G
    # shorter: both, and the ladder reads T3 once per iteration
    p = E.plan_memory(o, v, 3 * G)
    assert p.t3_file and p.pp_file and p.pp_bytes < 4 * v ** 4 and p.pp_order == 'batch'
    assert p.need <= 3 * G and p.slab_bytes < 1 << 29 and len(E.pp_slabs(v, p.pp_bytes)) > 1
    # small o, large v: T3 stays in memory, the large operator is sliced slab-wise
    p = E.plan_memory(5, 282, 60 * G)
    assert not p.t3_file and p.pp_file and p.pp_order == 'slab' and p.need <= 60 * G
    assert len(E.pp_slabs(282, p.pp_bytes)) > 1
    # the bare operator exists already: where it is does not change
    assert not E.plan_memory(o, v, 3 * G, pp_cached='memory').pp_file
    assert E.plan_memory(o, v, 30 * G, pp_cached='file').pp_file
    # below the fixed part: an error before anything is allocated
    with pytest.raises(MemoryError, match='o v\\^3'):
        E.plan_memory(o, v, 1 * G)
    # the budget is soft: the limit decides
    assert E.plan_memory(o, v, 1 * G, limit=6 * G).t3_file


@needs_c
def test_kernel_plans_from_max_memory(mf, monkeypatch, tmp_path):
    '''max_memory drives the plan: enough gives the run in memory, the least
    that works puts T3 and the pp operator in the scratch file (same energy),
    less raises before the iterations.'''
    from zccsdt import memory
    monkeypatch.setenv('ZCCSDT_TMPDIR', str(tmp_path))
    monkeypatch.setattr(memory, 'resident', lambda: 0)
    ints = R.from_scf(mf, ncore=1, full=False)
    no, nv = ints.nocc, ints.nvir
    full = E.plan_memory(no, nv).need
    ref = E.kernel(ints, conv_tol=1e-10, conv_tol_normt=1e-7, max_memory=(full + 1e5) / 1e6)
    assert not (ref.plan.t3_file or ref.plan.pp_file) and ref.plan.need == full
    low = full
    while True:                                   # the smallest budget that is accepted
        try:
            E.plan_memory(no, nv, low - 20_000)
        except MemoryError:
            break
        low -= 20_000
    assert low < full
    fresh = R.from_scf(mf, ncore=1, full=False)
    r = E.kernel(fresh, conv_tol=1e-10, conv_tol_normt=1e-7, max_memory=low / 1e6)
    assert r.plan.t3_file and r.plan.pp_file and r.plan.pp_bytes is not None
    assert r.plan.need <= low and len(E.pp_slabs(nv, r.plan.pp_bytes)) > 1
    assert r.converged and abs(r.e_tot - ref.e_tot) < 1e-9
    # the arrays cached on the integrals by that run do not shrink the budget
    again = E.kernel(fresh, max_cycle=1, max_memory=low / 1e6)
    assert again.plan.pp_file and again.plan.t3_file
    assert not E.kernel(ints, max_cycle=1, incore_complete=True, max_memory=1).plan.t3_file
    with pytest.raises(MemoryError, match='max_memory'):
        E.kernel(R.from_scf(mf, ncore=1, full=False), max_cycle=1, max_memory=1)
    del r, again
    assert list(tmp_path.iterdir()) == []
