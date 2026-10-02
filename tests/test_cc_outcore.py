'''Spin-orbital / spinor CCSDT when the large arrays do not fit in memory:
T3 and its residual as mapped arrays of a scratch file, the packed <ab||de>
there as well and its dressed rows in slabs (zccsdt.cc.ccsdt_opt.MemoryPlan).
The sliced and mapped paths must reproduce the run in memory.'''

import numpy as np
import pytest
from pyscf import gto, scf

from zccsdt import eris as E, eris_spin, memory, scratch
from zccsdt.cc import ccsdt, ccsdt_opt, diis, solver, t3pack
from zccsdt.cc.ccsdt_opt import MemoryPlan, plan_memory
from zccsdt.scratch import Scratch, is_mapped

from conftest import water_ghf, water_x2c


@pytest.fixture(scope='module')
def rhf():
    mol = gto.M(atom='F 0 0 0; H 0 0 0.917', basis='6-31g', verbose=0)
    return scf.RHF(mol).run(conv_tol=1e-12)


@pytest.fixture(scope='module', params=['real', 'complex'])
def fh(request, rhf):
    ghf = scf.addons.convert_to_ghf(rhf)
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
    t3p = ccsdt.pack_t3(ccsdt.fullasym(rand(nocc, nocc, nocc, nvir, nvir, nvir)))
    if not cplx:
        t3p = t3p.real.copy()
    return t1, t2, t3p


def test_pair_slabs_cover_all_rows():
    assert ccsdt_opt.pair_slabs(10, 80) == [(0, 10)]
    assert ccsdt_opt.pair_slabs(0, 80, 1) == [(0, 0)]
    for nrow, mb in ((10, 1), (10, 240), (10, 250), (7, 10 ** 6)):
        slabs = ccsdt_opt.pair_slabs(nrow, 80, mb)
        assert slabs[0][0] == 0 and slabs[-1][1] == nrow
        assert all(x[1] == y[0] for x, y in zip(slabs, slabs[1:]))
        assert all(q1 > q0 and (q1 == q0 + 1 or (q1 - q0) * 80 <= mb) for q0, q1 in slabs)


def test_dressed_rows_equal_whole_operator(fh):
    '''Rows of the dressed packed <ab||de> (and of the T3 ladder operator)
    against the whole-operator dressing.'''
    t1, t2, _ = _rand_amps(fh, 5)
    om, vm = t3pack.maps_for(fh)
    ref = ccsdt_opt.dressed_vvvv_pp(fh, t1, om, vm)[0]
    mm, nn = om.pairs[:, 0], om.pairs[:, 1]
    md, ne = vm.pairs[:, 0], vm.pairs[:, 1]
    t2pp = np.ascontiguousarray(t2[mm, nn][:, md, ne])
    goovv = np.asarray(fh.oovv)[mm, nn][:, md, ne]
    F = np.asarray(fh.fock)[fh.nocc:, fh.nocc:]
    full = ref + t2pp.T @ goovv
    ccsdt_opt.add_one_body_pairs(full, F, vm, 0.5)
    for q0, q1 in ((0, vm.npair), (0, 7), (7, 40), (40, vm.npair)):
        out = np.empty((q1 - q0, vm.npair), dtype=ref.dtype)
        ccsdt_opt.dressed_vvvv_rows(fh, t1, om, vm, q0, q1, out)
        np.testing.assert_allclose(out, ref[q0:q1], atol=1e-12)
        ccsdt_opt.dressed_vvvv_rows(fh, t1, om, vm, q0, q1, out, t2pp=t2pp)
        ccsdt_opt.add_one_body_pairs(out, F, vm, 0.5, rows=(q0, q1))
        np.testing.assert_allclose(out, full[q0:q1], atol=1e-12)


@pytest.mark.parametrize('use_c', [True, False])
def test_sliced_step_equals_step_in_memory(fh, use_c, monkeypatch):
    '''The step with the dressed operator in slabs of rows, slab-wise and
    batch-wise, with one or many batches of triples.'''
    from zccsdt.cc import t3kern
    if use_c and not t3kern.USE_C:
        pytest.skip('C library not built')
    monkeypatch.setattr(t3kern, 'USE_C', use_c)
    t1, t2, t3p = _rand_amps(fh, 1)
    opts = solver.Options()
    ref = ccsdt_opt.update_amps(opts, t1, t2, t3p, fh)
    row = t3pack.maps_for(fh)[1].npair * t3p.dtype.itemsize
    for pp_bytes, order, mem in ((5 * row, 'slab', 2000), (5 * row, 'batch', 2000),
                                 (row, 'batch', 0.01), (7 * row, 'slab', 0.01),
                                 (10 ** 9, 'slab', 2000)):
        plan = MemoryPlan(pp_bytes=pp_bytes, pp_order=order, mem_mb=mem)
        new = ccsdt_opt.update_amps(opts, t1, t2, t3p, fh, plan=plan)
        for a, b, name in zip(new, ref, ('t1', 't2', 't3')):
            assert abs(a - b).max() < 1e-12 * max(1.0, abs(b).max()), (name, pp_bytes, order)


@pytest.mark.parametrize('io_mode', [1, 0])
def test_mapped_step_equals_step_in_memory(fh, io_mode, monkeypatch, tmp_path):
    '''T3 and the new T3 as mapped arrays of a scratch file, with and without
    the I/O hints.'''
    monkeypatch.setenv('ZCCSDT_TMPDIR', str(tmp_path))
    monkeypatch.setattr(scratch, 'IO_MODE', io_mode)
    monkeypatch.setattr(scratch, '_HINTS', [True])          # as in a run short of memory
    t1, t2, t3p = _rand_amps(fh, 2)
    opts = solver.Options()
    ref = ccsdt_opt.update_amps(opts, t1, t2, t3p, fh)
    row = t3pack.maps_for(fh)[1].npair * t3p.dtype.itemsize
    scr = Scratch(prefix='zccsdt_test_')
    T3m = scr.create('t3', t3p.size, t3p.dtype)
    T3m[:] = t3p.ravel()
    Rm = scr.create('r3', t3p.size, t3p.dtype)
    Rm[:] = 7.0                         # must be cleared
    for plan in (None, MemoryPlan(pp_bytes=3 * row, pp_order='batch', mem_mb=0.01),
                 MemoryPlan(pp_bytes=4 * row, pp_order='slab', mem_mb=0.01)):
        new = ccsdt_opt.update_amps(opts, t1, t2, T3m, fh, out=Rm, plan=plan)
        assert np.shares_memory(new[2], Rm)
        for a, b, name in zip(new, ref, ('t1', 't2', 't3')):
            assert abs(a - b).max() < 1e-12 * max(1.0, abs(b).max()), name
    np.testing.assert_array_equal(T3m, t3p.ravel())         # T3 itself is only read
    scr.close()
    assert list(tmp_path.iterdir()) == []


@pytest.fixture(scope='module')
def uhf():
    mol = gto.M(atom='N 0 0 0; H 0 0.8036 0.6347; H 0 -0.8036 0.6347', basis='6-31g', spin=1,
                verbose=0)
    return scf.UHF(mol).run(conv_tol=1e-12)


@pytest.mark.parametrize('form', ['rform', 'flip', 'uhf'])
def test_sliced_step_spin_blocked(rhf, uhf, form):
    '''U/R-form integrals: the pair blocks of the dressed operator in slabs
    of rows (alpha-alpha rows serving beta-beta for RHF), slab-wise and
    batch-wise, against the step in memory.'''
    e = eris_spin.from_scf(uhf if form == 'uhf' else rhf, frozen=2 if form == 'uhf' else 4)
    opts = solver.Options(spin_flip=form == 'flip')
    eng = ccsdt_opt.engine_for(e, np.float64, opts.mem_mb, opts.spin_flip)
    if not ccsdt_opt.sliceable(e, eng):
        pytest.skip('C library not built')
    r0 = solver.kernel(e, 'ccsdt', solver.Options(max_cycle=2, spin_flip=opts.spin_flip,
                                                  incore_complete=True))
    ref = [np.array(x) for x in ccsdt_opt.update_amps(opts, r0.t1, r0.t2, r0.t3, e)]
    row = 8 * max(hi - lo for lo, hi in t3pack.maps_for(e)[1].blocks)
    for pp_bytes, order, mem in ((3 * row, 'slab', 2000), (5 * row, 'batch', 2000),
                                 (row, 'batch', 0.01), (7 * row, 'slab', 0.01),
                                 (10 ** 9, 'slab', 2000)):
        plan = MemoryPlan(pp_bytes=pp_bytes, pp_order=order, mem_mb=mem)
        new = ccsdt_opt.update_amps(opts, r0.t1, r0.t2, r0.t3, e, plan=plan)
        for a, b, name in zip(new, ref, ('t1', 't2', 't3')):
            assert abs(a - b).max() < 1e-12 * max(1.0, abs(b).max()), (name, pp_bytes, order)


def test_spin_engine_buffers(uhf):
    '''The spin-blocked engine takes its batch buffers from one work array:
    never more than buffer_bytes says, given back when the budget changes;
    one engine per integral object whatever the budget; and a step that is
    short of memory keeps no ovvv-sized array for the next one.'''
    from zccsdt.cc import t3engine
    e = eris_spin.from_scf(uhf, frozen=2)
    opts = solver.Options()
    eng = ccsdt_opt.engine_for(e, np.float64)
    if not ccsdt_opt.sliceable(e, eng):
        pytest.skip('C library not built')
    r0 = solver.kernel(e, 'ccsdt', solver.Options(max_cycle=2, incore_complete=True))
    ref = [np.array(x) for x in ccsdt_opt.update_amps(opts, r0.t1, r0.t2, r0.t3, e)]
    assert '_bare_vvvo' in e.__dict__ and ccsdt_opt._persistent(e, 'bv')[0] is not None
    assert ccsdt_opt._persistent(e, 'vvvv')[0] is not None
    row = 8 * max(hi - lo for lo, hi in t3pack.maps_for(e)[1].blocks)
    for mem, pp_bytes, order in ((2000, None, 'slab'), (0.05, None, 'slab'),
                                 (0.3, 3 * row, 'batch'), (0.02, 5 * row, 'slab')):
        plan = MemoryPlan(pp_bytes=pp_bytes, pp_order=order, mem_mb=mem)
        new = ccsdt_opt.update_amps(opts, r0.t1, r0.t2, r0.t3, e, plan=plan)
        for a, b in zip(new, ref):
            assert abs(a - b).max() < 1e-12 * max(1.0, abs(b).max())
        assert ccsdt_opt.engine_for(e, np.float64) is eng and eng.mem_mb == mem
        assert 0 < t3engine.workspace_bytes() <= eng.buffer_bytes(mem)
        if pp_bytes is not None:
            assert '_bare_vvvo' not in e.__dict__
            assert ccsdt_opt._persistent(e, 'bv')[0] is None
            assert ccsdt_opt._persistent(e, 'vvvv')[0] is None
    assert ccsdt_opt.cached_bytes(e) >= t3engine.workspace_bytes() > 0
    eng.set_memory(1.0)
    assert t3engine.workspace_bytes() == 0


def test_spin_blocked_vvvv_built_out_of_core(uhf, monkeypatch, tmp_path):
    '''With a small max_memory the U-form pair blocks are built in batches
    into mapped scratch arrays, equal to those built in memory.'''
    monkeypatch.setenv('ZCCSDT_TMPDIR', str(tmp_path))
    ref = eris_spin.from_scf(uhf, frozen=2)
    new = eris_spin.from_scf(uhf, frozen=2, max_memory=memory.resident() / 1e6 + 0.2)
    assert not any(is_mapped(b) for b in ref.vvvv_blocks)
    assert all(is_mapped(b) for b in new.vvvv_blocks)
    for a, b in zip(new.vvvv_blocks, ref.vvvv_blocks):
        np.testing.assert_allclose(a, b, atol=1e-11)
    kw = dict(conv_tol=1e-8, conv_tol_normt=1e-5, max_cycle=4)
    a = solver.kernel(ref, 'ccsdt', solver.Options(incore_complete=True, **kw))
    row = 8 * max(b.shape[1] for b in new.vvvv_blocks)
    b = solver.kernel(new, 'ccsdt', solver.Options(
        plan=MemoryPlan(pp_file=True, pp_bytes=11 * row, pp_order='batch'), **kw))
    assert abs(a.e_tot - b.e_tot) < 1e-9
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('form', ['real', 'complex', 'spin', 'rform', 'flip'])
def test_kernel_out_of_core(rhf, form, monkeypatch, tmp_path):
    '''Whole runs with T3 mapped (its residual in the DIIS storage) and the
    packed <ab||de> in the scratch file, for the dense and the spin-blocked
    engines, against the run in memory.'''
    monkeypatch.setattr(diis, 'INCORE_SIZE', 1000)
    monkeypatch.setenv('ZCCSDT_TMPDIR', str(tmp_path))
    ghf = scf.addons.convert_to_ghf(rhf)
    flip = form == 'flip'
    if form == 'complex':
        e = E.from_scf(ghf, mo_coeff=E.rotate_occ_vir(ghf.mo_coeff, ghf.mo_occ, seed=4, frozen=4),
                       frozen=4)
    elif form in ('real', 'spin'):
        e = E.from_scf(ghf, frozen=4, spin_order=form == 'spin')
    else:
        e = eris_spin.from_scf(rhf, frozen=4)
    kw = dict(conv_tol=1e-9, conv_tol_normt=1e-6, spin_flip=flip)
    ref = solver.kernel(e, 'ccsdt', solver.Options(incore_complete=True, **kw))
    assert ref.converged and not is_mapped(ref.t3) and not ref.plan.t3_file
    row = t3pack.maps_for(e)[1].npair * ref.t3.dtype.itemsize
    for plan in (MemoryPlan(t3_file=True),
                 MemoryPlan(t3_file=True, pp_file=True, pp_bytes=4 * row, pp_order='batch',
                            mem_mb=0.02),
                 MemoryPlan(t3_file=True, r3_file=False),       # the residual in memory
                 MemoryPlan(t3_file=True, r3_file=False, pp_file=True, pp_bytes=5 * row,
                            pp_order='batch'),
                 MemoryPlan(pp_bytes=9 * row, pp_order='slab')):
        r = solver.kernel(e, 'ccsdt', solver.Options(plan=plan, **kw))
        # two runs differ by threaded summation order, at the level of conv_tol
        assert r.converged and abs(r.ncycle - ref.ncycle) <= 1
        assert abs(r.e_tot - ref.e_tot) < 1e-8
        assert abs(r.t3 - ref.t3).max() < 1e-5
        assert is_mapped(r.t3) == plan.t3_file and r.plan is plan
        if plan.pp_file and form in ('real', 'complex'):
            assert is_mapped(e._vvvv_pp)
        if plan.pp_file and form in ('rform', 'flip'):
            assert all(is_mapped(b) for b in e.vvvv_blocks)
            assert e.vvvv_blocks[2] is e.vvvv_blocks[0]     # still shared between spins
    assert list(tmp_path.iterdir()) == []       # scratch files are unlinked when opened


def test_restart_into_mapped_t3(fh, monkeypatch, tmp_path):
    '''Amplitudes given in memory continue in a mapped T3.'''
    monkeypatch.setenv('ZCCSDT_TMPDIR', str(tmp_path))
    kw = dict(conv_tol=1e-9, conv_tol_normt=1e-6)
    r0 = solver.kernel(fh, 'ccsdt', solver.Options(max_cycle=3, **kw))
    ref = solver.kernel(fh, 'ccsdt', solver.Options(**kw), t1=r0.t1, t2=r0.t2, t3=r0.t3)
    r = solver.kernel(fh, 'ccsdt', solver.Options(plan=MemoryPlan(t3_file=True), **kw),
                      t1=r0.t1, t2=r0.t2, t3=r0.t3)
    assert r.converged and is_mapped(r.t3) and abs(r.ncycle - ref.ncycle) <= 1
    assert abs(r.e_tot - ref.e_tot) < 1e-8


@pytest.mark.parametrize('kind', ['ghf', 'x2c'])
def test_packed_vvvv_built_out_of_core(kind, monkeypatch, tmp_path):
    '''With a small max_memory the packed <ab||cd> is built in batches into a
    mapped scratch array and equals the one built in memory.'''
    monkeypatch.setenv('ZCCSDT_TMPDIR', str(tmp_path))
    mf = water_ghf() if kind == 'ghf' else water_x2c()
    ref = E.from_scf(mf, frozen=2, packed_vvvv=True)
    assert not is_mapped(ref._vvvv_pp) and ref._scratch is None
    with monkeypatch.context() as m:        # a budget too small for the packed vvvv
        m.setattr(memory, 'budget', lambda max_memory=None: 200_000)
        new = E.from_scf(mf, frozen=2, packed_vvvv=True)
    assert is_mapped(new._vvvv_pp) and new._scratch is not None
    np.testing.assert_allclose(new._vvvv_pp, ref._vvvv_pp, atol=1e-11)
    for name in E.BLOCKS[:-1]:
        np.testing.assert_allclose(getattr(new, name), getattr(ref, name), atol=1e-11)
    kw = dict(conv_tol=1e-8, conv_tol_normt=1e-5, max_cycle=4)
    a = solver.kernel(ref, 'ccsdt', solver.Options(incore_complete=True, **kw))
    row = new._vvvv_pp.shape[1] * new._vvvv_pp.dtype.itemsize
    b = solver.kernel(new, 'ccsdt', solver.Options(
        plan=MemoryPlan(pp_file=True, pp_bytes=11 * row, pp_order='batch'), **kw))
    assert abs(a.e_tot - b.e_tot) < 1e-9
    assert list(tmp_path.iterdir()) == []


def test_integrals_refuse_what_cannot_fit(monkeypatch):
    '''The integral build says so before it starts if the ovvv block and
    the transformation buffers cannot fit.'''
    monkeypatch.setattr(memory, 'available', lambda: 300 * 10 ** 6)
    with pytest.raises(MemoryError, match='o v\\^3'):
        E.from_scf(water_ghf(), frozen=2, packed_vvvv=True)


def test_place_vvvv_moves_the_bare_operator(monkeypatch, tmp_path):
    monkeypatch.setenv('ZCCSDT_TMPDIR', str(tmp_path))
    e = E.from_scf(water_ghf(), frozen=2, packed_vvvv=True)
    V = np.array(e._vvvv_pp)
    ccsdt_opt.place_vvvv(e, MemoryPlan())
    assert not is_mapped(e._vvvv_pp)
    ccsdt_opt.place_vvvv(e, MemoryPlan(pp_file=True))
    assert is_mapped(e._vvvv_pp)
    np.testing.assert_array_equal(e._vvvv_pp, V)
    M = e._vvvv_pp
    ccsdt_opt.place_vvvv(e, MemoryPlan(pp_file=True))      # already there
    assert e._vvvv_pp is M


def test_plan_memory(monkeypatch):
    monkeypatch.setattr(ccsdt_opt, 'overhead', lambda: 700_000_000)
    o, v, isz = 10, 106, 16                 # X2C H2O / cc-pVTZ
    f = o * v ** 3 * isz
    nvp = v * (v - 1) // 2
    pp, row = nvp * nvp * isz, nvp * isz
    t3 = 120 * (v * (v - 1) * (v - 2) // 6) * isz
    GB = 10 ** 9
    p = plan_memory(o, v, isz)              # no budget: everything in memory
    assert (p.t3_file, p.pp_file, p.pp_bytes, p.mem_mb) == (False, False, None, 2000.0)
    assert 3 * f + pp + 2 * t3 < p.need < 4 * GB        # (3.71 GB measured)
    assert plan_memory(o, v, isz, budget=p.need) == p
    # a little less: the bare operator (in memory now) stays, the dressed
    # rows come in one slab, the buffers shrink
    p1 = plan_memory(o, v, isz, budget=3 * GB)
    assert not p1.t3_file and not p1.pp_file and p1.pp_bytes == pp and p1.mem_mb < 2000
    assert p1.need <= 3 * GB + pp
    # less: the bare operator goes to the file, its dressed rows still come at once
    p2 = plan_memory(o, v, isz, budget=int(2.5 * GB))
    assert p2.pp_file and not p2.t3_file and p2.pp_bytes == pp and p2.need <= 2.5 * GB + pp
    # less: slabs of dressed rows next to T3 in memory
    p2 = plan_memory(o, v, isz, budget=2 * GB)
    assert p2.pp_file and not p2.t3_file and v * row <= p2.pp_bytes < pp
    assert p2.need <= 2 * GB + pp
    # less: T3 goes to the file, its residual stays in memory (the passes
    # only read T3)
    pr = plan_memory(o, v, isz, budget=int(1.7 * GB))
    assert pr.t3_file and pr.r3_file is False and pr.pp_order == 'batch'
    assert pr.need <= 1.7 * GB + pp and pr.need > t3
    # still less: T3 goes to the file as well and the ladder runs batch-wise;
    # the kept operator is then read once per batch of triples, so what is
    # left goes to the buffers and the slabs stay thin
    p3 = plan_memory(o, v, isz, budget=int(1.3 * GB))
    assert p3.t3_file and p3.r3_file and p3.pp_file and p3.pp_order == 'batch'
    assert v * row <= p3.pp_bytes < p2.pp_bytes and p3.need <= 1.3 * GB + pp
    assert p3.mem_mb > 1.5 * f / 1e6
    # ... but not beyond three times the buffers asked for
    p3 = plan_memory(o, v, isz, budget=int(1.3 * GB), mem_mb=150.0)
    assert p3.t3_file and 286 < p3.mem_mb <= 3 * 286 and p3.need <= 1.3 * GB + pp
    # the ovvv-sized arrays and the buffers must fit
    with pytest.raises(MemoryError, match='o v\\^3'):
        plan_memory(o, v, isz, budget=int(0.5 * GB))
    # a hard limit above the soft budget only decides about the error
    assert plan_memory(o, v, isz, budget=int(0.5 * GB), limit=3 * GB).t3_file
    # the operator in a file already: nothing to move, nothing to gain
    p4 = plan_memory(o, v, isz, budget=int(2.6 * GB), pp_where='file')
    assert p4.pp_file and p4.need <= 2.6 * GB
    # spin-blocked forms: T3 may go to the file, the pp operator stays
    p5 = plan_memory(o, v, 8, budget=2 * GB, t3_bytes=t3 // 4, pp_bytes=pp // 4,
                     sliceable=False)
    assert p5.pp_bytes is None and not p5.pp_file
    p6 = plan_memory(o, v, 8, budget=int(1.3 * GB), t3_bytes=t3 // 4, pp_bytes=pp // 4,
                     sliceable=False)
    assert p6.t3_file and p6.pp_bytes is None


def test_plan_memory_spin_engine(uhf, monkeypatch):
    '''The planner with the numbers of the spin-blocked engine (U/R-form
    integrals): its work array counts next to everything else, ovvv-sized
    arrays are kept between the iterations only when memory is not short,
    and the buffers never grow beyond what was asked for.'''
    monkeypatch.setattr(ccsdt_opt, 'overhead', lambda: 0)
    e = eris_spin.from_scf(uhf, frozen=2)
    eng = ccsdt_opt.engine_for(e, np.float64)
    om, vm = t3pack.maps_for(e)
    o, v = om.n, vm.n
    f = o * v ** 3 * 8
    blocks = {id(b): b for b in e.vvvv_blocks}
    pp = sum(b.nbytes for b in blocks.values())
    row = max(b.shape[1] * 8 for b in blocks.values())
    t3 = eng.size * 8
    small = 8 * o * o * v * v * 8 + 2 * om.ntri ** 2 * 8
    mem = 0.5                               # MB: a few pairs and triples per batch
    kw = dict(t3_bytes=t3, pp_bytes=pp, row_bytes=row, engine=eng, mem_mb=mem,
              pp_where='file')
    a, b, kept = ccsdt_opt.OVVV_SPIN
    kw.update(ovvv=ccsdt_opt.OVVV_SPIN, dress_bytes=0)
    assert eng.kept_buffers and kept > 0 and eng.mem_min() < mem
    assert eng.buffer_bytes(eng.mem_min()) <= eng.buffer_bytes(mem) <= eng.buffer_bytes(2000)
    # everything in memory: one more f of ovvv-sized arrays is kept
    p = plan_memory(o, v, 8, **kw)
    assert not p.t3_file and p.pp_bytes is None and p.mem_mb == mem
    ovvv = max(int((a + kept) * f), int((b + kept) * f) + int(mem * 250_000))
    assert p.need == small + eng.buffer_bytes(mem) + ovvv + 2 * t3 + pp
    assert plan_memory(o, v, 8, budget=p.need, **kw) == p
    # a byte less: the dressed rows in slabs, nothing kept, smaller buffers
    q = plan_memory(o, v, 8, budget=p.need - 1, **kw)
    assert q.pp_bytes is not None and q.mem_mb <= mem and q.need < p.need
    # the least that works: the smallest buffers, T3 in the file, the
    # thinnest slabs batch-wise; the buffers do not grow
    least = small + eng.buffer_bytes(eng.mem_min()) \
        + max(int(a * f), int(b * f) + int(eng.mem_min() * 250_000)) + min(v * row, pp)
    q = plan_memory(o, v, 8, budget=least, **kw)
    assert q.t3_file and q.pp_file and q.pp_order == 'batch'
    assert q.pp_bytes == min(v * row, pp) and q.mem_mb == eng.mem_min() and q.need == least
    with pytest.raises(MemoryError, match='o v\\^3'):
        plan_memory(o, v, 8, budget=least - 1, **kw)
    # more room with T3 in the file goes to the slabs, not to the buffers
    q2 = plan_memory(o, v, 8, budget=least + t3 // 2, **kw)
    if q2.t3_file:
        assert q2.mem_mb <= mem and q2.pp_bytes >= q.pp_bytes


def test_kernel_plans_from_max_memory(fh, monkeypatch, tmp_path):
    '''solver.kernel turns max_memory into a plan, and incore_complete keeps
    everything in memory whatever the budget.'''
    monkeypatch.setenv('ZCCSDT_TMPDIR', str(tmp_path))
    seen = {}
    real = ccsdt_opt.plan_memory

    def spy(nocc, nvir, itemsize=16, budget=None, limit=None, pp_where='memory', **kw):
        seen.update(budget=budget, limit=limit, pp_where=pp_where, kw=kw)
        return real(nocc, nvir, itemsize, budget, limit, pp_where, **kw)

    monkeypatch.setattr(ccsdt_opt, 'plan_memory', spy)
    kw = dict(conv_tol=1e-8, conv_tol_normt=1e-5, max_cycle=2)
    r = solver.kernel(fh, 'ccsdt', solver.Options(incore_complete=True, **kw))
    assert seen['budget'] is None and not r.plan.t3_file
    mm = memory.resident() / 1e6 + 5000.0
    r = solver.kernel(fh, 'ccsdt', solver.Options(max_memory=mm, **kw))
    assert abs(seen['budget'] - ccsdt_opt.cached_bytes(fh) - 5e9) < 3e8
    assert seen['limit'] == seen['budget']
    assert seen['kw']['sliceable'] and seen['kw']['t3_bytes'] == r.t3.nbytes
    assert not r.plan.t3_file and r.plan.pp_bytes is None
    with pytest.raises(MemoryError, match='max_memory'):
        solver.kernel(fh, 'ccsdt', solver.Options(max_memory=1.0, **kw))
