'''The disk of the scratch location as a planned resource (zccsdt.disk):

- the budget: max_disk, ZCCSDT_MAX_DISK / scratch.limit(), the free space;
- the placement of the DIIS vectors over a scan of the two budgets:
  monotone, within both, every case of the rule reached;
- Scratch.create(): no room is an OSError with the numbers before anything
  is allocated, and a failed allocation gives back what it had taken;
- the DIIS store with some vectors in memory and the others in the file;
- whole runs of the solvers with the disk forced low, in each case of the
  rule, against the unrestricted run; the options that are kept;
- the report of the front end, and the dry run of the benzene example.'''
import dataclasses
import errno
import gc
import io
import os
import tracemalloc

import numpy as np
import pytest
from pyscf import gto, scf

from zccsdt import disk, eris as E, memory, scratch
from zccsdt.cc import ccsdt_opt, ccsdtq, diis, lambda_opt, solver
from zccsdt.cc.diis import DIIS
from zccsdt.scratch import Scratch

WATER = 'O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587'


# ------------------------------------------------------------------ the plan

def test_plan_over_a_scan_of_the_budgets():
    '''plan_diis over a grid of (memory, disk): the subspace never shrinks
    when a budget grows, the vectors never take more than either budget,
    and (a), (b), (c), (d) are all reached; (e) with diis_t3 = True.'''
    big, space = 1000, 8
    pair = 2 * big
    grid = [0, 1, pair - 1, pair, 2 * pair, 3 * pair - 1, 3 * pair, 5 * pair, 8 * pair - 1, 8 * pair, 20 * pair, None]
    seen = set()
    plans = {}
    for m in grid:
        for d in grid:
            p = plans[m, d] = disk.plan_diis(big, space, m, d)
            seen.add(p.case)
            assert p.requested == space and p.pair == pair
            assert d is None or p.disk <= d
            assert m is None or p.memory <= m
            if p.with_big:
                assert 3 <= p.space == p.n_file + p.n_memory <= space
                assert p.reduced == (p.space < space)
                # the file first: nothing in memory while the disk has room
                assert p.n_memory == 0 or d is not None and (p.n_file + 1) * pair > d
            else:       # (d): not even three pairs
                assert p.case == 'd' and p.reduced and p.disk == p.memory == 0 and p.space == space
                assert (d // pair) + (m // pair) < 3
    assert seen == {'a', 'b', 'c', 'd'}

    def size(p):        # (d) below every subspace with T3
        return p.space if p.with_big else 0
    key = [float('inf') if x is None else x for x in grid]
    for i, m in enumerate(grid):
        for j, d in enumerate(grid):
            for i2 in range(i, len(grid)):
                for j2 in range(j, len(grid)):
                    assert key[i2] >= key[i] and key[j2] >= key[j]
                    assert size(plans[grid[i2], grid[j2]]) >= size(plans[m, d])
    # the cases one by one
    assert disk.plan_diis(big, 8, 0, 8 * pair).case == 'a'
    p = disk.plan_diis(big, 8, 3 * pair, 6 * pair)
    assert (p.case, p.n_file, p.n_memory, p.space) == ('b', 6, 2, 8)
    p = disk.plan_diis(big, 8, pair, 4 * pair)
    assert (p.case, p.n_file, p.n_memory, p.space) == ('c', 4, 1, 5) and 'reduced from the 8' in p.text()
    p = disk.plan_diis(big, 8, pair, pair)
    assert p.case == 'd' and not p.with_big and 'without T3' in p.text()
    # (e): T3 asked for explicitly
    with pytest.raises(disk.DiskFullError, match='diis_t3 = True') as info:
        disk.plan_diis(big, 8, pair, pair, with_big=True)
    assert info.value.errno == errno.ENOSPC and isinstance(info.value, OSError)
    assert disk.plan_diis(big, 8, pair, 2 * pair, with_big=True).space == 3
    # a subspace below the floor that is asked for is not a reduction
    p = disk.plan_diis(big, 2, 0, 2 * pair)
    assert (p.case, p.space) == ('a', 2) and disk.plan_diis(big, 2, 0, pair).case == 'd'
    # other floors
    assert disk.plan_diis(big, 8, 0, 4 * pair, floor=5).case == 'd'
    assert disk.plan_diis(big, 8, 0, 5 * pair, floor=5).space == 5


def test_plan_keeps_the_explicit_options():
    big, pair = 1000, 2000
    # incore_complete: all in memory, or a MemoryError with the numbers - never the file
    p = disk.plan_diis(big, 8, 8 * pair, 0, incore=True)
    assert (p.n_memory, p.n_file, p.disk, p.case) == (8, 0, 0, 'm')
    assert disk.plan_diis(big, 8, None, 0, incore=True).n_memory == 8
    with pytest.raises(MemoryError, match='incore_complete: 8 DIIS vectors'):
        disk.plan_diis(big, 8, 8 * pair - 1, 100 * pair, incore=True)
    # diis_t3 = False: no large part whatever fits
    p = disk.plan_diis(big, 8, None, None, with_big=False)
    assert not p.with_big and p.disk == p.memory == 0 and not p.reduced and 'diis_t3 = False' in p.text()
    # small vectors (nothing goes to a file) and no DIIS
    assert disk.plan_diis(0, 8, 0, 0).n_memory == 8 and disk.plan_diis(big, 0, 0, 0).space == 0
    # which parts of a vector go to the file
    assert disk.file_bytes([10, 500, 5000], 16, 1000) == 16 * 5000
    assert disk.file_bytes([10, 5000, 50000], 8, 1000) == 8 * 55000
    assert disk.file_bytes([10, 500], 8, 1000) == 0


def test_budget(monkeypatch, tmp_path):
    '''max_disk (MB, less what the open scratch files hold), the limit of
    the process, the free space less the reserve; never above the free
    space.'''
    monkeypatch.setenv('ZCCSDT_TMPDIR', str(tmp_path))
    free = [10_000_000_000]
    monkeypatch.setattr(scratch, 'disk_free', lambda d=None: free[0])
    prev = scratch.limit(None)
    try:
        assert disk.budget() == free[0] - int(0.02 * free[0])
        free[0] = 1 << 40
        assert disk.budget() == (1 << 40) - (1 << 30)       # the reserve: at most 1 GB
        assert disk.budget(5.0) == 5_000_000
        assert disk.budget(1e12) == 1 << 40                 # not more than is free
        s = Scratch(prefix='zccsdt_test_')
        s.create('a', 250_000)
        assert scratch.allocated() == 2_000_000 and disk.budget(5.0) == 3_000_000
        assert disk.budget(1.0) == 0
        with disk.limit(4.0):                               # the limit of a run
            assert scratch.limit() == 4_000_000 and disk.budget() == 2_000_000
            with disk.limit(None):                          # (no option: unchanged)
                assert disk.budget() == 2_000_000
        assert scratch.limit() is None
        assert disk.require(1_000_000, 'T3', 5.0) == 2_000_000
        with pytest.raises(disk.DiskFullError, match=r'puts 3.5 MB for T3 into scratch files .* max_disk 5.0 MB'):
            disk.require(3_500_000, 'T3', 5.0)
        s.close()
        assert scratch.allocated() == 0 and disk.budget(5.0) == 5_000_000
        free[0] = None
        assert disk.budget() is None and disk.budget(5.0) == 5_000_000
    finally:
        scratch.limit(prev)


def test_environment_variable(monkeypatch):
    monkeypatch.setenv('ZCCSDT_MAX_DISK', '12.5')
    assert scratch._env_limit() == 12_500_000
    monkeypatch.setenv('ZCCSDT_MAX_DISK', '')
    assert scratch._env_limit() is None


# ---------------------------------------------------------------- allocation

def test_no_room_is_an_error_before_anything_is_allocated(monkeypatch, tmp_path):
    '''create() with less room than the dataset: OSError (ENOSPC) with the
    numbers; posix_fallocate is not called and the file has not grown.'''
    monkeypatch.setenv('ZCCSDT_TMPDIR', str(tmp_path))
    calls = []
    real = os.posix_fallocate
    monkeypatch.setattr(os, 'posix_fallocate', lambda *a: (calls.append(a), real(*a))[1])
    # the limit of the run
    s = Scratch(prefix='zccsdt_test_', what='T3')
    with disk.limit(1.0):
        a = s.create('a', 100_000)                          # 0.8 MB
        size = os.fstat(s.raw.fileno()).st_size
        with pytest.raises(OSError, match=r'no room for 0.001 GB for T3 .*max_disk / ZCCSDT_MAX_DISK 0.001 GB') as info:
            s.create('b', 100_000)
        assert info.value.errno == errno.ENOSPC and len(calls) == 1
        assert os.fstat(s.raw.fileno()).st_size == size and scratch.allocated() == 800_000
        a[:] = 1.0                                          # (the first array is whole)
        assert s.create('c', 20_000).nbytes == 160_000
    s.close()
    assert scratch.allocated() == 0
    # a full file system (statvfs)

    class Full:
        f_bavail, f_frsize = 10, 4096
    monkeypatch.setattr(os, 'statvfs', lambda d: Full)
    assert scratch.disk_free(str(tmp_path)) == 40960
    s = Scratch(prefix='zccsdt_test_', what='a DIIS vector', hint=' or a smaller diis_space')
    with pytest.raises(OSError, match='no room for .* a DIIS vector .* or a smaller diis_space') as info:
        s.create('a', 10_000)
    assert info.value.errno == errno.ENOSPC and len(calls) == 2
    assert s.create('b', 5_000).nbytes == 40_000
    s.close()


def test_failed_allocation_gives_the_space_back(monkeypatch, tmp_path):
    '''The disk fills between the check and the allocation (another job):
    OSError at creation, and what the file system had taken for the dataset
    before it ran out (XFS keeps it) is released.'''
    monkeypatch.setenv('ZCCSDT_TMPDIR', str(tmp_path))
    seen = []

    def full(fd, offset, size):
        os.ftruncate(fd, offset + size // 2)                # half of it was there
        seen.append((fd, offset, os.fstat(fd).st_size))
        raise OSError(errno.ENOSPC, 'No space left on device')
    monkeypatch.setattr(os, 'posix_fallocate', full)
    s = Scratch(prefix='zccsdt_test_', what='T3')
    with pytest.raises(OSError, match='cannot allocate .* for T3') as info:
        s.create('a', 1_000_000)
    fd, offset, size = seen[0]
    assert info.value.errno == errno.ENOSPC and size == offset + 4_000_000
    assert os.fstat(fd).st_size == offset and scratch.allocated() == 0
    s.close()


@pytest.mark.parametrize('where', ['/dev/shm', '/tmp'])
def test_allocation_on_the_node_local_file_systems(where):
    '''posix_fallocate takes the space at once on tmpfs and on the local
    disk: the free space drops by the size of the dataset when it is
    created, before a byte is written.'''
    if not os.path.isdir(where) or not os.access(where, os.W_OK):
        pytest.skip('no %s' % where)
    n = 16 << 20
    before = scratch.disk_free(where)
    if before < 4 * n:
        pytest.skip('%s is full' % where)
    import tempfile
    with tempfile.TemporaryDirectory(dir=where, prefix='zccsdt_test_') as d:
        s = Scratch(os.path.join(d, 'a.h5'))
        s.create('a', n // 8)
        assert os.stat(s.filename).st_blocks * 512 >= n
        # (other processes write there too: half of it at least)
        assert before - scratch.disk_free(where) >= n // 2
        s.close()


# ----------------------------------------------------------------- the store

def _steps(d, sizes, nit, api='classic', seed=0):
    rng = np.random.default_rng(seed)
    x = [rng.standard_normal(n) for n in sizes]
    out = []
    for it in range(nit):
        new = [0.5 * a + 0.1 * rng.standard_normal(a.size) / (it + 1) for a in x]
        if api == 'buffers':
            sx, se = d.buffers(new)
            for a, b, c, o in zip(sx, se, new, x):
                a[:] = c
                b[:] = c - o
            x = [np.array(a) for a in d.update(sx, errors=se)]
        else:
            x = [np.array(a) for a in d.update(new)]
        out.append(x)
    return out


@pytest.mark.parametrize('api', ['classic', 'buffers'])
def test_split_store(monkeypatch, tmp_path, api):
    '''Some vectors in memory, the others in the file: the same vectors as
    all in memory, through and beyond a full subspace; the vectors of the
    file are all allocated when the first one is stored.'''
    monkeypatch.setattr(diis, 'INCORE_SIZE', 100)
    monkeypatch.setenv('ZCCSDT_TMPDIR', str(tmp_path))
    sizes = (50, 700)
    ref = _steps(DIIS(space=5, incore=True), sizes, 12, api)
    for k in (0, 2, 5):
        d = DIIS(space=5, in_memory=k)
        base = scratch.allocated()
        got = _steps(d, sizes, 2, api)
        # 5 - k pairs of the large part at once
        assert scratch.allocated() - base == (5 - k) * 2 * 700 * 8
        d.close()
        d = DIIS(space=5, in_memory=k)
        got = _steps(d, sizes, 12, api)
        assert d.on_disk() == (k < 5) and len(d.xs) == 5
        assert sum(isinstance(x[1], np.memmap) for x in d.xs) == 5 - k
        assert sum(isinstance(e[1], np.memmap) for e in d.es) == 5 - k
        assert not any(isinstance(x[0], np.memmap) for x in d.xs)
        for a, b in zip(ref, got):
            for p, q in zip(a, b):
                assert np.array_equal(p, q)
        d.close()
        assert scratch.allocated() == base


def test_first_vector_fails_when_the_file_does_not_hold_all(monkeypatch, tmp_path):
    '''The vectors of the file are allocated together: a disk that holds
    three of five says so with the first one, not in the fourth iteration.'''
    monkeypatch.setattr(diis, 'INCORE_SIZE', 100)
    monkeypatch.setenv('ZCCSDT_TMPDIR', str(tmp_path))
    pair = 2 * 700 * 8
    d = DIIS(space=5)
    with disk.limit(3.5 * pair / 1e6):
        with pytest.raises(OSError, match='no room .* a DIIS vector') as info:
            d.buffers([np.zeros(50), np.zeros(700)])
    assert info.value.errno == errno.ENOSPC
    d.close()
    assert scratch.allocated() == 0


# ------------------------------------------------------------- CCSDT, dense

@pytest.fixture(scope='module')
def hf():
    mol = gto.M(atom='F 0 0 0; H 0 0 0.917', basis='6-31g', verbose=0)
    return scf.RHF(mol).run(conv_tol=1e-12)


@pytest.fixture(scope='module')
def hf_run(hf):
    eris = E.from_scf(scf.addons.convert_to_ghf(hf), frozen=4)
    kw = dict(conv_tol=1e-9, conv_tol_normt=1e-6)
    ref = solver.kernel(eris, 'ccsdt', solver.Options(incore_complete=True, **kw))
    assert ref.converged and ref.diis_plan.n_memory == 8 and ref.t3.size >= 1000
    return eris, kw, ref


def _mb(nbytes):
    return nbytes / 1e6


def test_spin_orbital_solver_in_each_case(hf_run, monkeypatch, tmp_path):
    '''Packed-T3 CCSDT with the disk budget forced low: (a) the file, (b)
    file and memory - the energy and the iterations of the run without a
    limit - (c) a smaller subspace, (d) DIIS without T3: the same energy.
    With T3 mapped the memory side gives nothing, and the residual of (d)
    is one more array of the file.'''
    eris, kw, ref = hf_run
    monkeypatch.setattr(diis, 'INCORE_SIZE', 1000)
    monkeypatch.setenv('ZCCSDT_TMPDIR', str(tmp_path))
    t3 = ref.t3.nbytes
    pair = 2 * t3
    tight = ccsdt_opt.MemoryPlan(spare=0)           # everything in memory, nothing beyond
    mapped = ccsdt_opt.MemoryPlan(t3_file=True, disk=t3)

    def run(**o):
        r = solver.kernel(eris, 'ccsdt', solver.Options(**kw, **o))
        assert r.converged and abs(r.e_tot - ref.e_tot) < 1e-8 and abs(r.t3 - ref.t3).max() < 1e-5
        assert scratch.allocated() == 0 and list(tmp_path.iterdir()) == []
        p = r.diis_plan
        assert p.disk <= p.disk_budget and (p.memory_budget is None or p.memory <= p.memory_budget)
        return r, p

    full, p = run()
    assert (p.case, p.n_file, p.space) == ('a', 8, 8) and p.pair == pair
    r, p = run(max_disk=_mb(8.5 * pair), plan=tight)
    assert (p.case, p.n_file) == ('a', 8) and r.ncycle == full.ncycle and r.e_tot == full.e_tot
    r, p = run(max_disk=_mb(5.5 * pair))
    assert (p.case, p.n_file, p.n_memory, p.space) == ('b', 5, 3, 8)
    assert r.ncycle == full.ncycle and abs(r.e_tot - full.e_tot) < 1e-12
    r, p = run(max_disk=_mb(4.5 * pair), plan=tight)
    assert (p.case, p.n_file, p.n_memory, p.space) == ('c', 4, 0, 4) and p.reduced
    r, p = run(max_disk=_mb(4.5 * pair), plan=dataclasses.replace(tight, spare=int(1.5 * pair)))
    assert (p.case, p.n_file, p.n_memory, p.space) == ('c', 4, 1, 5)
    r, p = run(max_disk=_mb(1.5 * pair), plan=tight)
    assert p.case == 'd' and not p.with_big and p.disk == 0 and not scratch.is_mapped(r.t3)
    without, p = run(diis_t3=False)
    assert not p.with_big and not p.reduced and without.ncycle == r.ncycle and without.e_tot == r.e_tot
    # T3 mapped by the memory plan: it comes first
    r, p = run(max_disk=_mb(9.5 * t3), plan=mapped)
    assert (p.case, p.n_file, p.n_memory, p.disk_budget) == ('c', 4, 0, int(8.5 * t3)) and scratch.is_mapped(r.t3)
    r, p = run(max_disk=_mb(4.5 * t3), plan=mapped)
    assert p.case == 'd' and scratch.is_mapped(r.t3) and r.ncycle == without.ncycle


def test_spin_orbital_solver_stops_before_it_allocates(hf_run, monkeypatch, tmp_path):
    '''(e): what is asked for explicitly and cannot be had is an error with
    the numbers before the first iteration, with nothing allocated.'''
    eris, kw, ref = hf_run
    monkeypatch.setattr(diis, 'INCORE_SIZE', 1000)
    monkeypatch.setenv('ZCCSDT_TMPDIR', str(tmp_path))
    t3 = ref.t3.nbytes
    made = []
    monkeypatch.setattr(Scratch, 'create', lambda self, *a, **k: made.append(a) or 1 / 0)
    monkeypatch.setattr(ccsdt_opt, 'update_amps', lambda *a, **k: made.append('step') or 1 / 0)
    tight = ccsdt_opt.MemoryPlan(spare=0)
    # diis_t3 = True and not even three vectors
    with pytest.raises(disk.DiskFullError, match=r'DIIS with T3 \(diis_t3 = True\) needs 70.4 kB per stored .* '
                       r'3 of them at least \(8 asked for\): 0.2 MB; the scratch location .* has 0.1 MB'):
        solver.kernel(eris, 'ccsdt', solver.Options(max_disk=_mb(3 * t3), diis_t3=True, plan=tight, **kw))
    # the mapped T3 of the memory plan does not fit
    mapped = ccsdt_opt.MemoryPlan(t3_file=True, disk=t3)
    with pytest.raises(disk.DiskFullError, match='the memory plan puts 35.2 kB .* max_disk 17.6 kB'):
        solver.kernel(eris, 'ccsdt', solver.Options(max_disk=_mb(0.5 * t3), plan=mapped, **kw))
    # ... nor T3 and its residual without DIIS on T3
    with pytest.raises(disk.DiskFullError, match='puts 70.4 kB for T3 and its residual'):
        solver.kernel(eris, 'ccsdt', solver.Options(max_disk=_mb(1.5 * t3), plan=mapped, **kw))
    # incore_complete and a node without the memory
    monkeypatch.setattr(memory, 'available', lambda: 10 * t3)
    full = ccsdt_opt.MemoryPlan(need=1 << 40)       # (more than the node has left)
    with pytest.raises(MemoryError, match='incore_complete: 8 DIIS vectors with T3 take 0.6 MB'):
        solver.kernel(eris, 'ccsdt', solver.Options(incore_complete=True, plan=full, **kw))
    assert made == [] and scratch.allocated() == 0 and scratch.limit() is None


def test_memory_of_the_split_store(monkeypatch, tmp_path):
    '''The vectors that the plan keeps in memory take what the plan says:
    the peak of a run with three of them in memory is that of the run with
    all in the file plus three pairs (and less than one T3 more).'''
    monkeypatch.setattr(diis, 'INCORE_SIZE', 1000)
    monkeypatch.setenv('ZCCSDT_TMPDIR', str(tmp_path))
    mol = gto.M(atom=WATER, basis='6-31g', verbose=0)
    eris = E.from_scf(scf.addons.convert_to_ghf(scf.RHF(mol).run(conv_tol=1e-12)), frozen=2)
    size = ccsdt_opt.engine_for(eris, np.float64).size
    t3 = 8 * size
    o, v = eris.nocc, eris.nmo - eris.nocc
    # (the packed T2 is above the threshold here and goes where T3 goes)
    pair = 2 * disk.file_bytes([o * v, o * (o - 1) // 2 * (v * (v - 1) // 2), size], 8)
    assert t3 > 250_000 and 2 * t3 < pair < 2.5 * t3
    kw = dict(conv_tol=1e-12, conv_tol_normt=1e-12, max_cycle=7, diis_space=4)
    solver.kernel(eris, 'ccsdt', solver.Options(max_cycle=1))      # (what the first step leaves allocated)
    peaks = []
    for n_mem in (0, 3):
        dp = disk.DiisPlan(4, 4, n_file=4 - n_mem, n_memory=n_mem, pair=pair)
        gc.collect()
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        r = solver.kernel(eris, 'ccsdt', solver.Options(diis_plan=dp, **kw))
        peaks.append(tracemalloc.get_traced_memory()[1] - base)
        tracemalloc.stop()
        assert r.ncycle == 7 and r.diis_plan is dp
        del r
    assert 2.5 * pair < peaks[1] - peaks[0] <= 3 * pair + 0.25 * t3, (peaks, pair, t3)


# ----------------------------------------------------- the other solvers

def test_spin_adapted_solver_in_each_case(monkeypatch, tmp_path):
    from zccsdt.sa import rccsdt as S, rccsdt_ref as R
    mol = gto.M(atom=WATER, basis='6-31g', verbose=0)
    ints = R.from_scf(scf.RHF(mol).run(conv_tol=1e-12), ncore=1, full=False)
    kw = dict(conv_tol=1e-10, conv_tol_normt=1e-7)
    ref = S.kernel(ints, incore_complete=True, **kw)
    monkeypatch.setattr(diis, 'INCORE_SIZE', 1000)
    monkeypatch.setenv('ZCCSDT_TMPDIR', str(tmp_path))
    t3 = ref.t3.nbytes
    big = t3 + ref.t2.nbytes                # (T2 is above the threshold here: in the file too)
    pair = 2 * big
    tight = S.MemoryPlan(spare=0)
    mapped = S.MemoryPlan(t3_file=True, disk=2 * t3)

    def run(**o):
        r = S.kernel(ints, **kw, **o)
        assert r.converged and abs(r.e_tot - ref.e_tot) < 1e-9
        assert scratch.allocated() == 0 and list(tmp_path.iterdir()) == []
        assert r.diis_t3 == r.diis_plan.with_big
        return r, r.diis_plan

    full, p = run()
    assert (p.case, p.n_file, p.pair) == ('a', 8, pair) and full.diis_on_disk
    r, p = run(max_disk=_mb(5.5 * pair))
    assert (p.case, p.n_file, p.n_memory) == ('b', 5, 3) and r.ncycle == full.ncycle
    assert abs(r.e_tot - full.e_tot) < 1e-12
    r, p = run(max_disk=_mb(4.5 * pair), plan=tight)
    assert (p.case, p.space, p.n_file) == ('c', 4, 4)
    r, p = run(max_disk=_mb(1.5 * pair), plan=tight)
    assert p.case == 'd' and not r.diis_on_disk
    without, p = run(diis_t3=False)
    assert not p.with_big and without.ncycle == r.ncycle
    # T3 and its residual mapped by the memory plan come first
    r, p = run(max_disk=_mb(2 * t3 + 3.5 * pair), plan=mapped)
    assert (p.case, p.space) == ('c', 3) and scratch.is_mapped(r.t3)
    r, p = run(max_disk=_mb(2 * t3 + 2.5 * pair), plan=mapped)
    assert p.case == 'd' and scratch.is_mapped(r.t3)
    with pytest.raises(disk.DiskFullError, match='diis_t3 = True'):
        S.kernel(ints, max_disk=_mb(2 * t3 + 2.5 * pair), plan=mapped, diis_t3=True, **kw)
    with pytest.raises(disk.DiskFullError, match='the memory plan puts'):
        S.kernel(ints, max_disk=_mb(1.5 * t3), plan=mapped, **kw)
    assert scratch.allocated() == 0 and scratch.limit() is None


def test_memory_plans_say_what_they_map():
    '''disk and spare of the memory plans of CCSDT over a scan of the
    budget: nothing on disk with everything in memory; T3 (and what else
    is mapped) counted when memory is short; need + spare is the budget.'''
    from zccsdt.sa import rccsdt as S
    from zccsdt.sa.t3blocks import Blocks
    for name, plan_memory, args, t3, pp in (
            ('sa', S.plan_memory, (8, 120), 8 * Blocks.count(8) * 120 ** 3,
             8 * ((120 * 121 // 2) ** 2 + (120 * 119 // 2) ** 2)),
            ('so', lambda o, v, b=None: ccsdt_opt.plan_memory(o, v, 16, b), (30, 186),
             16 * 4060 * 1055240, 16 * (186 * 185 // 2) ** 2)):
        full = plan_memory(*args)
        assert full.disk == 0 and full.spare is None and not full.t3_file
        seen = set()
        for f in (2.0, 1.0, 0.9, 0.7, 0.5, 0.35, 0.25, 0.15, 0.1, 0.05, 0.02):
            try:
                pl = plan_memory(*args, int(f * full.need))
            except MemoryError:
                continue
            r3 = pl.t3_file if pl.r3_file is None else pl.r3_file
            kept = pl.pp_order == 'batch' and pl.pp_bytes is not None and pl.pp_bytes < pp
            # (the residual of the spin-orbital solver is built in the DIIS storage)
            want = (t3 if pl.t3_file else 0) + (t3 if r3 and name == 'sa' else 0) \
                + (pp if pl.pp_file else 0) + (pp if kept else 0)
            assert pl.disk == want, (name, f, pl)
            assert pl.spare == max(0, int(f * full.need) - pl.need)
            seen.add((bool(pl.t3_file), bool(pl.pp_file)))
        assert (False, False) in seen and (True, True) in seen, (name, seen)


@pytest.fixture(scope='module')
def water_sto():
    mol = gto.M(atom=WATER, basis='sto-3g', verbose=0)
    eris = E.from_scf(scf.addons.convert_to_ghf(scf.RHF(mol).run(conv_tol=1e-12)), packed_vvvv=True)
    r3 = solver.kernel(eris, 'ccsdt', solver.Options(conv_tol=1e-11, conv_tol_normt=1e-9))
    assert r3.converged
    return eris, r3


def test_ccsdtq_in_each_case(water_sto, monkeypatch, tmp_path):
    '''CCSDTQ: the DIIS vectors of T4 (and T3) by the same rule; (d) is
    DIIS without T4.'''
    eris, r3 = water_sto
    monkeypatch.setenv('ZCCSDT_TMPDIR', str(tmp_path))
    monkeypatch.setattr(ccsdtq, 'INCORE_SIZE', 100)
    no, nv = eris.nocc, eris.nmo - eris.nocc
    amps = dict(t1=r3.t1, t2=r3.t2, t3=r3.t3)
    kw = dict(conv_tol=1e-10, conv_tol_normt=1e-8)
    ref = ccsdtq.kernel(eris, ccsdtq.Options(incore_complete=True, **kw), **amps)
    assert ref.converged and ref.diis_plan.case == 'm' and ref.diis_plan.n_memory == 8
    pl = ccsdtq.plan(no, nv, 8, None)
    assert pl.diis_file and pl.disk == 0 and pl.spare is None
    tight = dataclasses.replace(pl, spare=0)
    # (the packed T2 is above the threshold of this test and goes to the file with T3 and T4)
    pair = 2 * (ref.t4.nbytes + ref.t3.nbytes + 8 * (no * (no - 1) // 2) * (nv * (nv - 1) // 2))

    def run(**o):
        r = ccsdtq.kernel(eris, ccsdtq.Options(**kw, **o), **amps)
        assert r.converged and abs(r.e_tot - ref.e_tot) < 1e-8
        assert scratch.allocated() == 0 and list(tmp_path.iterdir()) == []
        return r, r.diis_plan

    full, p = run(plan=pl)
    assert (p.case, p.n_file, p.pair, p.what) == ('a', 8, pair, 'T4')
    r, p = run(plan=pl, max_disk=_mb(5.5 * pair))
    assert (p.case, p.n_file, p.n_memory) == ('b', 5, 3) and r.ncycle == full.ncycle
    assert abs(r.e_tot - full.e_tot) < 1e-12
    r, p = run(plan=tight, max_disk=_mb(4.5 * pair))
    assert (p.case, p.space) == ('c', 4)
    r, p = run(plan=tight, max_disk=_mb(1.5 * pair))
    assert p.case == 'd' and not p.with_big
    without, p = run(plan=pl, diis_t4=False)
    assert not p.with_big and without.ncycle == r.ncycle
    with pytest.raises(disk.DiskFullError, match=r'DIIS with T4 \(diis_t4 = True\)'):
        ccsdtq.kernel(eris, ccsdtq.Options(plan=tight, max_disk=_mb(1.5 * pair), diis_t4=True, **kw), **amps)
    # T4 and its residual mapped by the memory plan: they come first, and (d) alternates the two
    t4 = ref.t4.nbytes
    mapped = dataclasses.replace(tight, t4_file=True, r4_file=True, disk=2 * t4)
    r, p = run(plan=mapped, max_disk=_mb(2 * t4 + 3.5 * pair))
    assert (p.case, p.space) == ('c', 3) and scratch.is_mapped(r.t4)
    r, p = run(plan=mapped, max_disk=_mb(2 * t4 + 0.5 * pair))
    assert p.case == 'd' and scratch.is_mapped(r.t4) and r.ncycle == without.ncycle
    with pytest.raises(disk.DiskFullError, match='the memory plan puts'):
        ccsdtq.kernel(eris, ccsdtq.Options(plan=mapped, max_disk=_mb(1.5 * t4), **kw), **amps)
    assert scratch.allocated() == 0 and scratch.limit() is None
    # the plan says what it maps and what it leaves
    low = ccsdtq.plan(no, nv, 8, pl.floor)
    assert low.t4_file and low.r4_file and low.xv_file and low.disk > 2 * t4 and low.spare == 0


def test_lambda_in_each_case(water_sto, monkeypatch, tmp_path):
    '''The Lambda equations of CCSDT: l3 is the large part.'''
    eris, r3 = water_sto
    monkeypatch.setenv('ZCCSDT_TMPDIR', str(tmp_path))
    monkeypatch.setattr(diis, 'INCORE_SIZE', 100)
    kw = dict(conv_tol_normt=1e-8)
    ref = lambda_opt.kernel(eris, r3.t1, r3.t2, r3.t3, solver.Options(incore_complete=True, **kw))
    assert ref.converged and ref.diis_plan.n_memory == 8
    o, v = r3.t1.shape
    big = ref.l3.nbytes + 8 * (o * (o - 1) // 2) * (v * (v - 1) // 2)       # (l2 is above the threshold here)
    pair = 2 * big
    dtype = ref.l3.dtype
    tight = lambda_opt.plan_for(eris, r3.t1, dtype, solver.Options())
    tight.spare = 0

    def run(**o):
        r = lambda_opt.kernel(eris, r3.t1, r3.t2, r3.t3, solver.Options(**kw, **o))
        assert r.converged and abs(r.l3 - ref.l3).max() < 1e-6 and abs(r.l2 - ref.l2).max() < 1e-6
        assert scratch.allocated() == 0 and list(tmp_path.iterdir()) == []
        return r, r.diis_plan

    full, p = run()
    assert (p.case, p.n_file, p.pair, p.what) == ('a', 8, pair, 'l3')
    r, p = run(max_disk=_mb(5.5 * pair))
    assert (p.case, p.n_file, p.n_memory) == ('b', 5, 3) and r.ncycle == full.ncycle
    r, p = run(max_disk=_mb(3.5 * pair), plan=tight)
    assert (p.case, p.space) == ('c', 3)
    r, p = run(max_disk=_mb(1.5 * pair), plan=tight)
    assert p.case == 'd' and not p.with_big
    without, p = run(diis_t3=False)
    assert not p.with_big and without.ncycle == r.ncycle
    with pytest.raises(disk.DiskFullError, match=r'DIIS with l3 \(diis_t3 = True\)'):
        lambda_opt.kernel(eris, r3.t1, r3.t2, r3.t3, solver.Options(max_disk=_mb(1.5 * pair), plan=tight,
                                                                    diis_t3=True, **kw))
    assert scratch.allocated() == 0 and scratch.limit() is None


# ------------------------------------------------------- report and dry run

def test_front_end_reports_the_placement(hf, monkeypatch, tmp_path):
    '''max_disk on the classes: the report has the disk budget and where the
    DIIS vectors are, with the numbers of a reduction; the energy is that
    of the run without a limit.'''
    import zccsdt
    monkeypatch.setattr(diis, 'INCORE_SIZE', 1000)
    monkeypatch.setenv('ZCCSDT_TMPDIR', str(tmp_path))

    def run(cls, mf, frozen, **o):
        cc = cls(mf, frozen=frozen)
        cc.stdout, cc.verbose, cc.conv_tol = io.StringIO(), 4, 1e-9
        for k, v in o.items():
            setattr(cc, k, v)
        cc.kernel()
        assert cc.converged and scratch.allocated() == 0 and scratch.limit() is None
        return cc, cc.stdout.getvalue()

    for cls, mf, frozen in ((zccsdt.GCCSDT, scf.addons.convert_to_ghf(hf), 4), (zccsdt.RCCSDT, hf, 2)):
        assert cls.max_disk is None and cls.diis_t3 is None
        ref, out = run(cls, mf, frozen)
        p = ref.diis_plan_used
        assert p.case == 'a' and p.n_file == 8
        assert ' disk              budget ' in out and 'less a reserve' in out
        assert '   DIIS vectors      8 vectors with T3, ' in out and 'they take %s of disk and 0 B of memory' % (
            zccsdt.ccsdt._size(p.disk)) in out
        # (the memory beyond the plan: none)
        monkeypatch.setattr(memory, 'budget', lambda max_memory=None: 0 if max_memory == -1 else None)
        plan = dataclasses.replace(ref.plan_used, spare=0)
        cc, out = run(cls, mf, frozen, max_disk=_mb(4.5 * p.pair), plan=plan)
        q = cc.diis_plan_used
        assert (q.case, q.space, q.n_file) == ('c', 4, 4) and abs(cc.e_tot - ref.e_tot) < 1e-8
        assert 'budget %s for scratch files (max_disk' % zccsdt.ccsdt._size(int(4.5 * p.pair)) in out
        assert '4 vectors with T3' in out and 'reduced from the 8 asked for (diis_space): disk' in out
        cc, out = run(cls, mf, frozen, max_disk=_mb(1.5 * p.pair), plan=plan)
        assert cc.diis_plan_used.case == 'd' and abs(cc.e_tot - ref.e_tot) < 1e-8
        assert '8 vectors without T3 (it takes its Jacobi step): not even 3 pairs of' in out
        with pytest.raises(disk.DiskFullError, match='diis_t3 = True'):
            run(cls, mf, frozen, max_disk=_mb(1.5 * p.pair), plan=plan, diis_t3=True)
        assert scratch.limit() is None
        monkeypatch.undo()
        monkeypatch.setattr(diis, 'INCORE_SIZE', 1000)
        monkeypatch.setenv('ZCCSDT_TMPDIR', str(tmp_path))


def test_dry_run_of_the_benzene_example():
    '''Benzene / cc-pVDZ with X2C, 30 occupied and 186 virtual spinors: the
    plans for 900 GB of disk with 320 GB and with 1.4 TB of memory (the
    example of docs/usage.md), from the sizes alone.'''
    from math import comb
    t3 = 16 * comb(30, 3) * comb(186, 3)
    assert round(t3 / 1e9, 1) == 68.5
    plan, p = disk.dry_run(30, 186, complex, max_memory=320000, max_disk=900000)
    assert not plan.t3_file and plan.disk == 0 and plan.need < 160e9
    assert (p.case, p.space, p.n_file, p.n_memory, p.pair) == ('c', 7, 6, 1, 2 * t3)
    assert plan.need + p.memory <= 320e9 and plan.disk + p.disk <= 900e9
    text = disk.describe(plan, p)
    assert '7 vectors with T3, 137.1 GB per pair: 6 in a scratch file (822.6 GB of disk), 1 in memory' in text
    plan, p = disk.dry_run(30, 186, complex, max_memory=1400000, max_disk=900000)
    assert (p.case, p.space, p.n_file, p.n_memory) == ('b', 8, 6, 2)
    assert plan.need + p.memory <= 1400e9 and plan.disk + p.disk <= 900e9
    # without limits: the default placement, 1.1 TB of disk
    plan, p = disk.dry_run(30, 186, complex)
    assert (p.case, p.n_file) == ('a', 8) and round(p.disk / 1e12, 1) == 1.1
    # little memory: T3 is mapped and comes out of the disk first; nothing of DIIS in memory
    plan, p = disk.dry_run(30, 186, complex, max_memory=100000, max_disk=900000)
    assert plan.t3_file and plan.disk >= t3 and (p.case, p.space, p.n_memory) == ('c', 6, 0)
    assert plan.disk + p.disk <= 900e9
    plan, p = disk.dry_run(30, 186, complex, max_memory=100000, max_disk=400000)
    assert p.case == 'd' and plan.disk + p.disk <= 400e9
    with pytest.raises(disk.DiskFullError):
        disk.dry_run(30, 186, complex, max_memory=100000, max_disk=60000)
    with pytest.raises(disk.DiskFullError, match='diis_t3 = True'):
        disk.dry_run(30, 186, complex, max_memory=100000, max_disk=400000, diis_t3=True)
    with pytest.raises(MemoryError, match='incore_complete'):
        disk.dry_run(30, 186, complex, max_memory=320000, max_disk=900000, incore_complete=True)
    # over a scan of both budgets: within both, the subspace monotone
    last = {}
    for mem in (60000, 100000, 200000, 320000, 600000, 1400000):
        for dsk in (100000, 300000, 600000, 900000, 1200000):
            try:
                plan, p = disk.dry_run(30, 186, complex, max_memory=mem, max_disk=dsk)
            except (MemoryError, disk.DiskFullError):
                assert not last.get(dsk)
                continue
            assert plan.need + p.memory <= mem * 1e6 and plan.disk + p.disk <= dsk * 1e6
            n = p.space if p.with_big else 0
            assert n >= last.get(dsk, 0) or plan.t3_file
            last[dsk] = n
