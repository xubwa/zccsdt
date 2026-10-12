'''CC4 without stored T4: the memory plan of the pass over the occupied
quadruples and its processes (zccsdt.cc.cc4fly.plan, run_sectors):

- the rules of the plan: a floor with an error that names the numbers,
  nothing shrinks when the budget grows, never more than the budget;
- the peak of a pass (tracemalloc) against the plan, in sectors and on the
  packed layout, with everything in memory and at the floor;
- several processes against one: the same R2, R3 (real and complex), the
  kernel of the row against that of the (Q) gradient;
- the front end: the plan of the run, the processes, an error before the
  first iteration where the memory does not hold the floor.

HBr / cc-pVDZ-DK, X2C, 28 spinors frozen (8 + 28, the labels 2 m_j).'''
import tracemalloc

import numpy as np
import pytest
from pyscf import gto, x2c

import zccsdt
from zccsdt.cc import cc4fly, ccsdt_opt, t4engine

# the packed layout against the sectors: what the integrals have outside the blocks of the labels
DENSE = 1e-9


@pytest.fixture(scope='module')
def hbr():
    mf = x2c.UHF(gto.M(atom='Br 0 0 0; H 0 0 1.4145', basis='cc-pvdz-dk', verbose=0))
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-8
    mf.kernel()
    cc = zccsdt.CC4(mf, frozen=28)
    cc.verbose, cc.max_cycle = 0, 3
    cc.kernel()
    eris = cc._eris
    assert eris.labels is not None and cc.plan_quadruples.sectors
    T3 = np.ascontiguousarray(ccsdt_opt.packed_t3(eris, cc.t3))
    assert np.size(cc.t3) < T3.size             # (the iterations ran in the blocks of the labels)
    ints = t4engine.Dressed.from_eris(eris, cc.t1)
    e = np.asarray(eris.fock).diagonal().real
    return mf, cc, eris, ints, e, T3


def _err(a, b):
    return max(abs(x - y).max() / max(abs(y).max(), 1e-30) for x, y in zip(a, b))


def _close(a, b, tol=1e-13):
    err = _err(a, b)
    assert err < tol, err
    return True


def test_plan_rules(hbr, monkeypatch):
    mf, cc, eris, ints, e, T3 = hbr
    no, nv, lab = 8, 28, np.asarray(eris.labels)
    full = cc4fly.plan(no, nv, 8, None, lab, 0, workers=6)
    assert full.sectors and full.workers == 6 and full.xo_workers == 6 and full.xv == 7
    assert full.need >= full.fixed + 6 * (full.per_worker + cc4fly.WORKER_HEAP)
    with pytest.raises(MemoryError, match=r'need \d+ MB .* \d+ MB are left'):
        cc4fly.plan(no, nv, 8, full.floor - 1, lab, 0, workers=6)
    least = cc4fly.plan(no, nv, 8, full.floor, lab, 0, workers=6)
    assert (least.workers, least.xv) == (1, 3) and least.need == least.floor == full.floor
    last = least
    for budget in np.linspace(full.floor, 1.02 * full.need, 80):
        p = cc4fly.plan(no, nv, 8, int(budget), lab, 0, workers=6)
        assert p.need <= budget and p.floor == full.floor
        assert p.workers >= last.workers and p.xv >= last.xv and p.xo_workers <= p.workers
        assert p.workers == 1 or p.xv == full.xv        # (first Xv for all occupied indices, then processes)
        last = p
    assert (last.workers, last.xv, last.need) == (full.workers, full.xv, full.need)
    # the packed layout: one process, the batches of the (Q) plan
    dense = cc4fly.plan(no, nv, 8, None)
    assert not dense.sectors and dense.workers == 1 and dense.floor <= dense.need
    with pytest.raises(MemoryError, match='MB are left'):
        cc4fly.plan(no, nv, 8, dense.floor - 1)
    assert cc4fly.plan(no, nv, 8, dense.floor).need <= dense.floor
    # by default one process where T4 is small, else the cores - or what the budget holds
    assert cc4fly.plan(no, nv, 8, None, lab, 0).workers == 1
    monkeypatch.setattr(cc4fly, '_cores', lambda: 24)
    big = np.concatenate((np.resize([1, -1, 3, -3, 1, -1], 26), np.resize([1, -1, 3, -3, 5, -5, 7, -7, 1, -1], 60)))
    p24 = cc4fly.plan(26, 60, 8, None, big, 0)
    assert p24.workers == 24
    few = cc4fly.plan(26, 60, 8, p24.fixed + 5 * (p24.per_worker + cc4fly.WORKER_HEAP), big, 0)
    assert 2 <= few.workers <= 5 and few.need <= p24.fixed + 5 * (p24.per_worker + cc4fly.WORKER_HEAP)


def _peak(fn):
    tracemalloc.start()
    try:
        base = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        out = fn()
        return out, tracemalloc.get_traced_memory()[1] - base
    finally:
        tracemalloc.stop()


def test_peak_within_plan(hbr):
    mf, cc, eris, ints, e, T3 = hbr
    no, nv, lab = 8, 28, np.asarray(eris.labels)
    full = cc4fly.plan(no, nv, 8, None, lab, 0, workers=1)
    ref, peak = _peak(lambda: cc4fly.run_sectors(ints, e, lab, 0, cc.t2, T3, plan_=full))
    assert peak <= full.need, (peak, full.need)
    assert peak > 0.2 * full.need       # (the plan is not far above what a pass takes)
    least = cc4fly.plan(no, nv, 8, full.floor, lab, 0, workers=1)
    assert least.xv == 3 < full.xv
    out, peak = _peak(lambda: cc4fly.run_sectors(ints, e, lab, 0, cc.t2, T3, plan_=least))
    assert peak <= least.need and _close(out, ref)
    # the packed layout, everything in memory and at the floor of its plan
    dense = cc4fly.plan(no, nv, 8, None)
    out, peak = _peak(lambda: cc4fly.run(ints, e, cc.t2, T3, plan_=dense))
    assert peak <= dense.need and _close(out, ref, DENSE)
    tight = cc4fly.plan(no, nv, 8, dense.floor)
    assert tight.dense.cols22 < dense.dense.cols22
    out, peak = _peak(lambda: cc4fly.run(ints, e, cc.t2, T3, plan_=tight))
    assert peak <= tight.need and _close(out, ref, DENSE)


def test_processes(hbr, monkeypatch):
    mf, cc, eris, ints, e, T3 = hbr
    lab = np.asarray(eris.labels)
    ref = cc4fly.run_sectors(ints, e, lab, 0, cc.t2, T3, workers=1)
    stats = {}
    out = cc4fly.run_sectors(ints, e, lab, 0, cc.t2, T3, workers=3, stats=stats)
    assert stats['workers'] == 3 and stats['shared'] > 0 and _close(out, ref)
    # complex amplitudes, a frequency, two drives
    ph = np.exp(0.3j)
    drives = [(ints, ph * cc.t2, ph * T3, 0.5), (ints, cc.t2, T3 * (1 + 0j), -0.25j)]
    kw = dict(omega=0.1, drives=drives)
    refz = cc4fly.run_sectors(ints, e, lab, 0, None, None, workers=1, **kw)
    assert _close(cc4fly.run_sectors(ints, e, lab, 0, None, None, workers=2, **kw), refz)
    assert _close(cc4fly.run(ints, e, None, None, **kw), refz, DENSE)
    # the kernel of the row against that of the (Q) gradient
    monkeypatch.setattr(cc4fly, 'LEAN', False)
    assert _close(cc4fly.run_sectors(ints, e, lab, 0, cc.t2, T3, workers=1), ref)
    assert _close(cc4fly.run_sectors(ints, e, lab, 0, None, None, workers=1, **kw), refz)


def test_a_failing_process_is_an_error(hbr, monkeypatch):
    mf, cc, eris, ints, e, T3 = hbr

    def broken(self, b, I, XVs):
        raise ValueError('quadruple %d' % I)
    monkeypatch.setattr(cc4fly._SectorPass, 'rhs', broken)
    with pytest.raises(RuntimeError, match='ValueError: quadruple'):
        cc4fly.run_sectors(ints, e, np.asarray(eris.labels), 0, cc.t2, T3, workers=2)


def test_q_in_processes(hbr):
    '''(Q) in sectors with the occupied quadruples shared out among
    processes (quadruples_sym.kernel_processes) against one process.'''
    from zccsdt.cc import quadruples_sym
    mf, cc, eris, ints, e, T3 = hbr
    ref = quadruples_sym.kernel(eris, cc.t2, T3)
    out = quadruples_sym.kernel(eris, cc.t2, T3, workers=3)
    assert abs(ref[1]) > 1e-6 and max(abs(a - b) for a, b in zip(ref, out)) < 1e-13 * abs(ref[1])
    part = quadruples_sym.kernel(eris, cc.t2, T3, stride=3)
    seen = []
    out = quadruples_sym.kernel(eris, cc.t2, T3, stride=3, workers=2, progress=lambda *a: seen.append(a))
    assert max(abs(a - b) for a, b in zip(part, out)) < 1e-13 * abs(ref[1]) and seen[-1][0] == seen[-1][1]


def test_front_end(hbr):
    mf, cc, eris, ints, e, T3 = hbr
    assert cc.plan_quadruples.workers == 1 and cc.t4 is None
    two = zccsdt.CC4(mf, frozen=28)
    two.verbose, two.max_cycle, two.workers = 0, 3, 2
    two.kernel()
    assert two.plan_quadruples.workers == 2 and two.quadruples_stats['workers'] == 2
    assert abs(two.e_corr - cc.e_corr) < 1e-12
    low = zccsdt.CC4(mf, frozen=28)
    low.verbose, low.max_memory = 0, 1
    with pytest.raises(MemoryError):
        low.kernel()
