'''Timing of CC3 - T3 stored, and without stored triples - against CCSDT and
against CCSD(T) on the same spinors (docs/ccn.md, section 9):

    python bench/ccn/cc3_bench.py --geom GEOM --basis BASIS --frozen N [--nvir N] --what WHAT
                                  [--niter 4] [--symmetry 1|0] [--real 1|0] [--sigma 1]

  WHAT   ccsdt | cc3 | cc3-fly : NITER iterations of zccsdt.CCSDT / CC3 (store_t3 True / False);
         with --sigma 1 also the time of a sigma vector of the excited states
         (stored: one Davidson step of two roots; cc3-fly: one folded sigma vector)
         ccsd_t : NITER CCSD iterations and (T) of zccsdt.zccsd.DirectZCCSD
  --nvir keep the lowest N virtual spinors; --symmetry 0: the packed layout
  (no label blocks); --real 0: the spinors as the SCF left them (complex).
         cc4 | q | ccsdtq : NITER iterations of zccsdt.CC4 (no stored T4; --workers N: processes
         over the occupied quadruples, default the cores), of CCSDT and then (Q) (--workers N:
         in N processes; default: one process), of CCSDTQ
One measurement per process; the line RESULT has the numbers.  X2C-1e
spinors of PySCF (x2c.UHF).  ZC_MAXMEM (MB): max_memory.  For cc4 the memory is sampled
by phase (mem_gb: the largest sum of the proportional set sizes of the process and of its
child processes - every page once, shared arrays included; mem_parts_gb: the anonymous
memory of the process, the private pages of the children, the rest; mem_by_phase_gb: the
maximum under each running timer and in each part of the pass over the quadruples).
'''
import argparse
import json
import os
import time

import numpy as np
from pyscf import gto, x2c

import zccsdt
from zccsdt import memory, timing

ap = argparse.ArgumentParser()
ap.add_argument('--geom', required=True)
ap.add_argument('--basis', required=True)
ap.add_argument('--frozen', type=int, default=0)
ap.add_argument('--nvir', type=int, default=0)
ap.add_argument('--what', required=True)
ap.add_argument('--niter', type=int, default=4)
ap.add_argument('--symmetry', type=int, default=1)
ap.add_argument('--real', type=int, default=1)
ap.add_argument('--sigma', type=int, default=0)
ap.add_argument('--workers', type=int, default=0)
a = ap.parse_args()
maxmem = float(os.environ['ZC_MAXMEM']) if os.environ.get('ZC_MAXMEM') else None
mol = gto.M(atom=a.geom, basis=a.basis, verbose=0)
mf = x2c.UHF(mol)
mf.conv_tol = 1e-11
mf.kernel()
nocc = int(mol.nelectron)
order = np.argsort(mf.mo_energy, kind='stable')
frozen = a.frozen
if a.nvir:
    frozen = [int(i) for i in order[:a.frozen]] + [int(i) for i in order[nocc + a.nvir:]]
nthr = int(os.environ.get('OMP_NUM_THREADS', 0))
rec = dict(what=a.what, threads=nthr, basis=a.basis, frozen=a.frozen, nvir=a.nvir, symmetry=a.symmetry, real=a.real,
           host=os.uname().nodename, job=os.environ.get('SLURM_JOB_ID'))


class Memory:
    '''Resident memory by phase, sampled by a thread: the proportional set
    sizes of this process and of its children (the processes of the pass
    over the quadruples) summed - every page once, the arrays they share
    included - with the anonymous memory of this process and the private
    pages of the children as parts.'''

    def __init__(self, dt=1.0):
        import threading
        self.dt, self.peak, self.by, self.parts = dt, 0, {}, (0, 0, 0)
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def sample(self):
        from zccsdt.cc import cc4fly
        phase = timing._stack[-1] if timing._stack else '-'
        if cc4fly.PHASE[0]:
            phase = 'cc4.pass:' + cc4fly.PHASE[0]
        me = os.getpid()
        try:
            with open('/proc/self/task/%d/children' % me) as f:
                kids = [int(x) for x in f.read().split()]
        except OSError:
            kids = []
        anon = memory.resident()
        priv = sum(cc4fly.private_bytes(k) for k in kids)
        tot = cc4fly.pss_bytes(me) + sum(cc4fly.pss_bytes(k) for k in kids)
        if tot > self.peak:
            self.peak, self.parts = tot, (anon, priv, tot - anon - priv)
        self.by[phase] = max(self.by.get(phase, 0), tot)

    def _run(self):
        while not self._done.wait(self.dt):
            try:
                self.sample()
            except (OSError, ValueError, IndexError):
                pass

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._done.set()
        self._thread.join()

    def record(self):
        gb = lambda x: round(x / 1e9, 3)                                   # noqa: E731
        return dict(mem_gb=gb(self.peak), mem_parts_gb=[gb(x) for x in self.parts],
                    mem_by_phase_gb={k: gb(v) for k, v in sorted(self.by.items(), key=lambda kv: -kv[1])[:8]})


def stages(snap, depth=2):
    return {'/'.join(p[1:]): [round(w, 3), round(c / w, 1) if w > 0 else 0] for p, (w, c, n) in timing.since(snap).items()
            if 1 < len(p) <= depth and w > 0.02}


if a.what == 'ccsd_t':
    from zccsdt.zccsd import DirectZCCSD
    cc = DirectZCCSD(mf, frozen=frozen)
    cc.verbose, cc.max_cycle = 0, a.niter
    if maxmem:
        cc.max_memory = maxmem
    stamps = []
    cc.callback = lambda envs: stamps.append(time.perf_counter())
    t0 = time.perf_counter()
    eris = cc.ao2mo()
    stamps.append(time.perf_counter())
    cc.kernel(eris=eris)
    it = np.diff(stamps)
    t1 = time.perf_counter()
    et = cc.ccsd_t()
    rec.update(integrals=stamps[0] - t0, iterations=[round(float(x), 3) for x in it], t=time.perf_counter() - t1,
               e_corr=float(cc.e_corr), e_t=float(np.real(et)), nocc=int(cc.nocc), nmo=int(cc.nmo),
               rss_gb=memory.resident() / 1e9, dtype=str(cc.t2.dtype))
elif a.what in ('cc4', 'q', 'ccsdtq'):
    # CC4 without stored T4 (the CCSDT step in blocks, the quadruples in the sectors of the
    # labels), (Q) after NITER iterations of CCSDT (in blocks), CCSDTQ (in blocks)
    cls = {'cc4': zccsdt.CC4, 'q': zccsdt.CCSDT, 'ccsdtq': zccsdt.CCSDTQ}[a.what]
    cc = cls(mf, frozen=frozen)
    cc.verbose, cc.max_cycle = 0, a.niter
    cc.symmetry, cc.real_spinors = bool(a.symmetry), bool(a.real)
    if maxmem:
        cc.max_memory = maxmem
    if a.what == 'cc4' and a.workers:
        cc.workers = a.workers
    snap = timing.snapshot()
    t0 = time.perf_counter()
    with Memory() as mem:
        cc.kernel()
    if a.what == 'cc4':
        q = cc.plan_quadruples
        q0 = cc._qplan      # (made before the iterations; q: that of the last pass)
        rec.update(mem.record(), workers=q.workers, xo_workers=q.xo_workers, plan_q_gb=q.need / 1e9,
                   workers_least=q0.workers, plan_least_gb=q0.need / 1e9, plan_step_gb=cc.plan_used.need / 1e9,
                   stats={k: v for k, v in (cc.quadruples_stats or {}).items() if k != 'plan'}, maxmem=maxmem)
    rec.update(total=time.perf_counter() - t0, iterations=[round(float(x), 3) for x in cc.timings],
               e_corr=float(np.real(cc.e_corr)), nocc=int(cc.t1.shape[0]), nvir_active=int(cc.t1.shape[1]),
               dtype=str(cc.t2.dtype), peak_gb=(getattr(cc, 'peak_memory', 0) or 0) / 1e9,
               rss_gb=memory.resident() / 1e9, stages=stages(snap, 3))
    if a.what == 'q':
        snap = timing.snapshot()
        t0 = time.perf_counter()
        eq = cc.ccsdt_q(workers=a.workers or None)
        rec.update(q_workers=a.workers)
        rec.update(q=time.perf_counter() - t0, e_q=float(eq), q_stages=stages(snap, 2), rss_q_gb=memory.resident() / 1e9)
else:
    cls = zccsdt.CCSDT if a.what == 'ccsdt' else zccsdt.CC3
    cc = cls(mf, frozen=frozen)
    cc.verbose, cc.max_cycle = 0, a.niter
    cc.symmetry, cc.real_spinors = bool(a.symmetry), bool(a.real)
    if a.what == 'cc3-fly':
        cc.store_t3 = False
    if maxmem:
        cc.max_memory = maxmem
    snap = timing.snapshot()
    t0 = time.perf_counter()
    cc.kernel()
    rec.update(total=time.perf_counter() - t0, iterations=[round(float(x), 3) for x in cc.timings], e_corr=cc.e_corr,
               nocc=int(cc.t1.shape[0]), nvir_active=int(cc.t1.shape[1]), dtype=str(cc.t2.dtype),
               t3_elements=0 if cc.t3 is None else int(np.size(cc.t3)),
               peak_gb=(cc.peak_memory or 0) / 1e9, stages=stages(snap, 3))
    if a.sigma:
        from zccsdt.cc import eom, eom_opt
        eris = cc._eris
        fly = a.what == 'cc3-fly'
        g = eom_opt.prepare(eris, cc.t1, cc.t2, cc.t3, cc._options(None, None), model=cc.model if cc.model != 'ccsdt' else 'cc',
                            **({'fly': True} if fly else {}))
        om, vm = g.om, g.vm
        rng = np.random.default_rng(0)
        m1, m2 = g.masks(0) if om.spins is not None else (1.0, 1.0)
        r1 = (rng.standard_normal((om.n, vm.n)) * m1).astype(g.dtype)
        r2 = (rng.standard_normal((om.npair, vm.npair)) * m2).astype(g.dtype)
        r3 = None if fly else rng.standard_normal(g.eng.size).astype(g.dtype)
        times = []
        for rep in range(2):
            snap = timing.snapshot()
            t0 = time.perf_counter()
            if fly:
                eom_opt.sigma(g, r1, r2, omega=0.3)
            else:
                eom_opt.sigma(g, r1, r2, r3)
            times.append(round(time.perf_counter() - t0, 3))
        rec.update(sigma=times, sigma_stages=stages(snap, 3), rss_sigma_gb=memory.resident() / 1e9)
print('RESULT ' + json.dumps(rec), flush=True)
