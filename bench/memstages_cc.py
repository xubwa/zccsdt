'''Anonymous memory by stage of a CCSDT iteration of the spin-orbital / spinor
engine (peak resident anonymous memory inside each timer): the data behind
the memory model of zccsdt.cc.ccsdt_opt.plan_memory.

usage: python bench/memstages_cc.py MOL BASIS KIND [PLAN-FIELDS] [FROZEN] [NITER]
  KIND: ghf | x2c (spinors / spin orbitals, dense engine), uhf | rhf (U / R-form
  integrals, spin-blocked engine; rhf with the closed-shell spin flip);
  PLAN-FIELDS: fields of an explicit ccsdt_opt.MemoryPlan,
  e.g. "t3_file=True, pp_file=True, pp_bytes=64<<20, pp_order='batch', mem_mb=300"
  ZC_MAXMEM: max_memory in MB;  ZC_MEM_MB: batch buffers (default 2000);
  ZC_WARM=1: one iteration in a kernel call of its own first, as a restart
  does (the second call plans with what the first one left allocated)
'''
import os
import sys
import threading
import time

import numpy as np
from pyscf import gto, scf, x2c

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profile_iter import GEOM, SPIN                       # noqa: E402
from zccsdt import eris as E, timing                      # noqa: E402
from zccsdt.cc import ccsdt_opt, solver, t3engine         # noqa: E402

# the timers whose peaks are reported (the innermost one that is running)
STAGES = ('t1_dress', 'dress.ovvv', 'dress.vvvv_pp', 'pass1.operands', 'pass1', 'intermediates',
          'wv.Bv', 'wv.vvoo', 'wv.zcorr', 'wv.vvvv_t1', 'wv.oovo', 'wv.t2ovvv', 'wv.dWv',
          'imd.Wvooo', 'imd.ring_F', 'ccsd', 'imd.Wpp', 'imd.O6', 'pp_ladder', 'pass2',
          'hh_ladder', 't3_update', 'diis')


def anon():
    '''Resident anonymous memory of this process in GB.'''
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('RssAnon'):
                return int(line.split()[1]) / 1e6


def stage_watch(stages, modules):
    '''Start a sampler thread that books the anonymous memory under the
    innermost running timer whose name is in ``stages`` (the timers of
    zccsdt.timing and of ``modules`` are wrapped).  Returns (peak, cur,
    stop): the peaks in GB by stage, the current stage (cur[0], which the
    caller may set between the timers) and the event that ends the sampling.'''
    peak, cur, stop = {}, ['setup'], threading.Event()

    def watch():
        while not stop.wait(0.01):
            a = anon()
            if a > peak.get(cur[0], 0):
                peak[cur[0]] = a

    threading.Thread(target=watch, daemon=True).start()
    plain = timing.timer

    class Staged:
        '''timing.timer that also names the stage the sampler books under.'''

        def __init__(self, name):
            self.name, self.t = name, plain(name)

        def __enter__(self):
            self.prev = cur[0]
            if self.name in stages:
                cur[0] = self.name
            return self.t.__enter__()

        def __exit__(self, *a):
            cur[0] = self.prev
            return self.t.__exit__(*a)

    for m in (timing,) + tuple(modules):
        if hasattr(m, 'timer'):
            m.timer = Staged
    return peak, cur, stop


def main():
    peak, cur, stop = stage_watch(STAGES, (ccsdt_opt, solver, t3engine))

    name, basis, kind = sys.argv[1:4]
    plan = eval('ccsdt_opt.MemoryPlan(%s)' % sys.argv[4]) \
        if len(sys.argv) > 4 and sys.argv[4] else None
    frozen = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    niter = int(sys.argv[6]) if len(sys.argv) > 6 else 3
    mem = {'max_memory': float(os.environ['ZC_MAXMEM'])} if os.environ.get('ZC_MAXMEM') else {}
    mol = gto.M(atom=GEOM[name], basis=basis, spin=SPIN.get(name, 0), verbose=0)
    if kind == 'x2c':
        mf = x2c.UHF(mol).run(conv_tol=1e-10)
    elif kind == 'uhf':
        mf = scf.UHF(mol).run(conv_tol=1e-10)
    elif kind == 'rhf':
        mf = scf.RHF(mol).run(conv_tol=1e-10)
    else:
        mf = scf.addons.convert_to_ghf(scf.RHF(mol).run(conv_tol=1e-10))
    print('after SCF: anonymous %.2f GB' % anon())
    t0 = time.perf_counter()
    cur[0] = 'integrals'
    if kind in ('uhf', 'rhf'):
        from zccsdt import eris_spin
        eris = eris_spin.from_scf(mf, frozen=frozen, **mem)
    else:
        eris = E.from_scf(mf, frozen=frozen, packed_vvvv=True, **mem)
    del mf
    o, v = eris.nocc, eris.nmo - eris.nocc
    isz = np.dtype(eris.dtype).itemsize
    nvp = v * (v - 1) // 2
    nto, nvt = o * (o - 1) * (o - 2) // 6, v * (v - 1) * (v - 2) // 6
    print('%s/%s %s  nocc=%d nvir=%d dtype=%s  integrals %.1fs  anonymous %.2f GB'
          % (name, basis, kind, o, v, eris.dtype, time.perf_counter() - t0, anon()))
    print('sizes: o v^3 %.3f GB, packed vvvv %.3f GB, T3 %.3f GB'
          % (o * v ** 3 * isz / 1e9, nvp * nvp * isz / 1e9, nto * nvt * isz / 1e9))
    cur[0] = '(between the timers)'
    kw = dict(verbose=True, plan=plan, spin_flip=kind == 'rhf',
              mem_mb=float(os.environ.get('ZC_MEM_MB', 2000)), **mem)
    amps = {}
    if os.environ.get('ZC_WARM', '0') == '1':
        r0 = solver.kernel(eris, 'ccsdt', solver.Options(max_cycle=1, **kw))
        print('plan of the first call: %s' % r0.plan)
        print('  anonymous after it %.2f GB, peak so far %.2f GB'
              % (anon(), max(peak.values())))
        amps = dict(t1=r0.t1, t2=r0.t2, t3=r0.t3)
        del r0
    r = solver.kernel(eris, 'ccsdt', solver.Options(max_cycle=niter, **kw), **amps)
    stop.set()
    print('plan: %s' % r.plan)
    for e in set(ccsdt_opt._ENGINES.values()):
        w = getattr(e, '_work', None)       # (engines with a work array of their own)
        print('engine %s mem_mb=%.0f: work array %.2f GB'
              % (type(e).__name__, e.mem_mb, 0 if w is None else w.nbytes / 1e9))
    if hasattr(t3engine, 'workspace_bytes'):
        print('work array of the spin-blocked engine: %.2f GB' % (t3engine.workspace_bytes() / 1e9))
    print('kept on the integral object: %.2f GB' % (ccsdt_opt.cached_bytes(eris) / 1e9))
    print('iterations: ' + ' '.join('%.2f' % t for t in r.timings))
    for k, val in sorted(peak.items(), key=lambda kv: -kv[1]):
        print('  %-24s peak anonymous %.2f GB' % (k, val))
    for k, sec, calls in timing.report(len(r.timings))[:14]:
        print('  %-16s %8.3fs' % (k, sec))


if __name__ == '__main__':
    main()
