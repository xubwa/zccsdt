'''Anonymous memory by stage of a CCSDT iteration of the spin-orbital / spinor
engine (peak resident anonymous memory inside each timer): the data behind
the memory model of zccsdt.cc.ccsdt_opt.plan_memory.

usage: python bench/memstages_cc.py MOL BASIS KIND [PLAN-FIELDS] [FROZEN] [NITER]
  KIND: ghf | x2c;  PLAN-FIELDS: fields of an explicit ccsdt_opt.MemoryPlan,
  e.g. "t3_file=True, pp_file=True, pp_bytes=64<<20, pp_order='batch', mem_mb=300"
  ZC_MAXMEM: max_memory in MB;  ZC_MEM_MB: batch buffers (default 2000)
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
STAGES = ('t1_dress', 'dress.vvvv_pp', 'pass1.operands', 'pass1', 'intermediates', 'wv.Bv',
          'wv.vvoo', 'wv.vvvv_t1', 'wv.oovo', 'wv.t2ovvv', 'wv.dWv', 'imd.Wvooo', 'imd.ring_F',
          'ccsd', 'imd.Wpp', 'imd.O6', 'pp_ladder', 'pass2', 'hh_ladder', 't3_update', 'diis')


def anon():
    '''Resident anonymous memory of this process in GB.'''
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('RssAnon'):
                return int(line.split()[1]) / 1e6


def main():
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
            if self.name in STAGES:
                cur[0] = self.name
            return self.t.__enter__()

        def __exit__(self, *a):
            cur[0] = self.prev
            return self.t.__exit__(*a)

    for m in (timing, ccsdt_opt, solver, t3engine):
        m.timer = Staged

    name, basis, kind = sys.argv[1:4]
    plan = eval('ccsdt_opt.MemoryPlan(%s)' % sys.argv[4]) \
        if len(sys.argv) > 4 and sys.argv[4] else None
    frozen = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    niter = int(sys.argv[6]) if len(sys.argv) > 6 else 3
    mem = {'max_memory': float(os.environ['ZC_MAXMEM'])} if os.environ.get('ZC_MAXMEM') else {}
    mol = gto.M(atom=GEOM[name], basis=basis, spin=SPIN.get(name, 0), verbose=0)
    if kind == 'x2c':
        mf = x2c.UHF(mol).run(conv_tol=1e-10)
    else:
        mf = scf.addons.convert_to_ghf(scf.RHF(mol).run(conv_tol=1e-10))
    print('after SCF: anonymous %.2f GB' % anon())
    t0 = time.perf_counter()
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
    opts = solver.Options(max_cycle=niter, verbose=True, plan=plan,
                          mem_mb=float(os.environ.get('ZC_MEM_MB', 2000)), **mem)
    r = solver.kernel(eris, 'ccsdt', opts)
    stop.set()
    print('plan: %s' % r.plan)
    print('iterations: ' + ' '.join('%.2f' % t for t in r.timings))
    for k, val in sorted(peak.items(), key=lambda kv: -kv[1]):
        print('  %-24s peak anonymous %.2f GB' % (k, val))
    for k, sec, calls in timing.report(len(r.timings))[:14]:
        print('  %-16s %8.3fs' % (k, sec))


if __name__ == '__main__':
    main()
