'''Per-term timing of CCSDT (or CCSD) iterations.

usage: python bench/profile_iter.py MOLECULE BASIS KIND [NITER] [FROZEN]
  MOLECULE: h2o | n2 | h2s | nh2 | o2 | c6h6     KIND: ghf | x2c | uhf
  (uhf: U-form integrals straight from a UHF reference, UCCSDT)
environment: ZC_METHOD=ccsd times CCSD iterations instead (default ccsdt);
             ZC_PACKED=1 builds the integrals block-wise with packed vvvv
             (needed for large bases); ZC_MEM_MB sets the batch buffer budget;
             ZC_SPIN=1 (ghf only) uses alpha-first orbitals and the spin-blocked
             T3 engine (UCCSDT); ZC_FLIP=1 adds closed-shell spin flip (RCCSDT);
             ZC_RU=1 (ghf only) builds R-form integrals straight from the RHF
             (eris_spin, no GHF step).
             ZC_MAXMEM: max_memory in MB for the integrals and the iterations
             (default: 80% of the available memory); ZC_PLAN: fields of an
             explicit ccsdt_opt.MemoryPlan, e.g. 't3_file=True, mem_mb=200'.
'''
import os
import sys
import threading
import time
import resource
import numpy as np
from pyscf import gto, scf, x2c

from zccsdt import eris as E, timing
from zccsdt.cc import ccsdt_opt, solver

GEOM = {
    'h2o': 'O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
    'n2': 'N 0 0 0; N 0 0 1.0977',
    'h2s': 'S 0 0 0; H 0 0.9618 0.9272; H 0 -0.9618 0.9272',
    'nh2': 'N 0 0 0; H 0 0.8036 0.6347; H 0 -0.8036 0.6347',
    'o2': 'O 0 0 0; O 0 0 1.2075',
    'c6h6': '''C 1.397 0 0; C 0.6985 1.209837 0; C -0.6985 1.209837 0; C -1.397 0 0;
               C -0.6985 -1.209837 0; C 0.6985 -1.209837 0; H 2.481 0 0;
               H 1.2405 2.148609 0; H -1.2405 2.148609 0; H -2.481 0 0;
               H -1.2405 -2.148609 0; H 1.2405 -2.148609 0''',
}
SPIN = {'nh2': 1, 'o2': 2}       # 2S of the open-shell molecules


class MemWatch(threading.Thread):
    '''Peak resident memory of this process, sampled: anonymous (the arrays
    of the program) and file-backed (mapped scratch arrays, reclaimable).'''

    def __init__(self, dt=0.1):
        super().__init__(daemon=True)
        self.dt, self.anon, self.file, self._done = dt, 0, 0, threading.Event()

    def run(self):
        while not self._done.wait(self.dt):
            with open('/proc/self/status') as f:
                kb = {l.split(':')[0]: int(l.split()[1]) for l in f if l.startswith('Rss')}
            self.anon = max(self.anon, kb.get('RssAnon', 0))
            self.file = max(self.file, kb.get('RssFile', 0))

    def stop(self):
        self._done.set()
        self.join()


def main():
    mol_name, basis, kind = sys.argv[1:4]
    niter = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    frozen = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    mol = gto.M(atom=GEOM[mol_name], basis=basis, spin=SPIN.get(mol_name, 0), verbose=0)
    ru = os.environ.get('ZC_RU', '0') == '1' and kind == 'ghf'
    if kind == 'uhf':
        mf = scf.UHF(mol).run(conv_tol=1e-10)
    elif kind == 'x2c':
        mf = x2c.UHF(mol).run(conv_tol=1e-10)
    elif ru:
        mf = scf.RHF(mol).run(conv_tol=1e-10)
    else:
        mf = scf.addons.convert_to_ghf(scf.RHF(mol).run(conv_tol=1e-10))
    t0 = time.perf_counter()
    mem = {'max_memory': float(os.environ['ZC_MAXMEM'])} if os.environ.get('ZC_MAXMEM') else {}
    watch = MemWatch()
    watch.start()
    packed = os.environ.get('ZC_PACKED', '0') == '1'
    spin = os.environ.get('ZC_SPIN', '0') == '1' and kind == 'ghf'
    if kind == 'uhf' or ru:
        from zccsdt import eris_spin
        eris = eris_spin.from_scf(mf, frozen=frozen,
                                  vir_batch=int(os.environ.get('ZC_VBATCH', 16)), **mem)
    else:
        eris = E.from_scf(mf, frozen=frozen, packed_vvvv=packed,
                          vir_batch=int(os.environ.get('ZC_VBATCH', 16)), spin_order=spin,
                          **mem)
    del mf
    t_ints = time.perf_counter() - t0
    nocc = eris.nocc
    nvir = eris.nmo - nocc
    print('%s/%s %s  nocc=%d nvir=%d dtype=%s  integrals %.1fs (peak anonymous %.2f GB)'
          % (mol_name, basis, kind, nocc, nvir, eris.dtype, t_ints, watch.anon / 1e6))
    watch.stop()
    kw = dict(verbose=True, mem_mb=float(os.environ.get('ZC_MEM_MB', 2000)),
              spin_flip=os.environ.get('ZC_FLIP', '0') == '1', **mem)
    if os.environ.get('ZC_PLAN'):
        kw['plan'] = eval('ccsdt_opt.MemoryPlan(%s)' % os.environ['ZC_PLAN'])
    watch = MemWatch()
    watch.start()
    # one warm-up iteration (engine setup, caches, buffers), then the timed
    # ones with the same memory plan: the plan of a fresh run (a second call
    # would plan again, with what the first one left allocated)
    method = os.environ.get('ZC_METHOD', 'ccsdt')
    r0 = solver.kernel(eris, method, solver.Options(max_cycle=1, **kw))
    timing.reset()
    kw['plan'] = r0.plan
    r = solver.kernel(eris, method, solver.Options(max_cycle=niter, **kw),
                      t1=r0.t1, t2=r0.t2, t3=r0.t3)
    watch.stop()
    per = len(r.timings)
    print('iterations: ' + ' '.join('%.2f' % t for t in r.timings))
    print('mean iteration %.3fs (warm-up iteration %.3fs; timers below exclude it)'
          % (np.mean(r.timings), r0.timings[0]))
    for name, sec, calls in timing.report(per):
        print('  %-14s %9.3fs  %5.1f calls' % (name, sec, calls))
    print('plan: %s' % r.plan)
    print('peak RSS %.2f GB (iterations: anonymous %.2f GB, file-backed %.2f GB)'
          % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, watch.anon / 1e6,
             watch.file / 1e6))


if __name__ == '__main__':
    main()
