'''Per-term timing of the closed-shell spin-adapted CCSDT (zccsdt.sa.rccsdt).

usage: python bench/profile_sa.py MOLECULE BASIS [NITER] [NCORE]
  MOLECULE: h2o | n2 | h2s | ch2o | c2h4 | c6h6     NCORE: frozen core (spatial orbitals)
  ZC_INCORE=1: all DIIS vectors in memory (default: T3 vectors in the scratch
  file, as in PySCF);  ZC_DIIS_SPACE: DIIS subspace size (default 8)
'''
import os
import sys
import threading
import time
import resource

import numpy as np
from pyscf import gto, scf

from zccsdt import timing
from zccsdt.sa import rccsdt, rccsdt_ref

GEOM = {
    'h2o': 'O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
    'n2': 'N 0 0 0; N 0 0 1.0977',
    'h2s': 'S 0 0 0; H 0 0.9618 0.9272; H 0 -0.9618 0.9272',
    'ch2o': 'C 0 0 0; O 0 0 1.205; H 0 0.9447 -0.5847; H 0 -0.9447 -0.5847',
    'c2h4': 'C 0.000000 0.000000 0.669500; C 0.000000 0.000000 -0.669500; H 0.000000 0.929781 1.232595; H 0.000000 -0.929781 1.232595; H 0.000000 0.929781 -1.232595; H 0.000000 -0.929781 -1.232595',
    'c6h6': 'C 1.397000 0.000000 0.000000; C 0.698500 1.209837 0.000000; C -0.698500 1.209837 0.000000; C -1.397000 0.000000 0.000000; C -0.698500 -1.209837 0.000000; C 0.698500 -1.209837 0.000000; H 2.481000 0.000000 0.000000; H 1.240500 2.148609 0.000000; H -1.240500 2.148609 0.000000; H -2.481000 0.000000 0.000000; H -1.240500 -2.148609 0.000000; H 1.240500 -2.148609 0.000000',
}

# Angstrom per bohr of CFOUR (PySCF: 0.52917721092).  With it the geometry is
# the one CFOUR builds from the same Cartesian input, and the energies of
# every iteration can be compared with its output (the two constants shift
# the geometry by 2e-7 and the H2O/cc-pVTZ SCF energy by 1.4e-8 Eh).
CFOUR_BOHR = 0.5291771


class MemWatch(threading.Thread):
    '''Peak resident memory of this process, sampled: anonymous (the arrays
    of the program) and file-backed (mapped DIIS vectors, reclaimable).'''

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
    name, basis = sys.argv[1:3]
    niter = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    ncore = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    mol = gto.M(atom=GEOM[name], basis=basis, unit=CFOUR_BOHR, verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-10)
    t0 = time.perf_counter()
    ints = rccsdt_ref.from_scf(mf, ncore=ncore, full=False)
    print('%s/%s  nocc=%d nvir=%d (spatial)  integrals %.1fs'
          % (name, basis, ints.nocc, ints.nvir, time.perf_counter() - t0))
    opts = dict(incore_complete=os.environ.get('ZC_INCORE', '0') == '1',
                diis_space=int(os.environ.get('ZC_DIIS_SPACE', '8')), verbose=True)
    watch = MemWatch()
    watch.start()
    timing.reset()
    r = rccsdt.kernel(ints, max_cycle=niter + 1, **opts)
    del r                       # its T3 must not count in the memory of the timed run
    timing.reset()
    # time a further run of niter iterations without the warm-up costs
    r = rccsdt.kernel(ints, max_cycle=niter, **opts)
    watch.stop()
    print('iterations: ' + ' '.join('%.2f' % t for t in r.timings))
    print('mean iteration %.3fs' % np.mean(r.timings[1:] or r.timings))
    if len(r.timings) > opts['diis_space'] + 2:
        print('mean iteration with a full DIIS subspace %.3fs'
              % np.mean(r.timings[opts['diis_space'] + 1:]))
    for k, sec, calls in timing.report(len(r.timings)):
        print('  %-14s %9.3fs  %5.1f calls' % (k, sec, calls))
    print('DIIS vectors: %s' % ('T3 part in the scratch file' if r.diis_on_disk else 'in memory'))
    print('peak RSS %.2f GB (anonymous %.2f GB, file-backed %.2f GB)'
          % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, watch.anon / 1e6,
             watch.file / 1e6))


if __name__ == '__main__':
    main()
