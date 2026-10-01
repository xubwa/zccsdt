'''Per-term timing of the closed-shell spin-adapted CCSDT (zccsdt.sa.rccsdt).

usage: python bench/profile_sa.py MOLECULE BASIS [NITER] [NCORE]
  MOLECULE: h2o | n2 | h2s     NCORE: frozen core (spatial orbitals)
'''
import os
import sys
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
}


def main():
    name, basis = sys.argv[1:3]
    niter = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    ncore = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    mol = gto.M(atom=GEOM[name], basis=basis, verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-10)
    t0 = time.perf_counter()
    ints = rccsdt_ref.from_scf(mf, ncore=ncore, full=False)
    print('%s/%s  nocc=%d nvir=%d (spatial)  integrals %.1fs'
          % (name, basis, ints.nocc, ints.nvir, time.perf_counter() - t0))
    timing.reset()
    r = rccsdt.kernel(ints, max_cycle=niter + 1, verbose=True)
    timing.reset()
    # time a further run of niter iterations without the warm-up costs
    r = rccsdt.kernel(ints, max_cycle=niter, verbose=True)
    print('iterations: ' + ' '.join('%.2f' % t for t in r.timings))
    print('mean iteration %.3fs' % np.mean(r.timings[1:] or r.timings))
    for k, sec, calls in timing.report(len(r.timings)):
        print('  %-14s %9.3fs  %5.1f calls' % (k, sec, calls))
    print('peak RSS %.2f GB' % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6))


if __name__ == '__main__':
    main()
