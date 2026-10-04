'''(T) of zccsdt.zccsd.DirectZCCSD with the terms of the Lambda equations and
of the one-particle density next to the (T) energy alone: wall times with
their timers and peak memory of the two runs of zccsd.triples.kernel, one
after the other on the amplitudes of the same CCSD.

usage: python bench/profile_t_lambda.py MOLECULE BASIS
  MOLECULE: h2o | n2 | c6h6 | ... (the geometries of bench/profile_iter.py)
X2C-1e spinors from PySCF (x2c.UHF), all electrons.
environment: ZC_MAXMEM: max_memory in MB (default 24000); ZC_KRAMERS: 0 or 1:
             DirectZCCSD.kramers; ZC_THREE_M: 0 or 1: three_m of (T);
             ZC_CCSD_CYCLES: at most so many CCSD cycles (the times of (T) do
             not depend on the convergence of the amplitudes; default 100);
             ZC_T_BATCH_GB: triples.BATCH_BYTES in GB.
'''
import os
import resource
import sys
import time

import numpy as np
from pyscf import gto, x2c

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profile_iter import GEOM, SPIN, MemWatch                           # noqa: E402

from zccsdt import blas, timing                                         # noqa: E402
from zccsdt.zccsd import DirectZCCSD, triples                           # noqa: E402
from zccsdt.zccsd import kernels as K                                   # noqa: E402


def main():
    mol_name, basis = sys.argv[1:3]
    maxmem = float(os.environ.get('ZC_MAXMEM', 24000))
    mol = gto.M(atom=GEOM[mol_name], basis=basis, spin=SPIN.get(mol_name, 0), verbose=0,
                max_memory=maxmem)
    mf = x2c.UHF(mol).run(conv_tol=1e-10)
    mycc = DirectZCCSD(mf)
    mycc.verbose, mycc.max_cycle = 0, int(os.environ.get('ZC_CCSD_CYCLES', 100))
    mycc.conv_tol, mycc.conv_tol_normt = 1e-9, 1e-7
    if 'ZC_KRAMERS' in os.environ:
        mycc.kramers = bool(int(os.environ['ZC_KRAMERS']))
    three_m = bool(int(os.environ['ZC_THREE_M'])) if 'ZC_THREE_M' in os.environ else None
    if 'ZC_T_BATCH_GB' in os.environ:
        triples.BATCH_BYTES = int(float(os.environ['ZC_T_BATCH_GB']) * 2 ** 30)
    t0 = time.perf_counter()
    eris = mycc.ao2mo()
    mycc.kernel(eris=eris)
    t1 = time.perf_counter()
    o, v = mycc.t1.shape
    # as DirectZCCSD.solve_lambda: t2 packed, the layouts of the CCSD step and the
    # AO operators of its ladder released
    t2p = K.pack_t2(np.ascontiguousarray(mycc.t2))
    t1a = np.ascontiguousarray(mycc.t1)
    mycc.t2 = None
    if getattr(eris, 'residual', None) is not None:
        eris.residual.close()
        eris.residual = None
    if getattr(eris, 'aoladder', None) is not None:
        eris.aoladder.close()
    print('%s/%s DirectZCCSD  nocc=%d nvir=%d  Kramers pairs: %s  max_memory %.0f MB  threads %s'
          % (mol_name, basis, o, v, bool(eris.kramers), maxmem,
             os.environ.get('OMP_NUM_THREADS', '?')))
    print('matrix products: ' + blas.backend()[1])
    print('integrals and CCSD %.1f s, converged %s, E_corr %.10f; packed T2 %.2f GB'
          % (t1 - t0, mycc.converged, mycc.e_corr, 16e-9 * K.npair(o) * K.npair(v)))
    mycc.verbose = 4
    wall = {}
    for label, terms in (('(T) energy', None), ('(T) energy and the terms of Lambda', True)):
        watch = MemWatch()
        watch.start()
        timing.reset()
        ta = time.perf_counter()
        out = triples.kernel(mycc, eris, t1a, t2p, three_m=three_m, terms=terms)
        wall[terms] = time.perf_counter() - ta
        watch.stop()
        e_t = out if terms is None else out[0]
        print('%s: %.1f s, E(T) %.10f, peak anonymous memory %.2f GB'
              % (label, wall[terms], e_t, watch.anon / 1e6))
        for name, w, calls in timing.report():
            if name.startswith('t.'):
                print('  %-12s %9.1f s  %5.1f %%  (%d calls)'
                      % (name[2:], w, 100 * w / wall[terms], calls))
        if terms:
            l1, l2p, goo, gvv, dvo = out[1]
            print('  largest elements: l1 %.3e  l2 %.3e  goo %.3e  gvv %.3e  dvo %.3e'
                  % tuple(abs(x).max() for x in (l1, l2p, goo, gvv, dvo)))
            del out, l1, l2p
    print('with the terms of Lambda: %.2f times the (T) energy' % (wall[True] / wall[None]))
    print('peak RSS %.2f GB' % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6))


if __name__ == '__main__':
    main()
