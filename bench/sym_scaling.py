'''CCSDT iterations and (Q) of a linear molecule on X2C spinors through the
front end, in the blocks of the symmetry labels and dense, in one process on
the same spinors: the time of an iteration (the first one, with its setup,
apart), of (Q), the cores busy and the peak memory of each.  Run it once per
number of threads (OMP_NUM_THREADS) for the scaling.

usage: python bench/sym_scaling.py GEOM BASIS FROZEN [NITER] [MODES]
       e.g.  python bench/sym_scaling.py 'Br 0 0 0; H 0 0 1.4145' cc-pvqz-dk 10 4 dense,blocks
'''
import os
import sys
import time

import numpy as np
from pyscf import gto, x2c

import zccsdt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profile_sa import MemWatch                                         # noqa: E402

geom, basis, frozen = sys.argv[1], sys.argv[2], int(sys.argv[3])
niter = int(sys.argv[4]) if len(sys.argv) > 4 else 4
modes = (sys.argv[5] if len(sys.argv) > 5 else 'dense,blocks').split(',')
mol = gto.M(atom=geom, basis=basis, verbose=0)
mf = x2c.UHF(mol)
mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-8
mf.kernel()
print('%s / %s: E(SCF) = %.10f, %d threads' % (geom, basis, mf.e_tot, int(os.environ.get('OMP_NUM_THREADS', 0))),
      flush=True)
out = {}
for mode in modes:
    cc = zccsdt.CCSDT(mf, frozen=frozen)
    cc.verbose, cc.symmetry, cc.max_cycle = 3, mode == 'blocks', niter
    cc.conv_tol, cc.conv_tol_normt = 1e-12, 1e-10
    marks = [(time.perf_counter(), time.process_time())]
    cc.callback = lambda info: marks.append((time.perf_counter(), time.process_time()))
    watch = MemWatch()
    watch.start()
    cc.kernel()
    watch.stop()
    wall = np.diff([m[0] for m in marks])
    cpu = np.diff([m[1] for m in marks])
    eris = cc._eris
    no, nv = eris.nocc, eris.nmo - eris.nocc
    rest = slice(1, None) if len(wall) > 1 else slice(0, None)
    print('== %s: %d + %d spinors, %s, T3 %d elements (%.2f GB)' % (mode, no, nv, np.asarray(cc.t2).dtype,
                                                                     np.size(cc.t3), np.asarray(cc.t3).nbytes / 1e9))
    print('   CCSDT iterations (s): %s' % ' '.join('%.2f' % w for w in wall))
    print('   iteration after the first: %.2f s, %.1f cores busy; peak anonymous memory %.2f GB; E_corr after %d: %.12f'
          % (wall[rest].mean(), cpu[rest].sum() / wall[rest].sum(), watch.anon / 1e6, len(wall), cc.e_corr), flush=True)
    watch = MemWatch()
    watch.start()
    t0, c0 = time.perf_counter(), time.process_time()
    eq = cc.ccsdt_q()
    tq, cq = time.perf_counter() - t0, time.process_time() - c0
    watch.stop()
    print('   (Q): %.2f s, %.1f cores busy; peak anonymous memory %.2f GB; E[Q] = %.12f  E(Q) = %.12f'
          % (tq, cq / tq, watch.anon / 1e6, cc.e_q5, eq), flush=True)
    out[mode] = (wall[rest].mean(), tq, cc.e_corr, cc.e_q5, eq)
    del cc
if len(out) == 2:
    d, b = out['dense'], out['blocks']
    print('dense / blocks: iteration %.2f, (Q) %.2f;  blocks - dense: E_corr %.1e  E[Q] %.1e  E(Q) %.1e'
          % (d[0] / b[0], d[1] / b[1], b[2] - d[2], b[3] - d[3], b[4] - d[4]))
