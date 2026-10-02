'''Time per iteration and memory of PySCF's RHF-CCSDT (pyscf.cc.rccsdt /
rccsdt_highm, in PySCF since December 2025), for the comparison with
zccsdt.sa.rccsdt (bench/profile_sa.py: same molecules, geometry constant and
frozen core).

usage: python bench/pyscf/profile_rccsdt.py MOLECULE BASIS [NITER] [NCORE]
  PS_BACKEND : einsum backend, numpy (PySCF's default) | pyscf | pytblis
               (the one PySCF recommends)
  PS_COMPACT : 1 (default) T3 stored for i <= j <= k (pyscf.cc.rccsdt);
               0 full T3 (pyscf.cc.rccsdt_highm, "if memory is sufficient")
  PS_BLK     : blksize,blksize_oovv,blksize_oooo of the compact code
               (default: PySCF's 8,4,4)
  PS_INCORE  : 1 keeps the DIIS vectors in memory (incore_complete)
  PS_CONV    : 1 runs to convergence (conv_tol 1e-10) instead of NITER
               iterations and prints the energy
  PS_LIBCCSDT: path of another build of PySCF's libccsdt.so (its C helpers;
               PySCF builds them with -O2 and without -march=native by default)
'''
import os
import sys
import threading
import time

import numpy as np
from pyscf import cc, gto, lib, scf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from profile_sa import GEOM, CFOUR_BOHR, MemWatch          # noqa: E402


def main():
    name, basis = sys.argv[1:3]
    niter = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    ncore = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    backend = os.environ.get('PS_BACKEND', 'numpy')
    compact = os.environ.get('PS_COMPACT', '1') == '1'
    conv = os.environ.get('PS_CONV', '0') == '1'
    nthreads = int(os.environ.get('OMP_NUM_THREADS', lib.num_threads()))
    lib.num_threads(nthreads)
    if backend == 'pytblis':
        import pytblis
        pytblis.set_num_threads(nthreads)
    if os.environ.get('PS_LIBCCSDT'):
        import ctypes
        from pyscf.cc import rccsdt, rccsdt_highm
        rccsdt._libccsdt = rccsdt_highm._libccsdt = ctypes.CDLL(os.environ['PS_LIBCCSDT'])
    mol = gto.M(atom=GEOM[name], basis=basis, unit=CFOUR_BOHR, verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-10)
    mycc = cc.RCCSDT(mf, frozen=ncore or None, compact_tamps=compact)
    mycc.set_einsum_backend(backend)
    mycc.incore_complete = os.environ.get('PS_INCORE', '0') == '1'
    mycc.max_cycle = 200 if conv else niter + 1
    mycc.conv_tol, mycc.conv_tol_normt = (1e-10, 1e-7) if conv else (1e-14, 1e-14)
    mycc.verbose = int(os.environ.get('PS_VERBOSE', 4))
    if compact and os.environ.get('PS_BLK'):
        mycc.blksize, mycc.blksize_oovv, mycc.blksize_oooo = (
            int(x) for x in os.environ['PS_BLK'].split(','))
    stamps = []
    mycc.callback = lambda envs: stamps.append(time.perf_counter())
    nocc = int(np.count_nonzero(mf.mo_occ > 0)) - ncore
    print('%s/%s  nocc=%d nvir=%d (spatial)  PySCF %s  %s T3, einsum backend %s, %d threads%s'
          % (name, basis, nocc, mf.mo_coeff.shape[1] - ncore - nocc, __import__('pyscf').__version__,
             'compact' if compact else 'full', backend, nthreads,
             ', libccsdt ' + os.environ['PS_LIBCCSDT'] if os.environ.get('PS_LIBCCSDT') else ''))
    watch = MemWatch()
    watch.start()
    t0, c0 = time.perf_counter(), time.process_time()
    mycc.kernel()
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    watch.stop()
    # the callback is called after the amplitude update of each iteration:
    # the time between two calls is one iteration (update, DIIS, energy)
    its = np.diff(stamps)
    print('iterations: ' + ' '.join('%.2f' % t for t in its))
    if len(its):
        print('mean iteration %.3fs' % np.mean(its[1:] if len(its) > 1 else its))
    print('total %.1fs for %d iterations (integrals and setup included), %.2f cores busy'
          % (wall, mycc.cycles, cpu / wall))
    print('E_corr = %.12f after %d iterations%s' % (mycc.e_corr, mycc.cycles,
                                                    ' (converged)' if mycc.converged else ''))
    print('peak anonymous memory %.2f GB, file-backed %.2f GB'
          % (watch.anon / 1e6, watch.file / 1e6))


if __name__ == '__main__':
    main()
