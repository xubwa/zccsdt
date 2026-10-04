'''The Lambda equations of zccsdt.zccsd.DirectZCCSD after a converged CCSD:
wall times of the intermediates and of the iterations with their timers, for
the comparison with CFOUR's xx2chbar and xx2clcc (bench/cfour/slurm_x2ccc.sh
with CFOUR_PROGS up to xx2clcc).

usage: python bench/profile_lambda.py MOLECULE BASIS
  MOLECULE: h2o | n2 | c6h6 | ... (the geometries of bench/profile_iter.py)
X2C-1e spinors from PySCF (x2c.UHF), all electrons.  CCSD and Lambda stop
when the norm of the change of the amplitudes is below ZC_CONV (default
1e-7; CFOUR's CC_CONV=7 asks the largest change to be below 1e-7).
environment: ZC_MAXMEM: max_memory in MB (default 24000); ZC_KRAMERS: 0 or 1:
             DirectZCCSD.kramers; ZC_THREE_M: 0 or 1: DirectZCCSD.three_m;
             ZC_CONV; ZC_AO_LADDER: DirectZCCSD.ao_ladder;
             ZC_CCSD_CYCLES, ZC_LAMBDA_CYCLES: at most so many cycles (for
             the time and the memory of an iteration of a large molecule:
             Lambda then runs on amplitudes that are not converged).
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
from zccsdt.zccsd import DirectZCCSD, amps, lam                         # noqa: E402
from zccsdt.zccsd import kernels as K                                   # noqa: E402
from zccsdt.cc.diis import DIIS                                         # noqa: E402


def main():
    mol_name, basis = sys.argv[1:3]
    maxmem = float(os.environ.get('ZC_MAXMEM', 24000))
    conv = float(os.environ.get('ZC_CONV', 1e-7))
    mol = gto.M(atom=GEOM[mol_name], basis=basis, spin=SPIN.get(mol_name, 0), verbose=0,
                max_memory=maxmem)
    mf = x2c.UHF(mol).run(conv_tol=1e-10)
    mycc = DirectZCCSD(mf)
    mycc.verbose, mycc.max_cycle = 0, int(os.environ.get('ZC_CCSD_CYCLES', 100))
    mycc.conv_tol, mycc.conv_tol_normt = 0.01 * conv, conv
    if 'ZC_KRAMERS' in os.environ:
        mycc.kramers = bool(int(os.environ['ZC_KRAMERS']))
    if 'ZC_THREE_M' in os.environ:
        mycc.three_m = bool(int(os.environ['ZC_THREE_M']))
    if os.environ.get('ZC_AO_LADDER'):
        mycc.ao_ladder = os.environ['ZC_AO_LADDER']
    watch = MemWatch()
    watch.start()
    t0 = time.perf_counter()
    eris = mycc.ao2mo()
    t1 = time.perf_counter()
    cycles = []
    mycc.callback = lambda envs: cycles.append(time.perf_counter())
    mycc.kernel(eris=eris)
    t2 = time.perf_counter()
    res = eris.residual
    o, v = res.nocc, res.nvir
    peak_ccsd = watch.anon
    watch.stop()
    # as lam.kernel: t2 packed, the unpacked array released, then the plan of the
    # Lambda iterations - the budget less what is theirs
    t2p = K.pack_t2(np.ascontiguousarray(mycc.t2))
    t1a = np.ascontiguousarray(mycc.t1)
    mycc.t2 = None
    res.replan(mycc.max_memory, mycc.diis_space, None, mycc.three_m,
               reserve=lam.reserve_bytes(o, v))
    print('%s/%s DirectZCCSD  nocc=%d nvir=%d  Kramers pairs: %s  max_memory %.0f MB  threads %s'
          % (mol_name, basis, o, v, bool(eris.kramers), maxmem,
             os.environ.get('OMP_NUM_THREADS', '?')))
    print('matrix products: ' + blas.backend()[1])
    print('plan of the Lambda iterations: %s; Lambda working set %.2f GB'
          % (res.describe(), lam.working_set(res.plan, o, v) / 1e9))
    print('integrals %.1f s; CCSD %.1f s: %d cycles (%.2f s each), converged %s, E_corr %.10f'
          % (t1 - t0, t2 - t1, len(cycles), (t2 - t1) / max(len(cycles), 1), mycc.converged,
             mycc.e_corr))

    # the iterations of lam.kernel, timed one by one
    watch = MemWatch()
    watch.start()
    timing.reset()
    t3 = time.perf_counter()
    L = lam.Lambda(res, t1a, t2p)
    t4 = time.perf_counter()
    l1, l2p = t1a.conj(), K.copy(t2p, conj=True)
    adiis = DIIS(space=mycc.diis_space, filename=mycc.diis_file, incore=res.plan.diis_incore)
    times, snaps, norm = [], [], 1.0
    for it in range(int(os.environ.get('ZC_LAMBDA_CYCLES', 100))):
        snaps.append(timing.snapshot())
        ta = time.perf_counter()
        with timing.timer('update_lambda'):
            l1n, l2n = L.update_packed(l1, l2p, mycc.level_shift)
        with timing.timer('lambda.diis'):
            norm = np.sqrt(K.diff_norm2(l1n, l1) + K.diff_norm2(l2n, l2p))
            x1, x2 = adiis.update([l1n, l2n])
            l1, l2p = x1.reshape(o, v), x2.reshape(l2n.shape)
            del l1n, l2n
        times.append(time.perf_counter() - ta)
        if norm < conv:
            break
    snaps.append(timing.snapshot())
    t5 = time.perf_counter()
    adiis.close()
    watch.stop()
    n = len(times)
    print('Lambda: intermediates %.2f s; %d iterations in %.1f s, converged %s (norm %.1e); '
          'intermediates and iterations %.1f s'
          % (t4 - t3, n, t5 - t4, norm < conv, norm, t5 - t3))
    print('Lambda iterations: ' + ' '.join('%.2f' % t for t in times[:6]) + ' ... s')
    if n > 1:
        per = n - 1
        steps = timing.since(snaps[1], snaps[-1])
        mean = sum(times[1:]) / per
        print('mean iteration %.3f s (iterations 2-%d)' % (mean, n))

        def show(parent, indent):
            for name, w, c, calls in timing.children(steps, parent):
                if w / per < 0.002 * mean:
                    continue
                print('%s%-*s %9.3f s  %5.1f %%  (%g calls)' % (
                    indent, 32 - len(indent), name, w / per, 100 * w / per / mean, calls / per))
                show(parent + (name,), indent + '  ')
        show((), '  ')
    print('|lambda1 - conj t1| %.2e  |lambda2 - conj t2| %.2e'
          % (abs(l1 - t1a.conj()).max(), abs(l2p - t2p.conj()).max()))
    print('peak anonymous memory: integrals and CCSD %.2f GB, Lambda %.2f GB; peak RSS %.2f GB'
          % (peak_ccsd / 1e6, watch.anon / 1e6,
             resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6))
    L.close()


if __name__ == '__main__':
    main()
