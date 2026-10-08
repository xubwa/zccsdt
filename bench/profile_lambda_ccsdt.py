'''Time per iteration of the CCSDT amplitude equations and of its Lambda
equations (zccsdt.cc.lambda_opt) in one process, with the stages of each.

usage: python bench/profile_lambda_ccsdt.py MOLECULE BASIS KIND [NITER] [FROZEN]
  MOLECULE: h2o | hbr | n2       KIND: ghf (spin orbitals of RHF, real) | x2c (spinors, complex)
environment: ZC_MEM_MB the batch buffer budget (default 2000); ZC_MAXMEM
             max_memory in MB (default: 80% of the available memory).

NITER amplitude iterations from the MP2 guess after a warm-up iteration, then
the intermediates of the Lambda equations and NITER Lambda iterations after a
warm-up iteration, from conj(t) of the amplitudes reached: the cost of an
iteration does not depend on the amplitudes being converged.  For every stage:
wall seconds per iteration and CPU / wall, the number of cores it keeps busy.
'''
import os
import resource
import sys
import time

import numpy as np
from pyscf import gto, scf, x2c

from zccsdt import eris as E, timing
from zccsdt.cc import ccsdt_opt, lambda_opt, solver

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profile_iter import MemWatch          # noqa: E402

GEOM = {
    'h2o': 'O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
    'hbr': 'Br 0 0 0; H 0 0 1.4145',
    'n2': 'N 0 0 0; N 0 0 1.0977',
}


def tree(times, per, parent=(), indent=1, floor=0.005):
    for name, w, c, n in timing.children(times, parent):
        if w / per < floor:
            continue
        print('%s%-*s %9.3f s  %5.1f cores  %6.1f calls'
              % ('  ' * indent, 26 - 2 * indent, name, w / per, c / w if w > 0 else 0.0, n / per))
        tree(times, per, tuple(parent) + (name,), indent + 1, floor)


def main():
    mol_name, basis, kind = sys.argv[1:4]
    niter = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    frozen = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    mol = gto.M(atom=GEOM[mol_name], basis=basis, verbose=0)
    if kind == 'x2c':
        mf = x2c.UHF(mol).run(conv_tol=1e-10)
    else:
        mf = scf.addons.convert_to_ghf(scf.RHF(mol).run(conv_tol=1e-10))
    mem = {'max_memory': float(os.environ['ZC_MAXMEM'])} if os.environ.get('ZC_MAXMEM') else {}
    t0 = time.perf_counter()
    eris = E.from_scf(mf, frozen=frozen, packed_vvvv=True, **mem)
    del mf
    nocc, nvir = eris.nocc, eris.nmo - eris.nocc
    print('%s/%s %s  nocc=%d nvir=%d dtype=%s  integrals %.1fs  threads %s'
          % (mol_name, basis, kind, nocc, nvir, eris.dtype, time.perf_counter() - t0,
             os.environ.get('OMP_NUM_THREADS')), flush=True)
    kw = dict(verbose=True, mem_mb=float(os.environ.get('ZC_MEM_MB', 2000)), **mem)

    # ---- amplitude iterations
    watch = MemWatch()
    watch.start()
    r0 = solver.kernel(eris, 'ccsdt', solver.Options(max_cycle=1, **kw))
    timing.reset()
    r = solver.kernel(eris, 'ccsdt', solver.Options(max_cycle=niter, plan=r0.plan, **kw),
                      t1=r0.t1, t2=r0.t2, t3=r0.t3)
    watch.stop()
    t_amp = float(np.mean(r.timings))
    print('amplitude iterations: ' + ' '.join('%.2f' % t for t in r.timings))
    print('mean amplitude iteration %.3f s (warm-up %.3f s)   peak anonymous %.2f GB   plan %s'
          % (t_amp, r0.timings[0], watch.anon / 1e6, r.plan))
    tree(timing.snapshot(), len(r.timings))
    sys.stdout.flush()
    t1, t2, t3 = r.t1, r.t2, np.array(r.t3)
    del r, r0
    ccsdt_opt.release_cached(eris)

    # ---- Lambda iterations
    from zccsdt import memory
    base = memory.resident()            # the integrals and T3
    watch = MemWatch()
    watch.start()
    timing.reset()
    l0 = lambda_opt.kernel(eris, t1, t2, t3, solver.Options(max_cycle=1, **kw))
    setup = timing.snapshot()
    print('Lambda: intermediates %.3f s' % l0.setup)
    tree(setup, 1, ('lam.setup',))
    timing.reset()
    lr = lambda_opt.kernel(eris, t1, t2, t3, solver.Options(max_cycle=niter, plan=l0.plan, **kw),
                           l0.l1, l0.l2, l0.l3)
    watch.stop()
    t_lam = float(np.mean(lr.timings))
    print('Lambda iterations: ' + ' '.join('%.2f' % t for t in lr.timings))
    print('mean Lambda iteration %.3f s (warm-up %.3f s)   ratio to the amplitude iteration %.2f'
          % (t_lam, l0.timings[0], t_lam / t_amp))
    times = {p: v for p, v in timing.snapshot().items() if p[0] != 'lam.setup'}
    tree(times, len(lr.timings))
    plan = lr.plan
    print('plan: need %.2f GB (setup %.2f, iteration %.2f, buffers mem_mb %.0f)'
          % (plan.need / 1e9, plan.setup / 1e9, plan.iteration / 1e9, plan.mem_mb))
    t3_bytes = t3.nbytes
    print('Lambda: peak anonymous %.2f GB, %.2f GB above the %.2f GB before it (integrals, T3); '
          'the plan without T3: %.2f GB;  file-backed %.2f GB;  peak RSS of the process %.2f GB'
          % (watch.anon / 1e6, watch.anon / 1e6 - base / 1e9, base / 1e9,
             (plan.need - t3_bytes) / 1e9, watch.file / 1e6,
             resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6))


if __name__ == '__main__':
    main()
