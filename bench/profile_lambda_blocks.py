'''The CCSDT amplitude equations and the Lambda equations on the packed block
and in the blocks of the symmetry labels, in one process on the same
integrals and amplitudes: time per iteration, cores busy, peak memory, and
the agreement of the two Lambda.

usage: python bench/profile_lambda_blocks.py MOLECULE BASIS [NITER] [FROZEN]
  MOLECULE: hbr | n2 (linear, on the z axis: X2C spinors adapted to m_j, the
  labels 2 m_j of the front end, real integrals)
environment: ZC_MEM_MB the batch buffer budget (default 2000); ZC_MAXMEM
             max_memory in MB (default: 80% of the available memory).

The amplitudes are converged in the blocks first (the front end).  Then, the
packed block before the blocks each time: NITER amplitude iterations from
the converged amplitudes after a warm-up iteration, and the Lambda
equations to convergence from conj(t).  For every stage of an iteration:
wall seconds per iteration and CPU / wall, the number of cores it keeps
busy.  Peak memory: the anonymous resident memory of the process, sampled,
above what was resident before the phase.
'''
import copy
import os
import sys
import time

import numpy as np
from pyscf import gto, x2c

import zccsdt
from zccsdt import memory, timing
from zccsdt.cc import ccsdt_opt, lambda_opt, solver

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profile_iter import MemWatch          # noqa: E402
from profile_lambda_ccsdt import tree       # noqa: E402

GEOM = {'hbr': 'Br 0 0 0; H 0 0 1.4145', 'n2': 'N 0 0 0; N 0 0 1.0977'}


class Phase:
    '''Wall and CPU time and the peak of the anonymous memory of a phase.'''

    def __enter__(self):
        self.base = memory.resident()
        self.watch = MemWatch()
        self.watch.start()
        timing.reset()
        self.w, self.c = time.perf_counter(), time.process_time()
        return self

    def __exit__(self, *exc):
        self.w, self.c = time.perf_counter() - self.w, time.process_time() - self.c
        self.watch.stop()
        self.peak = self.watch.anon * 1024
        self.times = timing.snapshot()


def iterations(label, run):
    '''A warm-up iteration and the timed ones; returns the mean time and the
    cores busy of the timed iterations.'''
    stat = []
    r0 = run(1, None, lambda info: None)
    with Phase() as ph:
        r = run(None, r0, lambda info: stat.append((info['wall'], info['cpu'])))
    w, c = np.array(stat).T
    print('%s: %s' % (label, ' '.join('%.2f' % t for t in w)))
    print('  mean %.3f s per iteration, %.1f cores busy (warm-up %.3f s);  peak anonymous %.2f GB, '
          '%.2f GB above the %.2f GB before it'
          % (w.mean(), c.sum() / w.sum(), r0.timings[0], ph.peak / 1e9, (ph.peak - ph.base) / 1e9,
             ph.base / 1e9))
    return r, ph, float(w.mean()), float(c.sum() / w.sum())


def main():
    mol_name, basis = sys.argv[1:3]
    niter = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    frozen = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    mol = gto.M(atom=GEOM[mol_name], basis=basis, verbose=0)
    mf = x2c.UHF(mol)
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-8
    mf.kernel()
    mem = {'max_memory': float(os.environ['ZC_MAXMEM'])} if os.environ.get('ZC_MAXMEM') else {}
    cc = zccsdt.GCCSDT(mf, frozen=frozen)
    cc.verbose = 3
    cc.conv_tol, cc.conv_tol_normt = 1e-10, 1e-9
    if mem:
        cc.max_memory = mem['max_memory']
    t0 = time.perf_counter()
    cc.kernel()
    eris = cc._eris
    nocc, nvir = eris.nocc, eris.nmo - eris.nocc
    labels = np.asarray(eris.labels)
    print('%s/%s X2C  nocc=%d nvir=%d dtype=%s  labels of the occupied %s, of the virtual %s  '
          'integrals and CCSDT %.1fs (converged %s, %d cycles)  threads %s'
          % (mol_name, basis, nocc, nvir, eris.dtype, np.unique(labels[:nocc]),
             np.unique(labels[nocc:]), time.perf_counter() - t0, cc.converged,
             getattr(cc, 'cycles', -1), os.environ.get('OMP_NUM_THREADS')), flush=True)
    t1, t2, t3b = cc.t1, cc.t2, np.array(cc.t3)
    del cc, mf
    dense = copy.copy(eris)
    dense.use_labels = False
    dense.__dict__.pop('_buffers', None)
    t3d = ccsdt_opt.packed_t3(eris, t3b)
    print('T3: %d elements in the blocks, %d on the packed block (%.1f%%)'
          % (t3b.size, t3d.size, 100.0 * t3b.size / t3d.size), flush=True)
    kw = dict(verbose=False, mem_mb=float(os.environ.get('ZC_MEM_MB', 2000)), **mem)
    out = {}

    # ---- amplitude iterations
    for name, e, t3 in (('packed block', dense, t3d), ('blocks', eris, t3b)):
        ccsdt_opt.release_cached(eris)
        ccsdt_opt.release_cached(dense)

        def run(n, r0, cb, e=e, t3=t3):
            o = solver.Options(max_cycle=n or niter, conv_tol=1e-30, conv_tol_normt=1e-30,
                               plan=r0 and r0.plan, callback=cb, **kw)
            return solver.kernel(e, 'ccsdt', o, t1=t1, t2=t2, t3=t3)
        r, ph, w, cores = iterations('amplitude iterations, ' + name, run)
        tree(ph.times, len(r.timings))
        out['amp', name] = (w, cores, ph.peak, ph.base)
        del r
        sys.stdout.flush()
    ccsdt_opt.release_cached(eris)
    ccsdt_opt.release_cached(dense)

    # ---- Lambda equations
    lam = {}
    for name, e, t3, blocks in (('packed block', dense, t3d, False), ('blocks', eris, t3b, True)):
        stat = []
        with Phase() as ph:
            r = lambda_opt.kernel(e, t1, t2, t3, solver.Options(
                conv_tol_normt=1e-9, callback=lambda info: stat.append((info['wall'], info['cpu'])),
                **kw), blocks=blocks)
        w, c = np.array(stat[1:]).T             # (without the first iteration)
        plan = r.plan
        print('Lambda, %s: converged %s in %d cycles; intermediates %.3f s; iterations %s'
              % (name, r.converged, r.ncycle, r.setup, ' '.join('%.2f' % t for t in r.timings)))
        print('  mean %.3f s per iteration, %.1f cores busy (first %.3f s);  peak anonymous %.2f GB, '
              '%.2f GB above the %.2f GB before it'
              % (w.mean(), c.sum() / w.sum(), r.timings[0], ph.peak / 1e9, (ph.peak - ph.base) / 1e9,
                 ph.base / 1e9))
        print('  plan: need %.2f GB, without T3 %.2f GB (setup %.2f, iteration %.2f, buffers %.2f, '
              'DIIS in memory %.2f, mem_mb %.0f)'
              % (plan.need / 1e9, (plan.need - t3.nbytes) / 1e9, plan.setup / 1e9,
                 plan.iteration / 1e9, plan.buffers / 1e9, plan.diis / 1e9, plan.mem_mb))
        print('  intermediates:')
        tree(ph.times, 1, ('lam.setup',))
        print('  an iteration:')
        tree({p: v for p, v in ph.times.items() if p[0] != 'lam.setup'}, r.ncycle)
        out['lam', name] = (float(w.mean()), float(c.sum() / w.sum()), ph.peak, ph.base)
        lam[name] = (r.l1, r.l2, lambda_opt.packed_l3(eris, r.l3) if blocks else r.l3.reshape(t3d.shape),
                     r.ncycle)
        del r
        ccsdt_opt.release_cached(e)
        sys.stdout.flush()

    a, b = lam['packed block'], lam['blocks']
    no = nocc
    fov, oovv = np.asarray(eris.fock)[:no, no:], np.asarray(eris.oovv)
    pe = [np.einsum('ia,ia', fov.conj(), x[0]) + 0.25 * np.einsum('ijab,ijab', oovv.conj(), x[1])
          for x in (a, b)]
    print('blocks against the packed block: max |dl1| %.2e  |dl2| %.2e  |dl3| %.2e;  cycles %d and %d;  '
          'pseudo-energy sum f l1 + 1/4 sum <ab||ij> l2: %.12f and %.12f (difference %.1e)'
          % (abs(a[0] - b[0]).max(), abs(a[1] - b[1]).max(), abs(a[2] - b[2]).max(), a[3], b[3],
             pe[0].real, pe[1].real, abs(pe[0] - pe[1])))
    print('\nsummary (s per iteration, cores busy, peak anonymous GB above the phase start)')
    for kind in ('amp', 'lam'):
        d, k = out[kind, 'packed block'], out[kind, 'blocks']
        print('  %-9s  packed block %8.3f s %4.1f cores %6.2f GB   blocks %8.3f s %4.1f cores %6.2f GB   '
              'ratio %.2f' % ({'amp': 'amplitude', 'lam': 'Lambda'}[kind], d[0], d[1],
                              (d[2] - d[3]) / 1e9, k[0], k[1], (k[2] - k[3]) / 1e9, d[0] / k[0]))

    # ---- the blocks under a budget a quarter of the way from the floor of the plan to all buffers
    if os.environ.get('ZC_TIGHT') == '1':
        del lam, a, b
        ccsdt_opt.release_cached(eris)
        eng = lambda_opt.block_engine(eris, t3b.dtype)
        space = solver.Options().diis_space
        floor = lambda_opt.plan_memory_blocks(eng, None, 1e-3, space)
        full = lambda_opt.plan_memory_blocks(eng, None, kw['mem_mb'], space)
        budget = floor.need + (full.need - floor.need) // 4
        plan = lambda_opt.plan_memory_blocks(eng, budget, kw['mem_mb'], space)
        del eng
        stat = []
        with Phase() as ph:
            r = lambda_opt.kernel(eris, t1, t2, t3b, solver.Options(
                max_cycle=4, plan=plan, callback=lambda info: stat.append((info['wall'], info['cpu'])),
                **kw), blocks=True)
        w, c = np.array(stat[1:]).T
        print('\nLambda in blocks under a budget: the plan needs %.3f GB at least and %.3f GB with all '
              'buffers; budget %.3f GB -> mem_mb %.1f, need %.3f GB, buffers %.3f GB (all: %.3f GB)'
              % (floor.need / 1e9, full.need / 1e9, budget / 1e9, plan.mem_mb, plan.need / 1e9,
                 plan.buffers / 1e9, full.buffers / 1e9))
        print('  mean %.3f s per iteration, %.1f cores busy;  peak anonymous %.2f GB, %.3f GB above the '
              '%.2f GB before it; the plan without T3: %.3f GB'
              % (w.mean(), c.sum() / w.sum(), ph.peak / 1e9, (ph.peak - ph.base) / 1e9, ph.base / 1e9,
                 (plan.need - t3b.nbytes) / 1e9))


if __name__ == '__main__':
    main()
