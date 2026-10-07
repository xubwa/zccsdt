'''The CCSDTQ iterations in the blocks of the symmetry labels
(zccsdt.cc.ccsdtq_sym) or dense (zccsdt.cc.ccsdtq): a linear molecule on the
z axis with X2C spinors (labels 2 m_j), from the CCSDT amplitudes of the
front end.  One mode per process, so that the peak memory is that of the
mode: run 'dense' first and 'sym' after it in one job for a comparison.

usage: python bench/sym_ccsdtq.py MODE GEOM BASIS FROZEN [NVIR] [NITER]
  MODE: dense | sym
  NVIR: the lowest NVIR virtual spinors only (0: all)
  NITER: CCSDTQ iterations (default 0: until convergence)
  ZC_MAXMEM: max_memory in MB (default: 80% of the available memory)
  ZC_TRACE=1 (sym): one more evaluation of the T4 equation under
        tracemalloc, its peak against the count of the plan
       e.g.  python bench/sym_ccsdtq.py sym 'Br 0 0 0; H 0 0 1.4145' cc-pvtz-dk 28
'''
import copy
import gc
import os
import sys
import time
import tracemalloc
from math import comb

import numpy as np
from pyscf import gto, x2c

import zccsdt
from zccsdt import timing
from zccsdt.ccsdt import _PeakMemory, _rss_anon
from zccsdt.cc import ccsdt_opt, ccsdtq, ccsdtq_sym, t4engine, t4engine_sym


def main():
    mode, geom, basis, frozen = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    nvir = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    niter = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    maxmem = float(os.environ['ZC_MAXMEM']) if os.environ.get('ZC_MAXMEM') else None
    mol = gto.M(atom=geom, basis=basis, verbose=0)
    mf = x2c.UHF(mol)
    mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-8
    mf.kernel()
    order = np.argsort(mf.mo_energy)
    nocc = int((mf.mo_occ > 0).sum())
    cc = zccsdt.GCCSDT(mf, frozen=list(order[:frozen]) + (list(order[nocc + nvir:]) if nvir else []))
    cc.verbose = 0
    cc.conv_tol, cc.conv_tol_normt = 1e-10, 1e-8
    t0 = time.perf_counter()
    cc.kernel()
    eris = cc.ao2mo()
    no, nv = eris.nocc, eris.nmo - eris.nocc
    lay = t4engine_sym.layout_for(eris)
    dtype = np.result_type(eris.fock, eris.oovv)
    print('%s  %s  %d + %d spinors, %s  E(SCF) = %.10f  E_corr(CCSDT) = %.12f  (%d iterations, %.1f s)'
          % (mode, basis, no, nv, np.dtype(dtype).name, mf.e_tot, cc.e_corr, cc.cycles, time.perf_counter() - t0))
    nt4 = comb(no, 4) * comb(nv, 4)
    print('T4: %d elements packed, %d in the blocks of %d labels (%.1f %%), %d sectors; T3: %d of %d'
          % (nt4, lay.size, lay.ncls, 100.0 * lay.size / nt4, len(lay.sectors), int(lay.t3off[-1]),
             comb(no, 3) * comb(nv, 3)))
    t1, t2, t3 = cc.t1, cc.t2, cc.t3
    if mode == 'dense':
        t3 = ccsdt_opt.packed_t3(eris, t3)
        ccsdt_opt.release_cached(eris)
        eris = copy.copy(eris)
        eris.use_labels = False
        eris.__dict__.pop('_buffers', None)
        mod, plan = ccsdtq, ccsdtq.plan_for(eris, dtype, maxmem)
        print('plan: needs %.0f MB besides the integrals; batches of %.0f MB; T4 %s, residual %s, Xv %s, DIIS %s'
              % (plan.need / 1e6, plan.batch_bytes / 1e6, *('file' if x else 'memory' for x in
                 (plan.t4_file, plan.r4_file, plan.xv_file, plan.diis_file))))
    else:
        mod, plan = ccsdtq_sym, ccsdtq_sym.plan_for(eris, dtype, maxmem)
        print('plan: needs %.0f MB besides the integrals (floor %.0f MB); batches of %.0f MB; DIIS vectors '
              'of T4 in %s' % (plan.need / 1e6, plan.floor / 1e6, plan.batch_bytes / 1e6,
                               'a file' if plan.diis_file else 'memory'))
    del cc
    gc.collect()
    rows = []

    def report(info):
        rows.append((info['wall'], info['cpu']))
        print(' %3d  %18.12f  %10.2e  %9.2e  %8.2f s  %4.1f cores'
              % (info['cycle'], info['e_corr'], info['de'], info['normt'], info['wall'],
                 info['cpu'] / info['wall']), flush=True)
    opts = ccsdtq.Options(conv_tol=1e-10, conv_tol_normt=1e-8, max_memory=maxmem, plan=plan, callback=report)
    if niter:
        opts.max_cycle = niter
    base = _rss_anon()
    snap = timing.snapshot()
    t0 = time.perf_counter()
    with _PeakMemory() as peak:
        res = mod.kernel(eris, opts, t1, t2, t3)
    wall = time.perf_counter() - t0
    w = np.array(rows)
    later = w[1:] if len(w) > 1 else w
    print('%s: E_corr(CCSDTQ) = %.12f  %d iterations (converged %s)  %.1f s'
          % (mode, res.e_corr, res.ncycle, res.converged, wall))
    print('%s: %.2f s per iteration (after the first: %.2f s), %.1f cores busy; peak anonymous memory %.0f MB '
          '(%.0f MB before the iterations)' % (mode, w[:, 0].mean(), later[:, 0].mean(),
                                                later[:, 1].sum() / later[:, 0].sum(), peak.peak / 1e6, base / 1e6))
    tree = timing.since(snap)
    for key in sorted(tree):
        if tree[key][0] >= 0.005 * wall and len(key) <= 3:
            print('   %-44s %9.2f s wall %9.2f s CPU' % ('/'.join(key), tree[key][0], tree[key][1]))
    if mode == 'sym' and os.environ.get('ZC_TRACE'):
        ints = t4engine.Dressed.from_eris(eris, res.t1)
        isz = np.dtype(dtype).itemsize
        gc.collect()
        tracemalloc.start()
        t4engine_sym.residual(lay, res.t2, res.t3, res.t4, ints, plan.batch_bytes)
        tp = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        print('T4 equation alone: peak of the allocations %.0f MB, counted by the plan %.0f MB'
              % (tp / 1e6, t4engine_sym.memory(lay, isz, plan.batch_bytes) / 1e6))


if __name__ == '__main__':
    main()
