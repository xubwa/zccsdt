'''Time and memory of CCSDT(Q)_Lambda next to (Q) on spin orbitals or spinors:
the dense codes (zccsdt.cc.quadruples, quadruples_lambda) and those in
symmetry sectors (quadruples_sym, quadruples_lambda_sym), one after the
other on the same amplitudes.

usage: python bench/profile_q_lambda.py MOLECULE BASIS KIND [NITER] [FROZEN]
  KIND: ghf   RHF orbitals as spin orbitals (real); the labels of the
              sectors are 2 m_s (the front end has none for them)
        x2c   spinors (ZC_HAM, ZC_SCF, ... as in bench/spinors.py); a linear
              molecule on the z axis has the labels 2 m_j
  NITER: CCSDT iterations before (default 0: until convergence).  The cost
         does not depend on the amplitudes, the values do.
  FROZEN: frozen occupied spin orbitals or spinors
  ZC_LAMBDA=1: solve the Lambda equations (else lambda = conj(t), which
         costs the same and gives (Q) again)
  ZC_MAXMEM: max_memory in MB (default: 80% of the available memory)
'''
import copy
import os
import sys
import time
from math import comb

import numpy as np
from pyscf import scf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spinors                                                          # noqa: E402
from profile_sa import MemWatch                                         # noqa: E402

import zccsdt                                                           # noqa: E402
from zccsdt import memory, timing                                       # noqa: E402
from zccsdt.cc import (ccsdt_opt, quadruples as QD, quadruples_lambda as QL,       # noqa: E402
                       quadruples_lambda_sym as QLS, quadruples_sym as QS)


def spin_labels(cc, eris):
    '''2 m_s of the active spin orbitals of a GHF determinant made of RHF
    orbitals (occupied, then virtual).'''
    C = np.asarray(cc.mo_coeff)
    occ = np.asarray(cc.mo_occ) > 0
    idx = np.concatenate((np.where(occ)[0], np.where(~occ)[0]))
    assert len(idx) == eris.nmo, 'frozen spin orbitals: not here'
    up = np.linalg.norm(C[:C.shape[0] // 2, idx], axis=0) / np.linalg.norm(C[:, idx], axis=0)
    assert (abs(up - 1) < 1e-6).sum() + (abs(up) < 1e-6).sum() == len(idx)
    return np.where(up > 0.5, 1, -1)


def main():
    name, basis, kind = sys.argv[1:4]
    niter = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    frozen = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    mol = spinors.molecule(name, basis)
    t0 = time.perf_counter()
    if kind == 'x2c':
        mf = spinors.scf(mol)
    else:
        mf = scf.addons.convert_to_ghf(scf.RHF(mol).run(conv_tol=1e-11))
    print('%s/%s %s  E(SCF) = %.10f  %.1f s' % (name, basis, kind, mf.e_tot, time.perf_counter() - t0),
          flush=True)
    cc = zccsdt.CCSDT(mf, frozen=frozen or None)
    cc.verbose = 3
    cc.conv_tol, cc.conv_tol_normt = 1e-10, 1e-8
    if niter:
        cc.max_cycle = niter
    max_memory = float(os.environ['ZC_MAXMEM']) if os.environ.get('ZC_MAXMEM') else None
    cc.max_memory = max_memory
    cc.run()
    solved = os.environ.get('ZC_LAMBDA') == '1'
    if solved:
        t0 = time.perf_counter()
        cc.solve_lambda()
        print('Lambda: converged %s, %.1f s' % (cc.converged_lambda, time.perf_counter() - t0), flush=True)
    eris = cc._eris
    t2, t3 = cc.t2, ccsdt_opt.packed_t3(eris, cc.t3)
    ccsdt_opt.release_cached(eris)
    l2, l3 = (cc.l2, np.asarray(cc.l3)) if solved else (np.conj(t2), np.conj(t3))
    if getattr(eris, 'labels', None) is None and kind == 'ghf':
        eris = copy.copy(eris)
        eris.labels, eris.modulus = spin_labels(cc, eris), 0
    no, nv = eris.nocc, eris.nmo - eris.nocc
    dtype = np.result_type(t2, t3, l2, l3, eris.oovv)
    isz = np.dtype(dtype).itemsize
    budget = memory.budget(max_memory)
    print('%d occupied, %d virtual, %s; T3 %.2f GB; CCSDT %d iterations (converged %s); lambda %s; budget %s'
          % (no, nv, dtype, t3.nbytes / 1e9, cc.cycles or 0, cc.converged,
             'solved' if solved else 'conj(t)', 'none' if budget is None else '%.1f GB' % (budget / 1e9)), flush=True)
    sym = QLS.available(eris)
    if sym:
        sec = QS.Sectors(eris)
        print('labels of the virtual spinors: %s%s' % (
            {k: len(v) for k, v in sec.V.items()}, ' modulo %d' % sec.modulus if sec.modulus else ''))
    runs = [('(Q) dense', lambda: QD.plan(no, nv, isz, budget),
             lambda pl: QD.kernel(eris, t2, t3, plan_=pl))]
    if sym:
        runs.append(('(Q) sectors', lambda: QS.plan(eris, isz, budget),
                     lambda pl: QS.kernel(eris, t2, t3, plan_=pl)))
    runs.append(('(Q)_L dense', lambda: QL.plan(no, nv, isz, budget),
                 lambda pl: QL.kernel(eris, t2, t3, l2, l3, plan_=pl)))
    if sym:
        runs.append(('(Q)_L sectors', lambda: QLS.plan(eris, isz, budget),
                     lambda pl: QLS.kernel(eris, t2, t3, l2, l3, plan_=pl)))
    with open('/proc/self/status') as f:
        base = {l.split(':')[0]: int(l.split()[1]) for l in f if l.startswith('Rss')}['RssAnon']
    print('anonymous memory before (amplitudes, integrals): %.2f GB' % (base / 1e6))
    out = {}
    for tag, mkplan, fn in runs:
        pl = mkplan()
        dense = getattr(pl, 'dense', pl)
        snap = timing.snapshot()
        watch = MemWatch()
        watch.start()
        t0, c0 = time.perf_counter(), time.process_time()
        e = fn(pl)
        dt, cpu = time.perf_counter() - t0, time.process_time() - c0
        watch.stop()
        out[tag] = (e, dt)
        print('%-14s %8.2f s  %5.2f cores busy  peak anonymous %.2f GB  plan %.2f GB (cols %d/%d, %d/%d; xv %d)'
              '  E[Q] = %.12f  E(Q) = %.12f'
              % (tag, dt, cpu / dt, watch.anon / 1e6, pl.need / 1e9, dense.cols22, comb(nv, 2), dense.cols13,
                 comb(nv, 3), dense.xv, e[0], e[1]), flush=True)
        for path, (w, c, k) in sorted(timing.since(snap).items(), key=lambda x: -x[1][0]):
            if path[-1].split('.')[0] in ('q', 'ql', 'qs', 'qls') and len(path) <= 2:
                print('    %-12s %9.2f s  %5.1f cores  %7d calls' % (path[-1], w, c / w if w else 0, k))
    print('(Q)_L / (Q): dense %.2f' % (out['(Q)_L dense'][1] / out['(Q) dense'][1])
          + ('   sectors %.2f   dense / sectors: (Q) %.2f  (Q)_L %.2f' % (
              out['(Q)_L sectors'][1] / out['(Q) sectors'][1], out['(Q) dense'][1] / out['(Q) sectors'][1],
              out['(Q)_L dense'][1] / out['(Q)_L sectors'][1]) if sym else ''))
    if sym:
        print('sectors - dense: (Q) %.1e %.1e   (Q)_L %.1e %.1e' % (
            *np.subtract(out['(Q) sectors'][0], out['(Q) dense'][0]),
            *np.subtract(out['(Q)_L sectors'][0], out['(Q)_L dense'][0])))


if __name__ == '__main__':
    main()
