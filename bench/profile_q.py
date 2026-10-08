'''Time and memory of the (Q) step on spin orbitals or spinors
(zccsdt.GCCSDT.ccsdt_q, zccsdt.cc.quadruples).

usage: python bench/profile_q.py MOLECULE BASIS KIND [NITER] [FROZEN]
  KIND: ghf   RHF orbitals as spin orbitals (real)
        x2c   spinors (complex; ZC_HAM, ZC_SCF, ... as in bench/spinors.py)
        rhf   the closed-shell code (zccsdt.sa) for the same molecule
  NITER: CCSDT iterations before (Q) (default 0: until convergence).  The
         cost of (Q) does not depend on the amplitudes, its value does.
  FROZEN: frozen occupied spin orbitals or spinors (rhf: orbitals)
  ZC_MAXMEM: max_memory in MB (default: 80% of the available memory)
  ZC_QPLAN: 'cols22, cols13, xv, rows' of an explicit quadruples.Plan
  ZC_INCORE=1: the DIIS vectors of CCSDT in memory
'''
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
from zccsdt.cc import quadruples                                        # noqa: E402


def main():
    name, basis, kind = sys.argv[1:4]
    niter = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    frozen = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    mol = spinors.molecule(name, basis)
    t0 = time.perf_counter()
    if kind == 'x2c':
        mf = spinors.scf(mol)
    else:
        mf = scf.RHF(mol).run(conv_tol=1e-11)
        if kind == 'ghf':
            mf = scf.addons.convert_to_ghf(mf)
    print('%s/%s %s  E(SCF) = %.10f  %.1f s' % (name, basis, kind, mf.e_tot, time.perf_counter() - t0),
          flush=True)
    cc = zccsdt.CCSDT(mf, frozen=frozen or None)
    cc.verbose = 3
    cc.conv_tol, cc.conv_tol_normt = 1e-10, 1e-7
    if niter:
        cc.max_cycle = niter
    if os.environ.get('ZC_MAXMEM'):
        cc.max_memory = float(os.environ['ZC_MAXMEM'])
    if os.environ.get('ZC_INCORE') == '1':          # DIIS vectors in memory (no scratch file)
        cc.incore_complete = True
    cc.run()
    snap = timing.snapshot()
    watch = MemWatch()
    watch.start()
    t0, c0 = time.perf_counter(), time.process_time()
    if kind == 'rhf' or not os.environ.get('ZC_QPLAN'):
        cc.ccsdt_q()
        e5, e6 = cc.e_q5, cc.e_q
    else:
        eris = cc._eris
        no, nv = eris.nocc, eris.nmo - eris.nocc
        pl = quadruples.explicit_plan(no, nv, np.dtype(eris.dtype).itemsize,
                                      *(int(x) for x in os.environ['ZC_QPLAN'].split(',')))
        print('plan: %s' % pl)
        e5, e6 = quadruples.kernel(eris, cc.t2, cc.t3, plan_=pl)
    dt, cpu = time.perf_counter() - t0, time.process_time() - c0
    watch.stop()
    print('(Q) %.2f s   E[Q] = %.12f  E(Q) = %.12f  (CCSDT: %d iterations, converged %s)'
          % (dt, e5, e6, cc.cycles or 0, cc.converged))
    print('(Q) CPU time %.2f s (%.2f cores busy); peak anonymous memory %.2f GB'
          % (cpu, cpu / dt, watch.anon / 1e6))
    for path, (w, c, k) in sorted(timing.since(snap).items(), key=lambda x: -x[1][0]):
        if path[-1].startswith('q.') and len(path) <= 2:
            print('  %-12s %9.2f s  %5.1f cores  %7d calls' % (path[-1], w, c / w if w else 0, k))
    if kind != 'rhf':
        eris = cc._eris
        no, nv = eris.nocc, eris.nmo - eris.nocc
        nQ, nA3 = comb(nv, 2), comb(nv, 3)
        cplx = np.iscomplexobj(eris.oovv)
        fma = comb(no, 4) * (nQ * nQ * (4 * nv + 4 * no + 6) + nv * nA3 * (6 * no + 3 * nv))
        gemm = sum(w for path, (w, c, k) in timing.since(snap).items() if path[-1] == 'q.gemm')
        print('  %d occupied + %d virtual, %s: %.3e multiplications in the products; %.1f GFLOP/s '
              'over the time in them' % (no, nv, 'complex' if cplx else 'real', fma,
                                         2 * fma * (4 if cplx else 1) / max(gemm, 1e-9) / 1e9))
        print('  budget now %s' % memory.budget(cc.max_memory))


if __name__ == '__main__':
    main()
