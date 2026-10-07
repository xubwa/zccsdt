'''Time and memory of CCSDTQ on spin orbitals or spinors (zccsdt.GCCSDTQ,
zccsdt.cc.ccsdtq, zccsdt.cc.t4engine).

usage: python bench/profile_ccsdtq.py MOLECULE BASIS KIND [NITER] [FROZEN]
  KIND: ghf   RHF orbitals as spin orbitals (real)
        x2c   spinors (ZC_HAM, ZC_SCF, ZC_AXIAL, ... as in bench/spinors.py;
              complex unless the SCF keeps the symmetry of the axis)
  NITER: CCSDTQ iterations (default 0: until convergence); the CCSDT step
         before them is converged
  FROZEN: frozen occupied spin orbitals or spinors
  ZC_MAXMEM: max_memory in MB (default: 80% of the available memory)
  ZC_P=1: the (P) correction after the iterations (with the amplitudes as
          they are then: its cost does not depend on them);
          ZC_P_MAXMEM: max_memory of that step
  ZC_Q=1: the (Q) correction of the CCSDT step
  ZC_REAL=0: the spinors as the SCF left them (complex) also for a linear
          molecule on the z axis (default: adapted to m_j, real)
'''
import os
import sys
import time

from pyscf import scf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spinors                                                          # noqa: E402

import zccsdt                                                           # noqa: E402


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
    cc3 = zccsdt.GCCSDT(mf, frozen=frozen or None)
    cc = zccsdt.GCCSDTQ(mf, frozen=frozen or None)
    for x in (cc3, cc):
        x.verbose = 4
        x.real_spinors = os.environ.get('ZC_REAL', '1') != '0'
        x.conv_tol, x.conv_tol_normt = 1e-10, 1e-7
        if os.environ.get('ZC_MAXMEM'):
            x.max_memory = float(os.environ['ZC_MAXMEM'])
    cc3.kernel()
    if os.environ.get('ZC_Q'):
        cc3.ccsdt_q()
    if niter:
        cc.max_cycle = niter
    cc._eris = cc3._eris
    cc.kernel(cc3.t1, cc3.t2, cc3.t3)
    print('E(CCSDT) = %.12f   E(CCSDTQ) = %.12f after %d iterations (converged %s)'
          % (cc3.e_tot, cc.e_tot, cc.cycles, cc.converged))
    if os.environ.get('ZC_P'):
        del cc3
        if os.environ.get('ZC_P_MAXMEM'):
            cc.max_memory = float(os.environ['ZC_P_MAXMEM'])
        cc.ccsdtq_p()
        print('E(P) = %.12e' % cc.e_p)


if __name__ == '__main__':
    main()
