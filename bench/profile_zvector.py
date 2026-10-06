'''The orbital gradient of the CCSD Lagrangian (zccsdt.zccsd.orbgrad) and the
Z-vector equations (zccsdt.zccsd.zvector) after CCSD and Lambda: wall times
with their timers and peak memory, next to an iteration of the CCSD and of
the Lambda equations of the same run.

usage: python bench/profile_zvector.py MOLECULE BASIS
  MOLECULE: h2o | n2 | c6h6 | ... (the geometries of bench/profile_iter.py)
X2C-1e spinors from PySCF (x2c.UHF), all electrons.
environment: ZC_MAXMEM: max_memory in MB (default 24000); ZC_KRAMERS: 0 or 1:
             DirectZCCSD.kramers; ZC_CCSD_CYCLES, ZC_LAMBDA_CYCLES: at most
             so many cycles (the gradient is then that of amplitudes which
             are not converged: for its time and memory only);
             ZC_ZVECTOR=0: without the Z-vector equations;
             ZC_FROZEN: so many frozen occupied spinors: the relaxed density
             as a whole (zvector.kernel: the gradient among the active
             spinors, the blocks with a frozen index through
             zccsd.orbgrad_frozen, then the Z-vector equations).
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
from zccsdt.zccsd import DirectZCCSD, orbgrad, zvector                  # noqa: E402


def main():
    mol_name, basis = sys.argv[1:3]
    maxmem = float(os.environ.get('ZC_MAXMEM', 24000))
    mol = gto.M(atom=GEOM[mol_name], basis=basis, spin=SPIN.get(mol_name, 0), verbose=0,
                max_memory=maxmem)
    t0 = time.perf_counter()
    mf = x2c.UHF(mol).run(conv_tol=1e-10)
    t_scf = time.perf_counter() - t0
    frozen = int(os.environ.get('ZC_FROZEN', 0))
    mycc = DirectZCCSD(mf, frozen=frozen)
    mycc.verbose, mycc.max_cycle = 0, int(os.environ.get('ZC_CCSD_CYCLES', 100))
    mycc.conv_tol, mycc.conv_tol_normt = 1e-9, 1e-7
    if 'ZC_KRAMERS' in os.environ:
        mycc.kramers = bool(int(os.environ['ZC_KRAMERS']))
    cycles = []
    mycc.callback = lambda envs: cycles.append(time.perf_counter())
    t0 = time.perf_counter()
    eris = mycc.ao2mo()
    t1 = time.perf_counter()
    mycc.kernel(eris=eris)
    t2 = time.perf_counter()
    o, v = mycc.t1.shape
    print('%s/%s DirectZCCSD  nocc=%d nvir=%d  frozen %d  Kramers pairs: %s  max_memory %.0f MB  '
          'threads %s' % (mol_name, basis, o, v, frozen, bool(eris.kramers), maxmem,
                          os.environ.get('OMP_NUM_THREADS', '?')))
    print('matrix products: ' + blas.backend()[1])
    print('SCF %.1f s; integrals %.1f s; CCSD %.1f s: %d cycles (%.2f s each), converged %s, '
          'E_corr %.10f' % (t_scf, t1 - t0, t2 - t1, len(cycles),
                            (t2 - t1) / max(len(cycles), 1), mycc.converged, mycc.e_corr))
    mycc.max_cycle = int(os.environ.get('ZC_LAMBDA_CYCLES', 100))
    mycc.verbose = 4
    t0 = time.perf_counter()
    mycc.solve_lambda()
    t_lam = time.perf_counter() - t0
    print('Lambda %.1f s, converged %s' % (t_lam, mycc.converged_lambda))

    del eris
    watch = MemWatch()
    watch.start()
    timing.reset()
    t0 = time.perf_counter()
    if frozen:
        G = zvector.kernel(mycc, tol=1e-8)[2]       # z
    else:
        G = orbgrad.kernel(mycc)
    t_og = time.perf_counter() - t0
    watch.stop()
    print('%s: %.1f s, largest element %.3e, peak anonymous memory %.2f GB'
          % ('relaxed density (gradient with the frozen spinors and Z-vector equations)' if frozen
             else 'orbital gradient', t_og, abs(G).max(), watch.anon / 1e6))
    for name, w, calls in timing.report():
        if name.startswith('og.'):
            print('  %-12s %9.2f s  %5.1f %%  (%d calls)' % (name[3:], w, 100 * w / t_og, calls))
    for name, w, calls in timing.report():
        if not name.startswith('og.') and w > 0.02 * t_og:
            print('    %-16s %9.2f s  (%d calls)' % (name, w, calls))
    print('that is %.2f CCSD iterations' % (t_og * len(cycles) / (t2 - t1)))

    if os.environ.get('ZC_ZVECTOR', '1') != '0' and not frozen:
        C, mo_e, nocc, frozen = zvector.full_space(mycc)
        R, eai = zvector.response(mf, C, mo_e, nocc)
        t0 = time.perf_counter()
        R(np.asarray(-0.5 * G / eai))
        t_fock = time.perf_counter() - t0
        t0 = time.perf_counter()
        veff = zvector.scf_veff(mf, mycc.max_memory)
        t_setup = time.perf_counter() - t0
        R, eai = zvector.response(mf, C, mo_e, nocc, veff)
        t0 = time.perf_counter()
        R(np.asarray(-0.5 * G / eai))
        print('one Fock build of the response: %.2f s with the SCF object, %.2f s with '
              'zvector.scf_veff (made and checked in %.2f s)'
              % (t_fock, time.perf_counter() - t0, t_setup))
        t0 = time.perf_counter()
        conv, z = zvector.solve(mf, C, mo_e, nocc, -0.5 * G, tol=1e-8, verbose=4, veff=veff)
        print('Z-vector equations: %.1f s, converged %s, largest element %.3e'
              % (time.perf_counter() - t0, conv, abs(z).max()))
        # all of it as DirectZCCSD.make_rdm1(relaxed=True) does it: the density of t and
        # lambda, the gradient, the Fock operator and its check, the equations
        del veff, R
        watch = MemWatch()
        watch.start()
        t0 = time.perf_counter()
        z2 = zvector.kernel(mycc, tol=1e-8)[2]
        t_all = time.perf_counter() - t0
        watch.stop()
        print('relaxed density, all of it (zvector.kernel): %.1f s, peak anonymous memory %.2f GB; '
              'z differs by %.1e' % (t_all, watch.anon / 1e6, abs(z2 - z).max()))
    print('peak RSS %.2f GB' % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6))


if __name__ == '__main__':
    main()
