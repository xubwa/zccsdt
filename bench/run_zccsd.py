'''A whole spinor CCSD(T) with zccsdt.zccsd.DirectZCCSD, to convergence: wall
times of the integrals, the CCSD iterations and (T), for the comparison with
a whole run of CFOUR's xx2cvtran and xx2cccsd (bench/cfour/slurm_x2ccc.sh).

usage: python bench/run_zccsd.py MOLECULE BASIS
  MOLECULE: h2o | n2 | c6h6 | ... (the geometries of bench/profile_iter.py)
X2C-1e spinors from PySCF (x2c.UHF), all electrons.  The iterations stop
when the energy changes by less than 1e-9 and the norm of the change of
[t1, packed t2] is below 1e-7 (ZC_CONV); that norm bounds the largest change
of an amplitude, which is what CFOUR's CC_CONV=7 asks to be below 1e-7.
environment: ZC_MAXMEM: max_memory in MB (default 24000); ZC_KRAMERS: 0 or 1:
             DirectZCCSD.kramers (default: Kramers pairs if the reference has
             them); ZC_CONV: the threshold of the norm (default 1e-7);
             ZC_AO_LADDER: DirectZCCSD.ao_ladder (memory | direct | file).
'''
import os
import resource
import sys
import time

from pyscf import gto, x2c

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profile_iter import GEOM, SPIN, MemWatch                           # noqa: E402

from zccsdt import blas                                                 # noqa: E402
from zccsdt.zccsd import DirectZCCSD                                    # noqa: E402


def main():
    mol_name, basis = sys.argv[1:3]
    maxmem = float(os.environ.get('ZC_MAXMEM', 24000))
    conv = float(os.environ.get('ZC_CONV', 1e-7))
    mol = gto.M(atom=GEOM[mol_name], basis=basis, spin=SPIN.get(mol_name, 0), verbose=0,
                max_memory=maxmem)
    t0 = time.perf_counter()
    mf = x2c.UHF(mol).run(conv_tol=1e-10)
    t_scf = time.perf_counter() - t0

    mycc = DirectZCCSD(mf)
    mycc.verbose, mycc.max_cycle = 4, 100
    mycc.conv_tol, mycc.conv_tol_normt = 0.01 * conv, conv
    if 'ZC_KRAMERS' in os.environ:
        mycc.kramers = bool(int(os.environ['ZC_KRAMERS']))
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
    e_t = mycc.ccsd_t()
    t3 = time.perf_counter()
    watch.stop()

    nocc, nvir = mycc.t1.shape
    print('%s/%s DirectZCCSD  nocc=%d nvir=%d  Kramers pairs: %s  max_memory %.0f MB  threads %s'
          % (mol_name, basis, nocc, nvir, bool(eris.kramers), maxmem,
             os.environ.get('OMP_NUM_THREADS', '?')))
    print('matrix products: ' + blas.backend()[1])
    print('SCF %.1f s  E(SCF) %.10f' % (t_scf, mf.e_tot))
    print('integrals %.1f s' % (t1 - t0))
    print('CCSD %.1f s: %d cycles (%.2f s each), converged %s, E_corr %.10f'
          % (t2 - t1, len(cycles), (t2 - t1) / max(len(cycles), 1), mycc.converged, mycc.e_corr))
    print('(T) %.1f s: E(T) %.10f' % (t3 - t2, e_t))
    print('integrals + CCSD + (T): %.1f s; peak anonymous memory %.2f GB, peak RSS %.2f GB'
          % (t3 - t0, watch.anon / 1e6, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6))


if __name__ == '__main__':
    main()
