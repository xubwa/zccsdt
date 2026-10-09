'''One zccsdt run of the thread-scaling comparison with CFOUR's ncc
(bench/ncc_scaling/slurm_scaling.sh): CCSDT iterations and the (Q) correction
at the nuclei of a geometry file, with the DIIS vectors in memory
(incore_complete).

usage: python bench/ncc_scaling/zc_run.py KIND GEOMETRY BASIS NCORE NITER Q
  KIND:     rhf - closed-shell spin-adapted program (zccsdt.CCSDT on RHF: RCCSDT)
            ghf - spin-orbital program on the spin orbitals of that RHF
                  (zccsdt.CCSDT on a GHF object: GCCSDT, real arithmetic)
  GEOMETRY: 'symbol x y z' per atom in bohr (the file CFOUR's ZMAT is made of)
  BASIS:    PySCF name;  NCORE: frozen core orbitals (spatial)
  NITER:    number of CCSDT iterations (the time per iteration is the mean of
            iterations 2..NITER, as for ncc); 0: converged (|dE| < 1e-11,
            residual 1e-8), which is the run whose energies are compared
  Q:        1 - then the (Q) correction with the amplitudes at hand (its cost
            does not depend on them; its energy means something only with
            NITER = 0)
  ZC_MAXMEM: max_memory in MB
Prints lines 'ZC key = value' that summary.py reads.
'''
import os
import resource
import sys
import time

from pyscf import gto, scf

import zccsdt


def main():
    kind, geometry, basis = sys.argv[1:4]
    ncore, niter, with_q = int(sys.argv[4]), int(sys.argv[5]), sys.argv[6] == '1'
    with open(geometry) as f:
        mol = gto.M(atom=f.read(), unit='Bohr', basis=basis, verbose=0)
    if kind == 'rhf':
        mf = scf.RHF(mol)
        frozen = ncore or None
    elif kind == 'ghf':
        mf = scf.RHF(mol)
        frozen = 2 * ncore or None
    else:
        raise SystemExit('KIND: rhf | ghf')
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-9, 300
    mf.kernel()
    if kind == 'ghf':
        mf = mf.to_ghf()
    print('ZC case = %s %s %s: %d basis functions, %d electrons, %d frozen core orbitals'
          % (kind, os.path.basename(geometry), basis, mol.nao_nr(), mol.nelectron, ncore))
    print('ZC threads = %s' % os.environ.get('OMP_NUM_THREADS'))
    print('ZC e_scf = %.12f' % mf.e_tot, flush=True)
    cc = zccsdt.CCSDT(mf, frozen=frozen)
    cc.verbose = 4
    cc.incore_complete = True
    if niter:
        cc.max_cycle = niter
    else:
        cc.conv_tol, cc.conv_tol_normt = 1e-11, 1e-8
    if os.environ.get('ZC_MAXMEM'):
        cc.max_memory = float(os.environ['ZC_MAXMEM'])
    cpus = []

    def callback(info):
        cpus.append(info.get('cpu'))
    cc.callback = callback
    t0, c0 = time.perf_counter(), time.process_time()
    cc.kernel()
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    t = list(cc.timings)
    print('ZC class = %s' % type(cc).__name__)
    print('ZC iterations = ' + ' '.join('%.3f' % x for x in t))
    if all(c is not None for c in cpus) and cpus:
        print('ZC iteration_cpu = ' + ' '.join('%.3f' % x for x in cpus))
    print('ZC cycles = %d' % cc.cycles)
    print('ZC converged = %s' % cc.converged)
    print('ZC ccsdt_wall = %.2f' % wall)
    print('ZC ccsdt_cpu = %.2f' % cpu)
    print('ZC diis = %s' % cc.diis_plan_used)
    print('ZC peak_anon_gb = %.2f' % (cc.peak_memory / 1e9))
    print('ZC e_corr = %.12f' % cc.e_corr)
    print('ZC e_ccsdt = %.12f' % cc.e_tot, flush=True)
    if with_q:
        t0, c0 = time.perf_counter(), time.process_time()
        cc.ccsdt_q()
        print('ZC q_wall = %.2f' % (time.perf_counter() - t0))
        print('ZC q_cpu = %.2f' % (time.process_time() - c0))
        print('ZC e_q5 = %.12f' % cc.e_q5)
        print('ZC e_q = %.12f' % cc.e_q)
        print('ZC e_ccsdtq = %.12f' % cc.e_tot_q)
    print('ZC maxrss_gb = %.2f' % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6))


if __name__ == '__main__':
    main()
