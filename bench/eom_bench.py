'''EOM-EE-CCSDT through the front end, timed against the CCSDT iteration of
the same run (docs/eom.md, docs/benchmarks.md):

    python -u bench/eom_bench.py h2o [NROOTS]         H2O / cc-pVTZ, RHF as GHF spin orbitals, all electrons (10 + 106)
    python -u bench/eom_bench.py hbr [NROOTS]         HBr / cc-pVTZ-DK, X2C spinors, 28 frozen (8 + 78), label blocks
    python -u bench/eom_bench.py hbr_dense [NROOTS]   the same on the packed block (all states)

Environment: EB_GENBAS - a GENBAS with O:PVTZ and H:PVTZ (CFOUR's), so that
H2O has the basis of the MRCC run of bench/slurm_eom.sh (default: PySCF's
cc-pVTZ); EB_MAXMEM - max_memory in MB (default 26000); EB_CONV - the
threshold of the EOM residual (default 1e-6).

Prints the seconds per CCSDT iteration (median), per sigma vector (the
timer around the sigma vectors of the triples stage, and the whole stage
by its number of sigma vectors), the sigma vectors per root, the peak of
the resident memory against the plan, and the stage timers.'''
import os
import sys
import time

import numpy as np
from pyscf import gto, scf, x2c

import zccsdt
from zccsdt import timing
from zccsdt.ccsdt import _PeakMemory

H2O_BOHR = '''O 0.000000000000000 0.000000000000000 0.000000000000000
H 0.000000000000000 1.430522676295752 1.109269235119691
H 0.000000000000000 -1.430522676295752 1.109269235119691'''

case = sys.argv[1]
nroots = int(sys.argv[2]) if len(sys.argv) > 2 else (4 if case != 'hbr' else 3)
if case == 'h2o':
    basis = 'cc-pvtz'
    if os.environ.get('EB_GENBAS'):
        from zccsdt.genbas import read_genbas
        basis = {el: read_genbas(os.environ['EB_GENBAS'], '%s:PVTZ' % el) for el in 'OH'}
    mol = gto.M(atom=H2O_BOHR, unit='Bohr', basis=basis, verbose=0)
    rhf = scf.RHF(mol).run(conv_tol=1e-12)
    print('E(RHF) = %.12f' % rhf.e_tot)
    cc = zccsdt.CCSDT(scf.addons.convert_to_ghf(rhf))
else:
    mol = gto.M(atom='Br 0 0 0; H 0 0 1.4145', basis='cc-pvtz-dk', verbose=0)
    mf = x2c.UHF(mol)
    mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-8
    mf.kernel()
    print('E(X2C SCF) = %.12f' % mf.e_tot)
    cc = zccsdt.CCSDT(mf, frozen=28)
    cc.symmetry = case == 'hbr'
cc.verbose = 4
cc.max_memory = float(os.environ.get('EB_MAXMEM', 26000))
cc.conv_tol, cc.conv_tol_normt = 1e-10, 1e-8
cc.kernel()
it = float(np.median(cc.timings))
print('CCSDT: E = %.12f, %d iterations, median %.2f s per iteration' % (cc.e_tot, cc.cycles, it), flush=True)
timing.reset()
t0 = time.perf_counter()
with _PeakMemory() as peak:
    e, res = cc.eom_ee(nroots=nroots, conv_tol=float(os.environ.get('EB_CONV', 1e-6)))
wall = time.perf_counter() - t0
snap = {str(k): v for k, v in timing.snapshot().items()}
print('\n== %s: %d roots, EOM-EE-CCSDT in %.1f s' % (case, len(e), wall))
for k, x in enumerate(e):
    print('   root %d  %.10f Eh  %.6f eV  |r| %.1e  converged %s' % (k + 1, x, x * 27.211386245988, res.residuals[k], res.converged[k]))
tr = res.timings
nsig = res.nsigma
print('   CCSDT iteration            %8.2f s' % it)
print('   sigma vectors              %d (%.1f per root) in %d iterations, %d collapses; guess stage: %d singles-doubles vectors, %.1f s'
      % (nsig, nsig / len(e), res.niter, res.ncollapse, res.guess.nsigma, tr['guess']))
print('   triples stage              %8.2f s = %.2f s per sigma vector = %.2f CCSDT iterations each'
      % (tr['iterations'], tr['iterations'] / nsig, tr['iterations'] / nsig / it))
print('   prepare                    %8.2f s' % tr['prepare'])
print('   peak resident memory       %.2f GB; plan %.2f GB (%s)' % ((peak.peak or 0) / 1e9, res.plan.need / 1e9, res.plan.text()))
print('   timers (wall s, cpu s, calls):')
for k in sorted(snap):
    v = snap[k]
    w = v[0] if isinstance(v, (tuple, list)) else v
    if 'eom' in k and w > 0.005 * wall:
        print('     %-40s %s' % (k, ' '.join('%.2f' % y if isinstance(y, float) else str(y) for y in (v if isinstance(v, (tuple, list)) else [v]))))
