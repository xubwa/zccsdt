'''Left-hand vectors of EOM-EE-CCSDT through the front end, timed against the
right-hand ones of the same run (docs/eom.md, section 12):

    python -u bench/eom_left_bench.py h2o [NROOTS]         H2O / cc-pVTZ, RHF as GHF spin orbitals (10 + 106), packed block
    python -u bench/eom_left_bench.py hbr [NROOTS] [SHIFT] HBr / cc-pVTZ-DK, X2C, 28 frozen (8 + 78), label blocks;
                                                           SHIFT: the label shift (default 0)
    python -u bench/eom_left_bench.py hbr_dense [NROOTS]   the same on the packed block

Environment: EB_MAXMEM (MB, default 26000), EB_CONV (default 1e-6), EB_BASIS
(default cc-pvtz / cc-pvtz-dk), EB_DENSITY=1: the excited-state densities,
dipole moments and transition moments as well (2: with the response of the
amplitudes); EB_LEFTMEM: the left solve with max_memory set to the resident
memory at its start plus so many MB (the plan under a budget: smaller
buffers, a smaller subspace, the vectors in a scratch file).

Prints seconds per sigma vector and per left product, the iterations of
both sides, the eigenvalues of both sides, biorthogonality, and the peak of
the resident memory of the left solve against its plan.'''
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
nroots = int(sys.argv[2]) if len(sys.argv) > 2 else 2
shift = int(sys.argv[3]) if len(sys.argv) > 3 else None
conv = float(os.environ.get('EB_CONV', 1e-6))
if case == 'h2o':
    mol = gto.M(atom=H2O_BOHR, unit='Bohr', basis=os.environ.get('EB_BASIS', 'cc-pvtz'), verbose=0)
    rhf = scf.RHF(mol).run(conv_tol=1e-12)
    print('E(RHF) = %.12f' % rhf.e_tot)
    cc = zccsdt.CCSDT(scf.addons.convert_to_ghf(rhf))
else:
    mol = gto.M(atom='Br 0 0 0; H 0 0 1.4145', basis=os.environ.get('EB_BASIS', 'cc-pvtz-dk'), verbose=0)
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
t0 = time.perf_counter()
e, res = cc.eom_ee(nroots=nroots, conv_tol=conv, shift=shift)
t_right = time.perf_counter() - t0
timing.reset()
from zccsdt import memory
from zccsdt.cc import ccsdt_opt
ccsdt_opt.release_cached(cc._eris)
base = memory.resident()            # (T3, the right-hand vectors and the operands of their sigma vector)
if os.environ.get('EB_LEFTMEM'):    # the left solve under a budget of so many MB above what is resident now
    cc.max_memory = base / 1e6 + float(os.environ['EB_LEFTMEM'])
    print('left solve: max_memory %.0f MB = resident %.0f MB + %s MB' % (cc.max_memory, base / 1e6, os.environ['EB_LEFTMEM']))
t0 = time.perf_counter()
with _PeakMemory() as peak:
    left = cc.eom_left(conv_tol=conv)
t_left = time.perf_counter() - t0
snap = {str(k): v for k, v in timing.snapshot().items()}
tr, tl = res.timings, left.timings
print('\n== %s%s: %d roots' % (case, '' if shift is None else ' shift %+d' % shift, len(e)))
for k in range(len(e)):
    print('   root %d  right %.10f  left %.10f  difference %.1e  |res| %.1e / %.1e'
          % (k + 1, e[k], left.e[k], left.e[k] - e[k], res.residuals[k], left.residuals[k]))
print('   CCSDT iteration            %8.2f s' % it)
print('   right: %d sigma vectors in %d iterations, %.2f s each (%.2f CCSDT iterations); guess stage %.1f s, prepare %.1f s; '
      'in all %.1f s' % (res.nsigma, res.niter, tr['iterations'] / res.nsigma, tr['iterations'] / res.nsigma / it,
                         tr['guess'], tr['prepare'], t_right))
print('   left:  %d products      in %d iterations, %.2f s each (%.2f CCSDT iterations); intermediates %.1f s; in all %.1f s'
      % (left.nsigma, left.niter, tl['iterations'] / left.nsigma, tl['iterations'] / left.nsigma / it, tl['prepare'], t_left))
print('   left / right: per product %.2f, products %.2f, iterations stage %.2f, in all %.2f'
      % (tl['iterations'] / left.nsigma / (tr['iterations'] / res.nsigma), left.nsigma / res.nsigma,
         tl['iterations'] / tr['iterations'], t_left / t_right))
print('   biorthogonality (different levels, before the normalization) %.1e; |diag(L R) - 1| %.1e'
      % (left.biorthogonality, abs(left.overlap.diagonal() - 1).max()))
t3b = np.asarray(cc.t3).nbytes
print('   left: resident memory %.2f GB before, peak %.2f GB: %.2f GB more; plan %.2f GB less T3 (%.2f GB, there before) = %.2f GB (%s)'
      % (base / 1e9, (peak.peak or 0) / 1e9, ((peak.peak or 0) - base) / 1e9, left.plan.need / 1e9, t3b / 1e9,
         (left.plan.need - t3b) / 1e9, left.plan.text()))
print('   left: buffers of the passes %.0f MB; vectors mapped: %s' % (left.plan.lam.mem_mb, left.plan.mapped))
print('   timers of the left solve (wall s, cpu s, calls):')
for k in sorted(snap):
    v = snap[k]
    w = v[0] if isinstance(v, (tuple, list)) else v
    if w > 0.01 * t_left:
        print('     %-40s %s' % (k, ' '.join('%.2f' % y if isinstance(y, float) else str(y) for y in (v if isinstance(v, (tuple, list)) else [v]))))
if os.environ.get('EB_DENSITY'):
    resp = os.environ.get('EB_DENSITY') == '2'
    cc.solve_lambda()
    lam_it = None
    base = memory.resident()
    t0 = time.perf_counter()
    with _PeakMemory() as peak:
        out = cc.eom_properties(response=resp)
    print('   densities%s, dipole and transition moments of %d states: %.1f s (%.1f CCSDT iterations per state); '
          'resident memory %.2f GB before, peak %.2f GB'
          % (' with the response of the amplitudes' if resp else '', len(e), time.perf_counter() - t0,
             (time.perf_counter() - t0) / len(e) / it, base / 1e9, (peak.peak or 0) / 1e9))
    for st in out.states:
        print('   state %d: %s; Lambda cycles of the response %d' % (st.root + 1, ', '.join('%s %.1f s' % kv for kv in st.timings.items()), st.lambda_cycles))
    for line in out.text():
        print('   ' + line)
