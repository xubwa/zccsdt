'''EOM-CCSDT of a linear molecule in the blocks of the symmetry labels, timed
against the CCSDT iteration of the same run (docs/eom.md, section 8):

    python -u bench/eom2_bench.py tz [PARTS]    HBr / cc-pVTZ-DK, X2C, 28 frozen (8 + 78)
    python -u bench/eom2_bench.py qz [PARTS]    HBr / cc-pVQZ-DK, X2C, 10 frozen, the lowest 120 virtual spinors (26 + 120)

PARTS (comma-separated; default "ee,ip"):
    ee    the lowest states of the label shifts E2_SHIFTS (default "0,2,-2,4,-4"; of a pair
          +-D the first is computed, the reference being symmetric under time reversal),
          E2_NROOTS (default 3) roots each
    ip    the ionization energies of the valence spinors with short vectors, by the label
          of the electron (E2_IP_NROOTS roots per label, default 2)
    ea    the same for attached states

Environment: E2_CONV (threshold of the EOM residual, default 1e-6), E2_MAXMEM
(max_memory in MB, default 80000), E2_CC_TOL (default 1e-9).

Prints, per eigenproblem: the length of the vectors, the sigma vectors per
root, the seconds per sigma vector (the triples stage by its number of
sigma vectors) against the seconds per CCSDT iteration (median), the peak of
the resident memory against the plan, and the timers of the stages.'''
import os
import sys
import time
from math import comb

import numpy as np
from pyscf import gto, x2c

import zccsdt
from zccsdt import timing
from zccsdt.cc import eom
from zccsdt.ccsdt import _PeakMemory, _rss_anon

case = sys.argv[1]
parts = (sys.argv[2] if len(sys.argv) > 2 else 'ee,ip').split(',')
conv = float(os.environ.get('E2_CONV', 1e-6))
basis, nfrozen, nvir = {'tz': ('cc-pvtz-dk', 28, 0), 'qz': ('cc-pvqz-dk', 10, 120), 'dz': ('cc-pvdz-dk', 28, 0)}[case]
mol = gto.M(atom='Br 0 0 0; H 0 0 1.4145', basis=basis, verbose=0)
mf = x2c.UHF(mol)
mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-7
mf.kernel()
print('E(X2C SCF) = %.12f' % mf.e_tot, flush=True)
frozen = nfrozen
if nvir:            # the lowest occupied and the highest virtual spinors left out
    order, nocc = np.argsort(mf.mo_energy), int((np.asarray(mf.mo_occ) > 0).sum())
    frozen = [int(i) for i in order[:nfrozen]] + [int(i) for i in order[nocc + nvir:]]
cc = zccsdt.CCSDT(mf, frozen=frozen)
cc.verbose = 4
cc.max_memory = float(os.environ.get('E2_MAXMEM', 80000))
cc.conv_tol = float(os.environ.get('E2_CC_TOL', 1e-9))
cc.conv_tol_normt = 100 * cc.conv_tol
t0 = time.perf_counter()
cc.kernel()
it = float(np.median(cc.timings))
eris = cc._eris
print('CCSDT: E = %.12f, %d iterations in %.1f s, median %.3f s per iteration; %d + %d spinors, T3 %d elements in blocks (%d packed)'
      % (cc.e_tot, cc.cycles, time.perf_counter() - t0, it, eris.nocc, eris.nmo - eris.nocc, np.size(cc.t3),
         comb(eris.nocc, 3) * comb(eris.nmo - eris.nocc, 3)), flush=True)
print('labels of the active spinors:', ' '.join(str(int(x)) for x in eris.labels))
cc.verbose = 4


def one(what, call, n):
    timing.reset()
    base = _rss_anon()
    t0 = time.perf_counter()
    with _PeakMemory() as peak:
        e, res = call()
    wall = time.perf_counter() - t0
    tr, nsig = res.timings, res.nsigma
    sizes = [res.r1[0].size, res.r2[0].size, res.r3[0].size]
    per = tr['iterations'] / nsig
    print('\n== %s: %d roots in %.1f s; vectors %d + %d + %d' % (what, len(e), wall, *sizes))
    for k, x in enumerate(e):
        print('   root %d  %.10f Eh  %.6f eV  |r| %.1e  %%R1 %.1f  converged %s'
              % (k + 1, x, x * 27.211386245988, res.residuals[k], 100 * res.weights[k][0], res.converged[k]))
    print('   CCSDT iteration            %8.3f s' % it)
    print('   sigma vectors              %d (%.1f per root) in %d iterations, %d collapses; guess stage: %d singles-doubles vectors, %.1f s'
          % (nsig, nsig / len(e), res.niter, res.ncollapse, res.guess.nsigma, tr['guess']))
    print('   triples stage              %8.2f s = %.3f s per sigma vector = %.2f CCSDT iterations each' % (tr['iterations'], per, per / it))
    print('   prepare                    %8.2f s' % tr['prepare'])
    print('   resident memory            %.2f GB before, peak %.2f GB; plan %.2f GB (fixed %.2f GB; %s)'
          % ((base or 0) / 1e9, (peak.peak or 0) / 1e9, res.plan.need / 1e9, res.plan.fixed / 1e9, res.plan.text()))
    snap = timing.snapshot()
    print('   timers (wall s, cpu s, calls; per sigma vector of the triples stage in brackets where it applies):')
    for path in sorted(snap):
        w, c, calls = snap[path]
        if path[0].startswith('eom') and w > 0.004 * wall and len(path) <= 4:
            print('     %-52s %8.2f %9.2f %6d' % ('/'.join(path), w, c, calls))
    print('RESULT %s case=%s n3=%d nroots=%d nsigma=%d per_root=%.1f sigma_s=%.4f iter_s=%.4f ratio=%.3f peak_gb=%.3f plan_gb=%.3f wall=%.1f'
          % (what.replace(' ', '_'), case, sizes[2], len(e), nsig, nsig / len(e), per, it, per / it, (peak.peak or 0) / 1e9,
             res.plan.need / 1e9, wall), flush=True)
    return res


if 'ee' in parts:
    shifts = [int(x) for x in os.environ.get('E2_SHIFTS', '0,2,-2,4,-4').split(',')]
    n = int(os.environ.get('E2_NROOTS', 3))
    done = set()
    print('time reversal:', eom.time_reversal_symmetric(eris))
    for d in shifts:
        if -d in done and eom.time_reversal_symmetric(eris):
            print('\n== EE shift %+d: the time-reversed states of shift %+d (not computed)' % (d, -d))
            continue
        one('EE shift %+d' % d, lambda: cc.eom_ee(nroots=n, conv_tol=conv, shift=d), n)
        done.add(d)
for kind in ('ip', 'ea'):
    if kind in parts:
        n = int(os.environ.get('E2_IP_NROOTS', 2))
        sh = eom.ghost_short(eris, cc.t1, cc.t2, cc.t3, kind)
        done = set()
        for d in sorted(sh.labels(), key=lambda d: (abs(d), -d)):
            if -d in done and eom.time_reversal_symmetric(eris):
                continue
            if abs(d) > int(os.environ.get('E2_IP_MAXLABEL', 3)):
                continue
            one('%s label %+d' % (kind.upper(), d), lambda: getattr(cc, 'eom_' + kind)(nroots=n, conv_tol=conv, shift=d), n)
            done.add(d)
