'''The one-shot quadruples correction of EOM-CCSDT excitation energies
(GCCSDT.eom_quadruples; docs/eom.md, section 14) against converged EOM-CC4:

    python -u bench/eom_q_bench.py h2o              H2O / 6-31G, RHF as spin orbitals, all electrons (10 + 16):
                                                    the M_S = 0 component of the lowest triplet and the lowest singlet
    python -u bench/eom_q_bench.py hbr [SHIFT] [N]  HBr / cc-pVDZ-DK, X2C spinors, 28 frozen (8 + 28): N roots of
                                                    the label shift SHIFT (default 0, 2)

(the cases of docs/ccn.md, section 12).  Per state:

  A  from CCSDT: omega(EOM-CCSDT) + L (fold + t4) R + W . rho, everything at the CCSDT amplitudes;
  B  from the CC4 ground state: the Jacobian of CCSDT at the CC4 amplitudes (the excitation energies "without
     the T4 terms", (i) of docs/native.md 19.3), its left and right vectors, + L (fold + t4) R;
     and the same with conj(R) in the place of L (the Rayleigh quotient of the right-hand vector, (ii) there);
  the converged EOM-CC4 of zccsdt.cc.cc4fly.eom on the same integrals.
'''
import copy
import sys
import time

import numpy as np
from pyscf import gto, scf, x2c

import zccsdt
from zccsdt.cc import cc4fly

which = sys.argv[1]
conv = 1e-8
if which == 'h2o':
    import os
    from zccsdt.genbas import read_genbas
    genbas = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'refs', 'GENBAS')
    mol = gto.M(atom='O 0 0 0; H 0 1.430522676295752 1.109269235119691; H 0 -1.430522676295752 1.109269235119691',
                basis={x: read_genbas(genbas, x + ':6-31G') for x in ('O', 'H')}, unit='Bohr', verbose=0)
    mf = scf.addons.convert_to_ghf(scf.RHF(mol).run(conv_tol=1e-13, conv_tol_grad=1e-9))
    frozen, shift, nroots, nall = None, 0, 2, 5
else:
    mol = gto.M(atom='Br 0 0 0; H 0 0 1.4145', basis='cc-pvdz-dk', verbose=0)
    mf = x2c.UHF(mol)
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-8
    mf.kernel()
    frozen = 28
    shift = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    nroots = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    nall = 14


def make(cls):
    cc = cls(mf, frozen=frozen) if frozen is not None else cls(mf)
    cc.verbose = 0
    cc.conv_tol, cc.conv_tol_normt = 1e-12, 1e-10
    return cc


t0 = time.time()
cc = make(zccsdt.CCSDT)
cc.kernel()
print('CCSDT  E = %.12f  (%.1f s)' % (cc.e_tot, time.time() - t0), flush=True)
t0 = time.time()
e, res = cc.eom_ee(nroots=nroots, shift=shift, conv_tol=conv)
t_right = time.time() - t0
t0 = time.time()
cc.eom_left(conv_tol=conv)
t_left = time.time() - t0
t0 = time.time()
cc.eom_density(response=True)
t_resp = time.time() - t0
t0 = time.time()
qa = cc.eom_quadruples()
t_q = time.time() - t0
print('EOM-CCSDT: right %.1f s, left %.1f s, response vectors and densities %.1f s, quadruples %.1f s (rho %.1f s; per state %s)'
      % (t_right, t_left, t_resp, t_q, qa.seconds_rho, ', '.join('%.1f' % c.seconds for c in qa.states)), flush=True)

t0 = time.time()
c4 = make(zccsdt.CC4)
c4.kernel()
print('CC4    E = %.12f  (%.1f s);  E(CC4) - E(CCSDT) = %.10f,  Lambda . rho = %.10f'
      % (c4.e_tot, time.time() - t0, c4.e_tot - cc.e_tot, qa.e_ground), flush=True)
assert np.size(c4.t3) == np.size(cc.t3)
# B: the Jacobian of CCSDT at the CC4 amplitudes
ci = make(zccsdt.CCSDT)
ci._eris, ci.t1, ci.t2, ci.t3 = c4._eris, c4.t1, c4.t2, c4.t3
ci._sorted = getattr(c4, '_sorted', None)
import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    ei, ri = ci.eom_ee(nroots=nroots, shift=shift, conv_tol=conv)
    ci.eom_left(conv_tol=conv)
    qb = ci.eom_quadruples(ground=False)
# converged EOM-CC4: all states of the packed block (the states of a shift among them)
t0 = time.time()
e4_eris = c4._eris
if which != 'h2o' and shift:
    e4_eris = copy.copy(c4._eris)
    e4_eris.no_sectors = True
e4, sol = cc4fly.eom(e4_eris, c4.t1, c4.t2, c4.t3, nroots=nall if (which == 'h2o' or shift) else nroots + 1, conv_tol=conv,
                     max_memory=c4.max_memory)
print('EOM-CC4 converged (%.1f s): %s' % (time.time() - t0, ' '.join('%.9f' % x for x in e4)), flush=True)

print('\n%s%s: Hartree' % (which, '' if which == 'h2o' else ' shift %+d' % shift))
print('root  omega(CCSDT)   A: fold      t4        ground    estimate      | (i) at CC4 T  B: fold     t4       estimate      '
      'Rayleigh (ii) | converged CC4 | error A    error B    error (ii)  error (i)  error CCSDT')
for k in range(len(e)):
    a, b = qa.states[k], qb.states[k]
    # the converged root next to the estimate (the states are close and ordered alike)
    conv4 = e4[np.argmin(abs(e4 - b.estimate))]
    ray = b.omega + b.rayleigh
    print('%3d  %.9f  %+.3e %+.3e %+.3e  %.9f  |  %.9f  %+.3e %+.3e  %.9f  %.9f  |  %.9f  | %+.2e  %+.2e  %+.2e  %+.2e  %+.2e'
          % (k + 1, a.omega, a.fold, a.t4, a.ground, a.estimate, b.omega, b.fold, b.t4, b.estimate, ray, conv4,
             a.estimate - conv4, b.estimate - conv4, ray - conv4, b.omega - conv4, a.omega - conv4))
    print('     ground-state shift: A %+.4e against (i) - omega(CCSDT) = %+.4e;  T4 effect beyond (i): converged %+.4e, B %+.4e (%.0f %%), '
          'Rayleigh %+.4e (%.0f %%)' % (a.ground, b.omega - a.omega, conv4 - b.omega, b.fold + b.t4,
                                      100 * (b.fold + b.t4) / (conv4 - b.omega), b.rayleigh, 100 * b.rayleigh / (conv4 - b.omega)))
