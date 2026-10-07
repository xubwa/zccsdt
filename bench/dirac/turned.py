'''A linear molecule turned away from the z axis: the X2C spinors are complex
then, MRCC cannot take them, and the correlation energies must be those of
the molecule on the axis (bench/dirac/write_integrals.py, run_integrals.py
and mrcc_rel.sh) - the test of the complex code on the numbers of MRCC.

    python bench/dirac/turned.py HEAVY LIGHT DISTANCE BASIS FROZEN NVIR [ALPHA BETA]

HEAVY at the origin, LIGHT at DISTANCE (Angstrom) in the direction of the
polar angle BETA and the azimuth ALPHA (degrees; default 61, 37); FROZEN
occupied spinors, NVIR virtual spinors kept (the lowest).
'''
import sys

import numpy as np
from pyscf import gto, x2c

import zccsdt

heavy, light, dist, basis = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
frozen, nvir = int(sys.argv[5]), int(sys.argv[6])
alpha, beta = (float(x) for x in sys.argv[7:9]) if len(sys.argv) > 8 else (37.0, 61.0)
a, b = np.radians(alpha), np.radians(beta)
r = dist * np.array([np.cos(a) * np.sin(b), np.sin(a) * np.sin(b), np.cos(b)])
mol = gto.M(atom=[[heavy, (0, 0, 0)], [light, tuple(r)]], basis=basis, verbose=0)
mf = x2c.UHF(mol)
mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-8
mf.kernel()
if not mf.converged:
    raise SystemExit('the SCF did not converge')
print('%s at (%.5f, %.5f, %.5f) Angstrom;  E(SCF) = %.12f' % (light, *r, mf.e_tot))
order = np.argsort(mf.mo_energy)
nocc = int((mf.mo_occ > 0).sum())
cc = zccsdt.CCSDTQ(mf, frozen=list(order[:frozen]) + list(order[nocc + nvir:]))
cc.verbose = 4
cc.conv_tol, cc.conv_tol_normt = 1e-11, 1e-9
cc.kernel()
e_q = cc.ccsdt_q()
e_p = cc.ccsdtq_p()
eris = cc._eris
print('integrals %s: largest imaginary part of <ij||ab> %.3f;  T4 %s: largest imaginary part %.1e'
      % (np.asarray(eris.oovv).dtype, abs(np.asarray(eris.oovv).imag).max(), cc.t4.dtype, abs(cc.t4.imag).max()))
print('correlation energies:')
for name, e in (('CCSDT', cc.ccsdt_run.e_corr), ('CCSDT[Q]', cc.ccsdt_run.e_corr + cc.ccsdt_run.e_q5),
                ('CCSDT(Q)', cc.ccsdt_run.e_corr + e_q), ('CCSDTQ', cc.e_corr), ('CCSDTQ(P)', cc.e_corr + e_p)):
    print('  %-10s %.12f' % (name, e))
