'''EOM-EE-CCSD of zccsdt on the spinors and integrals of a CFOUR run (STD:
ABCDTYPE=STANDARD, up to xx2cvtran) against the roots that xx2ceomcc printed
in another run of the same input with ABCDTYPE=AOBASIS (AO):

    python bench/cfour/eom_ccsd_compare.py STD AO NROOTS [NGUESS]

bench/cfour/eom_ccsd.sh makes both directories.'''
import re, sys, numpy as np
from zccsdt import cfour
from zccsdt.cc import eom, solver
std, ao, nroots = sys.argv[1], sys.argv[2], int(sys.argv[3])
nguess = int(sys.argv[4]) if len(sys.argv) > 4 else nroots
eris = cfour.from_cfour(std)
rt = solver.kernel(eris, 'ccsd', solver.Options(conv_tol=1e-12, conv_tol_normt=1e-10))
out = open(ao + '/xx2cccsd.out').read()
ecf = float(re.findall(r'CCSD correlation energy\s+(-?\d+\.\d+)', out)[-1])
print('%d + %d spinors, %s; E_corr(CCSD) zccsdt %.12f  xx2cccsd %.12f  diff %.1e' % (eris.nocc, eris.fock.shape[0] - eris.nocc, eris.fock.dtype, float(np.real(rt.e_corr)), ecf, float(np.real(rt.e_corr)) - ecf))
res = eom.kernel(eris, rt.t1, rt.t2, None, nroots=nroots, nguess=nguess, conv_tol=1e-9, max_space=max(60, 6 * nroots), verbose='v' in sys.argv)
print('zccsdt roots:', ' '.join('%.10f' % x for x in res.e), ' sigma vectors', res.nsigma, 'iterations', res.niter)
eo = open(ao + '/xx2ceomcc.out').read()
ce = [float(x) for x in re.findall(r'eigenvalue in a\.u\.\s+(-?\d+\.\d+(?:E[-+]\d+)?)', eo)]
cyc = re.findall(r'converged in\s+(\d+)\s+cycles', eo)
for k, x in enumerate(ce):
    print('xx2ceomcc root %d  %.12f  cycles %s  nearest zccsdt root differs by %.1e' % (k + 1, x, cyc[k] if k < len(cyc) else '?', abs(res.e - x).min()))
