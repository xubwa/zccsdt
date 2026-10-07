'''zccsdt on the MO integrals of a directory with MRCONEE and MDCINT (those
of DIRAC, or written by bench/dirac/write_integrals.py): CCSDT, (Q), CCSDTQ,
(P).

    python bench/dirac/run_integrals.py DIR [ccsdt(q) | ccsdtq | ccsdtq(p)]

ZC_MAXMEM (MB) sets max_memory.
'''
import os
import sys

import zccsdt
from zccsdt import dirac

d = sys.argv[1]
what = sys.argv[2].lower() if len(sys.argv) > 2 else 'ccsdtq'
eris = dirac.from_dirac(os.path.join(d, 'MRCONEE'), os.path.join(d, 'MDCINT'))
full = what.startswith('ccsdtq')
cc = (zccsdt.GCCSDTQ if full else zccsdt.GCCSDT).from_integrals(eris, verbose=4)
cc.conv_tol, cc.conv_tol_normt = 1e-11, 1e-9
if os.environ.get('ZC_MAXMEM'):
    cc.max_memory = float(os.environ['ZC_MAXMEM'])
cc.kernel()
run3 = cc.ccsdt_run if full else cc
e_q = cc.ccsdt_q()
if what == 'ccsdtq(p)':
    cc.ccsdtq_p()
print('correlation energies: CCSDT %.12f  CCSDT[Q] %.12f  CCSDT(Q) %.12f' % (
    run3.e_corr, run3.e_corr + run3.e_q5, run3.e_corr + e_q)
    + ('  CCSDTQ %.12f' % cc.e_corr if full else '')
    + ('  CCSDTQ(P) %.12f' % (cc.e_corr + cc.e_p) if what == 'ccsdtq(p)' else ''))
print('E(CCSDT)     = %.12f' % run3.e_tot)
print('E(CCSDT[Q])  = %.12f' % (run3.e_tot + run3.e_q5))
print('E(CCSDT(Q))  = %.12f' % (run3.e_tot + e_q))
if full:
    print('E(CCSDTQ)    = %.12f' % cc.e_tot)
if what == 'ccsdtq(p)':
    print('E(CCSDTQ(P)) = %.12f' % cc.e_tot_p)
