'''zccsdt.CC2 / CCSD / CC3 with eom_ee against the outputs of
bench/ccn/cfour_eom.sh (CFOUR: xvcc, xvee), H2O, on CFOUR's basis entries
and geometry:

    python bench/ccn/compare_cfour.py OUTDIR BASIS [NROOTS [CC2,CCSD,CC3]]

Prints the SCF and total energies of both programs and, for every
excitation energy of CFOUR (singlets), the nearest root of zccsdt among
NROOTS (default 14) of shift 0 of 2 m_s - the singlets and the M_S = 0
components of the triplets of the spin orbitals.
'''
import os
import re
import sys

import numpy as np
from pyscf import gto, scf

import zccsdt
from zccsdt.genbas import read_genbas

out_dir, basis = sys.argv[1], sys.argv[2]
nroots = int(sys.argv[3]) if len(sys.argv) > 3 else 14
names = sys.argv[4].split(',') if len(sys.argv) > 4 else ('CC2', 'CCSD', 'CC3')
genbas = os.path.join(os.environ.get('CFOUR_BIN', os.path.expanduser('~/apps/cfour-x2cso/bin')), '..', 'basis', 'GENBAS')
geom = '''O 0.000000000000000 0.000000000000000 0.000000000000000
H 0.000000000000000 1.430522676295752 1.109269235119691
H 0.000000000000000 -1.430522676295752 1.109269235119691'''
mol = gto.M(atom=geom, basis={x: read_genbas(genbas, x + ':' + basis) for x in ('O', 'H')}, unit='Bohr', verbose=0)
mf = scf.RHF(mol).run(conv_tol=1e-13, conv_tol_grad=1e-9)
print('E(SCF) = %.12f' % mf.e_tot)
for name in names:
    text = open(os.path.join(out_dir, 'h2o_%s_%s.out' % (basis, name))).read()
    escf = float(re.findall(r'E\(SCF\)=\s+(\S+)', text)[-1])
    ref = [float(x) for x in re.findall(r'Converged eigenvalue:\s+(\S+)', text)]
    etot = [float(x) for x in re.findall(r'Total EOMEE-\S+ electronic energy\s+(\S+)', text)]
    cc = getattr(zccsdt, name)(mf)
    cc.verbose = 0
    cc.conv_tol, cc.conv_tol_normt = 1e-12, 1e-10
    cc.kernel()
    e, res = cc.eom_ee(nroots=nroots, shift=0, conv_tol=1e-8, max_cycle=200)
    ground = etot[0] - ref[0]
    print('%s: E(SCF) CFOUR %.12f (zccsdt - CFOUR %.1e)   E: CFOUR %.12f  zccsdt %.12f  (%.1e)' % (
        name, escf, mf.e_tot - escf, ground, cc.e_tot, cc.e_tot - ground), flush=True)
    for w in ref:
        k = int(np.argmin(abs(e - w)))
        print('   CFOUR %.12f  zccsdt %.12f  diff %9.1e  (root %d, converged %s, %%R1 %.1f)' % (
            w, e[k], e[k] - w, k + 1, res.converged[k], 100 * res.weights[k][0] / res.weights[k].sum()), flush=True)
