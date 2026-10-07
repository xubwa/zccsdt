'''MO integrals of a PySCF calculation as MRCONEE and MDCINT (zccsdt.dirac),
for MRCC through its interface to DIRAC - the reference of the high-order
methods on spinors, on the same integrals as zccsdt:

    python bench/dirac/write_integrals.py KIND 'GEOMETRY' BASIS DIR [FROZEN [NVIR]]
    bench/dirac/mrcc_rel.sh DIR 'CCSDTQ(P)'
    python bench/dirac/run_integrals.py DIR            # zccsdt on the same files

KIND   rhf   an RHF reference as GHF spin orbitals (real; alpha unbarred)
       x2c   X2C spinors (pyscf.x2c.UHF) of a linear molecule on the z axis,
             adapted to m_j: real integrals
FROZEN, NVIR   frozen occupied spinors, virtual spinors kept (even numbers)
'''
import sys

import numpy as np
from pyscf import gto, scf, x2c

from zccsdt import dirac

kind, geom, basis, out = sys.argv[1:5]
frozen = int(sys.argv[5]) if len(sys.argv) > 5 else 0
nvir = int(sys.argv[6]) if len(sys.argv) > 6 else None
mol = gto.M(atom=geom, basis=basis, verbose=0)
if kind == 'rhf':
    mf = scf.addons.convert_to_ghf(scf.RHF(mol).run(conv_tol=1e-12, conv_tol_grad=1e-8))
elif kind == 'x2c':
    mf = x2c.UHF(mol)
    mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-8
    mf.kernel()
    e = np.sort(mf.mo_energy)
    print('X2C SCF converged: %s; largest splitting of a Kramers pair %.1e' % (mf.converged, abs(e[1::2] - e[::2]).max()))
    if not mf.converged:
        raise SystemExit('the SCF did not converge')
else:
    raise SystemExit(__doc__)
x = dirac.integrals(mf, frozen, nvir)
n = dirac.write(out, x)
print('E(SCF) = %.12f; %d spinors (%d occupied), %d integrals in %s/MRCONEE, MDCINT'
      % (mf.e_tot, len(x.energy), x.nelec.sum(), n, out))
print('energies of the occupied pairs:', np.array2string(x.energy[:x.nelec[0]], precision=5, max_line_width=200))
