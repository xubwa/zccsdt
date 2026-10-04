'''Dipole moment of the spinor CCSD of zccsdt.zccsd.DirectZCCSD, without and
with orbital relaxation (zccsd.zvector: small molecules), for the comparison
with CFOUR's xx2czvec / xsox2cprop after xx2cccsd, xx2chbar, xx2clcc and
xx2cccdenocbp (bench/cfour/slurm_x2ccc.sh with CFOUR_PROGS).

usage: python bench/check_dipole.py MOLECULE BASIS
X2C-1e spinors from PySCF (x2c.UHF), all electrons; the dipole integrals of
the 2c AO basis without a picture-change transformation; atomic units.
'''
import os
import sys

import numpy as np
from pyscf import gto, x2c

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profile_iter import GEOM, SPIN                                     # noqa: E402

from zccsdt.zccsd import DirectZCCSD                                    # noqa: E402


def main():
    mol_name, basis = sys.argv[1:3]
    mol = gto.M(atom=GEOM[mol_name], basis=basis, spin=SPIN.get(mol_name, 0), verbose=0)
    mf = x2c.UHF(mol).run(conv_tol=1e-11)
    mycc = DirectZCCSD(mf)
    mycc.verbose, mycc.conv_tol, mycc.conv_tol_normt = 0, 1e-11, 1e-8
    mycc.kernel()
    l1, l2 = mycc.solve_lambda()
    nuclear = np.einsum('i,ix->x', mol.atom_charges(), mol.atom_coords())
    with mol.with_common_orig((0, 0, 0)):
        r = mol.intor('int1e_r_spinor')
    scf = nuclear - np.einsum('xpq,qp->x', r, mf.make_rdm1()).real
    print('%s/%s  E(SCF) %.10f  E_corr(CCSD) %.10f  Lambda converged %s'
          % (mol_name, basis, mf.e_tot, mycc.e_corr, mycc.converged_lambda))
    fmt = '%-34s %14.8f %14.8f %14.8f'
    print(fmt % ('dipole (a.u.): SCF', *scf))
    print(fmt % ('CCSD, density at fixed spinors', *mycc.dip_moment(l1=l1, l2=l2, unit='AU')))
    print(fmt % ('CCSD, with orbital relaxation',
                 *mycc.dip_moment(l1=l1, l2=l2, relaxed=True, unit='AU')))


if __name__ == '__main__':
    main()
