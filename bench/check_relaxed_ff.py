'''The relaxed CCSD(T) dipole moment of zccsdt.zccsd against finite differences
of its own energy: E(SCF) + E(CCSD) + E(T) with the SCF repeated in the field
(central differences at eps and 2 eps, Richardson).  The check of the Lambda
equations, the densities and the Z-vector equations that needs no other
program; the field is the picture-change transformed dipole operator of the
SCF, added to its one-electron Hamiltonian.

usage: python bench/check_relaxed_ff.py MOLECULE BASIS [EPS]
  MOLECULE, BASIS and the environment as bench/profile_t_relaxed.py
  (ZC_FROZEN, ZC_NUCMOD, ZC_HAM, ZC_SCF, ZC_MAXMEM); EPS: default 2e-3
environment: ZC_ANALYTIC: the analytic dipole (z, a.u.) of another run to
             compare with; without it the dipole is computed here first.
'''
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spinors                                                          # noqa: E402

from zccsdt.zccsd import DirectZCCSD, rdm                               # noqa: E402


def main():
    mol_name, basis = sys.argv[1:3]
    eps = float(sys.argv[3]) if len(sys.argv) > 3 else 2e-3
    maxmem = float(os.environ.get('ZC_MAXMEM', 24000))
    frozen = int(os.environ.get('ZC_FROZEN', 0))
    mol = spinors.molecule(mol_name, basis, max_memory=maxmem)

    def energy(field=None, dm0=None):
        t0 = time.perf_counter()
        mf = spinors.scf(mol, 1e-12, 1e-9, field, dm0)
        mycc = DirectZCCSD(mf, frozen=frozen)
        mycc.verbose, mycc.max_cycle = 0, 200
        mycc.conv_tol, mycc.conv_tol_normt = 1e-11, 1e-9
        mycc.kernel()
        e = np.array([mf.e_tot, mycc.e_corr, mycc.ccsd_t()])
        if not (mf.converged and mycc.converged):
            print('  not converged: SCF %s, CCSD %s' % (mf.converged, mycc.converged))
        return mf, mycc, e, time.perf_counter() - t0

    mf0, cc0, e0, wall = energy()
    o, v = cc0.t1.shape
    print('%s/%s  nocc=%d nvir=%d  frozen %d  Hamiltonian %s  %s nuclei  SCF of %s'
          % (mol_name, basis, o, v, frozen, os.environ.get('ZC_HAM', 'x2c1e'),
             'Gaussian' if mol.nucmod else 'point', os.environ.get('ZC_SCF', 'pyscf')))
    print('E(SCF) %.10f  E_corr(CCSD) %.10f  E(T) %.10f  (%.0f s)' % (*e0, wall))
    r = rdm.dipole_integrals(cc0, True)[2]
    nuclear = np.einsum('i,ix->x', mol.atom_charges(), mol.atom_coords())[2]
    dm0 = mf0.make_rdm1()
    if 'ZC_ANALYTIC' in os.environ:
        analytic = float(os.environ['ZC_ANALYTIC'])
    else:
        t0 = time.perf_counter()
        analytic = cc0.dip_moment(with_t=True, relaxed=True, unit='AU')[2]
        print('analytic relaxed CCSD(T) dipole (z, a.u.): %.9f  (%.0f s)'
              % (analytic, time.perf_counter() - t0))
    del cc0
    d = {}
    for k in (1, 2):
        e = []
        for sign in (1, -1):
            _, _, ek, wall = energy(sign * k * eps * r, dm0)
            print('  field %+.4f: E(SCF) %.10f  E_corr(CCSD) %.10f  E(T) %.10f  (%.0f s)'
                  % (sign * k * eps, *ek, wall))
            e.append(ek)
        d[k] = (e[0] - e[1]) / (2 * k * eps)
    rich = (4 * d[1] - d[2]) / 3
    for i, name in enumerate(('SCF', 'CCSD correlation', '(T)')):
        print('d E(%s)/d eps: %.10f  (eps: %.10f, 2 eps: %.10f)'
              % (name, rich[i], d[1][i], d[2][i]))
    print('dipole z from finite differences (a.u.): SCF %.9f  CCSD %.9f  CCSD(T) %.9f'
          % (nuclear - rich[0], nuclear - rich[:2].sum(), nuclear - rich.sum()))
    print('analytic relaxed CCSD(T) %.9f; analytic - finite differences: %.2e'
          % (analytic, analytic - (nuclear - rich.sum())))


if __name__ == '__main__':
    main()
