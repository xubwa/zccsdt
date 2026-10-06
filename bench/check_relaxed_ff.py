'''The relaxed CCSD(T) dipole moment of zccsdt.zccsd against finite differences
of its own energy: E(SCF) + E(CCSD) + E(T) with the SCF repeated in the field
(central differences at eps and 2 eps, Richardson).  The check of the Lambda
equations, the densities and the Z-vector equations that needs no other
program; the field is the picture-change transformed dipole operator of the
SCF, added to its one-electron Hamiltonian.

usage: python bench/check_relaxed_ff.py MOLECULE BASIS [EPS]
  MOLECULE, BASIS and the environment as bench/profile_t_relaxed.py
  (ZC_FROZEN, ZC_VIRCUT, ZC_NUCMOD, ZC_HAM, ZC_SCF, ZC_MAXMEM); EPS: default
  2e-3.  With ZC_VIRCUT the virtual spinors above the cutoff are counted
  without the field, and as many of the highest are frozen in it.
environment: ZC_ANALYTIC: the analytic dipole (z, a.u.) of another run to
             compare with (then only the SCF is run without the field);
             without it the dipole is computed here first.
             ZC_FF_OPERATOR=FILE.npz:KEY: another operator in the spinor AO
             basis in the place of the dipole operator (one of CFOUR's, from
             bench/cfour/props.py, with ZC_GEOMETRY for its frame), e.g. the
             effective electric field; the analytic value is then its trace
             with the relaxed density of the file ZC_DENSITIES.
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
    vircut = float(os.environ.get('ZC_VIRCUT', np.inf))
    nhigh = []
    mol = spinors.molecule(mol_name, basis, max_memory=maxmem)

    def energy(field=None, dm0=None, scf_only=False):
        t0 = time.perf_counter()
        mf = spinors.scf(mol, 1e-12, 1e-9, field, dm0)
        if not nhigh:
            nhigh.append(int(np.count_nonzero((mf.mo_energy > vircut) & (mf.mo_occ == 0))))
        nmo = mf.mo_energy.size
        high = list(np.argsort(mf.mo_energy, kind='stable')[nmo - nhigh[0]:])
        mycc = DirectZCCSD(mf, frozen=list(range(frozen)) + sorted(high) if high else frozen)
        mycc.verbose, mycc.max_cycle = 0, 200
        mycc.conv_tol, mycc.conv_tol_normt = 1e-11, 1e-9
        if scf_only:
            return mf, mycc, np.array([mf.e_tot, 0.0, 0.0]), time.perf_counter() - t0
        mycc.kernel()
        e = np.array([mf.e_tot, mycc.e_corr, mycc.ccsd_t()])
        if not (mf.converged and mycc.converged):
            print('  not converged: SCF %s, CCSD %s' % (mf.converged, mycc.converged))
        return mf, mycc, e, time.perf_counter() - t0

    operator = os.environ.get('ZC_FF_OPERATOR')
    given = bool(os.environ.get('ZC_ANALYTIC')) or bool(operator)
    # (a doublet: the SCF of the run with the densities and this one have to
    # be the same Kramers partner for what is odd under time reversal)
    saved = np.load(os.environ['ZC_DENSITIES']) if os.environ.get('ZC_DENSITIES') else None
    mf0, cc0, e0, wall = energy(dm0=saved['scf'] if saved is not None else None, scf_only=given)
    print('%s/%s  %d spinors, %d electrons  frozen %d occupied, %d virtual  Hamiltonian %s  '
          '%s nuclei  SCF of %s'
          % (mol_name, basis, mol.nao_2c(), mol.nelectron, frozen, nhigh[0],
             os.environ.get('ZC_HAM', 'x2c1e'), 'Gaussian' if mol.nucmod else 'point',
             os.environ.get('ZC_SCF', 'pyscf')))
    print('without the field: E(SCF) %.10f' % e0[0] + ('' if given else
          '  E_corr(CCSD) %.10f  E(T) %.10f' % tuple(e0[1:])) + '  (%.0f s)' % wall)
    dm0 = mf0.make_rdm1()
    if operator:
        # the property is the derivative itself: no charge, no nuclear part
        path, key = operator.rsplit(':', 1)
        r, nuclear, sign, name = np.load(path)[key], 0.0, 1.0, key
        analytic = np.einsum('pq,qp', r, saved['relaxed']).real
        print('%s: trace with the SCF density %.10f, with the relaxed CCSD(T) density %.10f'
              % (key, np.einsum('pq,qp', r, dm0).real, analytic))
    else:
        r = rdm.dipole_integrals(cc0, True)[2]
        nuclear = np.einsum('i,ix->x', mol.atom_charges(), mol.atom_coords())[2]
        sign, name = -1.0, 'dipole z'
        if given:
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
        for sgn in (1, -1):
            _, _, ek, wall = energy(sgn * k * eps * r, dm0)
            print('  field %+.1e: E(SCF) %.10f  E_corr(CCSD) %.10f  E(T) %.10f  (%.0f s)'
                  % (sgn * k * eps, *ek, wall))
            e.append(ek)
        d[k] = (e[0] - e[1]) / (2 * k * eps)
    rich = (4 * d[1] - d[2]) / 3
    for i, part in enumerate(('SCF', 'CCSD correlation', '(T)')):
        print('d E(%s)/d eps: %.10f  (eps: %.10f, 2 eps: %.10f)'
              % (part, rich[i], d[1][i], d[2][i]))
    print('%s from finite differences (a.u.): SCF %.9f  CCSD %.9f  CCSD(T) %.9f'
          % (name, nuclear + sign * rich[0], nuclear + sign * rich[:2].sum(),
             nuclear + sign * rich.sum()))
    print('analytic relaxed CCSD(T) %.9f; analytic - finite differences: %.2e'
          % (analytic, analytic - (nuclear + sign * rich.sum())))


if __name__ == '__main__':
    main()
