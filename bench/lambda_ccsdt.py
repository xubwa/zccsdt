'''The left-hand side of CCSDT from the dense reference (zccsdt.cc.lambda_ref)
for a small molecule: Lambda, the CCSDT(Q)_Lambda energy, and the
one-particle density against a finite field.

usage: python bench/lambda_ccsdt.py MOLECULE BASIS [complex] [field]
    MOLECULE: bh, h2o;  BASIS: a name in bench/refs/GENBAS (STO-3G, 6-31G)
    complex: the two spin orbitals of every spatial orbital mixed by a
             random SU(2) matrix (complex integrals and amplitudes)
    field:   also the derivative of the CCSDT energy with respect to a
             dipole field along z at fixed orbitals, by central differences
             of CCSDT energies (dense solver), against trace(z d1)

RHF orbitals as spin orbitals, all electrons.  The numbers to compare:
MRCC calc=CCSDT(Q)_L (bench/refs/mrcc/*/MINP.ql.bohr with bench/mrcc_nr.sh)
and CFOUR CALC=CCSDT(Q)_L with CC_PROG=NCC (bench/refs/cfour/*/ZMAT.ql.bohr).
'''
import os
import sys
import time

import numpy as np
from pyscf import gto, scf

from zccsdt import eris as E
from zccsdt.cc import ccsdtq_ref as R, ccsdtq_terms as Q, lambda_ref as LR, solver
from zccsdt.genbas import read_genbas

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'refs')
GEOM = {'bh': 'B 0 0 0; H 0 0 1.2324', 'h2o': 'O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587'}


def spin_mixed(ghf, seed=4):
    '''The orbitals with the two spin orbitals of every spatial orbital
    mixed by a random SU(2) matrix (tests/test_ccsdtq_ref.py).'''
    rng = np.random.default_rng(seed)
    n = ghf.mo_coeff.shape[1]
    u = np.zeros((n, n), dtype=complex)
    order = np.argsort(ghf.mo_energy, kind='stable')
    for k in range(0, n, 2):
        p, q = order[k], order[k + 1]
        a = rng.standard_normal(4)
        a /= np.linalg.norm(a)
        u[np.ix_([p, q], [p, q])] = [[a[0] + 1j * a[1], a[2] + 1j * a[3]],
                                     [-a[2] + 1j * a[3], a[0] - 1j * a[1]]]
    return ghf.mo_coeff @ u


def main():
    name, basis = sys.argv[1:3]
    cplx, field = 'complex' in sys.argv[3:], 'field' in sys.argv[3:]
    atoms = sorted({a.split()[0] for a in GEOM[name].split(';')})
    mol = gto.M(atom=GEOM[name], verbose=0,
                basis={el: read_genbas(os.path.join(ROOT, 'GENBAS'), '%s:%s' % (el.upper(), basis)) for el in atoms})
    mf = scf.RHF(mol)
    mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-9
    mf.kernel()
    ghf = scf.addons.convert_to_ghf(mf)
    C = spin_mixed(ghf) if cplx else ghf.mo_coeff
    eris = E.from_scf(ghf, mo_coeff=C)
    fock, g, mo_e, nocc = R.integrals(eris)
    nvir = len(mo_e) - nocc
    print('%s / %s, %d + %d spin orbitals, %s;  E(SCF) = %.12f'
          % (name, basis, nocc, nvir, 'complex' if cplx else 'real', mf.e_tot), flush=True)
    t0 = time.time()
    r = solver.kernel(eris, 'ccsdt', solver.Options(conv_tol=1e-12, conv_tol_normt=1e-10))
    t3 = Q.expand(np.asarray(r.t3).reshape(len(Q.tuples(nocc, 3)), -1), nocc, nvir, 3)
    amps = [np.asarray(r.t1), np.asarray(r.t2), t3]
    print('CCSDT       %.12f   (%d iterations, %.0f s)' % (r.e_tot, r.ncycle, time.time() - t0), flush=True)
    e5, e6 = R.quadruples(amps[1], amps[2], g, mo_e, nocc)
    print('CCSDT[Q]    %.12f\nCCSDT(Q)    %.12f' % (r.e_tot + e5, r.e_tot + e6), flush=True)
    t0 = time.time()
    rl = LR.kernel(fock, g, mo_e, nocc, amps, conv_tol_normt=1e-10, verbose='-v' in sys.argv)
    print('Lambda: %d iterations, converged %s, %.0f s;  |l - conj(t)|: %.2e %.2e %.2e'
          % ((rl.ncycle, rl.converged, time.time() - t0)
             + tuple(abs(x - t.conj()).max() for x, t in zip(rl.lams, amps))), flush=True)
    l5, l6 = LR.quadruples_lambda(amps[1], amps[2], rl.lams[1], rl.lams[2], g, mo_e, nocc)
    print('CCSDT[Q]_L  %.12f\nCCSDT(Q)_L  %.12f   (correction %.6e, with conj(t) for lambda %.6e)'
          % (r.e_tot + l5, r.e_tot + l6, l6, e6), flush=True)
    # the density: the energy, and the dipole moment along z
    d1, d2 = LR.densities(amps, rl.lams, nocc)
    e_el = LR.energy(fock, g, nocc, d1, d2)
    print('E from the densities - E(CCSDT): %.2e' % abs(e_el + mol.energy_nuc() - r.e_tot))
    nao = mol.nao_nr()
    with mol.with_common_orig((0, 0, 0)):
        z = mol.intor('int1e_r')[2]
    zmo = C[:nao].conj().T @ z @ C[:nao] + C[nao:].conj().T @ z @ C[nao:]
    dz = np.einsum('pq,pq', zmo, d1)
    print('trace(z d1) = %.12f %+.1ej   (SCF: %.12f)' % (dz.real, dz.imag, np.trace(zmo[:nocc, :nocc]).real))
    if field:
        def e_ccsdt(eps):
            f = fock + eps * zmo
            x = R.kernel(f, g, mo_e, nocc, 'ccsdt', conv_tol=1e-13, conv_tol_normt=1e-11)
            assert x.converged
            return np.trace(f[:nocc, :nocc]).real - 0.5 * np.einsum('ijij', g[:nocc, :nocc, :nocc, :nocc]).real + x.e_corr
        for eps in (2e-3, 1e-3):
            e = {k: e_ccsdt(k * eps) for k in (-2, -1, 1, 2)}
            fd2 = (e[1] - e[-1]) / (2 * eps)
            fd4 = (8 * (e[1] - e[-1]) - (e[2] - e[-2])) / (12 * eps)
            print('field %.0e: dE/dF = %.12f (two points), %.12f (four points);  - trace(z d1) = %.2e'
                  % (eps, fd2, fd4, fd4 - dz.real), flush=True)


if __name__ == '__main__':
    main()
