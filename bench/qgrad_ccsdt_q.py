'''The relaxed one-particle density of CCSDT(Q) from the dense reference
(zccsdt.cc.qgrad_ref) against finite fields: the derivative of
E(SCF) + E_corr(CCSDT) + E(Q) with respect to a dipole field along z in the
one-electron Hamiltonian, the SCF repeated in the field.

usage: python bench/qgrad_ccsdt_q.py CASE [nfrozen ndrop] [steps=4e-3,2e-3,1e-3]
    CASE: MOLECULE:BASIS:KIND, MOLECULE in h2o, bh, lih, hf; BASIS a name in
          bench/refs/GENBAS (STO-3G, 6-31G); KIND one of
            real      RHF orbitals as spin orbitals
            complex   every pair of them mixed by its own SU(2) matrix
            x2c       spinors of pyscf.x2c.UHF (PySCF's basis of that name)
    nfrozen, ndrop: the lowest occupied spinors frozen, the highest virtual
          ones left out

The energies in the field are those of the production codes (cc.solver,
cc.quadruples); the analytic derivative is that of the dense reference for
the amplitudes of the production solver, and next to it that of the
production codes (zccsdt.cc.quadruples_grad, lambda_opt, density_opt).
Four-point central differences
(fourth order) at every step, and the Richardson extrapolation of two
consecutive steps, (16 f(h) - f(2h)) / 15.
'''
import os
import sys
import time

import numpy as np
from pyscf import gto, scf, x2c

from zccsdt import eris as E
from zccsdt.cc import (ccsdtq_terms as Q, density_opt, lambda_opt, qgrad_ref as G, quadruples, quadruples_grad,
                       solver)
from zccsdt.genbas import read_genbas

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'refs')
GEOM = {'bh': 'B 0 0 0; H 0 0 1.2324', 'lih': 'Li 0 0 0; H 0 0 1.5957',
        'h2o': 'O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', 'hf': 'H 0 0 0; F 0 0 0.9168'}


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


def setup(name, basis, kind):
    '''(make_scf(eps) -> (SCF in the field, its spinors), z in the AO basis
    of that SCF).'''
    if kind == 'x2c':
        mol = gto.M(atom=GEOM[name], basis=basis, verbose=0)
        with mol.with_common_orig((0, 0, 0)):
            z = mol.intor('int1e_r_spinor')[2]
    else:
        atoms = sorted({a.split()[0] for a in GEOM[name].split(';')})
        mol = gto.M(atom=GEOM[name], verbose=0, basis={
            el: read_genbas(os.path.join(ROOT, 'GENBAS'), '%s:%s' % (el.upper(), basis)) for el in atoms})
        with mol.with_common_orig((0, 0, 0)):
            z1 = mol.intor('int1e_r')[2]
        z = np.kron(np.eye(2), z1)

    def make_scf(eps):
        mf = x2c.UHF(mol) if kind == 'x2c' else scf.RHF(mol)
        h0 = mf.get_hcore()
        v = z if kind == 'x2c' else z1
        mf.get_hcore = lambda *args: h0 + eps * v
        mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-10, 200
        mf.kernel()
        assert mf.converged
        if kind == 'x2c':
            return mf, mf.mo_coeff
        ghf = scf.addons.convert_to_ghf(mf)
        h2 = np.kron(np.eye(2), h0 + eps * z1)
        ghf.get_hcore = lambda *args: h2
        return ghf, spin_mixed(ghf) if kind == 'complex' else ghf.mo_coeff
    return make_scf, z


def energies(make_scf, eps, nfrozen, ndrop):
    '''(E(SCF), E_corr(CCSDT), E(Q), SCF, spinors, eris, amplitudes) in the field.'''
    mf, C = make_scf(eps)
    n = C.shape[1]
    assert (mf.mo_occ[:-1] >= mf.mo_occ[1:]).all()
    eris = E.from_scf(mf, mo_coeff=C[:, :n - ndrop], mo_occ=mf.mo_occ[:n - ndrop], frozen=nfrozen)
    r = solver.kernel(eris, 'ccsdt', solver.Options(conv_tol=1e-13, conv_tol_normt=1e-11, incore_complete=True))
    assert r.converged
    e_q = quadruples.kernel(eris, r.t2, r.t3)[1]
    return mf.e_tot, float(np.real(r.e_corr)), e_q, mf, C, eris, r


def main():
    name, basis, kind = sys.argv[1].split(':')
    rest = [a for a in sys.argv[2:] if '=' not in a]
    opts = dict(a.split('=') for a in sys.argv[2:] if '=' in a)
    nfrozen, ndrop = (int(rest[0]), int(rest[1])) if rest else (0, 0)
    steps = [float(x) for x in opts.get('steps', '4e-3,2e-3,1e-3').split(',')]
    make_scf, z = setup(name, basis, kind)
    t0 = time.time()
    e_scf, e_cc, e_q, mf, C, eris, r = energies(make_scf, 0.0, nfrozen, ndrop)
    o, n = eris.nocc, np.asarray(eris.fock).shape[0]
    print('%s / %s %s: %d + %d spin orbitals, %d frozen, %d dropped' % (name, basis, kind, o, n - o, nfrozen, ndrop))
    print('E(SCF) %.12f  E_corr(CCSDT) %.12f  E(Q) %.6e   (%.1f s)' % (e_scf, e_cc, e_q, time.time() - t0))
    t3 = Q.expand(np.asarray(r.t3).reshape(len(Q.tuples(o, 3)), -1), o, n - o, 3)
    zmo = C.conj().T @ z @ C
    out = {}
    for with_q in (False, True):
        t0 = time.time()
        rel = G.relaxed_rdm1(mf, C, nfrozen, ndrop, with_q=with_q, amps=[r.t1, r.t2, t3])
        out[with_q] = (np.einsum('pq,qp', zmo, rel.relaxed).real, np.einsum('pq,qp', zmo, rel.unrelaxed).real)
        if with_q:
            assert abs(rel.e_q - e_q) < 1e-12, (rel.e_q, e_q)
            so, sv = slice(nfrozen, o + nfrozen), slice(o + nfrozen, n + nfrozen)
            Go, Gv = rel.gradient[so, so], rel.gradient[sv, sv]
            print('orbital gradient among active occupied %.2e, among active virtual %.2e' % (abs(Go).max(), abs(Gv).max()))
        print('%-9s relaxed %.12f  unrelaxed %.12f   (dense reference, %.1f s)'
              % ('CCSDT(Q)' if with_q else 'CCSDT', *out[with_q], time.time() - t0))
    # the production codes: the pass over the quadruples, Lambda with the sources, the densities
    t0 = time.time()
    q = quadruples_grad.kernel(eris, r.t2, r.t3)
    rl = lambda_opt.kernel(eris, r.t1, r.t2, r.t3, solver.Options(conv_tol_normt=1e-11, incore_complete=True),
                           sources=(None, q.s2, q.s3))
    nall = C.shape[1]
    prod = density_opt.relaxed_rdm1(
        mf, eris, r.t1, r.t2, r.t3, rl.l1, rl.l2, rl.l3, C[:, nfrozen:nall - ndrop],
        frozen_occ=C[:, :nfrozen] if nfrozen else None, dropped_vir=C[:, nall - ndrop:] if ndrop else None,
        tol=1e-11, q=q)
    pv = np.einsum('pq,qp', zmo, prod.relaxed).real
    print('CCSDT(Q)  relaxed %.12f  (packed amplitudes, %.1f s): %+.1e off the dense reference; densities %.1e, '
          'orbital gradient %.1e' % (pv, time.time() - t0, pv - out[True][0], abs(prod.relaxed - rel.relaxed).max(),
                                     abs(prod.gradient - rel.gradient).max()))
    fd = {}
    for h in steps:
        t0 = time.time()
        e = {k: energies(make_scf, k * h, nfrozen, ndrop)[:3] for k in (-2, -1, 1, 2)}
        d = [(8 * (e[1][x] - e[-1][x]) - (e[2][x] - e[-2][x])) / (12 * h) for x in range(3)]
        fd[h] = (d[0] + d[1], d[0] + d[1] + d[2])
        print('step %.0e: finite field CCSDT %.12f (analytic %+.1e)  CCSDT(Q) %.12f (analytic %+.1e)   (%.1f s)'
              % (h, fd[h][0], out[False][0] - fd[h][0], fd[h][1], out[True][0] - fd[h][1], time.time() - t0))
    for h2, h in zip(steps, steps[1:]):
        if abs(h2 / h - 2) < 1e-12:
            x = [(16 * fd[h][k] - fd[h2][k]) / 15 for k in (0, 1)]
            print('steps %.0e, %.0e extrapolated: CCSDT %.12f (analytic %+.1e)  CCSDT(Q) %.12f (analytic %+.1e)'
                  % (h2, h, x[0], out[False][0] - x[0], x[1], out[True][0] - x[1]))
    print('(Q) part of the relaxed derivative: analytic %.12f' % (out[True][0] - out[False][0]))


if __name__ == '__main__':
    main()
