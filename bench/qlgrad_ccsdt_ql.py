'''The relaxed one-particle density of CCSDT(Q)_Lambda from the dense
reference (zccsdt.cc.qlgrad_ref; docs/lambda.md, section 12) against finite
fields: the derivative of E(SCF) + E_corr(CCSDT) + E(Q)_L with respect to
the strength of a one-electron operator in the core Hamiltonian, the SCF
repeated and Lambda solved again at every strength.

usage: python bench/qlgrad_ccsdt_ql.py CASE [nfrozen ndrop] [steps=4e-3,2e-3,1e-3]
                                       [charge=X,Y,Z] [energies=q1,q2,..]
    CASE: MOLECULE:BASIS:KIND as in bench/qgrad_ccsdt_q.py (real, complex, x2c)
    nfrozen, ndrop: the lowest occupied spinors frozen, the highest virtual
          ones left out
    charge=X,Y,Z: the operator is the potential of a unit point charge at
          (X, Y, Z) bohr, -1 / |r - R|, in the place of the dipole operator z
          (what a program without finite fields can be given: MRCC's point
          charges); the energies printed then have the repulsion of the
          charge and the nuclei in them
    energies=..: only print the total energies at these strengths

The energies in the field are those of the production codes (cc.solver,
cc.lambda_opt, cc.quadruples_lambda); the analytic derivative is that of the
dense reference for the amplitudes and Lambda of the production solvers.
Four-point central differences (fourth order) at every step, and the
Richardson extrapolation of two consecutive steps, (16 f(h) - f(2h)) / 15.
'''
import os
import sys
import time

import numpy as np
from pyscf import gto, scf, x2c

from qgrad_ccsdt_q import GEOM, ROOT, spin_mixed
from zccsdt import eris as E
from zccsdt.cc import ccsdtq_terms as Q, lambda_opt, qgrad_ref as G, qlgrad_ref as QL, quadruples_lambda, solver
from zccsdt.genbas import read_genbas


def setup(name, basis, kind, charge=None):
    '''(make_scf(eps) -> (SCF with eps * V in the core Hamiltonian, its
    spinors), V in the AO basis of that SCF, the derivative of the nuclear
    repulsion with respect to eps).'''
    if kind == 'x2c':
        mol = gto.M(atom=GEOM[name], basis=basis, verbose=0)
    else:
        atoms = sorted({a.split()[0] for a in GEOM[name].split(';')})
        mol = gto.M(atom=GEOM[name], verbose=0, basis={
            el: read_genbas(os.path.join(ROOT, 'GENBAS'), '%s:%s' % (el.upper(), basis)) for el in atoms})
    suffix = '_spinor' if kind == 'x2c' else ''
    if charge is None:
        with mol.with_common_orig((0, 0, 0)):
            v1 = mol.intor('int1e_r' + suffix)[2]
        nuc = 0.0
    else:
        with mol.with_rinv_origin(charge):
            v1 = -mol.intor('int1e_rinv' + suffix)
        nuc = sum(mol.atom_charge(i) / np.linalg.norm(mol.atom_coord(i) - np.asarray(charge))
                  for i in range(mol.natm))
    v = v1 if kind == 'x2c' else np.kron(np.eye(2), v1)

    def make_scf(eps):
        mf = x2c.UHF(mol) if kind == 'x2c' else scf.RHF(mol)
        h0 = mf.get_hcore()
        mf.get_hcore = lambda *args: h0 + eps * v1
        mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-10, 200
        mf.kernel()
        assert mf.converged
        if kind == 'x2c':
            return mf, mf.mo_coeff
        ghf = scf.addons.convert_to_ghf(mf)
        h2 = np.kron(np.eye(2), h0 + eps * v1)
        ghf.get_hcore = lambda *args: h2
        return ghf, spin_mixed(ghf) if kind == 'complex' else ghf.mo_coeff
    return make_scf, v, nuc


OPTS = dict(conv_tol=1e-13, conv_tol_normt=1e-11, incore_complete=True)


def energies(make_scf, eps, nfrozen, ndrop):
    '''(E(SCF), E_corr(CCSDT), E(Q), E(Q)_L, SCF, spinors, eris, amplitudes,
    Lambda) in the field.'''
    mf, C = make_scf(eps)
    n = C.shape[1]
    assert (mf.mo_occ[:-1] >= mf.mo_occ[1:]).all()
    eris = E.from_scf(mf, mo_coeff=C[:, :n - ndrop], mo_occ=mf.mo_occ[:n - ndrop], frozen=nfrozen)
    r = solver.kernel(eris, 'ccsdt', solver.Options(**OPTS))
    assert r.converged
    rl = lambda_opt.kernel(eris, r.t1, r.t2, r.t3, solver.Options(**OPTS))
    assert rl.converged
    e_ql = quadruples_lambda.kernel(eris, r.t2, r.t3, rl.l2, rl.l3)[1]
    return mf.e_tot, float(np.real(r.e_corr)), e_ql, mf, C, eris, r, rl


def main():
    name, basis, kind = sys.argv[1].split(':')
    rest = [a for a in sys.argv[2:] if '=' not in a]
    opts = dict(a.split('=') for a in sys.argv[2:] if '=' in a)
    nfrozen, ndrop = (int(rest[0]), int(rest[1])) if rest else (0, 0)
    steps = [float(x) for x in opts.get('steps', '4e-3,2e-3,1e-3').split(',')]
    charge = [float(x) for x in opts['charge'].split(',')] if 'charge' in opts else None
    make_scf, v, nuc = setup(name, basis, kind, charge)
    if 'energies' in opts:
        for q in (float(x) for x in opts['energies'].split(',')):
            e = energies(make_scf, q, nfrozen, ndrop)[:3]
            print('strength %+.6f: E(SCF) %.12f  E(CCSDT) %.12f  E(CCSDT(Q)_L) %.12f'
                  % (q, e[0] + q * nuc, e[0] + e[1] + q * nuc, sum(e) + q * nuc))
        return
    t0 = time.time()
    e_scf, e_cc, e_ql, mf, C, eris, r, rl = energies(make_scf, 0.0, nfrozen, ndrop)
    o, n = eris.nocc, np.asarray(eris.fock).shape[0]
    print('%s / %s %s: %d + %d spin orbitals, %d frozen, %d dropped; operator: %s'
          % (name, basis, kind, o, n - o, nfrozen, ndrop, 'z' if charge is None else 'unit charge at %s bohr' % charge))
    print('E(SCF) %.12f  E_corr(CCSDT) %.12f  E(Q)_L %.6e   (%.1f s)' % (e_scf, e_cc, e_ql, time.time() - t0))

    def dense(x):
        return Q.expand(np.asarray(x).reshape(len(Q.tuples(o, 3)), -1), o, n - o, 3)
    amps, lams = [r.t1, r.t2, dense(r.t3)], [rl.l1, rl.l2, dense(rl.l3)]
    vmo = C.conj().T @ v @ C
    t0 = time.time()
    ref = G.relaxed_rdm1(mf, C, nfrozen, ndrop, with_q=False, amps=amps)
    a_cc = (np.einsum('pq,qp', vmo, ref.relaxed).real, np.einsum('pq,qp', vmo, ref.unrelaxed).real)
    print('CCSDT        relaxed %.12f  unrelaxed %.12f   (dense reference, %.1f s)' % (*a_cc, time.time() - t0))
    t0 = time.time()
    rel = QL.relaxed_rdm1(mf, C, nfrozen, ndrop, amps=amps, lams=lams)
    a_ql = (np.einsum('pq,qp', vmo, rel.relaxed).real, np.einsum('pq,qp', vmo, rel.unrelaxed).real)
    assert abs(rel.e_ql - e_ql) < 1e-12, (rel.e_ql, e_ql)
    t = rel.terms
    so, sv = slice(nfrozen, o + nfrozen), slice(o + nfrozen, n + nfrozen)
    print('CCSDT(Q)_L   relaxed %.12f  unrelaxed %.12f   (dense reference, %.1f s)' % (*a_ql, time.time() - t0))
    print('Im e %.1e; iterations of zeta %d, of lambda\' %d; |zeta| %.1e %.1e %.1e; |lambda\' - lambda| %.1e %.1e %.1e'
          % (t.e.imag, *t.ncycle, *[abs(z).max() for z in t.zeta], *[abs(a - b).max() for a, b in zip(t.lams, lams)]))
    print('orbital gradient among active occupied %.2e, among active virtual %.2e'
          % (abs(rel.gradient[so, so]).max(), abs(rel.gradient[sv, sv]).max()))
    if charge is not None:
        print('with the nuclei: CCSDT %.12f  CCSDT(Q)_L %.12f' % (a_cc[0] + nuc, a_ql[0] + nuc))
    fd = {}
    for h in steps:
        t0 = time.time()
        e = {k: energies(make_scf, k * h, nfrozen, ndrop)[:3] for k in (-2, -1, 1, 2)}
        d = [(8 * (e[1][x] - e[-1][x]) - (e[2][x] - e[-2][x])) / (12 * h) for x in range(3)]
        fd[h] = (d[0] + d[1], d[0] + d[1] + d[2])
        print('step %.0e: finite field CCSDT %.12f (analytic %+.1e)  CCSDT(Q)_L %.12f (analytic %+.1e)   (%.1f s)'
              % (h, fd[h][0], a_cc[0] - fd[h][0], fd[h][1], a_ql[0] - fd[h][1], time.time() - t0))
    for h2, h in zip(steps, steps[1:]):
        if abs(h2 / h - 2) < 1e-12:
            x = [(16 * fd[h][k] - fd[h2][k]) / 15 for k in (0, 1)]
            print('steps %.0e, %.0e extrapolated: CCSDT %.12f (analytic %+.1e)  CCSDT(Q)_L %.12f (analytic %+.1e)'
                  % (h2, h, x[0], a_cc[0] - x[0], x[1], a_ql[0] - x[1]))
    print('(Q)_L part of the relaxed derivative: analytic %.12f' % (a_ql[0] - a_cc[0]))


if __name__ == '__main__':
    main()
