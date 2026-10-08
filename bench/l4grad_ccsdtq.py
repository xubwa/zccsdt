'''The relaxed one-particle density of CCSDTQ from the dense reference
(zccsdt.cc.lambda4_ref; docs/lambda.md, section 13) against finite fields:
the derivative of E(SCF) + E_corr(CCSDTQ) with respect to a dipole field
along z in the one-electron Hamiltonian, the SCF repeated in the field.

usage: python bench/l4grad_ccsdtq.py CASE [nfrozen ndrop] [steps=4e-3,2e-3,1e-3] [dense=1] [zmat=1]
    CASE: MOLECULE:BASIS:KIND as in bench/qgrad_ccsdt_q.py (real, complex, x2c);
          nh3 besides its molecules
    nfrozen, ndrop: the lowest occupied spinors frozen, the highest virtual
          ones left out
    dense=1: the amplitudes at zero field from the dense reference
          (ccsdtq_ref.kernel) too, not from the production code
    zmat=1: only print the input of CFOUR for the case (CALC=CCSDTQ,
          CC_PROG=NCC, PROPS=FIRST_ORDER, the geometry in bohr as PySCF has it)

The energies in the field are those of the production code
(zccsdt.cc.ccsdtq); the analytic derivative is that of the dense reference:
Lambda from the tables of lambda4_terms for the amplitudes of the
production solver, densities, the orbital gradient and the Z-vector
equations of the CCSD line.  Four-point central differences (fourth order)
at every step, and the Richardson extrapolation of two consecutive steps,
(16 f(h) - f(2h)) / 15.

With four active electrons CCSDTQ is the full CI of the active spinors:
the unrelaxed densities are then compared with those of
pyscf.fci.fci_dhf_slow on the same integrals (fci()).
'''
import os
import sys
import time

import numpy as np

from pyscf import gto, scf, x2c

from qgrad_ccsdt_q import GEOM, ROOT, spin_mixed
from zccsdt.genbas import read_genbas
from zccsdt import eris as E
from zccsdt.cc import ccsdtq, ccsdtq_ref as R, ccsdtq_terms as Q, lambda4_ref as L4

OPTS = dict(conv_tol=1e-13, conv_tol_normt=1e-11, incore_complete=True)
# (C3v to all digits: CFOUR turns the molecule into its principal axes, and the dipole has to stay on one)
GEOM['nh3'] = ('N 0 0 0.1173; H 0 %.15f -0.2737; H %.15f %.15f -0.2737; H %.15f %.15f -0.2737'
               % (0.9377, 0.9377 * 0.75 ** 0.5, -0.9377 / 2, -0.9377 * 0.75 ** 0.5, -0.9377 / 2))

ZMAT = '''%s
%s

*CFOUR(BASIS=%s,REF=RHF,CC_PROG=NCC
COORDINATES=CARTESIAN,UNITS=BOHR,SYMMETRY=OFF
FROZEN_CORE=%s,SCF_CONV=11,CC_CONV=11,CC_MAXCYC=200,LINEQ_CONV=11
SCF_MAXCYC=200,MEMORY_SIZE=2,MEM_UNIT=GB
CALC=CCSDTQ,PROPS=FIRST_ORDER)
'''


def setup(name, basis, kind):
    '''(make_scf(eps) -> (SCF with eps * z in the core Hamiltonian, its
    spinors), z in the AO basis of that SCF): bench/qlgrad_ccsdt_ql.setup
    for the dipole operator, with an SCF that starts again from its last
    density when it has not reached conv_tol = 1e-13 (the X2C SCF of BH
    stops short of it in some runs and not in others, at the same
    strength).'''
    if kind == 'x2c':
        mol = gto.M(atom=GEOM[name], basis=basis, verbose=0)
    else:
        atoms = sorted({a.split()[0] for a in GEOM[name].split(';')})
        mol = gto.M(atom=GEOM[name], verbose=0, basis={
            el: read_genbas(os.path.join(ROOT, 'GENBAS'), '%s:%s' % (el.upper(), basis)) for el in atoms})
    with mol.with_common_orig((0, 0, 0)):
        v1 = mol.intor('int1e_r' + ('_spinor' if kind == 'x2c' else ''))[2]
    v = v1 if kind == 'x2c' else np.kron(np.eye(2), v1)

    def make_scf(eps):
        mf = x2c.UHF(mol) if kind == 'x2c' else scf.RHF(mol)
        h0 = mf.get_hcore()
        mf.get_hcore = lambda *args: h0 + eps * v1
        mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-10, 200
        mf.kernel()
        for attempt in range(5):
            if mf.converged:
                break
            print('(SCF at strength %g not converged in %d cycles: again from its density)' % (eps, mf.max_cycle))
            mf.kernel(dm0=mf.make_rdm1())
        assert mf.converged
        if kind == 'x2c':
            return mf, mf.mo_coeff
        ghf = scf.addons.convert_to_ghf(mf)
        h2 = np.kron(np.eye(2), h0 + eps * v1)
        ghf.get_hcore = lambda *args: h2
        return ghf, spin_mixed(ghf) if kind == 'complex' else ghf.mo_coeff
    return make_scf, v


def fci(h, g, nelec):
    '''(E, d1[p,q] = <p+ q>, d2[p,q,r,s] = <p+ q+ s r>) of the lowest state
    of ``nelec`` electrons in the spin orbitals of h[p,q] and <pq||rs>, by
    the diagonalization of the Hamiltonian matrix among all determinants:
    the sigma vectors and density matrices of pyscf.fci.fci_dhf_slow
    (determinants of spinors, complex integrals) with (pq|rs) = 1/2
    <pr||qs>, whose antisymmetrized part is <pr||qs>.  For a few hundred
    determinants and a state that is not degenerate.'''
    from pyscf.fci import cistring, fci_dhf_slow as F
    n = h.shape[0]
    eri = 0.5 * g.transpose(0, 2, 1, 3)
    h2e = F.absorb_h1e(h, eri, n, nelec, 0.5)
    nd = cistring.num_strings(n, nelec)
    H = np.array([F.contract_2e(h2e, x, n, nelec) for x in np.eye(nd, dtype=h2e.dtype)]).T
    assert abs(H - H.conj().T).max() < 1e-12
    e, c = np.linalg.eigh(H)
    assert e[1] - e[0] > 1e-6
    d1, d2 = F.make_rdm12(c[:, 0].copy(), n, nelec)     # <p+ q>, <p+ r+ s q>
    return e[0], d1, d2.transpose(0, 2, 1, 3)


def energies(make_scf, eps, nfrozen, ndrop):
    '''(E(SCF), E_corr(CCSDTQ), SCF, spinors, eris, amplitudes) in the field.'''
    mf, C = make_scf(eps)
    n = C.shape[1]
    assert (mf.mo_occ[:-1] >= mf.mo_occ[1:]).all()
    eris = E.from_scf(mf, mo_coeff=C[:, :n - ndrop], mo_occ=mf.mo_occ[:n - ndrop], frozen=nfrozen)
    r = ccsdtq.kernel(eris, ccsdtq.Options(**OPTS))
    assert r.converged
    return mf.e_tot, float(np.real(r.e_corr)), mf, C, eris, r


def main():
    name, basis, kind = sys.argv[1].split(':')
    rest = [a for a in sys.argv[2:] if '=' not in a]
    opts = dict(a.split('=') for a in sys.argv[2:] if '=' in a)
    nfrozen, ndrop = (int(rest[0]), int(rest[1])) if rest else (0, 0)
    steps = [float(x) for x in opts.get('steps', '4e-3,2e-3,1e-3').split(',')]
    make_scf, v = setup(name, basis, kind)
    if opts.get('zmat'):
        mol = make_scf(0.0)[0].mol
        print(ZMAT % ('%s ccsdtq_%s_props' % (name, basis.lower().replace('-', '')),
                      '\n'.join('%s %.15f %.15f %.15f' % (mol.atom_symbol(i), *mol.atom_coord(i))
                                for i in range(mol.natm)), basis, 'ON' if nfrozen else 'OFF'))
        return
    t0 = time.time()
    e_scf, e_cc, mf, C, eris, r = energies(make_scf, 0.0, nfrozen, ndrop)
    o, n = eris.nocc, np.asarray(eris.fock).shape[0]
    print('%s / %s %s: %d + %d spin orbitals, %d frozen, %d dropped'
          % (name, basis, kind, o, n - o, nfrozen, ndrop))
    print('E(SCF) %.12f  E_corr(CCSDTQ) %.12f   (production, %.1f s)' % (e_scf, e_cc, time.time() - t0))

    def dense(x, k):
        return Q.expand(np.asarray(x).reshape(len(Q.tuples(o, k)), -1), o, n - o, k)
    amps = [r.t1, r.t2, dense(r.t3, 3), dense(r.t4, 4)]
    fock, g, mo_e, _ = R.integrals(eris)
    if opts.get('dense'):
        t0 = time.time()
        rd = R.kernel(fock, g, mo_e, o, 'ccsdtq', conv_tol=1e-13, conv_tol_normt=1e-11)
        assert rd.converged
        print('dense CCSDTQ: E_corr %+.1e off, amplitudes %s off   (%d iterations, %.1f s)'
              % (rd.e_corr - e_cc, ' '.join('%.1e' % abs(a - b).max() for a, b in zip(rd.amps, amps)),
                 rd.ncycle, time.time() - t0))
        amps = rd.amps
    t0 = time.time()
    rl = L4.kernel(fock, g, mo_e, o, amps, conv_tol_normt=1e-11, verbose=bool(opts.get('verbose')))
    assert rl.converged
    t1 = time.time()
    print('Lambda: %d iterations, %.1f s (%.1f s each); |lambda - conj(t)| %s; |lambda| %s'
          % (rl.ncycle, t1 - t0, (t1 - t0) / rl.ncycle,
             ' '.join('%.1e' % abs(a - np.conj(b)).max() for a, b in zip(rl.lams, amps)),
             ' '.join('%.1e' % abs(a).max() for a in rl.lams)))
    arrays = L4._arrays(fock, g, o, amps, rl.lams)
    res = [L4.L.residual(k, arrays, o) for k in (1, 2, 3)]
    res.append(L4.residual4_unique(arrays, o, blocked=amps[3].size >= L4.BLOCKED))
    del arrays
    d1, d2 = L4.densities(amps, rl.lams, o)
    h = fock - np.einsum('piqi->pq', g[:, :o, :, :o])
    lag = L4.energy(fock, g, o, d1, d2)
    e0 = np.trace(h[:o, :o]) + 0.5 * np.einsum('ijij', g[:o, :o, :o, :o])
    G = L4.one_index(h, g, d1, d2)
    print('residuals of Lambda %s; Lagrangian from the densities - E %.1e; trace(d1) - N %.1e'
          % (' '.join('%.1e' % abs(x).max() for x in res), abs(lag - e0 - e_cc), abs(np.trace(d1) - o)))
    print('one-index transformed Lagrangian (active spinors): occupied-occupied %.1e, virtual-virtual %.1e, '
          'excitations %.1e, de-excitations (the orbital gradient) %.1e; dressed table - bare %.1e'
          % (abs(G[:o, :o]).max(), abs(G[o:, o:]).max(), abs(G[o:, :o]).max(), abs(G[:o, o:]).max(),
             abs(L4.orbital_gradient(fock, g, o, amps, rl.lams, dressed=True) + G[:o, o:].T).max()))
    if o == 4 and n <= 14:
        t0 = time.time()
        e, f1, f2 = fci(h, g, o)
        print('full CI of the active spinors: E_corr - E_corr(CCSDTQ) %+.1e; d1 - d1(FCI) %.1e, d2 - d2(FCI) %.1e; '
              'd1 - d1^+ %.1e   (%.1f s)' % ((e - e0).real - e_cc, abs(d1 - f1).max(), abs(d2 - f2).max(),
                                             abs(d1 - d1.conj().T).max(), time.time() - t0))
    vmo = C.conj().T @ v @ C
    t0 = time.time()
    rel = L4.relaxed_rdm1(mf, C, nfrozen, ndrop, amps=amps, lams=rl.lams)
    a = (np.einsum('pq,qp', vmo, rel.relaxed).real, np.einsum('pq,qp', vmo, rel.unrelaxed).real)
    print('CCSDTQ  relaxed %.12f  unrelaxed %.12f  (SCF %.12f)   (dense reference, %.1f s); |z| %.1e'
          % (*a, np.trace(vmo[:o + nfrozen, :o + nfrozen]).real, time.time() - t0, abs(rel.z).max()))
    mol = mf.mol
    znuc = sum(mol.atom_charge(i) * mol.atom_coord(i)[2] for i in range(mol.natm))
    print('dipole moment along z, atomic units: relaxed %.10f  unrelaxed %.10f' % (znuc - a[0], znuc - a[1]))
    fd = {}
    for hh in steps:
        t0 = time.time()
        e = {k: energies(make_scf, k * hh, nfrozen, ndrop)[:2] for k in (-2, -1, 1, 2)}
        d = [(8 * (e[1][x] - e[-1][x]) - (e[2][x] - e[-2][x])) / (12 * hh) for x in range(2)]
        fd[hh] = d[0] + d[1]
        print('step %.0e: finite field %.12f (analytic %+.1e)   (%.1f s)' % (hh, fd[hh], a[0] - fd[hh], time.time() - t0))
    for h2, hh in zip(steps, steps[1:]):
        if abs(h2 / hh - 2) < 1e-12:
            x = (16 * fd[hh] - fd[h2]) / 15
            print('steps %.0e, %.0e extrapolated: %.12f (analytic %+.1e)' % (h2, hh, x, a[0] - x))


if __name__ == '__main__':
    main()
