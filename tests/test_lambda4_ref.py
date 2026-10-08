'''The dense reference of the left-hand side of CCSDTQ (zccsdt.cc.lambda4_ref):

- against the exact left-hand state of a model Hamiltonian in a small Fock
  space (tests/fockspace.py: the Lambda equations as a linear system with the
  matrix of exp(-T) H exp(T), densities as expectation values);
- against the full CI where CCSDTQ is exact (four electrons: LiH), with real
  and with complex spin orbitals: densities, and no orbital gradient;
- the relaxed dipole derivative against finite fields with the SCF repeated,
  all electrons and with frozen and dropped spinors (BH / STO-3G).

Longer checks (H2O, NH3, X2C spinors, CFOUR): bench/l4grad_ccsdtq.py,
docs/lambda.md section 13.'''
import numpy as np
import pytest
from pyscf import fci, scf

import fockspace as F
from test_ccsdtq_ref import _scf, _spin_mixed
from test_lambda_ref import _model
from zccsdt import eris as E
from zccsdt.cc import ccsdtq, ccsdtq_ref as R, ccsdtq_terms as Q, lambda4_ref as LR

OPTS = dict(conv_tol=1e-13, conv_tol_normt=1e-11, incore_complete=True)


# ------------------------------------------------------------ model Hamiltonian

@pytest.fixture(scope='module')
def model():
    '''Five electrons in ten spin orbitals (CCSDTQ is not the full CI),
    complex Hermitian, f not diagonal: converged CCSDTQ, the exact left
    state of the Fock space, and Lambda of the tables.'''
    no, nv = 5, 5
    f, v = _model(no, nv)
    r = R.kernel(f, v, f.diagonal().real, no, 'ccsdtq', conv_tol=1e-13, conv_tol_normt=1e-11)
    assert r.converged
    e = np.trace(f[:no, :no]) - 0.5 * np.einsum('ijij', v[:no, :no, :no, :no]) + R.energy(r.amps[0], r.amps[1], f, v, no)
    ex = F.Left(F.Fock(no, nv), f, v, r.amps)
    assert abs(ex.residuals()).max() < 1e-11 and abs(ex.energy() - e) < 1e-12
    assert abs(r.amps[3]).max() > 1e-5 and abs(r.amps[0]).max() > 0.05
    lam = ex.solve()
    assert abs(ex.lambda_residual(lam)).max() < 1e-12
    rl = LR.kernel(f, v, f.diagonal().real, no, r.amps, conv_tol_normt=1e-12)
    assert rl.converged
    return no, nv, f, v, r.amps, e, ex, lam, rl.lams


def test_lambda_is_the_left_eigenvector(model):
    '''The iterations of the term table give the solution of the linear
    system <0| (1 + Lambda) (Hbar - E) |mu> = 0 among the singly to
    quadruply excited determinants; it is not conj(t).  The residual of
    Lambda_4 by occupied quadruples is that of the full arrays.'''
    no, nv, f, v, amps, e, ex, lam, lams = model
    want = ex.tensors(lam)
    for k in range(4):
        assert abs(lams[k] - want[k]).max() < 1e-11
        assert abs(want[k] - amps[k].conj()).max() > 1e-5
    arrays = LR._arrays(f, v, no, amps, [x + 0.1 * t.conj() for x, t in zip(lams, amps)])
    a, b = LR.residual4_unique(arrays, no), LR.residual4_unique(arrays, no, blocked=True)
    assert abs(a).max() > 1e-6 and abs(a - b).max() < 1e-14
    assert abs(Q.expand(a, no, nv, 4) - LR.L.residual(4, arrays, no)).max() < 1e-14


def test_densities_are_expectation_values(model):
    '''d1 and d2 of the tables are <0| (1 + Lambda) exp(-T) p+ q exp(T) |0>
    and the same of p+ q+ s r; they give the energy; d1 has the number of
    electrons and is not Hermitian (by 6e-7 here: CCSDTQ of five electrons
    is close to the full CI, whose density is).'''
    no, nv, f, v, amps, e, ex, lam, lams = model
    d1, d2 = LR.densities(amps, lams, no)
    assert abs(d1 - ex.rdm1(lam)).max() < 1e-11
    assert abs(d2 - ex.rdm2(lam)).max() < 1e-11
    assert abs(LR.energy(f, v, no, d1, d2) - e) < 1e-11
    assert abs(np.trace(d1) - no) < 1e-12
    assert abs(d1 - d1.conj().T).max() > 1e-7
    assert abs(np.einsum('pqrq->pr', d2) - (no - 1) * d1).max() < 1e-11


def test_orbital_gradient(model):
    '''For converged t and lambda the Lagrangian does not change with
    rotations among occupied or among virtual spin orbitals nor with the
    excitations a+_a a_i; the de-excitation half is the table of the
    dressed basis, and half of it is the orbital gradient of
    zccsdt.zccsd.zvector with the Hermitian densities.'''
    from zccsdt.zccsd import zvector
    no, nv, f, v, amps, e, ex, lam, lams = model
    d1, d2 = LR.densities(amps, lams, no)
    h = f - np.einsum('piqi->pq', v[:, :no, :, :no])
    G = LR.one_index(h, v, d1, d2)
    gnh = LR.orbital_gradient(f, v, no, amps, lams)
    scale = abs(gnh).max()
    assert scale > 1e-7                                 # (6.4e-7: what pentuples would take away)
    assert abs(G[:no, :no]).max() < 1e-12 and abs(G[no:, no:]).max() < 1e-12
    assert abs(G[no:, :no]).max() < 1e-12
    assert abs(LR.orbital_gradient(f, v, no, amps, lams, dressed=True) - gnh).max() < 1e-12
    dm1, dm2 = LR.pyscf_rdms(d1, d2)
    eri = 0.5 * v.transpose(0, 2, 1, 3)
    Gz = zvector.orbital_gradient(h, eri, dm1, dm2)
    assert abs(Gz[no:, :no] - 0.5 * gnh).max() < 1e-12


# ----------------------------------------------------------------- the full CI

def _fci_spin_orbital(mf, ghf):
    '''(E, d1[p,q] = <p+ q>, d2[p,q,r,s] = <p+ q+ s r>) of the full CI of
    the RHF ``mf`` (pyscf.fci.direct_spin1) in the spin orbitals of
    convert_to_ghf, each of which is one orbital with one spin.'''
    nmo = mf.mo_coeff.shape[1]
    nao = mf.mol.nao_nr()
    cis = fci.direct_spin1.FCI(mf.mol)
    cis.conv_tol, cis.max_cycle = 1e-14, 200
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    from pyscf import ao2mo
    e, c = cis.kernel(h1, ao2mo.kernel(mf.mol, mf.mo_coeff), nmo, mf.mol.nelec, tol=1e-14, lindep=1e-15)
    (da, db), (daa, dab, dbb) = cis.make_rdm12s(c, nmo, mf.mol.nelec)      # <p+ q>; <p+ r+ s q> as [p,q,r,s]
    C = ghf.mo_coeff
    spin = np.array([0 if abs(C[:nao, p]).max() > 0.5 else 1 for p in range(2 * nmo)])
    orb = np.array([int(np.argmax(abs(mf.mo_coeff.T @ mf.get_ovlp() @ C[spin[p] * nao:(spin[p] + 1) * nao, p])))
                    for p in range(2 * nmo)])
    sign = np.array([np.sign((mf.mo_coeff.T @ mf.get_ovlp() @ C[spin[p] * nao:(spin[p] + 1) * nao, p])[orb[p]])
                     for p in range(2 * nmo)])
    n = 2 * nmo
    d1 = np.zeros((n, n))
    chem = np.zeros((n, n, n, n))                       # <p+ r+ s q>
    two = {(0, 0): daa, (0, 1): dab, (1, 0): dab.transpose(2, 3, 0, 1), (1, 1): dbb}
    ix = [np.where(spin == s)[0] for s in (0, 1)]
    for s in (0, 1):
        d1[np.ix_(ix[s], ix[s])] = (da, db)[s][np.ix_(orb[ix[s]], orb[ix[s]])]
        for t in (0, 1):
            x = two[s, t][np.ix_(orb[ix[s]], orb[ix[s]], orb[ix[t]], orb[ix[t]])]
            chem[np.ix_(ix[s], ix[s], ix[t], ix[t])] = x
            if s != t:                                  # <p(s)+ r(t)+ s(s) q(t)> = -<p+ r+ q s>
                chem[np.ix_(ix[s], ix[t], ix[t], ix[s])] = -x.transpose(0, 3, 2, 1)
    d1 = d1 * sign[:, None] * sign[None, :]
    chem = np.einsum('pqrs,p,q,r,s->pqrs', chem, sign, sign, sign, sign)
    return e + mf.mol.energy_nuc(), d1, chem.transpose(0, 2, 1, 3)


@pytest.mark.parametrize('cplx', [False, True])
def test_four_electrons_full_ci(cplx):
    '''LiH / STO-3G, 4 + 8 spin orbitals: CCSDTQ is the full CI, so
    <0| (1 + Lambda) exp(-T) is the bra of the normalized ground state -
    d1 and d2, not made Hermitian, are the density matrices of
    pyscf.fci.direct_spin1 (in the spin orbitals; with complex spin
    orbitals rotated like them) and Hermitian; the Lagrangian is stationary
    in every orbital rotation, the Z-vector vanishes and the relaxed
    density is the unrelaxed one.'''
    mf = _scf('lih')
    ghf = scf.addons.convert_to_ghf(mf)
    e_fci, f1, f2 = _fci_spin_orbital(mf, ghf)
    C = ghf.mo_coeff
    if cplx:
        C, u = _spin_mixed(ghf)
        f1 = u.T @ f1 @ u.conj()                        # p'+ = sum_p u[p,p'] p+
        f2 = np.einsum('pqrs,pt,qu,rv,sw->tuvw', f2, u, u, u.conj(), u.conj())
    eris = E.from_scf(ghf, mo_coeff=C)
    fock, g, mo_e, no = R.integrals(eris)
    nv = len(mo_e) - no
    assert (abs(g.imag).max() > 0.01) == cplx
    r = ccsdtq.kernel(eris, ccsdtq.Options(**OPTS))
    assert r.converged and abs(mf.e_tot + r.e_corr - e_fci) < 1e-11
    amps = [r.t1, r.t2] + [Q.expand(np.asarray(x).reshape(len(Q.tuples(no, k)), -1), no, nv, k)
                           for k, x in ((3, r.t3), (4, r.t4))]
    rl = LR.kernel(fock, g, mo_e, no, amps, conv_tol_normt=1e-11)
    assert rl.converged
    assert all(abs(a - b.conj()).max() > 1e-5 for a, b in zip(rl.lams, amps))
    d1, d2 = LR.densities(amps, rl.lams, no)
    assert abs(d1 - f1).max() < 1e-9 and abs(d2 - f2).max() < 1e-9
    assert abs(d1 - d1.conj().T).max() < 1e-10
    h = fock - np.einsum('piqi->pq', g[:, :no, :, :no])
    assert abs(LR.one_index(h, g, d1, d2)).max() < 1e-10
    assert abs(LR.orbital_gradient(fock, g, no, amps, rl.lams, dressed=True)).max() < 1e-10
    rel = LR.relaxed_rdm1(ghf, C, amps=amps, lams=rl.lams)
    assert abs(rel.z).max() < 1e-10 and abs(rel.relaxed - rel.unrelaxed).max() < 1e-10
    assert abs(rel.unrelaxed - f1.T).max() < 1e-9


# ---------------------------------------------------------------- finite fields

def _field(name, cplx):
    '''run(eps) -> (SCF in the field eps * z, its spin orbitals), and z in
    the AO basis of the spin orbitals.'''
    mol = _scf(name).mol
    with mol.with_common_orig((0, 0, 0)):
        z = mol.intor('int1e_r')[2]
    h0 = scf.RHF(mol).get_hcore()

    def run(eps):
        mf = scf.RHF(mol)
        mf.get_hcore = lambda *args: h0 + eps * z
        mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-10
        mf.kernel()
        ghf = scf.addons.convert_to_ghf(mf)
        h2 = np.kron(np.eye(2), h0 + eps * z)
        ghf.get_hcore = lambda *args: h2
        return ghf, _spin_mixed(ghf)[0] if cplx else ghf.mo_coeff
    return run, np.kron(np.eye(2), z)


@pytest.mark.parametrize('cplx,nfrozen,ndrop', [(False, 0, 0), (True, 2, 2)])
def test_relaxed_dipole_against_finite_fields(cplx, nfrozen, ndrop):
    '''BH / STO-3G, 6 + 6 spin orbitals (CCSDTQ is not the full CI), and
    with the 1s spin orbitals frozen and the two highest virtual ones left
    out, complex: the relaxed density of lambda4_ref gives the derivative
    of E(SCF) + E_corr(CCSDTQ) with respect to a field along z with the SCF
    repeated in the field (energies of the production code; central
    differences of fourth order at two steps, extrapolated).  The unrelaxed
    one is another number.'''
    run, z = _field('bh', cplx)

    def energy(eps):
        mf, C = run(eps)
        n = C.shape[1]
        eris = E.from_scf(mf, mo_coeff=C[:, :n - ndrop], mo_occ=mf.mo_occ[:n - ndrop], frozen=nfrozen)
        r = ccsdtq.kernel(eris, ccsdtq.Options(**OPTS))
        assert r.converged
        return mf.e_tot + float(np.real(r.e_corr)), mf, C, eris, r
    _, mf, C, eris, r = energy(0.0)
    no, nv = eris.nocc, np.asarray(eris.fock).shape[0] - eris.nocc
    amps = [r.t1, r.t2] + [Q.expand(np.asarray(x).reshape(len(Q.tuples(no, k)), -1), no, nv, k)
                           for k, x in ((3, r.t3), (4, r.t4))]
    rel = LR.relaxed_rdm1(mf, C, nfrozen, ndrop, amps=amps)
    zmo = C.conj().T @ z @ C
    relaxed = np.einsum('pq,qp', zmo, rel.relaxed).real
    unrelaxed = np.einsum('pq,qp', zmo, rel.unrelaxed).real
    fd = []
    for eps in (2e-3, 1e-3):
        e = {k: energy(k * eps)[0] for k in (-2, -1, 1, 2)}
        fd.append((8 * (e[1] - e[-1]) - (e[2] - e[-2])) / (12 * eps))
    assert abs(fd[1] - relaxed) < 2e-9
    assert abs((16 * fd[1] - fd[0]) / 15 - relaxed) < 2e-10
    assert abs(unrelaxed - relaxed) > (1e-6 if nfrozen else 5e-8)
