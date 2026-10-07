'''The dense reference of the left-hand side of CCSDT (zccsdt.cc.lambda_ref):

- against the exact left-hand state of a model Hamiltonian in a small Fock
  space (tests/fockspace.py: the Lambda equations as a linear system with the
  matrix of exp(-T) H exp(T), densities as expectation values) and against
  derivatives of the CCSDT energy by finite differences;
- against the CCSD code where T3 and Lambda_3 are left out (pyscf.cc
  .gccsd_lambda, gccsd_rdm; zccsdt.zccsd.rdm, zvector);
- for molecules: the dipole derivative of the CCSDT energy at fixed
  orbitals, and CCSDT(Q)_Lambda against MRCC and CFOUR, with real and with
  complex spin orbitals.  Inputs and energies of the reference runs:
  bench/refs.'''
import json
import os

import numpy as np
import pytest
from pyscf import cc, gto, scf

import fockspace as F
from test_ccsdtq_ref import _scf, _spin_mixed
from zccsdt import eris as E
from zccsdt.cc import ccsdtq_ref as R, lambda_ref as LR
from zccsdt.genbas import read_genbas

ROOT = os.path.join(os.path.dirname(__file__), '..', 'bench', 'refs')
REFS = json.load(open(os.path.join(ROOT, 'reference_energies.json')))


# ------------------------------------------------------------ model Hamiltonian

def _model(no, nv, seed=1):
    '''A complex Hermitian Hamiltonian with a gap, not diagonal in f (so
    that t1 is not small), weakly correlated: (f, <pq||rs>).'''
    n = no + nv
    rng = np.random.default_rng(seed)
    v = 0.02 * F.hermitian_integrals(rng, n)
    e = np.concatenate([-1.0 - 0.3 * rng.random(no), 1.0 + 0.3 * rng.random(nv)])
    x = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return np.diag(e) + 0.05 * (x + x.conj().T), v


def _ccsdt(f, v, no):
    '''(amplitudes, total energy) of CCSDT: the energy is complex (the
    functional is holomorphic in the Hamiltonian, and a complex Hermitian
    one without a symmetry has no real coupled-cluster energy).'''
    r = R.kernel(f, v, f.diagonal().real, no, 'ccsdt', conv_tol=1e-13, conv_tol_normt=1e-11)
    assert r.converged
    e_ref = np.trace(f[:no, :no]) - 0.5 * np.einsum('ijij', v[:no, :no, :no, :no])
    return r.amps, e_ref + R.energy(r.amps[0], r.amps[1], f, v, no)


@pytest.fixture(scope='module')
def model():
    no, nv = 4, 5
    f, v = _model(no, nv)
    amps, e = _ccsdt(f, v, no)
    ex = F.Left(F.Fock(no, nv), f, v, amps)
    assert abs(ex.residuals()).max() < 1e-11 and abs(ex.energy() - e) < 1e-12
    assert abs(e.imag) > 1e-7 and abs(amps[2]).max() > 1e-3 and abs(amps[0]).max() > 0.05
    lam = ex.solve()
    assert abs(ex.lambda_residual(lam)).max() < 1e-12
    r = LR.kernel(f, v, f.diagonal().real, no, amps, conv_tol_normt=1e-12)
    assert r.converged
    return no, nv, f, v, amps, e, ex, lam, r.lams


def test_lambda_is_the_left_eigenvector(model):
    '''The iterations of the term table give the solution of the linear
    system <0| (1 + Lambda) (Hbar - E) |mu> = 0 in the determinant basis;
    it is not conj(t).'''
    no, nv, f, v, amps, e, ex, lam, lams = model
    want = ex.tensors(lam)
    for k in range(3):
        assert abs(lams[k] - want[k]).max() < 1e-11
        assert abs(want[k] - amps[k].conj()).max() > 1e-3


def test_densities_are_expectation_values(model):
    '''d1 and d2 of the tables are <0| (1 + Lambda) exp(-T) p+ q exp(T) |0>
    and the same of p+ q+ s r; they give the energy; d1 has the number of
    electrons and is not Hermitian.'''
    no, nv, f, v, amps, e, ex, lam, lams = model
    d1, d2 = LR.densities(amps, lams, no)
    assert abs(d1 - ex.rdm1(lam)).max() < 1e-11
    assert abs(d2 - ex.rdm2(lam)).max() < 1e-11
    assert abs(LR.energy(f, v, no, d1, d2) - e) < 1e-11
    assert abs(np.trace(d1) - no) < 1e-12
    assert abs(d1 - d1.conj().T).max() > 1e-6
    # the two-particle density has the one-particle one in it
    assert abs(np.einsum('pqrq->pr', d2) - (no - 1) * d1).max() < 1e-11


def test_densities_are_energy_derivatives(model):
    '''dE/dx of the CCSDT energy, converged again for every Hamiltonian at
    fixed orbitals, for a one-electron and for a two-electron perturbation
    (complex Hermitian): the trace with d1, and 1/4 of that with d2.
    Central differences of fourth order; with the step halved the
    difference falls by 2^4, and what is left at the smaller step is below
    2e-10 of the derivative (steps 2e-4 and 1e-4).'''
    no, nv, f, v, amps, e, ex, lam, lams = model
    n = no + nv
    d1, d2 = LR.densities(amps, lams, no)
    rng = np.random.default_rng(9)
    x = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    one = x + x.conj().T
    two = F.hermitian_integrals(rng, n)
    fock2 = np.einsum('piqi->pq', two[:, :no, :, :no])          # h stays: f changes with <pq||rs>
    for (df, dv), analytic in (((one, 0 * two), np.einsum('pq,pq', one, d1)),
                               ((fock2, two), 0.25 * np.einsum('pqrs,pqrs', two, d2))):
        err = []
        for eps in (2e-4, 1e-4):
            ee = {k: _ccsdt(f + k * eps * df, v + k * eps * dv, no)[1] for k in (-2, -1, 1, 2)}
            fd = (8 * (ee[1] - ee[-1]) - (ee[2] - ee[-2])) / (12 * eps)
            err.append(abs(fd - analytic))
        assert abs(analytic.imag) > 1e-6                        # (a complex number is compared)
        assert err[1] < 2e-10 * abs(analytic) and err[1] < err[0]
        assert 10 < err[0] / err[1] < 24 or err[0] < 1e-10 * abs(analytic)


def test_orbital_gradient(model):
    '''For converged t and lambda the Lagrangian does not change with
    rotations among occupied or among virtual spin orbitals nor with the
    excitations a+_a a_i; what is left, the de-excitation half, is the table
    of the dressed basis, and half of it is the orbital gradient of
    zccsdt.zccsd.zvector with the Hermitian densities.'''
    from zccsdt.zccsd import zvector
    no, nv, f, v, amps, e, ex, lam, lams = model
    d1, d2 = LR.densities(amps, lams, no)
    h = f - np.einsum('piqi->pq', v[:, :no, :, :no])
    G = LR.one_index(h, v, d1, d2)
    gnh = LR.orbital_gradient(f, v, no, amps, lams)
    scale = abs(gnh).max()
    assert scale > 1e-6
    assert abs(G[:no, :no]).max() < 1e-7 * scale and abs(G[no:, no:]).max() < 1e-7 * scale
    assert abs(G[no:, :no]).max() < 1e-7 * scale
    assert abs(LR.orbital_gradient(f, v, no, amps, lams, dressed=True) - gnh).max() < 1e-8 * scale
    dm1, dm2 = LR.pyscf_rdms(d1, d2)
    eri = 0.5 * v.transpose(0, 2, 1, 3)                 # (pq|rs) with (pq|rs) - (ps|rq) = <pr||qs>
    Gz = zvector.orbital_gradient(h, eri, dm1, dm2)
    assert abs(Gz[no:, :no] - 0.5 * gnh).max() < 1e-8 * scale
    assert abs(Gz[:no, no:] + 0.5 * gnh.conj().T).max() < 1e-8 * scale
    # in general (amplitudes that solve nothing): Gz[y,x] = -1/2 (G[x,y] - conj(G[y,x]))
    rng = np.random.default_rng(2)
    amps = [F.complex_antisym(rng, no, nv, k, s) for k, s in ((1, 0.3), (2, 0.2), (3, 0.1))]
    lams = [F.complex_antisym(rng, no, nv, k, s) for k, s in ((1, 0.3), (2, 0.2), (3, 0.1))]
    d1, d2 = LR.densities(amps, lams, no)
    G = LR.one_index(h, v, d1, d2)
    Gz = zvector.orbital_gradient(h, eri, *LR.pyscf_rdms(d1, d2))
    assert abs(Gz + 0.5 * (G.T - G.conj())).max() < 1e-12 * abs(G).max()
    # and the Hermitian densities give the real part of the Lagrangian
    lag = LR.energy(f, v, no, d1, d2)
    dm1, dm2 = LR.pyscf_rdms(d1, d2)
    assert abs(np.einsum('pq,qp', h, dm1) + 0.5 * np.einsum('pqrs,pqrs', eri, dm2) - lag.real) < 1e-12 * abs(lag)


# ------------------------------------------------------------------ CCSD limit

@pytest.mark.parametrize('cplx', [False, True])
def test_reduces_to_ccsd(cplx):
    '''Without T3 and Lambda_3 the tables are those of CCSD: lambda1 and
    lambda2 of pyscf.cc.gccsd_lambda make the residuals of rank 1 and 2
    vanish, and the densities are those of pyscf.cc.gccsd_rdm - the
    one-particle one before it is made Hermitian is gamma1 of
    zccsdt.zccsd.rdm block by block.  H2O / STO-3G, also with complex spin
    orbitals.'''
    from zccsdt.zccsd import rdm
    mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis='sto-3g', verbose=0)
    ghf = scf.addons.convert_to_ghf(scf.RHF(mol).run(conv_tol=1e-13))
    if cplx:
        ghf.mo_coeff = _spin_mixed(ghf)[0]
    mycc = cc.GCCSD(ghf)
    mycc.conv_tol, mycc.conv_tol_normt = 1e-12, 1e-10
    mycc.kernel()
    mycc.conv_tol_normt = 1e-10
    l1, l2 = mycc.solve_lambda()
    fock, g, mo_e, no = R.integrals(E.from_scf(ghf, mo_coeff=ghf.mo_coeff))
    assert (abs(g.imag).max() > 0.01) == cplx
    nv = len(mo_e) - no
    zero = np.zeros((no,) * 3 + (nv,) * 3)
    amps, lams = [mycc.t1, mycc.t2, zero], [l1, l2, zero]
    res = LR.residuals(lams, amps, fock, g, no)
    assert abs(res[0]).max() < 1e-8 and abs(res[1]).max() < 1e-8
    assert abs(res[2]).max() > 1e-4                             # (Lambda_3 would not vanish)
    d1, d2 = LR.densities(amps, lams, no)
    dm1, dm2 = LR.pyscf_rdms(d1, d2)
    assert abs(dm1 - mycc.make_rdm1()).max() < 1e-9
    assert abs(dm2 - mycc.make_rdm2()).max() < 1e-9
    doo, dov, dvo, dvv = rdm.gamma1(mycc.t1, mycc.t2, l1, l2)   # D[p,q] = <q+ p>, without the reference
    assert abs(d1[:no, :no].T - np.eye(no) - doo).max() < 1e-12
    assert abs(d1[no:, :no].T - dov).max() < 1e-12              # <a+ i> = l1[i,a]
    assert abs(d1[:no, no:].T - dvo).max() < 1e-12
    assert abs(d1[no:, no:].T - dvv).max() < 1e-12


# -------------------------------------------------------------------- molecules

def _molecule(name, cplx):
    '''(scf, <z> in the spin orbitals, fock, <pq||rs>, orbital energies,
    nocc) for bh or h2o with the STO-3G basis of bench/refs/GENBAS.'''
    if name == 'bh':
        mf = _scf('bh')
    else:
        basis = {el: read_genbas(os.path.join(ROOT, 'GENBAS'), '%s:STO-3G' % el) for el in 'OH'}
        mf = scf.RHF(gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis=basis, verbose=0))
        mf.run(conv_tol=1e-13)
    ghf = scf.addons.convert_to_ghf(mf)
    C = _spin_mixed(ghf)[0] if cplx else ghf.mo_coeff
    nao = mf.mol.nao_nr()
    with mf.mol.with_common_orig((0, 0, 0)):
        z = mf.mol.intor('int1e_r')[2]
    zmo = C[:nao].conj().T @ z @ C[:nao] + C[nao:].conj().T @ z @ C[nao:]
    return (mf, zmo) + R.integrals(E.from_scf(ghf, mo_coeff=C))


@pytest.mark.parametrize('name,cplx', [('bh', False), ('bh', True), ('h2o', False), ('h2o', True)])
def test_ccsdt_q_lambda_against_mrcc_and_cfour(name, cplx):
    '''CCSDT(Q)_Lambda of MRCC (calc=CCSDT(Q)_L) for BH and H2O with STO-3G,
    and of CFOUR's ncc for H2O, which also prints the part of Lambda_2,
    CCSDT[Q]_L; the same energies with complex spin orbitals.  With conj(t)
    in the place of lambda the correction is CCSDT(Q), which is another
    number.'''
    mf, zmo, fock, g, mo_e, nocc = _molecule(name, cplx)
    ref = REFS[name + '_sto3g']
    assert (abs(g.imag).max() > 0.01) == cplx
    rt = R.kernel(fock, g, mo_e, nocc, 'ccsdt', conv_tol=1e-12, conv_tol_normt=1e-10)
    rl = LR.kernel(fock, g, mo_e, nocc, rt.amps, conv_tol_normt=1e-10)
    assert rt.converged and rl.converged
    e = mf.e_tot + rt.e_corr
    assert abs(e - ref['recommended']['e_ccsdt']) < 1e-10
    t2, t3 = rt.amps[1:]
    e5, e6 = LR.quadruples_lambda(t2, t3, rl.lams[1], rl.lams[2], g, mo_e, nocc)
    assert abs(e + e6 - ref['recommended']['e_ccsdt(q)_l']) < 1e-10
    assert abs(e + e6 - ref['mrcc']['bohr']['e_ccsdt(q)_l']) < 1e-10
    if name == 'h2o':
        ncc = ref['cfour']['NCC']['bohr']
        assert abs(e5 - ncc['e_ccsdt[q]_l_corr']) < 2e-11 and abs(e6 - ncc['e_ccsdt(q)_l_corr']) < 2e-11
    # conj(t) for lambda: (Q) itself
    q5, q6 = LR.quadruples_lambda(t2, t3, t2.conj(), t3.conj(), g, mo_e, nocc)
    assert abs(q6 - R.quadruples(t2, t3, g, mo_e, nocc)[1]) < 1e-14
    assert abs(e + q6 - ref['recommended']['e_ccsdt(q)']) < 1e-10
    assert abs(q6 - e6) > 1e-7


@pytest.mark.parametrize('name,cplx', [('bh', True), ('h2o', False)])
def test_dipole_derivative(name, cplx):
    '''The derivative of the CCSDT energy with respect to a field along z
    in the Hamiltonian, the orbitals fixed (unrelaxed), is the trace of z
    with the one-particle density: central differences of fourth order of
    energies converged in the field.  It is not the derivative that conj(t)
    for lambda would give.'''
    mf, zmo, fock, g, mo_e, nocc = _molecule(name, cplx)

    def energy(eps):
        f = fock + eps * zmo
        r = R.kernel(f, g, mo_e, nocc, 'ccsdt', conv_tol=1e-13, conv_tol_normt=1e-11)
        assert r.converged
        return np.trace(f[:nocc, :nocc]).real - 0.5 * np.einsum('ijij', g[:nocc, :nocc, :nocc, :nocc]).real + r.e_corr, r
    amps = energy(0.0)[1].amps
    rl = LR.kernel(fock, g, mo_e, nocc, amps, conv_tol_normt=1e-11)
    assert rl.converged
    d1, d2 = LR.densities(amps, rl.lams, nocc)
    dz = np.einsum('pq,pq', zmo, d1)
    assert abs(dz.imag) < 1e-12
    assert abs(LR.energy(fock, g, nocc, d1, d2) + mf.mol.energy_nuc() - mf.e_tot - energy(0.0)[1].e_corr) < 1e-11
    eps = 1e-3
    e = {k: energy(k * eps)[0] for k in (-2, -1, 1, 2)}
    fd = (8 * (e[1] - e[-1]) - (e[2] - e[-2])) / (12 * eps)
    assert abs(fd - dz.real) < 2e-10
    scf_value = np.trace(zmo[:nocc, :nocc]).real
    assert abs(dz.real - scf_value) > 1e-2
    other = np.einsum('pq,pq', zmo, LR.densities(amps, [t.conj() for t in amps], nocc)[0]).real
    assert abs(other - dz.real) > 1e-5


@pytest.mark.parametrize('name,cplx', [('h2o', False), ('bh', True)])
def test_relaxed_dipole_through_the_zvector_code(name, cplx):
    '''The orbital gradient of the table goes into the Z-vector equations
    of the CCSD line as it is (zccsdt.zccsd.zvector.solve with
    G[a,i] = 1/2 Gnh[a,i], relaxed_rdm1 on the Hermitian density): the
    relaxed CCSDT dipole derivative is that of E(SCF) + E_corr(CCSDT) with
    the SCF repeated in the field (central differences of fourth order).
    The unrelaxed one is another number.'''
    from zccsdt.zccsd import zvector
    mol = _molecule(name, False)[0].mol
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
        C = _spin_mixed(ghf)[0] if cplx else ghf.mo_coeff
        fock, g, mo_e, nocc = R.integrals(E.from_scf(ghf, mo_coeff=C))
        r = R.kernel(fock, g, mo_e, nocc, 'ccsdt', conv_tol=1e-13, conv_tol_normt=1e-11)
        assert r.converged
        return mf.e_tot + r.e_corr, (ghf, C, fock, g, mo_e, nocc, r.amps)

    ghf, C, fock, g, mo_e, nocc, amps = run(0.0)[1]
    rl = LR.kernel(fock, g, mo_e, nocc, amps, conv_tol_normt=1e-11)
    dm1 = LR.pyscf_rdms(*LR.densities(amps, rl.lams, nocc))[0]
    G = 0.5 * LR.orbital_gradient(fock, g, nocc, amps, rl.lams, dressed=True)
    conv, zv = zvector.solve(ghf, C, mo_e, nocc, -G, tol=1e-11)
    assert rl.converged and conv
    nao = mol.nao_nr()
    zmo = C[:nao].conj().T @ z @ C[:nao] + C[nao:].conj().T @ z @ C[nao:]
    unrelaxed = np.einsum('pq,qp', zmo, dm1).real
    relaxed = np.einsum('pq,qp', zmo, zvector.relaxed_rdm1(dm1, zv, nocc)).real
    eps = 1e-3
    e = {k: run(k * eps)[0] for k in (-2, -1, 1, 2)}
    fd = (8 * (e[1] - e[-1]) - (e[2] - e[-2])) / (12 * eps)
    assert abs(fd - relaxed) < 2e-10
    assert abs(unrelaxed - relaxed) > 5e-6
