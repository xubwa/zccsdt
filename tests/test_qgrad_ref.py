'''The dense reference of the first derivatives of CCSDT(Q)
(zccsdt.cc.qgrad_ref; docs/lambda.md, section 11):

- the derivatives of E(Q) with respect to t2, t3, the integrals and the
  orbital energies against differences of the dense E(Q), real and complex;
- the orbital gradient among occupied and among virtual spin orbitals
  against the Fock-matrix density of (Q) (the form that needs no canonical
  orbitals);
- the relaxed dipole derivative against E(SCF) + E_corr(CCSDT) + E(Q) with
  the SCF repeated in the field, and against CFOUR's ncc (CALC=CCSDT(Q),
  PROPS=FIRST_ORDER), with frozen and dropped spin orbitals and with complex
  ones.'''
import numpy as np
import pytest
from pyscf import gto, scf

import fockspace as F
from test_ccsdtq_ref import _scf, _spin_mixed
from zccsdt import eris as E
from zccsdt.cc import ccsdtq_ref as R, qgrad_ref as G

WATER = 'O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587'
# CFOUR ncc, H2O / STO-3G (bench/refs/GENBAS), RHF, all electrons, the geometry above in bohr
# (bench/refs/cfour/h2o_sto3g/ZMAT.qprops.bohr): the dipole moment along z of the relaxed
# density, sum Z_A z_A - trace(z D), ten decimals printed
CFOUR_DIPOLE = {'ccsdt': 0.6359340712, 'ccsdt(q)': 0.6358501673}


def _fd(f, eps):
    return (8 * (f(eps) - f(-eps)) - (f(2 * eps) - f(-2 * eps))) / (12 * eps)


@pytest.mark.parametrize('cplx', [False, True])
def test_derivatives_of_the_correction(cplx):
    '''dE(Q) = Re [sum_(i<j,a<b) s2 dt2 + sum s3 dt3 + sum d2 d<pq||rs>] +
    sum de d e_p for amplitudes that solve nothing: four-point differences
    of the dense E(Q) along random directions (complex ones: t and conj(t)
    are independent, and s is twice the Wirtinger derivative).'''
    no, nv = 5, 5
    n = no + nv
    rng = np.random.default_rng(3)
    if cplx:
        mk = lambda k, s: F.complex_antisym(rng, no, nv, k, s)             # noqa: E731
        mg = lambda: F.hermitian_integrals(rng, n)                          # noqa: E731
    else:
        mk = lambda k, s: s * F.random_antisym(rng, no, nv, k)              # noqa: E731
        mg = lambda: F.random_integrals(rng, n)                             # noqa: E731
    t2, t3, g = mk(2, 0.2), mk(3, 0.1), 0.3 * mg()
    e = np.concatenate([-1 - rng.random(no), 1 + rng.random(nv)])
    gr = G.quadruples_gradient(t2, t3, g, e, no)
    assert abs(gr.e_q - G.energy(t2, t3, g, e, no)) < 1e-12 * abs(gr.e_q)
    assert np.iscomplexobj(gr.s3) == cplx
    d2, d3, dg, de = mk(2, 1.0), mk(3, 1.0), mg(), rng.standard_normal(n)
    checks = [(lambda x: G.energy(t2 + x * d2, t3, g, e, no), np.einsum('ijab,ijab', gr.s2, d2).real / 4),
              (lambda x: G.energy(t2, t3 + x * d3, g, e, no), np.einsum('ijkabc,ijkabc', gr.s3, d3).real / 36),
              (lambda x: G.energy(t2, t3, g + x * dg, e, no), np.einsum('pqrs,pqrs', gr.d2, dg).real),
              (lambda x: G.energy(t2, t3, g, e + x * de, no), gr.de @ de)]
    for f, analytic in checks:
        assert abs(analytic) > 1e-2
        assert abs(_fd(f, 1e-3) - analytic) < 2e-9 * abs(analytic)
    # the sources are antisymmetric, t1 has none
    assert abs(gr.s2 + gr.s2.transpose(1, 0, 2, 3)).max() < 1e-13 * abs(gr.s2).max()
    assert abs(gr.s3 + gr.s3.transpose(0, 1, 2, 4, 3, 5)).max() < 1e-13 * abs(gr.s3).max()


def _molecule(name, cplx):
    '''make_scf(eps) -> (GHF in the field eps z, its spin orbitals), z in the AO basis.'''
    mol = _scf(name).mol if name != 'h2o' else gto.M(atom=WATER, basis='sto-3g', verbose=0)
    with mol.with_common_orig((0, 0, 0)):
        z = mol.intor('int1e_r')[2]
    h0 = scf.RHF(mol).get_hcore()

    def make_scf(eps):
        mf = scf.RHF(mol)
        mf.get_hcore = lambda *args: h0 + eps * z
        mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-10
        mf.kernel()
        ghf = scf.addons.convert_to_ghf(mf)
        h2 = np.kron(np.eye(2), h0 + eps * z)
        ghf.get_hcore = lambda *args: h2
        return ghf, _spin_mixed(ghf)[0] if cplx else ghf.mo_coeff
    return mol, make_scf, np.kron(np.eye(2), z)


def _energy(make_scf, eps, nfrozen, ndrop):
    '''E(SCF) + E_corr(CCSDT) + E(Q) in the field, all dense.'''
    mf, C = make_scf(eps)
    fock, g, mo_e, o = R.integrals(E.from_scf(mf, mo_coeff=C))
    a = slice(nfrozen, len(mo_e) - ndrop)
    r = R.kernel(fock[a, a], g[a, a, a, a], mo_e[a], o - nfrozen, 'ccsdt', conv_tol=1e-13, conv_tol_normt=1e-11)
    assert r.converged
    return mf.e_tot + r.e_corr + R.quadruples(r.amps[1], r.amps[2], g[a, a, a, a], mo_e[a], o - nfrozen)[1]


@pytest.mark.parametrize('name,cplx,nfrozen,ndrop', [('h2o', False, 0, 0), ('bh', True, 0, 0), ('bh', True, 2, 2),
                                                    ('lih', False, 0, 0)])
def test_relaxed_dipole_against_finite_field(name, cplx, nfrozen, ndrop):
    '''The trace of z with the relaxed density of CCSDT(Q) is the derivative
    of E(SCF) + E_corr(CCSDT) + E(Q), the SCF repeated in the field
    (four-point differences; two steps extrapolated for LiH, whose
    differences have a large truncation error).  The density of CCSDT is
    another number, and so is the one without the response of the spin
    orbitals.  H2O: the dipole moment of CFOUR's ncc.'''
    mol, make_scf, z = _molecule(name, cplx)
    mf, C = make_scf(0.0)
    zmo = C.conj().T @ z @ C
    rel = G.relaxed_rdm1(mf, C, nfrozen, ndrop)
    assert abs(rel.relaxed - rel.relaxed.conj().T).max() < 1e-13
    assert abs(np.trace(rel.relaxed) - mol.nelectron) < 1e-10
    value = np.einsum('pq,qp', zmo, rel.relaxed).real
    if name == 'lih':
        f = [_fd(lambda x: _energy(make_scf, x, nfrozen, ndrop), h) for h in (1e-3, 5e-4)]
        fd = (16 * f[1] - f[0]) / 15
    else:
        fd = _fd(lambda x: _energy(make_scf, x, nfrozen, ndrop), 1e-3)
    print('%s %s %d frozen %d dropped: relaxed %.12f finite field %.12f, difference %.1e'
          % (name, 'complex' if cplx else 'real', nfrozen, ndrop, value, fd, value - fd))
    assert abs(value - fd) < 2e-10
    ccsdt = G.relaxed_rdm1(mf, C, nfrozen, ndrop, with_q=False, amps=rel.amps)
    other = np.einsum('pq,qp', zmo, ccsdt.relaxed).real
    assert abs(other - value) > 1e-6 and abs(np.einsum('pq,qp', zmo, rel.unrelaxed).real - value) > 1e-7
    if name == 'h2o':
        nuc = sum(mol.atom_charge(i) * mol.atom_coord(i)[2] for i in range(mol.natm))
        assert abs(nuc - value - CFOUR_DIPOLE['ccsdt(q)']) < 1.5e-10
        assert abs(nuc - other - CFOUR_DIPOLE['ccsdt']) < 1.5e-10


@pytest.mark.parametrize('name,cplx,nfrozen', [('lih', True, 0), ('bh', True, 0)])
def test_gradient_among_occupied_and_among_virtual(name, cplx, nfrozen):
    '''The orbital gradient of CCSDT(Q) among the active occupied and among
    the active virtual spin orbitals - from the densities of the CCSDT
    Lagrangian with the sources in lambda and those of E(Q) - is (e_q - e_p)
    times the Hermitian part of the Fock-matrix density of (Q), transposed:
    the form of (Q) that solves D y = r with the whole Fock matrix does not
    change with these rotations, and it differs from E(Q) by the
    off-diagonal elements of f alone.'''
    mol, make_scf, z = _molecule(name, cplx)
    mf, C = make_scf(0.0)
    rel = G.relaxed_rdm1(mf, C, nfrozen)
    fock, g, mo_e, o = R.integrals(E.from_scf(mf, mo_coeff=C))
    a = slice(nfrozen, None)
    gam = G.fock_density(rel.amps[1], rel.amps[2], g[a, a, a, a], mo_e[a], o - nfrozen)
    assert abs(gam.diagonal() - rel.q.de).max() < 1e-14
    ea, oa = mo_e[a], o - nfrozen
    want = (ea[None, :] - ea[:, None]) * 0.5 * (gam.T + gam.conj())
    Ga = rel.gradient[a, a]
    size = 0.0
    for s in (slice(0, oa), slice(oa, None)):
        assert abs(Ga[s, s] - want[s, s]).max() < 1e-11
        size = max(size, abs(Ga[s, s]).max())
    assert size > 1e-9
