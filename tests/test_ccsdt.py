import numpy as np
import pytest
from pyscf import gto, scf, fci
from pyscf.fci import fci_dhf_slow

from zccsdt import eris as E
from zccsdt.cc import solver, ccsdt, bruteforce
from conftest import water_ghf, three_electron, socutils_available

TIGHT = solver.Options(conv_tol=1e-12, conv_tol_normt=1e-9)


def test_ccsdt_equals_fci_three_electrons_real():
    mf = three_electron('ghf')
    uhf = scf.UHF(mf.mol).run(conv_tol=1e-12)
    e_fci = fci.FCI(uhf).kernel()[0]
    r = solver.kernel(E.from_scf(mf), 'ccsdt', TIGHT)
    assert r.converged
    assert abs(r.e_tot - e_fci) < 1e-9


def test_ccsdt_equals_fci_three_electrons_spinor():
    mf = three_electron('x2c')
    C = mf.mo_coeff
    h1 = C.conj().T @ mf.get_hcore() @ C
    eri = E._mo_chemist(mf.mol, C, 'spinor')
    e_fci = fci_dhf_slow.kernel(h1, eri, C.shape[1], 3,
                                ecore=mf.energy_nuc(), verbose=0)[0]
    r = solver.kernel(E.from_scf(mf), 'ccsdt', TIGHT)
    assert r.converged
    assert abs(r.e_tot - e_fci) < 1e-9


def test_ccsdt_complex_rotation_invariance():
    '''RHF -> GHF -> complex unitary within active occ and within vir.'''
    mf = water_ghf('sto-3g')
    ref = solver.kernel(E.from_scf(mf, frozen=2), 'ccsdt', TIGHT)
    C = E.rotate_occ_vir(mf.mo_coeff, mf.mo_occ, seed=7, frozen=2)
    rot = solver.kernel(E.from_scf(mf, mo_coeff=C, frozen=2), 'ccsdt', TIGHT)
    assert ref.converged and rot.converged
    assert abs(rot.e_corr.imag) < 1e-10
    assert abs(rot.e_corr - ref.e_corr) < 1e-10


@pytest.fixture(scope='module')
def fh_small():
    '''FH/6-31G, GHF, 6 frozen spin orbitals: nocc=4 < nvir=12, so the
    bilinear t2*t3 terms are exercised; complex via a random rotation.'''
    mol = gto.M(atom='F 0 0 0; H 0 0 0.917', basis='6-31g', verbose=0)
    ghf = scf.addons.convert_to_ghf(scf.RHF(mol).run(conv_tol=1e-12))
    C = E.rotate_occ_vir(ghf.mo_coeff, ghf.mo_occ, seed=11, frozen=6)
    return E.from_scf(ghf, mo_coeff=C, frozen=6)


def _rand_amps(nocc, nvir, seed, with_t1):
    rng = np.random.default_rng(seed)

    def rand(*sh):
        return 0.05 * (rng.random(sh) - 0.5 + 1j * (rng.random(sh) - 0.5))
    t1 = rand(nocc, nvir) if with_t1 else np.zeros((nocc, nvir), complex)
    t2 = rand(nocc, nocc, nvir, nvir)
    t2 = t2 - t2.transpose(1, 0, 2, 3)
    t2 = t2 - t2.transpose(0, 1, 3, 2)
    t3 = ccsdt.fullasym(rand(nocc, nocc, nocc, nvir, nvir, nvir))
    return t1, t2, t3


@pytest.mark.parametrize('seed,with_t1', [(1, False), (2, True), (3, True)])
def test_t3_residual_vs_bruteforce(fh_small, seed, with_t1):
    e = fh_small
    nocc, nmo = e.nocc, e.nmo
    g = bruteforce.build_g(e, nocc, nmo)
    fock = np.asarray(e.fock)
    t1, t2, t3 = _rand_amps(nocc, nmo - nocc, seed, with_t1)
    r3_ref = bruteforce.ccsdt_residuals(fock, g, nocc, nmo, t1, t2, t3)[2]
    ge, fdr = ccsdt._t1_dress(g, fock, t1, nocc, nmo)
    r3 = ccsdt._t3_residual(t2, t3, ge, fdr, nocc, nmo)
    scale = abs(r3_ref).max()
    assert abs(r3 - r3_ref).max() < 1e-12 * max(1.0, scale)
    if ccsdt._USE_CPACK:
        r3p = ccsdt._t3_residual_packed(t2, t3, ge, fdr, nocc, nmo)
        assert abs(r3p - ccsdt.pack_t3(r3_ref)).max() < 1e-12 * max(1.0, scale)


@pytest.mark.slow
def test_converged_energy_vs_bruteforce(fh_small):
    e = fh_small
    nocc, nmo = e.nocc, e.nmo
    g = bruteforce.build_g(e, nocc, nmo)
    e_ref = bruteforce.kernel(np.asarray(e.fock), g, nocc, nmo, e.mo_energy,
                              conv_tol=1e-12)[0]
    r = solver.kernel(e, 'ccsdt', TIGHT)
    assert r.converged
    assert abs(r.e_corr - e_ref) < 1e-9


@pytest.mark.socutils
@pytest.mark.skipif(not socutils_available(), reason='socutils not importable')
def test_ccsdt_x2c_vs_socutils():
    from socutils.scf import spinor_hf
    from socutils.cc.zccsdt import ZCCSDT
    from conftest import WATER
    mol = gto.M(atom=WATER, basis='sto-3g', verbose=0)
    mf = spinor_hf.SCF(mol).run(conv_tol=1e-12)
    ref = ZCCSDT(mf, frozen=2)
    ref.conv_tol = 1e-12
    ref.conv_tol_normt = 1e-9
    ref.verbose = 0
    ref.kernel()
    r = solver.kernel(E.from_scf(mf, frozen=2), 'ccsdt', TIGHT)
    assert abs(r.e_corr - ref.e_corr) < 1e-9
