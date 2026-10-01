'''Spin-sorted (alpha-first) orbitals with spin-grouped enumerations must
give exactly the same CC energies as the lexicographic code.'''
import functools
import numpy as np
import pytest
from pyscf import gto, scf

from zccsdt import eris as E, h5io
from zccsdt.cc import solver, t3pack

TIGHT = solver.Options(conv_tol=1e-11, conv_tol_normt=1e-8)


@functools.lru_cache(maxsize=None)
def nh2_uhf():
    mol = gto.M(atom='N 0 0 0; H 0 0.8 0.6; H 0 -0.8 0.6', basis='6-31g', spin=1,
                verbose=0)
    return scf.addons.convert_to_ghf(scf.UHF(mol).run(conv_tol=1e-12))


@functools.lru_cache(maxsize=None)
def h2o_rhf():
    mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis='6-31g',
                verbose=0)
    return scf.addons.convert_to_ghf(scf.RHF(mol).run(conv_tol=1e-12))


def test_spin_grouped_maps():
    om = t3pack.triple_map(6, [0, 0, 0, 1, 1, 1])
    g = om.tri_grp
    assert np.all(np.diff(g) >= 0) and list(om.tri_off) == [0, 1, 10, 19, 20]
    assert list(om.pair_off) == [0, 3, 12, 15]
    # maps are consistent with the enumeration
    for I, (i, j, k) in enumerate(om.triples):
        for r, (x, P) in enumerate(zip(om.free[:, I], om.pair[:, I])):
            assert om.tri[x, P] == I and om.sgn[x, P] == om.rsgn[r]


@pytest.mark.parametrize('frozen', [0, 2])
def test_spin_order_integrals(frozen, tmp_path):
    mf = nh2_uhf()
    e = E.from_scf(mf, frozen=frozen, spin_order=True)
    assert np.all(np.diff(e.spin_occ) >= 0) and np.all(np.diff(e.spin_vir) >= 0)
    # Fock and <ij||ab> vanish between different spins
    so, sv = e.spin_occ, e.spin_vir
    s = np.concatenate((so, sv))
    assert abs(e.fock[s[:, None] != s[None, :]]).max() < 1e-12
    h5io.save(tmp_path / 's.h5', e)
    back = h5io.load(tmp_path / 's.h5')
    np.testing.assert_array_equal(back.spin_occ, so)


@pytest.mark.parametrize('system', ['uhf', 'rhf'])
@pytest.mark.parametrize('method', ['ccsd', 'ccsdt'])
def test_spin_grouped_energy_equals_lexicographic(system, method):
    mf = nh2_uhf() if system == 'uhf' else h2o_rhf()
    lex = E.from_scf(mf, frozen=2)
    spn = E.from_scf(mf, frozen=2, spin_order=True)
    a = solver.kernel(lex, method, TIGHT)
    b = solver.kernel(spn, method, TIGHT)
    assert a.converged and b.converged
    assert abs(a.e_corr - b.e_corr) < 1e-10


def _rand_amps(e, seed, eng):
    rng = np.random.default_rng(seed)
    nocc, nvir = e.nocc, e.nmo - e.nocc
    so, sv = e.spin_occ, e.spin_vir
    t1 = 0.05 * (rng.random((nocc, nvir)) - 0.5) * (so[:, None] == sv[None, :])
    t2 = 0.05 * (rng.random((nocc, nocc, nvir, nvir)) - 0.5)
    t2 = t2 - t2.transpose(1, 0, 2, 3)
    t2 = t2 - t2.transpose(0, 1, 3, 2)
    ms = (so[:, None, None, None] + so[None, :, None, None]
          == sv[None, None, :, None] + sv[None, None, None, :])
    t2 = t2 * ms
    t3 = 0.05 * (rng.random(eng.size) - 0.5)
    return t1, t2, t3


@pytest.mark.parametrize('system', ['uhf', 'rhf'])
def test_spin_block_step_equals_dense(system):
    from zccsdt.cc import ccsdt_opt, t3engine
    mf = nh2_uhf() if system == 'uhf' else h2o_rhf()
    e = E.from_scf(mf, frozen=2, spin_order=True)
    eng = ccsdt_opt.engine_for(e, np.float64)
    assert isinstance(eng, t3engine.SpinBlockT3)
    t1, t2, t3 = _rand_amps(e, 1, eng)
    opts = solver.Options()
    b1, b2, b3 = ccsdt_opt.update_amps(opts, t1, t2, t3, e)
    # the same step with the dense engine on the same (spin-sorted) integrals
    om, vm = t3pack.maps_for(e)
    dense = t3engine.DenseT3(om, vm, np.float64, 2000)
    key = (id(om), id(vm), np.dtype(np.float64).str, 2000.0, False)
    saved = ccsdt_opt._ENGINES[key]
    try:
        ccsdt_opt._ENGINES[key] = dense
        d1, d2, d3 = ccsdt_opt.update_amps(opts, t1, t2, eng.L.to_dense(t3).ravel(), e)
    finally:
        ccsdt_opt._ENGINES[key] = saved
    assert abs(b1 - d1).max() < 1e-12
    assert abs(b2 - d2).max() < 1e-12
    D3 = d3.reshape(om.ntri, vm.ntri)
    assert abs(eng.L.to_dense(b3) - D3).max() < 1e-12
    # spin-forbidden blocks of the dense step stay zero
    assert abs(D3[om.tri_grp[:, None] != vm.tri_grp[None, :]]).max() < 1e-14
