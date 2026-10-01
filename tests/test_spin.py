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


@pytest.mark.parametrize('system', ['uhf', 'rhf'])
@pytest.mark.parametrize('cplx', [False, True])
def test_spin_block_c_equals_numpy(system, cplx):
    from zccsdt.cc import t3engine
    mf = nh2_uhf() if system == 'uhf' else h2o_rhf()
    e = E.from_scf(mf, frozen=2, spin_order=True)
    om, vm = t3pack.maps_for(e)
    dt = np.complex128 if cplx else np.float64
    eng = t3engine.SpinBlockT3(om, vm, dt, 0.05)       # tiny budget: many batches
    if not eng.use_c:
        pytest.skip('C kernels not built')
    rng = np.random.default_rng(3)
    no, nv, npo, npv = om.n, vm.n, om.npair, vm.npair
    so, sv, gP, gQ = om.spins, vm.spins, om.pair_grp, vm.pair_grp

    def rnd(*s):
        x = rng.random(s) - 0.5
        return x + 1j * (rng.random(s) - 0.5) if cplx else x
    T3 = rnd(eng.size)
    G_Pd = rnd(nv + no, npo, nv)
    G_Pd[:nv] *= gP[None, :, None] == sv[:, None, None] + sv[None, None, :]
    G_Pd[nv:] *= gP[None, :, None] == so[:, None, None] + sv[None, None, :]
    G3 = rnd(no, npv, no + nv)
    G3[:, :, :no] *= so[:, None, None] + so[None, None, :] == gQ[None, :, None]
    G3[:, :, no:] *= so[:, None, None] + sv[None, None, :] == gQ[None, :, None]
    gpp = rnd(npo, npv) * (gP[:, None] == gQ[None, :])
    fov = rnd(no, nv) * (so[:, None] == sv[None, :])
    args = (T3, G_Pd, G3.reshape(no * npv, -1), gpp, fov)
    for x, y in zip(eng._pass1_c(*args), eng._pass1_np(*args)):
        np.testing.assert_allclose(x, y, atol=1e-12)
    Wr = rnd(nv, no, nv, no) * ((sv[:, None, None, None] - so[None, :, None, None])
                                == (sv[None, None, :, None] - so[None, None, None, :]))
    t2mn = rnd(npo, nv, nv) * (gP[:, None, None] == sv[None, :, None] + sv[None, None, :])
    Wv = rnd(npv, nv, no) * (gQ[:, None, None] == sv[None, :, None] + so[None, None, :])
    Wo = rnd(no, nv, npo) * (so[:, None, None] + sv[None, :, None] == gP[None, None, :])
    t2P = rnd(no, no, npv) * (so[:, None, None] + so[None, :, None] == gQ[None, None, :])
    Wpp = rnd(npv, npv) * (gQ[:, None] == gQ[None, :])
    O6 = rnd(om.ntri, om.ntri) * (om.tri_grp[:, None] == om.tri_grp[None, :])
    rargs = (T3, Wr.reshape(nv * no, -1), t2mn, Wv, Wo, t2P, Wpp, O6)
    np.testing.assert_allclose(eng._residual_c(*rargs), eng._residual_np(*rargs), atol=1e-11)


@pytest.mark.parametrize('use_c', [True, False])
def test_rccsdt_spin_flip(use_c, monkeypatch):
    '''Closed shell: storing only the T3 blocks with 0 or 1 beta indices and
    obtaining the others by spin flip gives the same CCSDT energy.'''
    if use_c and not t3pack.USE_C:
        pytest.skip('C kernels not built')
    monkeypatch.setattr(t3pack, 'USE_C', use_c)
    from zccsdt.cc import ccsdt_opt
    ccsdt_opt._ENGINES.clear()
    e = E.from_scf(h2o_rhf(), frozen=2, spin_order=True)
    full = solver.kernel(e, 'ccsdt', TIGHT)
    o = solver.Options(conv_tol=1e-11, conv_tol_normt=1e-8, spin_flip=True)
    r = solver.kernel(e, 'ccsdt', o)
    eng_u = ccsdt_opt.engine_for(e, np.float64, o.mem_mb, False)
    eng_r = ccsdt_opt.engine_for(e, np.float64, o.mem_mb, True)
    assert eng_r.size < 0.6 * eng_u.size
    assert r.converged and abs(r.e_corr - full.e_corr) < 1e-10
    ccsdt_opt._ENGINES.clear()


def test_spin_flip_rejects_open_shell():
    from zccsdt.cc import ccsdt_opt
    e = E.from_scf(nh2_uhf(), frozen=2, spin_order=True)
    with pytest.raises(ValueError):
        ccsdt_opt.engine_for(e, np.float64, 2000, True)
