import numpy as np
import pytest
from pyscf import gto, scf

from zccsdt import eris as E, eris_spin
from zccsdt.cc import t3pack
from zccsdt.cc.ccsdt_opt import vvvv_packed


def _systems():
    nh2 = gto.M(atom='N 0 0 0; H 0 0.8 0.6; H 0 -0.8 0.6', basis='6-31g', spin=1, verbose=0)
    h2o = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis='6-31g', verbose=0)
    return {'uhf': scf.UHF(nh2).run(conv_tol=1e-12), 'rhf': scf.RHF(h2o).run(conv_tol=1e-12)}


SYS = _systems()


@pytest.mark.parametrize('kind', ['uhf', 'rhf'])
@pytest.mark.parametrize('frozen', [0, 2])
def test_spin_eris_match_ghf_spin_order(kind, frozen):
    mf = SYS[kind]
    se = eris_spin.from_scf(mf, frozen=frozen, vir_batch=3)
    ref = E.from_scf(scf.addons.convert_to_ghf(mf), frozen=frozen, spin_order=True)
    np.testing.assert_array_equal(se.spin_occ, ref.spin_occ)
    np.testing.assert_array_equal(se.spin_vir, ref.spin_vir)
    np.testing.assert_allclose(se.fock, ref.fock, atol=1e-10)
    assert abs(se.e_hf - ref.e_hf) < 1e-10
    for k in E.BLOCKS[:-1]:
        np.testing.assert_allclose(se.g[k].to_dense(), getattr(ref, k), atol=1e-10,
                                   err_msg=k)
    vm = t3pack.triple_map(len(se.spin_vir), se.spin_vir)
    for a, b in zip(se.vvvv_blocks, vvvv_packed(ref, vm)):
        np.testing.assert_allclose(a, b, atol=1e-10)
    if kind == 'rhf':
        # spin-flipped blocks share storage
        g = se.g['oovv'].blocks
        assert g[(1, 1, 1, 1)] is g[(0, 0, 0, 0)] and g[(1, 0, 1, 0)] is g[(0, 1, 0, 1)]
        assert se.vvvv_blocks[2] is se.vvvv_blocks[0]
    # the dense view reproduces the spin-orbital ERIs (vvvv via the packed form)
    d = se.to_dense_eris()
    for a, b in zip(vvvv_packed(d, vm), vvvv_packed(ref, vm)):
        np.testing.assert_allclose(a, b, atol=1e-10)


@pytest.mark.parametrize('kind', ['uhf', 'rhf'])
def test_step_with_spin_eris_equals_ghf_spin_order(kind):
    '''A CCSDT step with U/R-form integrals equals the step with the GHF
    spin-ordered (dense spin-orbital) integrals.'''
    from zccsdt.cc import ccsdt_opt, solver
    mf = SYS[kind]
    se = eris_spin.from_scf(mf, frozen=2, vir_batch=3)
    ref = E.from_scf(scf.addons.convert_to_ghf(mf), frozen=2, spin_order=True)
    eng = ccsdt_opt.engine_for(ref, np.float64)
    rng = np.random.default_rng(7)
    so, sv = ref.spin_occ, ref.spin_vir
    no, nv = len(so), len(sv)
    t1 = 0.05 * (rng.random((no, nv)) - 0.5) * (so[:, None] == sv[None, :])
    t2 = 0.05 * (rng.random((no, no, nv, nv)) - 0.5)
    t2 = t2 - t2.transpose(1, 0, 2, 3)
    t2 = t2 - t2.transpose(0, 1, 3, 2)
    t2 *= (so[:, None, None, None] + so[None, :, None, None]
           == sv[None, None, :, None] + sv[None, None, None, :])
    if kind == 'rhf':   # closed shell: alpha/beta (spin-flip) symmetric amplitudes
        na, va = int(np.sum(so == 0)), int(np.sum(sv == 0))
        fo = np.concatenate((np.arange(na, no), np.arange(na)))
        fv = np.concatenate((np.arange(va, nv), np.arange(va)))
        t1 = 0.5 * (t1 + t1[fo][:, fv])
        t2 = 0.5 * (t2 + t2[fo][:, fo][:, :, fv][:, :, :, fv])
    t3 = 0.05 * (rng.random(eng.size) - 0.5)
    o = solver.Options()
    a = ccsdt_opt.update_amps(o, t1, t2, t3, ref)
    b = ccsdt_opt.update_amps(o, t1, t2, t3, se)
    for x, y, name in zip(a, b, ('t1', 't2', 't3')):
        assert abs(x - y).max() < 1e-11, name
    a = ccsdt_opt.ccsd_update_amps(o, t1, t2, ref)
    b = ccsdt_opt.ccsd_update_amps(o, t1, t2, se)
    for x, y in zip(a, b):
        assert abs(x - y).max() < 1e-11


@pytest.mark.parametrize('kind', ['uhf', 'rhf'])
def test_converged_energies_with_spin_eris(kind):
    from zccsdt.cc import solver, ccsdt_opt
    mf = SYS[kind]
    se = eris_spin.from_scf(mf, frozen=2)
    ref = E.from_scf(scf.addons.convert_to_ghf(mf), frozen=2)
    tight = solver.Options(conv_tol=1e-11, conv_tol_normt=1e-8)
    flip = solver.Options(conv_tol=1e-11, conv_tol_normt=1e-8, spin_flip=(kind == 'rhf'))
    for meth, o in (('ccsd', tight), ('ccsdt', flip)):
        a = solver.kernel(ref, meth, tight)
        b = solver.kernel(se, meth, o)
        assert b.converged and abs(a.e_corr - b.e_corr) < 1e-10, meth
        assert abs(a.e_tot - b.e_tot) < 1e-9
    ccsdt_opt._ENGINES.clear()


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_vvvv_t1_blocks_c_equals_numpy(dtype, monkeypatch):
    '''Spin-block vvvv x t1 C kernels against the numpy path (unequal numbers
    of alpha and beta orbitals, as for an open-shell reference).'''
    from zccsdt.cc import spin_imds, t3pack
    rng = np.random.default_rng(3)
    sv = np.array([0] * 7 + [1] * 5)
    so = np.array([0] * 3 + [1] * 2)
    vm, om = t3pack.triple_map(len(sv), sv), t3pack.triple_map(len(so), so)

    def rand(*shape):
        x = rng.standard_normal(shape)
        return x + 1j * rng.standard_normal(shape) if dtype == np.complex128 else x

    Vd = [rand(hi - lo, hi - lo) for lo, hi in vm.blocks]
    t1 = rand(len(so), len(sv)) * (so[:, None] == sv[None, :])
    W0 = rand(vm.npair, len(sv), len(so))
    Wc = spin_imds.vvvv_t1_blocks(W0.copy(), Vd, t1, om, vm)
    monkeypatch.setattr(t3pack, 'USE_C', False)
    Wn = spin_imds.vvvv_t1_blocks(W0.copy(), Vd, t1, om, vm)
    assert abs(Wc - Wn).max() < 1e-12
