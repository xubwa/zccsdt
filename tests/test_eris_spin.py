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
