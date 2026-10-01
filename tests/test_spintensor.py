import numpy as np
import pytest

from zccsdt.cc import t3pack
from zccsdt.cc.spintensor import (SpinTensor, orbital_space, pair_space, einsum,
                                  pack_pairs, unpack_pairs)

SO = np.array([0, 0, 0, 1, 1])          # 3 alpha, 2 beta occupied
SV = np.array([0, 0, 0, 0, 1, 1, 1])    # 4 alpha, 3 beta virtual


def _ms_mask(spins_list, sig):
    grids = np.meshgrid(*spins_list, indexing='ij')
    return sum(sg * g for sg, g in zip(sig, grids)) == 0


def _rand(spins_list, sig, rng):
    shape = tuple(len(s) for s in spins_list)
    return rng.standard_normal(shape) * _ms_mask(spins_list, sig)


def test_roundtrip_and_einsum():
    rng = np.random.default_rng(0)
    o, v = orbital_space('o', SO), orbital_space('v', SV)
    A = _rand([SO, SO, SV, SV], (-1, -1, 1, 1), rng)
    B = _rand([SV, SV, SV, SV], (1, 1, -1, -1), rng)
    C = _rand([SO, SV, SV, SO], (1, 1, -1, -1), rng)
    At = SpinTensor.from_dense(A, (o, o, v, v), sig=(-1, -1, 1, 1))
    Bt = SpinTensor.from_dense(B, (v, v, v, v), sig=(1, 1, -1, -1))
    Ct = SpinTensor.from_dense(C, (o, v, v, o), sig=(1, 1, -1, -1))
    np.testing.assert_array_equal(At.to_dense(), A)
    assert len(At.blocks) == 6
    np.testing.assert_allclose(einsum('ijab,abcd->ijcd', At, Bt).to_dense(),
                               np.einsum('ijab,abcd->ijcd', A, B), atol=1e-12)
    np.testing.assert_allclose(einsum('mbej,imae->ijab', Ct, At).to_dense(),
                               np.einsum('mbej,imae->ijab', C, A), atol=1e-12)
    np.testing.assert_allclose((At * 2 - At.transpose(1, 0, 2, 3)).to_dense(),
                               2 * A - A.transpose(1, 0, 2, 3), atol=1e-14)


def test_pack_unpack_pairs():
    rng = np.random.default_rng(1)
    o, v = orbital_space('o', SO), orbital_space('v', SV)
    om, vm = t3pack.triple_map(len(SO), SO), t3pack.triple_map(len(SV), SV)
    P, Q = pair_space('P', om), pair_space('Q', vm)
    A = _rand([SO, SO, SV, SV], (-1, -1, 1, 1), rng)
    A = A - A.transpose(1, 0, 2, 3)
    A = A - A.transpose(0, 1, 3, 2)
    At = SpinTensor.from_dense(A, (o, o, v, v), sig=(-1, -1, 1, 1))
    Ap = pack_pairs(pack_pairs(At, 2, Q), 0, P)
    ref = A[om.pairs[:, 0][:, None], om.pairs[:, 1][:, None], vm.pairs[:, 0], vm.pairs[:, 1]]
    np.testing.assert_allclose(Ap.to_dense(), ref, atol=1e-14)
    back = unpack_pairs(unpack_pairs(Ap, 0, o), 2, v)
    np.testing.assert_allclose(back.to_dense(), A, atol=1e-14)
