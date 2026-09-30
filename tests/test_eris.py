import numpy as np
import pytest
from pyscf.cc import gccsd

from zccsdt import eris as E
from conftest import water_ghf, water_x2c


def _check_antisym(e):
    for name in E.BLOCKS:
        b = getattr(e, name)
        if name[0] == name[1]:
            assert np.allclose(b, -b.transpose(1, 0, 2, 3), atol=1e-12), name
        if name[2] == name[3]:
            assert np.allclose(b, -b.transpose(0, 1, 3, 2), atol=1e-12), name


@pytest.mark.parametrize('kind', ['ghf', 'ghf-rotated', 'x2c'])
def test_reference_energy_and_antisymmetry(kind):
    if kind == 'x2c':
        mf = water_x2c()
        e = E.from_scf(mf)
    else:
        mf = water_ghf()
        C = None if kind == 'ghf' else E.rotate_occ_vir(mf.mo_coeff, mf.mo_occ, seed=5)
        e = E.from_scf(mf, mo_coeff=C)
    assert abs(E.reference_energy(e) - mf.e_tot) < 1e-10
    _check_antisym(e)
    # Hermiticity of the Fock matrix and <ij||ab> = <ab||ij>^* via ovvo/ovov
    assert np.allclose(e.fock, e.fock.conj().T, atol=1e-10)
    if kind != 'ghf':
        assert np.iscomplexobj(e.oovv) and abs(e.oovv.imag).max() > 1e-3


def test_blocks_match_pyscf_gccsd():
    mf = water_ghf()
    ours = E.from_scf(mf, frozen=2)
    ref = gccsd.GCCSD(mf, frozen=2).ao2mo()
    for name in E.BLOCKS:
        assert np.allclose(getattr(ours, name), np.asarray(getattr(ref, name)),
                           atol=1e-11), name
    assert np.allclose(ours.fock, ref.fock, atol=1e-11)


def test_rotation_keeps_frozen_space():
    mf = water_ghf()
    C = E.rotate_occ_vir(mf.mo_coeff, mf.mo_occ, seed=2, frozen=2)
    assert np.allclose(C[:, :2], mf.mo_coeff[:, :2])
    e = E.from_scf(mf, mo_coeff=C, frozen=2)
    ref = E.from_scf(mf, frozen=2)
    # occ-occ / vir-vir rotations leave the traces of the Fock blocks invariant
    n = e.nocc
    assert abs(np.trace(e.fock[:n, :n]) - np.trace(ref.fock[:n, :n])) < 1e-10
    assert abs(np.trace(e.fock[n:, n:]) - np.trace(ref.fock[n:, n:])) < 1e-10
