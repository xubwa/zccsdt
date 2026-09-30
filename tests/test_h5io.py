import numpy as np
import pytest

from zccsdt import eris as E, h5io
from zccsdt.cc import solver
from conftest import water_ghf, water_x2c


@pytest.mark.parametrize('kind', ['ghf', 'x2c'])
def test_roundtrip(tmp_path, kind):
    mf = water_ghf() if kind == 'ghf' else water_x2c()
    e = E.from_scf(mf, frozen=2)
    path = tmp_path / 'ints.h5'
    h5io.save(path, e)
    r = h5io.load(path)
    assert r.nocc == e.nocc
    assert r.e_hf == e.e_hf, (r.e_hf, e.e_hf)
    assert r.e_nuc == e.e_nuc, (r.e_nuc, e.e_nuc)
    assert r.dtype == e.dtype, (r.dtype, e.dtype)
    np.testing.assert_array_equal(r.fock, e.fock)
    for name in E.BLOCKS:
        np.testing.assert_allclose(getattr(r, name), getattr(e, name), atol=1e-14)


def test_ccsd_from_file(tmp_path):
    mf = water_ghf()
    e = E.from_scf(mf, frozen=2)
    h5io.save(tmp_path / 'ints.h5', e)
    o = solver.Options(conv_tol=1e-12, conv_tol_normt=1e-9)
    a = solver.kernel(e, 'ccsd', o)
    b = solver.kernel(h5io.load(tmp_path / 'ints.h5'), 'ccsd', o)
    assert abs(a.e_corr - b.e_corr) < 1e-12
