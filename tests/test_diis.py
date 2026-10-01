import numpy as np
import pytest
from pyscf.lib import diis as pdiis

from zccsdt.cc.diis import DIIS


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_matches_pyscf(dtype):
    rng = np.random.default_rng(0)
    ref = pdiis.DIIS(incore=True)
    ref.space = 4
    ours = DIIS(space=4)
    x = rng.standard_normal(1000).astype(dtype)
    for it in range(9):
        x = 0.9 * x + 0.01 * rng.standard_normal(1000)
        if dtype == np.complex128:
            x = x + 0.01j * rng.standard_normal(1000)
        a = ref.update(x.copy())
        b = ours.update(x.copy())
        np.testing.assert_allclose(b, a, rtol=1e-7, atol=1e-9)
        x = b
