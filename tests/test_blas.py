import numpy as np
import pytest

from zccsdt.blas import gemm


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_gemm_accumulate(dtype):
    rng = np.random.default_rng(0)
    A = rng.standard_normal((7, 5)).astype(dtype)
    B = rng.standard_normal((5, 9)).astype(dtype)
    C = rng.standard_normal((7, 9)).astype(dtype)
    want = 2.0 * A @ B - 0.5 * C
    got = gemm(A, B, C, alpha=2.0, beta=-0.5)
    assert got is C
    np.testing.assert_allclose(C, want, atol=1e-12)
    np.testing.assert_allclose(gemm(A, B), A @ B, atol=1e-12)
