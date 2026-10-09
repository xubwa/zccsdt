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
    # transposed (F-contiguous) operands
    np.testing.assert_allclose(gemm(B.T.copy().T.T, A.T), B.T @ A.T, atol=1e-12)
    At = np.asfortranarray(A)
    np.testing.assert_allclose(gemm(At, B), A @ B, atol=1e-12)


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('use_c', [True, False])
def test_gemm_tasks(dtype, use_c, monkeypatch):
    '''A list of products of parts of three arrays in one parallel region,
    each split over its columns, against one GEMM per product.'''
    from zccsdt import blas
    if not use_c:
        monkeypatch.setattr(blas, '_clib', lambda: None)
    rng = np.random.default_rng(1)

    def rnd(*shape):
        x = rng.standard_normal(shape)
        return (x + 1j * rng.standard_normal(shape)).astype(dtype) if dtype is np.complex128 else x
    A, B, C = rnd(40, 23), rnd(9000), rnd(30000)
    # (m, n, k, offset of A, lda, offset of B, ldb, offset of C, ldc); the
    # second and third share A, the last has nothing to sum over
    tasks = [(7, 250, 11, 2 * 23 + 3, 23, 5, 260, 100, 300),
             (12, 97, 20, 0, 23, 3000, 97, 5000, 97),
             (12, 97, 20, 0, 23, 5000, 100, 8000, 110),
             (3, 10, 0, 0, 23, 0, 10, 20000, 10),
             (0, 5, 4, 0, 23, 0, 5, 0, 5)]
    want = C.copy()
    for m, n, k, ao, lda, bo, ldb, co, ldc in tasks:
        for i in range(m):
            a = A.ravel()[ao + i * lda:ao + i * lda + k]
            b = np.array([B[bo + x * ldb:bo + x * ldb + n] for x in range(k)]).reshape(k, n)
            want[co + i * ldc:co + i * ldc + n] = 0.5 * want[co + i * ldc:co + i * ldc + n] - 2.0 * (a @ b)
    got = blas.gemm_tasks(tasks, A, B, C, alpha=-2.0, beta=0.5, cols=64)
    assert got is C
    np.testing.assert_allclose(C, want, atol=1e-12)
