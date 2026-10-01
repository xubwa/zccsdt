'''GEMM on C-ordered arrays with accumulation (beta), via SciPy BLAS.

numpy.matmul cannot accumulate into an existing array, which costs a
temporary of the output size plus an extra pass over memory.  A C-ordered
matrix is the transpose of an F-ordered one, so C = alpha A B + beta C is
computed as C^T = alpha B^T A^T + beta C^T with no copies.
'''

import numpy as np
from scipy.linalg import blas as _blas

_GEMM = {np.dtype(np.float64): _blas.dgemm, np.dtype(np.complex128): _blas.zgemm}


def gemm(A, B, C=None, alpha=1.0, beta=0.0):
    '''C = alpha * A @ B + beta * C in place for 2-D C-contiguous arrays.'''
    dtype = np.result_type(A, B) if C is None else C.dtype
    A = np.ascontiguousarray(A, dtype=dtype)
    B = np.ascontiguousarray(B, dtype=dtype)
    if C is None:
        C = np.empty((A.shape[0], B.shape[1]), dtype=dtype)
        beta = 0.0
    assert C.flags.c_contiguous and C.shape == (A.shape[0], B.shape[1])
    if A.shape[1] == 0:
        C *= beta
        return C
    out = _GEMM[np.dtype(dtype)](alpha, B.T, A.T, beta=beta, c=C.T, overwrite_c=1)
    if not np.shares_memory(out, C):
        C[...] = out.T
    return C
