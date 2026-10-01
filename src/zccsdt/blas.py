'''GEMM on C-ordered arrays with accumulation (beta).

Uses zc_gemm from the zccsdt C library when available, so that Python-side
contractions run on the same BLAS (MKL with the GNU OpenMP layer) and thread
pool as the C kernels; otherwise SciPy BLAS with transposed (F-order) views.
numpy.matmul cannot accumulate into an existing array, which costs a
temporary of the output size plus an extra pass over memory.
'''

import ctypes

import numpy as np
from scipy.linalg import blas as _blas

_GEMM = {np.dtype(np.float64): _blas.dgemm, np.dtype(np.complex128): _blas.zgemm}
_SFX = {np.dtype(np.float64): 'd', np.dtype(np.complex128): 'z'}


def _clib():
    from zccsdt.cc import t3kern
    if t3kern.USE_C and hasattr(t3kern._lib, 'zc_gemm_d'):
        return t3kern._lib
    return None


def _ld(a):
    '''(transposed flag, leading dimension) for a 2-D C- or F-contiguous view.'''
    if a.flags.c_contiguous:
        return 0, a.shape[1]
    if a.flags.f_contiguous:
        return 1, a.shape[0]
    return None


def gemm(A, B, C=None, alpha=1.0, beta=0.0):
    '''C = alpha * A @ B + beta * C in place.  A and B may be C- or
    F-contiguous 2-D views (transposes are free); C must be C-contiguous.'''
    dtype = np.result_type(A, B) if C is None else C.dtype
    if A.dtype != dtype or _ld(A) is None:
        A = np.ascontiguousarray(A, dtype=dtype)
    if B.dtype != dtype or _ld(B) is None:
        B = np.ascontiguousarray(B, dtype=dtype)
    if C is None:
        C = np.empty((A.shape[0], B.shape[1]), dtype=dtype)
        beta = 0.0
    # C must be the caller's own C-contiguous buffer: a reshape of a
    # non-contiguous array would be a silent copy and the result lost
    assert C.flags.c_contiguous and C.shape == (A.shape[0], B.shape[1])
    m, k = A.shape
    n = B.shape[1]
    if k == 0 or m == 0 or n == 0:
        C *= beta
        return C
    lib = _clib()
    if lib is not None:
        ta, lda = _ld(A)
        tb, ldb = _ld(B)
        al = np.array(alpha, dtype=dtype)
        be = np.array(beta, dtype=dtype)
        p = lambda x: x.ctypes.data_as(ctypes.c_void_p)
        i = ctypes.c_int64
        getattr(lib, 'zc_gemm_' + _SFX[np.dtype(dtype)])(
            ctypes.c_int(ta), ctypes.c_int(tb), i(m), i(n), i(k), p(al), p(A),
            i(lda), p(B), i(ldb), p(be), p(C), i(n))
        return C
    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    out = _GEMM[np.dtype(dtype)](alpha, B.T, A.T, beta=beta, c=C.T, overwrite_c=1)
    if not np.shares_memory(out, C):
        C[...] = out.T
    return C
