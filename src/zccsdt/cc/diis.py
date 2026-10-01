'''DIIS extrapolation (same scheme as pyscf.lib.diis.DIIS with incore=True:
error vector = new vector - previous extrapolated vector), with the
large-vector operations done by OpenMP C kernels when available.'''

import ctypes

import numpy as np
import scipy.linalg

from zccsdt.cc import t3pack
from zccsdt.timing import timer


def _vec_c(name, dtype):
    if not t3pack.USE_C or not hasattr(t3pack._lib, 'zc_vdot_d'):
        return None
    return getattr(t3pack._lib, 'zc_%s_%s' % (name, t3pack._sfx(dtype)))


def _sub(a, b, out=None):
    if out is None or out.shape != a.shape or out.dtype != a.dtype:
        out = np.empty_like(a)
    f = _vec_c('vsub', a.dtype)
    if f is None:
        np.subtract(a, b, out=out)
        return out
    f(ctypes.c_int64(a.size), t3pack._p(a), t3pack._p(b), t3pack._p(out))
    return out


def _dot(a, b):
    f = _vec_c('vdot', a.dtype)
    if f is None:
        return np.vdot(a, b)
    res = np.zeros(1, dtype=a.dtype)
    f(ctypes.c_int64(a.size), t3pack._p(a), t3pack._p(b), t3pack._p(res))
    return res[0]


def _lincomb(xs, c, out=None):
    dtype = xs[0].dtype
    c = np.ascontiguousarray(c, dtype=dtype)
    f = _vec_c('vlincomb', dtype)
    if f is None:
        res = c[0] * xs[0]
        for ci, xi in zip(c[1:], xs[1:]):
            res += ci * xi
        if out is None:
            return res
        out[...] = res
        return out
    if out is None:
        out = np.empty_like(xs[0])
    ptrs = (ctypes.c_void_p * len(xs))(*[x.ctypes.data for x in xs])
    f(ctypes.c_int64(out.size), ctypes.c_int64(len(xs)), ptrs, t3pack._p(c),
      t3pack._p(out))
    return out


class DIIS:
    '''DIIS on a vector given as a list of arrays (e.g. [t1, t2, t3] parts),
    so that no concatenated copy is ever formed.  update() returns the
    extrapolated parts; after it, err_norms holds |x_new - x_prev| per part
    (None on the first call).

    The caller may pass the error vectors itself (``errors``: x_new - x_prev
    in any metric, e.g. the scaled Jacobi step), write the extrapolation into
    its own arrays (``out``), and take the arrays of the vector that a full
    subspace is about to drop (``recycle``) to build the next vector in.'''

    def __init__(self, space=8):
        self.space = space
        self.xs, self.es = [], []
        self.H = None
        self.xprev = None
        self.err_norms = None

    def recycle(self):
        '''If the subspace is full, drop its oldest vector now and return its
        (x parts, error parts) for the caller to overwrite; (None, None)
        otherwise.'''
        if len(self.xs) < self.space:
            return None, None
        self.H = self.H[1:, 1:]
        return self.xs.pop(0), self.es.pop(0)

    def update(self, parts, errors=None, out=None):
        x = [np.ascontiguousarray(p).ravel() for p in parts]
        first = self.xprev is None and not self.xs
        if first:
            # the first vector only starts the sequence (as in pyscf)
            self.xprev = x if errors is None else True
            self.err_norms = None
            if out is None:
                return x
            res = []
            for xi, o in zip(x, out):
                if o is None:
                    res.append(xi)
                else:
                    o.ravel()[...] = xi
                    res.append(o.ravel())
            return res
        reuse = None
        if len(self.xs) == self.space:
            self.xs.pop(0)
            reuse = self.es.pop(0)          # recycle the oldest error vectors
            self.H = self.H[1:, 1:]
        if errors is not None:
            e = [np.ascontiguousarray(p).ravel() for p in errors]
        else:
            with timer('diis.sub'):
                e = [_sub(a, b, None if reuse is None else r)
                     for a, b, r in zip(x, self.xprev, reuse or [None] * len(x))]
        with timer('diis.dot'):
            self.err_norms = [np.sqrt(abs(_dot(ei, ei))) for ei in e]
        self.xs.append(x)
        self.es.append(e)
        n = len(self.es)
        dtype = np.result_type(*[xi.dtype for xi in x])
        H = np.zeros((n, n), dtype=dtype)
        if self.H is not None:
            H[:n - 1, :n - 1] = self.H
        with timer('diis.dot'):
            for i in range(n):
                H[n - 1, i] = sum(_dot(a, b) for a, b in zip(e, self.es[i]))
                H[i, n - 1] = np.conj(H[n - 1, i])
        self.H = H
        B = np.zeros((n + 1, n + 1), dtype=dtype)
        B[0, 1:] = B[1:, 0] = 1
        B[1:, 1:] = H
        g = np.zeros(n + 1, dtype=dtype)
        g[0] = 1
        w, v = scipy.linalg.eigh(B)
        if np.any(abs(w) < 1e-14):
            idx = abs(w) > 1e-14
            c = v[:, idx] @ ((v[:, idx].T.conj() @ g) / w[idx])
        else:
            c = np.linalg.solve(B, g)
        outs = [None] * len(x) if out is None else [None if o is None else o.ravel() for o in out]
        with timer('diis.lincomb'):
            res = [_lincomb([xk[j] for xk in self.xs], c[1:], outs[j]) for j in range(len(x))]
        # with caller-supplied errors the previous extrapolation is not kept
        self.xprev = res if errors is None else True
        return res
