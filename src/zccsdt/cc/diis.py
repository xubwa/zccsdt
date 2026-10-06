'''DIIS extrapolation (the scheme of pyscf.lib.diis.DIIS: error vector = new
vector - previous extrapolated vector), with the large-vector operations done
by OpenMP C kernels when available.

Storage follows the PySCF settings.  With ``incore=False`` (the default) the
parts of a vector with at least INCORE_SIZE elements are kept in an HDF5 file:
``filename`` if given (kept), else a temporary file in the scratch directory
(ZCCSDT_TMPDIR, else PySCF's lib.param.TMPDIR, which follows PYSCF_TMPDIR /
TMPDIR).  Smaller parts, and everything with ``incore=True``, stay in memory.
A caller that plans its memory and has decided for the file passes its own
threshold (``incore_size``): with the default a vector of 7 million complex
numbers stays in memory, 12 times over for a subspace of 6.

The file-backed vectors are contiguous HDF5 datasets that are accessed through
numpy.memmap instead of explicit reads and writes (zccsdt.scratch): the
kernels work directly on the page cache, so nothing is copied while the
vectors fit in memory, and the operating system writes them out and pages
them back in when they do not.
'''

import ctypes

import numpy as np
import scipy.linalg

from zccsdt.cc import t3pack
from zccsdt.scratch import Scratch, scratch_dir, written, unmap          # noqa: F401
from zccsdt.timing import timer

INCORE_SIZE = 10_000_000        # elements; smaller parts always stay in memory


def _vec_c(name, dtype):
    if not t3pack.USE_C or not hasattr(t3pack._lib, 'zc_vdot_d'):
        return None
    return getattr(t3pack._lib, 'zc_%s_%s' % (name, t3pack._sfx(dtype)), None)


def _sub(a, b, out=None):
    if out is None or out.shape != a.shape or out.dtype != a.dtype:
        out = np.empty(a.shape, dtype=a.dtype)
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


def _mdot(a, xs):
    '''[vdot(a, x) for x in xs] in one pass over a.'''
    f = _vec_c('vmdot', a.dtype)
    if f is None or len(xs) > 64:
        return np.array([_dot(a, x) for x in xs])
    res = np.zeros(len(xs), dtype=a.dtype)
    ptrs = (ctypes.c_void_p * len(xs))(*[x.ctypes.data for x in xs])
    f(ctypes.c_int64(a.size), ctypes.c_int64(len(xs)), t3pack._p(a), ptrs, t3pack._p(res))
    return res


def _lincomb(xs, c, out=None):
    dtype = xs[0].dtype
    c = np.ascontiguousarray(c, dtype=dtype)
    f = _vec_c('vlincomb', dtype)
    if f is None:
        res = c[0] * xs[0]
        for ci, xi in zip(c[1:], xs[1:]):
            res += ci * xi
        if out is None:
            return np.asarray(res)
        out[...] = res
        return out
    if out is None:
        out = np.empty(xs[0].shape, dtype=dtype)
    ptrs = (ctypes.c_void_p * len(xs))(*[x.ctypes.data for x in xs])
    f(ctypes.c_int64(out.size), ctypes.c_int64(len(xs)), ptrs, t3pack._p(c),
      t3pack._p(out))
    return out


def _copy(dst, src):
    if src.dtype == dst.dtype and _vec_c('vlincomb', dst.dtype) is not None:
        return _lincomb([src], [1.0], dst)
    dst[...] = src
    return dst


def _stored(a):
    '''A part was written into a mapped array: send it to the file now and
    drop it from the page tables (see zccsdt.scratch).'''
    written(a)
    unmap(a)
    return a


def _same(a, b):
    '''Do the 1-D contiguous arrays a and b cover the same memory?'''
    return (b is not None and a.size == b.size and a.dtype == b.dtype
            and a.__array_interface__['data'][0] == b.__array_interface__['data'][0])


def _Scratch(filename=None):
    '''Scratch file of the large DIIS vectors, datasets 'x<slot>.<part>' /
    'e<slot>.<part>'.'''
    return Scratch(filename, prefix='zccsdt_diis_', what='a DIIS vector',
                   hint=' or diis_file, a smaller diis_space, or keep the vectors in '
                        'memory (incore_complete)')


class DIIS:
    '''DIIS on a vector given as a list of arrays (e.g. [t1, t2, t3] parts),
    so that no concatenated copy is ever formed.  update() returns the
    extrapolated parts; after it, err_norms holds |x_new - x_prev| per part
    (None on the first call).

    ``filename`` / ``incore`` choose where the stored vectors live (module
    docstring).  The caller may pass the error vectors itself (``errors``:
    x_new - x_prev in any metric, e.g. the scaled Jacobi step) and have the
    extrapolation written into its own arrays (``out``).  To avoid copies it
    can also build the new vector directly in the storage: buffers() returns
    the arrays of the vector that the next update() stores.'''

    def __init__(self, space=8, filename=None, incore=False, incore_size=None):
        self.space = space
        self.filename = filename
        self.incore = incore
        # parts with fewer elements stay in memory whatever ``incore`` says
        self.incore_size = INCORE_SIZE if incore_size is None else int(incore_size)
        self.xs, self.es = [], []           # stored vectors, oldest first
        self.H = None
        self.xprev = None
        self.err_norms = None
        self._scratch = None
        self._nslot = 0
        self._free = []                     # storage of vectors not in the subspace
        self._pending = None                # storage handed out by buffers()

    # ---------------------------------------------------------------- storage

    def on_disk(self):
        '''Is any part of the stored vectors file-backed?'''
        return self._scratch is not None

    def _new_part(self, like, key, in_memory=True):
        '''Storage of one part: a file-backed array if it is large and not
        incore, else an array in memory (None without ``in_memory``).'''
        if self.incore or like.size < self.incore_size:
            return np.empty(like.size, dtype=like.dtype) if in_memory else None
        if self._scratch is None:
            self._scratch = _Scratch(self.filename)
        return self._scratch.create(key, like.size, like.dtype)

    def _take(self, like, own_x=True):
        '''Storage (x parts, error parts) for one more vector: that of the
        oldest vector if the subspace is full (which is dropped), else unused
        or new storage.  Without ``own_x`` the x parts that would live in
        memory are left to the caller's arrays (None).'''
        if len(self.xs) >= self.space:
            self.H = self.H[1:, 1:]
            return self.xs.pop(0), self.es.pop(0)
        if self._free:
            return self._free.pop()
        k = self._nslot
        self._nslot += 1
        like = [np.asarray(a) for a in like]
        return ([self._new_part(a, 'x%d.%d' % (k, j), own_x) for j, a in enumerate(like)],
                [self._new_part(a, 'e%d.%d' % (k, j)) for j, a in enumerate(like)])

    def buffers(self, like):
        '''(x parts, error parts) of flat arrays, shaped like the arrays of
        ``like``, in which the caller builds the next vector and its error
        before passing them to update(); they are stored without a copy.
        File-backed for large parts unless incore.'''
        if self._pending is None:
            self._pending = self._take(like)
        return self._pending

    def _adopt(self, a, s):
        '''The array to store for a part given as a, with storage s.'''
        if s is None or _same(a, s):
            return a if s is None else s
        if isinstance(s, np.memmap):
            return _stored(_copy(s, a))
        return a

    def close(self):
        '''Drop the stored vectors and close the scratch file.'''
        self.xs, self.es, self._free, self._pending = [], [], [], None
        self.H = None
        if self._scratch is not None:
            self._scratch.close()
            self._scratch = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ----------------------------------------------------------------- update

    def update(self, parts, errors=None, out=None):
        x = [np.ascontiguousarray(p).ravel() for p in parts]
        pending, self._pending = self._pending, None
        if self.xprev is None and not self.xs:
            # the first vector only starts the sequence (as in pyscf)
            self.err_norms = None
            own = []
            if pending is not None:
                self._free.append(pending)
                own = pending[0]
            res = []
            for j, xi in enumerate(x):
                o = None if out is None else out[j]
                if o is not None:
                    res.append(_stored(_copy(o.ravel(), xi)))
                elif any(_same(xi, s) for s in own):
                    res.append(np.array(xi))        # never hand out a storage array
                else:
                    res.append(xi)
            self.xprev = res if errors is None else True
            return res
        sx, se = pending if pending is not None else self._take(x, own_x=False)
        x = [self._adopt(xi, si) for xi, si in zip(x, sx)]
        if errors is not None:
            e = [self._adopt(np.ascontiguousarray(p).ravel(), si) for p, si in zip(errors, se)]
        else:
            if self.xprev is True:
                raise ValueError('update() without errors after a call with errors')
            with timer('diis.sub'):
                e = [_stored(_sub(a, b, r)) for a, b, r in zip(x, self.xprev, se)]
        for part in x + e:              # written by the caller or just above
            _stored(part)
        self.xs.append(x)
        self.es.append(e)
        n = len(self.es)
        dtype = np.result_type(*[xi.dtype for xi in x])
        H = np.zeros((n, n), dtype=dtype)
        if self.H is not None:
            H[:n - 1, :n - 1] = self.H
        with timer('diis.dot'):
            # all overlaps of the new error vector in one pass per part
            d = [_mdot(ej, [ek[j] for ek in self.es]) for j, ej in enumerate(e)]
        for ek in self.es:
            for part in ek:
                unmap(part)
        self.err_norms = [np.sqrt(abs(dj[n - 1])) for dj in d]
        H[n - 1, :] = sum(d)
        H[:, n - 1] = np.conj(H[n - 1, :])
        self.H = H
        # The overlaps relative to the largest one: the coefficients do not
        # depend on the scale of H, the threshold for dependent error vectors
        # does.  With an absolute one (pyscf.lib.diis has 1e-14) all of H is
        # below it once the errors are below 1e-7, and what is left of the
        # equations gives the mean of the stored vectors - slower than no
        # extrapolation at all (HBr: 0.82 per cycle from 1e-8 on, against
        # 0.64 without DIIS).
        scale = abs(np.diag(H)).max()
        Hs = H / scale if scale > 0 else H
        # Overlaps that are real to the rounding - those of vectors that are
        # symmetric under time reversal (Kramers pairs) - give real
        # coefficients: a real combination of such vectors is symmetric
        # again, while the imaginary parts that the rounding of a nearly
        # dependent system leaves in the coefficients broke the symmetry at
        # 1e-11.
        if np.iscomplexobj(Hs) and abs(Hs.imag).max() < 1e-10:
            Hs = Hs.real
        B = np.zeros((n + 1, n + 1), dtype=Hs.dtype)
        B[0, 1:] = B[1:, 0] = 1
        B[1:, 1:] = Hs
        g = np.zeros(n + 1, dtype=Hs.dtype)
        g[0] = 1
        w, v = scipy.linalg.eigh(B)
        if np.any(abs(w) < 1e-14):
            idx = abs(w) > 1e-14
            c = v[:, idx] @ ((v[:, idx].T.conj() @ g) / w[idx])
        else:
            c = np.linalg.solve(B, g)
        outs = [None] * len(x) if out is None else [None if o is None else o.ravel() for o in out]
        with timer('diis.lincomb'):
            res = [_stored(_lincomb([xk[j] for xk in self.xs], c[1:], outs[j]))
                   for j in range(len(x))]
        for xk in self.xs:
            for part in xk:
                unmap(part)
        # with caller-supplied errors the previous extrapolation is not kept
        self.xprev = res if errors is None else True
        return res
