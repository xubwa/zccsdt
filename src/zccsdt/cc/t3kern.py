'''ctypes binding of csrc/ccsdt_t3.c (T3 part of the CCSDT iteration in C).

Layouts are documented in the C file and match ccsdt_opt's numpy path.
'''

import ctypes
import os

import numpy as np

from zccsdt.cc.t3pack import _sfx
from zccsdt import timing


def _load():
    '''Prefer the MKL build (BLAS threads share the OpenMP pool), then the
    OpenBLAS build; $ZCCSDT_T3_LIB overrides.'''
    libdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lib')
    env = os.environ.get('ZCCSDT_T3_LIB')
    cands = [env] if env else [os.path.join(libdir, 'libzc_ccsdt_t3_mkl.so'),
                               os.path.join(libdir, 'libzc_ccsdt_t3.so')]
    for path in cands:
        try:
            return ctypes.CDLL(path)
        except OSError:
            continue
    return None


_lib = _load()
USE_C = _lib is not None


def _p(a):
    assert a.flags.c_contiguous
    return a.ctypes.data_as(ctypes.c_void_p)


def _i(x):
    return ctypes.c_int64(int(x))


def _c(a, dtype):
    return np.ascontiguousarray(a, dtype=dtype)


def _record(name, stats):
    for k, part in enumerate(('gather', 'gemm', 'reduce', 'drive')):
        if k >= len(stats) or (k == 3 and stats[k] == 0):
            continue
        timing.add('%s.%s' % (name, part), stats[k])


def _io(io):
    '''Pointer to the description of the mapped arrays (scratch.io_desc) or NULL.'''
    return None if io is None else _p(io)


def pass1(T3p, om, vm, batch, GPd, GmQ, gpp, fT, io=None):
    '''Returns Xacc (nv+no, no*nvp), Yall (npo, nv, ny), r1 (nv*no), rA (npo, nvp).
    ``io``: scratch.io_desc(T3, R) if either is a view of a mapped file (the
    kernels then read ahead, start the writes and unmap what they are done with).'''
    dt = T3p.dtype
    no, nv = om.n, vm.n
    ny = GmQ.shape[1]
    Xacc = np.zeros((nv + no, no * vm.npair), dtype=dt)
    Yall = np.zeros((om.npair, nv, ny), dtype=dt)
    r1 = np.zeros(nv * no, dtype=dt)
    rA = np.zeros((om.npair, vm.npair), dtype=dt)
    GPd, GmQ, gpp, fT = (_c(x, dt) for x in (GPd, GmQ, gpp, fT))
    st = np.zeros(4)
    getattr(_lib, 'zc_t3_pass1_' + _sfx(dt))(
        _p(T3p), _i(om.ntri), _i(vm.ntri), _i(no), _i(nv), _i(om.npair),
        _i(vm.npair), _p(om.tri_c), _p(om.sgn_c), _p(vm.tri_c), _p(vm.sgn_c),
        _i(batch), _p(GPd), _p(GmQ), _i(ny), _p(gpp), _p(fT),
        _p(Xacc), _p(Yall), _p(r1), _p(rA), _io(io), _p(st))
    _record('pass1', st)
    return Xacc, Yall, r1, rA


def pass2(T3p, om, vm, batch, Wring, Adrv, Bdrv, R, io=None):
    dt = T3p.dtype
    Wring, Adrv, Bdrv = (_c(x, dt) for x in (Wring, Adrv, Bdrv))
    st = np.zeros(4)
    getattr(_lib, 'zc_t3_pass2_' + _sfx(dt))(
        _p(T3p), _i(om.ntri), _i(vm.ntri), _i(om.n), _i(vm.n), _i(om.npair),
        _i(vm.npair), _p(om.tri_c), _p(om.sgn_c), _p(om.free_c), _p(om.pair_c),
        _p(vm.tri_c), _p(vm.sgn_c), _p(vm.free_c), _p(vm.pair_c), _i(batch),
        _p(Wring), _p(Adrv), _p(Bdrv), _p(R), _io(io), _p(st))
    _record('pass2', st)


def pp_ladder(T3p, om, vm, batch, Wpp, R, io=None):
    dt = T3p.dtype
    st = np.zeros(4)
    getattr(_lib, 'zc_t3_ppl_' + _sfx(dt))(
        _p(T3p), _i(om.ntri), _i(vm.ntri), _i(vm.n), _i(vm.npair),
        _p(vm.tri_c), _p(vm.sgn_c), _p(vm.free_c), _p(vm.pair_c), _i(batch),
        _p(_c(Wpp, dt)), _p(R), _io(io), _p(st))
    _record('ppl', st)


def update_into_r(T3p, R, eo, ev, io=None):
    '''In place: R <- T3p + R / (eo[:, None] - ev[None, :]).'''
    getattr(_lib, 'zc_t3_update_' + _sfx(T3p.dtype))(
        _p(T3p), _p(R), _i(T3p.shape[0]), _i(T3p.shape[1]),
        _p(_c(eo, np.float64)), _p(_c(ev, np.float64)), _io(io))


def occ_operator(O6, T3p, R, cols=0, io=None):
    '''R += O6 @ T3p (in place, BLAS beta = 1), ``cols`` columns per GEMM
    (0: all at once).'''
    dt = T3p.dtype
    getattr(_lib, 'zc_t3_occop_' + _sfx(dt))(
        _p(_c(O6, dt)), _p(T3p), _i(T3p.shape[0]), _i(T3p.shape[1]), _i(cols),
        _io(io), _p(R))


# The hh ladder and F_oo: from this many occupied spinors on, pair-driven
# (measured, docs/factorization.md 2.3); below, the matrix on packed triples.
HH_PAIR_MIN_OCC = int(os.environ.get('ZCCSDT_HH_PAIR_MIN_OCC', 16))
HH_PAIR_COLS = int(os.environ.get('ZCCSDT_HH_PAIR_COLS', 96))


def hh_lists(om, rows=None):
    '''For the block lo <= I < hi of the occupied triples (``rows``; default
    all) and every spectator index k, the triples of the block that contain
    k: (koff, krow, kpair, ksgn, woff) with the entries koff[k] <= e <
    koff[k+1] holding the local row krow[e] = tri[k, P] - lo, the pair
    kpair[e] = P and the sign ksgn[e] = sgn[k, P]; woff[k]: where the matrix
    of spectator k starts in the array of all of them (woff[-1]: its
    length).  Kept on the map.'''
    lo, hi = (0, om.ntri) if rows is None else rows
    cache = om.__dict__.setdefault('_hh_lists', {})
    if (lo, hi) not in cache:
        mask = (om.sgn != 0) & (om.tri >= lo) & (om.tri < hi)
        ks, Ps = np.nonzero(mask)                           # (sorted by k, then P)
        koff = np.searchsorted(ks, np.arange(om.n + 1)).astype(np.int64)
        nk = np.diff(koff)
        woff = np.concatenate(([0], np.cumsum(nk * nk))).astype(np.int64)
        cache[lo, hi] = (koff, np.ascontiguousarray(om.tri[ks, Ps] - lo, dtype=np.int64),
                         np.ascontiguousarray(Ps, dtype=np.int64),
                         np.ascontiguousarray(om.sgn[ks, Ps], dtype=np.int8), woff)
    return cache[lo, hi]


class OccOperator:
    '''The hh ladder and the F_oo term of the T3 equation,
      (O t)[ijk] = P(k/ij) 1/2 sum_mn W[m,n,i,j] t[mnk] - P(i/jk) sum_m F[m,i] t[mjk]
    (``trans``: its transpose, as in the Lambda equations), applied to blocks
    of packed arrays, R[lo:hi] += O[lo:hi, lo:hi] T[lo:hi], in one of two forms
    (docs/factorization.md 2.3):

      'matrix'  O on packed triples, C(o,3)^2, written down directly
                (t3pack.occ_operator_direct), and one GEMM;
      'pair'    the triples as (spectator; pair): a GEMM with W[(ij), (mn)]
                per spectator on gathered rows, o / 9 of the flops, T3 read
                three times instead of once.

    ``mode`` None: 'pair' from HH_PAIR_MIN_OCC occupied spinors on.  Arrays
    in a mapped file (``io``) always take the matrix, whose GEMM touches
    column blocks of a chosen size.'''

    def __init__(self, om, W, F, dtype, trans=False, mode=None):
        self.om, self.dtype, self.trans = om, np.dtype(dtype), trans
        self.W = np.ascontiguousarray(W, dtype=dtype)
        self.F = np.ascontiguousarray(F, dtype=dtype)
        if mode is None:
            mode = 'pair' if om.n >= HH_PAIR_MIN_OCC else 'matrix'
        if not (USE_C and hasattr(_lib, 'zc_t3_hhpair_d')) \
                or self.dtype not in (np.float64, np.complex128):
            mode = 'matrix'
        self.mode = mode
        self._mat, self._wk = {}, {}

    def matrix(self, rows=None):
        '''O[lo:hi, lo:hi] (``rows`` = (lo, hi); default: all of it).'''
        from zccsdt.cc import t3pack
        key = (0, self.om.ntri) if rows is None else tuple(rows)
        if key not in self._mat:
            O = t3pack.occ_operator_direct(self.om, self.W, self.F, key, self.dtype)
            self._mat = {key: np.ascontiguousarray(O.T) if self.trans else O}   # (one block is kept)
        return self._mat[key]

    def pair_matrices(self, rows=None):
        key = (0, self.om.ntri) if rows is None else tuple(rows)
        if key not in self._wk:
            om = self.om
            koff, krow, kpair, ksgn, woff = hh_lists(om, key)
            Wk = np.empty(int(woff[-1]), dtype=self.dtype)
            getattr(_lib, 'zc_t3_hhpair_w_' + _sfx(self.dtype))(
                _p(Wk), _p(woff), _i(om.n), _p(koff), _p(kpair), _p(ksgn), _p(om.pa_c),
                _p(om.pb_c), _p(self.W), _p(self.F), ctypes.c_int(int(self.trans)))
            self._wk = {key: Wk}
        return self._wk[key]

    def nbytes(self, rows=None):
        '''Bytes of what apply() allocates for the block ``rows``.'''
        lo, hi = (0, self.om.ntri) if rows is None else rows
        if self.mode == 'matrix':
            return (hi - lo) ** 2 * self.dtype.itemsize
        koff, woff = hh_lists(self.om, (lo, hi))[0], hh_lists(self.om, (lo, hi))[4]
        nthr = int(os.environ.get('OMP_NUM_THREADS') or len(os.sched_getaffinity(0)))
        return int(woff[-1] + 2 * nthr * HH_PAIR_COLS * np.diff(koff).max(initial=0)) \
            * self.dtype.itemsize

    def apply(self, T, R, rows=None, cols=0, io=None):
        '''R += O[rows, rows] @ T for T, R of shape (hi - lo, N), C-contiguous.
        ``cols``: the columns per GEMM of the matrix form (0: all).'''
        if T.shape[0] == 0 or T.shape[1] == 0:
            return R
        if self.mode == 'matrix' or io is not None:
            O = self.matrix(rows)
            if USE_C:
                occ_operator(O, T, R, cols, io)
            else:
                R += O @ T
            return R
        koff, krow, _, _, woff = hh_lists(self.om, rows)
        Wk = self.pair_matrices(rows)
        getattr(_lib, 'zc_t3_hhpair_' + _sfx(self.dtype))(
            _p(T), _p(R), _i(T.shape[0]), _i(T.shape[1]), _i(self.om.n), _p(koff), _p(krow),
            _p(Wk), _p(woff), _i(HH_PAIR_COLS))
        return R


def zero(a):
    '''a[...] = 0 for a C-contiguous float64 / complex128 array, threaded.'''
    from zccsdt.cc import t3pack
    lib = t3pack._lib
    if t3pack.USE_C and hasattr(lib, 'zc_vzero_d') and a.flags.c_contiguous \
            and a.dtype in (np.float64, np.complex128):
        lib.zc_vzero_d(_i(a.size * a.itemsize // 8), _p(a))
    else:
        a[...] = 0
    return a
