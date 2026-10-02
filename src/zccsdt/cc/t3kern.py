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
        timing._totals['%s.%s' % (name, part)] += stats[k]
        timing._counts['%s.%s' % (name, part)] += 1


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
