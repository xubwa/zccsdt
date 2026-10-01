'''ctypes binding of csrc/ccsdt_t3.c (T3 part of the CCSDT iteration in C).

Layouts are documented in the C file and match ccsdt_opt's numpy path.
'''

import ctypes
import os

import numpy as np

from zccsdt.cc.t3pack import _sfx


def _load():
    path = os.environ.get('ZCCSDT_T3_LIB') or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'lib', 'libzc_ccsdt_t3.so')
    try:
        return ctypes.CDLL(path)
    except OSError:
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


def pass1(T3p, om, vm, batch, GPd, GmQ, gpp, fT):
    '''Returns Xacc (nv+no, no*nvp), Yall (npo, nv, ny), r1 (nv*no), rA (npo, nvp).'''
    dt = T3p.dtype
    no, nv = om.n, vm.n
    ny = GmQ.shape[1]
    Xacc = np.zeros((nv + no, no * vm.npair), dtype=dt)
    Yall = np.zeros((om.npair, nv, ny), dtype=dt)
    r1 = np.zeros(nv * no, dtype=dt)
    rA = np.zeros((om.npair, vm.npair), dtype=dt)
    GPd, GmQ, gpp, fT = (_c(x, dt) for x in (GPd, GmQ, gpp, fT))
    getattr(_lib, 'zc_t3_pass1_' + _sfx(dt))(
        _p(T3p), _i(om.ntri), _i(vm.ntri), _i(no), _i(nv), _i(om.npair),
        _i(vm.npair), _p(om.tri_c), _p(om.sgn_c), _p(vm.tri_c), _p(vm.sgn_c),
        _i(batch), _p(GPd), _p(GmQ), _i(ny), _p(gpp), _p(fT),
        _p(Xacc), _p(Yall), _p(r1), _p(rA))
    return Xacc, Yall, r1, rA


def pass2(T3p, om, vm, batch, Wring, t2mnT, WvT2, WoT, t2P, R):
    dt = T3p.dtype
    Wring, t2mnT, WvT2, WoT, t2P = (_c(x, dt) for x in (Wring, t2mnT, WvT2, WoT, t2P))
    getattr(_lib, 'zc_t3_pass2_' + _sfx(dt))(
        _p(T3p), _i(om.ntri), _i(vm.ntri), _i(om.n), _i(vm.n), _i(om.npair),
        _i(vm.npair), _p(om.tri_c), _p(om.sgn_c), _p(om.free_c), _p(om.pair_c),
        _p(vm.tri_c), _p(vm.sgn_c), _p(vm.free_c), _p(vm.pair_c), _i(batch),
        _p(Wring), _p(t2mnT), _p(WvT2), _p(WoT), _p(t2P), _p(R))


def pp_ladder(T3p, om, vm, batch, Wpp, R):
    dt = T3p.dtype
    getattr(_lib, 'zc_t3_ppl_' + _sfx(dt))(
        _p(T3p), _i(om.ntri), _i(vm.ntri), _i(vm.n), _i(vm.npair),
        _p(vm.tri_c), _p(vm.sgn_c), _p(vm.free_c), _p(vm.pair_c), _i(batch),
        _p(_c(Wpp, dt)), _p(R))


def update(T3p, R, eo, ev):
    '''In place: T3p += R / (eo[:, None] - ev[None, :]).'''
    getattr(_lib, 'zc_t3_update_' + _sfx(T3p.dtype))(
        _p(T3p), _p(R), _i(T3p.shape[0]), _i(T3p.shape[1]),
        _p(_c(eo, np.float64)), _p(_c(ev, np.float64)))
