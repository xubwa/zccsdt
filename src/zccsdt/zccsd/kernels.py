'''The kernels of the spinor CCSD amplitude equations (csrc/zccsd_kernels.c,
OpenMP) with their numpy equivalents.

Pairs are i > j and a > b in numpy.tril_indices(n, -1) order; the "ring
layout" of a four-index array is the matrix [(i a), (j b)].  All arrays are
C-contiguous complex128 unless said otherwise; the output arguments are
written in place.

libzc_zccsd.so is looked for in zccsdt/lib ($ZCCSDT_ZCCSD_LIB overrides).
Without it, or with USE_C = False, numpy does the same on one thread (the
reference of the tests).
'''

import ctypes
import functools
import os

import numpy as np

_P, _I, _D = ctypes.c_void_p, ctypes.c_int64, ctypes.c_double
_SIGNATURES = {
    'zc_copy': [_P, _I, ctypes.c_int, _P],
    'zc_dotu': [_P, _P, _I, _P],
    'zc_diff_norm2': [_P, _P, _I],
    'zc_outer_pp': [_P, _P, _I, _I, _D, _P],
    'zc_ring_cols': [_P, _I, _I, _I, _I, _P],
    'zc_asym_pack_cols': [_P, _I, _I, _I, _I, _D, _P],
    'zc_asym_pack2': [_P, _I, _I, _I, _I, _D, _P],
    'zc_pack_tril_add': [_P, _I, _I, _I, _I, _D, _P],
    'zc_unpack_tril': [_P, _I, _I, _I, _I, _P],
    'zc_pack_t2': [_P, _I, _I, _P],
    'zc_unpack_t2': [_P, _I, _I, _P],
    'zc_div_packed': [_P, _P, _P, _I, _I],
    'zc_t2p_matvec': [_P, _P, _I, _I, _P],
    'zc_add_rows': [_P, _I, _P, _P, _P, _I, ctypes.c_int],
    'zc_ovvv_t1': [_P, _P, _I, _I, _I, _P, _P, _P],
    'zc_ladder_pack': [_P, _I, _I, _I, _I, _P, _P],
    'zc_ladder_unpack': [_P, _P, _I, _I, _I, _I, _P],
}


def _load():
    path = os.environ.get('ZCCSDT_ZCCSD_LIB') or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lib', 'libzc_zccsd.so')
    try:
        lib = ctypes.CDLL(path)
        for name, args in _SIGNATURES.items():
            f = getattr(lib, name)
            f.argtypes, f.restype = args, None
        lib.zc_diff_norm2.restype = _D
    except (OSError, AttributeError):       # not built, or built from an older source
        return None
    return lib


_lib = _load()
USE_C = _lib is not None


def _c(a, dtype=np.complex128):
    '''The address of an array the C side reads or writes in place.'''
    assert a.dtype == dtype and a.flags.c_contiguous, (a.dtype, a.flags.c_contiguous)
    return a.ctypes.data


@functools.lru_cache(maxsize=None)
def tril(n, k=-1):
    '''numpy.tril_indices(n, k), cached.'''
    i, j = np.tril_indices(n, k)
    i.setflags(write=False)
    j.setflags(write=False)
    return i, j


def npair(n):
    return n * (n - 1) // 2


# --------------------------------------------------------------------- vectors

def copy(src, dst=None, conj=False):
    '''dst = src (its complex conjugate with ``conj``) on all threads.'''
    if dst is None:
        dst = np.empty(src.shape, dtype=np.complex128)
    if USE_C:
        _lib.zc_copy(_c(src), src.size, int(conj), _c(dst))
        return dst
    np.conjugate(src, out=dst) if conj else np.copyto(dst, src)
    return dst


def dotu(x, y):
    '''sum x y, not conjugated.'''
    if USE_C:
        out = np.zeros(1, dtype=np.complex128)
        _lib.zc_dotu(_c(x), _c(y), x.size, _c(out))
        return out[0]
    return np.dot(x.ravel(), y.ravel())


def diff_norm2(x, y):
    '''||x - y||^2.'''
    if USE_C:
        return _lib.zc_diff_norm2(_c(x), _c(y), x.size)
    d = x - y
    return float(np.vdot(d, d).real)


# ------------------------------------------------------------------ amplitudes

def outer_pp(x, y, fac, R):
    '''R[(i>j),(a>b)] += fac (x[i,a] y[j,b] - x[i,b] y[j,a] - x[j,a] y[i,b] + x[j,b] y[i,a]).'''
    o, v = x.shape
    if USE_C:
        _lib.zc_outer_pp(_c(x), _c(y), o, v, fac, _c(R))
        return R
    (io, jo), (av, bv) = tril(o), tril(v)
    xi, xj, yi, yj = x[io], x[jo], y[io], y[jo]
    R += fac * (xi[:, av] * yj[:, bv] - xi[:, bv] * yj[:, av]
                - xj[:, av] * yi[:, bv] + xj[:, bv] * yi[:, av])
    return R


def pack_t2(t2, out=None):
    '''t2p[(i>j),(a>b)] = t2[i,j,a,b].'''
    o, v = t2.shape[0], t2.shape[2]
    if out is None:
        out = np.empty((npair(o), npair(v)), dtype=np.complex128)
    if USE_C:
        _lib.zc_pack_t2(_c(t2), o, v, _c(out))
        return out
    (io, jo), (av, bv) = tril(o), tril(v)
    out[:] = t2[io[:, None], jo[:, None], av, bv]
    return out


def unpack_t2(t2p, o, v, out=None):
    '''t2[i,j,a,b], antisymmetric in both pairs, from t2p[(i>j),(a>b)].'''
    if out is None:
        out = np.empty((o, o, v, v), dtype=np.complex128)
    if USE_C:
        _lib.zc_unpack_t2(_c(t2p), o, v, _c(out))
        return out
    (io, jo), (av, bv) = tril(o), tril(v)
    i, j = io[:, None], jo[:, None]
    out[:] = 0
    out[i, j, av, bv] = t2p
    out[j, i, av, bv] = -t2p
    out[i, j, bv, av] = -t2p
    out[j, i, bv, av] = t2p
    return out


def ring_cols(t2p, o, v, j0=0, j1=None, out=None):
    '''The columns j0 <= j < j1 of t2 in the ring layout: out[(n f),(j b)] =
    t2[n,j,f,b], an (o v) x ((j1 - j0) v) matrix.  The full matrix is
    symmetric.'''
    j1 = o if j1 is None else j1
    if out is None:
        out = np.empty((o * v, (j1 - j0) * v), dtype=np.complex128)
    if USE_C:
        _lib.zc_ring_cols(_c(t2p), o, v, j0, j1, _c(out))
        return out
    t2 = unpack_t2(t2p, o, v)
    out[:] = t2[:, j0:j1].transpose(0, 2, 1, 3).reshape(o * v, (j1 - j0) * v)
    return out


def asym_pack_cols(x, o, v, j0, j1, alpha, R):
    '''R[(i>j),(a>b)] += alpha (x[i,a,j,b] - x[j,a,i,b] - x[i,b,j,a] + x[j,b,i,a])
    from the columns j0 <= j < j1 of x in the ring layout: the terms whose
    third index lies in the range.'''
    if USE_C:
        _lib.zc_asym_pack_cols(_c(x), o, v, j0, j1, alpha, _c(R))
        return R
    (io, jo), (av, bv) = tril(o), tril(v)
    full = np.zeros((o, v, o, v), dtype=np.complex128)
    full[:, :, j0:j1] = x.reshape(o, v, j1 - j0, v)
    i, j = io[:, None], jo[:, None]
    R += alpha * (full[i, av, j, bv] - full[j, av, i, bv] - full[i, bv, j, av] + full[j, bv, i, av])
    return R


def _strides(n, v, order):
    '''(sp, sa) of the element (p, a, b) for the layouts 'pab' and 'apb'.'''
    return (v * v, v) if order == 'pab' else (v, n * v)


def asym_pack2(x, alpha, R, order='pab'):
    '''R[p,(a>b)] += alpha (x[p,a,b] - x[p,b,a]) for x[p,a,b] ('pab') or
    x[a,p,b] ('apb').'''
    n, v = R.shape[0], x.shape[2]
    if USE_C:
        _lib.zc_asym_pack2(_c(x), n, v, *_strides(n, v, order), alpha, _c(R))
        return R
    av, bv = tril(v)
    R += alpha * ((x[:, av, bv] - x[:, bv, av]) if order == 'pab'
                  else (x[av, :, bv] - x[bv, :, av]).T)
    return R


def unpack_tril(xp, v, order='pab', out=None):
    '''The antisymmetric matrices of the rows of xp[p,(a>b)]: out[p,a,b]
    ('pab') or out[a,p,b] ('apb').'''
    n = xp.shape[0]
    if out is None:
        out = np.empty((n, v, v) if order == 'pab' else (v, n, v), dtype=np.complex128)
    if USE_C:
        _lib.zc_unpack_tril(_c(xp), n, v, *_strides(n, v, order), _c(out))
        return out
    av, bv = tril(v)
    x = np.zeros((n, v, v), dtype=np.complex128)
    x[:, av, bv] = xp
    x[:, bv, av] = -xp
    out[:] = x if order == 'pab' else x.transpose(1, 0, 2)
    return out


def pack_tril_add(x, alpha, R, order='pab'):
    '''R[p,(a>b)] += alpha x[p,a,b] for x[p,a,b] ('pab') or x[a,p,b] ('apb').'''
    n, v = R.shape[0], x.shape[2]
    if USE_C:
        _lib.zc_pack_tril_add(_c(x), n, v, *_strides(n, v, order), alpha, _c(R))
        return R
    av, bv = tril(v)
    R += alpha * (x[:, av, bv] if order == 'pab' else x[av, :, bv].T)
    return R


def div_packed(R, eo, ev):
    '''R[(i>j),(a>b)] /= eo[i] + eo[j] - ev[a] - ev[b].'''
    o, v = eo.size, ev.size
    eo, ev = np.ascontiguousarray(eo, dtype=np.float64), np.ascontiguousarray(ev, dtype=np.float64)
    if USE_C:
        _lib.zc_div_packed(_c(R), _c(eo, np.float64), _c(ev, np.float64), o, v)
        return R
    (io, jo), (av, bv) = tril(o), tril(v)
    R /= (eo[io] + eo[jo])[:, None] - (ev[av] + ev[bv])[None, :]
    return R


def t2p_matvec(t2p, F, out):
    '''out[i,a] += sum_me t2[i,m,a,e] F[m,e].'''
    o, v = F.shape
    if USE_C:
        _lib.zc_t2p_matvec(_c(t2p), _c(F), o, v, _c(out))
        return out
    out += np.einsum('imae,me->ia', unpack_t2(t2p, o, v), F)
    return out


def add_rows(R, rows, signs, Y, conj=False):
    '''R[rows[k]] += signs[k] Y[k] (conj(Y[k]) with ``conj``); rows < 0 are
    skipped, the others distinct.'''
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    signs = np.ascontiguousarray(signs, dtype=np.float64)
    if USE_C:
        _lib.zc_add_rows(_c(R), R.shape[1], _c(rows, np.int64), _c(signs, np.float64), _c(Y),
                         rows.size, int(conj))
        return R
    k = rows >= 0
    R[rows[k]] += signs[k, None] * (Y[k].conj() if conj else Y[k])
    return R


# -------------------------------------------------------------------- <ma||ef>

def ovvv_t1(Om, t1, m, Wm, fv, x1):
    '''The <mb||ef> t1 terms of the occupied m from Om[b,(e>f)] = <mb||ef>:
    with A[b,e,j] = sum_f <mb||ef> t1[j,f], Wm[e,(j b)] += A[b,e,j] (the rows
    (m e) of W in the ring layout), fv[b,e] = A[b,e,m] and x1[j,b] =
    sum_e t1[m,e] A[b,e,j].'''
    o, v = t1.shape
    if USE_C:
        _lib.zc_ovvv_t1(_c(Om), _c(t1), v, o, m, _c(Wm), _c(fv), _c(x1))
        return
    av, bv = tril(v)
    U = np.zeros((v, v, v), dtype=np.complex128)
    U[:, av, bv] = Om
    U[:, bv, av] = -Om
    A = (U.reshape(v * v, v) @ t1.T).reshape(v, v, o)
    Wm.reshape(v, o, v)[:] += A.transpose(1, 2, 0)
    fv[:] = A[:, :, m]
    x1[:] = np.einsum('e,bej->jb', t1[m], A)


# ------------------------------------------------------------------- AO ladder

def ladder_pack(X, n, nao, order='xpy'):
    '''The spin blocks of X[p] (n matrices over (spin, AO), X[x,p,y] for
    'xpy', X[p,x,y] for 'pxy') as real rows: AR (6 n, nao (nao - 1) / 2) with
    aa, bb, (ab - ab^T) / 2, real parts then imaginary parts, and SR
    (2 n, nao (nao + 1) / 2) with (ab + ab^T) / 2.'''
    n2 = 2 * nao
    AR = np.empty((6 * n, nao * (nao - 1) // 2))
    SR = np.empty((2 * n, nao * (nao + 1) // 2))
    if USE_C:
        sp, sx = (n2 * n2, n2) if order == 'pxy' else (n2, n * n2)
        _lib.zc_ladder_pack(_c(X), n, nao, sp, sx, _c(AR, np.float64), _c(SR, np.float64))
        return AR, SR
    X = X.reshape(n, n2, n2) if order == 'pxy' else X.reshape(n2, n, n2).transpose(1, 0, 2)
    (pi, pj), (si, sj) = tril(nao), tril(nao, 0)
    ab = X[:, :nao, nao:]
    anti = np.concatenate((X[:, :nao, :nao][:, pi, pj], X[:, nao:, nao:][:, pi, pj],
                           0.5 * (ab[:, pi, pj] - ab[:, pj, pi])))
    sym = 0.5 * (ab[:, si, sj] + ab[:, sj, si])
    AR[:3 * n], AR[3 * n:] = anti.real, anti.imag
    SR[:n], SR[n:] = sym.real, sym.imag
    return AR, SR


def ladder_unpack(GA, GS, n, nao, order='xpy'):
    '''The inverse layout for GA = AR W_same and GS = SR W_plus: G[p] with aa
    and bb antisymmetric, ab = GS + GA_ab and ba = -ab^T, as G[x,p,y]
    ('xpy') or G[p,x,y] ('pxy').'''
    n2 = 2 * nao
    G = np.empty((n2, n, n2) if order == 'xpy' else (n, n2, n2), dtype=np.complex128)
    if USE_C:
        sp, sx = (n2 * n2, n2) if order == 'pxy' else (n2, n * n2)
        _lib.zc_ladder_unpack(_c(GA, np.float64), _c(GS, np.float64), n, nao, sp, sx, _c(G))
        return G
    (pi, pj), (si, sj) = tril(nao), tril(nao, 0)
    Ga = GA[:3 * n] + 1j * GA[3 * n:]
    Gs = GS[:n] + 1j * GS[n:]
    g = np.zeros((n, n2, n2), dtype=np.complex128)
    aa, bb, ab, ba = g[:, :nao, :nao], g[:, nao:, nao:], g[:, :nao, nao:], g[:, nao:, :nao]
    aa[:, pi, pj] = Ga[:n]
    aa[:, pj, pi] = -Ga[:n]
    bb[:, pi, pj] = Ga[n:2 * n]
    bb[:, pj, pi] = -Ga[n:2 * n]
    ab[:, si, sj] = Gs
    ab[:, sj, si] = Gs
    ab[:, pi, pj] += Ga[2 * n:]
    ab[:, pj, pi] -= Ga[2 * n:]
    ba[:] = -ab.transpose(0, 2, 1)
    G[:] = g if order == 'pxy' else g.transpose(1, 0, 2)
    return G
