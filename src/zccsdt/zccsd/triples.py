'''The (T) correction of the spinor CCSD in batches of virtual pairs.

The equations are those of PySCF's gccsd_t (and of zccsdt.zccsd.gccsd_t):
with <bc||ei> and <ma||jk> the integrals and D the orbital-energy
denominator,

    x[ijk,abc] = sum_e t2[j,k,a,e] <bc||ei> - sum_m t2[i,m,b,c] <ma||jk>
    w = P(a/bc) P(i/jk) x,      v = w + P(a/bc) P(i/jk) (t1[i,a] <bc||jk> + f[a,i] t2[j,k,b,c])
    E(T) = sum_{i>j>k, a>b>c} conj(v) w / D.

x is antisymmetric in (j,k) and in (b,c), so everything is kept on packed
pairs (numpy.tril_indices order):

    x_{a,(bc)}[i, (jk)] = sum_e <bc||ei> t2[j,k,a,e] - sum_m t2[i,m,b,c] <ma||jk>

is, for fixed a and pair (b>c), an o x o(o-1)/2 matrix, and for a > b > c

    Z[i, (jk)] = x_{a,(bc)} - x_{b,(ac)} + x_{c,(ab)},
    w[i>j>k]   = Z[i, (jk)] - Z[j, (ik)] + Z[k, (ij)].

The triples are done in batches of one a and a range of b (all c < b): the
x of a batch are six GEMMs on contiguous blocks of

    Op[(b>c), i, e] = <bc||ei>     (the stored <ie||bc>, transposed once)
    t2u[a, (j>k), e] = t2[j,k,a,e]
    t2x[(b>c), i, m] = t2[i,m,b,c]
    Mh[a, (j>k), m]  = <ma||jk>

(o^3 v^4 / 4 multiply-adds for the particle term in all, each x made once),
and an OpenMP kernel (csrc/zccsd_t.c) assembles Z for one (b, c) per thread
in its cache and adds up the energy.

Memory: Op is kept in memory if it fits next to the smallest batch, else it
is a mapped array of a scratch file (its rows are read in contiguous
blocks); the batches are sized from what ``max_memory`` leaves (at most
BATCH_BYTES); MemoryError if not even the batch of one b fits.

Timers (zccsdt.timing): 't.setup', 't.gemm', 't.kernel'.
'''

import ctypes

import numpy as np
from pyscf.lib import logger

from zccsdt import blas, memory, scratch
from zccsdt.timing import timer
from zccsdt.zccsd import kernels as K

BATCH_BYTES = 4 << 30       # the arrays of a batch, at most

_P, _I = ctypes.c_void_p, ctypes.c_int64
if K._lib is not None and hasattr(K._lib, 'zc_t_batch'):
    K._lib.zc_t_batch.argtypes = [_P, _P, _P, _I, _I, _I, _I, _I, _P, _P, _P, _P, _P, _P, _P]
    K._lib.zc_t_batch.restype = None
    K._lib.zc_t_ovvv.argtypes = [_P, _I, _I, _I, _I, _P]
    K._lib.zc_t_ovvv.restype = None
    _HAVE_C = True
else:
    _HAVE_C = False


def _t_batch(X1, G1, G2, a, b0, b1, t1, fvo, VpT, T2pT, eo, ev):
    '''The (T) energy of the triples (a, b, c), b0 <= b < b1, c < b, from
    X1[(bc) - p0, i, (jk)] = x_{a,(bc)}, G1[x, i, y - b0, (jk)] = x_{y,(ax)}
    (x < b1, b0 <= y < b1) and G2[x - b0, i, y, (jk)] (b0 <= x < b1, y < b0):
    a complex number whose imaginary part should vanish.'''
    o, v = t1.shape
    npo = K.npair(o)
    if _HAVE_C and K.USE_C:
        out = np.zeros(2)
        c = K._c
        K._lib.zc_t_batch(c(X1), c(G1), c(G2) if b0 else None, a, b0, b1, o, v, c(t1), c(fvo),
                          c(VpT), c(T2pT), c(eo, np.float64), c(ev, np.float64),
                          c(out, np.float64))
        return complex(out[0], out[1])
    nb, p0, pa = b1 - b0, b0 * (b0 - 1) // 2, a * (a - 1) // 2
    X1 = X1.reshape(-1, o, npo)
    G1 = G1.reshape(b1, o, nb, npo)
    G2 = G2.reshape(nb, o, b0, npo) if b0 else None
    i, j, k = np.array([(i, j, k) for i in range(o) for j in range(i) for k in range(j)]).T
    pjk, pik, pij = j * (j - 1) // 2 + k, i * (i - 1) // 2 + k, i * (i - 1) // 2 + j
    e = 0j
    for b in range(max(b0, 1), b1):
        for c in range(b):
            pbc = b * (b - 1) // 2 + c
            Z = X1[pbc - p0] - G1[c, :, b - b0]
            Z = Z + (G1[b, :, c - b0] if c >= b0 else G2[b - b0, :, c])
            V = (Z + t1[:, a, None] * VpT[pbc] + fvo[a][:, None] * T2pT[pbc]
                 - t1[:, b, None] * VpT[pa + c] - fvo[b][:, None] * T2pT[pa + c]
                 + t1[:, c, None] * VpT[pa + b] + fvo[c][:, None] * T2pT[pa + b])
            w = Z[i, pjk] - Z[j, pik] + Z[k, pij]
            u = V[i, pjk] - V[j, pik] + V[k, pij]
            e += np.sum(u.conj() * w / (eo[i] + eo[j] + eo[k] - ev[a] - ev[b] - ev[c]))
    return e


def _batches(a, o, nbytes):
    '''Ranges (b0, b1) of b < a whose arrays (x_{a,(bc)} for the pairs with b
    in the range, and the two rectangles of x_{y,(ax)}) take ``nbytes``.'''
    unit = 16 * o * K.npair(o)
    out, b0 = [], 0
    while b0 < a:
        b1 = b0 + 1
        while b1 < a and unit * (K.npair(b1 + 1) - K.npair(b0) + (b1 + 1) * (b1 + 1 - b0)
                                 + (b1 + 1 - b0) * b0) <= nbytes:
            b1 += 1
        out.append((b0, b1))
        b0 = b1
    return out


def kernel(mycc, eris, t1=None, t2=None, verbose=None, max_memory=None):
    '''E(T) for the amplitudes t1 and t2 (t2[i,j,a,b] or packed
    [(i>j), (a>b)]) and the integrals of DirectZCCSD.'''
    log = logger.new_logger(mycc, verbose)
    cput0 = (logger.process_clock(), logger.perf_counter())
    t1 = mycc.t1 if t1 is None else t1
    t2 = mycc.t2 if t2 is None else t2
    t1 = np.ascontiguousarray(t1, dtype=np.complex128)
    o, v = t1.shape
    npo, npv = K.npair(o), K.npair(v)
    if o < 3 or v < 3:
        return 0.0
    mo_e = np.asarray(eris.mo_energy, dtype=np.float64)
    eo, ev = np.ascontiguousarray(mo_e[:o]), np.ascontiguousarray(mo_e[o:])
    fvo = np.ascontiguousarray(np.asarray(eris.fock)[o:, :o], dtype=np.complex128)
    io, jo = K.tril(o)

    with timer('t.setup'):
        t2 = np.ascontiguousarray(t2, dtype=np.complex128)
        t2p = K.pack_t2(t2) if t2.ndim == 4 else t2
        t2u = K.unpack_tril(t2p, v, 'apb')                      # [a, (j>k), e] = t2[j,k,a,e]
        T2pT = np.ascontiguousarray(t2p.T)                      # [(b>c), (j>k)]
        t2x = K.unpack_tril(T2pT, o, 'pab')                     # [(b>c), i, m] = t2[i,m,b,c]
        del t2p
        oovv_p = getattr(eris, 'oovv_p', None)
        if oovv_p is None:
            oovv_p = K.pack_t2(np.ascontiguousarray(np.asarray(eris.oovv), dtype=np.complex128))
        VpT = np.ascontiguousarray(np.asarray(oovv_p).conj().T)     # <bc||jk>
        # Mh[a, (j>k), m] = <ma||jk> = <jk||ma>*
        Mh = np.ascontiguousarray(np.asarray(eris.ooov)[io, jo].transpose(2, 0, 1).conj(),
                                  dtype=np.complex128)

    # Op[(b>c), i, e] = <bc||ei> = -<ie||bc>*: in memory or a mapped scratch array
    budget = memory.budget(mycc.max_memory if max_memory is None else max_memory)
    floor = 16 * o * npo * 3 * v                    # the batch of one b
    size = 16 * npv * o * v
    scr = None
    if budget is not None and budget < floor:
        raise MemoryError('(T) needs %.1f MB for the batch of one virtual orbital, %.1f MB are '
                          'left of max_memory' % (floor / 1e6, budget / 1e6))
    if budget is None or size + floor <= budget:
        Op = np.empty((npv, o, v), dtype=np.complex128)
        nbytes = BATCH_BYTES if budget is None else min(BATCH_BYTES, budget - size)
    else:
        scr = scratch.Scratch(prefix='zccsd_t_', what='<bc||ei>',
                              hint=' or give the calculation more memory (max_memory)')
        Op = scr.create('Op', (npv, o, v), np.complex128)
        nbytes = min(BATCH_BYTES, budget)
    with timer('t.setup'):
        ovvv_p = getattr(eris, 'ovvv_p', None)
        if ovvv_p is None:
            bv, cv = K.tril(v)
            ovvv_p = np.asarray(eris.ovvv)[:, :, bv, cv]
        for i in range(o):
            slab = np.ascontiguousarray(ovvv_p[i], dtype=np.complex128)
            if _HAVE_C and K.USE_C:
                K._lib.zc_t_ovvv(K._c(slab), v, npv, o, i, K._c(Op))
            else:
                Op[:, i] = -slab.T.conj()
            scratch.unmap(slab)
        scratch.written(Op)
    log.info('(T): %d occupied and %d virtual spinors; <bc||ei> (%.2f GB) in %s, batches of '
             '%.2f GB', o, v, size / 1e9, 'memory' if scr is None else 'a scratch file',
             nbytes / 1e9)

    et = 0j
    done, total, nxt = 0, v * (v - 1) * (v - 2) // 6, 0.1
    for a in range(2, v):
        pa = a * (a - 1) // 2
        for b0, b1 in _batches(a, o, nbytes):
            nb, p0, p1 = b1 - b0, K.npair(b0), K.npair(b1)
            with timer('t.gemm'):
                Ta = t2u[a].T                                           # [e, (jk)]
                Ma = Mh[a].T                                            # [m, (jk)]
                # x_{a,(bc)} for the pairs (b, c) with b in the range
                X1 = blas.gemm(Op[p0:p1].reshape((p1 - p0) * o, v), Ta)
                blas.gemm(t2x[p0:p1].reshape((p1 - p0) * o, o), Ma, C=X1, alpha=-1.0, beta=1.0)
                # x_{y,(ax)} for x < b1, b0 <= y < b1 and for b0 <= x < b1, y < b0
                Ty = t2u[b0:b1].reshape(nb * npo, v).T
                My = Mh[b0:b1].reshape(nb * npo, o).T
                G1 = blas.gemm(Op[pa:pa + b1].reshape(b1 * o, v), Ty)
                blas.gemm(t2x[pa:pa + b1].reshape(b1 * o, o), My, C=G1, alpha=-1.0, beta=1.0)
                G2 = None
                if b0:
                    Ty = t2u[:b0].reshape(b0 * npo, v).T
                    My = Mh[:b0].reshape(b0 * npo, o).T
                    G2 = blas.gemm(Op[pa + b0:pa + b1].reshape(nb * o, v), Ty)
                    blas.gemm(t2x[pa + b0:pa + b1].reshape(nb * o, o), My, C=G2,
                              alpha=-1.0, beta=1.0)
            with timer('t.kernel'):
                et += _t_batch(X1, G1, G2, a, b0, b1, t1, fvo, VpT, T2pT, eo, ev)
            del X1, G1, G2
        done += pa
        if done >= nxt * total:
            log.info('(T): %3.0f %% of the virtual triples, %.1f s', 100 * done / total,
                     logger.perf_counter() - cput0[1])
            nxt = (int(10 * done / total) + 1) / 10
    if abs(et.imag) > 1e-6:
        log.warn('Non-zero imaginary part found in the (T) energy %s', et)
    del Op
    if scr is not None:
        scr.close()
    log.timer('%s(T)' % mycc.__class__.__name__, *cput0)
    log.note('%s(T) correction = %.15g', mycc.__class__.__name__, et.real)
    return et.real
