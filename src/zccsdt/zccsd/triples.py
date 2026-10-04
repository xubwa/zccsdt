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

The triples are done in batches - a range of a with all its b if that
fits, else one a and a range of b - (all c < b).  With the two arrays

    A[(b>c), i, (e | m)] = (<bc||ei> | t2[i,m,b,c])
    B[a, (j>k), (e | m)] = (t2[j,k,a,e] | -<ma||jk>)

(<bc||ei> is the stored <ie||bc>, transposed once) the particle and the
hole term of x are one product over the joint index (e | m), and the x of a
batch are two or three GEMMs on contiguous blocks of A and B: x_{a,(bc)} for
the pairs of the batch, and x_{y,(ax)} for the rectangles of (x, y) it needs
(o^3 v^4 / 4 multiply-adds for the particle term in all, each x made once
and written once).  An OpenMP kernel (csrc/zccsd_t.c) assembles Z for one
(b, c) per thread in its cache and adds up the energy.

With ``three_m`` A and B are kept as their real parts (Re, Im, Re + Im) and
every complex product is three real GEMMs, A_r B_r, A_i B_i and
(A_r + A_i)(B_r + B_i), instead of four; the three real results of a batch
go to the kernel as they are.  One and a half times the memory for a
quarter fewer flops: the default where the real parts of both operands and
a whole batch fit in memory.

Kramers pairs.  For spinors in time-reversal pairs, ordered [unbarred,
barred] in the virtual space (zccsdt.zccsd.kramers), the triples (a, b, c)
and (Ka, Kb, Kc) contribute complex-conjugate energies, and K turns a
triple with n barred virtuals into one with 3 - n: the triples with at most
one barred virtual - b in the unbarred half, whatever a - are one of each
pair.  ``kramers`` restricts the batches to them and doubles the sum: half
the GEMMs.  (For a range of barred a the pairs (b, c) are the same, those
below the barred half: one product for the x_{a,(bc)} of the range.)

Memory: A and B are kept in memory, the smaller first, as long as they fit
next to the smallest batch; else they are mapped arrays of a scratch file,
written and read in contiguous blocks.  The batches are sized from what
``max_memory`` leaves (at most BATCH_BYTES), and their x are written into
the same arrays, allocated once for the largest batch (new arrays for every
batch are mapped and zeroed by the system page by page: a fifth of the time
of the GEMMs).  MemoryError below three packed T2 and the batch of one b.

Timers (zccsdt.timing): 't.setup', 't.gemm', 't.kernel'.
'''

import ctypes

import numpy as np
from pyscf.lib import logger

from zccsdt import blas, memory, scratch
from zccsdt.timing import timer
from zccsdt.zccsd import kernels as K

# The x of a batch, at most.  Benzene/cc-pVDZ with Kramers pairs (job 31556395):
# 252.6 / 233.5 / 228.2 / 228.7 / 228.8 s for 0.5 / 1 / 2 / 4 / 8 GB with ZGEMM,
# 234.1 / 223.5 / 223.9 s for 2 / 4 / 8 GB with three real GEMMs (which hold two
# thirds of the elements in the same bytes)
BATCH_BYTES = 4 << 30

_P, _I = ctypes.c_void_p, ctypes.c_int64
if K._lib is not None and hasattr(K._lib, 'zc_t_batch'):
    K._lib.zc_t_batch.argtypes = [_P, _I, _I, _P, _I, _P, _I, ctypes.c_int, _I, _I, _I, _I, _I,
                                  _P, _P, _P, _P, _P, _P, _P]
    K._lib.zc_t_batch.restype = None
    for _f, _a in (('zc_t_ovvv', [_P, _I, _I, _I, _I, _I, _I, _I, ctypes.c_int, _P, _P, _P]),
                   ('zc_t_t2u', [_P, _I, _I, _I, _I, _I, _I, ctypes.c_int, _P, _P, _P]),
                   ('zc_t_t2x', [_P, _I, _I, _I, _I, _I, _I, ctypes.c_int, _P, _P, _P])):
        getattr(K._lib, _f).argtypes, getattr(K._lib, _f).restype = _a, None
    _HAVE_C = True
else:
    _HAVE_C = False


def _complex(x):
    '''A complex array from its three real products (3M), or as it is.'''
    if isinstance(x, (tuple, list)):
        return (x[0] - x[1]) + 1j * (x[2] - x[0] - x[1])
    return x


def _t_batch(X1, c1, G1, g1, G2, a, b0, b1, t1, fvo, VpT, T2pT, eo, ev, terms=None):
    '''The (T) energy of the triples (a, b, c), b0 <= b < b1, c < b, from
    matrices with rows (pair or virtual, i) and columns (virtual, (jk)):

    X1[(p - p0, i), c1 + (jk)] = x_{a,(bc)}, p the pair (b, c), p0 the pair (b0, 0);
    G1[g1 + (x, i), (y - b0, (jk))] = x_{y,(ax)} for x < b1 and y >= b0;
    G2[(x - b0, i), (y, (jk))] = x_{y,(ax)} for b0 <= x < b1 and y < b0.

    The three are complex arrays or tuples of the three real products of
    the 3M scheme.  Returns a complex number whose imaginary part should
    vanish.  ``terms``: an object whose add(a, b, c, w, u) gets the
    connected triples w and w plus the disconnected ones u of every virtual
    triple, divided by the denominator and packed over i > j > k (the (T)
    terms of the Lambda equations, zccsd.lam_t.Terms; with numpy so far).'''
    o, v = t1.shape
    npo = K.npair(o)
    m3 = isinstance(X1, (tuple, list))
    first = lambda x: x[0] if m3 else x                         # noqa: E731
    s1, n1 = first(X1).shape[1], first(G1).shape[1]
    n2 = first(G2).shape[1] if b0 else 0
    if _HAVE_C and K.USE_C and terms is None:
        out = np.zeros(2)
        c = K._c
        dt = np.float64 if m3 else np.complex128

        def ptrs(x, row=0):
            x = (x if m3 else (x,)) if x is not None else ()
            p = [c(y, dt) + row * y.strides[0] for y in x] + [None] * 3
            return (ctypes.c_void_p * 3)(*p[:3])
        K._lib.zc_t_batch(ptrs(X1), s1, c1, ptrs(G1, g1), n1, ptrs(G2 if b0 else None), n2,
                          int(m3), a, b0, b1, o, v, c(t1), c(fvo), c(VpT), c(T2pT),
                          c(eo, np.float64), c(ev, np.float64), c(out, np.float64))
        return complex(out[0], out[1])
    p0, pa = b0 * (b0 - 1) // 2, a * (a - 1) // 2
    X1 = _complex(X1).reshape(-1, o, s1)[:, :, c1:c1 + npo]
    G1 = _complex(G1)[g1:g1 + b1 * o].reshape(b1, o, n1 // npo, npo)
    G2 = _complex(G2).reshape(b1 - b0, o, b0, npo) if b0 else None
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
            d = eo[i] + eo[j] + eo[k] - ev[a] - ev[b] - ev[c]
            e += np.sum(u.conj() * w / d)
            if terms is not None:
                terms.add(a, b, c, w / d, u / d)
    return e


def _a_range(a0, v, o, nbytes, xbytes=16):
    '''The largest a1 such that the arrays of the batch a0 <= a < a1 with all
    b < a (x_{a,(bc)} for the pairs below a1 - 1, x_{y,(ax)} for x < a and
    y < a1 - 1) take ``nbytes``; a0 if not even one a fits.'''
    unit = xbytes * o * K.npair(o)

    def size(a1):
        return unit * (K.npair(a1 - 1) * (a1 - a0) + (K.npair(a1) - K.npair(a0)) * (a1 - 1))
    a1 = a0
    while a1 < v and size(a1 + 1) <= nbytes:
        a1 += 1
    return a1


def _batches(a, o, nbytes, xbytes=16):
    '''Ranges (b0, b1) of b < a whose arrays (x_{a,(bc)} for the pairs with b
    in the range, and the two rectangles of x_{y,(ax)}) take ``nbytes``.'''
    unit = xbytes * o * K.npair(o)
    out, b0 = [], 0
    while b0 < a:
        b1 = b0 + 1
        while b1 < a and unit * (K.npair(b1 + 1) - K.npair(b0) + (b1 + 1) * (b1 + 1 - b0)
                                 + (b1 + 1 - b0) * b0) <= nbytes:
            b1 += 1
        out.append((b0, b1))
        b0 = b1
    return out


def _put(name, src, args, out, m3):
    '''One block of an operand through its kernel, or with numpy.'''
    if _HAVE_C and K.USE_C:
        dt = np.float64 if m3 else np.complex128
        ptrs = [K._c(x, dt) for x in out] + [None, None]
        getattr(K._lib, name)(src, *args, int(m3), *ptrs[:3])
        return True
    return False


def _store(x, out):
    '''The complex block x into out: one complex array, or the three real
    parts of the 3M scheme.'''
    if len(out) == 1:
        out[0][...] = x
    else:
        out[0][...], out[1][...] = x.real, x.imag
        np.add(out[0], out[1], out=out[2])


def memory_need(nocc, nvir, three_m=True):
    '''(bytes of the arrays that are always in memory - two packed T2 and a
    third while t2 is being packed -, bytes of the batch of one virtual
    orbital): the least (T) can run in is their sum.'''
    npo, npv, xbytes = K.npair(nocc), K.npair(nvir), 24 if three_m else 16
    return 48 * npo * npv, xbytes * nocc * npo * 3 * nvir


def kernel(mycc, eris, t1=None, t2=None, verbose=None, max_memory=None, three_m=None,
           kramers=None, terms=None):
    '''E(T) for the amplitudes t1 and t2 (t2[i,j,a,b] or packed
    [(i>j), (a>b)]) and the integrals of DirectZCCSD.  ``three_m``: three
    real GEMMs for each complex product (default: if the real parts of both
    operands and a whole batch fit in memory).  ``max_memory`` in MB (default: that of
    ``mycc``).  ``kramers``: the spinors are Kramers pairs in the
    order [unbarred, barred] and the amplitudes have their symmetry: only
    the triples with at most one barred virtual are done (default: if
    ``eris.kramers`` says so and the amplitudes pass kramers.symmetric).
    ``terms``: see _t_batch; terms.kramers is set to what is used here.'''
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
    budget = memory.budget(mycc.max_memory if max_memory is None else max_memory)
    if three_m is None:
        # the real parts, if both operands fit in memory with them next to a
        # whole batch: an operand in a scratch file or small batches cost
        # more than the three real GEMMs save
        three_m = budget is None or budget >= (
            memory_need(o, v, True)[0] + 24 * (npv * o + v * npo) * (v + o)
            + max(memory_need(o, v, True)[1], BATCH_BYTES))
    m3 = bool(three_m)
    nparts, dtype, xbytes = (3, np.float64, 24) if m3 else (1, np.complex128, 16)
    if kramers is None:
        kramers = bool(getattr(eris, 'kramers', False))
        if kramers:
            from zccsdt.zccsd.kramers import symmetric
            if not symmetric(t1, t2):
                log.info('(T): the amplitudes do not have the symmetry of the Kramers pairs: '
                         'all triples are done')
                kramers = False
    hv = v // 2 if kramers else v                   # b runs over the unbarred virtuals
    if terms is not None:
        terms.kramers = bool(kramers)

    # where the operands live: T2pT and VpT in memory; A and B in memory, the
    # smaller first, while they fit next to the smallest batch
    fixed, floor = memory_need(o, v, m3)
    n = v + o                                       # the joint index (e | m)
    shapes = {'A': (npv, o, n), 'B': (v, npo, n)}
    if budget is not None and budget < fixed + floor:
        raise MemoryError(
            '(T) needs %.1f MB (three packed T2 and the batch of one virtual orbital), %.1f MB '
            'are left of max_memory' % ((fixed + floor) / 1e6, budget / 1e6))
    left = None if budget is None else budget - fixed - floor
    ops, scr = {}, None
    for name in sorted(shapes, key=lambda k: np.prod(shapes[k], dtype=np.int64)):
        size = xbytes * int(np.prod(shapes[name], dtype=np.int64))
        if left is None or size <= left:
            ops[name] = tuple(np.empty(shapes[name], dtype=dtype) for _ in range(nparts))
            left = None if left is None else left - size
        else:
            if scr is None:
                scr = scratch.Scratch(prefix='zccsd_t_', what='the arrays of (T)',
                                      hint=' or give the calculation more memory (max_memory)')
            ops[name] = tuple(scr.create('%s%d' % (name, k), shapes[name], dtype)
                              for k in range(nparts))
    nbytes = BATCH_BYTES if left is None else min(BATCH_BYTES, floor + left)
    # the batches: (a, a1, None), a range of a with all its b - b < a, or with
    # Kramers pairs b below the barred half -, or (a, a + 1, (b0, b1)), one a
    # and a range of b; cap: the elements of the x of the largest batch
    plan, cap, a = [], 0, 2
    while a < v:
        if a < hv:
            a1 = _a_range(a, hv, o, nbytes, xbytes)
            size = K.npair(a1 - 1) * (a1 - a) + (K.npair(a1) - K.npair(a)) * (a1 - 1)
        else:       # barred a: x_{a,(bc)} of the range and x_{y,(ax)} of one a at a time
            a1 = a + max(0, min(v - a, (nbytes // (xbytes * o * npo) - hv * hv) // K.npair(hv)))
            size = K.npair(hv) * (a1 - a) + hv * hv
        if a1 > a:
            plan.append((a, a1, None))
            cap = max(cap, size)
            a = a1
        else:
            for b0, b1 in _batches(min(a, hv), o, nbytes, xbytes):
                plan.append((a, a + 1, (b0, b1)))
                cap = max(cap, K.npair(b1) - K.npair(b0) + (b1 - b0) * (b1 + b0))
            a += 1
    cap *= o * npo
    mapped = sorted(k for k in ops if scratch.is_mapped(ops[k][0]))
    log.info('(T): %d occupied and %d virtual spinors%s; %s; %d batches of at most %.2f GB; %s',
             o, v, ' in Kramers pairs (half of the virtual triples)' if kramers else '',
             'in a scratch file: ' + ', '.join(
                 '%s %.2f GB' % (k, xbytes * np.prod(shapes[k], dtype=np.int64) / 1e9)
                 for k in mapped) if mapped else 'all arrays in memory',
             len(plan), xbytes * cap / 1e9, 'three real GEMMs per product' if m3 else 'ZGEMM')
    log.info('(T): matrix products on %s', blas.backend()[1])

    def blocks(nrow, per_row):
        '''Row ranges of an operand: all at once in memory, else within nbytes.'''
        step = nrow if scr is None else max(1, min(nrow, nbytes // per_row))
        return [(r, min(r + step, nrow)) for r in range(0, nrow, step)]

    def done(out):
        for x in out:
            scratch.written(x)
            scratch.unmap(x)

    with timer('t.setup'):
        t2 = np.ascontiguousarray(t2, dtype=np.complex128)
        t2p = K.pack_t2(t2) if t2.ndim == 4 else t2
        T2pT = np.ascontiguousarray(t2p.T)                      # [(b>c), (j>k)]
        oovv_p = getattr(eris, 'oovv_p', None)
        if oovv_p is None:
            oovv_p = K.pack_t2(np.ascontiguousarray(np.asarray(eris.oovv), dtype=np.complex128))
        VpT = np.ascontiguousarray(np.asarray(oovv_p).conj().T)     # <bc||jk>
        # B[a, (j>k), (e | m)] = (t2[j,k,a,e] | -<ma||jk>), <ma||jk> = <jk||ma>*
        B = ops['B']
        ooov_p = np.asarray(eris.ooov)[io, jo]                  # [(j>k), m, a]
        for a0, a1 in blocks(v, xbytes * npo * n):
            out = [x[a0:a1] for x in B]
            if not _put('zc_t_t2u', K._c(t2p), (npo, v, a0, a1, n, 0), out, m3):
                _store(K.unpack_tril(t2p, v, 'apb')[a0:a1], [x[:, :, :v] for x in out])
            _store(-ooov_p[:, :, a0:a1].transpose(2, 0, 1).conj(), [x[:, :, v:] for x in out])
            done(out)
        # A[(b>c), i, (e | m)] = (<bc||ei> | t2[i,m,b,c]), <bc||ei> = -<ie||bc>* from
        # the integrals as the CCSD keeps them, [i, e, (b>c)]: blocks of pairs
        A = ops['A']
        ovvv_p = getattr(eris, 'ovvv_p', None)
        if ovvv_p is None:
            bv, cv = K.tril(v)
            ovvv_p = np.ascontiguousarray(np.asarray(eris.ovvv)[:, :, bv, cv],
                                          dtype=np.complex128)
        for q0, q1 in blocks(npv, xbytes * o * n):
            out = [x[q0:q1] for x in A]
            if not _put('zc_t_t2x', K._c(t2p), (o, npv, q0, q1, n, v), out, m3):
                _store(K.unpack_tril(np.ascontiguousarray(T2pT[q0:q1]), o, 'pab'),
                       [x[:, :, v:] for x in out])
            for i in range(o):
                src = ovvv_p[i]
                if not (src.dtype == np.complex128 and src.flags.c_contiguous and _put(
                        'zc_t_ovvv', src.ctypes.data + 16 * q0, (npv, v, q1 - q0, o, i, n, 0),
                        out, m3)):
                    _store(-np.asarray(src[:, q0:q1]).T.conj(), [x[:, i, :v] for x in out])
                scratch.unmap(src)
            done(out)
        del t2p, ooov_p

    # the x of all batches are written into the same arrays: arrays allocated
    # per batch are mapped and zeroed page by page every time, as much memory
    # traffic as the kernel's own
    pool = [np.empty(cap, dtype=dtype) for _ in range(nparts)]

    def products(rows, cols, off=0):
        '''x = A[rows] B[cols]^T over the joint index (e | m), for a block of
        rows (pairs, i) and of columns (virtual, jk): complex, or its three
        real products, in the arrays of the batch from element ``off`` on.
        Returns x and the element behind it.'''
        a, b = [rows(x) for x in A], [cols(x) for x in B]
        m, nc = a[0].shape[0], b[0].shape[1]
        end = off + m * nc
        assert end <= cap
        out = [blas.gemm(a[k], b[k], pool[k][off:end].reshape(m, nc)) for k in range(nparts)]
        return (out if m3 else out[0]), end

    def one_a(a, b0, b1):
        '''The triples of one a with b0 <= b < b1: x_{a,(bc)} for the pairs with
        b in the range, x_{y,(ax)} for x < b1, b0 <= y < b1 and for
        b0 <= x < b1, y < b0.'''
        pa, nb, p0, p1 = K.npair(a), b1 - b0, K.npair(b0), K.npair(b1)
        with timer('t.gemm'):
            X1, end = products(lambda x: x[p0:p1].reshape((p1 - p0) * o, n), lambda x: x[a].T)
            G1, end = products(lambda x: x[pa:pa + b1].reshape(b1 * o, n),
                               lambda x: x[b0:b1].reshape(nb * npo, n).T, end)
            G2 = None
            if b0:
                G2, end = products(lambda x: x[pa + b0:pa + b1].reshape(nb * o, n),
                                   lambda x: x[:b0].reshape(b0 * npo, n).T, end)
        with timer('t.kernel'):
            return _t_batch(X1, 0, G1, 0, G2, a, b0, b1, t1, fvo, VpT, T2pT, eo, ev, terms)

    et = 0j
    ntri = lambda a: K.npair(min(a, hv))                        # noqa: E731
    ndone, total, nxt = 0, sum(ntri(a) for a in range(v)), 0.1
    for a, a1, brange in plan:
        if brange is None and a >= hv:
            # barred a, unbarred b and c: x_{a,(bc)} as [(bc, i), (a, jk)] for the
            # pairs below the barred half, x_{y,(ax)} as [(x, i), (y, jk)] for one
            # a at a time (its pairs (a, x) are not next to those of a + 1)
            na, pr = a1 - a, K.npair(hv)
            with timer('t.gemm'):
                X1, end = products(lambda x: x[:pr].reshape(pr * o, n),
                                   lambda x: x[a:a1].reshape(na * npo, n).T)
            for aa in range(a, a1):
                pa = K.npair(aa)
                with timer('t.gemm'):
                    G1, _ = products(lambda x: x[pa:pa + hv].reshape(hv * o, n),
                                     lambda x: x[:hv].reshape(hv * npo, n).T, end)
                with timer('t.kernel'):
                    et += _t_batch(X1, (aa - a) * npo, G1, 0, None, aa, 0, hv,
                                   t1, fvo, VpT, T2pT, eo, ev, terms)
            ndone += na * ntri(a)
        elif brange is None:
            # x_{a,(bc)} as [(bc, i), (a, jk)] for the pairs below a1 - 1,
            # x_{y,(ax)} as [(a, x, i), (y, jk)] for y < a1 - 1
            na, pr, q0, q1 = a1 - a, K.npair(a1 - 1), K.npair(a), K.npair(a1)
            with timer('t.gemm'):
                X1, end = products(lambda x: x[:pr].reshape(pr * o, n),
                                   lambda x: x[a:a1].reshape(na * npo, n).T)
                G1, end = products(lambda x: x[q0:q1].reshape((q1 - q0) * o, n),
                                   lambda x: x[:a1 - 1].reshape((a1 - 1) * npo, n).T, end)
            with timer('t.kernel'):
                for aa in range(a, a1):
                    et += _t_batch(X1, (aa - a) * npo, G1, (K.npair(aa) - q0) * o, None, aa, 0, aa,
                                   t1, fvo, VpT, T2pT, eo, ev, terms)
            ndone += sum(ntri(aa) for aa in range(a, a1))
        else:
            et += one_a(a, *brange)
            if brange[1] == min(a, hv):
                ndone += ntri(a)
        if ndone >= nxt * total:
            log.info('(T): %3.0f %% of the virtual triples, %.1f s', 100 * ndone / total,
                     logger.perf_counter() - cput0[1])
            nxt = (int(10 * ndone / total) + 1) / 10
    if kramers:
        et = 2.0 * et.real + 0j
    if abs(et.imag) > 1e-6:
        log.warn('Non-zero imaginary part found in the (T) energy %s', et)
    del A, B, ops, pool
    if scr is not None:
        scr.close()
    log.timer('%s(T)' % mycc.__class__.__name__, *cput0)
    log.note('%s(T) correction = %.15g', mycc.__class__.__name__, et.real)
    return et.real
