'''The T4 equation of spin-orbital CCSDTQ on packed amplitudes, real or
complex: T3 and T4 on their unique blocks (i<j<k(<l), a<b<c(<d)), the
T1-dressed integrals in blocks.

    R4 = P(ab/cd) sum_{e<f} Wq(ab,ef) t(ijkl,efcd)           pp ladder, F_vv folded in
       + P(ij/kl) sum_{m<n} Wh(mn,ij) t(mnkl,abcd)           hh ladder, F_oo folded in
       + P(i/jkl) P(a/bcd) sum_{me} Wr(ma,ei) t(mjkl,ebcd)   ring
       + P(i/jkl) P(ab/cd) [ - sum_e Wv(ab,e,i) t(jkl,cde) + sum_m t(im,ab) Xo(m;jkl;cd) ]
       + P(ij/kl) P(a/bcd) [   sum_m Wo(m,a,ij) t(mkl,bcd) + sum_e t(ij,ae) Xv(e;kl;bcd) ]
       - P(i/jkl) P(d/abc) sum_{m<n} t(mni,abc) Y(mn;jkl;d)
       + 1/2 P(ij/kl) P(ab/cd) sum_{nf} Z(nf;ij;ab) t(nkl,fcd)

(the terms of zccsdt.cc.ccsdtq_terms, every product a group of them: tests),
and the terms of T4 in the equations of T2 and T3.

How the products are done (zccsdt.cc.t4pack for the views of the packed
arrays and the antisymmetrizing reductions):

- pp ladder: for the occupied quadruples in batches, T4 with its virtual
  indices in two pairs times Wq, reduced over the splits (ab|cd).  The same
  view gives, by thin products, the T4 part of Xo and the T4 terms of R2
  and of R3 with <vm||vv>.
- ring: T4 as t(m jkl; e bcd) in batches of the columns (bcd) times Wr,
  the slabs of the four splits (i|jkl) of a quadruple summed and reduced
  once over (a|bcd); the kernels walk through the block of columns.  The
  same view gives the T4 part of Xv and the T4 terms of R3 with f_ov and
  <oo||vo>.
- hh ladder: rows of T4 by two occupied pairs, in batches of columns.
- the two products with a six-index intermediate and the two of T3 with
  T3: for one occupied quadruple at a time, as in (Q)
  (zccsdt.cc.quadruples): the splits of the quadruple are summed by the
  matrix products themselves, into X[(ab),(cd)] and X[a,(bcd)], each
  reduced once.

So an element of the residual is gathered 26 times per iteration and one of
T4 22 times (pp and ring), where views and reductions on two occupied and
two virtual groups for every product would take about 190.

What is kept besides T4 and its residual: Xv for all pairs of occupied
indices (v^4 / 6 each: 48 / ((o-2)(o-3)) of T4), the batch arrays, and of
the size of T3 the operands of the T3 x T3 products (twice 9 x T3) and T3
in rows by one occupied index and a pair (3 x T3).
'''

from dataclasses import dataclass
from math import comb

import numpy as np

from zccsdt.blas import gemm
from zccsdt.cc import t4pack as P
from zccsdt.timing import timer

BATCH_BYTES = 1 << 30           # an operand or a product of one batch, at most
PASSES = 16                     # batches of columns of a product of one occupied quadruple, at most
RING_PASSES = 64                # batches of columns of the ring and of the hh ladder, at most
PIECES = ('pp', 'hh', 'r2', 'ring', 'b1', 'b2', 't3t3h', 't3t3r')


def _E(sub, *ops):
    return np.einsum(sub, *ops, optimize=True)


def _pairs(n):
    t = P.tuples(n, 2)
    return t[:, 0], t[:, 1]


def _batches(n, step):
    step = int(max(1, step))
    return [(x, min(n, x + step)) for x in range(0, n, step)]


def _one_body_pairs(F, n):
    '''W[(pq),(rs)] = F[p,r] d(q,s) - F[p,s] d(q,r) - F[q,r] d(p,s) + F[q,s] d(p,r)
    on pairs p<q, r<s: the one-body operator on a pair.'''
    p, q = _pairs(n)
    eye = np.eye(n)
    return (F[p[:, None], p[None, :]] * eye[q[:, None], q[None, :]]
            - F[p[:, None], q[None, :]] * eye[q[:, None], p[None, :]]
            - F[q[:, None], p[None, :]] * eye[p[:, None], q[None, :]]
            + F[q[:, None], q[None, :]] * eye[p[:, None], p[None, :]])


@dataclass
class Dressed:
    '''The T1-dressed Fock matrix and blocks of <pq||rs> of the T4 equation.'''
    nocc: int
    nvir: int
    f: np.ndarray           # [p, q]
    oovv: np.ndarray        # <mn||ef>  [m, n, e, f]
    oooo: np.ndarray        # <mn||kl>
    oovo: np.ndarray        # <mn||el>
    ovvo: np.ndarray        # <mb||ek>
    vovv: np.ndarray        # <dm||ef>
    ovoo: np.ndarray        # <md||kl>
    vvvo_p: np.ndarray      # <cd||el>  [(cd), e, l]
    vvvv_pp: np.ndarray     # <cd||ef>  [(cd), (ef)]

    @classmethod
    def from_dense(cls, f, v, nocc):
        '''From the full dressed tensor (tests).'''
        n = v.shape[0]
        o, V = slice(0, nocc), slice(nocc, n)
        a2, b2 = _pairs(n - nocc)
        return cls(nocc, n - nocc, f, v[o, o, V, V], v[o, o, o, o], v[o, o, V, o], v[o, V, V, o],
                   v[V, o, V, V], v[o, V, o, o], np.ascontiguousarray(v[V, V, V, o][a2, b2]),
                   np.ascontiguousarray(v[V, V, V, V][a2[:, None], b2[:, None], a2, b2]))

    @classmethod
    def from_eris(cls, eris, t1):
        '''From an integral container (zccsdt.eris.ERIs, <vv||vv> packed or
        full) and T1.'''
        from zccsdt.cc import ccsdt_opt as O, t3pack
        nocc, nvir = t1.shape
        om, vm = t3pack.triple_map(nocc), t3pack.triple_map(nvir)
        blk = {k: O.dress_block(eris, t1, k) for k in ('oooo', 'oovo', 'ovvo', 'vovv', 'ovoo')}
        vpp = O.dressed_vvvv_pp(eris, t1, om, vm)[0]
        # the bra of <cd||el> dressed, then its ket: + sum_f ~<cd||ef> t1[l,f]
        vvvo = np.ascontiguousarray(O.bra_dressed_vvvo_pp(eris, t1, om, vm), dtype=vpp.dtype)
        t3pack.pair_t1_contract(vvvo, vpp, t1, vm)
        return cls(nocc, nvir, O.dress_fock(eris, t1), np.asarray(eris.oovv), blk['oooo'], blk['oovo'],
                   blk['ovvo'], blk['vovv'], blk['ovoo'], vvvo, vpp)


def _sizes(no, nv):
    return comb(no, 2), comb(nv, 2), comb(no, 3), comb(nv, 3), comb(no, 4), comb(nv, 4)


def _columns(no, nv, isz, batch_bytes):
    '''Columns per batch: (pairs (cd) and triples (bcd) of the products of
    one occupied quadruple, triples of the ring, quadruples of the hh
    ladder, occupied quadruples of the pp ladder).'''
    nP, nQ, nI3, nA3, nI, nA = _sizes(no, nv)

    def cols(n, rows, passes):
        return int(min(n, max(-(-n // passes), batch_bytes // (isz * max(rows, 1)))))
    return (cols(nQ, nQ, PASSES), cols(nA3, nv, PASSES), cols(nA3, no * nv * nI3, RING_PASSES),
            cols(nA, nP * nP, RING_PASSES), cols(nI, nQ * nQ, nI))


def memory(no, nv, isz, batch_bytes=None):
    '''Bytes that residual() allocates at most (its results included: the
    residual of T4 is one of them), for products of ``batch_bytes`` per
    batch - the largest of its stages, each counted array by array.'''
    batch_bytes = batch_bytes or BATCH_BYTES
    nP, nQ, nI3, nA3, nI, nA = _sizes(no, nv)
    c22, c13, cring, chh, cpp = _columns(no, nv, isz, batch_bytes)
    ov, n = no * nv, no + nv
    # the layouts of T2 and of the integrals, the results
    small = 3 * nP * nQ + 2 * no * no * nQ + 2 * nP * nv * nv + nP * nv * no + nv * no * nQ + n * n
    res = nI * nA + nI3 * nA3 + nP * nQ
    t3 = no * nP * nA3 + 2 * nP * ov * nQ                   # T3 by rows, T3 on two groups, Z
    g = no * no * nP * nQ
    hs = nv * no * nI3 * nv
    y3, xo = nI3 * nv * nP, nI3 * no * nQ
    w3 = nQ * nv * no                                       # <vv||vo> on pairs
    wsmall = no ** 4 + 2 * no * no * nP + 3 * ov * ov + 2 * nP * nP + 2 * nP * ov
    first = small + res + t3 + g + w3 + no * nP * nv
    kept = small + res + t3 + y3 + xo + w3 + nP * ov + wsmall      # after the intermediates
    xv = nP * nv * nA3
    rows_w = min(nQ, max(1, batch_bytes // (2 * isz * nv * nv)))
    stages = [
        first + 3 * nv * nP * nQ + 2 * no * nv * nv,                            # the pass over T3
        first + nQ * nI3 * nv + 2 * y3 + hs,                                    # .. its end
        first + y3 + hs + nQ * nQ + wsmall + 2 * w3 + nv ** 3 * no + 2 * w3,    # Wv
        first + y3 + hs + nQ * nQ + wsmall + w3 + xo + no * nP * nQ + 6 * nI3 * nQ,     # Xo
        kept + nQ * nQ + xv + no * nv * nP + nv * nv * ov * 2 + nv * nv * nQ + 2 * rows_w * nv * nv,    # Xv
        kept + nQ * nQ + xv + 2 * nQ * cpp * nQ + nI3 * nv * nQ + (no + 8) * n * nQ,    # pp ladder
        kept + xv + 2 * ov * nI3 * cring + no * nP * cring + ov * ov + (no + 6) * n * ov + 3 * n * cring,   # ring
        kept + xv + 2 * nP * nP * chh + nP * nP,                                # hh ladder
        kept + xv + nQ * c22 + nv * c13 + 8 * n * nQ,                           # one occupied quadruple
    ]
    # tables of the splits of the occupied tuples and of the ring's block of columns, small arrays
    tables = 300 * nI + 9 * nP * nP + 60 * nI3 + 9 * nv * cring + (1 << 20)
    return isz * max(stages) + tables


def min_batch_bytes(no, nv, isz):
    '''The smallest ``batch_bytes`` that is of use: one occupied quadruple
    of the pp ladder, the other products in their largest numbers of
    batches.'''
    nP, nQ, nI3, nA3, nI, nA = _sizes(no, nv)
    return isz * max(nQ * nQ, no * nv * nI3 * -(-nA3 // RING_PASSES), nP * nP * -(-nA // RING_PASSES),
                     nQ * -(-nQ // PASSES), nv * -(-nA3 // PASSES))


class _Pool:
    '''Work arrays by name, kept between batches (a fresh large array pays
    its page faults again).'''

    def __init__(self):
        self.arrays = {}

    def __call__(self, name, shape, dtype):
        n = int(np.prod(shape))
        a = self.arrays.get(name)
        if a is None or a.size < n or a.dtype != dtype:
            # free before allocating again (the views that a caller still holds pin the old
            # array: every stage of residual() deletes its own)
            del a
            self.arrays[name] = None
            self.arrays[name] = np.empty(max(n, 1), dtype=dtype)
        return self.arrays[name][:n].reshape(shape)

    def clear(self):
        self.arrays.clear()


def residual(t2, T3, T4, ints, batch_bytes=None, only=None):
    '''(R4, the T4 terms of R3, the T4 term of R2), all packed, from
    t2[i,j,a,b], the packed T3 [C(no,3), C(nv,3)] and T4 [C(no,4), C(nv,4)]
    and the dressed integrals ``ints`` (Dressed).  ``only``: the pieces to
    evaluate (tests), of PIECES.'''
    def want(*pieces):
        return only is None or any(p in only for p in pieces)
    batch_bytes = batch_bytes or BATCH_BYTES
    no, nv = ints.nocc, ints.nvir
    dtype = np.result_type(T4, T3, t2, ints.oovv, ints.f)
    isz = np.dtype(dtype).itemsize
    nP, nQ, nI3, nA3, nI, nA = comb(no, 2), comb(nv, 2), comb(no, 3), comb(nv, 3), comb(no, 4), comb(nv, 4)
    T3 = np.ascontiguousarray(np.asarray(T3).reshape(nI3, nA3), dtype=dtype)
    T4 = np.ascontiguousarray(np.asarray(T4).reshape(nI, nA), dtype=dtype)
    o12, o13, o22 = P.split(no, 1, 2), P.split(no, 1, 3), P.split(no, 2, 2)
    i2, j2 = _pairs(no)
    a2, b2 = _pairs(nv)
    pairs_o = P.tuples(no, 2)
    c22, c13, cring, chh, cpp = _columns(no, nv, isz, batch_bytes)
    work = _Pool()

    def c_(x):
        return np.ascontiguousarray(x, dtype=dtype)

    # (timed: where a product spends its time)
    def expand2(*args, **kw):
        with timer('expand'):
            return P.tn_expand2(*args, **kw)

    def reduce2(*args, **kw):
        with timer('reduce'):
            return P.tn_reduce2(*args, **kw)

    def axpy_rows(*args, **kw):
        with timer('reduce'):
            return P.tn_axpy_rows(*args, **kw)

    def rows(*args, **kw):
        with timer('expand'):
            return P.tn_rows(*args, **kw)

    def mm(*args, **kw):
        with timer('gemm'):
            return gemm(*args, **kw)

    f = np.asarray(ints.f)
    foo, fov, fvv = f[:no, :no], c_(f[:no, no:]), f[no:, no:]
    oovv = np.asarray(ints.oovv)
    oovv_pp = c_(oovv[i2, j2][:, a2, b2])                               # [(mn), (ef)]
    oovv_xp = c_(oovv[:, :, a2, b2])                                    # [m, n, (ef)]
    oovv_ef = c_(oovv[i2, j2].transpose(1, 0, 2)).reshape(nv, nP * nv)  # [e, ((mn) f)]
    oovo = np.asarray(ints.oovo)
    oovo_pp = c_(oovo[i2, j2])                                          # [(mn), e, l]
    vovv = np.asarray(ints.vovv)
    vovv_p = c_(vovv[:, :, a2, b2])                                     # [d, m, (ef)]
    t2_pp = c_(t2[i2, j2][:, a2, b2])                                   # [(ij), (ab)]
    t2_xp = c_(t2[:, :, a2, b2])                                        # [i, m, (ab)]
    t2_px = c_(t2[i2, j2])                                              # [(ij), a, e]

    R4 = np.zeros((nI, nA), dtype=dtype)
    R3 = np.zeros((nI3, nA3), dtype=dtype)
    R2 = np.zeros((nP, nQ), dtype=dtype)

    # ------------------------------------------------ T3 with one integral
    # t3o[m, (kl), (bcd)] = t(mkl, bcd): rows of T3 by one occupied index and a pair
    t3o = np.empty((no * nP, nA3), dtype=dtype)
    P.tn_rows(t3o, T3, o12.pos.ravel(), o12.psg.ravel())
    t3o = t3o.reshape(no, nP, nA3)
    with timer('t4.t3pass'):
        # t(n kl; f cd) as [(kl), (n f), (cd)]
        T3x = np.empty((nP, no * nv, nQ), dtype=dtype)
        P.tn_expand2(T3x, T3, nv, 1, 2, P.occ_expand(o12, nv * nQ, no * nv * nQ), nQ, 1)
        Zt = np.zeros((nP * nQ, no * nv), dtype=dtype)                  # Z(nf;ij;ab) as [((ij), (ab)), (n f)]
        G = np.zeros((no * no, nP * nQ), dtype=dtype)                   # [(m l), ((jk), (cd))]
        WvT = np.empty((nQ, nv, no), dtype=dtype)                       # 1/2 <mn||ef> t(mnl,fcd)
        WoT = np.zeros((no, nP * nv), dtype=dtype)                      # 1/2 <mn||ef> t(nkl,efd) as [m, (kl), d]
        T3x4 = T3x.reshape(nP, no, nv, nQ)
        for x in range(no):
            S = c_(T3x4[:, x].transpose(1, 0, 2)).reshape(nv, nP * nQ)  # t(x jk; e cd) as [e, ((jk), (cd))]
            gemm(S.T, c_(oovv[x].transpose(1, 0, 2)).reshape(nv, no * nv), Zt, beta=1.0)
            gemm(c_(oovo[:, x].transpose(0, 2, 1)).reshape(no * no, nv), S, G, beta=1.0)
            WvT[:, :, x] = gemm(oovv_ef, c_(T3x4[:, x]).reshape(nP * nv, nQ)).T
            S2 = work('S', (nQ, nP * nv), dtype)                        # t(x kl; ef d) as [(ef), ((kl), d)]
            P.tn_expand2(S2, T3, nv, 2, 1, (o12.pos[x], np.arange(nP) * nv, o12.psg[x]), nP * nv, 1)
            gemm(c_(oovv_xp[:, x]), S2, WoT, beta=1.0)
            del S, S2
        work.clear()
        WoT = WoT.reshape(no, nP, nv).transpose(0, 2, 1)                # [m, d, (kl)]
        # t(jkl; ef c) as [(ef), (jkl), c]
        T3y = np.empty((nQ, nI3 * nv), dtype=dtype)
        P.tn_expand2(T3y, T3, nv, 2, 1, (np.arange(nI3), np.arange(nI3) * nv, np.ones(nI3)), nI3 * nv, 1)
        # Y(mn;jkl;d) = sum_{e<f} <mn||ef> t(jkl,efd) as [(jkl), d, (mn)]
        Y3 = c_(gemm(oovv_pp, T3y).reshape(nP, nI3, nv).transpose(1, 2, 0))
        Hs = gemm(vovv_p.reshape(nv * no, nQ), T3y).reshape(nv, no, nI3, nv)   # [d, m, (jkl), c]
        del T3y

    # ------------------------------------------------ intermediates of T2 and T3
    with timer('t4.imds'):
        Fv = fvv - 0.5 * _E('mnef,mndf->de', oovv, t2)
        Fo = foo + 0.5 * _E('mnef,lnef->ml', oovv, t2)
        Wpp = np.array(ints.vvvv_pp, dtype=dtype, order='C', copy=True)         # [(cd), (ef)]
        gemm(t2_pp.T, oovv_pp, Wpp, beta=1.0)
        Whh = np.asarray(ints.oooo) + 0.5 * _E('mnef,klef->mnkl', oovv, t2)
        Whh_p = c_(Whh[:, :, i2, j2])                                   # [m, n, (jk)]
        Wr = c_(np.asarray(ints.ovvo) + _E('mnef,lndf->mdel', oovv, t2))        # [m, d, e, l]
        Wh = c_(Whh[i2, j2][:, i2, j2] - _one_body_pairs(Fo, no) / 3)   # [(mn), (ij)]
        # Wv[(cd), e, l]
        Wv = np.array(ints.vvvo_p, dtype=dtype, order='C', copy=True)
        gemm(t2_pp.T, oovo_pp.reshape(nP, nv * no), Wv.reshape(nQ, nv * no), beta=1.0)
        x = _E('dmef,lmce->cdfl', vovv, t2)
        Wv += x[a2, b2]
        Wv -= x[b2, a2]
        del x
        Wv += _E('me,lmQ->Qel', fov, t2_xp)
        Wv += WvT
        del WvT
        Wv_t = c_(Wv.transpose(2, 1, 0))                                # [i, e, (ab)]
        del Wv
        # Wo[m, d, (kl)]
        Wo = np.array(np.asarray(ints.ovoo)[:, :, i2, j2], dtype=dtype, order='C', copy=True)
        x = _E('nmek,lnde->mdkl', oovo, t2)
        Wo += (x - x.transpose(0, 1, 3, 2))[:, :, i2, j2]
        Wo -= _E('dmQ,PQ->mdP', vovv_p, t2_pp)
        Wo -= _E('me,Pde->mdP', fov, t2_px)
        Wo -= WoT
        Wo_t = c_(Wo.transpose(2, 0, 1))                                # [(ij), m, a]
        del x, WoT, Wo
        # the part of Xo without T4, as [(jkl), m, (cd)], for one m at a time:
        # P(l/jk) [ <mn||el> t(njk,ecd) + 1/2 Whh(mn,jk) t(nl,cd) ] - <dm||ef> t(jkl,efc) + (c <-> d)
        Gm = G.reshape(no, no, nP, nQ)
        Xo = np.empty((nI3, no, nQ), dtype=dtype)
        t2_n = t2_xp.reshape(no, no * nQ)                               # [n, (l (cd))]
        for m in range(no):
            Gm[m] += 0.5 * gemm(c_(Whh_p[m].T), t2_n).reshape(nP, no, nQ).transpose(1, 0, 2)
            Xo[:, m] = P.fold(Gm[m], o12, 0, 1)
            Xo[:, m] -= (Hs[b2, m, :, a2] - Hs[a2, m, :, b2]).T
        del G, Gm, Hs

    # ---- the part of Xv without T4, for all pairs: [(kl), e, (bcd)]
    if want('b2'):
        with timer('t4.xv'):
            Xv = np.zeros((nP, nv, nA3), dtype=dtype)
            oovo_t = c_(oovo_pp.transpose(2, 1, 0))                             # [l, e, (mn)]
            vovv_ed = c_(vovv.transpose(2, 0, 1, 3)).reshape(nv * nv, no * nv)  # [(e d), (m f)]
            Wr_k = c_(Wr.transpose(3, 2, 1, 0)).reshape(no, nv * nv, no)        # [k, (e b), m]
            Mx = work('X', (nv * nv, nQ), dtype)
            for p, (k, l) in enumerate(pairs_o):
                # - P(kl) sum_{m<n} <mn||el> t(mnk, bcd)
                gemm(oovo_t[l], t3o[k], Xv[p], alpha=-1.0, beta=1.0)
                gemm(oovo_t[k], t3o[l], Xv[p], alpha=1.0, beta=1.0)
                # P(d/bc) [ <dm||ef> t(m kl; f, bc) + P(kl) sum_m Wr(m,d,e,k) t(ml,bc) ]
                gemm(vovv_ed, T3x[p], Mx)
                gemm(Wr_k[k], t2_xp[:, l], Mx, alpha=1.0, beta=1.0)
                gemm(Wr_k[l], t2_xp[:, k], Mx, alpha=-1.0, beta=1.0)
                P.tn_reduce(Xv[p], Mx.reshape(nv, nv, nQ), nv, 1, 2, outer=0)
            # + P(d/bc) 1/2 sum_f Wpp(bc,ef) t(kl,fd), from rows of Wpp
            for r0, r1 in _batches(nQ, batch_bytes // (2 * isz * nv * nv)):
                nr = r1 - r0
                W = work('W', (nr * nv, nv), dtype)                     # [(bc), e, f]
                P.tn_expand(W.reshape(nr, nv, nv), c_(Wpp[r0:r1]), nv, 1, 1, outer=0)
                for p in range(nP):
                    M2 = gemm(W, t2_px[p], work('X', (nr * nv, nv), dtype))             # [(bc), e, d]
                    P.tn_reduce(Xv[p], M2.reshape(nr, nv, nv), nv, 2, 1, rows=(r0, r1), scale=0.5, outer=1)
            del oovo_t, vovv_ed, Wr_k
            Mx = W = M2 = None
            work.clear()
    # Wq = Wpp + F_vv on pairs / 3, in the storage of Wpp
    from zccsdt.cc import ccsdt_opt, t3pack
    Wq = ccsdt_opt.add_one_body_pairs(Wpp, c_(Fv), t3pack.triple_map(nv), 1.0 / 3)
    del Wpp

    # ---- pp ladder with F_vv, in batches of occupied quadruples; from the same view of T4
    # (virtual indices in two pairs) the T4 part of Xo and the T4 terms of R2 and R3
    if want('pp', 'r2', 'b1'):
        with timer('t4.pp'):
            if want('b1'):
                # per n: <mn||ef> as [m, (ef)] above - <cn||ef> as [c, (ef)]
                GL = c_(np.concatenate((oovv_xp.transpose(1, 0, 2), -vovv_p.transpose(1, 0, 2)), axis=1))
                L4 = np.empty((4 * (no + nv), nQ), dtype=dtype)
                U3 = np.zeros((nI3, nv, nQ), dtype=dtype)               # [(ijk), c, (ab)]
            for I0, I1 in _batches(nI, cpp):
                nb = I1 - I0
                ent = (np.arange(I0, I1), np.arange(nb) * nQ, np.ones(nb))
                Tv = work('M', (nQ, nb * nQ), dtype)                    # t(I; ef, cd) as [(ef), (I, (cd))]
                expand2(Tv, T4, nv, 2, 2, ent, nb * nQ, 1)
                if want('r2'):
                    for k, I in enumerate(range(I0, I1)):
                        g = c_(oovv_pp[o22.idx1[I]] * o22.sgn[I][:, None])              # [6, (ef)]
                        R2[o22.idx2[I]] += mm(g, Tv[:, k * nQ:(k + 1) * nQ])
                if want('b1'):
                    # Xo(m;jkl;cd) += <mn||ef> t(n jkl; ef, cd);  R3 -= P(c/ab) <cm||ef> t(m ijk; ef, ab)
                    for k, I in enumerate(range(I0, I1)):
                        ns, Js, sg = o13.idx1[I], o13.idx2[I], o13.sgn[I]
                        for q in range(4):
                            np.multiply(GL[ns[q]], sg[q], out=L4[q * (no + nv):(q + 1) * (no + nv)])
                        out = mm(L4, Tv[:, k * nQ:(k + 1) * nQ], work('O', (4 * (no + nv), nQ), dtype))
                        out = out.reshape(4, no + nv, nQ)
                        for q in range(4):
                            Xo[Js[q]] += out[q, :no]
                            U3[Js[q]] += out[q, no:]
                if want('pp'):
                    X = mm(Wq, Tv, work('X', (nQ, nb * nQ), dtype))
                    reduce2(R4, X, nv, 2, 2, ent, nb * nQ, 1)
            if want('b1'):
                P.tn_reduce(R3, U3, nv, 1, 2, outer=0)
                del GL, L4, U3
            Tv = X = out = None                 # (views of the work arrays: they would pin them)
            work.clear()
    del Wq

    # ---- ring, in batches of the columns (bcd) of t(m jkl; e bcd); from the same view of T4
    # the T4 part of Xv and the T4 terms of R3 with f_ov and <mn||ek>
    if want('ring', 'b2'):
        with timer('t4.ring'):
            Wr2 = c_(Wr.transpose(3, 1, 0, 2)).reshape(no * nv, no * nv)        # [(i a), (m e)]
            fov_flat = fov.ravel()
            if want('b2'):
                # per n: - 1/2 <mn||ek> as [k, (m e)] above 1/2 <mn||ef> as [e, (m f)]
                GB = c_(np.concatenate((-0.5 * oovo.transpose(1, 3, 0, 2).reshape(no, no, no * nv),
                                        0.5 * oovv.transpose(1, 2, 0, 3).reshape(no, nv, no * nv)), axis=1))
                L3 = np.empty((3 * (no + nv), no * nv), dtype=dtype)
            erow, ebase, esg = P.occ_expand(o13, nv * nI3, 1)
            rrow, rbase, rsg = P.occ_reduce(o13, nv * nI3, 1)
            for c in _batches(nA3, cring):
                nc = c[1] - c[0]
                M = work('M', (no * nv, nI3 * nc), dtype)               # t(m J; e K) as [(m e), (J K)]
                with timer('expand'):
                    P.tn_expand1(M, T4, nv, 3, (erow, ebase * nc, esg), nI3 * nc, cols=c)
                if want('ring'):
                    R3[:, c[0]:c[1]] += (fov_flat @ M).reshape(nI3, nc)
                    X = mm(Wr2, M, work('X', (no * nv, nI3 * nc), dtype))
                    with timer('reduce'):
                        P.tn_reduce1(R4, X, nv, 3, (rrow, rbase * nc, rsg), nI3 * nc, cols=c)
                if want('b2'):
                    U = work('U', (no, nP, nc), dtype)                  # - <mn||ek> t(mn ij; e, abc) as [k, (ij), (abc)]
                    P.zero(U)
                    for J in range(nI3):
                        ns, ps, sg = o12.idx1[J], o12.idx2[J], o12.sgn[J]
                        for q in range(3):
                            np.multiply(GB[ns[q]], sg[q], out=L3[q * (no + nv):(q + 1) * (no + nv)])
                        out = mm(L3, M[:, J * nc:(J + 1) * nc], work('O', (3 * (no + nv), nc), dtype))
                        out = out.reshape(3, no + nv, nc)
                        for q in range(3):
                            U[:, ps[q]] += out[q, :no]
                            Xv[ps[q], :, c[0]:c[1]] += out[q, no:]
                    row, off, sg = P.occ_reduce(o12, nP * nc, nc)
                    axpy_rows(R3, U.reshape(no * nP, nc), row, off, sg, col0=c[0])
            del Wr2
            M = X = U = out = None
            work.clear()

    # ---- hh ladder with F_oo: batches of virtual quadruples
    if want('hh'):
        with timer('t4.hh'):
            WhT = c_(Wh.T)                                              # [(ij), (mn)]
            row, base, sg = P.occ_reduce(o22, nP, 1)
            for c0, c1 in _batches(nA, chh):
                nc = c1 - c0
                Th = work('M', (nP * nP, nc), dtype)                    # t(mn kl; A)
                rows(Th, T4, o22.pos.ravel(), o22.psg.ravel(), (c0, c1))
                X = mm(WhT, Th.reshape(nP, nP * nc), work('X', (nP, nP * nc), dtype))
                axpy_rows(R4, X.reshape(nP * nP, nc), row, base * nc, sg, col0=c0)
            Th = X = None
            work.clear()

    # ---- one occupied quadruple at a time: the products with the six-index intermediates
    # and T3 with T3, the splits of the quadruple summed by the products
    if want('b1', 'b2', 't3t3h', 't3t3r'):
        with timer('t4.rows'):
            Zt3 = Zt.reshape(nP, nQ, no * nv)
            LA, RA = (np.empty((4 * nv, nQ), dtype=dtype) for _ in range(2))
            LB, RB = (np.empty((4 * no, nQ), dtype=dtype) for _ in range(2))
            for I in range(nI):
                xs, Js, s13 = o13.idx1[I], o13.idx2[I], o13.sgn[I]
                ps, qs, s22 = o22.idx1[I], o22.idx2[I], o22.sgn[I]
                if want('b1'):
                    # - sum_e Wv(ab,e,i) t(jkl,cde) + sum_m t(im,ab) Xo(m;jkl;cd), the splits (i|jkl)
                    with timer('operands'):
                        for q in range(4):
                            x, J, sg = xs[q], Js[q], int(s13[q])
                            np.multiply(Wv_t[x], -sg, out=LA[q * nv:(q + 1) * nv])      # [e, (ab)]
                            P.tn_expand(RA[q * nv:(q + 1) * nv], T3[J], nv, 1, 2)       # t(jkl; e, cd)
                            np.multiply(t2_xp[x], sg, out=LB[q * no:(q + 1) * no])      # [m, (ab)]
                            RB[q * no:(q + 1) * no] = Xo[J]                             # [m, (cd)]
                # X[(ab),(cd)], in batches of the columns (cd)
                for c in (_batches(nQ, c22) if want('b1', 't3t3r') else ()):
                    X22 = work('M', (nQ, c[1] - c[0]), dtype)
                    beta = 0.0
                    if want('b1'):
                        mm(LA.T, RA[:, c[0]:c[1]], X22)
                        mm(LB.T, RB[:, c[0]:c[1]], X22, beta=1.0)
                        beta = 1.0
                    if want('t3t3r'):
                        # sum_{nf} Z(nf;ij;ab) t(nkl, fcd): the same for (ij,ab | kl,cd) and
                        # (kl,cd | ij,ab) - the three splits with i in the first pair, twice (the 1/2)
                        for q in range(3):
                            mm(Zt3[ps[q]], T3x[qs[q]][:, c[0]:c[1]], X22, alpha=float(s22[q]), beta=beta)
                            beta = 1.0
                    with timer('reduce'):
                        P.tn_reduce(R4[I], X22, nv, 2, 2, cols=c)
                # X[a,(bcd)], in batches of the columns (bcd)
                for c in (_batches(nA3, c13) if want('b2', 't3t3h') else ()):
                    X13 = work('X', (nv, c[1] - c[0]), dtype)
                    beta = 0.0
                    if want('b2'):
                        # sum_m Wo(m,a,ij) t(mkl,bcd) + sum_e t(ij,ae) Xv(e;kl;bcd), the splits (ij|kl)
                        for q in range(6):
                            sg = float(s22[q])
                            mm(Wo_t[ps[q]].T, t3o[:, qs[q], c[0]:c[1]], X13, alpha=sg, beta=beta)
                            mm(t2_px[ps[q]], Xv[qs[q]][:, c[0]:c[1]], X13, alpha=sg, beta=1.0)
                            beta = 1.0
                    if want('t3t3h'):
                        # - sum_{m<n} Y(mn;jkl;a) t(mni, bcd), the splits (i|jkl)
                        for q in range(4):
                            mm(Y3[Js[q]], t3o[xs[q]][:, c[0]:c[1]], X13, alpha=-float(s13[q]), beta=beta)
                            beta = 1.0
                    with timer('reduce'):
                        P.tn_reduce(R4[I], X13, nv, 1, 3, cols=c)
    return R4, R3, R2
