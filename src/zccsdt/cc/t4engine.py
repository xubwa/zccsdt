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

Every product is a matrix product of two operands in batches of columns:
a view of the packed T4 with its indices in two occupied and two virtual
groups (or a view of T3, or an intermediate), written by a kernel straight
in the layout of the product (zccsdt.cc.t4pack: tn_expand2, tn_rows), and
the product is added to the packed residual by the antisymmetrizing
reduction (tn_reduce2, tn_axpy_rows).  The batches run over the second
occupied and the second virtual group.

What is kept besides T4 and its residual: the batch arrays, and of the size
of T3 the operand Z (9 x T3) and T3 in rows by one occupied index and a
pair (3 x T3); Xv, the largest intermediate (v^4 / 6 per pair of occupied
indices), is formed for the pairs of a batch.
'''

from dataclasses import dataclass
from math import comb

import numpy as np

from zccsdt.blas import gemm
from zccsdt.cc import t4pack as P
from zccsdt.timing import timer

BATCH_BYTES = 1 << 30           # an operand or a product of one batch, at most
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


class _Pool:
    '''Work arrays by name, kept between batches (a fresh large array pays
    its page faults again).'''

    def __init__(self):
        self.arrays = {}

    def __call__(self, name, shape, dtype):
        n = int(np.prod(shape))
        a = self.arrays.get(name)
        if a is None or a.size < n or a.dtype != dtype:
            self.arrays[name] = None                    # (free before allocating again)
            a = self.arrays[name] = np.empty(max(n, 1), dtype=dtype)
        return a[:n].reshape(shape)


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
    work = _Pool()

    def c_(x):
        return np.ascontiguousarray(x, dtype=dtype)

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
        # slabs t(x jk; e cd) as [e, (jk), (cd)] and t(x kl; ef d) as [(ef), (kl), d], x = 0 ..
        Zt = np.zeros((nP * nQ, no * nv), dtype=dtype)                  # Z(nf;ij;ab) as [((ij), (ab)), (n f)]
        G = np.zeros((no * no, nP * nQ), dtype=dtype)                   # [(m l), ((jk), (cd))]
        WvT = np.empty((nQ, nv, no), dtype=dtype)                       # 1/2 <mn||ef> t(mnl,fcd)
        WoT = np.zeros((no, nP * nv), dtype=dtype)                      # 1/2 <mn||ef> t(nkl,efd) as [m, (kl), d]
        for x in range(no):
            S = work('S', (nv, nP * nQ), dtype)
            P.tn_expand2(S, T3, nv, 1, 2, (o12.pos[x], np.arange(nP) * nQ, o12.psg[x]), nP * nQ, 1)
            gemm(S.T, c_(oovv[x].transpose(1, 0, 2)).reshape(nv, no * nv), Zt, beta=1.0)
            gemm(c_(oovo[:, x].transpose(0, 2, 1)).reshape(no * no, nv), S, G, beta=1.0)
            St = c_(S.reshape(nv, nP, nQ).transpose(1, 0, 2)).reshape(nP * nv, nQ)      # [((mn) f), (cd)]
            WvT[:, :, x] = gemm(oovv_ef, St).T
            S2 = work('S', (nQ, nP * nv), dtype)
            P.tn_expand2(S2, T3, nv, 2, 1, (o12.pos[x], np.arange(nP) * nv, o12.psg[x]), nP * nv, 1)
            gemm(c_(oovv_xp[:, x]), S2, WoT, beta=1.0)
            del S, St, S2
        WoT = WoT.reshape(no, nP, nv).transpose(0, 2, 1)                # [m, d, (kl)]
        # t(jkl; ef c) as [(ef), (jkl), c]
        T3y = np.empty((nQ, nI3 * nv), dtype=dtype)
        P.tn_expand2(T3y, T3, nv, 2, 1, (np.arange(nI3), np.arange(nI3) * nv, np.ones(nI3)), nI3 * nv, 1)
        Y2 = gemm(oovv_pp, T3y)                                         # Y(mn;jkl;d) as [(mn), ((jkl), d)]
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
        Wq = c_(Wpp + _one_body_pairs(Fv, nv) / 3)
        Wh = c_(Whh[i2, j2][:, i2, j2] - _one_body_pairs(Fo, no) / 3)   # [(mn), (ij)]
        # Wv[(cd), e, l]
        Wv = np.array(ints.vvvo_p, dtype=dtype, order='C', copy=True)
        gemm(t2_pp.T, oovo_pp.reshape(nP, nv * no), Wv.reshape(nQ, nv * no), beta=1.0)
        x = _E('dmef,lmce->cdfl', vovv, t2)
        Wv += (x - x.transpose(1, 0, 2, 3))[a2, b2]
        Wv += _E('me,lmQ->Qel', fov, t2_xp)
        Wv += WvT
        # Wo[m, d, (kl)]
        Wo = np.array(np.asarray(ints.ovoo)[:, :, i2, j2], dtype=dtype, order='C', copy=True)
        x = _E('nmek,lnde->mdkl', oovo, t2)
        Wo += (x - x.transpose(0, 1, 3, 2))[:, :, i2, j2]
        Wo -= _E('dmQ,PQ->mdP', vovv_p, t2_pp)
        Wo -= _E('me,Pde->mdP', fov, t2_px)
        Wo -= WoT
        del x, WvT, WoT
        # the part of Xo without T4, [m, (jkl), (cd)]
        Gm = G.reshape(no, no, nP, nQ)
        Gm += 0.5 * _E('mnJ,nlQ->mlJQ', Whh_p, t2_xp)
        Xo0 = np.empty((no, nI3, nQ), dtype=dtype)
        for m in range(no):
            Xo0[m] = P.fold(Gm[m], o12, 0, 1)                           # P(l/jk)
        Xo0 -= _antisym_cd(Hs, a2, b2)                                  # - <dm||ef> t(jkl,efc), c and d exchanged
        del G, Gm, Hs

    # ---- pp ladder with F_vv: batches of occupied quadruples;  the T4 term of R2
    if want('pp', 'r2'):
        with timer('t4.pp'):
            for I0, I1 in _batches(nI, batch_bytes // (isz * nQ * nQ)):
                nb = I1 - I0
                ent = (np.arange(I0, I1), np.arange(nb) * nQ, np.ones(nb))
                Tv = work('M', (nQ, nb * nQ), dtype)                    # t(I; ef, cd) as [(ef), (I, (cd))]
                P.tn_expand2(Tv, T4, nv, 2, 2, ent, nb * nQ, 1)
                if want('r2'):
                    for k, I in enumerate(range(I0, I1)):
                        g = c_(oovv_pp[o22.idx1[I]] * o22.sgn[I][:, None])              # [6, (ef)]
                        R2[o22.idx2[I]] += gemm(g, Tv[:, k * nQ:(k + 1) * nQ])
                if want('pp'):
                    X = gemm(Wq, Tv, work('X', (nQ, nb * nQ), dtype))
                    P.tn_reduce2(R4, X, nv, 2, 2, ent, nb * nQ, 1)

    # ---- hh ladder with F_oo: batches of virtual quadruples
    if want('hh'):
        with timer('t4.hh'):
            WhT = c_(Wh.T)                                              # [(ij), (mn)]
            row, base, sg = P.occ_reduce(o22, nP, 1)
            for c0, c1 in _batches(nA, batch_bytes // (isz * nP * nP)):
                nc = c1 - c0
                Th = work('M', (nP * nP, nc), dtype)                    # t(mn kl; A)
                P.tn_rows(Th, T4, o22.pos.ravel(), o22.psg.ravel(), (c0, c1))
                X = gemm(WhT, Th.reshape(nP, nP * nc), work('X', (nP, nP * nc), dtype))
                P.tn_axpy_rows(R4, X.reshape(nP * nP, nc), row, base * nc, sg, col0=c0)

    # ---- ring, and the f_ov term of R3
    if want('ring'):
        with timer('t4.ring'):
            Wr2 = c_(Wr.transpose(3, 1, 0, 2)).reshape(no * nv, no * nv)        # [(i a), (m e)]
            fov_flat = fov.ravel()
            for b in _batches(nI3, max(1, min(nI3, int((batch_bytes / (isz * no * nv)) ** 0.5)))):
                nb = b[1] - b[0]
                for c in _batches(nA3, batch_bytes // (isz * no * nv * nb)):
                    nc = c[1] - c[0]
                    M = work('M', (no * nv, nb * nc), dtype)            # t(m J; e K) as [(m e), (J K)]
                    P.tn_expand2(M, T4, nv, 1, 3, P.occ_expand(o13, nv * nb * nc, nc, b), nb * nc, 1, cols=c)
                    R3[b[0]:b[1], c[0]:c[1]] += (fov_flat @ M).reshape(nb, nc)
                    X = gemm(Wr2, M, work('X', (no * nv, nb * nc), dtype))
                    P.tn_reduce2(R4, X, nv, 1, 3, P.occ_reduce(o13, nv * nb * nc, nc, b), nb * nc, 1, cols=c)

    # ---- (i, ab) x (jkl, cd): Wv with T3 and T2 with Xo;  the <cm||ef> term of R3
    if want('b1'):
        with timer('t4.b1'):
            L1 = c_(np.concatenate((-Wv.transpose(2, 0, 1), t2_xp.transpose(0, 2, 1)), axis=2))
            L1 = L1.reshape(no * nQ, nv + no)                           # [(i (ab)), e + m]
            vovv_m = vovv_p.reshape(nv, no * nQ)                        # [c, (m (ef))]
            oovv_m = oovv_xp.reshape(no, no * nQ)                       # [m, (n (ef))]
            for b in _batches(nI3, max(1, min(nI3, int((batch_bytes / (isz * no * nQ)) ** 0.5)))):
                nb = b[1] - b[0]
                rows_b = (np.arange(b[0], b[1]), None, np.ones(nb))
                for c in _batches(nQ, batch_bytes // (isz * no * nQ * nb)):
                    nc = c[1] - c[0]
                    ent = (rows_b[0], np.arange(nb) * nc, rows_b[2])
                    M = work('M', (no * nQ, nb * nc), dtype)            # t(n J; ef, cd) as [(n (ef)), (J (cd))]
                    P.tn_expand2(M, T4, nv, 2, 2, P.occ_expand(o13, nQ * nb * nc, nc, b), nb * nc, 1, cols=c)
                    # R3 -= P(c/ab) sum_{m, e<f} <cm||ef> t(m ijk; ef, ab)
                    U = gemm(vovv_m, M, work('U', (nv, nb * nc), dtype), alpha=-1.0)
                    P.tn_reduce2(R3, U, nv, 1, 2, ent, nb * nc, 1, cols=c)
                    Rt = work('Rt', (nv + no, nb * nc), dtype)
                    P.tn_expand2(Rt[:nv], T3, nv, 1, 2, ent, nb * nc, 1, cols=c)        # t(J; e, cd)
                    Rt[nv:] = Xo0[:, b[0]:b[1], c[0]:c[1]].reshape(no, nb * nc)
                    gemm(oovv_m, M, Rt[nv:], beta=1.0)                  # + <mn||ef> t(n J; ef, cd)
                    X = gemm(L1, Rt, work('X', (no * nQ, nb * nc), dtype))
                    P.tn_reduce2(R4, X, nv, 2, 2, P.occ_reduce(o13, nQ * nb * nc, nc, b), nb * nc, 1, cols=c)

    # ---- (ij, a) x (kl, bcd): Wo with T3 and T2 with Xv;  the <mn||ek> term of R3
    if want('b2'):
        with timer('t4.b2'):
            L2 = c_(np.concatenate((Wo.transpose(2, 1, 0), t2_px), axis=2)).reshape(nP * nv, no + nv)
            oovo_m = c_(oovo_pp.transpose(2, 0, 1)).reshape(no, nP * nv)        # [k, ((mn) e)]
            oovo_t = c_(oovo_pp.transpose(2, 1, 0))                             # [l, e, (mn)]
            vovv_ed = c_(vovv.transpose(2, 0, 1, 3)).reshape(nv * nv, no * nv)  # [(e d), (m f)]
            Wr_k = c_(Wr.transpose(3, 2, 1, 0)).reshape(no, nv * nv, no)        # [k, (e b), m]
            for b in _batches(nP, max(1, min(nP, int((batch_bytes / (isz * nP * nv)) ** 0.5)))):
                nb = b[1] - b[0]
                with timer('t4.xv'):
                    # the part of Xv without T4 for the pairs (kl) of the batch: [e, (kl), (bcd)]
                    Xv0 = work('Xv', (nv, nb, nA3), dtype)
                    P.zero(Xv0)
                    Rb = work('M', (no * nv, nb * nQ), dtype)           # t(m kl; f, bc) as [(m f), ((kl) (bc))]
                    P.tn_expand2(Rb, T3, nv, 1, 2, P.occ_expand(o12, nv * nb * nQ, nQ, b), nb * nQ, 1)
                    Mx = gemm(vovv_ed, Rb, work('X', (nv * nv, nb * nQ), dtype))        # [e, d, (kl), (bc)]
                    Mk = Mx.reshape(nv * nv, nb, nQ)
                    for n_, (k, l) in enumerate(pairs_o[b[0]:b[1]]):
                        # - P(kl) sum_{m<n} <mn||el> t(mnk, bcd)
                        gemm(oovo_t[l], t3o[k], Xv0[:, n_], alpha=-1.0, beta=1.0)
                        gemm(oovo_t[k], t3o[l], Xv0[:, n_], alpha=1.0, beta=1.0)
                        # + P(kl) sum_m Wr(m,b,e,k) t(ml,cd), to be antisymmetrized with Mx
                        gemm(Wr_k[k], t2_xp[:, l], Mk[:, n_], alpha=1.0, beta=1.0)
                        gemm(Wr_k[l], t2_xp[:, k], Mk[:, n_], alpha=-1.0, beta=1.0)
                    ent = (np.arange(nv * nb), (np.arange(nv)[:, None] * (nv * nb * nQ)
                                                + np.arange(nb)[None, :] * nQ).ravel(), np.ones(nv * nb))
                    P.tn_reduce2(Xv0.reshape(nv * nb, nA3), Mx, nv, 1, 2, ent, nb * nQ, 1)      # P(d/bc)
                    # + P(d/bc) 1/2 sum_f Wpp(bc,ef) t(kl,fd), from rows of Wpp
                    tl = c_(t2_px[b[0]:b[1]].transpose(1, 0, 2)).reshape(nv, nb * nv)   # [f, ((kl) d)]
                    for r0, r1 in _batches(nQ, batch_bytes // (isz * nv * nv * (nb + 1))):
                        nr = r1 - r0
                        W = work('W', (nr, nv, nv), dtype)              # [(bc), e, f]
                        P.tn_expand(W, c_(Wpp[r0:r1]), nv, 1, 1, outer=0)
                        M2 = gemm(W.reshape(nr * nv, nv), tl, work('X', (nr * nv, nb * nv), dtype))
                        P.tn_reduce(Xv0.reshape(nv * nb, nA3), M2.reshape(nr, nv * nb, nv), nv, 2, 1,
                                    rows=(r0, r1), scale=0.5, outer=1)
                for c in _batches(nA3, batch_bytes // (isz * nP * nv * nb)):
                    nc = c[1] - c[0]
                    M = work('M', (nP * nv, nb * nc), dtype)            # t(mn kl; f, bcd) as [((mn) f), ((kl) (bcd))]
                    P.tn_expand2(M, T4, nv, 1, 3, P.occ_expand(o22, nv * nb * nc, nc, b), nb * nc, 1, cols=c)
                    # R3 -= P(k/ij) sum_{m<n, e} <mn||ek> t(mn ij; e, abc)
                    U = gemm(oovo_m, M, work('U', (no, nb * nc), dtype), alpha=-1.0)
                    row, off, sg = P.occ_reduce(o12, nb * nc, nc, b)
                    P.tn_axpy_rows(R3, U.reshape(no * nb, nc), row, off, sg, col0=c[0])
                    Rt = work('Rt', (no + nv, nb * nc), dtype)
                    P.tn_rows(Rt[:no].reshape(no * nb, nc), T3, o12.pos[:, b[0]:b[1]].ravel(),
                              o12.psg[:, b[0]:b[1]].ravel(), c)         # t(m kl; bcd)
                    Rt[no:] = Xv0[:, :, c[0]:c[1]].reshape(nv, nb * nc)
                    gemm(oovv_ef, M, Rt[no:], beta=1.0)                 # + <mn||ef> t(mn kl; f, bcd)
                    X = gemm(L2, Rt, work('X', (nP * nv, nb * nc), dtype))
                    P.tn_reduce2(R4, X, nv, 1, 3, P.occ_reduce(o22, nv * nb * nc, nc, b), nb * nc, 1, cols=c)

    # ---- T3 with T3 through <oo||vv>
    if want('t3t3h'):
        with timer('t4.t3t3h'):
            # sum_{m<n} t(mni, abc) Y(mn;jkl;d) as [(i (abc)), ((jkl) d)], rows (abc) in batches
            # (the minus of the term is the sign of (abc|d) against (d|abc))
            for c in _batches(nA3, batch_bytes // (isz * no * nI3 * nv)):
                nc = c[1] - c[0]
                L3 = c_(t3o[:, :, c[0]:c[1]].transpose(0, 2, 1)).reshape(no * nc, nP)
                X = gemm(L3, Y2, work('X', (no * nc, nI3 * nv), dtype))
                P.tn_reduce2(R4, X, nv, 3, 1, P.occ_reduce(o13, nc * nI3 * nv, nv), nI3 * nv, 1, rows=c)
    if want('t3t3r'):
        with timer('t4.t3t3r'):
            # sum_{nf} Z(nf;ij;ab) t(nkl, fcd): the same for (ij,ab | kl,cd) and (kl,cd | ij,ab), so
            # the splits with i in the first pair are taken, twice (the 1/2): rows (ij) below (kl)
            for b in _batches(nP, max(1, min(nP, int((batch_bytes / (isz * nP * nQ)) ** 0.5)))):
                nb = b[1] - b[0]
                for c in _batches(nQ, batch_bytes // (isz * nP * nQ * nb)):
                    nc = c[1] - c[0]
                    Rb = work('M', (no * nv, nb * nc), dtype)           # t(n kl; f, cd) as [(n f), ((kl) (cd))]
                    P.tn_expand2(Rb, T3, nv, 1, 2, P.occ_expand(o12, nv * nb * nc, nc, b), nb * nc, 1, cols=c)
                    X = gemm(Zt[:b[1] * nQ], Rb, work('X', (b[1] * nQ, nb * nc), dtype))
                    P.tn_reduce2(R4, X, nv, 2, 2, P.occ_reduce(o22, nQ * nb * nc, nc, b, only=(0, 1, 2)),
                                 nb * nc, 1, cols=c)
    return R4, R3, R2


def _antisym_cd(Hs, a2, b2):
    '''[m, J, (cd)] = H[d, m, J, c] - H[c, m, J, d] for c < d.'''
    return (Hs[b2, :, :, a2] - Hs[a2, :, :, b2]).transpose(1, 2, 0)
