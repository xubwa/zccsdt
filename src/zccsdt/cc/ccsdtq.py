'''Spin-orbital CCSDTQ with T4 on its unique block (i<j<k<l, a<b<c<d), real
or complex.

Equations: the T1-dressed ones of zccsdt.cc.ccsdtq_terms, in the form

    R4 = P(ab/cd) sum_{e<f} Wpp(ab,ef) t(ijkl,efcd)          pp ladder, F_vv folded in
       + P(ij/kl) sum_{m<n} Whh(mn,ij) t(mnkl,abcd)          hh ladder, F_oo folded in
       + P(i/jkl) P(a/bcd) sum_{me} Wr(ma,ei) t(mjkl,ebcd)   ring
       + P(i/jkl) P(ab/cd) [ - sum_e Wv(ab,e,i) t(jkl,cde) + sum_m t(im,ab) Xo(m;jkl;cd) ]
       + P(ij/kl) P(a/bcd) [   sum_m Wo(m,a,ij) t(mkl,bcd) + sum_e t(ij,ae) Xv(e;kl;bcd) ]
       - P(i/jkl) P(d/abc) sum_{m<n} t(mni,abc) Y(mn;jkl;d)
       + 1/2 P(ij/kl) P(ab/cd) sum_{nf} Z(nf;ij;ab) t(nkl,fcd)

with F and the two-body operators W dressed by T2 as in the T3 equation, the
<vv||vo> and <ov||oo> intermediates Wv and Wo dressed by T2 and T3, the
six-index intermediates Xo and Xv (T2, T3 and T4 with one integral, an
index of T2 left to contract), and Y and Z (one integral with a T3).  Every
piece is the corresponding group of terms of the table (tests).

T4 is never unpacked.  Each product takes a view of the packed block
(zccsdt.cc.t4pack.view: the tuples split into the group that is summed and
the rest) in batches of columns and adds its antisymmetrized result back
(t4pack.reduce).  T2 and T3 are full arrays here, and the T1, T2, T3
equations without T4 are those of the reference code (zccsdt.cc.ccsdt and
the table); T4 adds its terms to R2 and R3.

This is the first version: numpy for the views and reductions, memory for
everything (T3 in full, T4 packed, the integrals as the full dressed
tensor).
'''

from dataclasses import dataclass
from math import comb

import numpy as np

from zccsdt.cc import ccsdt, ccsdtq_terms as Q, t4pack as P
from zccsdt.timing import timer

BATCH_BYTES = 1 << 30           # the product of one batch of columns, at most


def _E(sub, *ops):
    return np.einsum(sub, *ops, optimize=True)


def _pairs(n):
    t = P.tuples(n, 2)
    return t[:, 0], t[:, 1]


def _one_body_pairs(F, n):
    '''W[(pq),(rs)] = F[p,r] d(q,s) - F[p,s] d(q,r) - F[q,r] d(p,s) + F[q,s] d(p,r)
    on pairs p<q, r<s: the one-body operator on a pair.'''
    p, q = _pairs(n)
    eye = np.eye(n)
    return (F[p[:, None], p[None, :]] * eye[q[:, None], q[None, :]]
            - F[p[:, None], q[None, :]] * eye[q[:, None], p[None, :]]
            - F[q[:, None], p[None, :]] * eye[p[:, None], q[None, :]]
            + F[q[:, None], q[None, :]] * eye[p[:, None], p[None, :]])


def _batches(n, step):
    step = int(max(1, step))
    return [(x, min(n, x + step)) for x in range(0, n, step)]


def _antisym_single_pair(G, n, axis_single, axis_pair):
    '''sum over the splits (x | pair) of sorted triples of sgn G[.., x, .., pair, ..]:
    the triple index replaces the pair axis, the single axis is dropped.'''
    sp = P.split(n, 1, 2)
    G = np.moveaxis(G, (axis_single, axis_pair), (0, 1))
    out = 0
    for s in range(sp.nsplit):
        out = out + G[sp.idx1[:, s], sp.idx2[:, s]] * sp.sgn[:, s].reshape((-1,) + (1,) * (G.ndim - 2))
    # [triples, rest..] -> the triples where the pair axis was
    rest = [ax for ax in range(max(axis_single, axis_pair) + 1) if ax not in (axis_single, axis_pair)]
    target = axis_pair - (1 if axis_single < axis_pair else 0)
    return np.moveaxis(out, 0, target)


@dataclass
class Intermediates:
    '''What the T4 residual needs of F, V, T2 and T3 (all dense).'''
    Wq: np.ndarray      # [(ab),(ef)]  Wpp + F_vv on pairs
    Wh: np.ndarray      # [(mn),(ij)]  Whh + F_oo on pairs
    Wr: np.ndarray      # [m,a,e,i]
    Wv: np.ndarray      # [(ab),e,i]   dressed <ab||ei>
    Wo: np.ndarray      # [m,a,(ij)]   dressed <ma||ij>
    Whh: np.ndarray     # [m,n,(jk)]
    Wpp: np.ndarray     # [(bc),e,f]


def intermediates(t2, t3, f, v, nocc):
    '''From dressed f[p,q] and v = <pq||rs>, T2 and T3.'''
    n = v.shape[0]
    nvir = n - nocc
    o, V = slice(0, nocc), slice(nocc, n)
    oovv = v[o, o, V, V]
    i2, j2 = _pairs(nocc)
    a2, b2 = _pairs(nvir)
    Fv = f[V, V] - 0.5 * _E('mnef,mndf->de', oovv, t2)
    Fo = f[o, o] + 0.5 * _E('mnef,lnef->ml', oovv, t2)
    Wpp = v[V, V, V, V] + 0.5 * _E('mncd,mnef->cdef', t2, oovv)
    Whh = v[o, o, o, o] + 0.5 * _E('mnef,klef->mnkl', oovv, t2)
    Wr = v[o, V, V, o] + _E('mnef,lndf->mdel', oovv, t2)
    Wq = Wpp[a2[:, None], b2[:, None], a2[None, :], b2[None, :]] + _one_body_pairs(Fv, nvir) / 3
    Wh = Whh[i2[:, None], j2[:, None], i2[None, :], j2[None, :]] - _one_body_pairs(Fo, nocc) / 3
    # <cd||el> dressed: hole ladder, ring, f_ov and T3
    Wv = v[V, V, V, o] + 0.5 * _E('mnel,mncd->cdel', v[o, o, V, o], t2)
    x = _E('dmef,lmce->cdfl', v[V, o, V, V], t2)
    Wv += x - x.transpose(1, 0, 2, 3)
    Wv += _E('me,lmcd->cdel', f[o, V], t2)
    Wv += 0.5 * _E('mnef,mnlfcd->cdel', oovv, t3)
    # <md||kl> dressed
    Wo = v[o, V, o, o].copy()
    x = _E('nmek,lnde->mdkl', v[o, o, V, o], t2)
    Wo += x - x.transpose(0, 1, 3, 2)
    Wo -= 0.5 * _E('dmef,klef->mdkl', v[V, o, V, V], t2)
    Wo -= _E('me,klde->mdkl', f[o, V], t2)
    Wo -= 0.5 * _E('mnef,nklefd->mdkl', oovv, t3)
    return Intermediates(Wq, Wh, Wr, np.ascontiguousarray(Wv[a2, b2]), np.ascontiguousarray(Wo[:, :, i2, j2]),
                         np.ascontiguousarray(Whh[:, :, i2, j2]), np.ascontiguousarray(Wpp[a2, b2]))


def residual_t4(t2, t3, T4, f, v, nocc, batch_bytes=None, only=None):
    '''(R4 packed, the T4 terms of R3 packed, the T4 term of R2 packed) from
    dressed f and v.  ``only``: the pieces to evaluate (tests): 'pp', 'hh',
    'r2', 'ring', 'b1', 'b2', 't3t3h', 't3t3r'.'''
    def want(piece):
        return only is None or piece in only
    batch_bytes = batch_bytes or BATCH_BYTES
    n = v.shape[0]
    nvir = n - nocc
    o, V = slice(0, nocc), slice(nocc, n)
    isz = np.result_type(T4, v).itemsize
    dtype = np.result_type(T4, t2, v)
    nI, nA = T4.shape
    nP, nQ = comb(nocc, 2), comb(nvir, 2)
    nI3, nA3 = comb(nocc, 3), comb(nvir, 3)
    o13, o22, v13, v22 = P.split(nocc, 1, 3), P.split(nocc, 2, 2), P.split(nvir, 1, 3), P.split(nvir, 2, 2)
    i2, j2 = _pairs(nocc)
    a2, b2 = _pairs(nvir)
    i3, a3 = P.tuples(nocc, 3), P.tuples(nvir, 3)
    oovv = v[o, o, V, V]
    oovv_pp = np.ascontiguousarray(oovv[i2, j2][:, a2, b2])             # [(mn),(ef)]
    with timer('q.imds'):
        im = intermediates(t2, t3, f, v, nocc)
    R4 = np.zeros((nI, nA), dtype=dtype)
    R3 = np.zeros((nI3, nA3), dtype=dtype)
    R2 = np.zeros((nP, nQ), dtype=dtype)

    # ---- pp ladder with F_vv: batches of occupied quadruples
    with timer('q.pp'):
        for b in (_batches(nI, batch_bytes // (isz * nQ * nQ)) if want('pp') else ()):
            Tv = P.view(T4, None, v22, jo=b)                            # [I, (ef), (cd)]
            P.reduce(R4, np.matmul(im.Wq[None], Tv), None, v22, jo=b)
    # ---- hh ladder with F_oo, and the T4 term of R2
    with timer('q.hh'):
        for c in (_batches(nA, batch_bytes // (isz * nP * nP)) if want('hh') else ()):
            Th = P.view(T4, o22, None, jv=c)                            # [(mn), (kl), A]
            P.reduce(R4, np.tensordot(im.Wh.T, Th, axes=1), o22, None, jv=c)
    with timer('q.r2'):
        for c in (_batches(nQ, batch_bytes // (isz * nP * nP * nQ)) if want('r2') else ()):
            T22 = P.view(T4, o22, v22, jv=c)                            # [(mn), (ij), (ef), (ab)]
            R2[:, c[0]:c[1]] += _E('PQ,PIQA->IA', oovv_pp, T22)
    # ---- ring, and the f_ov term of R3
    Wr2 = np.ascontiguousarray(im.Wr.transpose(3, 1, 0, 2)).reshape(nocc * nvir, nocc * nvir)   # [(i a),(m e)]
    fov = np.ascontiguousarray(f[o, V])
    with timer('q.ring'):
        step_o = max(1, min(nI3, int((batch_bytes / (isz * nocc * nvir)) ** 0.5)))
        for b in (_batches(nI3, step_o) if want('ring') else ()):
            for c in _batches(nA3, batch_bytes // (isz * nocc * nvir * (b[1] - b[0]))):
                T13 = P.view(T4, o13, v13, jo=b, jv=c)                  # [m, J, e, K]
                nb, nc = b[1] - b[0], c[1] - c[0]
                R3[b[0]:b[1], c[0]:c[1]] += _E('me,mJeK->JK', fov, T13)
                M = np.ascontiguousarray(T13.transpose(0, 2, 1, 3)).reshape(nocc * nvir, nb * nc)
                X = (Wr2 @ M).reshape(nocc, nvir, nb, nc).transpose(0, 2, 1, 3)
                P.reduce(R4, X, o13, v13, jo=b, jv=c)

    # ---- (i, ab) x (jkl, cd): Wv with T3 and T2 with Xo;  the <cm||ef> term of R3
    vovv_pp = np.ascontiguousarray(v[V, o, V, V][:, :, a2, b2])         # [d, m, (ef)]
    oovo = v[o, o, V, o]
    # t3(jkl, cd e) as [e, (jkl), (cd)];  t3(njk, ecd) as [n, e, (jk), (cd)]
    t3_tp = t3[i3[:, 0], i3[:, 1], i3[:, 2]]                            # [(jkl), c, d, e]
    t3v = np.ascontiguousarray(t3_tp[:, a2, b2].transpose(2, 0, 1))     # [e, (jkl), (cd)]
    t3h = np.ascontiguousarray(t3[:, i2, j2][:, :, :, a2, b2])          # [n, (jk), e, (cd)]
    L1 = np.concatenate((-im.Wv.transpose(2, 0, 1), t2[:, :, a2, b2].transpose(0, 2, 1)), axis=2)   # [i, (ab), e + m]
    L1 = np.ascontiguousarray(L1.reshape(nocc * nQ, nvir + nocc))
    # the parts of Xo that do not have T4: [m, (jkl), (cd)]
    with timer('q.xo'):
        G = _E('mnel,nJeQ->mlJQ', oovo, t3h)                            # <mn||el> t3(njk,ecd)
        G += 0.5 * _E('mnJ,nlQ->mlJQ', im.Whh, t2[:, :, a2, b2])
        Xo0 = _antisym_single_pair(G, nocc, 1, 2)                       # P(l/jk)
        H = _E('dmQ,JQc->mJcd', vovv_pp, t3_tp[:, a2, b2])              # sum_{e<f} <dm||ef> t3(jkl,efc)
        Xo0 -= (H - H.transpose(0, 1, 3, 2))[:, :, a2, b2]
        del G, H
    with timer('q.b1'):
        step_o = max(1, min(nI3, int((batch_bytes / (isz * nocc * nQ)) ** 0.5)))
        for b in (_batches(nI3, step_o) if want('b1') else ()):
            for c in _batches(nQ, batch_bytes // (isz * nocc * nQ * (b[1] - b[0]))):
                nb, nc = b[1] - b[0], c[1] - c[0]
                T12 = P.view(T4, o13, v22, jo=b, jv=c)                  # [n, J, (ef), (cd)]
                # R3 += - P(c/ab) sum_{m, e<f} <cm||ef> t[m,(ijk),(ef),(ab)]
                U = -_E('cmQ,mJQA->JcA', vovv_pp, T12)
                P.reduce(R3[b[0]:b[1]], U, None, P.split(nvir, 1, 2), jv=c)
                Xo = Xo0[:, b[0]:b[1], c[0]:c[1]] + _E('mnQ,nJQC->mJC', oovv[:, :, a2, b2], T12)
                Rt = np.concatenate((t3v[:, b[0]:b[1], c[0]:c[1]], Xo), axis=0).reshape(nvir + nocc, nb * nc)
                X = (L1 @ Rt).reshape(nocc, nQ, nb, nc).transpose(0, 2, 1, 3)
                P.reduce(R4, X, o13, v22, jo=b, jv=c)

    # ---- (ij, a) x (kl, bcd): Wo with T3 and T2 with Xv;  the <mn||ek> term of R3
    oovo_pp = np.ascontiguousarray(oovo[i2, j2])                        # [(mn), e, l]
    t3_op = t3[:, :, :, a3[:, 0], a3[:, 1], a3[:, 2]]                   # [m, k, l, (bcd)]
    t3o = np.ascontiguousarray(t3_op[:, i2, j2])                        # [m, (kl), (bcd)]
    t3p = np.ascontiguousarray(t3_op[i2, j2])                           # [(mn), k, (bcd)]
    L2 = np.concatenate((im.Wo.transpose(2, 1, 0), t2[i2, j2]), axis=2)             # [(ij), a, m + e]
    L2 = np.ascontiguousarray(L2.reshape(nP * nvir, nocc + nvir))
    with timer('q.xv'):
        K = _E('Pel,PkA->eklA', oovo_pp, t3p)                           # sum_{m<n} <mn||el> t3(mnk,bcd)
        Xv0 = -(K - K.transpose(0, 2, 1, 3))[:, i2, j2]                 # - P(kl)
        Mx = _E('dmef,mPfQ->ePdQ', v[V, o, V, V], t3h)                  # <dm||ef> t3(mkl,fbc)
        Mx += 0.5 * _E('Qef,Pfd->ePdQ', im.Wpp, t2[i2, j2])
        Xv0 += _antisym_single_pair(Mx, nvir, 2, 3)                     # P(d/bc)
        N = _E('mbek,mlQ->eklbQ', im.Wr, t2[:, :, a2, b2])              # Wr(m,b,e,k) t2(ml,cd)
        N = (N - N.transpose(0, 2, 1, 3, 4))[:, i2, j2]                 # P(kl)
        Xv0 += _antisym_single_pair(N, nvir, 2, 3)                      # P(b/cd)
        del K, Mx, N
    with timer('q.b2'):
        step_o = max(1, min(nP, int((batch_bytes / (isz * nP * nvir)) ** 0.5)))
        for b in (_batches(nP, step_o) if want('b2') else ()):
            for c in _batches(nA3, batch_bytes // (isz * nP * nvir * (b[1] - b[0]))):
                nb, nc = b[1] - b[0], c[1] - c[0]
                T21 = P.view(T4, o22, v13, jo=b, jv=c)                  # [(mn), (kl), f, (bcd)]
                # R3 += - P(k/ij) sum_{m<n, e} <mn||ek> t[(mn),(ij),e,(abc)]
                U = -_E('Pek,PIeA->kIA', oovo_pp, T21)
                P.reduce(R3[:, c[0]:c[1]], U, P.split(nocc, 1, 2), None, jo=b)
                Xv = Xv0[:, b[0]:b[1], c[0]:c[1]] + _E('Pef,PKfA->eKA', oovv[i2, j2], T21)
                Rt = np.concatenate((t3o[:, b[0]:b[1], c[0]:c[1]], Xv), axis=0).reshape(nocc + nvir, nb * nc)
                X = (L2 @ Rt).reshape(nP, nvir, nb, nc).transpose(0, 2, 1, 3)
                P.reduce(R4, X, o22, v13, jo=b, jv=c)

    # ---- T3 with T3 through <oo||vv>
    with timer('q.t3t3'):
        # - sum_{m<n} t3(mni,abc) sum_{e<f} <mn||ef> t3(jkl,efd):  X[i, (jkl), d, (abc)]
        Y = _E('PQ,JQd->PJd', oovv_pp, t3_tp[:, a2, b2])                # [(mn), (jkl), d]
        L3 = np.ascontiguousarray(t3_op[i2, j2].transpose(1, 2, 0)).reshape(nocc, nA3, nP)   # [i, (abc), (mn)]
        Y2 = Y.reshape(nP, nI3 * nvir)
        for c in (_batches(nA3, batch_bytes // (isz * nocc * nI3 * nvir)) if want('t3t3h') else ()):
            X = -(L3[:, c[0]:c[1]].reshape(-1, nP) @ Y2).reshape(nocc, c[1] - c[0], nI3, nvir)
            P.reduce(R4, X.transpose(0, 2, 3, 1), o13, v13, jv=c)
        # 1/2 sum_{nf} Z(nf;ij;ab) t3(nkl,fcd):  X[(ij), (kl), (ab), (cd)]
        Z = _E('mnef,mPeQ->PQnf', oovv, t3h).reshape(nP * nQ, nocc * nvir)
        R = np.ascontiguousarray(t3h.transpose(0, 2, 1, 3)).reshape(nocc * nvir, nP, nQ)   # [(n f), (kl), (cd)]
        step_o = max(1, min(nP, int((batch_bytes / (isz * nP * nQ)) ** 0.5)))
        for b in (_batches(nP, step_o) if want('t3t3r') else ()):
            for c in _batches(nQ, batch_bytes // (isz * nP * nQ * (b[1] - b[0]))):
                nb, nc = b[1] - b[0], c[1] - c[0]
                X = (Z @ R[:, b[0]:b[1], c[0]:c[1]].reshape(nocc * nvir, nb * nc)).reshape(nP, nQ, nb, nc)
                P.reduce(R4, X.transpose(0, 2, 1, 3), o22, v22, jo=b, jv=c, scale=0.5)
    return R4, R3, R2


# ------------------------------------------------------------------ solver

def residuals(t1, t2, t3, T4, fock, g, nocc, batch_bytes=None):
    '''(R1, R2, R3 as full arrays, R4 packed) of CCSDTQ from the bare Fock
    matrix and <pq||rs>.'''
    nmo = g.shape[0]
    nvir = nmo - nocc
    with timer('q.dress'):
        ge, fdr = ccsdt._t1_dress(g, fock, t1, nocc, nmo)
        if not any(np.iscomplexobj(x) for x in (g, fock, t1, t2, t3, T4)):
            ge, fdr = np.ascontiguousarray(ge.real), np.ascontiguousarray(fdr.real)
    arrays = {'v': ge, 'f': fdr, 't2': t2, 't3': t3}
    with timer('q.ccsdt'):
        R1 = Q.residual(1, arrays, nocc, with_t4=False)
        R2 = Q.residual(2, arrays, nocc, with_t4=False)
        R3 = ccsdt._t3_residual(t2, t3, ge, fdr, nocc, nmo)
    R4, d3, d2 = residual_t4(t2, t3, T4, fdr, ge, nocc, batch_bytes)
    R3 += P.unpack(d3, nocc, nvir, 3)
    R2 += P.unpack(d2, nocc, nvir, 2)
    return R1, R2, R3, R4


def denominators(mo_energy, nocc, k):
    '''e_i + .. - e_a - .. on the unique block of rank k.'''
    eo, ev = np.asarray(mo_energy[:nocc]), np.asarray(mo_energy[nocc:])
    occ, vir = P.tuples(nocc, k), P.tuples(len(ev), k)
    return eo[occ].sum(axis=1)[:, None] - ev[vir].sum(axis=1)[None, :]


@dataclass
class Result:
    e_corr: float
    converged: bool
    ncycle: int
    t1: np.ndarray
    t2: np.ndarray              # full
    t3: np.ndarray              # full
    t4: np.ndarray              # packed [C(no,4), C(nv,4)]
    timings: list = None


def kernel(fock, g, mo_energy, nocc, conv_tol=1e-10, conv_tol_normt=1e-8, max_cycle=200, diis_space=8,
           batch_bytes=None, verbose=False, amps=None):
    '''CCSDTQ by Jacobi steps with DIIS from the MP2 amplitudes (or from
    ``amps`` = (t1, t2, t3, T4)).  fock: the Fock matrix of the spin
    orbitals; g = <pq||rs> over all of them; canonical orbitals are not
    assumed, but the steps divide by the differences of ``mo_energy``.'''
    import time
    from zccsdt.cc.ccsdtq_ref import energy
    from zccsdt.cc.diis import DIIS
    nmo = g.shape[0]
    nvir = nmo - nocc
    o, v = slice(0, nocc), slice(nocc, None)
    dtype = np.result_type(fock, g)
    den = [denominators(mo_energy, nocc, k) for k in (1, 2, 3, 4)]
    if amps is None:
        t1 = fock[v, o].T / den[0]
        t2p = P.pack(g[v, v, o, o].transpose(2, 3, 0, 1)) / den[1]
        t3p = np.zeros(den[2].shape, dtype=dtype)
        T4 = np.zeros(den[3].shape, dtype=dtype)
    else:
        t1, t2p, t3p, T4 = amps[0], P.pack(amps[1]), P.pack(amps[2]), amps[3]
    t2, t3 = P.unpack(t2p, nocc, nvir, 2), P.unpack(t3p, nocc, nvir, 3)
    e_old = energy(t1, t2, fock, g, nocc)
    adiis = DIIS(space=diis_space, incore=True)
    conv, timings = False, []
    for cycle in range(max_cycle):
        t0 = time.perf_counter()
        R1, R2, R3, R4 = residuals(t1, t2, t3, T4, fock, g, nocc, batch_bytes)
        new = [t1 + R1 / den[0], t2p + P.pack(R2) / den[1], t3p + P.pack(R3) / den[2], T4 + R4 / den[3]]
        del R1, R2, R3, R4
        normt = sum(np.linalg.norm(a - b) for a, b in zip(new, (t1, t2p, t3p, T4)))
        with timer('q.diis'):
            parts = adiis.update([x.ravel() for x in new])
        del new
        if dtype == np.float64:             # (the DIIS vectors are complex numbers)
            parts = [np.ascontiguousarray(p.real) if np.iscomplexobj(p) else p for p in parts]
        t1, t2p, t3p, T4 = (p.reshape(d.shape) for p, d in zip(parts, den))
        t2, t3 = P.unpack(t2p, nocc, nvir, 2), P.unpack(t3p, nocc, nvir, 3)
        e = energy(t1, t2, fock, g, nocc)
        timings.append(time.perf_counter() - t0)
        if verbose:
            print('cycle %3d  E_corr = %.13f  dE = %9.2e  |dt| = %.2e  %.2f s'
                  % (cycle + 1, e.real, (e - e_old).real, normt, timings[-1]), flush=True)
        done = abs(e - e_old) < conv_tol and normt < conv_tol_normt
        e_old = e
        if done:
            conv = True
            break
    adiis.close()
    return Result(float(np.real(e_old)), conv, cycle + 1, t1, t2, t3, T4, timings)
