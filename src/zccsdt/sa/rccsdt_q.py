'''CCSDT(Q) for closed shells on spatial orbitals (spin-adapted, block T3).

With the spatial (NOSA) quantities of zccsdt.sa.derive.q_spatial,

  E(Q) = 1/24 sum_{ijkl,abcd} z_ijkl^abcd / D sum_tau w(tau) r_ijkl^{tau(abcd)},
  r = A + B,  z = A + C  (z = C alone gives E[Q]),  w(tau) = sgn(tau) 2^cycles,

for the four pairs P_n = (o_n, x_n) of an occupied quadruple and its virtual
slots, the parts are sums over arrangements (p, q, r, s) of the pairs:

  A:  + Xp(p,q,r;s) = sum_e t_{o_p o_q o_r}^{x_p x_q e} <x_r x_s|e o_s>     (12)
      - Xh(p,q;r,s) = sum_m t_{o_p o_q m}^{x_p x_q x_r} <m x_s|o_r o_s>     (12)
  B:  + Yv(p,q|r,s) = sum_f I_{pq}[x_q,x_p,x_r;f] t_{o_r o_s}^{f x_s}       (12)
        I_{pq}[x_q,x_p,y;f] = sum_e t_{o_p o_q}^{e x_q} <x_p y|e f>
      + sum_m t_{o_p m}^{x_p x_q} G(o_q,o_r,o_s)[m;x_r,x_s]                 (24)
        G = 1/2 Go - G1 - G2 (oooo and the two ring integrals with one T2)
  C:  t_{o_p o_q}^{x_p x_q} <o_r o_s|x_r x_s>                               (6)

r and z are invariant under simultaneous permutations of the pairs, so only
sorted occupied quadruples are needed; quadruples with an index three times
do not contribute.  Bare integrals, canonical orbitals; T1 does not enter.
'''

from itertools import permutations

import numpy as np

from zccsdt.blas import gemm
from zccsdt.sa import derive


def unique_quadruples(nocc):
    '''[(i, j, k, l), number of distinct orderings] for i <= j <= k <= l with
    no index more than twice.'''
    out = []
    for i in range(nocc):
        for j in range(i, nocc):
            for k in range(j, nocc):
                for l in range(k, nocc):
                    q = (i, j, k, l)
                    cnt = [q.count(x) for x in set(q)]
                    if max(cnt) > 2:
                        continue
                    out.append((q, 24 // (2 ** cnt.count(2))))
    return out


class Context:
    '''Integral blocks and T2 intermediates shared by all quadruples.'''

    def __init__(self, ints, blocks, t2, t3b):
        self.no, self.nv = no, v = ints.nocc, ints.nvir
        self.blocks, self.t2, self.t3b = blocks, t2, t3b
        self.eo, self.ev = ints.mo_energy[:no], ints.mo_energy[no:]
        g = ints.block
        self.vvvo = g('vvvo')                       # <cd|el>
        self.ovoo = g('ovoo')                       # <md|kl>
        self.oovv = g('oovv')                       # <kl|cd>
        # (e x|y f) = <e y|x f>: Wc[e, x, y, f], contraction index e leading
        self.Wc = np.ascontiguousarray(g('vvvv').transpose(0, 2, 1, 3))
        # G[q,r,s][m; c, d] = 1/2 Go - G1 - G2
        G = 0.5 * np.einsum('mnqs,rncd->qrsmcd', g('oooo'), t2, optimize=True)
        G -= np.einsum('mdes,qrec->qrsmcd', g('ovvo'), t2, optimize=True)
        G -= np.einsum('mcqe,rsed->qrsmcd', g('ovov'), t2, optimize=True)
        self.G = G

    def cube(self, p, q, r):
        return self.blocks.cube(self.t3b, p, q, r)

    def I(self, op, oq):
        '''I[x_q, x_p, y, f] = sum_e t_{op oq}^{e x_q} <x_p y|e f>.'''
        v = self.nv
        return gemm(np.ascontiguousarray(self.t2[op, oq].T), self.Wc.reshape(v, -1)).reshape((v,) * 4)


def _put(out, x, order):
    '''out[x_0,x_1,x_2,x_3] += x, where axis n of x carries pair order[n].'''
    out += x.transpose(np.argsort(order))


def quad_arrays(ctx, quad):
    '''(A, B, C) as [x_0, x_1, x_2, x_3] arrays for an ordered occupied
    quadruple (straightforward version: one product and one transposed add
    per term).'''
    v = ctx.nv
    o = quad
    A, B, C = (np.zeros((v,) * 4) for _ in range(3))
    for p, q, r, s in permutations(range(4)):
        if p < q:
            # Xp and Xh: unordered {p,q} on T3
            x = gemm(np.ascontiguousarray(ctx.cube(o[p], o[q], o[r])).reshape(v * v, v),
                     np.ascontiguousarray(ctx.vvvo[:, :, :, o[s]].transpose(2, 0, 1)).reshape(v, -1))
            _put(A, x.reshape((v,) * 4), (p, q, r, s))
            T = np.stack([ctx.cube(o[p], o[q], m) for m in range(ctx.no)], axis=-1)   # [xp,xq,xr,m]
            x = gemm(T.reshape(-1, ctx.no), np.ascontiguousarray(ctx.ovoo[:, :, o[r], o[s]]))
            _put(A, -x.reshape((v,) * 4), (p, q, r, s))
            if r < s:
                x = ctx.t2[o[p], o[q]][:, :, None, None] * ctx.oovv[o[r], o[s]][None, None]
                _put(C, x, (p, q, r, s))
        if 0 in (p, q):
            # Yv: the pair of pairs containing pair 0 first
            x = gemm(ctx.I(o[p], o[q]).reshape(-1, v), np.ascontiguousarray(ctx.t2[o[r], o[s]]))
            _put(B, x.reshape((v,) * 4), (q, p, r, s))
        x = gemm(np.ascontiguousarray(ctx.t2[o[p]].transpose(1, 2, 0)).reshape(v * v, -1),
                 ctx.G[o[q], o[r], o[s]].reshape(ctx.no, -1))
        _put(B, x.reshape((v,) * 4), (p, q, r, s))
    return A, B, C


def _energy_dense(ctx, quad, A, B, C):
    '''(sum C M r / D, sum A M r / D) for one quadruple, numpy.'''
    ev = ctx.ev
    d = sum(ctx.eo[x] for x in quad) - (ev[:, None, None, None] + ev[None, :, None, None]
                                        + ev[None, None, :, None] + ev[None, None, None, :])
    r = A + B
    rt = sum(w * r.transpose(tau) for tau, w in derive.q_metric()) / d
    return float(np.vdot(C, rt)), float(np.vdot(A, rt))


def q_correction_simple(ints, blocks, t2, t3b):
    '''(E[Q], E(Q)) with one quadruple at a time in numpy (reference for the
    block algorithm; memory 10 v^4).'''
    ctx = Context(ints, blocks, t2, t3b)
    e5 = e6 = 0.0
    for quad, mult in unique_quadruples(ctx.no):
        A, B, C = quad_arrays(ctx, quad)
        c, a = _energy_dense(ctx, quad, A, B, C)
        e5 += mult / 24 * c
        e6 += mult / 24 * a
    return e5, e5 + e6


# ------------------------------------------------------------ block algorithm

# Accumulator layouts: three cyclic rotations (w, x, y) of the pairs 0, 1, 2
# with pair 3 last; accumulator c holds its terms as [x_w, x_x, x_y, x_3].
ROT = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
# pair sets of the T2 factor of C and their complements
PAIRS = ((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2), (1, 2, 0, 3), (1, 3, 0, 2), (2, 3, 0, 1))


def quad_energy(S, tc, gc, eocc, ev):
    '''(sum c M r / D, sum a M r / D) of one quadruple from its accumulators
    S = [A_0, A_1, A_2, B_0, B_1, B_2] (cyclic layouts ROT), with
    a = sum_c A_c, r = a + sum_c B_c and c = sum_k tc[k] x gc[k] (PAIRS).'''
    from zccsdt.cc import t3pack
    lib = t3pack._lib
    v = len(ev)
    if t3pack.USE_C and hasattr(lib, 'zc_sa_q_energy_d'):
        ptr = (t3pack.ctypes.c_void_p * 6)(*[x.ctypes.data for x in S])
        out = np.zeros(2)
        tcc, gcc, evc = (np.ascontiguousarray(x, dtype=np.float64) for x in (tc, gc, ev))
        lib.zc_sa_q_energy_d(t3pack._I64(v), ptr, t3pack._p(tcc), t3pack._p(gcc),
                             t3pack.ctypes.c_double(float(eocc)), t3pack._p(evc), t3pack._p(out))
        return float(out[0]), float(out[1])
    a, b = (sum(X[c].reshape((v,) * 4).transpose(np.argsort(ROT[c] + (3,))) for c in range(3))
            for X in (S[:3], S[3:]))
    cc = np.zeros((v,) * 4)
    for k, (p, q, r, s) in enumerate(PAIRS):
        _put(cc, tc[k][:, :, None, None] * gc[k][None, None], (p, q, r, s))
    d = eocc - (ev[:, None, None, None] + ev[None, :, None, None]
                + ev[None, None, :, None] + ev[None, None, None, :])
    rt = sum(w * (a + b).transpose(tau) for tau, w in derive.q_metric()) / d
    return float(np.vdot(cc, rt)), float(np.vdot(a, rt))


def q_correction(ints, blocks, t2, t3b):
    '''(E[Q], E(Q)) from converged spatial t2 and T3 blocks.

    One sorted occupied quadruple (i <= j <= k <= l) at a time.  Its terms
    are GEMMs whose outputs are stored in the cyclic layouts ROT directly, so
    nothing is transposed: for rotation (w, x, y)

      A_c: Xp as one [(x_w x_x), 4v] x [4v, (x_y x_3)] product (T3 cubes and
           <vv|vo> stacked along the contraction index); Xh with special pair
           w as [x_w, 3o] x [3o, (x_x x_y x_3)] and, for c = 0, with special
           pair 3 as [(x_0 x_1 x_2), 3o] x [3o, x_3];
      B_c: Ibar_c x T2 (pair 3 last) and T2 x Ibar_{c+1} (pair w first), with
           Ibar_c[x_w, x_x, y; f] = I_{wx} + I_{xw} kept for the pairs (i,j),
           (j,k), (k,i) while l runs; the T2 x G terms as one
           [(x_w x_x), 4o] x [4o, (x_y x_3)] product.

    The energy of the quadruple is one tiled pass over the six accumulators
    (quad_energy).  Memory: 11 v^4.'''
    from zccsdt.cc.t3pack import axpby
    from zccsdt.sa.t3blocks import gather_boxes
    from zccsdt.timing import timer
    ctx = Context(ints, blocks, t2, t3b)
    no, v = ctx.no, ctx.nv
    vv, v3 = v * v, v ** 3
    eo, ev = ctx.eo, ctx.ev
    VO = np.ascontiguousarray(ctx.vvvo.transpose(3, 0, 1, 2))      # [o][c,d,e] = <cd|eo>
    ovoo, oovv = ctx.ovoo, ctx.oovv
    Gh = ctx.G + ctx.G.transpose(0, 2, 1, 3, 5, 4)      # G[q,r,s][m,xr,xs] + G[q,s,r][m,xs,xr]
    Wc = ctx.Wc.reshape(v, v3)
    S = [np.empty((v,) * 4) for _ in range(6)]
    Ibar = [np.empty((v,) * 4) for _ in range(3)]
    tmp = np.empty((v,) * 4)
    As, Bs = np.empty((v, v, 4, v)), np.empty((4, v, v, v))
    As2, Bs2 = np.empty((v, v, 4, no)), np.empty((4, no, v, v))
    Tg = np.empty((3 * no, v, v, v))
    occ = np.arange(no)

    def cube(a, b, c):
        return blocks.cube(t3b, a, b, c)

    def build_I(dst, oa, ob):
        # Ibar[x_a, x_b, y, f] = I_ba[x_a | x_b, y, f] + I_ab[x_b | x_a, y, f]
        gemm(t2[ob, oa].T, Wc, dst.reshape(v, v3))
        gemm(t2[oa, ob].T, Wc, tmp.reshape(v, v3))
        axpby(dst, tmp.transpose(1, 0, 2, 3), 1.0, 1.0)

    def gather(trips):
        views = [blocks.view(a, b, c, v) for a, b, c in trips]
        gather_boxes(t3b, [x[0] for x in views], [x[1] for x in views], Tg)

    e5 = e6 = 0.0
    for i in range(no):
        for j in range(i, no):
            with timer('q.I'):
                build_I(Ibar[0], i, j)
            for k in range(j, no):
                if i == j == k:
                    continue
                with timer('q.I'):
                    build_I(Ibar[1], j, k)
                    build_I(Ibar[2], k, i)
                for l in range(k, no):
                    if j == k == l:
                        continue
                    o = (i, j, k, l)
                    mult = 24 // 2 ** sum(o[n] == o[n + 1] for n in range(3))
                    o3 = l
                    for c, (w, x, y) in enumerate(ROT):
                        ow, ox, oy = o[w], o[x], o[y]
                        # ---- A_c: Xp
                        with timer('q.stack'):
                            axpby(As[:, :, 0], cube(ow, ox, oy), 1.0, 0.0)
                            axpby(As[:, :, 1], cube(ow, ox, o3), 1.0, 0.0)
                            axpby(As[:, :, 2], VO[ox], 1.0, 0.0)
                            axpby(As[:, :, 3], VO[ow].transpose(1, 0, 2), 1.0, 0.0)
                            axpby(Bs[0], VO[o3].transpose(2, 0, 1), 1.0, 0.0)
                            axpby(Bs[1], VO[oy].transpose(2, 1, 0), 1.0, 0.0)
                            axpby(Bs[2], cube(ow, oy, o3), 1.0, 0.0)
                            axpby(Bs[3], cube(ox, oy, o3), 1.0, 0.0)
                        with timer('q.Xp'):
                            gemm(As.reshape(vv, 4 * v), Bs.reshape(4 * v, vv), S[c].reshape(vv, vv))
                        # ---- A_c: Xh, special pair w, m in the slot of x, y or 3
                        with timer('q.gather'):
                            gather([(m, oy, o3) for m in occ] + [(ox, m, o3) for m in occ]
                                   + [(ox, oy, m) for m in occ])
                        with timer('q.Xh'):
                            sm = np.hstack([-ovoo[:, :, r, ow].T for r in (ox, oy, o3)])
                            gemm(sm, Tg.reshape(3 * no, v3), S[c].reshape(v, v3), beta=1.0)
                        if c == 0:
                            # special pair 3, m in the slot of 0, 1 or 2
                            with timer('q.gather'):
                                gather([(m, o[1], o[2]) for m in occ] + [(o[0], m, o[2]) for m in occ]
                                       + [(o[0], o[1], m) for m in occ])
                            with timer('q.Xh'):
                                sm = np.vstack([-ovoo[:, :, r, o3] for r in o[:3]])
                                gemm(Tg.reshape(3 * no, v3).T, sm, S[0].reshape(v3, v), beta=1.0)
                        # ---- B_c: Yv and the T2 x G terms
                        with timer('q.Yv'):
                            gemm(Ibar[c].reshape(v3, v), t2[oy, o3], S[3 + c].reshape(v3, v))
                            gemm(t2[o3, ow].T, Ibar[(c + 1) % 3].reshape(v3, v).T,
                                 S[3 + c].reshape(v, v3), beta=1.0)
                        with timer('q.G'):
                            As2[:, :, 0] = t2[ow].transpose(1, 2, 0)
                            As2[:, :, 1] = t2[ox].transpose(2, 1, 0)
                            As2[:, :, 2] = Gh[o3, ow, ox].transpose(1, 2, 0)
                            As2[:, :, 3] = Gh[oy, ow, ox].transpose(1, 2, 0)
                            Bs2[0], Bs2[1] = Gh[ox, oy, o3], Gh[ow, oy, o3]
                            Bs2[2], Bs2[3] = t2[oy], t2[o3].transpose(0, 2, 1)
                            gemm(As2.reshape(vv, 4 * no), Bs2.reshape(4 * no, vv),
                                 S[3 + c].reshape(vv, vv), beta=1.0)
                    with timer('q.energy'):
                        tc = np.array([t2[o[p], o[q]] for p, q, r, s in PAIRS])
                        gc = np.array([oovv[o[r], o[s]] for p, q, r, s in PAIRS])
                        c5, a6 = quad_energy(S, tc, gc, sum(eo[n] for n in o), ev)
                    e5 += mult / 24 * c5
                    e6 += mult / 24 * a6
    return e5, e5 + e6
