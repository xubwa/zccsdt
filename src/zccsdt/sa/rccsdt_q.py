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
        G = 1/2 Go - G1 - G2 (oooo and the two ring integrals with one T2;
        formed per quadruple, o v^3 work)
  C:  t_{o_p o_q}^{x_p x_q} <o_r o_s|x_r x_s>                               (6)

r and z are invariant under simultaneous permutations of the pairs, so only
sorted occupied quadruples are needed; quadruples with an index three times
do not contribute.  Bare integrals, canonical orbitals; T1 does not enter.
'''

from itertools import permutations

import numpy as np

from zccsdt.blas import gemm, gemm_slices
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

    def __init__(self, ints, blocks, t2, t3b, wc=True):
        self.no, self.nv = no, v = ints.nocc, ints.nvir
        self.blocks, self.t2, self.t3b = blocks, t2, t3b
        self.eo, self.ev = ints.mo_energy[:no], ints.mo_energy[no:]
        g = ints.block
        self.vvvo = g('vvvo')                       # <cd|el>
        self.ovoo = g('ovoo')                       # <md|kl>
        self.oovv = g('oovv')                       # <kl|cd>
        # (e x|y f) = <e y|x f>: Wc[e, x, y, f], contraction index e leading
        # (the sliced algorithm reads rows of the packed integrals instead)
        if not wc:
            self.Wc = None
        elif ints.V is not None:
            self.Wc = np.ascontiguousarray(ints.V[no:, no:, no:, no:].transpose(0, 2, 1, 3))
        else:
            from pyscf import ao2mo
            Cv = np.ascontiguousarray(ints.mo_coeff[:, no:])
            self.Wc = np.ascontiguousarray(
                ao2mo.general(ints.mol, (Cv, Cv, Cv, Cv), compact=False).reshape((v,) * 4))
        # integrals of G with the summed index last / first
        self._oooo = np.ascontiguousarray(g('oooo').transpose(2, 3, 0, 1))      # [q,s][m,n]
        self._ovvo = np.ascontiguousarray(g('ovvo').transpose(3, 0, 1, 2))      # [s][m,d,e]
        self._ovov = np.ascontiguousarray(g('ovov').transpose(2, 0, 1, 3))      # [q][m,c,e]

    def G(self, q, r, s):
        '''G[m; c, d] = 1/2 Go - G1 - G2 for the occupied indices (q, r, s):
        Go = sum_n <mn|qs> t_rn^cd, G1 = sum_e <md|es> t_qr^ec,
        G2 = sum_e <mc|qe> t_rs^ed (an o^4 v^2 array if stored for all).

        The small products go through zccsdt.blas.gemm as well: calls into
        numpy's own BLAS between the threaded GEMMs of the quadruple loop
        (np.matmul is 2-3 times faster for these sizes alone) made those
        twice as slow (the two libraries share the OpenMP thread pool).'''
        no, v, t2 = self.no, self.nv, self.t2
        out = gemm(self._oooo[q, s], t2[r].reshape(no, -1), alpha=0.5).reshape(no, v, v)
        g1 = gemm(self._ovvo[s].reshape(no * v, v), t2[q, r]).reshape(no, v, v)      # [m,d,c]
        out -= g1.transpose(0, 2, 1)
        gemm(self._ovov[q].reshape(no * v, v), t2[r, s], out.reshape(no * v, v), alpha=-1.0, beta=1.0)
        return out

    def Gh(self, q, r, s):
        '''G(q,r,s)[m; x_r, x_s] + G(q,s,r)[m; x_s, x_r].'''
        return self.G(q, r, s) + self.G(q, s, r).transpose(0, 2, 1)

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
                 ctx.G(o[q], o[r], o[s]).reshape(ctx.no, -1))
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


def quad_energy(S, tc, gc, eocc, ev, mirror=False):
    '''(sum c M r / D, sum a M r / D) of one quadruple from its accumulators
    S = [A_0, A_1, A_2, B_0, B_1, B_2] (cyclic layouts ROT), with
    a = sum_c A_c, r = a + sum_c B_c and c = sum_k tc[k] x gc[k] (PAIRS).
    ``mirror``: a and r are the accumulated arrays plus their images under
    the exchange of the pairs 0 and 1 (x_0 <-> x_1).'''
    from zccsdt.cc import t3pack
    lib = t3pack._lib
    v = len(ev)
    if t3pack.USE_C and hasattr(lib, 'zc_sa_q_energy_d'):
        ptr = (t3pack.ctypes.c_void_p * 6)(*[x.ctypes.data for x in S])
        out = np.zeros(2)
        tcc, gcc, evc = (np.ascontiguousarray(x, dtype=np.float64) for x in (tc, gc, ev))
        lib.zc_sa_q_energy_d(t3pack._I64(v), ptr, t3pack._p(tcc), t3pack._p(gcc),
                             t3pack.ctypes.c_double(float(eocc)), t3pack._p(evc),
                             t3pack.ctypes.c_int(int(mirror)), t3pack._p(out))
        return float(out[0]), float(out[1])
    a, b = (sum(X[c].reshape((v,) * 4).transpose(np.argsort(ROT[c] + (3,))) for c in range(3))
            for X in (S[:3], S[3:]))
    if mirror:
        a, b = a + a.transpose(1, 0, 2, 3), b + b.transpose(1, 0, 2, 3)
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
    (quad_energy).  Quadruples with i = j or j = k use the exchange symmetry
    of the two equal pairs (fill_sym): about 60% of the v^5 work.
    Memory: 10 v^4.'''
    from zccsdt.cc.t3pack import axpby
    from zccsdt.sa.t3blocks import gather_boxes
    from zccsdt.timing import timer
    ctx = Context(ints, blocks, t2, t3b)
    no, v = ctx.no, ctx.nv
    vv, v3 = v * v, v ** 3
    eo, ev = ctx.eo, ctx.ev
    VO = np.ascontiguousarray(ctx.vvvo.transpose(3, 0, 1, 2))      # [o][c,d,e] = <cd|eo>
    ovoo, oovv = ctx.ovoo, ctx.oovv
    Wc = ctx.Wc.reshape(v, v3)
    S = [np.empty((v,) * 4) for _ in range(6)]
    Ibar = [np.empty((v,) * 4) for _ in range(3)]
    As, Bs = np.empty((v, v, 4, v)), np.empty((4, v, v, v))
    As2, Bs2 = np.empty((v, v, 4, no)), np.empty((4, no, v, v))
    Tg = np.empty((3 * no, v, v, v))
    occ = np.arange(no)

    def cube(a, b, c):
        return blocks.cube(t3b, a, b, c)

    def build_I(dst, oa, ob):
        # Ibar[x_a, x_b, y, f] = sum_e t_ab^{x_a e} (e x_b|y f) + t_ab^{e x_b} (x_a e|y f):
        # one product over all x_a, and one per x_a into its slice (no
        # transposed copy; (e x|y f) = (x e|y f))
        tab = np.ascontiguousarray(t2[oa, ob])
        gemm(tab, Wc, dst.reshape(v, v3))
        tT = np.ascontiguousarray(tab.T)
        d3, w3 = dst.reshape(v, v, vv), ctx.Wc.reshape(v, v, vv)
        for a in range(v):
            gemm(tT, w3[a], d3[a], beta=1.0)

    def gather(trips):
        views = [blocks.view(a, b, c, v) for a, b, c in trips]
        gather_boxes(t3b, [x[0] for x in views], [x[1] for x in views], Tg[:len(trips)])
        return Tg[:len(trips)].reshape(len(trips), v3)

    def xp(dst, ow, ox, oy, o3, sym=False):
        # T3 x <vv|vo> for the partition {w x | y 3}.  sym (o_w = o_x): the two
        # terms with T3 on the rows at half weight, one of the two others
        h, nb = (0.5, 3) if sym else (1.0, 4)
        with timer('q.stack'):
            axpby(As[:, :, 0], cube(ow, ox, oy), h, 0.0)
            axpby(As[:, :, 1], cube(ow, ox, o3), h, 0.0)
            axpby(As[:, :, 2], VO[ox], 1.0, 0.0)
            axpby(Bs[0], VO[o3].transpose(2, 0, 1), 1.0, 0.0)
            axpby(Bs[1], VO[oy].transpose(2, 1, 0), 1.0, 0.0)
            axpby(Bs[2], cube(ow, oy, o3), 1.0, 0.0)
            if not sym:
                axpby(As[:, :, 3], VO[ow].transpose(1, 0, 2), 1.0, 0.0)
                axpby(Bs[3], cube(ox, oy, o3), 1.0, 0.0)
        with timer('q.Xp'):
            gemm(As.reshape(vv, 4 * v)[:, :nb * v], Bs.reshape(4 * v, vv)[:nb * v],
                 dst.reshape(vv, vv))

    def gterms(dst, ow, ox, oy, o3, sym=False):
        # T2 x G for the ordered pairs (w,x), (x,w), (y,3), (3,y) on T2.  sym:
        # (w,x) once, the two others at half weight
        with timer('q.G'):
            As2[:, :, 0] = t2[ow].transpose(1, 2, 0)
            Bs2[0] = ctx.Gh(ox, oy, o3)
            if sym:
                As2[:, :, 1] = 0.5 * ctx.Gh(o3, ow, ox).transpose(1, 2, 0)
                As2[:, :, 2] = 0.5 * ctx.Gh(oy, ow, ox).transpose(1, 2, 0)
                Bs2[1], Bs2[2] = t2[oy], t2[o3].transpose(0, 2, 1)
            else:
                As2[:, :, 1] = t2[ox].transpose(2, 1, 0)
                As2[:, :, 2] = ctx.Gh(o3, ow, ox).transpose(1, 2, 0)
                As2[:, :, 3] = ctx.Gh(oy, ow, ox).transpose(1, 2, 0)
                Bs2[1] = ctx.Gh(ow, oy, o3)
                Bs2[2], Bs2[3] = t2[oy], t2[o3].transpose(0, 2, 1)
            nb = 3 if sym else 4
            gemm(As2.reshape(vv, 4 * no)[:, :nb * no], Bs2.reshape(4 * no, vv)[:nb * no],
                 dst.reshape(vv, vv), beta=1.0)

    def xh_first(dst, ow, terms, beta):
        # Xh with special pair w first: dst[x_w; (x_x x_y x_3)] = - sum over
        # terms (weight, o_r, triples of the gathered cubes)
        with timer('q.gather'):
            Tm = gather([t for _, _, trips in terms for t in trips])
        with timer('q.Xh'):
            sm = np.hstack([-wgt * ovoo[:, :, r, ow].T for wgt, r, _ in terms])
            gemm(sm, Tm, dst.reshape(v, v3), beta=beta)

    def xh_last(dst, o3, terms):
        # Xh with special pair 3 last: dst[(x_0 x_1 x_2); x_3] +=
        with timer('q.gather'):
            Tm = gather([t for _, _, trips in terms for t in trips])
        with timer('q.Xh'):
            sm = np.vstack([-wgt * ovoo[:, :, r, o3] for wgt, r, _ in terms])
            gemm(Tm.T, sm, dst.reshape(v3, v), beta=1.0)

    def fill_general(o):
        o3 = o[3]
        for c, (w, x, y) in enumerate(ROT):
            ow, ox, oy = o[w], o[x], o[y]
            xp(S[c], ow, ox, oy, o3)
            # Xh, special pair w: m in the slot of x, y or 3
            xh_first(S[c], ow, [(1.0, ox, [(m, oy, o3) for m in occ]),
                                (1.0, oy, [(ox, m, o3) for m in occ]),
                                (1.0, o3, [(ox, oy, m) for m in occ])], 1.0)
            if c == 0:
                xh_last(S[0], o3, [(1.0, o[0], [(m, o[1], o[2]) for m in occ]),
                                   (1.0, o[1], [(o[0], m, o[2]) for m in occ]),
                                   (1.0, o[2], [(o[0], o[1], m) for m in occ])])
            with timer('q.Yv'):
                gemm(Ibar[c].reshape(v3, v), t2[oy, o3], S[3 + c].reshape(v3, v))
                gemm(t2[o3, ow].T, Ibar[(c + 1) % 3].reshape(v3, v).T,
                     S[3 + c].reshape(v, v3), beta=1.0)
            gterms(S[3 + c], ow, ox, oy, o3)

    def fill_sym(o, Ia, Ib):
        # o_0 = o_1: r and z are invariant under the exchange of the pairs 0
        # and 1, which maps every term to a term.  Only one term of each
        # mirror pair is computed (terms that are their own mirror image at
        # half weight); the kernel adds the mirror image.  Ia = Ibar for the
        # pairs (0,1), Ib for (1,2); rotation 2 is the mirror of rotation 1.
        o0, o1, o2, o3 = o
        xp(S[0], o0, o1, o2, o3, sym=True)
        xh_first(S[0], o0, [(1.0, o1, [(m, o2, o3) for m in occ]),
                            (1.0, o2, [(o1, m, o3) for m in occ]),
                            (1.0, o3, [(o1, o2, m) for m in occ])], 1.0)
        xh_last(S[0], o3, [(1.0, o0, [(m, o1, o2) for m in occ]),
                           (0.5, o2, [(o0, o1, m) for m in occ])])
        xp(S[1], o1, o2, o0, o3)
        # special pair 2 (layout [x_2, x_0, x_1, x_3]): m in the slot of 0 or 3
        xh_first(S[2], o2, [(1.0, o0, [(m, o1, o3) for m in occ]),
                            (0.5, o3, [(o0, o1, m) for m in occ])], 0.0)
        with timer('q.Yv'):
            gemm(Ia.reshape(v3, v), t2[o2, o3], S[3].reshape(v3, v), alpha=0.5)
            gemm(t2[o3, o0].T, Ib.reshape(v3, v).T, S[3].reshape(v, v3), beta=1.0)
            gemm(Ib.reshape(v3, v), t2[o0, o3], S[4].reshape(v3, v))
            gemm(t2[o3, o2].T, Ia.reshape(v3, v).T, S[5].reshape(v, v3), alpha=0.5)
        gterms(S[3], o0, o1, o2, o3, sym=True)
        gterms(S[4], o1, o2, o0, o3)

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
                    # a repeated index among the first three pairs: relabel the
                    # pairs so that it is on the pairs 0 and 1
                    if i == j:
                        mirror = True
                        fill_sym(o, Ibar[0], Ibar[1])
                    elif j == k:
                        mirror = True
                        o = (j, k, i, l)
                        fill_sym(o, Ibar[1], Ibar[2])
                    else:
                        mirror = False
                        fill_general(o)
                    with timer('q.energy'):
                        tc = np.array([t2[o[p], o[q]] for p, q, r, s in PAIRS])
                        gc = np.array([oovv[o[r], o[s]] for p, q, r, s in PAIRS])
                        c5, a6 = quad_energy(S, tc, gc, sum(eo[n] for n in o), ev, mirror)
                    e5 += mult / 24 * c5
                    e6 += mult / 24 * a6
    return e5, e5 + e6


# ------------------------------------------------------- sliced algorithm

QT = 6          # tile size of the energy kernels (csrc/t3pack.c)


def slab_tiles(v, tiles=1):
    '''[(t0, t1)]: slabs of ``tiles`` kernel tiles of the largest virtual
    index (the indices t0 QT .. min(v, t1 QT) - 1).'''
    nt = (v + QT - 1) // QT
    return [(t, min(t + tiles, nt)) for t in range(0, nt, tiles)]


def q_correction_slabs(ints, blocks, t2, t3b, tiles=1, erows=None):
    '''(E[Q], E(Q)) with the v^4 work arrays cut into slabs.

    The virtual quadruples are grouped by the slab [d0, d1) that holds their
    largest index.  For those, r and a are needed where one index (of pair
    p, say) is in the slab and the other three are below n = d1: the *face*
    p of the occupied quadruple, six arrays (n, n, n, d1 - d0) per face
    (parts A and B in the three cyclic layouts of the other pairs, the slab
    index last).  Face p is what the block algorithm computes for the
    quadruple with pair p moved to the end, restricted to those ranges, so
    every element of r is still computed once.  The energy kernel reads
    the 24 permuted positions of an element from the four faces
    (zc_sa_q_energy_faces).

    Only the Yv terms need <vv|vv>.  For every slab and every ordered
    occupied pair (a, b) two intermediates are formed from the rows
    E[(y f), tri(e x)] = (y f|e x) of the integrals (rccsdt.vvvv_rows; a
    mapped scratch array or ``erows``), packed over e >= x:

      N[y, ab, d, (e x)] = sum_f (y f|e x) t_ab^{f d}     y < n, d in the slab
      M[d, ab, w, (e x)] = sum_f (d f|e x) t_ab^{f w}     d in the slab, w < n

    (one pass over the rows y < n, resp. the rows of the slab).  A face then
    takes, per layout [x_w, x_x, x_y, d] and with 3 the slab pair,

      sum_e t_wx^{x_w e} N_y3[e, x_x, x_y, d] + (w <-> x)      (Yv(w,x|y,3), Yv(x,w|y,3))
      sum_e t_xy^{x_x e} M_3w[e, x_y, x_w, d] + (x <-> y)      (Yv(x,y|3,w), Yv(y,x|3,w))

    the second into the layout [x_x, x_y, x_w, d].

    Memory: 24 n^3 nd (faces) + 2 o^2 n nd v^2 / 2 (N and M) + 2 v n^2 nd
    elements with nd = 6 ``tiles``, instead of 10 v^4.'''
    from zccsdt.cc import t3pack
    from zccsdt.cc.t3pack import axpby
    from zccsdt.sa.rccsdt import vvvv_rows
    from zccsdt.sa.t3blocks import gather_boxes
    from zccsdt.timing import timer
    lib = t3pack._lib
    I64, P = t3pack._I64, t3pack._p
    ctx = Context(ints, blocks, t2, t3b, wc=False)
    no, v = ctx.no, ctx.nv
    ps = v * (v + 1) // 2
    eo, ev = ctx.eo, np.ascontiguousarray(ctx.ev, dtype=np.float64)
    VO = np.ascontiguousarray(ctx.vvvo.transpose(3, 0, 1, 2))      # [o][c,d,e] = <cd|eo>
    ovoo, oovv = ctx.ovoo, ctx.oovv
    E = vvvv_rows(ints) if erows is None else erows
    occ = np.arange(no)
    npair = no * no
    t2p = t2.reshape(npair, v, v)
    nmax = QT * tiles
    pool = {}

    def buf(name, *shape):
        # work array: a prefix of a flat buffer of the largest size needed
        size = int(np.prod(shape, dtype=np.int64))
        big = pool.get(name)
        if big is None or big.size < size:
            big = pool[name] = np.empty(size)
        return big[:size].reshape(shape)

    gh = {}

    def Gh(q, r, s):
        # G(q,r,s) + G(q,s,r)^T does not depend on the slab: kept (o^4 v^2 at most)
        x = gh.get((q, r, s))
        if x is None:
            if len(gh) * no * v * v * 8 > 1 << 30:
                gh.clear()
            x = gh[q, r, s] = ctx.Gh(q, r, s)
        return x

    def cube(a, b, c):
        return blocks.cube(t3b, a, b, c)

    e5 = e6 = 0.0
    for t0, t1 in slab_tiles(v, tiles):
        d0, n = t0 * QT, min(v, t1 * QT)
        nd = n - d0
        nn, n3 = n * n, n * n * n
        with timer('q.NM'):
            # pair intermediates of the slab, rows packed over e >= x
            TN = np.ascontiguousarray(t2p[:, :, d0:n].transpose(0, 2, 1)).reshape(npair * nd, v)
            TM = np.ascontiguousarray(t2p[:, :, :n].transpose(0, 2, 1)).reshape(npair * n, v)
            N = buf('N', n, npair, nd, ps)
            M = buf('M', nd, npair, n, ps)
            for y in range(n):
                gemm(TN, E[y * v:(y + 1) * v], N[y].reshape(npair * nd, ps))
            for d in range(nd):
                gemm(TM, E[(d0 + d) * v:(d0 + d + 1) * v], M[d].reshape(npair * n, ps))
        As, Bs = buf('As', n, n, 4, v), buf('Bs', 4, v, n, nd)
        As2, Bs2 = buf('As2', n, n, 4, no), buf('Bs2', 4, no, n, nd)
        Nd, Md = buf('Nd', v, n, n, nd), buf('Md', v, n, n, nd)
        Sall = buf('S', 24, n, n, n, nd)

        def gather(trips, shape, lo):
            views = [blocks.view(a, b, c, v) for a, b, c in trips]
            offs = [o_ if o_ < 0 else o_ + int(np.dot(lo, st)) for o_, st in views]
            out = buf('Tg', len(trips), *shape)
            gather_boxes(t3b, offs, [st for _, st in views], out)
            return out.reshape(len(trips), -1)

        def unpack(dst, src, rows):
            with timer('q.unpack'):
                lib.zc_sa_q_unpack_d(P(dst), I64(v), I64(n), I64(len(rows)), P(src),
                                     P(np.ascontiguousarray(rows, dtype=np.int64)))

        def fill_face(o, S):
            o3 = o[3]
            for c, (w, x, y) in enumerate(ROT):
                ow, ox, oy = o[w], o[x], o[y]
                # ---- A_c: T3 x <vv|vo> for the partition {w x | y 3}
                with timer('q.stack'):
                    axpby(As[:, :, 0], cube(ow, ox, oy)[:n, :n], 1.0, 0.0)
                    axpby(As[:, :, 1], cube(ow, ox, o3)[:n, :n], 1.0, 0.0)
                    axpby(As[:, :, 2], VO[ox][:n, :n], 1.0, 0.0)
                    axpby(As[:, :, 3], VO[ow].transpose(1, 0, 2)[:n, :n], 1.0, 0.0)
                    axpby(Bs[0], VO[o3].transpose(2, 0, 1)[:, :n, d0:n], 1.0, 0.0)
                    axpby(Bs[1], VO[oy].transpose(2, 1, 0)[:, :n, d0:n], 1.0, 0.0)
                    axpby(Bs[2], cube(ow, oy, o3)[:, :n, d0:n], 1.0, 0.0)
                    axpby(Bs[3], cube(ox, oy, o3)[:, :n, d0:n], 1.0, 0.0)
                with timer('q.Xp'):
                    gemm(As.reshape(nn, 4 * v), Bs.reshape(4 * v, n * nd), S[c].reshape(nn, n * nd))
                # ---- Xh, special pair w first: m in the slot of x, y or 3
                with timer('q.gather'):
                    Tm = gather([(m, oy, o3) for m in occ] + [(ox, m, o3) for m in occ]
                                + [(ox, oy, m) for m in occ], (n, n, nd), (0, 0, d0))
                with timer('q.Xh'):
                    sm = np.ascontiguousarray(
                        np.hstack([-ovoo[:, :, r, ow].T for r in (ox, oy, o3)])[:n])
                    gemm(sm, Tm, S[c].reshape(n, nn * nd), beta=1.0)
                if c == 0:
                    # special pair 3 last: [(x_0 x_1 x_2); x_3]
                    with timer('q.gather'):
                        Tm = gather([(m, o[1], o[2]) for m in occ] + [(o[0], m, o[2]) for m in occ]
                                    + [(o[0], o[1], m) for m in occ], (n, n, n), (0, 0, 0))
                    with timer('q.Xh'):
                        sm = np.ascontiguousarray(
                            np.vstack([-ovoo[:, :, r, o3] for r in (o[0], o[1], o[2])])[:, d0:n])
                        gemm(Tm.T, sm, S[0].reshape(n3, nd), beta=1.0)
                # ---- B_c: Yv(w,x|y,3) + Yv(x,w|y,3) from N of the pair (y, 3)
                with timer('q.Yv'):
                    ab = oy * no + o3
                    unpack(Nd, N.reshape(-1, ps),
                           ((np.arange(n)[:, None] * npair + ab) * nd + np.arange(nd)[None, :]).ravel())
                    twx = t2[ow, ox]
                    gemm(np.ascontiguousarray(twx[:n]), Nd.reshape(v, nn * nd),
                         S[3 + c].reshape(n, nn * nd))
                    # (w <-> x): [x_w][x_x, (x_y d)] += t_wx^T N[:, x_w]
                    gemm_slices(twx.T[:n], Nd.reshape(v, n, n * nd),
                                S[3 + c].reshape(n, n, n * nd), beta=1.0)
            for c, (w, x, y) in enumerate(ROT):
                ow, ox, oy = o[w], o[x], o[y]
                c1 = (c + 1) % 3
                # ---- Yv(x,y|3,w) + Yv(y,x|3,w) from M of the pair (3, w): layout [x, y, w, 3]
                with timer('q.Yv'):
                    ab = o3 * no + ow
                    unpack(Md, M.reshape(-1, ps),
                           ((np.arange(nd)[None, :] * npair + ab) * n + np.arange(n)[:, None]).ravel())
                    txy = t2[ox, oy]
                    gemm(np.ascontiguousarray(txy[:n]), Md.reshape(v, nn * nd),
                         S[3 + c1].reshape(n, nn * nd), beta=1.0)
                    gemm_slices(txy.T[:n], Md.reshape(v, n, n * nd),
                                S[3 + c1].reshape(n, n, n * nd), beta=1.0)
                # ---- T2 x G for the ordered pairs (w,x), (x,w), (y,3), (3,y) on T2
                with timer('q.G'):
                    As2[:, :, 0] = t2[ow].transpose(1, 2, 0)[:n, :n]
                    As2[:, :, 1] = t2[ox].transpose(2, 1, 0)[:n, :n]
                    As2[:, :, 2] = Gh(o3, ow, ox).transpose(1, 2, 0)[:n, :n]
                    As2[:, :, 3] = Gh(oy, ow, ox).transpose(1, 2, 0)[:n, :n]
                    Bs2[0] = Gh(ox, oy, o3)[:, :n, d0:n]
                    Bs2[1] = Gh(ow, oy, o3)[:, :n, d0:n]
                    Bs2[2] = t2[oy][:, :n, d0:n]
                    Bs2[3] = t2[o3].transpose(0, 2, 1)[:, :n, d0:n]
                    gemm(As2.reshape(nn, 4 * no), Bs2.reshape(4 * no, n * nd),
                         S[3 + c].reshape(nn, n * nd), beta=1.0)

        for o, mult in unique_quadruples(no):
            # face p: the quadruple with pair p last; equal occupied indices
            # give equal faces
            faces, done, k = [], {}, 0
            for p in range(4):
                op = tuple(o[q] for q in range(4) if q != p) + (o[p],)
                if op not in done:
                    done[op] = Sall[6 * k:6 * k + 6]
                    k += 1
                    fill_face(op, done[op])
                faces.append(done[op])
            with timer('q.energy'):
                tc = np.array([t2[o[p], o[q]] for p, q, r, s in PAIRS])
                gc = np.array([oovv[o[r], o[s]] for p, q, r, s in PAIRS])
                ptr = (t3pack.ctypes.c_void_p * 24)(*[f[i].ctypes.data for f in faces for i in range(6)])
                out = np.zeros(2)
                lib.zc_sa_q_energy_faces_d(I64(v), I64(t0), I64(t1), ptr, P(tc), P(gc),
                                           t3pack.ctypes.c_double(float(sum(eo[i] for i in o))),
                                           P(ev), P(out))
            e5 += mult / 24 * out[0]
            e6 += mult / 24 * out[1]
    return e5, e5 + e6


def memory_needed(nocc, nvir):
    '''Bytes of the work arrays of q_correction (10 v^4 and smaller ones).'''
    v = nvir
    return 8 * (10 * v ** 4 + (2 * nocc + 8) * v ** 3 + 3 * nocc * v ** 3)


def sliced_memory(nocc, nvir, tiles=1):
    '''Bytes of ordinary memory of q_correction_slabs: per slab index 24 v^3
    (faces), o^2 v^3 (N and M) and 2 v^3 (their dense forms), besides the
    stacked operands and the gathered cubes.  The integral rows
    (4 v^4 bytes) are not counted: they may be mapped from the scratch file.'''
    o, v = nocc, nvir
    nd = min(QT * tiles, v)
    ps = v * (v + 1) // 2
    return 8 * (nd * (26 * v ** 3 + 2 * o * o * v * ps) + (4 + 5 * o) * v ** 3)


def _available_memory():
    '''Bytes the process can still allocate (node and memory cgroups), or
    None.'''
    from zccsdt import memory
    return memory.available()


def kernel(ints, blocks, t2, t3b, max_memory=None, tiles=None):
    '''(E[Q], E(Q)) for converged CCSDT amplitudes, with numpy's own BLAS
    kept single-threaded as in rccsdt.kernel.

    With enough memory for the work arrays of one occupied quadruple (10 v^4)
    this is the block algorithm q_correction.  Otherwise the v^4 arrays are
    cut into slabs of the largest virtual index (q_correction_slabs, six
    indices per slab), and the <vv|vv> integrals are read from a mapped
    scratch array unless they fit in memory as well.  ``max_memory`` (MB, as
    in rccsdt.kernel; default: 80% of the available memory) is the budget;
    ``tiles`` chooses explicitly: 0 for the block algorithm, else the number
    of kernel tiles (of six indices) per slab.  Raises MemoryError if the
    slabs do not fit either.'''
    from zccsdt import memory
    no, v = ints.nocc, ints.nvir
    ints.__dict__.pop('_sa_wpp', None)          # work buffers of the CCSDT iterations
    scr = erows = None
    if tiles is None:
        def budget():
            avail = _available_memory()
            if max_memory is not None:
                b = memory.budget(max_memory)
                return b, b
            return (None, None) if avail is None else (int(memory.FRACTION * avail), avail)

        soft, hard = budget()
        if soft is None or memory_needed(no, v) <= soft:
            tiles = 0
        else:
            # not enough for the block algorithm: free the in-memory packed
            # <vv|vv> of the iterations (rebuilt if needed again), take slabs
            packed = getattr(ints, '_sa_packed', None)
            if packed is not None and packed.get('scratch') is None:
                ints._sa_packed = None
                soft, hard = budget()
            tiles = 1
            if sliced_memory(no, v) > hard:
                raise MemoryError(
                    'CCSDT(Q) with %d virtual orbitals needs about %.1f GB with all work '
                    'arrays in memory (10 v^4) and %.1f GB in slabs; %.1f GB are available '
                    '(max_memory, or what the node and the memory limit of the job leave)'
                    % (v, memory_needed(no, v) / 1e9, sliced_memory(no, v) / 1e9, hard / 1e9))
            if sliced_memory(no, v) + 8 * v * v * (v * (v + 1) // 2) > soft:
                from zccsdt.sa.rccsdt import vvvv_rows
                from zccsdt.scratch import Scratch
                scr = Scratch(prefix='zccsdt_q_', what='the <vv|vv> integrals of (Q)',
                              hint=' or give the calculation more memory (max_memory)')
                erows = vvvv_rows(ints, out=scr.create('E', (v * v, v * (v + 1) // 2)),
                                  max_bytes=int(max(soft - sliced_memory(no, v), 1 << 28) // 2))

    def run():
        if tiles:
            return q_correction_slabs(ints, blocks, t2, t3b, tiles=tiles, erows=erows)
        return q_correction(ints, blocks, t2, t3b)

    try:
        try:
            from threadpoolctl import threadpool_limits
        except ImportError:
            return run()
        with threadpool_limits(limits={'openblas': 1}):
            return run()
    finally:
        if scr is not None:
            scr.close()
