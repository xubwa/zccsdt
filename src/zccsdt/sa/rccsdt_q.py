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
