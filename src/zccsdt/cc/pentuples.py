'''CCSDTQ(P) on spin orbitals or spinors, real or complex: the (P) correction
from the amplitudes of CCSDTQ, with T3 and T4 on their unique blocks.

    E(P) = sum conj(z) r / D,    r = <P| [V,T4] + [[V,T2],T3] + 1/6 [[[V,T2],T2],T2] |0>,
                                 z = <P| V T3 |0> + <P| V Y4 |0>_connected,
                                 Y4 = <Q| V T2 |0> / D4

summed over the unique pentuples i<j<k<l<m, a<b<c<d<e, D the differences of
the orbital energies; bare integrals, canonical orbitals, T1 does not enter.
This is the CCSDTQ(P) of MRCC for a Hartree-Fock reference (its CCSDTQ[P] is
the same number); the terms are those of zccsdt.cc.ccsdtq_terms, where the
left-hand side is discussed (zccsdt.cc.ccsdtq_ref).

Nothing of the size of T5 is kept: for one occupied pentuple at a time, r
and z over all virtual pentuples are matrix products on two groups of
virtual indices, each followed by the antisymmetrizing reduction onto
a<b<c<d<e (zccsdt.cc.t4pack).  With the splits of the pentuple stacked along
the summed index:

    X[(de),(abc)]   sum_f <de||fm> t(ijkl; abc f)             (ijkl|m), 5 splits     [V,T4]
                  + sum_g t(klm; g de) Xv(g; ij; abc)         (ij|klm), 10           T2 T3, <vv||vv> and <ov||vo>
                  + sum_n Xo(n; ijm; de) t(n kl; abc)         (ijm|kl), 10           T2 T3, <oo||oo> and <ov||vo>
                  + sum_n t(in; de) Xq(n; jklm; abc)          (i|jklm), 5            T2 T2 T2
    X[e,(abcd)]   - sum_n <ne||lm> t(ijkn; abcd)              (ijk|lm), 10           [V,T4]

and for z the first and the last line with Y4 in the place of T4, and
t(ijk; abc) <de||lm> (10 splits).  The intermediates of T2:

    Xv(g; ij; abc) = P(a/bc) [ sum_f <bc||gf> t(ij; fa) + P(ij) sum_n <na||gj> t(in; bc) ]
    Xo(n; ijm; de) = P(m/ij) [ sum_o <no||ij> t(om; de) + sum_f ( <ne||fm> t(ij; df) - <nd||fm> t(ij; ef) ) ]
    Xq(n; jklm; abc) = P(c/ab) [   1/2 P(j/klm) sum_o t(jo; ab) W(n, o; klm; c)
                                 -     P(jk/lm), j first:  sum_fg <cn||fg> t(jk; fa) t(lm; gb) + (a <-> b) ]
    W(n, o; klm; c) = P(m/kl) sum_f <no||fm> t(kl; fc)

Xv has v^4 / 6 elements per pair of occupied indices (as in the T4 equation
of CCSDTQ), Xq o v^3 / 6 per occupied quadruple.

The four products of the first group are one matrix product per pentuple,
(de) x 15 (v + o) x (abc); the ten splits of the last line take the
5 + 10 (o - 5) rows of T4 that they share once each.

Cost: o^5 v^5 (20 v + 15 o + 10) / 1440 multiplications in the products
with two and three virtual indices, a quarter of them for z.

Memory (plan()): besides T3, T4 and the packed <vv||vv>, which are given,
Y4 (as large as T4), Xv for all pairs, Xq for all occupied quadruples
(4 o / (v - 3) of T4) and the products of one batch of columns.  With less
memory the products are done in more batches, and Xq is made for the five
quadruples of a pentuple when it comes (o - 4 times each, at o v^4 per
quadruple: a few percent of the products).  Y4 and Xv are the floor.
'''

from dataclasses import dataclass
from math import comb

import numpy as np

from zccsdt.blas import gemm
from zccsdt.cc import t3pack, t4pack as P
from zccsdt.cc.ccsdt_opt import vvvv_packed
from zccsdt.cc.quadruples import _batches, _pairs, check_canonical, second_order_quadruples
from zccsdt.timing import timer

BATCH_BYTES = 1 << 30           # an operand or a product of one batch of columns, at most
PARTS = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'zt3', 'zy')


PASSES = 16                     # batches of columns of a product, at most
SLACK = 1 << 20                 # Python objects and small arrays


@dataclass
class Plan:
    cols23: int         # columns (abc) of one product X[(de),(abc)]
    cols14: int         # columns (abcd) of one product X[e,(abcd)]
    rows: int           # rows of <bc||ef> (and of the other operands) in one step of the intermediates
    store_xq: bool      # Xq for all occupied quadruples (else: for those of one pentuple)
    need: int           # bytes of the arrays of the calculation (without T3, T4 and <vv||vv>)
    floor: int          # the least that a calculation takes


def _sizes(no, nv):
    return (comb(no, 2), comb(nv, 2), comb(no, 3), comb(nv, 3), comb(no, 4), comb(nv, 4), comb(no, 5),
            comb(nv, 5))


def _kept(no, nv, isz):
    '''Bytes of what is there from the start to the end, whatever the plan
    (Xq apart), and the size of Xq for all quadruples and for five.'''
    nP, nQ, nI3, nA3, nI4, nA4, nI5, nA5 = _sizes(no, nv)
    amps = nI4 * nA4 + nP * nv * nA3 + nI3 * no * nQ                       # Y4, Xv, Xo
    layouts = nP * nQ + no * nv * nQ + nP * no * nv + no * no * nQ + nP * nv * nv   # integrals, T2 twice
    work = 2 * nA5 + (20 * nv + 15 * no + 10) * nQ + 20 * no * nv + 10 * nA4 + 10 * nA3
    tables = 8 * nA5 + nI5 * 600 + nI4 * (no * 9 + 250) + nI3 * (nP + no) * 9 + nP * nP * 9
    # (for the quadruples of one pentuple: with the operands that Xq is made of)
    five = 5 * no * nA3 + nI3 * no * no * nv + no * nv ** 3 + nQ * no * nv
    return isz * (amps + layouts + work) + tables + SLACK, isz * nI4 * no * nA3, isz * five


def _transient(no, nv, isz, cols23, cols14, rows, store_xq=True):
    '''Bytes of the largest set of arrays that live for a while: those of
    the formation of the intermediates, those of a batch of columns (with
    a step of Xq where that is made on the way).'''
    nP, nQ, nI3, nA3, nI4, nA4, nI5, nA5 = _sizes(no, nv)
    # (the left-hand quadruples: zccsdt.cc.quadruples in its smallest batches)
    y4 = nQ * -(-nQ // PASSES) + 3 * nA4
    xv = 3 * rows * nv * nv + no * no * (nv * nv + nQ)
    xo = no * nP * nQ + 2 * nP * nv * nv + no * no * nP + 4 * nI3 * nQ
    # a step of Xq: a few virtual indices a at a time, rows / o of them (one at least)
    step = 4 * max(rows, no) * nv * nv
    xq = max(no ** 3 * nv + 2 * no ** 3 * nP * nv + 4 * nI3 * no * no * nv, no * nv ** 3 + nQ * no * nv + step)
    # the work arrays stay from one product to the next: each as large as its largest use
    batch = max(15 * (nv + no) * cols23, 10 * no * cols14) + max(nQ * cols23, nv * cols14) + 10 * cols23
    if not store_xq:
        batch += step
    return isz * max(y4, xv, xo, xq, batch)


def _least(no, nv):
    nQ, nA3, nA4 = comb(nv, 2), comb(nv, 3), comb(nv, 4)
    return -(-nA3 // PASSES), -(-nA4 // PASSES), max(1, -(-nQ // PASSES))


def _floors(no, nv, isz):
    '''The least that a calculation takes with Xq for all quadruples, and
    with Xq made for the quadruples of one pentuple.'''
    kept, xq_all, xq_five = _kept(no, nv, isz)
    least = _least(no, nv)
    return (kept + xq_all + _transient(no, nv, isz, *least, True),
            kept + xq_five + _transient(no, nv, isz, *least, False))


def explicit_plan(no, nv, isz, cols23, cols14, rows, store_xq=True):
    '''The Plan with these batch sizes.'''
    kept, xq_all, xq_five = _kept(no, nv, isz)
    floor = min(_floors(no, nv, isz))
    need = kept + (xq_all if store_xq else xq_five) + _transient(no, nv, isz, cols23, cols14, rows, store_xq)
    return Plan(int(cols23), int(cols14), int(rows), bool(store_xq), int(need), int(floor))


def plan(no, nv, isz, budget=None, batch_bytes=None):
    '''The Plan for ``budget`` bytes (None: no limit).  The floor - below it
    MemoryError - is Y4, Xv, Xq for the quadruples of one pentuple with the
    operands it is made of (or Xq for all quadruples, where that is less)
    and the products in PASSES batches of columns; from there first Xq for
    all quadruples, then the products in fewer batches (of ``batch_bytes``
    at most), then more rows per step of the intermediates: one after the
    other, so that no part of a plan shrinks when the budget grows.'''
    batch_bytes = batch_bytes or BATCH_BYTES
    nP, nQ, nI3, nA3, nI4, nA4, nI5, nA5 = _sizes(no, nv)
    kept, xq_all, xq_five = _kept(no, nv, isz)
    min23, min14, min_rows = _least(no, nv)
    top23 = int(max(min23, min(nA3, batch_bytes // (isz * (nQ + 15 * (nv + no))))))
    top14 = int(max(min14, min(nA4, batch_bytes // (isz * (nv + 10 * no)))))
    top_rows = int(max(min_rows, min(nQ, batch_bytes // (isz * 4 * nv * nv))))
    if budget is None:
        return explicit_plan(no, nv, isz, top23, top14, top_rows, True)
    with_all, with_five = _floors(no, nv, isz)
    if budget < min(with_all, with_five):
        raise MemoryError('(P) needs %.0f MB besides T3, T4 and the integrals for %d occupied and %d virtual '
                          'spin orbitals (the second-order quadruples, as large as T4: %.0f MB); %.0f MB are left'
                          % (min(with_all, with_five) / 1e6, no, nv, isz * nI4 * nA4 / 1e6, budget / 1e6))
    store = budget >= with_all
    rest = budget - kept - (xq_all if store else xq_five)
    cols23, cols14, rows = min23, min14, min_rows
    if store:
        # the two products grow together, by the same fraction of what they lack
        def fits(x):
            return _transient(no, nv, isz, int(min23 + x * (top23 - min23)), int(min14 + x * (top14 - min14)),
                              min_rows) <= rest
        if fits(1.0):
            cols23, cols14 = top23, top14
            lo, hi = min_rows, top_rows
            while lo < hi:                      # the largest number of rows that fits
                mid = (lo + hi + 1) // 2
                if _transient(no, nv, isz, cols23, cols14, mid) <= rest:
                    lo = mid
                else:
                    hi = mid - 1
            rows = lo
        else:
            lo, hi = 0.0, 1.0
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if fits(mid) else (lo, mid)
            cols23, cols14 = int(min23 + lo * (top23 - min23)), int(min14 + lo * (top14 - min14))
    return explicit_plan(no, nv, isz, cols23, cols14, rows, store)


class Pentuples:
    '''r and z of (P) for one occupied pentuple at a time.  ``only``: the
    parts to evaluate (tests), of PARTS - the letters of the terms of r in
    the order of the table, 'zt3' and 'zy' for the two parts of z.'''

    def __init__(self, eris, t2, t3, t4, plan_=None, only=None):
        check_canonical(eris)
        self.nocc = no = eris.nocc
        self.nvir = nv = np.asarray(eris.fock).shape[0] - no
        self.dtype = dtype = np.result_type(t2, t3, t4, eris.oovv)
        isz = np.dtype(dtype).itemsize
        self.plan = pl = plan_ or plan(no, nv, isz)
        self.only = only
        want = self.want
        nP, nQ, nI3, nA3 = comb(no, 2), comb(nv, 2), comb(no, 3), comb(nv, 3)
        nI4, nA4, nA5 = comb(no, 4), comb(nv, 4), comb(nv, 5)
        self.nQ, self.nA3, self.nA4, self.nA5 = nQ, nA3, nA4, nA5
        i2, j2 = _pairs(no)
        a2, b2 = _pairs(nv)

        def c_(x):
            return np.ascontiguousarray(x, dtype=dtype)
        mo_e = np.asarray(eris.mo_energy)
        self.pents = P.tuples(no, 5)
        self.eo5 = mo_e[:no][self.pents].sum(axis=1)
        self.ev5 = P.tuple_sums(mo_e[no:], 5)
        self.o41, self.o32 = P.split(no, 4, 1), P.split(no, 3, 2)
        self.o31, self.o12 = P.split(no, 3, 1), P.split(no, 1, 2)
        o13, o22 = P.split(no, 1, 3), P.split(no, 2, 2)
        self.T3 = c_(np.asarray(t3).reshape(nI3, nA3))
        self.T4 = c_(np.asarray(t4).reshape(nI4, nA4))
        with timer('p.blocks'):
            x = np.asarray(eris.oovv)[i2, j2][:, a2, b2]                    # <de||lm> = <lm||de>*
            self.vvoo_t = c_(np.conj(x, out=x))                             # [(lm), (de)]
            # <de||fm> = - <mf||de>* as [m, f, (de)],  <ne||lm> = <lm||ne>* as [(lm), n, e]
            x = np.asarray(eris.ovvv)[:, :, a2, b2]
            self.vvvo_t = c_(np.negative(np.conj(x, out=x), out=x))
            x = np.asarray(eris.ooov)[i2, j2]
            self.ovoo_t = c_(np.conj(x, out=x))
        self.t2_xp = t2_xp = c_(t2[:, :, a2, b2])                           # [i, n, (ab)]
        tfc = c_(t2[i2, j2])                                                # [(ij), f, c]
        ovvo = np.asarray(eris.ovvo)
        with timer('p.y4'):
            # (the products of the left-hand quadruples in their smallest batches of columns)
            from zccsdt.cc import quadruples
            qp = quadruples.explicit_plan(no, nv, isz, -(-nQ // PASSES), -(-nA3 // PASSES), 3, 1)
            self.Y4 = c_(second_order_quadruples(eris, t2, qp)) if want('zy') else None

        # ---- Xv[(ij), g, (abc)]
        with timer('p.xv'):
            self.Xv = Xv = np.zeros((nP, nv, nA3), dtype=dtype)
            if want('c'):
                vpp = vvvv_packed(eris, t3pack.triple_map(nv))[0]           # [(bc), (gf)]
                for r0, r1 in _batches(nQ, pl.rows):
                    nr = r1 - r0
                    W = np.empty((nr, nv, nv), dtype=dtype)                 # [(bc), g, f]
                    P.tn_expand(W, c_(vpp[r0:r1]), nv, 1, 1, outer=0)
                    W = W.reshape(nr * nv, nv)
                    M = np.empty((nr * nv, nv), dtype=dtype)
                    for p in range(nP):
                        gemm(W, tfc[p], M)                                  # [(bc), g, a]
                        P.tn_reduce(Xv[p], M.reshape(nr, nv, nv), nv, 2, 1, rows=(r0, r1), outer=1)
                del W, M, vpp
            if want('e'):
                # P(ij) P(c/ab) sum_n <nc||gj> t(in; ab), from [(ab), n] [n, (g c)], rows (ab) in steps
                ovvo_t = c_(ovvo.transpose(3, 0, 2, 1)).reshape(no, no, nv * nv)        # [j, n, (g c)]
                t2_t = c_(t2_xp.transpose(0, 2, 1))                                     # [i, (ab), n]
                for r0, r1 in _batches(nQ, pl.rows):
                    K = np.empty((r1 - r0, nv * nv), dtype=dtype)
                    for p in range(nP):
                        for x, y, sg in ((i2[p], j2[p], 1.0), (j2[p], i2[p], -1.0)):
                            gemm(t2_t[x, r0:r1], ovvo_t[y], K)
                            P.tn_reduce(Xv[p], K.reshape(r1 - r0, nv, nv), nv, 2, 1, rows=(r0, r1), scale=sg, outer=1)
                del K, ovvo_t, t2_t

        # ---- Xo[(ijm), n, (de)]: for one n at a time G[m, (ij), (de)] and P(m/ij) of it
        with timer('p.xo'):
            self.Xo = np.zeros((nI3, no, nQ), dtype=dtype)
            if want('d', 'f'):
                oooo_t = c_(np.asarray(eris.oooo)[:, :, i2, j2].transpose(0, 2, 1))     # [n, (ij), o]
                t2_pc = tfc.reshape(nP * nv, nv)                                        # [(ij) d, f]
                G = np.zeros((no, nP, nQ), dtype=dtype)
                for n in range(no):
                    for m in range(no):
                        if want('f'):
                            # sum_f t(ij; d f) <ne||fm> as [(ij), d, e], then d < e of its antisymmetric part
                            K = gemm(t2_pc, c_(ovvo[n, :, :, m].T)).reshape(nP, nv, nv)
                            np.subtract(K[:, a2, b2], K[:, b2, a2], out=G[m])
                        if want('d'):                                                   # + sum_o <no||ij> t(om; de)
                            gemm(oooo_t[n], t2_xp[:, m], G[m], beta=1.0 if want('f') else 0.0)
                    self.Xo[:, n] = P.fold(G, self.o12, 0, 1)
                del G, oooo_t

        # ---- Xq[(jklm), n, (abc)]: its operands, and all of it where it is kept
        with timer('p.xq'):
            self.o13, self.o22, self.tfc = o13, o22, tfc
            self.W1 = self.V = None
            if want('g'):
                # W(n, o; klm; c) = P(m/kl) sum_f <no||fm> t(kl; fc), as [(klm), o, (n c)]
                oovo = -np.asarray(eris.ooov).transpose(0, 1, 3, 2)                     # <no||fm>
                G = np.einsum('nofm,Pfc->mPonc', oovo, tfc, optimize=True)
                self.W1 = c_(P.fold(G, self.o12, 0, 1)).reshape(nI3, no, no * nv)
                del G, oovo
            if want('h'):
                # <cn||fg> as [f, (n c g)]
                self.V = c_(-np.asarray(eris.ovvv).transpose(2, 0, 1, 3)).reshape(nv, no * nv * nv)
            self._zq = np.empty((nQ, no * nv), dtype=dtype)
            self.Xq = None
            if pl.store_xq:
                self.Xq = np.empty((nI4, no, nA3), dtype=dtype)
                for J in range(nI4):
                    self.xi_q(J, self.Xq[J])
                self.W1 = self.V = self._zq = self.tfc = None
            else:
                self._five = (np.empty((5, no, nA3), dtype=dtype), {})

        # ---- the work arrays of one pentuple
        self.r, self.z = np.empty(nA5, dtype=dtype), np.empty(nA5, dtype=dtype)
        self.LA = np.empty((15 * nv + 15 * no, nQ), dtype=dtype)
        self.LZ = np.empty((5 * nv + 10, nQ), dtype=dtype)
        self.LB, self.LC = (np.empty((10 * no, nv), dtype=dtype) for _ in range(2))
        self.Xv2 = self.Xv.reshape(nP * nv, nA3)
        self._buf = {}

    def want(self, *parts):
        return self.only is None or any(p in self.only for p in parts)

    def xi_q(self, J, out):
        '''Xq[n, (abc)] of the occupied quadruple J into ``out``.'''
        no, nv, nQ, o13, o22, tfc = self.nocc, self.nvir, self.nQ, self.o13, self.o22, self.tfc
        a2, b2 = _pairs(nv)
        P.zero(out)
        if self.W1 is not None:
            Z = self._zq
            for s in range(4):                                                          # (j|klm)
                gemm(self.t2_xp[o13.idx1[J, s]].T, self.W1[o13.idx2[J, s]], Z)          # [(ab), n, c]
                P.tn_reduce(out, Z.reshape(nQ, no, nv), nv, 2, 1, scale=0.5 * o13.sgn[J, s], outer=1)
        if self.V is not None:
            step = max(1, min(nv, self.plan.rows // max(no, 1)))
            for s in range(3):                                                          # (jk|lm), j in the first pair
                p, q, sg = o22.idx1[J, s], o22.idx2[J, s], float(o22.sgn[J, s])
                ta = np.ascontiguousarray(tfc[p].T)                                     # [a, f]
                for r0, r1 in _batches(nv, step):
                    U = gemm(ta[r0:r1], self.V)                                         # [a, (n c), g]
                    Z = gemm(U.reshape((r1 - r0) * no * nv, nv), tfc[q]).reshape(r1 - r0, no, nv, nv)
                    Zp = np.ascontiguousarray(Z[:, :, b2, a2] - Z[:, :, a2, b2])        # [a, n, (b c)]
                    P.tn_reduce(out, Zp, nv, 1, 2, rows=(r0, r1), scale=-sg, outer=1)
                    del U, Z, Zp
        return out

    def _xq_of(self, J4s):
        '''[Xq of the five quadruples of a pentuple].'''
        if self.Xq is not None:
            return [self.Xq[J] for J in J4s]
        store, where = self._five
        free = [k for k in range(5) if k not in [where.get(int(J)) for J in J4s]]
        for J in list(where):
            if J not in [int(x) for x in J4s]:
                del where[J]
        with timer('p.xq'):
            for J in J4s:
                if int(J) not in where:
                    where[int(J)] = k = free.pop()
                    self.xi_q(int(J), store[k])
        return [store[where[int(J)]] for J in J4s]

    def _work(self, name, shape):
        n = int(np.prod(shape))
        a = self._buf.get(name)
        if a is None or a.size < n:
            # (the old one goes before the new one comes)
            del a
            self._buf[name] = None
            self._buf[name] = np.empty(max(n, 1), dtype=self.dtype)
        return self._buf[name][:n].reshape(shape)

    def both(self, I):
        '''(r, z) for the occupied pentuple I, [C(nv,5)] each (work arrays:
        valid until the next call).'''
        no, nv, nQ, nA3, nA4 = self.nocc, self.nvir, self.nQ, self.nA3, self.nA4
        want, T3, T4, Y4 = self.want, self.T3, self.T4, self.Y4
        o41, o32, o31, o12 = self.o41, self.o32, self.o31, self.o12
        J4s, xs, s41 = o41.idx1[I], o41.idx2[I], o41.sgn[I]
        J3s, ps, s32 = o32.idx1[I], o32.idx2[I], o32.sgn[I]
        r, z, LA, LZ, LB, LC = self.r, self.z, self.LA, self.LZ, self.LB, self.LC
        P.zero(r)
        P.zero(z)
        # the rows of the stacked operands: [V,T4] | Xo with T3 | T2 with Xq | T3 with Xv
        ka, kd, kg = 5 * nv, 5 * nv + 10 * no, 5 * nv + 15 * no
        Xq = self._xq_of(J4s) if want('g', 'h') else None
        with timer('p.operands'):
            for s in range(5):
                # (ijkl|m): - <de||fm> for t(ijkl; f abc) = - t(ijkl; abc f);  (i|jklm): t(in; de)
                np.multiply(self.vvvo_t[xs[s]], -float(s41[s]), out=LA[s * nv:(s + 1) * nv])
                np.multiply(self.t2_xp[xs[s]], float(s41[s]), out=LA[kd + s * no:kd + (s + 1) * no])
            LZ[:ka] = LA[:ka]
            for s in range(10):
                # (ijm|kl): Xo(n; ijm; de);  (ijk|lm): <de||lm>, - <ne||lm>
                np.multiply(self.Xo[J3s[s]], float(s32[s]), out=LA[ka + s * no:ka + (s + 1) * no])
                np.multiply(self.vvoo_t[ps[s]], float(s32[s]), out=LZ[ka + s])
                np.multiply(self.ovoo_t[ps[s]], -float(s32[s]), out=LB[s * no:(s + 1) * no])
            if want('c', 'e'):
                # (klm|ij): t(klm; g, de), for Xv(g; ij; abc) with the sign of the split
                P.tn_expand(LA[kg:].reshape(10, nv, nQ), np.ascontiguousarray(T3[J3s]), nv, 1, 2, outer=0)
                rowsv = (ps[:, None] * nv + np.arange(nv)).ravel()
                sgnv = np.repeat(s32, nv)
            rows3, sgn3 = o12.pos[:, ps].T.ravel(), o12.psg[:, ps].T.ravel()            # t(n kl; abc)
            # t(ijk n; abcd): the rows of T4 that the ten splits (ijk|lm) and the n make, each once
            rows4, sgn4 = o31.pos[J3s].ravel(), o31.psg[J3s].ravel()
            rows4, inv = np.unique(rows4[sgn4 != 0], return_inverse=True)
            LC[:len(rows4)] = 0
            np.add.at(LC, inv, LB[sgn4 != 0] * sgn4[sgn4 != 0][:, None])
            one4 = np.ones(len(rows4), dtype=np.int8)
            T4r = np.ascontiguousarray(T4[J4s])
            Y4r = np.ascontiguousarray(Y4[J4s]) if Y4 is not None else None
        for c in _batches(nA3, self.plan.cols23):
            nc = c[1] - c[0]
            # ---- r
            R = self._work('R', (15 * nv + 15 * no, nc))
            with timer('p.operands'):
                if want('a'):
                    P.tn_expand(R[:ka].reshape(5, nv, nc), T4r, nv, 1, 3, cols=c, outer=0)
                else:
                    P.zero(R[:ka])
                if want('d', 'f'):
                    P.tn_rows(R[ka:kd], T3, rows3, sgn3, cols=c)
                else:
                    P.zero(R[ka:kd])
                for s in range(5):
                    R[kd + s * no:kd + (s + 1) * no] = Xq[s][:, c[0]:c[1]] if Xq is not None else 0
                if want('c', 'e'):
                    P.tn_rows(R[kg:], self.Xv2, rowsv, sgnv, cols=c)
            X = self._work('X', (nQ, nc))
            with timer('p.gemm'):
                gemm((LA if want('c', 'e') else LA[:kg]).T, R if want('c', 'e') else R[:kg], X)
            with timer('p.reduce'):
                P.tn_reduce(r, X, nv, 2, 3, cols=c)
            # ---- z
            R = R[:ka + 10]
            with timer('p.operands'):
                if want('zy'):
                    P.tn_expand(R[:ka].reshape(5, nv, nc), Y4r, nv, 1, 3, cols=c, outer=0)
                else:
                    P.zero(R[:ka])
                R[ka:] = T3[J3s, c[0]:c[1]] if want('zt3') else 0
            with timer('p.gemm'):
                gemm(LZ.T, R, X)
            with timer('p.reduce'):
                P.tn_reduce(z, X, nv, 2, 3, cols=c)
        R = X = None                    # (views of the work arrays: they would pin them)
        for c in _batches(nA4, self.plan.cols14):
            nc = c[1] - c[0]
            R = self._work('R', (len(rows4), nc))
            X = self._work('X', (nv, nc))
            for amp, out, part in ((T4, r, 'b'), (Y4, z, 'zy')):
                if not want(part):
                    continue
                with timer('p.operands'):
                    P.tn_rows(R, amp, rows4, one4, cols=c)
                with timer('p.gemm'):
                    gemm(LC[:len(rows4)].T, R, X)
                with timer('p.reduce'):
                    P.tn_reduce(out, X, nv, 1, 4, cols=c)
        return r, z


def kernel(eris, t2, t3, t4, max_memory=None, plan_=None):
    '''E(P) from converged CCSDTQ amplitudes: t2[i,j,a,b], T3 and T4 packed
    ([C(no,3), C(nv,3)] and [C(no,4), C(nv,4)], or flat), on an integral
    container (zccsdt.eris.ERIs).'''
    no = eris.nocc
    nv = np.asarray(eris.fock).shape[0] - no
    if no < 5 or nv < 5:
        return 0.0
    if plan_ is None:
        from zccsdt import memory
        isz = np.dtype(np.result_type(t2, t3, t4, eris.oovv)).itemsize
        plan_ = plan(no, nv, isz, memory.budget(max_memory))
    p = Pentuples(eris, t2, t3, t4, plan_)
    e = 0.0
    for I in range(len(p.pents)):
        r, z = p.both(I)
        with timer('p.energy'):
            e += 0.5 * P.energy2(z, r, r, p.ev5, p.eo5[I])[0]
    return float(e)


def rows(eris, t2, t3, t4, plan_=None, only=None):
    '''(r, z) for all pentuples, [C(no,5), C(nv,5)] each: for tests and
    small cases.'''
    p = Pentuples(eris, t2, t3, t4, plan_, only)
    out = [np.zeros((len(p.pents), p.nA5), dtype=p.dtype) for _ in range(2)]
    for I in range(len(p.pents)):
        out[0][I], out[1][I] = p.both(I)
    return out
