'''(P) in symmetry sectors: zccsdt.cc.pentuples for spinors that carry a
label of an Abelian group (the labels: zccsdt.cc.quadruples_sym).

For an occupied pentuple of total label M only the virtual pentuples of
total M are there, and the two kinds of products of zccsdt.cc.pentuples fall
into blocks:

    X[(de), (abc)]   one block per label q of (de): the pairs of label q
                     times the triples of label M - q; of the stacked rows
                     of a split only the summed indices with the label that
                     q leaves
    X[e, (abcd)]     one block per label p of e: the virtual spinors of
                     label p times the quadruples of label M - p

for r and for z each.  The blocks of one occupied pentuple are formed by
matrix products on the rows and columns that the labels allow, and one
kernel (zc_p_sector_energy, csrc/psym.c) walks through the virtual pentuples
of the sector, sums the ten and the five splits of each from the blocks and
adds the energy: nothing of the size of the virtual pentuples is stored, and
no tables of their size are made.

The two large intermediates of T2 are in blocks too, formed from blocks:

    Xv(g; ij; abc)      for the g of one label the triples of one label
    Xq(n; jklm; abc)    for the n of one label the triples of one label

T3 and T4 are the packed blocks of the dense code; what a product needs of
them is read where it is (quad_rows13(), quad_pick()).  The second-order
quadruples of the left-hand side (as large as T4), the intermediate
Xo(n; ijm; de) and the layouts of T2 and of the integrals with two virtual
indices are those of the dense code.
'''

import ctypes
import os
from dataclasses import dataclass
from math import comb

import numpy as np

from zccsdt.blas import gemm
from zccsdt.cc import pentuples as PT, quadruples as Q, quadruples_sym as QS, t3pack, t4pack as P
from zccsdt.cc.ccsdt_opt import vvvv_packed
from zccsdt.timing import timer

_I64, _INT, _DBL = ctypes.c_int64, ctypes.c_int, ctypes.c_double
_p = P._p
PASSES = Q.PASSES


def _load_lib():
    path = os.environ.get('ZCCSDT_PSYM_LIB') or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lib', 'libzc_psym.so')
    try:
        return ctypes.CDLL(path)
    except OSError:
        return None


_lib = _load_lib()


def available(eris):
    '''Whether the container has labels and the kernels are built.'''
    return QS.available(eris) and _lib is not None


def _fn(name, dtype):
    return getattr(_lib, 'zc_p_%s_%s' % (name, P._SFX[np.dtype(dtype)]))


def _i64(x):
    return np.ascontiguousarray(x, dtype=np.int64)


def _c(x, dtype=None):
    return np.ascontiguousarray(x, dtype=dtype)


class Sectors(QS.Sectors):
    '''The classes of zccsdt.cc.quadruples_sym.Sectors, those of the virtual
    quadruples, the labels of the occupied tuples and the sizes of the
    blocks.'''

    def __init__(self, eris):
        super().__init__(eris)
        no = self.nocc
        k4 = np.rint(P.tuple_sums(self.kv.astype(np.float64), 4)).astype(np.int32)
        self.A4, self.pos4 = QS._classes(self.red(k4))
        del k4
        self.kv32 = _c(self.kv, np.int32)
        self.V32 = {k: _c(x, np.int32) for k, x in self.V.items()}
        self.olab = sorted(self.O)
        self.kpair = self.red(self.ko[P.tuples(no, 2)].sum(axis=1))
        self.k3 = self.red(self.ko[P.tuples(no, 3)].sum(axis=1))
        self.k4 = self.red(self.ko[P.tuples(no, 4)].sum(axis=1))
        self.k5 = self.red(self.ko[P.tuples(no, 5)].sum(axis=1))
        self.lid = {k: l for l, k in enumerate(self.vlab)}
        self._layout5 = {}

    def layout5(self, M):
        '''The blocks of the products of an occupied pentuple of total label
        M: [(q, offset, rows, columns)] and the size for X[(de), (abc)],
        [(p, offset, rows, columns)] and the size for X[e, (abcd)], and the
        same as the arrays of the kernel.'''
        if M not in self._layout5:
            b23, n23 = [], 0
            off23, nc23 = np.full(self.nq, -1, dtype=np.int64), np.zeros(self.nq, dtype=np.int64)
            for q in self.qlab:
                r = int(self.red(M - q))
                if r in self.A3:
                    nr, nc = len(self.Q2[q]), len(self.A3[r])
                    b23.append((q, n23, nr, nc))
                    off23[q - self.qmin], nc23[q - self.qmin] = n23, nc
                    n23 += nr * nc
            b14, n14 = [], 0
            off14, nc14 = np.full(len(self.vlab), -1, dtype=np.int64), np.zeros(len(self.vlab), dtype=np.int64)
            for l, p in enumerate(self.vlab):
                r = int(self.red(M - p))
                if r in self.A4:
                    nr, nc = len(self.V[p]), len(self.A4[r])
                    b14.append((p, n14, nr, nc))
                    off14[l], nc14[l] = n14, nc
                    n14 += nr * nc
            self._layout5[M] = (b23, n23, b14, n14, off23, nc23, off14, nc14)
        return self._layout5[M]

    def sizes5(self):
        '''Elements of the largest set of blocks X[(de), (abc)], of the
        largest set X[e, (abcd)] and of the largest block of these.'''
        lay = [self.layout5(int(M)) for M in np.unique(self.k5)]
        return (max(x[1] for x in lay), max(x[3] for x in lay),
                max([nr * nc for x in lay for _, _, nr, nc in x[2]] or [0]))

    # ---- the blocks of the intermediates

    def xv_layout(self, kij, kg):
        '''The products that Xv(g; ij; abc) is made from, for the g of label
        ``kg`` and a pair (ij) of label ``kij``: (label of the triples,
        [(label of (bc), label of a, offset)], size, offsets and widths for
        the kernel); None if there are no such triples.'''
        T = int(self.red(kg + kij))
        if T not in self.A3:
            return None
        todo, n = [], 0
        off, nd = np.full(self.nq, -1, dtype=np.int64), np.zeros(self.nq, dtype=np.int64)
        for q2 in self.qlab:
            ka = int(self.red(T - q2))
            if ka in self.V:
                todo.append((q2, ka, n))
                off[q2 - self.qmin], nd[q2 - self.qmin] = n, len(self.V[ka])
                n += len(self.Q2[q2]) * len(self.V[kg]) * len(self.V[ka])
        return T, todo, n, off, nd

    def xv_sizes(self):
        '''Elements of the blocks of Xv for all pairs of occupied indices,
        and of the largest set of products that one of them is made from.'''
        total = work = 0
        for kij, count in zip(*np.unique(self.kpair, return_counts=True)):
            for kg in self.vlab:
                lay = self.xv_layout(int(kij), kg)
                if lay is not None:
                    total += int(count) * len(self.V[kg]) * len(self.A3[lay[0]])
                    work = max(work, lay[2])
        return total, work

    def xq_layout(self, kJ, kn):
        '''The products that Xq(n; jklm; abc) is made from, for the n of
        label ``kn`` and a quadruple of label ``kJ``: (label of the triples,
        [(label of (ab), label of c, offset)] with its size, offsets and
        widths - the term with one integral <oo||vo> -, and
        [(label of a, label of b, label of c, offset)] with its size - the
        term with <vo||vv>); None if there are no such triples.'''
        T = int(self.red(kJ - kn))
        if T not in self.A3:
            return None
        nn = len(self.O[kn])
        todo, n = [], 0
        off, nd = np.full(self.nq, -1, dtype=np.int64), np.zeros(self.nq, dtype=np.int64)
        for q2 in self.qlab:
            kc = int(self.red(T - q2))
            if kc in self.V:
                todo.append((q2, kc, n))
                off[q2 - self.qmin], nd[q2 - self.qmin] = n, len(self.V[kc])
                n += len(self.Q2[q2]) * nn * len(self.V[kc])
        nl = len(self.vlab)
        todo2, n2 = [], 0
        off2 = np.full(nl * nl, -1, dtype=np.int64)
        ncs, nbs = np.zeros(nl * nl, dtype=np.int64), np.zeros(nl * nl, dtype=np.int64)
        for la, ka in enumerate(self.vlab):
            for lb, kb in enumerate(self.vlab):
                kc = int(self.red(T - ka - kb))
                if kc in self.V:
                    todo2.append((ka, kb, kc, n2))
                    off2[la * nl + lb], ncs[la * nl + lb], nbs[la * nl + lb] = n2, len(self.V[kc]), len(self.V[kb])
                    n2 += len(self.V[ka]) * nn * len(self.V[kc]) * len(self.V[kb])
        return T, todo, n, off, nd, todo2, n2, off2, ncs, nbs

    def xq_sizes(self):
        '''Elements of the blocks of Xq for all occupied quadruples, of the
        largest sets of products that one of them is made from (the two
        terms), of the largest product with one T2 on the way, and of the
        blocks of <vo||vv> that are kept for it.'''
        total = work = work2 = half = 0
        pair_labels = [int(x) for x in np.unique(self.kpair)]
        for kJ, count in zip(*np.unique(self.k4, return_counts=True)):
            for kn in self.olab:
                lay = self.xq_layout(int(kJ), kn)
                if lay is None:
                    continue
                nn = len(self.O[kn])
                total += int(count) * nn * len(self.A3[lay[0]])
                work, work2 = max(work, lay[2]), max(work2, lay[6])
                for ka, kb, kc, _ in lay[5]:
                    for kjk in pair_labels:
                        kg = int(self.red(kJ - kjk - kb))
                        if int(self.red(kjk - ka)) in self.V and kg in self.V:
                            half = max(half, len(self.V[ka]) * nn * len(self.V[kc]) * len(self.V[kg]))
        ints = 0
        for kn in self.olab:
            for kf in self.vlab:
                for kg in self.vlab:
                    kc = int(self.red(kf + kg - kn))
                    if kc in self.V:
                        ints += len(self.O[kn]) * len(self.V[kf]) * len(self.V[kg]) * len(self.V[kc])
        return total, work, work2, half, ints


@dataclass
class Plan:
    need: int               # bytes besides T3, T4 and the integrals
    sectors: object         # Sectors
    n23: int                # elements of the largest set of blocks X[(de), (abc)]
    n14: int                # and X[e, (abcd)]
    work: int               # elements of the rows read for a product
    kept: int               # bytes of what stays to the end
    parts: dict             # bytes by array (for the output and for tests)


def plan(eris, isz, budget=None):
    """The plan for ``budget`` bytes (None: no limit); MemoryError if the
    calculation in blocks does not fit (the dense code has plans for less:
    batches of columns and scratch files).  Counted: the second-order
    quadruples (as large as T4), Xo and the layouts of T2 and of the
    integrals as in the dense code, Xv for all occupied pairs and Xq for all
    occupied quadruples in blocks, the blocks of r and z of one occupied
    pentuple, the rows read for a product, the tables of the sectors, and
    the largest of what the intermediates are made with."""
    sec = Sectors(eris)
    no, nv = sec.nocc, sec.nvir
    nP, nQ, nI3, nA3, nI4, nA4, nI5, nA5 = PT._sizes(no, nv)
    n23, n14, blk14 = sec.sizes5()
    xv, xv_work = sec.xv_sizes()
    xq, xq_work, xq_work2, xq_half, vints = sec.xq_sizes()
    maxV, maxO = (max(len(x) for x in c.values()) for c in (sec.V, sec.O))
    max3 = max(len(x) for x in sec.A3.values())
    work = max(maxV, maxO) * max3 + blk14
    parts = {
        'second-order quadruples': isz * nI4 * nA4,
        'Xo': isz * nI3 * no * nQ,
        'Xv': isz * xv,
        'Xq': isz * xq,
        'layouts': isz * (nP * nQ + no * nv * nQ + nP * no * nv + no * no * nQ + nP * nv * nv),
        'tables': (12 * nA4 + 24 * nA3 + 16 * nv * nQ + 16 * nQ + nI5 * 600 + nI4 * (no * 9 + 250)
                   + nI3 * (nP + no) * 9 + nP * nP * 9 + Q.SLACK),
    }
    kept = sum(parts.values())
    # what is there for a while: the formation of the four intermediates, each with what it
    # is made of, then the blocks of a pentuple
    t_sec = 28 * nA4                                                    # the labels of the quadruples
    t_y4 = isz * (2 * nP * nQ + nQ * -(-nQ // PASSES) + 2 * nA4) + 16 * nA4
    t_xo = isz * (no * nP * nQ + 2 * nP * nv * nv + no * no * nP + 4 * nI3 * nQ)
    t_xv = isz * (sec.w_size() + 2 * xv_work + 2 * maxO * nv * nv)
    w1 = nI3 * no * no * nv
    t_w1 = isz * (no ** 3 * nv + 2 * no ** 3 * nP * nv + 3 * w1)
    t_xq = isz * (w1 + vints + xq_work + xq_work2 + 2 * xq_half + 2 * max(maxO, maxV) ** 2 * nQ)
    t_run = isz * (2 * n23 + 2 * n14 + work + 4 * max(maxV, maxO) * max(len(x) for x in sec.Q2.values()))
    parts['blocks of a pentuple'] = t_run
    need = kept + max(t_sec, t_y4, t_xo, t_xv, t_w1, t_xq, t_run)
    if budget is not None and need > budget:
        raise MemoryError('(P) in blocks of the symmetry labels needs %.0f MB besides T3, T4 and the integrals '
                          'for %d occupied and %d virtual spinors; %.0f MB are left'
                          % (need / 1e6, no, nv, budget / 1e6))
    return Plan(int(need), sec, int(n23), int(n14), int(work), int(kept), parts)


# ---- what is read of the packed quadruples (T4 and the second-order ones)

def quad_rows13(T4, J, nv, es, tri, out):
    '''out[e, t] = t(J; e, abc) for the virtual spinors ``es`` and the sorted
    triples ``tri`` of the occupied quadruple J of a packed block.'''
    _fn('expand13', out.dtype)(_p(T4[J]), _INT(nv), _p(es), _INT(len(es)), _p(tri), _I64(len(tri)), _p(out))
    return out


def quad_pick(T4, rows, sgn, cols, out):
    '''out[r, c] = sgn[r] * T4[rows[r], cols[c]] of a packed block.'''
    _fn('pick', out.dtype)(_p(T4), _I64(T4.shape[1]), _p(rows), _p(sgn), _I64(len(rows)), _p(cols),
                           _I64(len(cols)), _p(out))
    return out


class Pentuples:
    '''The operands of (P) in blocks and the energy of one occupied pentuple
    at a time.'''

    def __init__(self, eris, t2, t3, t4, plan_):
        Q.check_canonical(eris)
        self.sec = sec = plan_.sectors
        self.plan = plan_
        self.nocc = no = eris.nocc
        self.nvir = nv = np.asarray(eris.fock).shape[0] - no
        self.dtype = dtype = np.result_type(t2, t3, t4, eris.oovv)
        isz = np.dtype(dtype).itemsize
        nP, nQ, nI3, nA3, nI4, nA4, nI5, nA5 = PT._sizes(no, nv)
        i2, j2 = Q._pairs(no)
        a2, b2 = Q._pairs(nv)

        def c_(x):
            return np.ascontiguousarray(x, dtype=dtype)
        mo_e = np.asarray(eris.mo_energy)
        self.pents = P.tuples(no, 5)
        self.eo5 = mo_e[:no][self.pents].sum(axis=1)
        self.ev = _c(mo_e[no:], np.float64)
        self.o41, self.o32 = P.split(no, 4, 1), P.split(no, 3, 2)
        self.o31, self.o12 = P.split(no, 3, 1), P.split(no, 1, 2)
        self.o13, self.o22 = P.split(no, 1, 3), P.split(no, 2, 2)
        self.T3 = c_(np.asarray(t3).reshape(nI3, nA3))
        self.T4 = c_(np.asarray(t4).reshape(nI4, nA4))
        with timer('ps.blocks'):
            # the layouts of the dense code (zccsdt.cc.pentuples)
            x = np.asarray(eris.oovv)[i2, j2][:, a2, b2]                    # <de||lm> as [(lm), (de)]
            self.vvoo_t = c_(np.conj(x, out=x))
            x = np.asarray(eris.ovvv)[:, :, a2, b2]                         # <de||fm> as [m, f, (de)]
            self.vvvo_t = c_(np.negative(np.conj(x, out=x), out=x))
            x = np.asarray(eris.ooov)[i2, j2]                               # <ne||lm> as [(lm), n, e]
            self.ovoo_t = c_(np.conj(x, out=x))
            del x
            self.t2_xp = c_(t2[:, :, a2, b2])                               # [i, n, (ab)]
            self.tfc = c_(t2[i2, j2])                                       # [(ij), f, c]
        self.ovvo = ovvo = np.asarray(eris.ovvo)
        with timer('ps.y4'):
            qp = Q.explicit_plan(no, nv, isz, -(-nQ // PASSES), -(-nA3 // PASSES), 3, 1)
            self.Y4 = Q.second_order_quadruples(eris, t2, qp, np.empty((nI4, nA4), dtype=dtype))
        with timer('ps.xo'):
            # Xo[(ijm), n, (de)]: for one n at a time G[m, (ij), (de)] and P(m/ij) of it
            self.Xo = np.zeros((nI3, no, nQ), dtype=dtype)
            oooo_t = c_(np.asarray(eris.oooo)[:, :, i2, j2].transpose(0, 2, 1))         # [n, (ij), o]
            t2_pc = self.tfc.reshape(nP * nv, nv)                                       # [(ij) d, f]
            G = np.zeros((no, nP, nQ), dtype=dtype)
            for n in range(no):
                for m in range(no):
                    # sum_f t(ij; d f) <ne||fm> as [(ij), d, e], then d < e of its antisymmetric part
                    K = gemm(t2_pc, c_(ovvo[n, :, :, m].T)).reshape(nP, nv, nv)
                    np.subtract(K[:, a2, b2], K[:, b2, a2], out=G[m])
                    gemm(oooo_t[n], self.t2_xp[:, m], G[m], beta=1.0)                   # + sum_o <no||ij> t(om; de)
                self.Xo[:, n] = P.fold(G, self.o12, 0, 1)
            del G, K, oooo_t, t2_pc
        with timer('ps.xv'):
            self.vpp = vvvv_packed(eris, t3pack.triple_map(nv))[0]                      # [(bc), (gf)]
            self._m3 = np.empty(max(sec.xv_sizes()[1], 1), dtype=dtype)
            self.Xv = [self.xv_blocks(p) for p in range(nP)]
            sec._w.clear()
            self.vpp = self._m3 = None
        with timer('ps.xq'):
            # W(n, o; klm; c) = P(m/kl) sum_f <no||fm> t(kl; fc), as [(klm), o, n, c]
            oovo = -np.asarray(eris.ooov).transpose(0, 1, 3, 2)                         # <no||fm>
            G = np.einsum('nofm,Pfc->mPonc', oovo, self.tfc, optimize=True)
            self.W1 = c_(P.fold(G, self.o12, 0, 1))
            del G, oovo
            self.ovvv = np.asarray(eris.ovvv)
            self._v = {}
            sz = sec.xq_sizes()
            self._m3, self._z = np.empty(max(sz[1], 1), dtype=dtype), np.empty(max(sz[2], 1), dtype=dtype)
            self.Xq = [self.xq_blocks(J) for J in range(nI4)]
            self.W1 = self.ovvv = self._v = self._m3 = self._z = None
        # the blocks and the work arrays of one pentuple
        self.R23, self.Z23 = (np.empty(max(plan_.n23, 1), dtype=dtype) for _ in range(2))
        self.R14, self.Z14 = (np.empty(max(plan_.n14, 1), dtype=dtype) for _ in range(2))
        self.W = np.empty(max(plan_.work, 1), dtype=dtype)
        self.e = np.zeros(1)

    # ---- Xv(g; ij; abc) = P(a/bc) [ sum_f <bc||gf> t(ij; fa) + P(ij) sum_n <na||gj> t(in; bc) ]

    def xv_blocks(self, p):
        '''{label of g: Xv[g, (abc)]} for the occupied pair p: the g of that
        label and the triples of the label that goes with it.'''
        sec, dtype, ovvo = self.sec, self.dtype, self.ovvo
        i, j = (int(x) for x in P.tuples(self.nocc, 2)[p])
        t = self.tfc[p]                                                     # [f, a]
        fn = getattr(P._lib, 'zc_xv_sector_' + P._SFX[np.dtype(dtype)])
        out = {}
        for kg in sec.vlab:
            lay = sec.xv_layout(int(sec.kpair[p]), kg)
            if lay is None:
                continue
            T, todo, n, off, nd = lay
            G = sec.V[kg]
            for q2, ka, o in todo:
                rq, A = sec.Q2[q2], sec.V[ka]
                blk = self._m3[o:o + len(rq) * len(G) * len(A)]
                first = True
                kf = int(sec.red(q2 - kg))
                if kf in sec.V:
                    W = sec.w_block(self.vpp, q2, kg)                       # [(bc) g, f]
                    gemm(W, _c(t[np.ix_(sec.V[kf], A)]), blk.reshape(len(rq) * len(G), len(A)), alpha=2.0)
                    first = False
                for x, y, sg in ((i, j, 1.0), (j, i, -1.0)):
                    kn = int(sec.red(q2 - sec.ko[x]))
                    if kn in sec.O:
                        ns = sec.O[kn]
                        L = _c(self.t2_xp[x][np.ix_(ns, rq)].T)             # [(ab), n]
                        R = _c(ovvo[:, :, :, y][np.ix_(ns, A, G)].transpose(0, 2, 1), dtype)    # [n, g, c]
                        gemm(L, R.reshape(len(ns), len(G) * len(A)), blk.reshape(len(rq), len(G) * len(A)),
                             alpha=2.0 * sg, beta=0.0 if first else 1.0)
                        first = False
                if first:
                    blk[...] = 0
            tri = sec.tri[T]
            res = np.empty((len(G), len(tri)), dtype=dtype)
            fn(_INT(sec.nvir), _p(sec.kv32), _INT(sec.modulus), _p(tri), _I64(len(tri)), _INT(len(G)),
               _INT(sec.qmin), _INT(sec.nq), _p(off), _p(nd), _p(sec.posV), _p(sec.posQ), _p(self._m3), _p(res))
            out[kg] = res
        return out

    # ---- Xq(n; jklm; abc) = P(c/ab) [   1/2 P(j/klm) sum_o t(jo; ab) W(n, o; klm; c)
    #                                    -     P(jk/lm), j first:  sum_fg <cn||fg> t(jk; fa) t(lm; gb) + (a <-> b) ]

    def v_block(self, kf, kn, kc, kg):
        '''<cn||fg> as [f, (n c g)] for the classes of the four indices, kept.'''
        key = (kf, kn, kg)
        if key not in self._v:
            sec = self.sec
            x = self.ovvv[np.ix_(sec.O[kn], sec.V[kc], sec.V[kf], sec.V[kg])].transpose(2, 0, 1, 3)
            self._v[key] = np.ascontiguousarray(-x, dtype=self.dtype).reshape(len(sec.V[kf]), -1)
        return self._v[key]

    def xq_blocks(self, J):
        '''{label of n: Xq[n, (abc)]} for the occupied quadruple J: the n of
        that label and the triples of the label that goes with it.'''
        sec, dtype, o13, o22, tfc = self.sec, self.dtype, self.o13, self.o22, self.tfc
        fn = getattr(P._lib, 'zc_xv_sector_' + P._SFX[np.dtype(dtype)])
        fn2 = _fn('xq_sector', dtype)
        out = {}
        for kn in sec.olab:
            lay = sec.xq_layout(int(sec.k4[J]), kn)
            if lay is None:
                continue
            T, todo, n, off, nd, todo2, n2, off2, ncs, nbs = lay
            ns = sec.O[kn]
            nn = len(ns)
            self._m3[:n] = 0
            for s in range(4):                                              # (j|klm)
                j, J3, sg = int(o13.idx1[J, s]), int(o13.idx2[J, s]), float(o13.sgn[J, s])
                for q2, kc, o in todo:
                    ko_ = int(sec.red(q2 - sec.ko[j]))
                    if ko_ in sec.O:
                        os_, rq, C = sec.O[ko_], sec.Q2[q2], sec.V[kc]
                        L = _c(self.t2_xp[j][np.ix_(os_, rq)].T)            # [(ab), o]
                        R = _c(self.W1[J3][np.ix_(os_, ns, C)]).reshape(len(os_), nn * len(C))
                        gemm(L, R, self._m3[o:o + len(rq) * nn * len(C)].reshape(len(rq), nn * len(C)),
                             alpha=sg, beta=1.0)
            tri = sec.tri[T]
            res = np.empty((nn, len(tri)), dtype=dtype)
            fn(_INT(sec.nvir), _p(sec.kv32), _INT(sec.modulus), _p(tri), _I64(len(tri)), _INT(nn),
               _INT(sec.qmin), _INT(sec.nq), _p(off), _p(nd), _p(sec.posV), _p(sec.posQ), _p(self._m3), _p(res))
            self._z[:n2] = 0
            for s in range(3):                                              # (jk|lm), j in the first pair
                pj, pl_, sg = int(o22.idx1[J, s]), int(o22.idx2[J, s]), float(o22.sgn[J, s])
                for ka, kb, kc, o in todo2:
                    kf, kg = int(sec.red(sec.kpair[pj] - ka)), int(sec.red(sec.kpair[pl_] - kb))
                    if kf not in sec.V or kg not in sec.V:
                        continue
                    A, B, F, G = sec.V[ka], sec.V[kb], sec.V[kf], sec.V[kg]
                    U = gemm(_c(tfc[pj][np.ix_(F, A)].T), self.v_block(kf, kn, kc, kg))        # [a, (n c g)]
                    m = len(A) * nn * len(sec.V[kc])
                    gemm(U.reshape(m, len(G)), _c(tfc[pl_][np.ix_(G, B)]),
                         self._z[o:o + m * len(B)].reshape(m, len(B)), alpha=-sg, beta=1.0)
                    del U
            fn2(_INT(sec.nvir), _p(sec.kv32), _INT(sec.kmin), _p(sec.lid_of), _INT(len(sec.vlab)), _p(tri),
                _I64(len(tri)), _INT(nn), _p(off2), _p(ncs), _p(nbs), _p(sec.posV), _p(self._z), _DBL(1.0),
                _p(res))
            out[kn] = res
        return out

    # ---- one occupied pentuple

    def _work(self, nr, nc):
        return self.W[:nr * nc].reshape(nr, nc)

    def energy(self, I):
        '''Adds the energy of the occupied pentuple I to self.e.'''
        sec, dtype, nv = self.sec, self.dtype, self.nvir
        ko, kpair, k3 = sec.ko, sec.kpair, sec.k3
        o41, o32, o31, o12 = self.o41, self.o32, self.o31, self.o12
        T3, T4, Y4 = self.T3, self.T4, self.Y4
        M = int(sec.k5[I])
        b23, _, b14, _, off23, nc23, off14, nc14 = sec.layout5(M)
        J4s, xs, s41 = o41.idx1[I], o41.idx2[I], o41.sgn[I]
        J3s, ps, s32 = o32.idx1[I], o32.idx2[I], o32.sgn[I]
        with timer('ps.23'):
            for q, off, nr, nc in b23:
                T = int(sec.red(M - q))
                rq, ca, tri = sec.Q2[q], sec.A3[T], sec.tri[T]
                R = self.R23[off:off + nr * nc].reshape(nr, nc)
                Z = self.Z23[off:off + nr * nc].reshape(nr, nc)
                P.zero(R)
                P.zero(Z)
                for s in range(5):
                    x, J, sg = int(xs[s]), int(J4s[s]), float(s41[s])
                    kf = int(sec.red(q - ko[x]))
                    if kf in sec.V:             # (ijkl|m): <de||fm> t(ijkl; abc f) = - <de||fm> t(ijkl; f abc)
                        E = sec.V[kf]
                        L = _c(self.vvvo_t[x][np.ix_(E, rq)].T)
                        W = self._work(len(E), nc)
                        for amp, X in ((T4, R), (Y4, Z)):
                            gemm(L, quad_rows13(amp, J, nv, sec.V32[kf], tri, W), X, alpha=-sg, beta=1.0)
                    if kf in sec.O:             # (i|jklm): t(in; de) Xq(n; jklm; abc)
                        L = _c(self.t2_xp[x][np.ix_(sec.O[kf], rq)].T)
                        gemm(L, self.Xq[J][kf], R, alpha=sg, beta=1.0)
                for s in range(10):
                    J3, pp, sg = int(J3s[s]), int(ps[s]), float(s32[s])
                    kn = int(sec.red(k3[J3] - q))
                    if kn in sec.O:             # (ijm|kl): Xo(n; ijm; de) t(n kl; abc)
                        ns = sec.O[kn]
                        sn = o12.psg[ns, pp]
                        ns = ns[sn != 0]
                        if len(ns):
                            L = _c(self.Xo[J3][np.ix_(ns, rq)].T)
                            W = self._work(len(ns), nc)
                            quad_pick(T3, _i64(o12.pos[ns, pp]), _c(sn[sn != 0], np.float64), ca, W)
                            gemm(L, W, R, alpha=sg, beta=1.0)
                    if kn in sec.V:             # (klm|ij): t(klm; g de) Xv(g; ij; abc)
                        col, s3 = sec.t3_rows(kn, q)
                        L = _c((T3[J3][col] * s3).T)
                        gemm(L, self.Xv[pp][kn], R, alpha=sg, beta=1.0)
                    if kpair[pp] == q:          # (ijk|lm): t(ijk; abc) <de||lm>
                        gemm(_c(self.vvoo_t[pp][rq][:, None]), _c(T3[J3][ca][None, :]), Z, alpha=sg, beta=1.0)
        with timer('ps.14'):
            for pe, off, nr, nc in b14:
                va, ca = sec.V[pe], sec.A4[int(sec.red(M - pe))]
                rows, Ls = [], []
                for s in range(10):             # (ijk|lm): - <ne||lm> t(ijk n; abcd)
                    J3, pp, sg = int(J3s[s]), int(ps[s]), float(s32[s])
                    kn = int(sec.red(kpair[pp] - pe))
                    if kn in sec.O:
                        ns = sec.O[kn]
                        sn = o31.psg[J3, ns]
                        ns, sn = ns[sn != 0], sn[sn != 0]
                        if len(ns):
                            rows.append(o31.pos[J3, ns])
                            Ls.append((-sg * sn)[:, None] * self.ovoo_t[pp][np.ix_(ns, va)])
                R = self.R14[off:off + nr * nc].reshape(nr, nc)
                Z = self.Z14[off:off + nr * nc].reshape(nr, nc)
                if not rows:
                    P.zero(R)
                    P.zero(Z)
                    continue
                # (the rows of T4 that the ten splits share: each once)
                rows, inv = np.unique(np.concatenate(rows), return_inverse=True)
                L = np.zeros((len(rows), nr), dtype=dtype)
                np.add.at(L, inv, np.concatenate(Ls))
                L = _c(L.T)
                rows, one = _i64(rows), np.ones(len(rows))
                step = max(1, len(self.W) // len(rows))
                for c0 in range(0, nc, step):
                    c1 = min(nc, c0 + step)
                    W = self._work(len(rows), c1 - c0)
                    for amp, X in ((T4, R), (Y4, Z)):
                        gemm(L, quad_pick(amp, rows, one, ca[c0:c1], W), X[:, c0:c1])
        with timer('ps.sum'):
            _fn('sector_energy', dtype)(
                _INT(nv), _p(sec.kv32), _INT(sec.modulus), _INT(M), _p(self.ev), _DBL(float(self.eo5[I])),
                _INT(sec.kmin), _INT(sec.nk), _p(sec.lid_of), _p(sec.lstart), _p(sec.lmem),
                _INT(sec.qmin), _INT(sec.nq), _p(off23), _p(nc23), _p(off14), _p(nc14),
                _p(sec.posV), _p(sec.posQ), _p(sec.pos3), _p(sec.pos4), _p(self.R23), _p(self.Z23),
                _p(self.R14), _p(self.Z14), _p(self.e))


def kernel(eris, t2, t3, t4, max_memory=None, plan_=None):
    '''E(P) as zccsdt.cc.pentuples.kernel, in the symmetry sectors of
    ``eris.labels``.'''
    no = eris.nocc
    nv = np.asarray(eris.fock).shape[0] - no
    if no < 5 or nv < 5:
        return 0.0
    if plan_ is None:
        from zccsdt import memory
        isz = np.dtype(np.result_type(t2, t3, t4, eris.oovv)).itemsize
        plan_ = plan(eris, isz, memory.budget(max_memory))
    p = Pentuples(eris, t2, t3, t4, plan_)
    for I in range(len(p.pents)):
        p.energy(I)
    return float(p.e[0])
