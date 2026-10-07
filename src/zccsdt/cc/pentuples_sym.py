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
        self.pr32 = {k: _c(self.pairs[x], np.int32) for k, x in self.Q2.items()}
        # the pairs by label, for the kernel
        self.qlen = np.zeros(self.nq, dtype=np.int64)
        self.qlen[np.array(self.qlab) - self.qmin] = [len(self.Q2[k]) for k in self.qlab]
        self.qstart = np.concatenate(([0], np.cumsum(self.qlen)[:-1])).astype(np.int64)
        self.qprs = _c(np.concatenate([self.pr32[k] for k in self.qlab]))
        self.olab = sorted(self.O)
        self.kpair = self.red(self.ko[P.tuples(no, 2)].sum(axis=1))
        self.k3 = self.red(self.ko[P.tuples(no, 3)].sum(axis=1))
        self.k4 = self.red(self.ko[P.tuples(no, 4)].sum(axis=1))
        self.k5 = self.red(self.ko[P.tuples(no, 5)].sum(axis=1))
        self.lid = {k: l for l, k in enumerate(self.vlab)}
        self._layout5, self._xq_layout = {}, {}

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

    def u_rows(self, kjk, kn, kg):
        """The rows (a n c) of U(jk; a n c g) = sum_f t(jk; f a) <cn||fg> for
        the n of label ``kn``, the g of label ``kg`` and a pair (jk) of label
        ``kjk``, one class of a after the other:
        ([(label of a, label of c, first row, rows)], rows)."""
        rows, n = [], 0
        for ka in self.vlab:
            kc = int(self.red(kjk - ka + kg - kn))
            if kc in self.V:
                m = len(self.V[ka]) * len(self.O[kn]) * len(self.V[kc])
                rows.append((ka, kc, n, m))
                n += m
        return rows, n

    def xq_layout(self, kJ, kn):
        """The products that Xq(n; jklm; abc) is made from, for the n of
        label ``kn`` and a quadruple of label ``kJ``: (label of the triples;
        {label of (ab): offset}, size, offsets and widths of the blocks
        [(ab), n, c] - the term with one integral <oo||vo> -;
        [(label of b, offset, rows)], size, offsets and widths of the blocks
        [a, n, c, b] by the labels of a and b - the term with <vo||vv>: for
        one label of b the rows (a n c) of u_rows()); None if there are no
        such triples."""
        key = (kJ, kn)
        if key in self._xq_layout:
            return self._xq_layout[key]
        T = int(self.red(kJ - kn))
        if T not in self.A3:
            self._xq_layout[key] = None
            return None
        nn = len(self.O[kn])
        where, n = {}, 0
        off, nd = np.full(self.nq, -1, dtype=np.int64), np.zeros(self.nq, dtype=np.int64)
        for q2 in self.qlab:
            kc = int(self.red(T - q2))
            if kc in self.V:
                where[q2] = n
                off[q2 - self.qmin], nd[q2 - self.qmin] = n, len(self.V[kc])
                n += len(self.Q2[q2]) * nn * len(self.V[kc])
        nl = len(self.vlab)
        zrows, n2 = [], 0
        off2 = np.full(nl * nl, -1, dtype=np.int64)
        ncs, nbs = np.zeros(nl * nl, dtype=np.int64), np.zeros(nl * nl, dtype=np.int64)
        for lb, kb in enumerate(self.vlab):
            nb, m = len(self.V[kb]), 0
            for la, ka in enumerate(self.vlab):
                kc = int(self.red(T - ka - kb))
                if kc in self.V:
                    off2[la * nl + lb], ncs[la * nl + lb], nbs[la * nl + lb] = n2 + m * nb, len(self.V[kc]), nb
                    m += len(self.V[ka]) * nn * len(self.V[kc])
            if m:
                zrows.append((kb, n2, m))
                n2 += m * nb
        self._xq_layout[key] = (T, where, n, off, nd, zrows, n2, off2, ncs, nbs)
        return self._xq_layout[key]

    def xq_sizes(self):
        """Elements of the blocks of Xq for all occupied quadruples, of the
        largest sets of products that one of them is made from (the two
        terms), of U for the pairs of one smallest index, and of the blocks
        of <vo||vv>."""
        total = work = work2 = 0
        for kJ, count in zip(*np.unique(self.k4, return_counts=True)):
            for kn in self.olab:
                lay = self.xq_layout(int(kJ), kn)
                if lay is not None:
                    total += int(count) * len(self.O[kn]) * len(self.A3[lay[0]])
                    work, work2 = max(work, lay[2]), max(work2, lay[6])
        usz = {int(k): sum(self.u_rows(int(k), kn, kg)[1] * len(self.V[kg]) for kn in self.olab for kg in self.vlab)
               for k in np.unique(self.kpair)}
        pairs = P.tuples(self.nocc, 2)
        u = max(sum(usz[int(k)] for k in self.kpair[pairs[:, 0] == j]) for j in range(self.nocc - 1))
        ints = 0
        for kn in self.olab:
            for kf in self.vlab:
                for kg in self.vlab:
                    kc = int(self.red(kf + kg - kn))
                    if kc in self.V:
                        ints += len(self.O[kn]) * len(self.V[kf]) * len(self.V[kg]) * len(self.V[kc])
        return total, work, work2, u, ints


@dataclass
class Plan:
    need: int               # bytes besides T3, T4 and the integrals
    sectors: object         # Sectors
    n23: int                # elements of the largest set of blocks X[(de), (abc)]
    n14: int                # and X[e, (abcd)]
    work: int               # elements of the stacked rows of the right operand of a product
    lwork: int              # and of the left one
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
    xq, xq_work, xq_work2, xq_u, vints = sec.xq_sizes()
    maxV, maxO = (max(len(x) for x in c.values()) for c in (sec.V, sec.O))
    max3, max2 = (max(len(x) for x in c.values()) for c in (sec.A3, sec.Q2))
    rows = 15 * (maxV + maxO) + 10          # of the stacked operands of X[(de), (abc)], at most
    work, lwork = max(rows * max3, blk14), rows * max2
    parts = {
        'second-order quadruples': isz * nI4 * nA4,
        'Xo': isz * nI3 * no * nQ,
        'Xv': isz * xv,
        'Xq': isz * xq,
        'layouts': isz * (nP * nQ + no * nv * nQ + nP * no * nv + no * no * nQ + 2 * nP * nv * nv),
        'tables': (12 * nA4 + 24 * nA3 + 16 * nv * nQ + 24 * nQ + nI5 * 600 + nI4 * (no * 9 + 250)
                   + nI3 * (nP + no) * 9 + nP * nP * 9 + Q.SLACK),
    }
    kept = sum(parts.values())
    # what is there for a while: the formation of the intermediates, each with what it is made
    # of, then the blocks of a pentuple
    t_y4 = isz * nP * nQ
    t_xo = isz * (no * nP * nQ + 2 * nP * nv * nv + no * no * nP + 4 * nI3 * nQ)
    t_xv = isz * (sec.w_size() + xv_work + 2 * maxO * maxV * max(maxV, max2))
    w1 = nI3 * no * no * nv
    t_w1 = isz * (no ** 3 * nv + 2 * no ** 3 * nP * nv + 3 * w1)
    t_xq = isz * (2 * w1 + no * no * nQ + vints + no * nv * maxV * maxV + xq_work + xq_work2 + xq_u)
    t_run = isz * (2 * n23 + 2 * n14 + work + lwork + 10 * no * maxV)
    parts['blocks of a pentuple'] = t_run
    need = kept + max(t_y4, t_xo, t_xv, t_w1, t_xq, t_run)
    if budget is not None and need > budget:
        raise MemoryError('(P) in blocks of the symmetry labels needs %.0f MB besides T3, T4 and the integrals '
                          'for %d occupied and %d virtual spinors; %.0f MB are left'
                          % (need / 1e6, no, nv, budget / 1e6))
    return Plan(int(need), sec, int(n23), int(n14), int(work), int(lwork), int(kept), parts)


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
        nP, nQ, nI3, nA3, nI4, nA4, nI5, nA5 = PT._sizes(no, nv)
        i2, j2 = Q._pairs(no)
        a2, b2 = Q._pairs(nv)
        self._k = {name: _fn(name, dtype) for name in ('expand13', 'expand12', 'pick', 'copy', 'sector_energy',
                                                       'xq_sector', 'y4_row')}
        self._xv_sector = getattr(P._lib, 'zc_xv_sector_' + P._SFX[np.dtype(dtype)])

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
            # t(ij; f a) for the a of one label (the f: of the label that is left): [f, a]
            self.tblk = {(p, ka): c_(self.tfc[p][np.ix_(sec.V[kf], sec.V[ka])])
                         for p in range(nP) for ka in sec.vlab
                         for kf in [int(sec.red(sec.kpair[p] - ka))] if kf in sec.V}
        self.ovvo = ovvo = np.asarray(eris.ovvo)
        with timer('ps.y4'):
            self.Y4 = self.second_order_quadruples(mo_e)
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
            self.Xq = self.xq_all(eris)
        # the blocks and the work arrays of one pentuple
        self.R23, self.Z23 = (np.empty(max(plan_.n23, 1), dtype=dtype) for _ in range(2))
        self.R14, self.Z14 = (np.empty(max(plan_.n14, 1), dtype=dtype) for _ in range(2))
        self.L, self.R = np.empty(max(plan_.lwork, 1), dtype=dtype), np.empty(max(plan_.work, 1), dtype=dtype)
        self.e = np.zeros(1)
        self._sg, self._t3r, self._ix = {}, {}, {}

    # ---- the second-order quadruples <Q| V T2 |0> / D of the left-hand side

    def second_order_quadruples(self, mo_e):
        '''[C(no,4), C(nv,4)] as zccsdt.cc.quadruples.second_order_quadruples:
        each element that the labels allow from its 36 products
        t(ij; ab) <cd||kl>.'''
        sec, no, nv, dtype = self.sec, self.nocc, self.nvir, self.dtype
        i2, j2 = Q._pairs(no)
        a2, b2 = Q._pairs(nv)
        t2_pp = _c(self.t2_xp[i2, j2])                                      # [(ij), (ab)]
        quads = P.tuples(no, 4)
        eo4 = mo_e[:no][quads].sum(axis=1)
        out = np.empty((len(quads), comb(nv, 4)), dtype=dtype)
        o22 = self.o22
        for J in range(len(quads)):
            p1, p2 = _i64(o22.idx1[J]), _i64(o22.idx2[J])
            sgn, labels = _c(o22.sgn[J], np.float64), _c(sec.kpair[p1], np.int32)
            self._k['y4_row'](
                _INT(nv), _p(sec.kv32), _INT(sec.modulus), _INT(int(sec.k4[J])), _p(self.ev), _DBL(float(eo4[J])),
                _INT(sec.kmin), _INT(sec.nk), _p(sec.lid_of), _p(sec.lstart), _p(sec.lmem),
                _p(t2_pp), _p(self.vvoo_t), _p(p1), _p(p2), _p(sgn), _p(labels), _p(out[J]))
        return out

    # ---- Xv(g; ij; abc) = P(a/bc) [ sum_f <bc||gf> t(ij; fa) + P(ij) sum_n <na||gj> t(in; bc) ]

    def xv_blocks(self, p):
        '''{label of g: Xv[g, (abc)]} for the occupied pair p: the g of that
        label and the triples of the label that goes with it.'''
        sec, dtype, ovvo = self.sec, self.dtype, self.ovvo
        i, j = (int(x) for x in P.tuples(self.nocc, 2)[p])
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
                    gemm(W, self.tblk[p, ka], blk.reshape(len(rq) * len(G), len(A)), alpha=2.0)
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
            self._xv_sector(_INT(sec.nvir), _p(sec.kv32), _INT(sec.modulus), _p(tri), _I64(len(tri)), _INT(len(G)),
                            _INT(sec.qmin), _INT(sec.nq), _p(off), _p(nd), _p(sec.posV), _p(sec.posQ),
                            _p(self._m3), _p(res))
            out[kg] = res
        return out

    # ---- Xq(n; jklm; abc) = P(c/ab) [   1/2 P(j/klm) sum_o t(jo; ab) W(n, o; klm; c)
    #                                    -     P(jk/lm), j first:  sum_fg <cn||fg> t(jk; fa) t(lm; gb) + (a <-> b) ]

    def xq_all(self, eris):
        '''[{label of n: Xq[n, (abc)]}] for all occupied quadruples.  The
        product of <cn||fg> with the first T2, U(jk; a n c g), is the same
        for all quadruples with the pair (jk): it is formed for the pairs of
        one smallest index j at a time.'''
        sec, dtype, no = self.sec, self.dtype, self.nocc
        nI3 = comb(no, 3)

        def c_(x):
            return np.ascontiguousarray(x, dtype=dtype)
        # W(n, o; klm; c) = P(m/kl) sum_f <no||fm> t(kl; fc), [(klm), o, n, c], in the blocks of the
        # labels of o and n: [o, (n c)]
        oovo = -np.asarray(eris.ooov).transpose(0, 1, 3, 2)                         # <no||fm>
        G = np.einsum('nofm,Pfc->mPonc', oovo, self.tfc, optimize=True)
        W1 = P.fold(G, self.o12, 0, 1)
        del G, oovo
        w1 = {}
        for J3 in range(nI3):
            for ko_ in sec.olab:
                for kn in sec.olab:
                    kc = int(sec.red(sec.k3[J3] - kn - ko_))
                    if kc in sec.V:
                        os_, ns = sec.O[ko_], sec.O[kn]
                        w1[J3, ko_, kn] = c_(W1[J3][np.ix_(os_, ns, sec.V[kc])]).reshape(len(os_), -1)
        del W1
        # t(j o; ab) for the pairs of one label: [(ab), o]
        t2b = {(j, ko_): c_(self.t2_xp[j][np.ix_(sec.O[ko_], sec.Q2[q2])].T)
               for j in range(no) for ko_ in sec.olab for q2 in [int(sec.red(ko_ + sec.ko[j]))] if q2 in sec.Q2}
        # <cn||fg> as [f, (n c g)] in the blocks of the labels
        ovvv = np.asarray(eris.ovvv)
        vb = {}
        for kn in sec.olab:
            for kf in sec.vlab:
                for kg in sec.vlab:
                    kc = int(sec.red(kf + kg - kn))
                    if kc in sec.V:
                        x = ovvv[np.ix_(sec.O[kn], sec.V[kc], sec.V[kf], sec.V[kg])].transpose(2, 0, 1, 3)
                        vb[kf, kn, kg] = np.ascontiguousarray(-x, dtype=dtype).reshape(len(sec.V[kf]), -1)
        sz = sec.xq_sizes()
        self._m3, self._z = np.empty(max(sz[1], 1), dtype=dtype), np.empty(max(sz[2], 1), dtype=dtype)
        quads = P.tuples(no, 4)
        pair_of = P.ranks(no, 2)
        out = []
        for j in range(no - 3):
            U = {}
            for k in range(j + 1, no):
                p = int(pair_of[j, k])
                for kn in sec.olab:
                    for kg in sec.vlab:
                        rows = sec.u_rows(int(sec.kpair[p]), kn, kg)
                        if not rows[1]:
                            continue
                        u = np.zeros((rows[1], len(sec.V[kg])), dtype=dtype)
                        for ka, kc, o, m in rows[0]:
                            if (p, ka) in self.tblk:
                                gemm(self.tblk[p, ka].T, vb[int(sec.red(sec.kpair[p] - ka)), kn, kg],
                                     u[o:o + m].reshape(len(sec.V[ka]), -1))
                        U[p, kn, kg] = u
            for J in np.where(quads[:, 0] == j)[0]:
                out.append(self.xq_blocks(int(J), w1, t2b, U))
            del U
        self._m3 = self._z = None
        return out

    def xq_blocks(self, J, w1, t2b, U):
        '''{label of n: Xq[n, (abc)]} for the occupied quadruple J: the n of
        that label and the triples of the label that goes with it.'''
        sec, dtype, o13, o22 = self.sec, self.dtype, self.o13, self.o22
        out = {}
        for kn in sec.olab:
            lay = sec.xq_layout(int(sec.k4[J]), kn)
            if lay is None:
                continue
            T, where, n, off, nd, zrows, n2, off2, ncs, nbs = lay
            nn = len(sec.O[kn])
            self._m3[:n] = 0
            for s in range(4):                                              # (j|klm)
                j, J3, sg = int(o13.idx1[J, s]), int(o13.idx2[J, s]), float(o13.sgn[J, s])
                for ko_ in sec.olab:
                    at = where.get(int(sec.red(ko_ + sec.ko[j])))
                    if at is not None and (J3, ko_, kn) in w1:
                        L = t2b[j, ko_]                                     # [(ab), o]
                        R = w1[J3, ko_, kn]                                 # [o, (n c)]
                        gemm(L, R, self._m3[at:at + L.shape[0] * R.shape[1]].reshape(L.shape[0], R.shape[1]),
                             alpha=sg, beta=1.0)
            tri = sec.tri[T]
            res = np.empty((nn, len(tri)), dtype=dtype)
            self._xv_sector(_INT(sec.nvir), _p(sec.kv32), _INT(sec.modulus), _p(tri), _I64(len(tri)), _INT(nn),
                            _INT(sec.qmin), _INT(sec.nq), _p(off), _p(nd), _p(sec.posV), _p(sec.posQ),
                            _p(self._m3), _p(res))
            self._z[:n2] = 0
            for s in range(3):                                              # (jk|lm), j in the first pair
                pj, pl_, sg = int(o22.idx1[J, s]), int(o22.idx2[J, s]), float(o22.sgn[J, s])
                for kb, o, m in zrows:
                    u = U.get((pj, kn, int(sec.red(sec.kpair[pl_] - kb))))
                    if u is not None and (pl_, kb) in self.tblk:
                        tb = self.tblk[pl_, kb]                             # [g, b]
                        gemm(u, tb, self._z[o:o + m * tb.shape[1]].reshape(m, tb.shape[1]), alpha=-sg, beta=1.0)
            self._k['xq_sector'](
                _INT(sec.nvir), _p(sec.kv32), _INT(sec.kmin), _p(sec.lid_of), _INT(len(sec.vlab)), _p(tri),
                _I64(len(tri)), _INT(nn), _p(off2), _p(ncs), _p(nbs), _p(sec.posV), _p(self._z), _DBL(1.0),
                _p(res))
            out[kn] = res
        return out

    # ---- one occupied pentuple

    def _signs(self, n, sg):
        if (n, sg) not in self._sg:
            self._sg[n, sg] = np.full(n, sg)
        return self._sg[n, sg]

    def _index(self, i):
        if i not in self._ix:
            self._ix[i] = np.array([i], dtype=np.int64)
        return self._ix[i]

    def _t3_rows(self, kn, pp):
        '''(rows, signs) of t(n kl; abc) in the packed T3 for the n of label
        ``kn`` and the occupied pair ``pp`` (sign 0: n is one of k, l).'''
        if (kn, pp) not in self._t3r:
            ns = self.sec.O[kn]
            self._t3r[kn, pp] = (_i64(self.o12.pos[ns, pp]), _c(self.o12.psg[ns, pp], np.float64))
        return self._t3r[kn, pp]

    def energy(self, I):
        '''Adds the energy of the occupied pentuple I to self.e.'''
        sec, dtype, nv, K = self.sec, self.dtype, self.nvir, self._k
        ko, kpair, k3 = sec.ko, sec.kpair, sec.k3
        o41, o32, o31 = self.o41, self.o32, self.o31
        T3, T4, Y4 = self.T3, self.T4, self.Y4
        nA3 = T3.shape[1]
        M = int(sec.k5[I])
        b23, _, b14, _, off23, nc23, off14, nc14 = sec.layout5(M)
        J4s, xs, s41 = [[int(x) for x in a] for a in (o41.idx1[I], o41.idx2[I], o41.sgn[I])]
        J3s, ps, s32 = [[int(x) for x in a] for a in (o32.idx1[I], o32.idx2[I], o32.sgn[I])]
        pick, copy = K['pick'], K['copy']
        with timer('ps.23'):
            for q, off, nr, nc in b23:
                T = int(sec.red(M - q))
                rq, ca, tri, prs = sec.Q2[q], sec.A3[T], sec.tri[T], sec.pr32[q]
                p_rq, p_ca, i_nr, i_nc = _p(rq), _p(ca), _I64(nr), _I64(nc)
                # the rows of the stacked operands: [V,T4] | T2 with Xq | Xo with T3 | T3 with Xv
                kfs = [int(sec.red(q - ko[x])) for x in xs]
                kns = [int(sec.red(k3[J3] - q)) for J3 in J3s]
                segs, n = [], 0
                for kind, labels, cls in ((0, kfs, sec.V), (1, kfs, sec.O), (2, kns, sec.O), (3, kns, sec.V)):
                    for s, k in enumerate(labels):
                        if k in cls:
                            segs.append((kind, s, k, n, len(cls[k])))
                            n += len(cls[k])
                    if kind == 0:
                        na = n
                X = self.R23[off:off + nr * nc].reshape(nr, nc)
                Z = self.Z23[off:off + nr * nc].reshape(nr, nc)
                L, R = self.L[:n * nr].reshape(n, nr), self.R[:n * nc].reshape(n, nc)
                for kind, s, k, at, m in segs:
                    l_, r_, i_m = ctypes.c_void_p(L.ctypes.data + at * L.strides[0]), \
                        ctypes.c_void_p(R.ctypes.data + at * R.strides[0]), _I64(m)
                    if kind == 0:               # (ijkl|m): <de||fm> t(ijkl; abc f) = - <de||fm> t(ijkl; f abc)
                        pick(_p(self.vvvo_t[xs[s]]), _I64(self.vvvo_t.shape[2]), _p(sec.V[k]),
                             _p(self._signs(m, -float(s41[s]))), i_m, p_rq, i_nr, l_)
                        K['expand13'](_p(T4[J4s[s]]), _INT(nv), _p(sec.V32[k]), _INT(m), _p(tri), i_nc, _DBL(1.0), r_)
                    elif kind == 1:             # (i|jklm): t(in; de) Xq(n; jklm; abc)
                        pick(_p(self.t2_xp[xs[s]]), _I64(self.t2_xp.shape[2]), _p(sec.O[k]),
                             _p(self._signs(m, float(s41[s]))), i_m, p_rq, i_nr, l_)
                        copy(_p(self.Xq[J4s[s]][k]), _I64(m * nc), _DBL(1.0), r_)
                    elif kind == 2:             # (ijm|kl): Xo(n; ijm; de) t(n kl; abc)
                        pick(_p(self.Xo[J3s[s]]), _I64(self.Xo.shape[2]), _p(sec.O[k]),
                             _p(self._signs(m, float(s32[s]))), i_m, p_rq, i_nr, l_)
                        rows, sgn = self._t3_rows(k, ps[s])
                        pick(_p(T3), _I64(nA3), _p(rows), _p(sgn), i_m, p_ca, i_nc, r_)
                    else:                       # (klm|ij): t(klm; g de) Xv(g; ij; abc)
                        K['expand12'](_p(T3[J3s[s]]), _INT(nv), _p(sec.V32[k]), _INT(m), _p(prs), i_nr,
                                      _DBL(float(s32[s])), l_)
                        copy(_p(self.Xv[ps[s]][k]), _I64(m * nc), _DBL(1.0), r_)
                if n:
                    gemm(L.T, R, X)
                else:
                    P.zero(X)
                # z: the first rows with the second-order quadruples, and t(ijk; abc) <de||lm>
                for kind, s, k, at, m in segs:
                    if kind == 0:
                        K['expand13'](_p(Y4[J4s[s]]), _INT(nv), _p(sec.V32[k]), _INT(m), _p(tri), i_nc, _DBL(1.0),
                                      ctypes.c_void_p(R.ctypes.data + at * R.strides[0]))
                n = na
                for s in range(10):
                    if kpair[ps[s]] == q:
                        pick(_p(self.vvoo_t), _I64(self.vvoo_t.shape[1]), _p(self._index(ps[s])),
                             _p(self._signs(1, float(s32[s]))), _I64(1), p_rq, i_nr,
                             ctypes.c_void_p(self.L.ctypes.data + n * nr * self.L.itemsize))
                        pick(_p(T3), _I64(nA3), _p(self._index(J3s[s])), _p(self._signs(1, 1.0)), _I64(1), p_ca, i_nc,
                             ctypes.c_void_p(self.R.ctypes.data + n * nc * self.R.itemsize))
                        n += 1
                if n:
                    gemm(self.L[:n * nr].reshape(n, nr).T, self.R[:n * nc].reshape(n, nc), Z)
                else:
                    P.zero(Z)
        with timer('ps.14'):
            for pe, off, nr, nc in b14:
                va, ca = sec.V[pe], sec.A4[int(sec.red(M - pe))]
                rows, Ls = [], []
                for s in range(10):             # (ijk|lm): - <ne||lm> t(ijk n; abcd)
                    J3, pp, sg = J3s[s], ps[s], float(s32[s])
                    kn = int(sec.red(kpair[pp] - pe))
                    if kn in sec.O:
                        ns = sec.O[kn]
                        sn = o31.psg[J3, ns]
                        ns, sn = ns[sn != 0], sn[sn != 0]
                        if len(ns):
                            rows.append(o31.pos[J3, ns])
                            Ls.append((-sg * sn)[:, None] * self.ovoo_t[pp][np.ix_(ns, va)])
                X = self.R14[off:off + nr * nc].reshape(nr, nc)
                Z = self.Z14[off:off + nr * nc].reshape(nr, nc)
                if not rows:
                    P.zero(X)
                    P.zero(Z)
                    continue
                # (the rows of T4 that the ten splits share: each once)
                rows, inv = np.unique(np.concatenate(rows), return_inverse=True)
                L = np.zeros((len(rows), nr), dtype=dtype)
                np.add.at(L, inv, np.concatenate(Ls))
                L = _c(L.T)
                rows, one = _i64(rows), np.ones(len(rows))
                step = max(1, len(self.R) // len(rows))
                for c0 in range(0, nc, step):
                    c1 = min(nc, c0 + step)
                    W = self.R[:len(rows) * (c1 - c0)].reshape(len(rows), c1 - c0)
                    for amp, out in ((T4, X), (Y4, Z)):
                        gemm(L, quad_pick(amp, rows, one, ca[c0:c1], W), out[:, c0:c1])
        with timer('ps.sum'):
            K['sector_energy'](
                _INT(nv), _p(sec.kv32), _INT(sec.modulus), _INT(M), _p(self.ev), _DBL(float(self.eo5[I])),
                _INT(sec.kmin), _INT(sec.nk), _p(sec.lid_of), _INT(len(sec.vlab)), _p(sec.lstart), _p(sec.lmem),
                _INT(sec.qmin), _INT(sec.nq), _p(sec.qstart), _p(sec.qlen), _p(sec.qprs),
                _p(off23), _p(nc23), _p(off14), _p(nc14),
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
