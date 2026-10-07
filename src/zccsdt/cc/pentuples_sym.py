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

for r and for z each.  The blocks of one occupied pentuple are matrix
products on the rows and columns that the labels allow - the rows of all
splits and terms of a block stacked, as in the dense code -, and one kernel
(zc_p_sector_energy, csrc/psym.c) walks through the virtual pentuples of the
sector, sums the ten and the five splits of each from the blocks and adds
the energy: nothing of the size of the virtual pentuples is stored, and no
tables of their size are made.

The intermediates with virtual triples are in blocks too, formed from blocks
of T2 and of the integrals:

    Xv(g; ij; abc)      for the g of one label the triples of one label
    Xq(n; jklm; abc)    for the n of one label the triples of one label

(zc_xv_sector of csrc/tnpack.c and zc_p_xq_sector reduce the products onto
the triples), and of the second-order quadruples of the left-hand side only
the elements that the labels allow are evaluated (zc_p_y4_row).

T3 and T4 are the packed blocks of the dense code, and the second-order
quadruples are stored like T4; what a product needs of them is read where
it is (quad_row(), quad_pick()).  Xo(n; ijm; de) and the layouts of T2
and of the integrals with two virtual indices are those of the dense code.

Memory (plan()): all of this for one plan - where it does not fit,
MemoryError, and the dense code has plans for less (batches of columns,
scratch files).  With one label the blocks are the dense arrays, and r and z
on two groups of indices take more than the dense code's on the pentuples.
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
        self._layout5, self._xq = {}, {}

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

    # Xq(n; jklm; abc) is the sum of two products that are reduced onto the triples,
    #     G[(ab), n, c] = sum_o t(jo; ab) W(n, o; klm; c)                    (4 splits of jklm)
    #     Z[a, n, c, b] = sum_g U(jk; a n c g) t(lm; g b),  U = sum_f t(jk; f a) <cn||fg>   (3 splits)
    # for all labels of n in one matrix product each: the columns (n c) and the rows (a n c)
    # with the n of one label after those of another.

    def xq_cols(self, S):
        """The columns (n c) with the labels of n and c adding up to ``S``:
        ({label of n: (first column, label of c)}, columns)."""
        if ('c', S) not in self._xq:
            cols, n = {}, 0
            for kn in self.olab:
                kc = int(self.red(S - kn))
                if kc in self.V:
                    cols[kn] = (n, kc)
                    n += len(self.O[kn]) * len(self.V[kc])
            self._xq['c', S] = (cols, n)
        return self._xq['c', S]

    def xq_rows(self, S):
        """The rows (a n c) with the labels of a, n and c adding up to ``S``,
        the a of one label together and for each of them the columns
        xq_cols(S - label of a): ([(label of a, first row, rows for one a,
        columns)], rows)."""
        if ('r', S) not in self._xq:
            rows, n = [], 0
            for ka in self.vlab:
                cols, m = self.xq_cols(int(self.red(S - ka)))
                if m:
                    rows.append((ka, n, m, cols))
                    n += len(self.V[ka]) * m
            self._xq['r', S] = (rows, n)
        return self._xq['r', S]

    def xq_layout(self, kJ):
        """The two products for an occupied quadruple of label ``kJ``:
        [(label of (ab), offset, columns)] and the size of G,
        [(label of b, offset, rows)] and the size of Z, and for every label
        of n that has triples (those of label kJ - kn) the arrays of the
        kernel: {label of n: (label of the triples, offsets, strides and
        widths of the blocks of G and of Z)}."""
        if kJ in self._xq:
            return self._xq[kJ]
        gb, n = [], 0
        for q2 in self.qlab:
            m = self.xq_cols(int(self.red(kJ - q2)))[1]
            if m:
                gb.append((q2, n, m))
                n += len(self.Q2[q2]) * m
        zb, n2 = [], 0
        for kb in self.vlab:
            m = self.xq_rows(int(self.red(kJ - kb)))[1]
            if m:
                zb.append((kb, n2, m))
                n2 += m * len(self.V[kb])
        nl, kern = len(self.vlab), {}
        for kn in self.olab:
            T = int(self.red(kJ - kn))
            if T not in self.A3:
                continue
            goff, gst, gnd = (np.full(self.nq, x, dtype=np.int64) for x in (-1, 0, 0))
            for q2, at, m in gb:
                cols = self.xq_cols(int(self.red(kJ - q2)))[0]
                if kn in cols:
                    goff[q2 - self.qmin], gst[q2 - self.qmin] = at + cols[kn][0], m
                    gnd[q2 - self.qmin] = len(self.V[cols[kn][1]])
            off, ast, ncs, nbs = (np.full(nl * nl, x, dtype=np.int64) for x in (-1, 0, 0, 0))
            for kb, at, _ in zb:
                lb, nb = self.vlab.index(kb), len(self.V[kb])
                for ka, first, m, cols in self.xq_rows(int(self.red(kJ - kb)))[0]:
                    if kn in cols:
                        k = self.vlab.index(ka) * nl + lb
                        off[k], ast[k] = at + (first + cols[kn][0]) * nb, m * nb
                        ncs[k], nbs[k] = len(self.V[cols[kn][1]]), nb
            kern[kn] = (T, goff, gst, gnd, off, ast, ncs, nbs)
        self._xq[kJ] = (gb, n, zb, n2, kern)
        return self._xq[kJ]

    def xq_sizes(self):
        """Elements of the blocks of Xq for all occupied quadruples, of the
        largest G and Z that one of them is made from, of U for the pairs
        of one smallest index, and of the blocks of <vo||vv>."""
        total = work = work2 = 0
        for kJ, count in zip(*np.unique(self.k4, return_counts=True)):
            gb, n, zb, n2, kern = self.xq_layout(int(kJ))
            total += int(count) * sum(len(self.O[kn]) * len(self.A3[x[0]]) for kn, x in kern.items())
            work, work2 = max(work, n), max(work2, n2)
        usz = {int(k): sum(self.xq_rows(int(self.red(k + kg)))[1] * len(self.V[kg]) for kg in self.vlab)
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
    zwork: tuple            # the same for z, (left, right)
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
    zwork = ((5 * maxV + 10) * max2, (5 * maxV + 10) * max3)
    parts = {
        'second-order quadruples': isz * nI4 * nA4,
        'Xo': isz * nI3 * no * nQ,
        'Xv': isz * xv,
        'Xq': isz * xq,
        'layouts': isz * (nP * nQ + no * nv * nQ + nP * no * nv + no * no * nQ + 2 * nP * nv * nv),
        'tables': (12 * nA4 + 24 * nA3 + 40 * nQ + nI5 * 600 + nI4 * (no * 9 + 250)
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
    t_xq = isz * (2 * w1 + no * no * nQ + vints + 3 * no * nv * maxV * maxV + xq_work + xq_work2 + xq_u)
    t_run = isz * (2 * n23 + 2 * n14 + work + lwork + sum(zwork) + 2 * (5 + 10 * no) * nv + 10 * no * nv)
    parts['blocks of a pentuple'] = t_run
    need = kept + max(t_y4, t_xo, t_xv, t_w1, t_xq, t_run)
    if budget is not None and need > budget:
        raise MemoryError('(P) in blocks of the symmetry labels needs %.0f MB besides T3, T4 and the integrals '
                          'for %d occupied and %d virtual spinors; %.0f MB are left'
                          % (need / 1e6, no, nv, budget / 1e6))
    return Plan(int(need), sec, int(n23), int(n14), int(work), int(lwork), zwork, int(kept), parts)


# ---- what is read of the packed quadruples (T4 and the second-order ones)

def quad_row(T4, J):
    '''The address of the packed row of the occupied quadruple J: the
    kernel reads t(J; f, abc) from it for the f and the triples (abc) of a
    block (zc_p_operands23).'''
    return T4.ctypes.data + J * T4.strides[0]


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
        self._k = {name: _fn(name, dtype) for name in ('operands23', 'pick', 'sector_energy',
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
        self.LZ, self.RZ = (np.empty(max(n, 1), dtype=dtype) for n in plan_.zwork)
        self.e = np.zeros(1)
        # the addresses that the kernel of the operands is given
        adr = lambda x: x.ctypes.data                           # noqa: E731
        self._adr = {
            'V': {k: (adr(x), adr(sec.V32[k]), len(x)) for k, x in sec.V.items()},
            'O': {k: (adr(x), len(x)) for k, x in sec.O.items()},
            'Xq': [{k: adr(x) for k, x in blk.items()} for blk in self.Xq],
            'Xv': [{k: adr(x) for k, x in blk.items()} for blk in self.Xv],
        }
        self._t3r = {}

    # ---- the second-order quadruples <Q| V T2 |0> / D of the left-hand side

    def second_order_quadruples(self, mo_e):
        '''[C(no,4), C(nv,4)] as zccsdt.cc.quadruples.second_order_quadruples:
        each element that the labels allow from its 36 products
        t(ij; ab) <cd||kl>.'''
        sec, no, nv, dtype = self.sec, self.nocc, self.nvir, self.dtype
        i2, j2 = Q._pairs(no)
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
        sec, dtype, no, red = self.sec, self.dtype, self.nocc, self.sec.red

        def c_(x):
            return np.ascontiguousarray(x, dtype=dtype)

        def stack(parts):
            return parts[0] if len(parts) == 1 else c_(np.concatenate(parts, axis=1))
        # W(n, o; klm; c) = P(m/kl) sum_f <no||fm> t(kl; fc), [(klm), o, n, c], for the o of one
        # label: [o, (n c)]
        oovo = -np.asarray(eris.ooov).transpose(0, 1, 3, 2)                         # <no||fm>
        G = np.einsum('nofm,Pfc->mPonc', oovo, self.tfc, optimize=True)
        W1 = P.fold(G, self.o12, 0, 1)
        del G, oovo
        w1 = {}
        for J3 in range(len(W1)):
            for ko_ in sec.olab:
                cols = sec.xq_cols(int(red(sec.k3[J3] - ko_)))[0]
                if cols:
                    os_ = sec.O[ko_]
                    w1[J3, ko_] = stack([c_(W1[J3][np.ix_(os_, sec.O[kn], sec.V[kc])]).reshape(len(os_), -1)
                                         for kn, (_, kc) in cols.items()])
        del W1
        # t(j o; ab) for the pairs of one label: [(ab), o]
        t2b = {(j, ko_): c_(self.t2_xp[j][np.ix_(sec.O[ko_], sec.Q2[q2])].T)
               for j in range(no) for ko_ in sec.olab for q2 in [int(red(ko_ + sec.ko[j]))] if q2 in sec.Q2}
        # <cn||fg> for the f and the g of one label each: [f, (n c), g]
        ovvv = np.asarray(eris.ovvv)
        vb = {}
        for kf in sec.vlab:
            for kg in sec.vlab:
                cols = sec.xq_cols(int(red(kf + kg)))[0]
                if cols:
                    F, G = sec.V[kf], sec.V[kg]
                    vb[kf, kg] = stack([
                        c_(-ovvv[np.ix_(sec.O[kn], sec.V[kc], F, G)].transpose(2, 0, 1, 3)).reshape(len(F), -1, len(G))
                        for kn, (_, kc) in cols.items()]).reshape(len(F), -1)
        del ovvv
        sz = sec.xq_sizes()
        self._g, self._z = np.empty(max(sz[1], 1), dtype=dtype), np.empty(max(sz[2], 1), dtype=dtype)
        quads = P.tuples(no, 4)
        pair_of = P.ranks(no, 2)
        out = []
        for j in range(no - 3):
            U = {}
            for k in range(j + 1, no):
                p = int(pair_of[j, k])
                for kg in sec.vlab:
                    rows, n = sec.xq_rows(int(red(sec.kpair[p] + kg)))
                    if not n:
                        continue
                    ng = len(sec.V[kg])
                    U[p, kg] = u = np.zeros((n, ng), dtype=dtype)
                    for ka, first, m, _ in rows:
                        if (p, ka) in self.tblk:
                            na = len(sec.V[ka])
                            gemm(self.tblk[p, ka].T, vb[int(red(sec.kpair[p] - ka)), kg],
                                 u[first:first + na * m].reshape(na, m * ng))
            for J in np.where(quads[:, 0] == j)[0]:
                out.append(self.xq_blocks(int(J), w1, t2b, U))
            del U
        self._g = self._z = None
        return out

    def xq_blocks(self, J, w1, t2b, U):
        '''{label of n: Xq[n, (abc)]} for the occupied quadruple J: the n of
        that label and the triples of the label that goes with it.'''
        sec, dtype, o13, o22, red = self.sec, self.dtype, self.o13, self.o22, self.sec.red
        gb, n, zb, n2, kern = sec.xq_layout(int(sec.k4[J]))
        G, Z = self._g, self._z
        P.zero(G[:n])
        P.zero(Z[:n2])
        for s in range(4):                                                  # (j|klm)
            j, J3, sg = int(o13.idx1[J, s]), int(o13.idx2[J, s]), float(o13.sgn[J, s])
            for q2, at, m in gb:
                ko_ = int(red(q2 - sec.ko[j]))
                if (J3, ko_) in w1:
                    L = t2b[j, ko_]                                         # [(ab), o]
                    gemm(L, w1[J3, ko_], G[at:at + L.shape[0] * m].reshape(L.shape[0], m), alpha=sg, beta=1.0)
        for s in range(3):                                                  # (jk|lm), j in the first pair
            pj, pl_, sg = int(o22.idx1[J, s]), int(o22.idx2[J, s]), float(o22.sgn[J, s])
            for kb, at, m in zb:
                u = U.get((pj, int(red(sec.kpair[pl_] - kb))))
                if u is not None:
                    tb = self.tblk[pl_, kb]                                 # [g, b]
                    gemm(u, tb, Z[at:at + m * tb.shape[1]].reshape(m, tb.shape[1]), alpha=-sg, beta=1.0)
        out = {}
        for kn, (T, goff, gst, gnd, off, ast, ncs, nbs) in kern.items():
            tri, nn = sec.tri[T], len(sec.O[kn])
            out[kn] = res = np.empty((nn, len(tri)), dtype=dtype)
            self._k['xq_sector'](
                _INT(sec.nvir), _p(sec.kv32), _INT(sec.modulus), _INT(sec.kmin), _p(sec.lid_of), _INT(len(sec.vlab)),
                _p(tri), _I64(len(tri)), _INT(nn), _INT(sec.qmin), _INT(sec.nq), _p(goff), _p(gst), _p(gnd),
                _p(sec.posQ), _p(G), _p(off), _p(ast), _p(ncs), _p(nbs), _p(sec.posV), _p(Z), _p(res))
        return out

    # ---- one occupied pentuple

    def _t3_rows(self, kn, pp):
        '''(rows, signs) of t(n kl; abc) in the packed T3 for the n of label
        ``kn`` and the occupied pair ``pp`` (sign 0: n is one of k, l), as
        addresses.'''
        if (kn, pp) not in self._t3r:
            ns = self.sec.O[kn]
            rows, sgn = _i64(self.o12.pos[ns, pp]), _c(self.o12.psg[ns, pp], np.float64)
            self._t3r[kn, pp] = (rows.ctypes.data, sgn.ctypes.data, rows, sgn)
        return self._t3r[kn, pp]

    def energy(self, I):
        '''Adds the energy of the occupied pentuple I to self.e.'''
        sec, dtype, nv, K = self.sec, self.dtype, self.nvir, self._k
        ko, kpair, k3, red = sec.ko, sec.kpair, sec.k3, sec.red
        o41, o32, o31 = self.o41, self.o32, self.o31
        T3, T4, Y4 = self.T3, self.T4, self.Y4
        M = int(sec.k5[I])
        b23, _, b14, _, off23, nc23, off14, nc14 = sec.layout5(M)
        J4s, xs, s41 = [[int(x) for x in a] for a in (o41.idx1[I], o41.idx2[I], o41.sgn[I])]
        J3s, ps, s32 = [[int(x) for x in a] for a in (o32.idx1[I], o32.idx2[I], o32.sgn[I])]
        aV, aO, aXq, aXv = (self._adr[x] for x in ('V', 'O', 'Xq', 'Xv'))
        a_t3 = T3.ctypes.data
        t3_rows = [a_t3 + J3 * T3.strides[0] for J3 in J3s]
        t4_rows, y4_rows = ([quad_row(amp, J) for J in J4s] for amp in (T4, Y4))
        vvvo = [self.vvvo_t.ctypes.data + x * self.vvvo_t.strides[0] for x in xs]
        t2x = [self.t2_xp.ctypes.data + x * self.t2_xp.strides[0] for x in xs]
        xo = [self.Xo.ctypes.data + J3 * self.Xo.strides[0] for J3 in J3s]
        vvoo = [self.vvoo_t.ctypes.data + pp * self.vvoo_t.strides[0] for pp in ps]
        kx, k3s, kps = [int(ko[x]) for x in xs], [int(k3[J3]) for J3 in J3s], [int(kpair[pp]) for pp in ps]
        with timer('ps.23'):
            for q, off, nr, nc in b23:
                T = int(red(M - q))
                kfs, kns = [int(red(q - k)) for k in kx], [int(red(k - q)) for k in k3s]
                # the rows of the stacked operands: [V,T4] | T2 with Xq | Xo with T3 | T3 with Xv,
                # and those of z: the first with the second-order quadruples | t(ijk; abc) <de||lm>
                seg, scale, zseg, zscale, n = [], [], [], [], 0
                for s in range(5):
                    if kfs[s] in aV:
                        rows, es, m = aV[kfs[s]]
                        seg += [0, n, m, vvvo[s], rows, t4_rows[s], es, 0]
                        zseg += [4, n, m, vvvo[s], rows, y4_rows[s], es, 0]
                        scale.append(-s41[s])
                        n += m
                nz = n
                for s in range(5):
                    if kfs[s] in aO:
                        rows, m = aO[kfs[s]]
                        seg += [1, n, m, t2x[s], rows, aXq[J4s[s]][kfs[s]], 0, 0]
                        scale.append(s41[s])
                        n += m
                for s in range(10):
                    if kns[s] in aO:
                        rows, m = aO[kns[s]]
                        seg += [2, n, m, xo[s], rows, a_t3] + list(self._t3_rows(kns[s], ps[s])[:2])
                        scale.append(s32[s])
                        n += m
                for s in range(10):
                    if kns[s] in aV:
                        _, es, m = aV[kns[s]]
                        seg += [3, n, m, t3_rows[s], es, aXv[ps[s]][kns[s]], 0, 0]
                        scale.append(s32[s])
                        n += m
                    if kps[s] == q:
                        zseg += [5, nz, 1, vvoo[s], 0, t3_rows[s], 0, 0]
                        zscale.append(s32[s])
                        nz += 1
                X = self.R23[off:off + nr * nc].reshape(nr, nc)
                Z = self.Z23[off:off + nr * nc].reshape(nr, nc)
                seg = np.array(seg + zseg, dtype=np.int64)
                scale = np.array(scale + scale[:len(zseg) // 8 - len(zscale)] + zscale, dtype=np.float64)
                K['operands23'](_INT(nv), _I64(len(scale)), _p(seg), _p(scale), _p(sec.Q2[q]), _I64(nr),
                                _p(sec.pr32[q]), _p(sec.A3[T]), _I64(nc), _p(sec.tri[T]),
                                _p(self.L), _p(self.R), _p(self.LZ), _p(self.RZ))
                for Lx, Rx, m, out in ((self.L, self.R, n, X), (self.LZ, self.RZ, nz, Z)):
                    if m:
                        gemm(Lx[:m * nr].reshape(m, nr).T, Rx[:m * nc].reshape(m, nc), out)
                    else:
                        P.zero(out)
        with timer('ps.14'):
            # - <ne||lm> t(ijk n; abcd), (ijk|lm): the rows of T4 that the ten splits share, each
            # once, with the sum of the integrals that multiply it, [row, e]
            pos, sg4 = o31.pos[J3s], o31.psg[J3s] * (-np.array(s32))[:, None]           # [split, n]
            rows, inv = np.unique(pos[sg4 != 0], return_inverse=True)
            LC = np.zeros((len(rows), nv), dtype=dtype)
            np.add.at(LC, inv, (self.ovoo_t[ps] * sg4[:, :, None])[sg4 != 0])
            klab = red(M - sec.k4[rows])                # the label of e that goes with a row
            for pe, off, nr, nc in b14:
                X = self.R14[off:off + nr * nc].reshape(nr, nc)
                Z = self.Z14[off:off + nr * nc].reshape(nr, nc)
                sel = np.where(klab == pe)[0]
                if not len(sel):
                    P.zero(X)
                    P.zero(Z)
                    continue
                ca = sec.A4[int(red(M - pe))]
                L = _c(LC[np.ix_(sel, sec.V[pe])].T)
                rows_, one = _i64(rows[sel]), np.ones(len(sel))
                step = max(1, len(self.R) // len(sel))
                for c0 in range(0, nc, step):
                    c1 = min(nc, c0 + step)
                    W = self.R[:len(sel) * (c1 - c0)].reshape(len(sel), c1 - c0)
                    for amp, out in ((T4, X), (Y4, Z)):
                        gemm(L, quad_pick(amp, rows_, one, ca[c0:c1], W), out[:, c0:c1])
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
