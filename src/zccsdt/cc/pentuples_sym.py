'''(P) in symmetry sectors: zccsdt.cc.pentuples for spinors that carry a
label of an Abelian group (zccsdt.cc.quadruples_sym).
'''

import ctypes
import os
from dataclasses import dataclass
from math import comb

import numpy as np

from zccsdt.blas import gemm
from zccsdt.cc import pentuples as PT, quadruples_sym as QS, t4pack as P
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


class Sectors(QS.Sectors):
    '''The classes of zccsdt.cc.quadruples_sym.Sectors and those of the
    virtual quadruples.'''

    def __init__(self, eris):
        super().__init__(eris)
        k4 = np.rint(P.tuple_sums(self.kv.astype(np.float64), 4)).astype(np.int64)
        self.A4, self.pos4 = QS._classes(self.red(k4))
        self.kv32 = np.ascontiguousarray(self.kv, dtype=np.int32)
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

    def sizes5(self, pents):
        '''(elements of the largest X[(de), (abc)], of the largest
        X[e, (abcd)]) over the occupied pentuples ``pents``.'''
        Ms = np.unique(self.red(self.ko[pents].sum(axis=1)))
        return (max(self.layout5(int(M))[1] for M in Ms), max(self.layout5(int(M))[3] for M in Ms))


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


def kernel(eris, t2, t3, t4, max_memory=None, plan_=None):
    '''E(P) as zccsdt.cc.pentuples.kernel, in the symmetry sectors of
    ``eris.labels``.'''
    no = eris.nocc
    nv = np.asarray(eris.fock).shape[0] - no
    if no < 5 or nv < 5:
        return 0.0
    sec = Sectors(eris)
    p = PT.Pentuples(eris, t2, t3, t4)
    dtype = p.dtype
    pents = p.pents
    n23, n14 = sec.sizes5(pents)
    R23, Z23 = (np.empty(n23, dtype=dtype) for _ in range(2))
    R14, Z14 = (np.empty(n14, dtype=dtype) for _ in range(2))
    ko = sec.ko
    kpair = sec.red(ko[P.tuples(no, 2)].sum(axis=1))
    k3 = sec.red(ko[P.tuples(no, 3)].sum(axis=1))
    ev = np.ascontiguousarray(np.asarray(eris.mo_energy)[no:], dtype=np.float64)
    o41, o32, o31, o12 = p.o41, p.o32, p.o31, p.o12
    T3, T4, Y4, Xv2, Xq, Xo = p.T3, p.T4, p.Y4, p.Xv2, p.Xq, p.Xo
    efn = _fn('sector_energy', dtype)
    e = np.zeros(1)
    f64 = lambda x: np.ascontiguousarray(x, dtype=np.float64)
    i64 = lambda x: np.ascontiguousarray(x, dtype=np.int64)
    i32 = lambda x: np.ascontiguousarray(x, dtype=np.int32)
    for I in range(len(pents)):
        M = int(sec.red(ko[pents[I]].sum()))
        b23, _, b14, _, off23, nc23, off14, nc14 = sec.layout5(M)
        J4s, xs, s41 = o41.idx1[I], o41.idx2[I], o41.sgn[I]
        J3s, ps, s32 = o32.idx1[I], o32.idx2[I], o32.sgn[I]
        with timer('ps.23'):
            for q, off, nr, nc in b23:
                T = int(sec.red(M - q))
                rq, ca, tri = sec.Q2[q], i64(sec.A3[T]), sec.tri[T]
                R = R23[off:off + nr * nc].reshape(nr, nc)
                Z = Z23[off:off + nr * nc].reshape(nr, nc)
                R[...] = 0
                Z[...] = 0
                for s in range(5):
                    x, J, sg = int(xs[s]), int(J4s[s]), float(s41[s])
                    kf = int(sec.red(q - ko[x]))
                    if kf in sec.V:             # (ijkl|m): <de||fm> t(ijkl; abc f) = - <de||fm> t(ijkl; f abc)
                        E = sec.V[kf]
                        L = np.ascontiguousarray(p.vvvo_t[x][np.ix_(E, rq)].T)
                        W = np.empty((len(E), nc), dtype=dtype)
                        gemm(L, quad_rows13(T4, J, nv, i32(E), tri, W), R, alpha=-sg, beta=1.0)
                        gemm(L, quad_rows13(Y4, J, nv, i32(E), tri, W), Z, alpha=-sg, beta=1.0)
                    if kf in sec.O:             # (i|jklm): t(in; de) Xq(n; jklm; abc)
                        ns = sec.O[kf]
                        L = np.ascontiguousarray(p.t2_xp[x][np.ix_(ns, rq)].T)
                        gemm(L, np.ascontiguousarray(Xq[J][np.ix_(ns, ca)]), R, alpha=sg, beta=1.0)
                for s in range(10):
                    J3, pp, sg = int(J3s[s]), int(ps[s]), float(s32[s])
                    kn = int(sec.red(k3[J3] - q))
                    if kn in sec.O:             # (ijm|kl): Xo(n; ijm; de) t(n kl; abc)
                        ns = sec.O[kn]
                        sn = o12.psg[ns, pp]
                        ns = ns[sn != 0]
                        if len(ns):
                            L = np.ascontiguousarray(Xo[J3][np.ix_(ns, rq)].T)
                            W = np.empty((len(ns), nc), dtype=dtype)
                            quad_pick(T3, i64(o12.pos[ns, pp]), f64(sn[sn != 0]), ca, W)
                            gemm(L, W, R, alpha=sg, beta=1.0)
                    if kn in sec.V:             # (klm|ij): t(klm; g de) Xv(g; ij; abc)
                        G = sec.V[kn]
                        col, s3 = sec.t3_rows(kn, q)
                        L = np.ascontiguousarray((T3[J3][col] * s3).T)
                        gemm(L, np.ascontiguousarray(Xv2[np.ix_(pp * nv + G, ca)]), R, alpha=sg, beta=1.0)
                    if kpair[pp] == q:          # (ijk|lm): t(ijk; abc) <de||lm>
                        Z += sg * np.outer(p.vvoo_t[pp][rq], T3[J3][ca])
        with timer('ps.14'):
            for pe, off, nr, nc in b14:
                va, ca = sec.V[pe], i64(sec.A4[int(sec.red(M - pe))])
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
                            Ls.append((-sg * sn)[:, None] * p.ovoo_t[pp][np.ix_(ns, va)])
                R = R14[off:off + nr * nc].reshape(nr, nc)
                Z = Z14[off:off + nr * nc].reshape(nr, nc)
                if not rows:
                    R[...] = 0
                    Z[...] = 0
                    continue
                rows, inv = np.unique(np.concatenate(rows), return_inverse=True)
                L = np.zeros((len(rows), nr), dtype=dtype)
                np.add.at(L, inv, np.concatenate(Ls))
                L = np.ascontiguousarray(L.T)
                one = np.ones(len(rows))
                W = np.empty((len(rows), nc), dtype=dtype)
                gemm(L, quad_pick(T4, i64(rows), one, ca, W), R)
                gemm(L, quad_pick(Y4, i64(rows), one, ca, W), Z)
        with timer('ps.sum'):
            efn(_INT(nv), _p(sec.kv32), _INT(sec.modulus), _INT(M), _p(ev), _DBL(float(p.eo5[I])),
                _INT(sec.kmin), _INT(sec.nk), _p(sec.lid_of), _p(sec.lstart), _p(sec.lmem),
                _INT(sec.qmin), _INT(sec.nq), _p(off23), _p(nc23), _p(off14), _p(nc14),
                _p(sec.posV), _p(sec.posQ), _p(sec.pos3), _p(sec.pos4), _p(R23), _p(Z23), _p(R14), _p(Z14),
                _p(e))
    return float(e[0])
