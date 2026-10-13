'''EOM-DEA: the 5p3h operators at lowest order, folded (docs/eom.md, section
11) - on top of the complete rows of 2p, 3p1h and 4p2h of zccsdt.cc.dea.

The 5p3h row is kept to the lowest order in the fluctuation potential, as
the quadruples of CC4 and the pentuples of the (P) correction are:

    (omega - D5) r5 = <5p3h| [V, R4] + [[V, T2], R3] + [[V, R2], T3] + 1/2 [[[V, T2], T2], R2] |0>,

D5 the differences of the orbital energies (canonical spinors), V the bare
two-electron operator; and r5 enters the rows below it completely:

    sigma(3p1h) += <3p1h| V r5 |0>                    <mn||ef>
    sigma(4p2h) += <4p2h| (F~ + V~) r5 |0>            f~[m,e], <dm||ef>~, <mn||le>~ (T1-dressed).

The right-hand side is the quantity r of (P) (zccsdt.cc.pentuples) for the
amplitudes T + R on the padded spinors, on the rows with both ghosts: it is
a polynomial in the amplitudes and, as every residual, exactly linear in R
there - [V, T4 + R4] gives [V, R4], [[V, T2 + R2], T3 + R3] the two mixed
terms, the cubic term its part linear in R2.  The class of (P) is used as
it is, for the C(o,3) occupied pentuples of the sector only.

Nothing of the size of the 5p3h vector is stored: for one occupied triple
(i<j<k) at a time r5[(abcde)] is made, divided by omega - D5 and contracted
into the rows of 3p1h and 4p2h.  The eigenproblem depends on omega; it is
solved to self-consistency (zccsdt.cc.ccn.folded_roots).

For the states with two electrons more this is one level above the
complete 4p2h operators (which describe them as CCSD describes a ground
state), the triples-like operators 5p3h treated as CC3 / CCSDT-n treat
triples: at lowest order.
'''

import copy
import time
from math import comb

import numpy as np

from zccsdt.blas import gemm
from zccsdt.cc import eom, pentuples as PT, t4pack as P

PARTS = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h')        # r of (P), without its left-hand side


def _blocked(A, B, C, blocks, c0=0, c1=None):
    '''C[(q), (t) - c0] = sum_x A[x, (q)] B[x, (t) - c0] for the blocks (q, t) of
    ``blocks`` - pairs and triples whose labels add to the label of the row -
    with the x for which A is not zero there; the rest of C is zero.'''
    P.zero(C)
    for qsel, tsel in blocks:
        if c1 is not None:
            tsel = tsel[(tsel >= c0) & (tsel < c1)] - c0
        if not len(tsel) or not len(qsel):
            continue
        Aq = A[:, qsel]
        rows = np.flatnonzero(np.abs(Aq).max(axis=1) > 1e-9)
        if len(rows):
            C[np.ix_(qsel, tsel)] = Aq[rows].T @ B[np.ix_(rows, tsel)]
    return C


def drive(p, I, blocks=None):
    '''r of (P) for the occupied pentuple I (zccsdt.cc.pentuples.Pentuples.both
    without the left-hand side): a work array [C(nv,5)].  ``blocks``: the
    (pairs, triples) of virtual spinors that the labels of the row allow in
    X[(de), (abc)] (the product is done for these only).'''
    no, nv, nQ, nA3, nA4 = p.nocc, p.nvir, p.nQ, p.nA3, p.nA4
    T3, T4 = p.T3, p.T4
    o41, o32, o31, o12 = p.o41, p.o32, p.o31, p.o12
    J4s, xs, s41 = o41.idx1[I], o41.idx2[I], o41.sgn[I]
    J3s, ps, s32 = o32.idx1[I], o32.idx2[I], o32.sgn[I]
    r, LA, LB, LC = p.r, p.LA, p.LB, p.LC
    P.zero(r)
    ka, kd, kg = 5 * nv, 5 * nv + 10 * no, 5 * nv + 15 * no
    Xq = p._xq_of(J4s)
    for s in range(5):
        np.multiply(p.vvvo_t[xs[s]], -float(s41[s]), out=LA[s * nv:(s + 1) * nv])
        np.multiply(p.t2_xp[xs[s]], float(s41[s]), out=LA[kd + s * no:kd + (s + 1) * no])
    for s in range(10):
        np.multiply(p.Xo[J3s[s]], float(s32[s]), out=LA[ka + s * no:ka + (s + 1) * no])
        np.multiply(p.ovoo_t[ps[s]], -float(s32[s]), out=LB[s * no:(s + 1) * no])
    P.tn_expand(LA[kg:].reshape(10, nv, nQ), np.ascontiguousarray(T3[J3s]), nv, 1, 2, outer=0)
    rowsv = (ps[:, None] * nv + np.arange(nv)).ravel()
    sgnv = np.repeat(s32, nv)
    rows3, sgn3 = o12.pos[:, ps].T.ravel(), o12.psg[:, ps].T.ravel()
    rows4, sgn4 = o31.pos[J3s].ravel(), o31.psg[J3s].ravel()
    rows4, inv = np.unique(rows4[sgn4 != 0], return_inverse=True)
    LC[:len(rows4)] = 0
    np.add.at(LC, inv, LB[sgn4 != 0] * sgn4[sgn4 != 0][:, None])
    one4 = np.ones(len(rows4), dtype=np.int8)
    T4r = np.ascontiguousarray(T4[J4s])
    for c in PT._batches(nA3, p.plan.cols23):
        nc = c[1] - c[0]
        R = p._work('R', (15 * nv + 15 * no, nc))
        P.tn_expand(R[:ka].reshape(5, nv, nc), T4r, nv, 1, 3, cols=c, outer=0)
        P.tn_rows(R[ka:kd], T3, rows3, sgn3, cols=c)
        for s in range(5):
            R[kd + s * no:kd + (s + 1) * no] = Xq[s][:, c[0]:c[1]]
        P.tn_rows(R[kg:], p.Xv2, rowsv, sgnv, cols=c)
        X = p._work('X', (nQ, nc))
        if blocks is None:
            gemm(LA.T, R, X)
        else:
            _blocked(LA, R, X, blocks, c[0], c[1])
        P.tn_reduce(r, X, nv, 2, 3, cols=c)
    R = X = None
    for c in PT._batches(nA4, p.plan.cols14):
        nc = c[1] - c[0]
        R = p._work('R', (len(rows4), nc))
        X = p._work('X', (nv, nc))
        P.tn_rows(R, T4, rows4, one4, cols=c)
        gemm(LC[:len(rows4)].T, R, X)
        P.tn_reduce(r, X, nv, 1, 4, cols=c)
    return r


class Fold5:
    '''The matrix of the rows 2p, 3p1h, 4p2h with the 5p3h operators folded
    in at the frequency omega, on a problem of zccsdt.cc.dea of rank 4
    (Problem with the full 4p2h row, or ProblemSym).'''

    def __init__(self, base, max_memory=None, labels=None, parity=None, target=0, target_parity=0):
        '''``labels``, ``parity`` (one integer per spinor of the problem,
        occupied then virtual; 2 Omega, 0 / 1) with the ``target`` of the
        problem: the products with two and three virtual indices are done
        in the blocks that they allow.  Default: dense products.'''
        if base.rank != 4 or base.row4 != 'full':
            raise ValueError('the 5p3h row: on the complete rows to 4p2h')
        self.base = base
        self.no, self.nv = no, nv = base.no, base.nv
        self.dtype = base.dtype
        if self.dtype.kind != 'f':
            raise NotImplementedError('the folded 5p3h row: real arithmetic')
        g1, g2 = base.ghosts
        self.rp = rp = np.asarray(base.places)
        e = copy.copy(base.eris)
        e.use_labels = False
        e.__dict__.pop('_buffers', None)
        self.eris = e
        t5 = P.tuples(no + 2, 5)
        has = (t5 == g1).any(axis=1) & (t5 == g2).any(axis=1)
        back = np.full(no + 2, -1, dtype=np.int64)
        back[rp] = np.arange(no)
        self.rows = []
        for I in np.flatnonzero(has):
            t = t5[I]
            p1, p2 = int(np.argmax(t == g1)), int(np.argmax(t == g2))
            rest = tuple(int(back[x]) for x in t if x not in (g1, g2))
            self.rows.append((int(I), -1.0 if (p1 + p2 - 1) % 2 else 1.0, rest))
        ev = np.asarray(base.eris.mo_energy).real
        self.eo, self.ev5 = ev[rp], P.tuple_sums(ev[no + 2:], 5)
        isz = self.dtype.itemsize
        from zccsdt import memory
        self.plan = PT.plan(no + 2, nv, isz, None if max_memory is None else memory.budget(max_memory))
        # the integrals of the back terms: T1-dressed blocks on the spinors of eris
        ints = base.ints
        a2, b2 = P.tuples(nv, 2).T
        i2, j2 = P.tuples(no, 2).T
        self.fov = np.ascontiguousarray(np.asarray(ints.f)[rp, no + 2:])                         # [m, e]
        self.oovv = np.ascontiguousarray(np.asarray(ints.oovv)[rp[i2], rp[j2]][:, a2, b2])       # [(mn), (ef)]
        self.vovv = np.ascontiguousarray(np.asarray(ints.vovv)[:, rp][:, :, a2, b2].transpose(1, 0, 2))   # [m, d, (ef)]
        # <mn||le>~ = - <mn||el>~ as [(mn), l, e]
        self.ooov = np.ascontiguousarray(-np.asarray(ints.oovo)[rp[i2], rp[j2]][:, :, rp].transpose(0, 2, 1))
        self.pair = np.zeros((no, no), dtype=np.int64)
        self.pair[i2, j2] = self.pair[j2, i2] = np.arange(len(i2))
        self.nsigma, self.seconds, self.timing = 0, 0.0, {'drive': 0.0, 'back': 0.0, 'setup': 0.0, 'base': 0.0}
        self.blocks, self._buf = None, None
        if labels is not None:
            lab = np.asarray(labels, dtype=np.int64)
            par = np.zeros(len(lab), dtype=np.int64) if parity is None else np.asarray(parity, dtype=np.int64)
            lv, pv = lab[no:], par[no:]
            v2, v3 = P.tuples(nv, 2), P.tuples(nv, 3)
            gq, gt = {}, {}
            for j, k in enumerate(zip(lv[v2].sum(axis=1).tolist(), (pv[v2].sum(axis=1) % 2).tolist())):
                gq.setdefault(k, []).append(j)
            for j, k in enumerate(zip(lv[v3].sum(axis=1).tolist(), (pv[v3].sum(axis=1) % 2).tolist())):
                gt.setdefault(k, []).append(j)
            gq = {k: np.array(v) for k, v in gq.items()}
            gt = {k: np.array(v) for k, v in gt.items()}
            self.blocks, self.row_key = {}, {}
            for _, _, t in self.rows:
                M = (int(target + lab[list(t)].sum()), int(target_parity + par[list(t)].sum()) % 2)
                self.row_key[t] = M
                if M not in self.blocks:
                    self.blocks[M] = [(gq[q], gt[(M[0] - q[0], (M[1] - q[1]) % 2)]) for q in gq
                                      if (M[0] - q[0], (M[1] - q[1]) % 2) in gt]

    def back(self, r5, t, S3, S4, blocks=None):
        '''The terms of r5[(abcde)] of the occupied triple t = (i<j<k) in the
        rows of 3p1h (S3[i, (abc)]) and 4p2h (S4[(ij), (abcd)]).'''
        nv = self.nv
        nQ, nA3, nA4 = comb(nv, 2), comb(nv, 3), comb(nv, 4)
        if self._buf is None:           # (kept: new arrays of this size pay their page faults per triple)
            self._buf = (np.empty((nv, nA4), dtype=self.dtype), np.empty((nQ, nA3), dtype=self.dtype),
                         np.empty((nv, nA3), dtype=self.dtype))
        E14, E23, Z = self._buf
        P.tn_expand(E14, r5, nv, 1, 4)                                  # r5(e abcd)
        P.tn_expand(E23, r5, nv, 2, 3)                                  # r5(ef abc)
        for pos, sg in ((0, 1.0), (1, -1.0), (2, 1.0)):
            z = t[pos]
            m, n = [x for x in t if x != z]
            # sigma(3p1h)[z] += <mn||ef> r5(z m n; abc ef)
            S3[z] += sg * (self.oovv[self.pair[m, n]] @ E23)
            # sigma(4p2h)[(z l)] -= <mn||le> r5(z m n; abcd e), for every l but z
            C = self.ooov[self.pair[m, n]] @ E14                        # [l, (abcd)]
            for l in range(self.no):
                if l != z:
                    S4[self.pair[z, l]] -= sg * (1.0 if z < l else -1.0) * C[l]
            # the same position of the triple as the summed index m: (x y | m)
            x, y = m, n
            # sigma(4p2h)[(xy)] += f[m,e] r5(x y m; abcd e) + P(d/abc) <dm||ef> r5(x y m; abc ef)
            S4[self.pair[x, y]] += sg * (self.fov[z] @ E14)
            if blocks is None:
                gemm(self.vovv[z], E23, Z)                              # [d, (abc)]
            else:
                P.zero(Z)
                V = self.vovv[z]
                for qsel, tsel in blocks:
                    Vq = V[:, qsel]
                    rows = np.flatnonzero(np.abs(Vq).max(axis=1) > 1e-9)
                    if len(rows):
                        Z[np.ix_(rows, tsel)] = Vq[rows] @ E23[np.ix_(qsel, tsel)]
            P.tn_reduce(S4[self.pair[x, y]], Z, nv, 1, 3, scale=-sg)
        return S3, S4

    def _blocks_of(self, t):
        return None if self.blocks is None else self.blocks[self.row_key[t]]

    def rows5(self, x):
        '''The right-hand side of the 5p3h row for x, [(i<j<k), (abcde)] (tests, small cases).'''
        X2, X3, X4 = self.base.padded(x)
        pent = PT.Pentuples(self.eris, X2, X3, X4, self.plan, only=PARTS)
        return np.array([drive(pent, I, self._blocks_of(t)) * sgn for I, sgn, t in self.rows])

    def add(self, x, y, omega):
        '''y += the 5p3h terms of the rows of x at the frequency omega.'''
        base, no, nv = self.base, self.no, self.nv
        t0 = time.perf_counter()
        X2, X3, X4 = base.padded(x)
        pent = PT.Pentuples(self.eris, X2, X3, X4, self.plan, only=PARTS)
        del X2, X3, X4
        self.timing['setup'] += time.perf_counter() - t0
        S3 = np.zeros((no, comb(nv, 3)), dtype=self.dtype)
        S4 = np.zeros((comb(no, 2), comb(nv, 4)), dtype=self.dtype)
        for I, sgn, t in self.rows:
            t0 = time.perf_counter()
            blk = self._blocks_of(t)
            r5 = drive(pent, I, blk) * sgn
            r5 /= omega - (self.ev5 - self.eo[list(t)].sum())
            t1 = time.perf_counter()
            self.back(r5, t, S3, S4, blk)
            self.timing['drive'] += t1 - t0
            self.timing['back'] += time.perf_counter() - t1
        del pent
        base.add_dense(y, S3, S4)
        return base.project(y)

    def apply_at(self, omega):
        def apply(x, y):
            t0 = time.perf_counter()
            self.base.sigma(x, y)
            self.timing['base'] += time.perf_counter() - t0
            self.add(x, y, omega)
            self.nsigma += 1
            self.seconds += time.perf_counter() - t0
            return y
        return apply


def kernel(base, start, conv_tol=1e-8, omega_tol=1e-10, max_memory=None, verbose=False, stdout=None, **labels):
    '''The root of the folded problem that follows ``start`` = (omega,
    vector) - a root of ``base``, the complete rows to 4p2h: (omega, the
    Solved of zccsdt.cc.ccn.folded_roots, the Fold5).'''
    import sys
    from zccsdt.cc import ccn
    out = stdout or sys.stdout
    f5 = Fold5(base, max_memory, **labels)
    store = eom.Store(base.sizes, base.dtype)

    def log(k, outer, lam, gap, res, nsig):
        if verbose:
            print('  5p3h folded: step %d  omega = %.11f  |lambda - omega| = %.1e  residual %.1e  %d sigma vectors'
                  % (outer, lam, gap, res, nsig), file=out, flush=True)
    sol = ccn.folded_roots(f5.apply_at, base.precond, None, store, 1, conv_tol=conv_tol, omega_tol=omega_tol,
                           log=log, start=[start])
    return float(sol.e[0].real), sol, f5
