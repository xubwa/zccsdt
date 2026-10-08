'''The passes of a Lambda iteration of CCSDT over Lambda_3 in the blocks of a
symmetry label: the engine of ``zccsdt.cc.lambda_opt`` for the layout of
``t3engine.SpinBlockT3`` / ``T3Layout``.

l3 has the block structure of T3 (the labels of i, j, k add up to those of
a, b, c), so it is stored as T3 is: block g, the triples whose classes add
up to g, as a matrix (nI[g], nA[g]) in a flat vector.  BlockL3 has the
methods of ``lambda_opt.DenseL3`` and takes the same dense operands, in the
enumeration of pairs and triples of the grouped maps (t3pack.TripleMap with
classes); every pass is split into the blocks that the labels leave:

  pass 1   for occupied pairs of group gP and virtual pairs of group gQ the
           half-expanded l3 and T3 as [P, D, Q], D the (d, m) with
           class(d) - class(m) = gP - gQ (the composite index of the ring of
           the T3 equation): the density sum over (P, Q) is one GEMM per
           pair and difference of classes; the two stacked GEMMs with the
           operands [x, P, d] and [(m, Q), y] run per class of d and m (x
           and y are then of the class gP - class(d))
  pass 2   as pass 2 of the T3 equation in blocks: the ring per difference
           of classes with the transposed operator, the outer products where
           gP = gQ, the drive products per class of the summed index, one
           reduction per batch of pairs (zc_breduce2)
  ladder   as the pp ladder of the T3 equation in blocks, with the
           transposed blocks of the operator, and the contractions with Xq
           and X2 on the same gathered batches
  hh, gram, update   block by block

One class is no symmetry: the blocks are then the whole arrays, in
lexicographic order.  The buffers are views of the work array of the T3
engine (t3engine.workspace), which is as large as the passes of the T3
equation need it (pass 1 here takes two gathered blocks of a batch of pass
2, less than its products and one gathered block).

C kernels only (the gathers and reductions of csrc/t3pack.c); without them
lambda_opt uses the dense layout.
'''

import ctypes

import numpy as np

from zccsdt.blas import gemm
from zccsdt.cc import t3engine, t3kern, t3pack
from zccsdt.cc.t3engine import _CArgs, _batches, _cgather
from zccsdt.timing import timer


def available(t3eng):
    '''Can BlockL3 work on this T3 engine?  The block engine with its C
    kernels, all blocks stored.'''
    return (isinstance(t3eng, t3engine.SpinBlockT3) and bool(t3eng.use_c) and not t3eng.L.flip
            and t3kern.USE_C)


class BlockL3:
    '''The steps of a Lambda iteration that touch l3, T3 and the residual in
    the blocks of ``t3eng`` (a t3engine.SpinBlockT3 without spin flip).'''

    def __init__(self, t3eng):
        assert available(t3eng)
        e = self.t3 = t3eng
        self.om, self.vm, self.dtype, self.L = e.om, e.vm, np.dtype(e.dtype), e.L
        self.size = e.size
        om, vm = self.om, self.vm
        self.nc, self.npg = e.nc, e.npg
        self.vr = [(int(r[0]), int(r[-1]) + 1) if len(r) else (0, 0) for r in e.vrng]
        self.orr = [(int(r[0]), int(r[-1]) + 1) if len(r) else (0, 0) for r in e.orng]
        self.nQg = [int(vm.pair_off[g + 1] - vm.pair_off[g]) for g in range(e.npg)]
        # the composite index D_delta: (a, i) with class(a) - class(i) = delta,
        # as positions a no + i of the dense index (a, i)
        self.flat, self.nD = {}, {}
        for delta in e.deltas:
            a, i = e._rows(delta)
            self.flat[delta] = np.ascontiguousarray(a * om.n + i, dtype=np.int64)
            self.nD[delta] = len(a)
        self._wd = (None, None)

    @property
    def mem_mb(self):
        return self.t3.mem_mb

    @property
    def nb(self):
        return self.t3.nb

    def set_memory(self, mem_mb):
        self.t3.set_memory(mem_mb)

    def view(self, x):
        x = np.ascontiguousarray(np.asarray(x, dtype=self.dtype).ravel())
        assert x.size == self.size, 'l3 / T3 in the blocks of the labels: %d elements, not %d' \
            % (self.size, x.size)
        return x

    def buffer_bytes(self, mem_mb=None):
        '''Bytes of the work array with ``mem_mb``: that of the T3 engine.'''
        return self.t3.buffer_bytes(self.mem_mb if mem_mb is None else mem_mb)

    def _gather(self, X, pb, delta, Qlo, Qhi, out):
        '''out[P, D_delta, Q] = x[m, P_0, P_1, d, Q_0, Q_1] for the pairs
        ``pb`` and Qlo <= Q < Qhi; out flat.'''
        e, nQ, nr = self.t3, Qhi - Qlo, self.nD[delta]
        r0 = 0
        for sd, sm in e.dsub[delta]:
            d, m = e.vrng[sd], e.orng[sm]
            if len(d) and len(m):
                _cgather(e, X, pb, d, m, Qlo, Qhi, out, r0 * nQ, nr * nQ)
            r0 += len(d) * len(m)
        return out.reshape(len(pb), nr, nQ)

    # ---- pass 1

    def pass1(self, L3, T3, GPd, GmQ, l2pp, Xacc):
        '''As DenseL3.pass1: Xacc (nx, no nvp) is written; returns (Yall,
        Dr, D1).  The operands have the labels of the amplitude equations:
        GPd[x, P, d] with class(x) = group(P) - class(d), GmQ[(m, Q), y]
        with class(y) = group(Q) - class(m).'''
        e, om, vm, dtype = self.t3, self.om, self.vm, self.dtype
        no, nv, npo, nvp = om.n, vm.n, om.npair, vm.npair
        dm, nx, ny = nv * no, GPd.shape[0], GmQ.shape[1]
        assert nx == nv + no and ny == no + nv
        nc, vr, orr, nQg, nD = self.nc, self.vr, self.orr, self.nQg, self.nD
        L3, T3 = self.view(L3), self.view(T3)
        Yall = np.zeros((npo, nv, ny), dtype=dtype)
        Dr = np.zeros((dm, dm), dtype=dtype)
        D1 = np.zeros(dm, dtype=dtype)
        Xacc[...] = 0
        if not self.size:           # no triples
            return Yall, Dr, D1
        X3 = Xacc.reshape(nx, no, nvp)
        G3 = GmQ.reshape(no, nvp, ny)
        Dd = {delta: np.zeros((nD[delta],) * 2, dtype=dtype) for delta in e.deltas}
        D0 = np.zeros(nD[0], dtype=dtype)
        gy_cache = {}           # the operand [(m, Q), y] per (class of m, gQ, class of y)

        def gy(sm, gQ, sy):
            key = (sm, gQ, sy)
            if key not in gy_cache:
                (m0, m1), (j0, j1), (e0, e1) = orr[sm], orr[sy], vr[sy]
                Qlo, Qhi = vm.pair_off[gQ], vm.pair_off[gQ + 1]
                nj = j1 - j0
                G = np.empty((m1 - m0, Qhi - Qlo, nj + e1 - e0), dtype=dtype)
                t3pack.axpby(G[:, :, :nj], G3[m0:m1, Qlo:Qhi, j0:j1], 1.0, 0.0)
                t3pack.axpby(G[:, :, nj:], G3[m0:m1, Qlo:Qhi, no + e0:no + e1], 1.0, 0.0)
                gy_cache[key] = G.reshape((m1 - m0) * (Qhi - Qlo), -1)
            return gy_cache[key]

        for gP in range(self.npg):
            for pb in _batches(om.pair_off[gP], om.pair_off[gP + 1], e.nb):
                p0, p1, nbp = pb[0], pb[-1] + 1, len(pb)
                for gQ in range(self.npg):
                    delta = gP - gQ
                    if abs(delta) >= nc or not nQg[gQ] or not nD[delta]:
                        continue
                    Qlo, Qhi = vm.pair_off[gQ], vm.pair_off[gQ + 1]
                    nQ, nr = nQg[gQ], nD[delta]
                    n = nbp * nr * nQ
                    ws = e._workspace(2 * n)
                    with timer('gather'):
                        L = self._gather(L3, pb, delta, Qlo, Qhi, ws[:n])
                        T = self._gather(T3, pb, delta, Qlo, Qhi, ws[n:])
                    with timer('density'):
                        for k in range(nbp):
                            gemm(L[k], T[k].T, Dd[delta], beta=1.0)
                            if delta == 0:
                                D0 += T[k] @ l2pp[p0 + k, Qlo:Qhi]
                    with timer('gemm'):
                        r0 = 0
                        for sd, sm in e.dsub[delta]:
                            (d0, d1), (m0, m1) = vr[sd], orr[sm]
                            nd, nm = d1 - d0, m1 - m0
                            sy = gP - sd                    # the class of x and of y
                            rows = slice(r0, r0 + nd * nm)
                            r0 += nd * nm
                            if not (nd and nm and 0 <= sy < nc):
                                continue
                            (e0, e1), (j0, j1) = vr[sy], orr[sy]
                            ne, nj = e1 - e0, j1 - j0
                            if not ne + nj:
                                continue
                            G = np.concatenate((GPd[e0:e1, p0:p1, d0:d1],
                                                GPd[nv + j0:nv + j1, p0:p1, d0:d1]), axis=0)
                            Gy = gy(sm, gQ, sy)
                            if nr == nd * nm:       # one block of classes: L as it is
                                Lf = L.reshape(nbp * nd, nm * nQ)
                            else:                   # [(P, d), (m, Q)] in the storage of T
                                Lf = ws[n:n + nbp * nd * nm * nQ].reshape(nbp, nd * nm, nQ)
                                t3pack.axpby(Lf, L[:, rows], 1.0, 0.0)
                                Lf = Lf.reshape(nbp * nd, nm * nQ)
                            X = gemm(np.ascontiguousarray(G, dtype=dtype).reshape(ne + nj, -1), Lf)
                            Y = gemm(Lf, Gy).reshape(nbp, nd, nj + ne)
                            X = X.reshape(ne + nj, nm, nQ)
                            t3pack.axpby(X3[e0:e1, m0:m1, Qlo:Qhi], X[:ne])
                            t3pack.axpby(X3[nv + j0:nv + j1, m0:m1, Qlo:Qhi], X[ne:])
                            t3pack.axpby(Yall[p0:p1, d0:d1, j0:j1], Y[:, :, :nj])
                            t3pack.axpby(Yall[p0:p1, d0:d1, no + e0:no + e1], Y[:, :, nj:])
        for delta in e.deltas:
            fl = self.flat[delta]
            Dr[fl[:, None], fl] = Dd[delta]
        D1[self.flat[0]] = D0
        return Yall, Dr, D1

    def gram(self, L3, T3):
        '''G[I, J] = sum_A l3[I, A] t3[J, A] over the unique virtual triples
        (I and J of one group).'''
        om, L = self.om, self.L
        L3, T3 = self.view(L3), self.view(T3)
        G = np.zeros((om.ntri,) * 2, dtype=self.dtype)
        for g in L.stored:
            if L.nI[g] and L.nA[g]:
                I = slice(om.tri_off[g], om.tri_off[g + 1])
                G[I, I] = gemm(L.block(L3, g), L.block(T3, g).T)
        return G

    # ---- pass 2

    def _ring(self, WL):
        '''The blocks [D_delta, D_delta] of the ring operator [(a,i), (d,m)]
        (made once per operator).'''
        if self._wd[0] is not WL:
            self._wd = (WL, {delta: np.ascontiguousarray(WL[fl[:, None], fl], dtype=self.dtype)
                             for delta, fl in self.flat.items()})
        return self._wd[1]

    def ring_bytes(self):
        '''Bytes of the blocks of the ring operator (kept with it).'''
        return sum(n * n for n in self.nD.values()) * self.dtype.itemsize

    def pass2(self, L3, WL, Adrv, Bdrv, U, Vp, R, segs=None):
        '''As DenseL3.pass2.  ``segs``: the parts of the summed index of the
        drive products, (first column, 'v' or 'o') each (default: virtual,
        occupied, virtual, occupied, the layout of lambda_opt); a column of
        class s meets the pairs P and virtual indices c with group(P) -
        class(c) = s.'''
        e, om, vm, dtype = self.t3, self.om, self.vm, self.dtype
        no, nv, npo, nvp = om.n, vm.n, om.npair, vm.npair
        nc, npg, vr, orr, nD = self.nc, self.npg, self.vr, self.orr, self.nD
        K, nu = Bdrv.shape[0], U.shape[1]
        if segs is None:
            assert K == 2 * (nv + no)
            segs = ((0, 'v'), (nv, 'o'), (nv + no, 'v'), (2 * nv + no, 'o'))
        L3 = self.view(L3)
        B3 = Bdrv.reshape(K, no, nvp)
        Wd = None if WL is None else self._ring(WL)
        # the columns of class s: a range of every part
        kseg = [[(off + r[0], off + r[1]) for off, r in
                 ((off, vr[s] if kind == 'v' else orr[s]) for off, kind in segs)] for s in range(nc)]
        kcols = [np.concatenate([np.arange(a, b) for a, b in kseg[s]]).astype(np.int64)
                 for s in range(nc)]
        U0 = np.ascontiguousarray(U[self.flat[0]], dtype=dtype)
        bd_cache = {}           # B[x, (i, Q)] per (gQ, class of i), x of class gQ - class(i)

        def bd(gQ, si):
            if (gQ, si) not in bd_cache:
                (i0, i1), cols = orr[si], kcols[gQ - si]
                Qlo, Qhi = vm.pair_off[gQ], vm.pair_off[gQ + 1]
                B = np.empty((len(cols), i1 - i0, Qhi - Qlo), dtype=dtype)
                k = 0
                for a, b in kseg[gQ - si]:
                    if b > a:
                        t3pack.axpby(B[k:k + b - a], B3[a:b, i0:i1, Qlo:Qhi], 1.0, 0.0)
                    k += b - a
                bd_cache[gQ, si] = B.reshape(len(cols), -1)
            return bd_cache[gQ, si]

        nQg = np.array(self.nQg, dtype=np.int64)
        Qoff = np.ascontiguousarray(vm.pair_off[:npg], dtype=np.int64)
        fred2 = getattr(t3pack._lib, 'zc_breduce2_' + t3pack._sfx(dtype))
        I_, p_ = _CArgs.I, _CArgs.p
        for gP in range(npg):
            for pb in _batches(om.pair_off[gP], om.pair_off[gP + 1], e.nb):
                p0, p1, nbp = pb[0], pb[-1] + 1, len(pb)
                gQs = [gQ for gQ in range(npg) if abs(gP - gQ) < nc and nQg[gQ] > 0]
                if not gQs:
                    continue
                # the products [P, D, Q] of one gQ next to each other
                zoff = -np.ones((nbp, npg), dtype=np.int64)
                base, off = {}, 0
                for gQ in gQs:
                    sz = nD[gP - gQ] * int(nQg[gQ])
                    base[gQ] = off
                    zoff[:, gQ] = off + np.arange(nbp) * sz
                    off += nbp * sz
                tmax = max(nD[gP - gQ] * int(nQg[gQ]) for gQ in gQs) * nbp
                ws = e._workspace(off + tmax)
                Z = ws[:off]
                for gQ in gQs:
                    delta = gP - gQ
                    Qlo, Qhi = vm.pair_off[gQ], vm.pair_off[gQ + 1]
                    nQ, nr = int(nQg[gQ]), nD[delta]
                    n = nbp * nr * nQ
                    if not n:
                        continue
                    Zg = Z[base[gQ]:base[gQ] + n].reshape(nbp, nr, nQ)
                    with timer('gather'):
                        T = self._gather(L3, pb, delta, Qlo, Qhi, ws[off:off + n])
                    with timer('gemm'):
                        if Wd is None:
                            Zg[...] = 0
                        else:
                            for k in range(nbp):
                                gemm(Wd[delta], T[k], Zg[k])
                        if delta == 0 and nu:       # the outer products: gP = gQ
                            for k in range(nbp):
                                gemm(U0, np.ascontiguousarray(Vp[p0 + k, :, Qlo:Qhi], dtype=dtype),
                                     Zg[k], beta=1.0)
                    with timer('drive'):
                        r0 = 0
                        for sc, si in e.dsub[delta]:
                            (c0, c1), (i0, i1) = vr[sc], orr[si]
                            ncc, ni = c1 - c0, i1 - i0
                            se = gP - sc                    # the class of the summed index
                            rows = slice(r0, r0 + ncc * ni)
                            r0 += ncc * ni
                            if not (ncc and ni and 0 <= se < nc and len(kcols[se])):
                                continue
                            B = bd(gQ, si)
                            A = np.ascontiguousarray(Adrv[p0:p1, c0:c1][:, :, kcols[se]],
                                                     dtype=dtype).reshape(nbp * ncc, -1)
                            if nr == ncc * ni:      # one block of classes: into Z as it is
                                gemm(A, B, Zg.reshape(nbp * ncc, ni * nQ), alpha=-1.0, beta=1.0)
                            else:                   # the batch at once, in the storage of the gathered block
                                X = ws[off:off + nbp * ncc * ni * nQ].reshape(nbp * ncc, ni * nQ)
                                gemm(A, B, X)
                                t3pack.axpby(Zg[:, rows], X.reshape(nbp, ncc * ni, nQ), -1.0, 1.0)
                with timer('reduce'):
                    for tb, tx, tP, tsg, A0, A1 in e._red_targets(p0, p1):
                        fred2(p_(R), p_(e.Rcol), I_(len(tb)), p_(tb), p_(tx), p_(tP), p_(tsg),
                              I_(A0), I_(A1), I_(vm.ntri), p_(vm.free_c), p_(vm.pair_c),
                              p_(vm.pair_grp), p_(zoff), I_(npg), p_(nQg), p_(Qoff),
                              p_(e.rowpos), I_(no), p_(Z))

    def drive_bytes(self, K=None):
        '''Bytes of the copies of the drive operand [x, (i, Q)] in blocks
        that a pass 2 holds, and of those of the operand [(m, Q), y] in
        pass 1: the elements whose labels add up.'''
        nc, vr, orr, nQg = self.nc, self.vr, self.orr, self.nQg
        nx = [vr[s][1] - vr[s][0] + orr[s][1] - orr[s][0] for s in range(nc)]
        n = sum(nx[gQ - si] * (orr[si][1] - orr[si][0]) * nQg[gQ]
                for gQ in range(self.npg) for si in range(nc) if 0 <= gQ - si < nc)
        return n * self.dtype.itemsize

    # ---- pp ladder

    def _vir_args(self):
        vm = self.vm
        vspin = np.ascontiguousarray(vm.spins, dtype=np.int64)
        coff = np.array([r[0] for r in self.vr], dtype=np.int64)
        ncs = np.array([r[1] - r[0] for r in self.vr], dtype=np.int64)
        return vspin, coff, ncs

    def _vir_sizes(self, g, nb, ncs):
        '''Elements of the gathered [Q, I, c] per class of c, for nb triples
        of group g.'''
        return {sc: self.nQg[g - sc] * nb * int(ncs[sc]) for sc in range(self.nc)
                if 0 <= g - sc < self.npg and ncs[sc] and self.nQg[g - sc]}

    def ladder(self, L3, Wpp, R, Xq, X2):
        '''As DenseL3.ladder with the operator as the list of its diagonal
        pair blocks.  Returns (r2, r1).'''
        e, om, vm, dtype, L = self.t3, self.om, self.vm, self.dtype, self.L
        no, nv, npo, nvp = om.n, vm.n, om.npair, vm.npair
        L3 = self.view(L3)
        r2 = np.zeros((npo, nvp), dtype=dtype)
        r1 = np.zeros((nv, no), dtype=dtype)
        if not self.size:
            return r2, r1
        fg = getattr(t3pack._lib, 'zc_bgather_vir_' + t3pack._sfx(dtype))
        fr = getattr(t3pack._lib, 'zc_breduce_ppl_' + t3pack._sfx(dtype))
        I_, p_ = _CArgs.I, _CArgs.p
        vspin, coff, ncs = self._vir_args()
        dummy = np.zeros(1, dtype=dtype)
        for g in L.stored:
            A0, A1 = vm.tri_off[g], vm.tri_off[g + 1]
            if A1 == A0:
                continue
            for rows in _batches(om.tri_off[g], om.tri_off[g + 1], e.nbt):
                nb, i0 = len(rows), int(rows[0])
                size = self._vir_sizes(g, nb, ncs)
                if not size:
                    continue
                # workspace [X_0 | X_1 | ... | T]: the products stay alive for the reduction
                nx = sum(size.values()) if Wpp is not None else 0
                ws = e._workspace(nx + max(size.values()))
                X = [dummy] * self.nc
                hoff = np.zeros(self.nc, dtype=np.int64)
                xo = 0
                for sc in sorted(size):
                    h = g - sc
                    Qlo, Qhi = vm.pair_off[h], vm.pair_off[h + 1]
                    nQ, ncc = Qhi - Qlo, int(ncs[sc])
                    T = ws[nx:nx + size[sc]].reshape(nQ, nb, ncc)
                    with timer('gather'):
                        fg(p_(L3), p_(L.obase), p_(L.vcol), p_(L.osig), p_(L.vsig), p_(vm.tri_c),
                           p_(vm.sgn_c), I_(nvp), I_(nb), p_(rows), I_(Qlo), I_(nQ), I_(coff[sc]),
                           I_(ncc), p_(T))
                    Tf = T.reshape(nQ, nb * ncc)
                    if Wpp is not None:
                        X[sc] = ws[xo:xo + size[sc]].reshape(nQ, nb * ncc)
                        with timer('gemm'):
                            gemm(Wpp[h].T, Tf, X[sc])
                        xo += size[sc]
                        hoff[sc] = Qlo
                    with timer('small'):
                        # Xq[P, I, e]: P of the group of Q;  X2[Q, I, i]: i of the class of c
                        plo, phi = om.pair_off[h], om.pair_off[h + 1]
                        (c0, c1), (j0, j1) = self.vr[sc], self.orr[sc]
                        if phi > plo:
                            A = np.ascontiguousarray(Xq[plo:phi, i0:i0 + nb, c0:c1], dtype=dtype)
                            r2[plo:phi, Qlo:Qhi] += gemm(A.reshape(phi - plo, nb * ncc), Tf.T)
                        if j1 > j0:
                            B = np.ascontiguousarray(X2[Qlo:Qhi, i0:i0 + nb, j0:j1], dtype=dtype)
                            r1[c0:c1, j0:j1] += gemm(T.reshape(nQ * nb, ncc).T,
                                                     B.reshape(nQ * nb, j1 - j0))
                if Wpp is not None:
                    with timer('reduce'):
                        xs = (ctypes.c_void_p * self.nc)(*[x.ctypes.data for x in X])
                        fr(p_(R), p_(e.Rbase), p_(e.Rcol), I_(nb), p_(rows), I_(vm.ntri), I_(A0),
                           I_(A1), p_(vm.free_c), p_(vm.pair_c), p_(vspin), p_(hoff), p_(ncs),
                           p_(coff), xs)
        return r2, r1

    def contract_x(self, T3, gpp, vq, Xq, X2):
        '''The contractions of T3 that the ladder of a Lambda iteration
        meets, with Tv[Q, I, f] = t3[I, f, Q_0, Q_1]:

          Xq[P, I, e]  = sum_Q gpp[P, Q] Tv[Q, I, e]
          X2[(e<f), I, i] = A(ef) sum_Q vq[e, i, Q] Tv[Q, I, f]

        gpp (npo, nvp), vq (nv, no, nvp); Xq (npo, ntri_o, nv) and X2
        (nvp, ntri_o, no) are added to (zeros on entry).'''
        e, om, vm, dtype, L = self.t3, self.om, self.vm, self.dtype, self.L
        no, nv, nvp = om.n, vm.n, vm.npair
        T3 = self.view(T3)
        if not self.size:
            return
        fg = getattr(t3pack._lib, 'zc_bgather_vir_' + t3pack._sfx(dtype))
        I_, p_ = _CArgs.I, _CArgs.p
        vspin, coff, ncs = self._vir_args()
        for g in L.stored:
            if vm.tri_off[g + 1] == vm.tri_off[g]:
                continue
            for rows in _batches(om.tri_off[g], om.tri_off[g + 1], e.nbt):
                nb, i0 = len(rows), int(rows[0])
                size = self._vir_sizes(g, nb, ncs)
                for sc in sorted(size):
                    h = g - sc
                    Qlo, Qhi = vm.pair_off[h], vm.pair_off[h + 1]
                    nQ, ncc = Qhi - Qlo, int(ncs[sc])
                    f0, f1 = self.vr[sc]
                    T = e._workspace(size[sc]).reshape(nQ, nb, ncc)
                    fg(p_(T3), p_(L.obase), p_(L.vcol), p_(L.osig), p_(L.vsig), p_(vm.tri_c),
                       p_(vm.sgn_c), I_(nvp), I_(nb), p_(rows), I_(Qlo), I_(nQ), I_(coff[sc]),
                       I_(ncc), p_(T))
                    Tf = T.reshape(nQ, nb * ncc)
                    plo, phi = om.pair_off[h], om.pair_off[h + 1]
                    if phi > plo:
                        Xq[plo:phi, i0:i0 + nb, f0:f1] += gemm(
                            np.ascontiguousarray(gpp[plo:phi, Qlo:Qhi], dtype=dtype), Tf).reshape(
                                phi - plo, nb, ncc)
                    for se in range(self.nc):       # <ei||Q>: class(e) + class(i) = h
                        si = h - se
                        if not 0 <= si < self.nc:
                            continue
                        (e0, e1), (j0, j1) = self.vr[se], self.orr[si]
                        if e1 == e0 or j1 == j0:
                            continue
                        V = np.ascontiguousarray(vq[e0:e1, j0:j1, Qlo:Qhi], dtype=dtype)
                        Z = gemm(V.reshape(-1, nQ), Tf).reshape(e1 - e0, j1 - j0, nb, ncc)
                        del V
                        # [(e<f), I, i] += Z[e, i, I, f],  [(f<e), I, i] -= Z[e, i, I, f]
                        ee, ff = np.meshgrid(np.arange(e0, e1), np.arange(f0, f1), indexing='ij')
                        for sel, sign in ((ee < ff, 1.0), (ee > ff, -1.0)):
                            if not sel.any():
                                continue
                            ea, fa = ee[sel], ff[sel]
                            q = vm.pidx[np.minimum(ea, fa), np.maximum(ea, fa)]
                            part = Z[ea - e0, :, :, fa - f0].transpose(0, 2, 1)   # [q, I, i]
                            if sign < 0:
                                X2[q, i0:i0 + nb, j0:j1] -= part
                            else:
                                X2[q, i0:i0 + nb, j0:j1] += part
                            del part
                        del Z

    def x_bytes(self, mem_mb=None):
        '''Bytes of the transients of contract_x next to the work array: a
        product [e, i, I, f] of one class each of e, i and f for a batch of
        triples, a selection of it, and a product [P, I, e].'''
        _, nb1, nbt = self.t3._batch_sizes(self.mem_mb if mem_mb is None else mem_mb)
        nI = max([int(self.L.nI[g]) for g in self.L.stored] + [1])
        nv = max(r[1] - r[0] for r in self.vr)
        no = max(r[1] - r[0] for r in self.orr)
        nP = max([int(self.om.pair_off[g + 1] - self.om.pair_off[g]) for g in range(self.npg)])
        return min(nbt, nI) * nv * (2 * nv * no + nP) * self.dtype.itemsize

    # ---- the rest

    def occ(self, O6, L3, R):
        '''R += O6 @ l3, block by block of the occupied triples.'''
        om, L, dtype = self.om, self.L, self.dtype
        L3 = self.view(L3)
        for g in L.stored:
            if L.nI[g] == 0 or L.nA[g] == 0:
                continue
            I = slice(om.tri_off[g], om.tri_off[g + 1])
            Og = np.ascontiguousarray(O6[I, I], dtype=dtype)
            t3kern.occ_operator(Og, L.block(L3, g), L.block(R, g),
                                t3engine.occ_cols(L.nI[g], dtype.itemsize, self.mem_mb))

    def occ_bytes(self):
        '''Bytes of the largest block of the occupied operator (a copy).'''
        return max([int(n) ** 2 for n in self.L.nI] + [0]) * self.dtype.itemsize

    def target(self, out=None):
        return self.t3.target(out)

    def update(self, L3, R, eo, ev):
        '''l3 + R / D written over R.'''
        return self.t3.update(self.view(L3), R, eo, ev)
