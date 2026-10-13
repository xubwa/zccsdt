'''The passes of the Lambda equations in the blocks of the symmetry labels
(zccsdt.cc.lambda_blk.BlockL3) for a left vector with a label shift: the
left product of zccsdt.cc.eom_left for the states of a shift, in the
blocks of that shift (docs/eom.md, section 12.2).

l3 and its residual are flat arrays in the layout t3shift.ShiftLayout(s) -
(occupied triples of group g) x (virtual triples of group g + s) -, l1 and
l2 dense arrays with their elements where l_a - l_i = s (classes of the
maps); T3 and every operand of the ground state have no shift.  What the
shift changes in the passes of BlockL3:

  half-expanded l3   L[P, D_delta, Q] with delta = class(d) - class(m) has
                     its pairs Q in the group gP - delta + s
  pass 1             the density Dr[(a i), (g o)] of l3 with T3 couples the
                     index pairs of delta + s with those of delta; D1 (l2
                     with T3) lives in delta = -s; the two stacked products
                     have operands of different classes (x of class gP -
                     class(d), y of class gQ - class(m))
  Gram matrix        between the occupied triples of group g and g + s
  pass 2             the outer product with l1 in the blocks gQ = gP, that
                     with l2 in gQ = gP + s; of the four parts of the
                     summed index of the drive products the two whose left
                     operand is made of l (l2, Y1) have the class of the
                     others plus s
  ladder             the pairs of the gathered Lv[Q, I, c] are in the group
                     g + s - class(c); Xq meets the occupied pairs of group
                     g - class(c), X2 the occupied index of class class(c) - s
  occupied operator  block by block (it does not change the group)

The gathers and reductions are the kernels of csrc/t3pack.c, driven by the
tables of the layouts (t3shift.ShiftT3): no kernel was written.
'''

import ctypes

import numpy as np

from zccsdt.blas import gemm
from zccsdt.cc import t3engine, t3kern, t3pack
from zccsdt.cc.t3engine import _CArgs, _batches, _target
from zccsdt.timing import timer


class ShiftL3:
    '''The engine of lambda_opt.residuals for l3 of the label shift
    ``shift`` (classes), on a lambda_blk.BlockL3 ``base`` and the
    t3shift.ShiftT3 ``sh`` of its T3 engine.'''

    def __init__(self, base, sh, shift):
        self.base, self.sh, self.shift = base, sh, int(shift)
        self.t3 = base.t3
        self.om, self.vm, self.dtype = base.om, base.vm, base.dtype
        self.Ls, self.L0 = sh.layout(self.shift), sh.layout(0)
        self.size = self.Ls.size
        self.nc, self.npg, self.ntg = base.nc, base.npg, sh.ntg
        self.vr, self.orr, self.nQg, self.nD, self.flat = base.vr, base.orr, base.nQg, base.nD, base.flat

    @property
    def mem_mb(self):
        return self.base.mem_mb

    def view(self, x):
        x = np.ascontiguousarray(np.asarray(x, dtype=self.dtype).ravel())
        assert x.size in (self.size, self.base.size)
        return x

    def target(self, out=None):
        return _target(out, (self.size,), self.dtype, True)

    def buffer_bytes(self, mem_mb=None):
        mem = self.mem_mb if mem_mb is None else mem_mb
        return max(self.base.buffer_bytes(mem), self.sh.buffer_bytes(mem, self.shift))

    def _gather(self, lay, X, pb, gP, delta, Qlo, Qhi, out):
        '''out[P, D_delta, Q] = x[m, P_0, P_1, d, Q_0, Q_1] for the pairs
        ``pb`` of group gP of an array in the layout ``lay`` (zeros where it
        has no block); out flat.'''
        e, nQ, nr = self.t3, Qhi - Qlo, self.nD[delta]
        view = out.reshape(len(pb), nr, nQ)
        r0 = 0
        for sd, sm in e.dsub[delta]:
            d, m = e.vrng[sd], e.orng[sm]
            if len(d) and len(m):
                if lay.nA[gP + sm] > 0:
                    self.sh._gather(lay, X, pb, d, m, Qlo, Qhi, out, r0 * nQ, nr * nQ)
                else:
                    view[:, r0:r0 + len(d) * len(m)] = 0
            r0 += len(d) * len(m)
        return view

    # ---- pass 1

    def pass1(self, L3, T3, GPd, GmQ, l2pp, Xacc):
        '''BlockL3.pass1 for l3 and l2pp of the shift; Xacc, Yall and the
        rows of Dr have it then.'''
        e, om, vm, dtype, s = self.t3, self.om, self.vm, self.dtype, self.shift
        no, nv, npo, nvp = om.n, vm.n, om.npair, vm.npair
        dm, nx, ny = nv * no, GPd.shape[0], GmQ.shape[1]
        assert nx == nv + no and ny == no + nv
        nc, vr, orr, nQg, nD = self.nc, self.vr, self.orr, self.nQg, self.nD
        L3, T3 = self.view(L3), self.view(T3)
        Yall = np.zeros((npo, nv, ny), dtype=dtype)
        Dr = np.zeros((dm, dm), dtype=dtype)
        D1 = np.zeros(dm, dtype=dtype)
        Xacc[...] = 0
        if not self.size:
            return Yall, Dr, D1
        X3 = Xacc.reshape(nx, no, nvp)
        G3 = GmQ.reshape(no, nvp, ny)
        Dd = {}
        D0 = np.zeros(nD[-s], dtype=dtype) if abs(s) < nc else None
        gy_cache = {}

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

        nb = max(1, min(e.nb, self.sh._batch1(s) // 2))
        for gP in range(self.npg):
            for pb in _batches(om.pair_off[gP], om.pair_off[gP + 1], nb):
                p0, p1, nbp = pb[0], pb[-1] + 1, len(pb)
                for gQ in range(self.npg):
                    dT = gP - gQ
                    dL = dT + s
                    if not nQg[gQ] or abs(dL) >= nc or not nD[dL]:
                        continue
                    okT = abs(dT) < nc and nD[dT] > 0
                    Qlo, Qhi = vm.pair_off[gQ], vm.pair_off[gQ + 1]
                    nQ, nrL = nQg[gQ], nD[dL]
                    nrT = nD[dT] if okT else 0
                    nL, nT = nbp * nrL * nQ, nbp * nrT * nQ
                    ws = e._workspace(nL + max(nT, nL))
                    with timer('gather'):
                        L = self._gather(self.Ls, L3, pb, gP, dL, Qlo, Qhi, ws[:nL])
                        if okT:
                            T = self._gather(self.L0, T3, pb, gP, dT, Qlo, Qhi, ws[nL:nL + nT])
                    if okT:
                        with timer('density'):
                            D = Dd.get(dT)
                            if D is None:
                                D = Dd[dT] = np.zeros((nrL, nrT), dtype=dtype)
                            for k in range(nbp):
                                gemm(L[k], T[k].T, D, beta=1.0)
                                if dT == -s:
                                    D0 += T[k] @ l2pp[p0 + k, Qlo:Qhi]
                    with timer('gemm'):
                        r0 = 0
                        for sd, sm in e.dsub[dL]:
                            (d0, d1), (m0, m1) = vr[sd], orr[sm]
                            nd, nm = d1 - d0, m1 - m0
                            rows = slice(r0, r0 + nd * nm)
                            r0 += nd * nm
                            if not (nd and nm):
                                continue
                            sx, sy = gP - sd, gQ - sm       # the classes of x and of y
                            okx = 0 <= sx < nc and (vr[sx][1] - vr[sx][0] + orr[sx][1] - orr[sx][0]) > 0
                            oky = 0 <= sy < nc and (vr[sy][1] - vr[sy][0] + orr[sy][1] - orr[sy][0]) > 0
                            if not (okx or oky):
                                continue
                            if nrL == nd * nm:
                                Lf = L.reshape(nbp * nd, nm * nQ)
                            else:
                                Lf = ws[nL:nL + nbp * nd * nm * nQ].reshape(nbp, nd * nm, nQ)
                                t3pack.axpby(Lf, L[:, rows], 1.0, 0.0)
                                Lf = Lf.reshape(nbp * nd, nm * nQ)
                            if okx:
                                (e0, e1), (j0, j1) = vr[sx], orr[sx]
                                ne, nj = e1 - e0, j1 - j0
                                G = np.concatenate((GPd[e0:e1, p0:p1, d0:d1],
                                                    GPd[nv + j0:nv + j1, p0:p1, d0:d1]), axis=0)
                                X = gemm(np.ascontiguousarray(G, dtype=dtype).reshape(ne + nj, -1), Lf)
                                X = X.reshape(ne + nj, nm, nQ)
                                t3pack.axpby(X3[e0:e1, m0:m1, Qlo:Qhi], X[:ne])
                                t3pack.axpby(X3[nv + j0:nv + j1, m0:m1, Qlo:Qhi], X[ne:])
                            if oky:
                                (e0, e1), (j0, j1) = vr[sy], orr[sy]
                                ne, nj = e1 - e0, j1 - j0
                                Y = gemm(Lf, gy(sm, gQ, sy)).reshape(nbp, nd, nj + ne)
                                t3pack.axpby(Yall[p0:p1, d0:d1, j0:j1], Y[:, :, :nj])
                                t3pack.axpby(Yall[p0:p1, d0:d1, no + e0:no + e1], Y[:, :, nj:])
        for dT, D in Dd.items():
            Dr[self.flat[dT + s][:, None], self.flat[dT]] = D
        if D0 is not None:
            D1[self.flat[-s]] = D0
        return Yall, Dr, D1

    def gram(self, L3, T3):
        '''G[I, J] = sum_A l3[I, A] t3[J, A]: I of group g, J of group g + s.'''
        om, Ls, L0, s = self.om, self.Ls, self.L0, self.shift
        L3, T3 = self.view(L3), self.view(T3)
        G = np.zeros((om.ntri,) * 2, dtype=self.dtype)
        for g in range(self.ntg):
            h = g + s
            if not 0 <= h < self.ntg or not (Ls.nI[g] and Ls.nA[g] and L0.nI[h] and L0.nA[h]):
                continue
            I = slice(om.tri_off[g], om.tri_off[g + 1])
            J = slice(om.tri_off[h], om.tri_off[h + 1])
            G[I, J] = gemm(Ls.block(L3, g), L0.block(T3, h).T)
        return G

    # ---- pass 2

    def pass2(self, L3, WL, Adrv, Bdrv, U, Vp, R, segs=None):
        '''BlockL3.pass2 for l3 of the shift with the operands of
        lambda_opt.residuals: U = (l1, f), Vp = (<P||Q>, l2), and the four
        parts of the summed index of the drive products (virtual, occupied:
        the left operand without a shift; virtual, occupied: the left
        operand made of l).'''
        e, om, vm, dtype, s = self.t3, self.om, self.vm, self.dtype, self.shift
        no, nv, nvp = om.n, vm.n, vm.npair
        nc, npg, vr, orr, nD = self.nc, self.npg, self.vr, self.orr, self.nD
        K, nu = Bdrv.shape[0], U.shape[1]
        assert segs is None and K == 2 * (nv + no) and nu == 2
        L3 = self.view(L3)
        B3 = Bdrv.reshape(K, no, nvp)
        Wd = None if WL is None else self.base._ring(WL)
        kcache = {}

        def kparts(se):
            '''The columns of the summed index for the base class se: the
            ranges of the four parts.'''
            if se not in kcache:
                a = vr[se] if 0 <= se < nc else (0, 0)
                b = orr[se] if 0 <= se < nc else (0, 0)
                c = vr[se + s] if 0 <= se + s < nc else (0, 0)
                d = orr[se + s] if 0 <= se + s < nc else (0, 0)
                parts = [(a[0], a[1]), (nv + b[0], nv + b[1]), (nv + no + c[0], nv + no + c[1]),
                         (2 * nv + no + d[0], 2 * nv + no + d[1])]
                cols = np.concatenate([np.arange(x, y) for x, y in parts]).astype(np.int64)
                kcache[se] = (parts, cols)
            return kcache[se]
        Us = np.ascontiguousarray(U[self.flat[s], 0:1], dtype=dtype) if abs(s) < nc else None
        U0 = np.ascontiguousarray(U[self.flat[0], 1:2], dtype=dtype)
        bd_cache = {}

        def bd(gQ, si):
            if (gQ, si) not in bd_cache:
                parts, cols = kparts(gQ - si - s)
                (i0, i1) = orr[si]
                Qlo, Qhi = vm.pair_off[gQ], vm.pair_off[gQ + 1]
                B = np.empty((len(cols), i1 - i0, Qhi - Qlo), dtype=dtype)
                k = 0
                for a, b in parts:
                    if b > a:
                        t3pack.axpby(B[k:k + b - a], B3[a:b, i0:i1, Qlo:Qhi], 1.0, 0.0)
                    k += b - a
                bd_cache[gQ, si] = B.reshape(len(cols), -1)
            return bd_cache[gQ, si]

        nQg = np.array(self.nQg, dtype=np.int64)
        Qoff = np.ascontiguousarray(vm.pair_off[:npg], dtype=np.int64)
        fred2 = getattr(t3pack._lib, 'zc_breduce2_' + t3pack._sfx(dtype))
        I_, p_ = _CArgs.I, _CArgs.p
        nb = max(1, min(e.nb, self.sh._batch2(s, s)))
        for gP in range(npg):
            gQs = [gQ for gQ in range(npg) if abs(gP - gQ + s) < nc and nQg[gQ] > 0 and nD[gP - gQ + s]]
            if not gQs:
                continue
            for pb in _batches(om.pair_off[gP], om.pair_off[gP + 1], nb):
                p0, p1, nbp = pb[0], pb[-1] + 1, len(pb)
                targets = self.sh._targets(self.Ls, p0, p1)
                if not targets:
                    continue
                zoff = -np.ones((nbp, npg), dtype=np.int64)
                base, off = {}, 0
                for gQ in gQs:
                    sz = nD[gP - gQ + s] * int(nQg[gQ])
                    base[gQ] = off
                    zoff[:, gQ] = off + np.arange(nbp) * sz
                    off += nbp * sz
                tmax = max(nD[gP - gQ + s] * int(nQg[gQ]) for gQ in gQs) * nbp
                ws = e._workspace(off + tmax)
                Z = ws[:off]
                for gQ in gQs:
                    dL = gP - gQ + s
                    Qlo, Qhi = vm.pair_off[gQ], vm.pair_off[gQ + 1]
                    nQ, nr = int(nQg[gQ]), nD[dL]
                    n = nbp * nr * nQ
                    Zg = Z[base[gQ]:base[gQ] + n].reshape(nbp, nr, nQ)
                    with timer('gemm'):
                        if Wd is None:
                            Zg[...] = 0
                        else:
                            with timer('gather'):
                                T = self._gather(self.Ls, L3, pb, gP, dL, Qlo, Qhi, ws[off:off + n])
                            for k in range(nbp):
                                gemm(Wd[dL], T[k], Zg[k])
                        if gQ == gP and Us is not None:         # l1[i,a] <P||Q>
                            for k in range(nbp):
                                gemm(Us, np.ascontiguousarray(Vp[p0 + k, 0:1, Qlo:Qhi], dtype=dtype), Zg[k], beta=1.0)
                        if gQ == gP + s:                        # f[i,a] l2[P,Q]
                            for k in range(nbp):
                                gemm(U0, np.ascontiguousarray(Vp[p0 + k, 1:2, Qlo:Qhi], dtype=dtype), Zg[k], beta=1.0)
                    with timer('drive'):
                        r0 = 0
                        for sc, si in e.dsub[dL]:
                            (c0, c1), (i0, i1) = vr[sc], orr[si]
                            ncc, ni = c1 - c0, i1 - i0
                            rows = slice(r0, r0 + ncc * ni)
                            r0 += ncc * ni
                            cols = kparts(gP - sc)[1]
                            if not (ncc and ni and len(cols)):
                                continue
                            B = bd(gQ, si)
                            A = np.ascontiguousarray(Adrv[p0:p1, c0:c1][:, :, cols], dtype=dtype).reshape(nbp * ncc, -1)
                            if nr == ncc * ni:
                                gemm(A, B, Zg.reshape(nbp * ncc, ni * nQ), alpha=-1.0, beta=1.0)
                            else:
                                X = ws[off:off + nbp * ncc * ni * nQ].reshape(nbp * ncc, ni * nQ)
                                gemm(A, B, X)
                                t3pack.axpby(Zg[:, rows], X.reshape(nbp, ncc * ni, nQ), -1.0, 1.0)
                with timer('reduce'):
                    for tb, tx, tP, tsg, A0, A1 in targets:
                        fred2(p_(R), p_(self.Ls.vcol), I_(len(tb)), p_(tb), p_(tx), p_(tP), p_(tsg),
                              I_(A0), I_(A1), I_(vm.ntri), p_(vm.free_c), p_(vm.pair_c),
                              p_(vm.pair_grp), p_(zoff), I_(npg), p_(nQg), p_(Qoff),
                              p_(e.rowpos), I_(no), p_(Z))

    # ---- pp ladder

    def ladder(self, L3, Wpp, R, Xq, X2):
        '''BlockL3.ladder for l3 of the shift.  Returns (r2, r1).'''
        e, om, vm, dtype, Ls, s = self.t3, self.om, self.vm, self.dtype, self.Ls, self.shift
        no, nv, npo, nvp = om.n, vm.n, om.npair, vm.npair
        nc, npg, nQg = self.nc, self.npg, self.nQg
        L3 = self.view(L3)
        r2 = np.zeros((npo, nvp), dtype=dtype)
        r1 = np.zeros((nv, no), dtype=dtype)
        if not self.size:
            return r2, r1
        fg = getattr(t3pack._lib, 'zc_bgather_vir_' + t3pack._sfx(dtype))
        fr = getattr(t3pack._lib, 'zc_breduce_ppl_' + t3pack._sfx(dtype))
        I_, p_ = _CArgs.I, _CArgs.p
        vspin, coff, ncs = self.base._vir_args()
        dummy = np.zeros(1, dtype=dtype)
        nbt = max(1, min(e.nbt, self.sh._batch_pp(s, s)))
        for g in range(self.ntg):
            A0, A1 = Ls.vrange(g)
            if A1 == A0 or Ls.nI[g] == 0:
                continue
            classes = [sc for sc in range(nc) if 0 <= g + s - sc < npg and ncs[sc] and nQg[g + s - sc]]
            if not classes:
                continue
            for rows in _batches(om.tri_off[g], om.tri_off[g + 1], nbt):
                nb, i0 = len(rows), int(rows[0])
                size = {sc: nQg[g + s - sc] * nb * int(ncs[sc]) for sc in classes}
                nx = sum(size.values()) if Wpp is not None else 0
                ws = e._workspace(nx + max(size.values()))
                X = [dummy] * nc
                hoff = np.zeros(nc, dtype=np.int64)
                xo = 0
                for sc in classes:
                    h = g + s - sc
                    Qlo, Qhi = vm.pair_off[h], vm.pair_off[h + 1]
                    nQ, ncc = Qhi - Qlo, int(ncs[sc])
                    T = ws[nx:nx + size[sc]].reshape(nQ, nb, ncc)
                    with timer('gather'):
                        fg(p_(L3), p_(Ls.obase), p_(Ls.vcol), p_(Ls.osig), p_(Ls.vsig), p_(vm.tri_c),
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
                        # Xq[P, I, e]: P of the group g - class(c);  X2[Q, I, i]: i of the class class(c) - s
                        (c0, c1) = self.vr[sc]
                        hp = g - sc
                        if 0 <= hp < npg:
                            plo, phi = om.pair_off[hp], om.pair_off[hp + 1]
                            if phi > plo:
                                A = np.ascontiguousarray(Xq[plo:phi, i0:i0 + nb, c0:c1], dtype=dtype)
                                r2[plo:phi, Qlo:Qhi] += gemm(A.reshape(phi - plo, nb * ncc), Tf.T)
                        if 0 <= sc - s < nc:
                            j0, j1 = self.orr[sc - s]
                            if j1 > j0:
                                B = np.ascontiguousarray(X2[Qlo:Qhi, i0:i0 + nb, j0:j1], dtype=dtype)
                                r1[c0:c1, j0:j1] += gemm(T.reshape(nQ * nb, ncc).T, B.reshape(nQ * nb, j1 - j0))
                if Wpp is not None:
                    with timer('reduce'):
                        xs = (ctypes.c_void_p * nc)(*[x.ctypes.data for x in X])
                        fr(p_(R), p_(Ls.Rbase), p_(Ls.vcol), I_(nb), p_(rows), I_(vm.ntri), I_(A0),
                           I_(A1), p_(vm.free_c), p_(vm.pair_c), p_(vspin), p_(hoff), p_(ncs),
                           p_(coff), xs)
        return r2, r1

    # ---- the rest

    def occ(self, O6, L3, R):
        '''R += O6 @ l3, block by block of the occupied triples.'''
        om, Ls, dtype = self.om, self.Ls, self.dtype
        L3 = self.view(L3)
        for g in range(self.ntg):
            if Ls.nI[g] == 0 or Ls.nA[g] == 0:
                continue
            I = slice(om.tri_off[g], om.tri_off[g + 1])
            if isinstance(O6, t3kern.OccOperator):
                O6.apply(Ls.block(L3, g), Ls.block(R, g), (int(I.start), int(I.stop)),
                         t3engine.occ_cols(Ls.nI[g], dtype.itemsize, self.mem_mb))
                continue
            Og = np.ascontiguousarray(O6[I, I], dtype=dtype)
            t3kern.occ_operator(Og, Ls.block(L3, g), Ls.block(R, g),
                                t3engine.occ_cols(Ls.nI[g], dtype.itemsize, self.mem_mb))
