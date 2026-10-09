'''Rank-3 arrays with a label shift in the blocks of the symmetry labels,
and the passes of the T3 engine over them (docs/eom.md, section 6).

An operator that changes the sum of the labels by a fixed amount - the
excitation operator R of a state that is not totally symmetric - has
r[ijk,abc] where

    l_a + l_b + l_c - l_i - l_j - l_k = shift

(in units of the classes of the maps, t3pack.classes_of: the labels less
their minimum, divided by their step).  ShiftLayout stores its blocks: the
occupied triples of group g with the virtual triples of group g + shift.
shift = 0 is the layout of T3 (t3engine.T3Layout without spin flip).

Every array carries a shift, (sum of the labels of its creators) - (sum of
those of its annihilators): amplitudes r[i..,a..] virtual minus occupied,
integrals and intermediates <pq||rs> bra minus ket, f[p,q] p minus q.  A
product has the sum of the shifts of its factors.  ShiftT3 does pass 1
and the residual of t3engine.SpinBlockT3 for operands with shifts:

    pass1(T, sT; integrals, sG)         the outputs have shift sT + sG
    residual(T, sT; ring, pp and hh operators of shift sW;
             drive: t2 of shift sa with W_vvvo, W_vooo of shift sT + sW - sa)
                                        the result has shift sT + sW

with the C kernels of the engine as they are - the gathers and reductions
are driven by tables, and only the tables know the layout.  The operators
of a shift couple different blocks: W_pp the pair block h with h + sW (a
list of rectangular blocks by the column block), the ring operator the
classes (d m) of delta = s_d - s_m with delta + sW, the occupied operator
the triples of group g with g + sW (taken as blocks of the matrix).
'''

import ctypes

import numpy as np

from zccsdt import scratch
from zccsdt.blas import gemm, gemm_tasks
from zccsdt.cc import t3engine, t3kern, t3pack
from zccsdt.cc.t3engine import _CArgs, _batches, _target
from zccsdt.timing import timer


class ShiftLayout:
    '''The blocks (occupied triples of group g) x (virtual triples of group
    g + shift) of a flat array; the interface of t3engine.T3Layout.'''

    flip = False

    def __init__(self, om, vm, shift=0):
        self.om, self.vm, self.shift = om, vm, int(shift)
        self.ntg = ntg = om.ntg
        nAv = np.diff(vm.tri_off)
        self.nI = np.diff(om.tri_off)
        self.nA = np.array([nAv[g + self.shift] if 0 <= g + self.shift < ntg else 0 for g in range(ntg)],
                           dtype=np.int64)
        self.stored = list(range(ntg))
        self.boff = {}
        off = 0
        for g in self.stored:
            self.boff[g] = off
            off += int(self.nI[g] * self.nA[g])
        self.size = off
        self.obase = np.zeros(om.ntri, dtype=np.int64)
        for g in range(ntg):
            I = np.arange(om.tri_off[g], om.tri_off[g + 1])
            self.obase[I] = self.boff[g] + (I - om.tri_off[g]) * self.nA[g]
        self.vcol = np.ascontiguousarray(np.arange(vm.ntri) - vm.tri_off[vm.tri_grp], dtype=np.int64) \
            if vm.ntri else np.zeros(0, dtype=np.int64)
        self.osig = np.ones(om.ntri)
        self.vsig = np.ones(vm.ntri)
        # targets of a reduction: the triples whose block is not empty
        self.Rbase = np.where(self.nA[om.tri_grp] > 0, self.obase, -1).astype(np.int64) \
            if om.ntri else np.zeros(0, dtype=np.int64)

    def vrange(self, g):
        '''(A0, A1): the virtual triples of the block of occupied group g.'''
        h = g + self.shift
        if not 0 <= h < self.ntg:
            return 0, 0
        return int(self.vm.tri_off[h]), int(self.vm.tri_off[h + 1])

    def zeros(self, dtype):
        return np.zeros(self.size, dtype=dtype)

    def block(self, T3, g):
        o = self.boff[g]
        return T3[o:o + self.nI[g] * self.nA[g]].reshape(self.nI[g], self.nA[g])

    def blocks(self):
        '''[(slice of the flat array, shape, slice of the occupied triples,
        slice of the virtual triples)] of the blocks that are not empty.'''
        out = []
        for g in self.stored:
            if self.nI[g] and self.nA[g]:
                A0, A1 = self.vrange(g)
                out.append((slice(self.boff[g], self.boff[g] + int(self.nI[g] * self.nA[g])),
                            (int(self.nI[g]), int(self.nA[g])),
                            slice(int(self.om.tri_off[g]), int(self.om.tri_off[g + 1])), slice(A0, A1)))
        return out

    def to_dense(self, T3):
        '''[occupied triples, virtual triples] in the order of the maps.'''
        D = np.zeros((self.om.ntri, self.vm.ntri), dtype=T3.dtype)
        for where, shape, I, A in self.blocks():
            D[I, A] = T3[where].reshape(shape)
        return D

    def from_dense(self, D):
        out = np.zeros(self.size, dtype=D.dtype)
        for where, shape, I, A in self.blocks():
            out[where] = D[I, A].ravel()
        return out


def one_body_pairs(F, vm, rows, cols, scale, dtype):
    '''scale * (F_ad d_be - F_bd d_ae - F_ae d_bd + F_be d_ad) for the pairs
    (a<b) of block ``rows`` and (d<e) of block ``cols`` of ``vm``: the
    antisymmetrized one-body operator between two pair blocks
    (ccsdt_opt.add_one_body_pairs for a block with itself).'''
    (r0, r1), (c0, c1) = vm.blocks[rows], vm.blocks[cols]
    W = np.zeros((r1 - r0, c1 - c0), dtype=dtype)
    if r1 == r0 or c1 == c0:
        return W
    a, b = vm.pairs[r0:r1, 0], vm.pairs[r0:r1, 1]
    y = np.arange(vm.n)[None, :]
    rws = np.broadcast_to(np.arange(r1 - r0)[:, None], (r1 - r0, vm.n))
    for keep, act, s_lt, s_gt in ((b, a, 1, -1), (a, b, -1, 1)):
        k = keep[:, None]
        col = vm.pidx[np.minimum(y, k), np.maximum(y, k)] - c0
        valid = (y != k) & (col >= 0) & (col < c1 - c0)
        val = scale * np.where(y < k, s_lt, s_gt) * F[act[:, None], y]
        np.add.at(W, (rws[valid], col[valid]), val[valid])
    return W


class ShiftT3:
    '''Pass 1 and the residual of a SpinBlockT3 engine for operands with
    label shifts (see the module).  Needs the C kernels; no spin flip.'''

    def __init__(self, eng):
        assert isinstance(eng, t3engine.SpinBlockT3) and not eng.L.flip
        if not (eng.use_c and t3kern.USE_C):
            raise RuntimeError('arrays with a label shift need the C kernels (libzc_t3pack)')
        self.eng = eng
        self.om, self.vm, self.dtype = eng.om, eng.vm, eng.dtype
        self.nc, self.npg, self.ntg = eng.nc, eng.npg, eng.ntg
        self._L, self._redt = {}, {}
        om, vm = self.om, self.vm
        self.vr = [(int(r[0]), int(r[-1]) + 1) if len(r) else (0, 0) for r in eng.vrng]
        self.orr = [(int(r[0]), int(r[-1]) + 1) if len(r) else (0, 0) for r in eng.orng]
        self.nQ = [int(vm.pair_off[g + 1] - vm.pair_off[g]) for g in range(self.npg)]
        self.nP = [int(om.pair_off[g + 1] - om.pair_off[g]) for g in range(self.npg)]
        self.rows = {d: eng._rows(d) for d in eng.deltas}
        self.nD = {d: len(self.rows[d][0]) for d in eng.deltas}

    def layout(self, shift=0):
        s = int(shift)
        if s not in self._L:
            self._L[s] = ShiftLayout(self.om, self.vm, s)
        return self._L[s]

    def size(self, shift=0):
        return self.layout(shift).size

    def shifts(self):
        '''The shifts with a block that is not empty.'''
        return [s for s in range(1 - self.ntg, self.ntg) if self.layout(s).size]

    # ---- batches and buffers

    def _unit2(self, sT, so):
        '''Elements of the work array per occupied pair of a batch of pass 2.'''
        unit = 1
        for gP in range(self.npg):
            if not self.nP[gP]:
                continue
            out = [self.nD[gP - gQ + so] * self.nQ[gQ] for gQ in range(self.npg)
                   if abs(gP - gQ + so) < self.nc and self.nQ[gQ]]
            inp = [self.nD[gP - gQ + sT] * self.nQ[gQ] for gQ in range(self.npg)
                   if abs(gP - gQ + sT) < self.nc and abs(gP - gQ + so) < self.nc and self.nQ[gQ]]
            unit = max(unit, sum(out) + max(inp, default=0))
        return unit

    def _batch2(self, sT, so, mem_mb=None):
        isz = np.dtype(self.dtype).itemsize
        mem_mb = self.eng.mem_mb if mem_mb is None else mem_mb
        return max(1, int(mem_mb * 1e6 * 2 / 3 // max(self._unit2(sT, so) * isz, 1)))

    def _unit1(self, sT):
        '''Elements of one gathered block of pass 1 per occupied pair.'''
        nv = [b - a for a, b in self.vr]
        no = [b - a for a, b in self.orr]
        unit = 1
        for gP in range(self.npg):
            if not self.nP[gP]:
                continue
            for sd in range(self.nc):
                for sm in range(self.nc):
                    gQ = gP + sm - sd + sT
                    if 0 <= gQ < self.npg:
                        unit = max(unit, nv[sd] * no[sm] * self.nQ[gQ])
        return unit

    def _batch1(self, sT, mem_mb=None):
        isz = np.dtype(self.dtype).itemsize
        mem_mb = self.eng.mem_mb if mem_mb is None else mem_mb
        return max(1, int(mem_mb * 1e6 / 2 // max(self._unit1(sT) * isz, 1)))

    def _unit_pp(self, sT, so):
        '''Elements per occupied triple of a batch of the pp ladder.'''
        ncs = [b - a for a, b in self.vr]
        unit = 1
        for g in range(self.ntg):
            out = [self.nQ[g + so - sc] * ncs[sc] for sc in range(self.nc) if 0 <= g + so - sc < self.npg]
            inp = [self.nQ[g + sT - sc] * ncs[sc] for sc in range(self.nc) if 0 <= g + sT - sc < self.npg]
            unit = max(unit, sum(out) + max(inp, default=0))
        return unit

    def _batch_pp(self, sT, so, mem_mb=None):
        isz = np.dtype(self.dtype).itemsize
        mem_mb = self.eng.mem_mb if mem_mb is None else mem_mb
        return max(1, int(mem_mb * 1e6 * 2 / 3 // max(self._unit_pp(sT, so) * isz, 1)))

    def buffer_bytes(self, mem_mb, shift):
        '''Bytes of the work array that the passes of a sigma vector of
        this shift take with a buffer budget of ``mem_mb`` MB.'''
        isz = np.dtype(self.dtype).itemsize
        n = 1
        for sT, so in ((shift, shift), (0, shift)):
            n = max(n, min(self._batch2(sT, so, mem_mb), max(self.nP)) * self._unit2(sT, so))
            n = max(n, min(self._batch_pp(sT, so, mem_mb), int(max(self.layout(0).nI, default=1)))
                    * self._unit_pp(sT, so))
            n = max(n, min(self._batch1(sT, mem_mb), max(self.nP)) * self._unit1(sT))
        return n * isz

    def occ_bytes(self, shift):
        '''Bytes of the largest block of the occupied operator of a shift
        (occ_blocks: the matrix on the triples of the groups g .. g + shift).'''
        om = self.om
        n = 0
        for g in range(self.ntg):
            h = g + shift
            if 0 <= h < self.ntg:
                n = max(n, int(om.tri_off[max(g, h) + 1] - om.tri_off[min(g, h)]))
        return n * n * np.dtype(self.dtype).itemsize

    # ---- pass 1

    def pass1(self, T3, sT, G_Pd, G_mQ, gpp, fov, sG=0, xe=True, yj=True):
        '''SpinBlockT3.pass1 for T3 of shift sT and integrals of shift sG:
        Xacc (nv+no, no, nvp), Yall (npo, nv, no+nv), r1 (nv, no), rA (npo,
        nvp).  ``gpp`` None: no r1; ``xe``, ``yj`` False: the rows e of
        G_Pd, the columns j of G_mQ are zero and are left out.'''
        eng, om, vm, dtype = self.eng, self.om, self.vm, self.dtype
        L = self.layout(sT)
        nocc, nvir, npv, nc, npg = om.n, vm.n, vm.npair, self.nc, self.npg
        Xacc = np.zeros((nvir + nocc, nocc, npv), dtype=dtype)
        Yall = np.zeros((om.npair, nvir, nocc + nvir), dtype=dtype)
        r1 = np.zeros((nvir, nocc), dtype=dtype)
        rA = np.zeros((om.npair, npv), dtype=dtype)
        G3 = G_mQ.reshape(nocc, npv, nocc + nvir)
        if gpp is not None:
            gpp = np.ascontiguousarray(gpp, dtype=dtype)
        vr, orr = self.vr, self.orr
        gy_cache = {}

        def gy(sm, gQ, sy):
            key = (sm, gQ, sy)
            if key not in gy_cache:
                (m0, m1), (j0, j1), (e0, e1) = orr[sm], orr[sy], vr[sy]
                Qlo, Qhi = vm.pair_off[gQ], vm.pair_off[gQ + 1]
                nj = (j1 - j0) if yj else 0
                G = np.empty((m1 - m0, Qhi - Qlo, nj + e1 - e0), dtype=dtype)
                if nj:
                    t3pack.axpby(G[:, :, :nj], G3[m0:m1, Qlo:Qhi, j0:j1], 1.0, 0.0)
                t3pack.axpby(G[:, :, nj:], G3[m0:m1, Qlo:Qhi, nocc + e0:nocc + e1], 1.0, 0.0)
                gy_cache[key] = G.reshape((m1 - m0) * (Qhi - Qlo), -1)
            return gy_cache[key]

        nb1 = self._batch1(sT)
        for gP in range(npg):
            for pb in _batches(om.pair_off[gP], om.pair_off[gP + 1], nb1):
                p0, p1 = pb[0], pb[-1] + 1
                for sd in range(nc):
                    for sm in range(nc):
                        gQ = gP + sm - sd + sT
                        if not 0 <= gQ < npg or L.nA[gP + sm] == 0:
                            continue
                        d, m = eng.vrng[sd], eng.orng[sm]
                        Qlo, Qhi = vm.pair_off[gQ], vm.pair_off[gQ + 1]
                        nd, nm, nQ = len(d), len(m), Qhi - Qlo
                        if nd == 0 or nm == 0 or nQ == 0:
                            continue
                        (d0, d1), (m0, m1) = vr[sd], orr[sm]
                        sx = gP - sd - sG             # class of the rows (e | j) of G_Pd
                        sy = gP - sd + sT + sG        # class of the columns (j | e) of G_mQ
                        do_x = 0 <= sx < nc and ((xe and vr[sx][1] > vr[sx][0]) or orr[sx][1] > orr[sx][0])
                        do_y = 0 <= sy < nc and ((yj and orr[sy][1] > orr[sy][0]) or vr[sy][1] > vr[sy][0])
                        do_r1 = gpp is not None and gP - gQ == sG
                        do_ra = sm - sd == sG
                        if not (do_x or do_y or do_r1 or do_ra):
                            continue
                        with timer('pass1.gather'):
                            T = eng._workspace(len(pb) * nd * nm * nQ).reshape(len(pb), nd, nm, nQ)
                            self._gather(L, T3, pb, d, m, Qlo, Qhi, T, 0, nd * nm * nQ)
                            Tf = T.reshape(len(pb) * nd, nm * nQ)
                        if do_x:
                            (e0, e1), (j0, j1) = vr[sx], orr[sx]
                            ne = (e1 - e0) if xe else 0
                            G = np.concatenate((G_Pd[e0:e0 + ne, p0:p1, d0:d1],
                                                G_Pd[nvir + j0:nvir + j1, p0:p1, d0:d1]), axis=0)
                            with timer('pass1.gemm'):
                                Gc = np.ascontiguousarray(G, dtype=dtype).reshape(G.shape[0], -1)
                                X = np.empty((Gc.shape[0], Tf.shape[1]), dtype=dtype)
                                gemm_tasks([(Gc.shape[0], Tf.shape[1], Gc.shape[1], 0, Gc.shape[1],
                                             0, Tf.shape[1], 0, Tf.shape[1])], Gc, Tf, X, cols=256)
                            with timer('pass1.acc'):
                                X = X.reshape(-1, nm, nQ)
                                if ne:
                                    t3pack.axpby(Xacc[e0:e1, m0:m1, Qlo:Qhi], X[:ne])
                                t3pack.axpby(Xacc[nvir + j0:nvir + j1, m0:m1, Qlo:Qhi], X[ne:])
                        if do_y:
                            (e0, e1), (j0, j1) = vr[sy], orr[sy]
                            nj = (j1 - j0) if yj else 0
                            with timer('pass1.gemm'):
                                Gy = gy(sm, gQ, sy)
                                nrow, ncol, ny = Tf.shape[0], Tf.shape[1], Gy.shape[1]
                                Y = np.empty((nrow, ny), dtype=dtype)
                                r0 = np.arange(0, nrow, 8, dtype=np.int64)
                                tasks = np.empty((len(r0), 9), dtype=np.int64)
                                tasks[:] = (0, ny, ncol, 0, ncol, 0, ny, 0, ny)
                                tasks[:, 0] = np.minimum(r0 + 8, nrow) - r0
                                tasks[:, 3] = r0 * ncol
                                tasks[:, 7] = r0 * ny
                                gemm_tasks(tasks, Tf, Gy, Y, cols=max(ny, 1))
                                Y = Y.reshape(len(pb), nd, -1)
                            with timer('pass1.acc'):
                                if nj:
                                    t3pack.axpby(Yall[p0:p1, d0:d1, j0:j1], Y[:, :, :nj])
                                t3pack.axpby(Yall[p0:p1, d0:d1, nocc + e0:nocc + e1], Y[:, :, nj:])
                        if do_r1 or do_ra:
                            with timer('pass1.r1rA'):
                                nbp, ndm = len(pb), nd * nm
                                ks = np.arange(nbp, dtype=np.int64)
                                tasks = np.empty((nbp, 9), dtype=np.int64)
                                if do_r1:       # r1[(d m)] += T[P, (d m), Q] gpp[P, Q]
                                    acc = np.empty((nbp, ndm), dtype=dtype)
                                    tasks[:] = (ndm, 1, nQ, 0, nQ, 0, 1, 0, 1)
                                    tasks[:, 3] = ks * (ndm * nQ)
                                    tasks[:, 5] = (p0 + ks) * npv + Qlo
                                    tasks[:, 7] = ks * ndm
                                    gemm_tasks(tasks, T, gpp, acc)
                                    r1[d0:d1, m0:m1] += acc.sum(axis=0).reshape(nd, nm)
                                if do_ra:       # rA[P, Q] += f[(d m)] T[P, (d m), Q]
                                    fdm = np.ascontiguousarray(fov[m0:m1, d0:d1].T, dtype=dtype).ravel()
                                    tasks[:] = (1, nQ, ndm, 0, ndm, 0, nQ, 0, nQ)
                                    tasks[:, 5] = ks * (ndm * nQ)
                                    tasks[:, 7] = (p0 + ks) * npv + Qlo
                                    gemm_tasks(tasks, fdm, T, rA, beta=1.0, cols=max(nQ, 1))
        scratch.unmap(T3)
        return Xacc, Yall, r1, rA

    def _gather(self, L, T3, Pidx, didx, midx, Qlo, Qhi, out, off, ldP):
        om, vm = self.om, self.vm
        vci, vcs = self._vtab()
        f = getattr(t3pack._lib, 'zc_bgather2_' + t3pack._sfx(T3.dtype))
        I, p = _CArgs.I, _CArgs.p
        f(p(T3), p(L.obase), p(L.osig), p(om.tri_c), p(om.sgn_c), I(om.npair), p(vci),
          p(vcs), I(vm.npair), I(len(Pidx)), p(Pidx), I(len(didx)), p(didx), I(len(midx)),
          p(midx), I(Qlo), I(Qhi - Qlo), I(ldP), p(out, off))

    def _vtab(self):
        '''The virtual tables of the gathers (t3engine._vtables): the column
        of a virtual triple in its group does not depend on the shift.'''
        t = getattr(self, '_vt', None)
        if t is None:
            vm, L = self.vm, self.layout(0)
            sg = vm.sgn_c.astype(np.int64)
            ci = np.where(sg != 0, L.vcol[vm.tri_c], 0)
            assert ci.max(initial=0) < 2 ** 31
            t = self._vt = (np.ascontiguousarray(ci, dtype=np.int32), np.ascontiguousarray(sg.astype(np.int8)))
        return t

    # ---- residual

    def residual(self, T3, sT, WringT, t2mn, Wv, Wo, t2P, Wpp, O6, sW=0, sa=0, out=None, clear=True):
        '''SpinBlockT3.residual for T3 of shift sT: the ring operator
        WringT, the pp operator Wpp and the occupied operator O6 have the
        shift sW, the amplitudes of the drive terms (t2mn, t2P) the shift
        sa and their operators Wv, Wo the shift sT + sW - sa.  The result
        (flat, layout(sT + sW)) is new or accumulated in ``out``.

        Wpp: the blocks by their column block h, of shape (pairs of block
        h + sW, pairs of block h) - for sW = 0 the list of the engine.
        O6: a t3kern.OccOperator (sW = 0), or (W_oooo, F_oo) of shift sW.'''
        so = sT + sW
        Lin, Lout = self.layout(sT), self.layout(so)
        R = _target(out, (Lout.size,), self.dtype, clear)
        if Lout.size == 0:
            return R
        with timer('pass2'):
            self._pass2(T3, Lin, Lout, WringT, t2mn, Wv, Wo, t2P, R, sW, sa)
        with timer('pp_ladder'):
            if Lin.size:
                self._ppl(T3, Lin, Lout, Wpp, R, sW)
        scratch.unmap(T3)
        with timer('hh_ladder'):
            if Lin.size:
                self._hh(T3, Lin, Lout, O6, R, sW)
        return R

    def _targets(self, Lout, p0, p1):
        '''SpinBlockT3._red_targets for the layout of the result.'''
        key = (Lout.shift, p0, p1)
        if key not in self._redt:
            om = self.om
            out = []
            for g in range(self.ntg):
                A0, A1 = Lout.vrange(g)
                its = np.where((Lout.Rbase >= 0) & (om.tri_grp == g))[0]
                if A1 == A0 or len(its) == 0:
                    continue
                tb, tx, tP, ts = [], [], [], []
                for r, sg in ((0, 1.0), (1, -1.0), (2, 1.0)):
                    P = om.pair_c[r, its]
                    sel = (P >= p0) & (P < p1)
                    tb.append(Lout.Rbase[its[sel]])
                    tx.append(om.free_c[r, its[sel]])
                    tP.append(P[sel] - p0)
                    ts.append(np.full(int(sel.sum()), sg))
                tb = np.concatenate(tb)
                if len(tb) == 0:
                    continue
                c = lambda x, t=np.int64: np.ascontiguousarray(np.concatenate(x), dtype=t)  # noqa: E731
                out.append((np.ascontiguousarray(tb, dtype=np.int64), c(tx), c(tP), c(ts, np.float64), A0, A1))
            if len(self._redt) > 4096:
                self._redt.clear()
            self._redt[key] = out
        return self._redt[key]

    def _pass2(self, T3, Lin, Lout, WringT, t2mn, Wv, Wo, t2P, R, sW, sa):
        eng, om, vm, dtype = self.eng, self.om, self.vm, self.dtype
        nocc, nvir, nc, npg = om.n, vm.n, self.nc, self.npg
        vr, orr, nD = self.vr, self.orr, self.nD
        sT, so = Lin.shift, Lout.shift
        sb = so - sa
        Wd = {}
        if Lin.size:
            for din in eng.deltas:
                dout = din + sW
                if abs(dout) < nc and nD[din] and nD[dout]:
                    fo = self.rows[dout][0] * nocc + self.rows[dout][1]
                    fi = self.rows[din][0] * nocc + self.rows[din][1]
                    Wd[din] = np.ascontiguousarray(WringT[fo[:, None], fi], dtype=dtype)
        # drive right-hand sides B[(e|m), (i, Q)] per (gQ, s_i): e of class gQ - s_i - sb
        # (W_vvvo[Q, e, i]), m of class gQ - s_i - sa (t2[m, i, Q])
        Bd = {}
        for gQ in range(npg):
            Qlo, Qhi = vm.pair_off[gQ], vm.pair_off[gQ + 1]
            if Qhi == Qlo:
                continue
            for si in range(nc):
                i0, i1 = orr[si]
                se, sm = gQ - si - sb, gQ - si - sa
                parts = []
                e0, e1 = vr[se] if 0 <= se < nc else (0, 0)
                m0, m1 = orr[sm] if 0 <= sm < nc else (0, 0)
                if i1 == i0 or (e1 == e0 and m1 == m0):
                    continue
                parts.append(Wv[Qlo:Qhi, e0:e1, i0:i1].transpose(1, 2, 0))          # [e, i, Q]
                parts.append(t2P[m0:m1, i0:i1, Qlo:Qhi])                            # [m, i, Q]
                Bd[gQ, si] = (np.ascontiguousarray(np.concatenate(parts, axis=0), dtype=dtype).reshape(
                    (e1 - e0) + (m1 - m0), -1), (e0, e1), (m0, m1))
        nQg = np.array(self.nQ, dtype=np.int64)
        Qoff = np.ascontiguousarray(vm.pair_off[:npg], dtype=np.int64)
        fred2 = getattr(t3pack._lib, 'zc_breduce2_' + t3pack._sfx(dtype))
        I_, p_ = _CArgs.I, _CArgs.p
        nb2 = self._batch2(sT, so)
        for gP in range(npg):
            gQs = [gQ for gQ in range(npg) if abs(gP - gQ + so) < nc and nQg[gQ] > 0 and nD[gP - gQ + so]]
            if not gQs:
                continue
            for pb in _batches(om.pair_off[gP], om.pair_off[gP + 1], nb2):
                p0, p1, nbp = pb[0], pb[-1] + 1, len(pb)
                targets = self._targets(Lout, p0, p1)
                if not targets:
                    continue
                zoff = -np.ones((nbp, npg), dtype=np.int64)
                per, loc = 0, {}
                for gQ in gQs:
                    loc[gQ] = per
                    per += nD[gP - gQ + so] * int(nQg[gQ])
                for gQ in gQs:
                    zoff[:, gQ] = np.arange(nbp) * per + loc[gQ]
                off = nbp * per
                tmax = max([nD[gP - gQ + sT] * int(nQg[gQ]) for gQ in gQs if (gP - gQ + sT) in Wd] + [0]) * nbp
                ws = eng._workspace(off + tmax)
                Z = ws[:off]
                Zk = Z.reshape(nbp, per)
                for gQ in gQs:
                    dout, din = gP - gQ + so, gP - gQ + sT
                    Qlo, Qhi = vm.pair_off[gQ], vm.pair_off[gQ + 1]
                    nQ = Qhi - Qlo
                    # rows (a, i) of the result: only those whose group g = gP + s(i) has a block
                    r0, sub = 0, []
                    for sa_, si in eng.dsub[dout]:
                        n = (vr[sa_][1] - vr[sa_][0]) * (orr[si][1] - orr[si][0])
                        if n and Lout.nA[gP + si] > 0:
                            sub.append((r0, n, sa_, si))
                        r0 += n
                    if not sub:
                        continue
                    if din in Wd:
                        nr = nD[din]
                        T = ws[off:off + nbp * nr * nQ]
                        r0 = 0
                        with timer('pass2.gather'):
                            for sd, sm in eng.dsub[din]:
                                d, m = eng.vrng[sd], eng.orng[sm]
                                if len(d) and len(m):
                                    if Lin.nA[gP + sm] > 0:
                                        self._gather(Lin, T3, pb, d, m, Qlo, Qhi, T, r0 * nQ, nr * nQ)
                                    else:
                                        T.reshape(nbp, nr, nQ)[:, r0:r0 + len(d) * len(m)] = 0
                                r0 += len(d) * len(m)
                        with timer('pass2.ring'):
                            tasks = np.empty((len(sub), nbp, 9), dtype=np.int64)
                            for j, (r0, n, _, _) in enumerate(sub):
                                tasks[j] = (n, nQ, nr, r0 * nr, nr, 0, nQ, 0, nQ)
                                tasks[j, :, 5] = np.arange(nbp) * (nr * nQ)
                                tasks[j, :, 7] = zoff[:, gQ] + r0 * nQ
                            gemm_tasks(tasks, Wd[din], T, Z)
                    else:
                        Zk[:, loc[gQ]:loc[gQ] + nD[dout] * nQ] = 0
                    with timer('pass2.drive'):
                        for r0, n, sc, si in sub:
                            if (gQ, si) not in Bd:
                                continue
                            B, (e0, e1), (m0, m1) = Bd[gQ, si]
                            (c0, c1), (i0, i1) = vr[sc], orr[si]
                            ncc, ni = c1 - c0, i1 - i0
                            # (e of class gP - sc + sa = gQ - si - sb, m of class gP - sc + sb)
                            A = np.ascontiguousarray(np.concatenate(
                                (t2mn[p0:p1, e0:e1, c0:c1].transpose(0, 2, 1),
                                 Wo[m0:m1, c0:c1, p0:p1].transpose(2, 1, 0)), axis=2), dtype=dtype)
                            kk = A.shape[2]
                            tasks = np.empty((nbp, 9), dtype=np.int64)
                            tasks[:] = (ncc, ni * nQ, kk, 0, kk, 0, ni * nQ, 0, ni * nQ)
                            tasks[:, 3] = np.arange(nbp) * (ncc * kk)
                            tasks[:, 7] = zoff[:, gQ] + r0 * nQ
                            gemm_tasks(tasks, A, B, Z, alpha=-1.0, beta=1.0)
                with timer('pass2.reduce'):
                    for tb, tx, tP, tsg, A0, A1 in targets:
                        fred2(p_(R), p_(Lout.vcol), I_(len(tb)), p_(tb), p_(tx), p_(tP),
                              p_(tsg), I_(A0), I_(A1), I_(vm.ntri), p_(vm.free_c),
                              p_(vm.pair_c), p_(vm.pair_grp), p_(zoff), I_(npg), p_(nQg), p_(Qoff),
                              p_(eng.rowpos), I_(nocc), p_(Z))

    def _ppl(self, T3, Lin, Lout, Wpp, R, sW):
        eng, om, vm, dtype = self.eng, self.om, self.vm, self.dtype
        nc, npg, nQ, vr = self.nc, self.npg, self.nQ, self.vr
        sT, so = Lin.shift, Lout.shift
        lib = t3pack._lib
        fg = getattr(lib, 'zc_bgather_vir_' + t3pack._sfx(dtype))
        fr = getattr(lib, 'zc_breduce_ppl_' + t3pack._sfx(dtype))
        I_, p_ = _CArgs.I, _CArgs.p
        Wh = {}
        for h in range(npg):
            if 0 <= h + sW < npg and nQ[h] and nQ[h + sW] and Wpp[h] is not None:
                Wh[h] = np.ascontiguousarray(Wpp[h], dtype=dtype)
                assert Wh[h].shape == (nQ[h + sW], nQ[h])
        vspin = np.ascontiguousarray(vm.spins, dtype=np.int64)
        coff = np.array([vr[s][0] for s in range(nc)], dtype=np.int64)
        ncs = np.array([vr[s][1] - vr[s][0] for s in range(nc)], dtype=np.int64)
        dummy = np.zeros(1, dtype=dtype)
        nbt = self._batch_pp(sT, so)
        for g in range(self.ntg):
            A0, A1 = Lout.vrange(g)
            if A1 == A0 or Lin.nA[g] == 0 or Lin.nI[g] == 0:
                continue
            size, inp = {}, {}
            for sc in range(nc):
                ho = g + so - sc
                if 0 <= ho < npg and ncs[sc] and nQ[ho]:
                    size[sc] = nQ[ho] * int(ncs[sc])
                    if (ho - sW) in Wh:
                        inp[sc] = nQ[ho - sW] * int(ncs[sc])
            if not inp:
                continue
            nx = sum(size.values())
            for rows in _batches(om.tri_off[g], om.tri_off[g + 1], nbt):
                nb = len(rows)
                X = [dummy] * nc
                hoff = np.zeros(nc, dtype=np.int64)
                ws = eng._workspace((nx + max(inp.values())) * nb)
                xo = 0
                for sc in sorted(size):
                    ho = g + so - sc
                    X[sc] = ws[xo:xo + size[sc] * nb].reshape(nQ[ho], -1)
                    if sc in inp:
                        hi = ho - sW
                        T = ws[nx * nb:nx * nb + inp[sc] * nb].reshape(nQ[hi], nb, ncs[sc])
                        fg(p_(T3), p_(Lin.obase), p_(Lin.vcol), p_(Lin.osig), p_(Lin.vsig),
                           p_(vm.tri_c), p_(vm.sgn_c), I_(vm.npair), I_(nb), p_(rows),
                           I_(vm.pair_off[hi]), I_(nQ[hi]), I_(coff[sc]), I_(ncs[sc]), p_(T))
                        gemm(Wh[hi], T.reshape(nQ[hi], -1), X[sc])
                    else:
                        X[sc][:] = 0
                    xo += size[sc] * nb
                    hoff[sc] = vm.pair_off[ho]
                xs = (ctypes.c_void_p * nc)(*[x.ctypes.data for x in X])
                fr(p_(R), p_(Lout.Rbase), p_(Lout.vcol), I_(nb), p_(rows), I_(vm.ntri),
                   I_(A0), I_(A1), p_(vm.free_c), p_(vm.pair_c), p_(vspin), p_(hoff),
                   p_(ncs), p_(coff), xs)

    def occ_blocks(self, W, F, sW):
        '''g -> O[triples of group g, triples of group g + sW] of the
        occupied operator of (W_oooo, F_oo) (t3pack.occ_operator_direct on
        the range of triples that holds both groups).'''
        om = self.om
        for g in range(self.ntg):
            h = g + sW
            if not 0 <= h < self.ntg:
                continue
            I0, I1, J0, J1 = om.tri_off[g], om.tri_off[g + 1], om.tri_off[h], om.tri_off[h + 1]
            if I1 == I0 or J1 == J0:
                continue
            lo, hi = int(min(I0, J0)), int(max(I1, J1))
            O = t3pack.occ_operator_direct(om, W, F, (lo, hi), self.dtype)
            yield g, np.ascontiguousarray(O[I0 - lo:I1 - lo, J0 - lo:J1 - lo])

    def _hh(self, T3, Lin, Lout, O6, R, sW):
        om, dtype = self.om, self.dtype
        isz = np.dtype(dtype).itemsize
        if not isinstance(O6, tuple):
            assert sW == 0
            for g in range(self.ntg):
                if Lin.nI[g] == 0 or Lin.nA[g] == 0:
                    continue
                Tg, Rg = Lin.block(T3, g), Lout.block(R, g)
                rows = (int(om.tri_off[g]), int(om.tri_off[g + 1]))
                if isinstance(O6, t3kern.OccOperator):
                    O6.apply(Tg, Rg, rows, t3engine.occ_cols(Lin.nI[g], isz, self.eng.mem_mb), None)
                else:
                    gemm(np.ascontiguousarray(O6[rows[0]:rows[1], rows[0]:rows[1]], dtype=dtype), Tg, Rg, beta=1.0)
            return
        W, F = O6
        for g, O in self.occ_blocks(W, F, sW):
            h = g + sW
            if Lout.nA[g] == 0 or Lin.nA[h] == 0:
                continue
            gemm(O, Lin.block(T3, h), Lout.block(R, g), beta=1.0)
