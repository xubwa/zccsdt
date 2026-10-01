'''T3 engines: storage of the packed T3 amplitudes and the T3-dependent steps
of a CCSDT iteration (pass 1, the T3 residual and the update).

DenseT3       T3 stored as the full packed matrix T3p[I, A] (i<j<k, a<b<c);
              C kernels (csrc/ccsdt_t3.c) or numpy.
SpinBlockT3   orbitals carry alpha/beta labels (spin-grouped enumerations,
              see t3pack.TripleMap): only Ms-conserving blocks (number of beta
              indices g = 0..3 equal in I and A) are stored and every
              contraction is split into spin blocks, skipping zero blocks.
              With flip=True (closed-shell RHF) only g = 0, 1 are stored and
              computed; g = 2, 3 follow from alpha <-> beta spin flip.

Both engines take the same dense intermediates and return the same dense
pass-1 quantities, so ccsdt_opt.update_amps does not depend on the engine.
'''

import ctypes

import numpy as np

from zccsdt.blas import gemm
from zccsdt.cc import t3pack, t3kern
from zccsdt.timing import timer


def _batches(lo, hi, size):
    return [np.arange(i, min(i + size, hi)) for i in range(lo, hi, size)]


def pair_batch_size(nocc, nvir, itemsize, mem_mb):
    '''Occupied pairs per batch for half-expanded buffers of
    nocc * nvir * nvpair elements each (a few are alive at once).'''
    per_pair = 3 * nocc * nvir * nvir * max(nvir - 1, 1) // 2 * itemsize
    return max(1, int(mem_mb * 1e6 // max(per_pair, 1)))


# --------------------------------------------------------------------- dense

class DenseT3:
    def __init__(self, om, vm, dtype, mem_mb):
        self.om, self.vm, self.dtype, self.mem_mb = om, vm, dtype, mem_mb
        self.size = om.ntri * vm.ntri
        nocc, nvir = om.n, vm.n
        self.nb = pair_batch_size(nocc, nvir, np.dtype(dtype).itemsize, mem_mb)

    def view(self, T3):
        return np.ascontiguousarray(np.asarray(T3, dtype=self.dtype).reshape(
            self.om.ntri, self.vm.ntri))

    def pass1(self, T3, G_Pd, G_mQ, gpp, fov):
        '''Returns Xacc (nv+no, no, nvp), Yall (npo, nv, no+nv), r1 (nv, no),
        rA (npo, nvp); see ccsdt_opt for their meaning.'''
        om, vm, dtype = self.om, self.vm, self.dtype
        nocc, nvir = om.n, vm.n
        T3p = self.view(T3)
        fovT = np.ascontiguousarray(fov.T).ravel()                 # [(e, m)]
        if t3kern.USE_C:
            Xacc, Yall, r1, rA = t3kern.pass1(T3p, om, vm, self.nb, G_Pd, G_mQ, gpp, fovT)
            return Xacc.reshape(nvir + nocc, nocc, vm.npair), Yall, r1.reshape(nvir, nocc), rA
        Xacc = np.zeros((nvir + nocc, nocc * vm.npair), dtype=dtype)
        Yall = np.zeros((om.npair, nvir, nocc + nvir), dtype=dtype)
        r1 = np.zeros(nvir * nocc, dtype=dtype)
        rA = np.zeros((om.npair, vm.npair), dtype=dtype)
        for pb in _batches(0, om.npair, self.nb):
            nbp = len(pb)
            T = t3pack.gather_half(T3p, om, vm, pb)              # (Pb, d, m, Q)
            Tf = T.reshape(nbp * nvir, nocc * vm.npair)
            Xacc += np.ascontiguousarray(G_Pd[:, pb]).reshape(nvir + nocc, -1) @ Tf
            Yall[pb] = (Tf @ G_mQ).reshape(nbp, nvir, nocc + nvir)
            Tm = T.reshape(nbp, nvir * nocc, vm.npair)
            for k, P in enumerate(pb):
                r1 += Tm[k] @ gpp[P]
            rA[pb] = np.matmul(fovT, Tm)
        return Xacc.reshape(nvir + nocc, nocc, vm.npair), Yall, r1.reshape(nvir, nocc), rA

    def residual(self, T3, WringT, t2mn, Wv, Wo, t2P, Wpp, O6):
        '''T3 residual (ring + drive, pp ladder, hh ladder) as a dense
        (ntri_o, ntri_v) array.'''
        om, vm, dtype = self.om, self.vm, self.dtype
        nocc, nvir = om.n, vm.n
        T3p = self.view(T3)
        if isinstance(Wpp, (list, tuple)):          # diagonal pair blocks
            if len(Wpp) == 1:
                Wpp = Wpp[0]
            else:                                   # (dense engine on spin maps)
                full = np.zeros((vm.npair, vm.npair), dtype=np.result_type(*Wpp))
                for (lo, hi), blk in zip(vm.blocks, Wpp):
                    full[lo:hi, lo:hi] = blk
                Wpp = full
        R = np.zeros((om.ntri, vm.ntri), dtype=dtype)
        WvT2 = np.ascontiguousarray(Wv.transpose(1, 2, 0)).reshape(nvir, -1)   # [e,(i,Q)]
        t2P_flat = t2P.reshape(nocc, -1)                                     # [m,(k,Q)]
        with timer('pass2'):
            if t3kern.USE_C:
                Adrv = np.concatenate((t2mn.transpose(0, 2, 1), Wo.transpose(2, 1, 0)),
                                      axis=2)                 # (P, c, e | m)
                Bdrv = np.concatenate((WvT2, t2P_flat), axis=0)  # (e | m, (i, Q))
                t3kern.pass2(T3p, om, vm, self.nb, WringT, Adrv, Bdrv, R)
            else:
                for pb in _batches(0, om.npair, self.nb):
                    nbp = len(pb)
                    T = t3pack.gather_half(T3p, om, vm, pb)
                    Tm = T.reshape(nbp, nvir * nocc, vm.npair)
                    Z = np.empty((nbp, nvir * nocc, vm.npair), dtype=dtype)
                    for k in range(nbp):
                        np.matmul(WringT, Tm[k], out=Z[k])
                    Zf = Z.reshape(nbp * nvir, nocc * vm.npair)
                    gemm(t2mn[pb].transpose(0, 2, 1).reshape(nbp * nvir, nvir), WvT2,
                         Zf, alpha=-1.0, beta=1.0)
                    gemm(Wo[:, :, pb].transpose(2, 1, 0).reshape(nbp * nvir, nocc),
                         t2P_flat, Zf, alpha=-1.0, beta=1.0)
                    t3pack.add_reduce_split(R, Z, om, vm, pb)
        with timer('pp_ladder'):
            nbt = max(1, int(self.mem_mb * 1e6 //
                             (3 * vm.npair * nvir * np.dtype(dtype).itemsize)))
            if t3kern.USE_C:
                t3kern.pp_ladder(T3p, om, vm, nbt, Wpp, R)
            else:
                for rows in _batches(0, om.ntri, nbt):
                    T = t3pack.gather_vir_half(T3p, vm, rows)
                    X = gemm(Wpp, T.reshape(vm.npair, -1))
                    t3pack.add_reduce_pair_first(R, X, vm, rows)
        with timer('hh_ladder'):
            if t3kern.USE_C:
                t3kern.occ_operator(O6, T3p, R)
            else:
                R += O6.astype(dtype) @ T3p
        return R

    def update(self, T3, R, eo, ev):
        T3p = self.view(T3)
        if t3kern.USE_C:
            t3kern.update_into_r(T3p, R, eo, ev)                 # R <- T3p + R / D
            return R.ravel()
        return (T3p + R / (eo[:, None] - ev[None, :])).ravel()


# ---------------------------------------------------------------- spin-block

class T3Layout:
    '''Storage of the Ms-conserving blocks of packed T3.

    Block g (g beta indices in both I and A) is a dense (nI[g], nA[g]) matrix
    at offset boff[g] of a flat vector.  For any occupied triple I and virtual
    triple A of equal group, the element T3[I, A] is
        osig[I] * vsig[A] * flat[obase[I] + vcol[A]].
    Without flip all blocks are stored and the signs are +1.  With flip
    (closed shell) only g = 0, 1 are stored; for g = 2, 3 the tables point to
    the alpha <-> beta flipped element of block 3 - g with the sign of
    re-sorting the flipped indices.
    '''

    def __init__(self, om, vm, flip=False):
        so, sv = om.spins, vm.spins
        self.om, self.vm, self.flip = om, vm, flip
        self.nI = np.diff(om.tri_off)
        self.nA = np.diff(vm.tri_off)
        self.stored = [0, 1] if flip else [0, 1, 2, 3]
        self.boff = {}
        off = 0
        for g in self.stored:
            self.boff[g] = off
            off += int(self.nI[g] * self.nA[g])
        self.size = off
        self.obase = np.zeros(om.ntri, dtype=np.int64)
        self.vcol = np.zeros(vm.ntri, dtype=np.int64)
        self.osig = np.ones(om.ntri)
        self.vsig = np.ones(vm.ntri)
        for g in range(4):
            I = np.arange(om.tri_off[g], om.tri_off[g + 1])
            A = np.arange(vm.tri_off[g], vm.tri_off[g + 1])
            if g in self.stored:
                self.obase[I] = self.boff[g] + (I - om.tri_off[g]) * self.nA[g]
                self.vcol[A] = A - vm.tri_off[g]
            else:
                fI, sI = _flip_triples(om, I)
                fA, sA = _flip_triples(vm, A)
                h = 3 - g
                self.obase[I] = self.boff[h] + (fI - om.tri_off[h]) * self.nA[h]
                self.vcol[A] = fA - vm.tri_off[h]
                self.osig[I] = sI
                self.vsig[A] = sA

    def zeros(self, dtype):
        return np.zeros(self.size, dtype=dtype)

    def block(self, T3, g):
        o = self.boff[g]
        return T3[o:o + self.nI[g] * self.nA[g]].reshape(self.nI[g], self.nA[g])

    def to_dense(self, T3):
        om, vm = self.om, self.vm
        D = np.zeros((om.ntri, vm.ntri), dtype=T3.dtype)
        for g in range(4):
            I = np.arange(om.tri_off[g], om.tri_off[g + 1])
            A = np.arange(vm.tri_off[g], vm.tri_off[g + 1])
            D[I[:, None], A] = (self.osig[I][:, None] * self.vsig[A]
                                * T3[self.obase[I][:, None] + self.vcol[A]])
        return D

    def from_dense(self, D):
        om, vm = self.om, self.vm
        out = np.zeros(self.size, dtype=D.dtype)
        for g in self.stored:
            I = slice(om.tri_off[g], om.tri_off[g + 1])
            A = slice(vm.tri_off[g], vm.tri_off[g + 1])
            self.block(out, g)[:] = D[I, A]
        return out


def _flip_triples(m, idx):
    '''Indices and re-sorting signs of the alpha <-> beta flipped triples.'''
    s = m.spins
    na = int(np.sum(s == 0))
    assert 2 * na == m.n, 'spin flip needs equal numbers of alpha and beta'
    tri = m.triples[idx]
    ft = np.where(tri < na, tri + na, tri - na)
    order = np.argsort(ft, axis=1, kind='stable')
    fs = np.take_along_axis(ft, order, axis=1)
    # parity of the sorting permutation
    inv = (order[:, 0] > order[:, 1]).astype(int) + (order[:, 0] > order[:, 2]) \
        + (order[:, 1] > order[:, 2])
    sign = np.where(inv % 2 == 0, 1.0, -1.0)
    tidx = -np.ones((m.n,) * 3, dtype=np.int64)
    tidx[m.triples[:, 0], m.triples[:, 1], m.triples[:, 2]] = np.arange(m.ntri)
    return tidx[fs[:, 0], fs[:, 1], fs[:, 2]], sign


class SpinBlockT3:
    '''Spin-blocked engine (numpy reference implementation).'''

    def __init__(self, om, vm, dtype, mem_mb, flip=False):
        self.om, self.vm, self.dtype, self.mem_mb = om, vm, dtype, mem_mb
        self.L = T3Layout(om, vm, flip)
        self.size = self.L.size
        so, sv = om.spins, vm.spins
        self.orng = [np.where(so == s)[0] for s in (0, 1)]
        self.vrng = [np.where(sv == s)[0] for s in (0, 1)]
        self.nb = pair_batch_size(om.n, vm.n, np.dtype(dtype).itemsize, mem_mb)
        # composite (d, m) / (a, i) index sets D_delta, delta = s(vir) - s(occ),
        # as lists of (s_vir, s_occ) sub-blocks, each vir-major
        self.dsub = {-1: [(0, 1)], 0: [(0, 0), (1, 1)], 1: [(1, 0)]}
        self.use_c = t3pack.USE_C and hasattr(t3pack._lib, 'zc_bgather_d')
        L = self.L
        # targets of the residual: stored groups only (R in the stored layout)
        self.Rbase = np.where(np.isin(om.tri_grp, L.stored), L.obase, -1).astype(np.int64)
        self.Rcol = np.ascontiguousarray(L.vcol)
        # position of (a, x) inside its D_delta (delta = s(a) - s(x))
        self.rowpos = -np.ones((vm.n, om.n), dtype=np.int64)
        for delta in (-1, 0, 1):
            a, i = self._rows(delta)
            self.rowpos[a, i] = np.arange(len(a))

    # ---- gathers (numpy reference)
    def gather(self, T3, P, d, m, Qlo, Qhi):
        '''T[P, d, m, Q] = t3[m, P_0, P_1, d, Q_0, Q_1] for the given index
        arrays P, d, m and the pair range Q in [Qlo, Qhi); shape
        (len(P), len(d), len(m), Qhi - Qlo).'''
        om, vm, L = self.om, self.vm, self.L
        I = om.tri[m[None, :], P[:, None]]                       # (nP, nm)
        so = om.sgn[m[None, :], P[:, None]] * L.osig[I]
        Q = np.arange(Qlo, Qhi)
        A = vm.tri[d[:, None], Q[None, :]]                       # (nd, nQ)
        sv = vm.sgn[d[:, None], Q[None, :]] * L.vsig[A]
        addr = L.obase[I][:, None, :, None] + L.vcol[A][None, :, None, :]
        return T3[addr] * so[:, None, :, None] * sv[None, :, None, :]

    # ---- pass 1
    def pass1(self, T3, G_Pd, G_mQ, gpp, fov):
        if self.use_c:
            return self._pass1_c(T3, G_Pd, G_mQ, gpp, fov)
        return self._pass1_np(T3, G_Pd, G_mQ, gpp, fov)

    def _pass1_np(self, T3, G_Pd, G_mQ, gpp, fov):
        om, vm, dtype = self.om, self.vm, self.dtype
        nocc, nvir = om.n, vm.n
        npv = vm.npair
        Xacc = np.zeros((nvir + nocc, nocc, npv), dtype=dtype)
        Yall = np.zeros((om.npair, nvir, nocc + nvir), dtype=dtype)
        r1 = np.zeros((nvir, nocc), dtype=dtype)
        rA = np.zeros((om.npair, npv), dtype=dtype)
        G3 = G_mQ.reshape(nocc, npv, nocc + nvir)
        for gP in range(3):
            for pb in _batches(om.pair_off[gP], om.pair_off[gP + 1], self.nb):
                for sd in (0, 1):
                    for sm in (0, 1):
                        gQ = gP + sm - sd
                        if not 0 <= gQ <= 2:
                            continue
                        d, m = self.vrng[sd], self.orng[sm]
                        Qlo, Qhi = vm.pair_off[gQ], vm.pair_off[gQ + 1]
                        if len(d) == 0 or len(m) == 0 or Qhi == Qlo:
                            continue
                        T = self.gather(T3, pb, d, m, Qlo, Qhi)  # (P, d, m, Q)
                        Tf = T.reshape(len(pb) * len(d), len(m) * (Qhi - Qlo))
                        sy = gP - sd                  # spin of e/j (X) and n/a (Y)
                        if 0 <= sy <= 1:
                            ey, jy = self.vrng[sy], nvir + self.orng[sy]
                            rows = np.concatenate((ey, jy))
                            G = np.ascontiguousarray(G_Pd[rows][:, pb][:, :, d]).reshape(
                                len(rows), -1)
                            X = (G @ Tf).reshape(len(rows), len(m), Qhi - Qlo)
                            Xacc[rows[:, None, None], m[None, :, None],
                                 np.arange(Qlo, Qhi)[None, None, :]] += X
                            cols = np.concatenate((self.orng[sy], nocc + self.vrng[sy]))
                            Gy = np.ascontiguousarray(G3[m][:, Qlo:Qhi][:, :, cols]).reshape(
                                -1, len(cols))
                            Y = (Tf @ Gy).reshape(len(pb), len(d), len(cols))
                            Yall[pb[:, None, None], d[None, :, None], cols[None, None, :]] += Y
                        if sd == sm:                  # gQ == gP: r1 and rA terms
                            r1[d[:, None], m] += np.einsum('Pdmq,Pq->dm', T, gpp[pb, Qlo:Qhi])
                            rA[pb[:, None], np.arange(Qlo, Qhi)] += np.einsum(
                                'md,Pdmq->Pq', fov[m[:, None], d], T)
        return Xacc, Yall, r1, rA

    # ---- residual
    def residual(self, T3, WringT, t2mn, Wv, Wo, t2P, Wpp, O6):
        '''T3 residual on the stored blocks (flat, layout of T3).'''
        if self.use_c:
            return self._residual_c(T3, WringT, t2mn, Wv, Wo, t2P, Wpp, O6)
        return self._residual_np(T3, WringT, t2mn, Wv, Wo, t2P, Wpp, O6)

    def _residual_np(self, T3, WringT, t2mn, Wv, Wo, t2P, Wpp, O6):
        '''Numpy reference: blocked GEMMs, scattered into dense buffers and
        reduced with the dense kernels.'''
        om, vm, dtype, L = self.om, self.vm, self.dtype, self.L
        nocc, nvir = om.n, vm.n
        Rd = np.zeros((om.ntri, vm.ntri), dtype=dtype)
        with timer('pass2'):
            self._pass2(T3, WringT, t2mn, Wv, Wo, t2P, Rd)
        with timer('pp_ladder'):
            self._ppl(T3, Wpp, Rd)
        with timer('hh_ladder'):
            for g in L.stored:
                I = slice(om.tri_off[g], om.tri_off[g + 1])
                A = slice(vm.tri_off[g], vm.tri_off[g + 1])
                Rd[I, A] += O6[I, I] @ L.block(T3, g)
        return L.from_dense(Rd)

    def _rows(self, delta):
        '''(vir index, occ index) arrays of D_delta, sub-blocks vir-major.'''
        a, i = [], []
        for sa, si in self.dsub[delta]:
            aa, ii = np.meshgrid(self.vrng[sa], self.orng[si], indexing='ij')
            a.append(aa.ravel())
            i.append(ii.ravel())
        return np.concatenate(a), np.concatenate(i)

    def _pass2(self, T3, WringT, t2mn, Wv, Wo, t2P, Rd):
        om, vm, dtype = self.om, self.vm, self.dtype
        nocc, nvir = om.n, vm.n
        Wd = {}
        for delta in (-1, 0, 1):
            a, i = self._rows(delta)
            flat = a * nocc + i
            Wd[delta] = np.ascontiguousarray(WringT[flat[:, None], flat])
        top = 1 if self.L.flip else 2          # flip: targets g <= 1 need gP <= 1
        for gP in range(top + 1):
            for pb in _batches(om.pair_off[gP], om.pair_off[gP + 1], self.nb):
                nbp = len(pb)
                Zfull = np.zeros((nbp, nvir, nocc, vm.npair), dtype=dtype)
                for gQ in range(3):
                    delta = gP - gQ
                    if abs(delta) > 1:
                        continue
                    Qlo, Qhi = vm.pair_off[gQ], vm.pair_off[gQ + 1]
                    if Qhi == Qlo:
                        continue
                    Qs = np.arange(Qlo, Qhi)
                    blocks = []
                    for sd, sm in self.dsub[delta]:
                        d, m = self.vrng[sd], self.orng[sm]
                        T = self.gather(T3, pb, d, m, Qlo, Qhi)
                        blocks.append(T.reshape(nbp, len(d) * len(m), Qhi - Qlo))
                    T = np.concatenate(blocks, axis=1)              # (P, D, Q)
                    Z = np.matmul(Wd[delta], T)                      # (P, D, Q)
                    r0 = 0
                    for sc, si in self.dsub[delta]:
                        c, i = self.vrng[sc], self.orng[si]
                        nr = len(c) * len(i)
                        Zs = Z[:, r0:r0 + nr].reshape(nbp, len(c), len(i), -1)
                        se = gP - sc
                        if 0 <= se <= 1:
                            e = self.vrng[se]
                            mm = self.orng[se]
                            Zs -= np.einsum('Pec,qei->Pciq', t2mn[pb][:, e][:, :, c],
                                            Wv[Qs][:, e][:, :, i])
                            Zs -= np.einsum('mcP,miq->Pciq', Wo[mm][:, c][:, :, pb],
                                            t2P[mm][:, i][:, :, Qs])
                        Zfull[:, c[:, None, None], i[None, :, None], Qs[None, None, :]] = Zs
                        r0 += nr
                t3pack.add_reduce_split(Rd, Zfull.reshape(nbp, nvir * nocc, vm.npair),
                                        om, vm, pb)

    def _ppl(self, T3, Wpp, Rd):
        om, vm, dtype, L = self.om, self.vm, self.dtype, self.L
        nvir = vm.n
        for g in L.stored:
            rows = np.arange(om.tri_off[g], om.tri_off[g + 1])
            if len(rows) == 0:
                continue
            Xfull = np.zeros((vm.npair, len(rows), nvir), dtype=dtype)
            for sc in (0, 1):
                h = g - sc
                if not 0 <= h <= 2:
                    continue
                Qlo, Qhi = vm.pair_off[h], vm.pair_off[h + 1]
                c = self.vrng[sc]
                if Qhi == Qlo or len(c) == 0:
                    continue
                # T[Q, I, c] = t3[I, Q_0, Q_1, c]
                Q = np.arange(Qlo, Qhi)
                A = vm.tri[c[None, :], Q[:, None]]                  # (Q, c)
                sv = vm.sgn[c[None, :], Q[:, None]] * L.vsig[A]
                addr = L.obase[rows][None, :, None] + L.vcol[A][:, None, :]
                T = T3[addr] * L.osig[rows][None, :, None] * sv[:, None, :]
                Wh = np.ascontiguousarray(Wpp[h])
                X = (Wh @ T.reshape(Qhi - Qlo, -1)).reshape(Qhi - Qlo, len(rows), len(c))
                Xfull[Qlo:Qhi, :, c[0]:c[-1] + 1] = X
            t3pack.add_reduce_pair_first(Rd, Xfull, vm, rows)

    def update(self, T3, R, eo, ev):
        L, om, vm = self.L, self.om, self.vm
        if self.use_c and t3kern.USE_C:
            for g in L.stored:
                if L.nI[g] == 0 or L.nA[g] == 0:
                    continue
                I = slice(om.tri_off[g], om.tri_off[g + 1])
                A = slice(vm.tri_off[g], vm.tri_off[g + 1])
                t3kern.update_into_r(L.block(T3, g), L.block(R, g), eo[I], ev[A])
            return R
        out = np.empty_like(R)
        for g in L.stored:
            I = slice(om.tri_off[g], om.tri_off[g + 1])
            A = slice(vm.tri_off[g], vm.tri_off[g + 1])
            L.block(out, g)[:] = L.block(T3, g) + L.block(R, g) / (
                eo[I][:, None] - ev[A][None, :])
        return out


class _CArgs:
    '''ctypes helpers for the blocked kernels.'''
    @staticmethod
    def I(x):
        return ctypes.c_int64(int(x))

    @staticmethod
    def p(a, off=0):
        return ctypes.c_void_p(int(a.ctypes.data) + int(off) * a.itemsize)


def _cgather(eng, T3, Pidx, didx, midx, Qlo, Qhi, out, off, ldP):
    om, vm, L = eng.om, eng.vm, eng.L
    f = getattr(t3pack._lib, 'zc_bgather_' + t3pack._sfx(T3.dtype))
    I, p = _CArgs.I, _CArgs.p
    f(p(T3), p(L.obase), p(L.vcol), p(L.osig), p(L.vsig), p(om.tri_c), p(om.sgn_c),
      I(om.npair), p(vm.tri_c), p(vm.sgn_c), I(vm.npair), I(len(Pidx)), p(Pidx),
      I(len(didx)), p(didx), I(len(midx)), p(midx), I(Qlo), I(Qhi - Qlo), I(ldP),
      p(out, off))


def _spin_block_c_methods():
    def _pass1_c(self, T3, G_Pd, G_mQ, gpp, fov):
        om, vm, dtype = self.om, self.vm, self.dtype
        nocc, nvir = om.n, vm.n
        npv = vm.npair
        Xacc = np.zeros((nvir + nocc, nocc, npv), dtype=dtype)
        Yall = np.zeros((om.npair, nvir, nocc + nvir), dtype=dtype)
        r1 = np.zeros((nvir, nocc), dtype=dtype)
        rA = np.zeros((om.npair, npv), dtype=dtype)
        G3 = G_mQ.reshape(nocc, npv, nocc + nvir)
        vr = [(r[0], r[-1] + 1) if len(r) else (0, 0) for r in self.vrng]
        orr = [(r[0], r[-1] + 1) if len(r) else (0, 0) for r in self.orng]
        gy_cache = {}           # Y-GEMM operands depend on (sm, gQ, sy) only

        def gy(sm, gQ, sy):
            key = (sm, gQ, sy)
            if key not in gy_cache:
                (m0, m1), (j0, j1), (e0, e1) = orr[sm], orr[sy], vr[sy]
                Qlo, Qhi = vm.pair_off[gQ], vm.pair_off[gQ + 1]
                G = np.concatenate((G3[m0:m1, Qlo:Qhi, j0:j1],
                                    G3[m0:m1, Qlo:Qhi, nocc + e0:nocc + e1]), axis=2)
                gy_cache[key] = np.ascontiguousarray(G).reshape((m1 - m0) * (Qhi - Qlo), -1)
            return gy_cache[key]

        for gP in range(3):
            for pb in _batches(om.pair_off[gP], om.pair_off[gP + 1], self.nb):
                p0, p1 = pb[0], pb[-1] + 1
                for sd in (0, 1):
                    for sm in (0, 1):
                        gQ = gP + sm - sd
                        if not 0 <= gQ <= 2:
                            continue
                        d, m = self.vrng[sd], self.orng[sm]
                        Qlo, Qhi = vm.pair_off[gQ], vm.pair_off[gQ + 1]
                        nd, nm, nQ = len(d), len(m), Qhi - Qlo
                        if nd == 0 or nm == 0 or nQ == 0:
                            continue
                        T = np.empty((len(pb), nd, nm, nQ), dtype=dtype)
                        _cgather(self, T3, pb, d, m, Qlo, Qhi, T, 0, nd * nm * nQ)
                        Tf = T.reshape(len(pb) * nd, nm * nQ)
                        sy = gP - sd
                        if 0 <= sy <= 1:
                            e0, e1 = vr[sy]
                            j0, j1 = orr[sy]
                            G = np.concatenate((G_Pd[e0:e1, p0:p1, vr[sd][0]:vr[sd][1]],
                                                G_Pd[nvir + j0:nvir + j1, p0:p1,
                                                     vr[sd][0]:vr[sd][1]]), axis=0)
                            X = gemm(np.ascontiguousarray(G).reshape(G.shape[0], -1), Tf)
                            X = X.reshape(-1, nm, nQ)
                            m0, m1 = orr[sm]
                            Xacc[e0:e1, m0:m1, Qlo:Qhi] += X[:e1 - e0]
                            Xacc[nvir + j0:nvir + j1, m0:m1, Qlo:Qhi] += X[e1 - e0:]
                            Y = gemm(Tf, gy(sm, gQ, sy)).reshape(len(pb), nd, -1)
                            d0, d1 = vr[sd]
                            Yall[p0:p1, d0:d1, j0:j1] += Y[:, :, :j1 - j0]
                            Yall[p0:p1, d0:d1, nocc + e0:nocc + e1] += Y[:, :, j1 - j0:]
                        if sd == sm:
                            d0, d1 = vr[sd]
                            m0, m1 = orr[sm]
                            Tm = T.reshape(len(pb), nd * nm, nQ)
                            acc = np.zeros(nd * nm, dtype=dtype)
                            for k, P in enumerate(pb):
                                acc += Tm[k] @ gpp[P, Qlo:Qhi]
                            r1[d0:d1, m0:m1] += acc.reshape(nd, nm)
                            fdm = np.ascontiguousarray(fov[m0:m1, d0:d1].T).ravel()
                            rA[p0:p1, Qlo:Qhi] += np.matmul(fdm, Tm)
        return Xacc, Yall, r1, rA

    def _residual_c(self, T3, WringT, t2mn, Wv, Wo, t2P, Wpp, O6):
        om, vm, dtype, L = self.om, self.vm, self.dtype, self.L
        R = np.zeros(L.size, dtype=dtype)
        with timer('pass2'):
            self._pass2_c(T3, WringT, t2mn, Wv, Wo, t2P, R)
        with timer('pp_ladder'):
            self._ppl_c(T3, Wpp, R)
        with timer('hh_ladder'):
            for g in L.stored:
                if L.nI[g] == 0 or L.nA[g] == 0:
                    continue
                I = slice(om.tri_off[g], om.tri_off[g + 1])
                gemm(np.ascontiguousarray(O6[I, I], dtype=dtype), L.block(T3, g),
                     L.block(R, g), beta=1.0)
        return R

    def _pass2_c(self, T3, WringT, t2mn, Wv, Wo, t2P, R):
        om, vm, dtype, L = self.om, self.vm, self.dtype, self.L
        nocc, nvir = om.n, vm.n
        vr = [(r[0], r[-1] + 1) if len(r) else (0, 0) for r in self.vrng]
        orr = [(r[0], r[-1] + 1) if len(r) else (0, 0) for r in self.orng]
        Wd, nD = {}, {}
        for delta in (-1, 0, 1):
            a, i = self._rows(delta)
            flat = a * nocc + i
            Wd[delta] = np.ascontiguousarray(WringT[flat[:, None], flat], dtype=dtype)
            nD[delta] = len(a)
        # drive right-hand sides B[(e|m), (i, Q)] per (gQ, s_i), e and m of spin gQ - s_i
        Bd = {}
        for gQ in range(3):
            Qlo, Qhi = vm.pair_off[gQ], vm.pair_off[gQ + 1]
            for si in (0, 1):
                se = gQ - si
                if not 0 <= se <= 1 or Qhi == Qlo:
                    continue
                (e0, e1), (i0, i1), (m0, m1) = vr[se], orr[si], orr[se]
                B1 = Wv[Qlo:Qhi, e0:e1, i0:i1].transpose(1, 2, 0)          # [e, i, Q]
                B2 = t2P[m0:m1, i0:i1, Qlo:Qhi]                            # [m, i, Q]
                Bd[gQ, si] = np.ascontiguousarray(np.concatenate((B1, B2), axis=0),
                                                  dtype=dtype).reshape(
                    (e1 - e0) + (m1 - m0), -1)
        nQg = np.array([vm.pair_off[g + 1] - vm.pair_off[g] for g in range(3)], dtype=np.int64)
        Qoff = np.ascontiguousarray(vm.pair_off[:3], dtype=np.int64)
        lib = t3pack._lib
        fred = getattr(lib, 'zc_breduce_pass2_' + t3pack._sfx(dtype))
        I_, p_ = _CArgs.I, _CArgs.p
        top = 1 if L.flip else 2
        for gP in range(top + 1):
            for pb in _batches(om.pair_off[gP], om.pair_off[gP + 1], self.nb):
                p0, p1, nbp = pb[0], pb[-1] + 1, len(pb)
                gQs = [gQ for gQ in range(3) if abs(gP - gQ) <= 1 and nQg[gQ] > 0]
                zoff = -np.ones((nbp, 3), dtype=np.int64)
                off = 0
                for k in range(nbp):
                    for gQ in gQs:
                        zoff[k, gQ] = off
                        off += nD[gP - gQ] * nQg[gQ]
                Z = np.empty(off, dtype=dtype)
                for gQ in gQs:
                    delta = gP - gQ
                    Qlo, Qhi = vm.pair_off[gQ], vm.pair_off[gQ + 1]
                    nQ, nr = Qhi - Qlo, nD[delta]
                    T = np.empty(nbp * nr * nQ, dtype=dtype)
                    r0 = 0
                    t_ = timer('pass2.gather'); t_.__enter__()
                    for sd, sm in self.dsub[delta]:
                        d, m = self.vrng[sd], self.orng[sm]
                        if len(d) and len(m):
                            _cgather(self, T3, pb, d, m, Qlo, Qhi, T, r0 * nQ, nr * nQ)
                        r0 += len(d) * len(m)
                    t_.__exit__(None, None, None)
                    T = T.reshape(nbp, nr, nQ)
                    # rows (a, i) only for target groups that are computed:
                    # g = gP + s(i) must be stored (all for UCCSDT; with spin
                    # flip the pairs gP = 1 only need i alpha)
                    r0, sub = 0, []
                    for sa, si in self.dsub[delta]:
                        n = (vr[sa][1] - vr[sa][0]) * (orr[si][1] - orr[si][0])
                        if gP + si in L.stored and n:
                            sub.append((r0, n))
                        r0 += n
                    t_ = timer('pass2.ring'); t_.__enter__()
                    for k in range(nbp):
                        for r0, n in sub:
                            zo = zoff[k, gQ] + r0 * nQ
                            gemm(Wd[delta][r0:r0 + n], T[k], Z[zo:zo + n * nQ].reshape(n, nQ))
                    t_.__exit__(None, None, None)
                    t_ = timer('pass2.drive'); t_.__enter__()
                    # drive terms on the (s_c, s_i) sub-blocks of D_delta
                    r0 = 0
                    for sc, si in self.dsub[delta]:
                        (c0, c1), (i0, i1) = vr[sc], orr[si]
                        nc, ni = c1 - c0, i1 - i0
                        se = gP - sc
                        if nc and ni and 0 <= se <= 1 and (gQ, si) in Bd \
                                and gP + si in L.stored:
                            (e0, e1), (m0, m1) = vr[se], orr[se]
                            for k, P in enumerate(pb):
                                A = np.concatenate((t2mn[P, e0:e1, c0:c1].T,
                                                    Wo[m0:m1, c0:c1, P].T), axis=1)
                                zo = zoff[k, gQ] + r0 * nQ
                                Zs = Z[zo:zo + nc * ni * nQ].reshape(nc, ni * nQ)
                                gemm(np.ascontiguousarray(A, dtype=dtype), Bd[gQ, si], Zs,
                                     alpha=-1.0, beta=1.0)
                        r0 += nc * ni
                    t_.__exit__(None, None, None)
                t_ = timer('pass2.reduce'); t_.__enter__()
                fred(p_(R), p_(self.Rbase), p_(self.Rcol), I_(om.ntri), p_(om.tri_grp),
                     p_(om.free_c), p_(om.pair_c), I_(vm.ntri), p_(vm.tri_off),
                     p_(vm.free_c), p_(vm.pair_c), p_(vm.pair_grp), I_(p0), I_(p1),
                     p_(zoff), p_(nQg), p_(Qoff), p_(self.rowpos), I_(nocc), p_(Z))
                t_.__exit__(None, None, None)

    def _ppl_c(self, T3, Wpp, R):
        om, vm, dtype, L = self.om, self.vm, self.dtype, self.L
        nvir = vm.n
        vr = [(r[0], r[-1] + 1) if len(r) else (0, 0) for r in self.vrng]
        lib = t3pack._lib
        fg = getattr(lib, 'zc_bgather_vir_' + t3pack._sfx(dtype))
        fr = getattr(lib, 'zc_breduce_ppl_' + t3pack._sfx(dtype))
        I_, p_ = _CArgs.I, _CArgs.p
        Wh = {}
        for h in range(3):
            Qlo, Qhi = vm.pair_off[h], vm.pair_off[h + 1]
            Wh[h] = np.ascontiguousarray(Wpp[h], dtype=dtype)
        vspin = np.ascontiguousarray(vm.spins, dtype=np.int64)
        coff = np.array([vr[0][0], vr[1][0]], dtype=np.int64)
        ncs = np.array([vr[0][1] - vr[0][0], vr[1][1] - vr[1][0]], dtype=np.int64)
        dummy = np.zeros(1, dtype=dtype)
        nbt = max(1, int(self.mem_mb * 1e6 // (3 * vm.npair * nvir * np.dtype(dtype).itemsize)))
        for g in L.stored:
            A0, A1 = vm.tri_off[g], vm.tri_off[g + 1]
            if A1 == A0:
                continue
            for rows in _batches(om.tri_off[g], om.tri_off[g + 1], nbt):
                nb = len(rows)
                X = [dummy, dummy]
                hoff = np.zeros(2, dtype=np.int64)
                for sc in (0, 1):
                    h = g - sc
                    if not 0 <= h <= 2 or ncs[sc] == 0:
                        continue
                    Qlo, Qhi = vm.pair_off[h], vm.pair_off[h + 1]
                    if Qhi == Qlo:
                        continue
                    T = np.empty((Qhi - Qlo, nb, ncs[sc]), dtype=dtype)
                    fg(p_(T3), p_(L.obase), p_(L.vcol), p_(L.osig), p_(L.vsig),
                       p_(vm.tri_c), p_(vm.sgn_c), I_(vm.npair), I_(nb), p_(rows),
                       I_(Qlo), I_(Qhi - Qlo), I_(coff[sc]), I_(ncs[sc]), p_(T))
                    X[sc] = gemm(Wh[h], T.reshape(Qhi - Qlo, -1))
                    hoff[sc] = Qlo
                fr(p_(R), p_(self.Rbase), p_(self.Rcol), I_(nb), p_(rows), I_(vm.ntri),
                   I_(A0), I_(A1), p_(vm.free_c), p_(vm.pair_c), p_(vspin), p_(hoff),
                   p_(ncs), p_(coff), p_(X[0]), p_(X[1]))

    return _pass1_c, _residual_c, _pass2_c, _ppl_c


(SpinBlockT3._pass1_c, SpinBlockT3._residual_c, SpinBlockT3._pass2_c,
 SpinBlockT3._ppl_c) = _spin_block_c_methods()


def make_engine(om, vm, dtype, mem_mb, flip=False):
    if om.spins is None:
        return DenseT3(om, vm, dtype, mem_mb)
    return SpinBlockT3(om, vm, dtype, mem_mb, flip=flip)
