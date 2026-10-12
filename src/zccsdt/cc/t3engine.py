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

T3 and its residual may be views of a mapped scratch file (``io``, set by
update_amps when memory is short, see scratch.io_desc): the kernels then
announce the rows they are about to read and unmap the rows of T3 they have
read, and the new T3 is sent to the file when the update has written it;
nothing below makes a copy of either array.

Batch buffers: both engines size their batches by ``mem_mb`` (set_memory) and
tell the memory planner what that costs (buffer_bytes, mem_min,
``kept_buffers``: see ccsdt_opt.plan_memory).  The dense engine allocates its
buffers per pass; the spin-blocked engine takes them from one work array that
stays allocated between the passes and the iterations (workspace).
'''

import ctypes

import numpy as np

from zccsdt import scratch
from zccsdt.blas import gemm, gemm_tasks
from zccsdt.cc import t3pack, t3kern
from zccsdt.timing import timer


_RED2 = True        # loop-inverted pass-2 reduction (zc_breduce2)


def _batches(lo, hi, size):
    return [np.arange(i, min(i + size, hi)) for i in range(lo, hi, size)]


def pair_batch_size(nocc, nvir, itemsize, mem_mb):
    '''Occupied pairs per batch for half-expanded buffers of
    nocc * nvir * nvpair elements each (a few are alive at once).'''
    per_pair = 3 * nocc * nvir * nvir * max(nvir - 1, 1) // 2 * itemsize
    return max(1, int(mem_mb * 1e6 // max(per_pair, 1)))


def occ_cols(nrow, itemsize, mem_mb):
    '''Columns per GEMM of the occupied operator: blocks of T3 and of the
    residual of mem_mb / 8 each.'''
    return max(64, int(mem_mb * 1e6 / 8 // max(nrow * itemsize, 1)))


_WORK = [None]      # the work array of the spin-blocked engine


def workspace(n, dtype):
    '''A work array of at least n elements, kept between batches, passes and
    iterations (fresh large arrays pay page faults on every first touch).
    One user at a time: callers take non-overlapping views.  It is released
    when the buffer budget of an engine changes (set_memory).'''
    w = _WORK[0]
    if w is None or w.size < n or w.dtype != dtype:
        _WORK[0] = w = None                     # free before allocating again
        w = _WORK[0] = np.empty(n, dtype=dtype)
    return w[:n]


def release_workspace():
    _WORK[0] = None


def workspace_bytes():
    '''Bytes of the work array that is allocated now.'''
    return 0 if _WORK[0] is None else _WORK[0].nbytes


def _target(out, shape, dtype, clear):
    '''The residual array: ``out`` viewed in ``shape`` (zeroed if ``clear``),
    or new zeros.'''
    if out is None:
        return np.zeros(shape, dtype=dtype)
    R = out.reshape(shape)
    assert R.dtype == dtype and R.flags.c_contiguous and np.shares_memory(R, out)
    if clear:
        t3kern.zero(R)
    return R


# --------------------------------------------------------------------- dense

class DenseT3:
    kept_buffers = False        # (for plan_memory) the buffers live for one pass only

    def __init__(self, om, vm, dtype, mem_mb):
        self.om, self.vm, self.dtype = om, vm, dtype
        self.size = om.ntri * vm.ntri
        self.io = None
        self.set_memory(mem_mb)

    def set_memory(self, mem_mb):
        '''Batch sizes for a buffer budget of ``mem_mb`` MB.'''
        self.mem_mb = mem_mb
        self.nb = pair_batch_size(self.om.n, self.vm.n, np.dtype(self.dtype).itemsize, mem_mb)

    def _f(self):
        return self.om.n * self.vm.n ** 3 * np.dtype(self.dtype).itemsize

    def mem_min(self):
        '''The smallest useful ``mem_mb``: one occupied pair per batch.'''
        return 1.5 * self._f() / 1e6

    def buffer_bytes(self, mem_mb, max_rows=None):
        '''Bytes of the batch buffers of a pass with ``mem_mb``: two thirds of
        it for the half-expanded T3 batches, at least one occupied pair and
        at most all of them (f per pair in pass 2).'''
        f = self._f()
        return int(min(max(mem_mb * 1e6 * 2 / 3, f), max(self.om.npair, 1) * f))

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
            Xacc, Yall, r1, rA = t3kern.pass1(T3p, om, vm, self.nb, G_Pd, G_mQ, gpp, fovT,
                                              self.io)
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

    def residual(self, T3, WringT, t2mn, Wv, Wo, t2P, Wpp, O6, out=None, clear=True):
        '''T3 residual (ring + drive, pp ladder, hh ladder) as a dense
        (ntri_o, ntri_v) array: new, or accumulated in ``out`` (zeroed first
        unless ``clear`` is False).  Wpp = None leaves the pp ladder out (it
        is then in ``out`` already, see pp_ladder_rows).'''
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
        R = _target(out, (om.ntri, vm.ntri), dtype, clear)
        t2P_flat = t2P.reshape(nocc, -1)                                     # [m,(k,Q)]
        with timer('pass2'):
            if t3kern.USE_C:
                Adrv = np.concatenate((t2mn.transpose(0, 2, 1), Wo.transpose(2, 1, 0)),
                                      axis=2)                 # (P, c, e | m)
                # (e | m, (i, Q)): Wv[Q, e, i] transposed, then t2P, in one array
                Bdrv = np.empty((nvir + nocc, nocc * vm.npair), dtype=dtype)
                t3pack.axpby(Bdrv[:nvir].reshape(nvir, nocc, vm.npair),
                             np.asarray(Wv, dtype=dtype).transpose(1, 2, 0), 1.0, 0.0)
                Bdrv[nvir:] = t2P_flat
                if WringT is None:      # no ring term (CC3): the drive terms, T3 is not read
                    t3kern.drive(om, vm, self.nb, Adrv, Bdrv, R, self.io)
                else:
                    t3kern.pass2(T3p, om, vm, self.nb, WringT, Adrv, Bdrv, R, self.io)
                del Bdrv
            elif WringT is None:
                WvT2 = np.ascontiguousarray(Wv.transpose(1, 2, 0)).reshape(nvir, -1)
                for pb in _batches(0, om.npair, self.nb):
                    nbp = len(pb)
                    Z = np.zeros((nbp, nvir * nocc, vm.npair), dtype=dtype)
                    Zf = Z.reshape(nbp * nvir, nocc * vm.npair)
                    gemm(t2mn[pb].transpose(0, 2, 1).reshape(nbp * nvir, nvir), WvT2,
                         Zf, alpha=-1.0, beta=1.0)
                    gemm(Wo[:, :, pb].transpose(2, 1, 0).reshape(nbp * nvir, nocc),
                         t2P_flat, Zf, alpha=-1.0, beta=1.0)
                    t3pack.add_reduce_split(R, Z, om, vm, pb)
            else:
                WvT2 = np.ascontiguousarray(Wv.transpose(1, 2, 0)).reshape(nvir, -1)  # [e,(i,Q)]
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
            if Wpp is None:
                pass
            elif t3kern.USE_C:
                t3kern.pp_ladder(T3p, om, vm, nbt, Wpp, R, self.io)
            else:
                for rows in _batches(0, om.ntri, nbt):
                    T = t3pack.gather_vir_half(T3p, vm, rows)
                    X = gemm(Wpp, T.reshape(vm.npair, -1))
                    t3pack.add_reduce_pair_first(R, X, vm, rows)
        with timer('hh_ladder'):
            if O6 is None:
                pass
            elif isinstance(O6, t3kern.OccOperator):
                O6.apply(T3p, R, None, occ_cols(om.ntri, np.dtype(dtype).itemsize, self.mem_mb),
                         self.io)
            elif t3kern.USE_C:
                t3kern.occ_operator(O6, T3p, R,
                                    occ_cols(om.ntri, np.dtype(dtype).itemsize, self.mem_mb),
                                    self.io)
            else:
                R += O6.astype(dtype) @ T3p
        return R

    def target(self, out=None, clear=True):
        '''The array the residual is accumulated in (see residual).'''
        return _target(out, (self.om.ntri, self.vm.ntri), self.dtype, clear)

    def pp_ladder_rows(self, T3, slabs, R, order='slab', max_rows=None):
        '''pp ladder with the operator given in slabs of rows: ``slabs()``
        returns an iterator of (q0, q1, W[q0:q1]) over all rows of W(ab,de).
        order 'slab': for each slab all occupied triples (T3 is gathered again
        for every slab, slabs() is called once); 'batch': for each batch of
        triples all slabs (T3 is read once, slabs() is called once per batch).
        ``max_rows``: the largest q1 - q0.  R as returned by target().'''
        om, vm, dtype = self.om, self.vm, self.dtype
        nvir, nvp, isz = vm.n, vm.npair, np.dtype(dtype).itemsize
        T3p = self.view(T3)
        max_rows = nvp if max_rows is None else max_rows
        # the gathered T3 and its product: two thirds of the buffer budget, as
        # the buffers of the other passes
        nbt = max(1, int(self.mem_mb * 1e6 * 2 / 3 // max((nvp + max_rows) * nvir * isz, 1)))
        nbt = min(nbt, max(om.ntri, 1))
        ws = np.empty((nvp + max_rows) * nbt * nvir, dtype=dtype)

        def gather(rows):
            r0, r1 = rows[0] * vm.ntri, (rows[-1] + 1) * vm.ntri
            scratch.need(T3p, r0, r1)
            scratch.need(R, r0, r1)
            with timer('ppl.gather'):
                Tv = t3pack.gather_vir_half(
                    T3p, vm, rows, out=ws[:nvp * len(rows) * nvir].reshape(nvp, len(rows), nvir))
            scratch.unmap(T3p, r0, r1)
            return Tv

        def apply(Tv, rows, q0, q1, W):
            nq = q1 - q0
            X = ws[nvp * nbt * nvir:nvp * nbt * nvir + nq * len(rows) * nvir]
            X = X.reshape(nq, len(rows) * nvir)
            with timer('ppl.gemm'):
                gemm(W, Tv.reshape(nvp, -1), X)
            with timer('ppl.reduce'):
                t3pack.add_reduce_pair_first(R, X.reshape(nq, len(rows), nvir), vm, rows,
                                             pairs=(q0, q1))

        batches = _batches(0, om.ntri, nbt)
        if order == 'slab':
            for q0, q1, W in slabs():
                for rows in batches:
                    apply(gather(rows), rows, q0, q1, W)
        else:
            for rows in batches:
                Tv = gather(rows)
                for q0, q1, W in slabs():
                    apply(Tv, rows, q0, q1, W)

    def update(self, T3, R, eo, ev):
        '''T3 + R / D, written over R.'''
        T3p = self.view(T3)
        if t3kern.USE_C:
            t3kern.update_into_r(T3p, R, eo, ev, self.io)        # R <- T3p + R / D
            return R.ravel()
        R /= eo[:, None] - ev[None, :]
        R += T3p
        scratch.written(R)
        return R.ravel()


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
        self.ntg = om.ntg
        assert not flip or om.ncls == 2
        self.stored = [0, 1] if flip else list(range(om.ntg))
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
        for g in range(self.ntg):
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
        for g in range(self.ntg):
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

    kept_buffers = True         # (for plan_memory) one work array that stays allocated

    def __init__(self, om, vm, dtype, mem_mb, flip=False):
        self.om, self.vm, self.dtype = om, vm, dtype
        self.L = T3Layout(om, vm, flip)
        self.size = self.L.size
        so, sv = om.spins, vm.spins
        assert om.ncls == vm.ncls
        self.nc, self.npg, self.ntg = nc, npg, ntg = om.ncls, om.npg, om.ntg
        self.cls = range(nc)
        self.deltas = range(1 - nc, nc)
        self.orng = [np.where(so == s)[0] for s in range(nc)]
        self.vrng = [np.where(sv == s)[0] for s in range(nc)]
        # composite (d, m) / (a, i) index sets D_delta, delta = s(vir) - s(occ),
        # as lists of (s_vir, s_occ) sub-blocks, each vir-major
        self.dsub = {d: [(sa, sa - d) for sa in range(nc) if 0 <= sa - d < nc] for d in self.deltas}
        self.mem_mb = None
        self.set_memory(mem_mb)
        self.use_c = t3pack.USE_C and hasattr(t3pack._lib, 'zc_bgather_d')
        self.io = None
        L = self.L
        # targets of the residual: stored groups only (R in the stored layout)
        self.Rbase = np.where(np.isin(om.tri_grp, L.stored), L.obase, -1).astype(np.int64)
        self.Rcol = np.ascontiguousarray(L.vcol)
        # position of (a, x) inside its D_delta (delta = s(a) - s(x))
        self.rowpos = -np.ones((vm.n, om.n), dtype=np.int64)
        for delta in self.deltas:
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

    # ---- batch sizes and buffers
    def _batch_sizes(self, mem_mb):
        '''(pairs per batch of pass 2, of pass 1, triples per batch of the pp
        ladder) for a buffer budget of ``mem_mb`` MB.'''
        om, vm = self.om, self.vm
        isz = np.dtype(self.dtype).itemsize
        nb = pair_batch_size(om.n, vm.n, isz, mem_mb)
        # pass 1 keeps one gathered block T[P, d, m, Q] (single spins of d, m)
        # per pair alive: larger batches than pass 2, for larger GEMMs
        nmax = max(len(r) for r in self.orng) * max(len(r) for r in self.vrng)
        nqmax = max(vm.pair_off[g + 1] - vm.pair_off[g] for g in range(self.npg))
        per_pair = 2 * nmax * nqmax * isz
        nb1 = max(nb, int(mem_mb * 1e6 // max(per_pair, 1)))
        nbt = max(1, int(mem_mb * 1e6 // max(3 * vm.npair * vm.n * isz, 1)))
        return nb, nb1, nbt

    def _pass2_unit(self):
        '''Elements of the work array per occupied pair of a batch of pass 2:
        the products of all pair groups and one gathered block, for the group
        of occupied pairs that takes most.'''
        om, vm = self.om, self.vm
        no, nv = [len(r) for r in self.orng], [len(r) for r in self.vrng]
        nQ = [int(vm.pair_off[g + 1] - vm.pair_off[g]) for g in range(self.npg)]
        nD = {d: sum(nv[sa] * no[si] for sa, si in self.dsub[d]) for d in self.deltas}
        unit = 1
        for gP in range(self.npg):
            sizes = [nD[gP - gQ] * nQ[gQ] for gQ in range(self.npg)
                     if abs(gP - gQ) < self.nc and nQ[gQ]]
            if sizes and om.pair_off[gP + 1] > om.pair_off[gP]:
                unit = max(unit, sum(sizes) + max(sizes))
        return unit

    def _pass2_batch(self, mem_mb):
        '''Occupied pairs per batch of pass 2 (C kernels).  The pair batch of
        the dense layout (pair_batch_size) counts o v C(v,2) elements per
        pair; in blocks a pair takes a fraction of that (_pass2_unit), and
        batches of that size - three pairs with 26 + 120 spinors - make
        thousands of gathers, products and reductions too small for the
        threads.  As many pairs as two thirds of ``mem_mb`` hold (the share
        of the batch buffers in the dense engine), at least that batch.'''
        isz = np.dtype(self.dtype).itemsize
        nb = self._batch_sizes(mem_mb)[0]
        return max(nb, int(mem_mb * 1e6 * 2 / 3 // max(self._pass2_unit() * isz, 1)))

    def set_memory(self, mem_mb):
        '''Batch sizes for a buffer budget of ``mem_mb`` MB.  A change gives
        the work array back: it is allocated again with the new sizes.'''
        if mem_mb == self.mem_mb:
            return
        self.mem_mb = mem_mb
        self.nb, self.nb1, self.nbt = self._batch_sizes(mem_mb)
        self.nb2 = self._pass2_batch(mem_mb)
        self.__dict__.pop('_redt', None)        # (targets cached per batch of pairs)
        release_workspace()

    def mem_min(self):
        '''The smallest useful ``mem_mb``: every pass works with batches of
        one pair or one triple whatever the budget, so a quarter of f.'''
        return 0.25 * self.om.n * self.vm.n ** 3 * np.dtype(self.dtype).itemsize / 1e6

    def _row_unit(self):
        '''Elements of the gathered T3 per occupied triple in the pp ladder
        by rows (both spins of the free index).'''
        vm, L = self.vm, self.L
        ncs = [len(r) for r in self.vrng]
        nQ = [int(vm.pair_off[h + 1] - vm.pair_off[h]) for h in range(self.npg)]
        return max([sum(nQ[g - sc] * ncs[sc] for sc in self.cls if 0 <= g - sc < self.npg)
                    for g in L.stored] + [1])

    def buffer_bytes(self, mem_mb, max_rows=None):
        '''Bytes of the work array with ``mem_mb``: the largest of what the
        passes take from it (``max_rows``: rows per slab of the pp ladder by
        rows; default: a whole pair block).'''
        om, vm, L = self.om, self.vm, self.L
        isz = np.dtype(self.dtype).itemsize
        nb, nb1, nbt = self._batch_sizes(mem_mb)
        nb = self._pass2_batch(mem_mb)          # (pass 2 with the C kernels; else smaller)
        no, nv = [len(r) for r in self.orng], [len(r) for r in self.vrng]
        npg, nc = self.npg, self.nc
        nP = [int(om.pair_off[g + 1] - om.pair_off[g]) for g in range(npg)]
        nQ = [int(vm.pair_off[g + 1] - vm.pair_off[g]) for g in range(npg)]
        nI = [int(om.tri_off[g + 1] - om.tri_off[g]) for g in range(self.ntg)]
        nD = {d: sum(nv[sa] * no[si] for sa, si in self.dsub[d]) for d in self.deltas}
        n = 1
        for gP in range(npg):
            # pass 1: one gathered block T[P, d, m, Q]
            for sd in self.cls:
                for sm in self.cls:
                    if 0 <= gP + sm - sd < npg:
                        n = max(n, min(nb1, nP[gP]) * nv[sd] * no[sm] * nQ[gP + sm - sd])
            # pass 2: the products of all pair groups and one gathered block
            sizes = [nD[gP - gQ] * nQ[gQ] for gQ in range(npg) if abs(gP - gQ) < nc and nQ[gQ]]
            if sizes:
                n = max(n, min(nb, nP[gP]) * (sum(sizes) + max(sizes)))
        for g in L.stored:
            # pp ladder: the products of both spins of the free index and one
            # gathered block (operator in memory), or the gathered blocks and
            # one product (operator by rows: two thirds of the budget)
            sizes = [nQ[g - sc] * nv[sc] for sc in self.cls if 0 <= g - sc < npg and nQ[g - sc]]
            if sizes and nI[g]:
                n = max(n, min(nbt, nI[g]) * (sum(sizes) + max(sizes)))
        per = self._row_unit() + (max(nQ) if max_rows is None else max_rows) * max(nv)
        n = max(n, min(max(nI[g] for g in L.stored) * per,
                       max(int(mem_mb * 1e6 * 2 / 3 // isz), per)))
        return n * isz

    def _workspace(self, n):
        '''n elements of the work array (see workspace).'''
        return workspace(n, self.dtype)

    # ---- pass 1

    def _red_targets(self, p0, p1):
        '''Targets of the pass-2 reduction for the occupied pairs [p0, p1):
        per virtual group g, (Rbase[it], free_r(it), pair_r(it) - p0, rs_r,
        A0, A1) for the computed triples it with pair_r(it) in the batch.'''
        cache = self.__dict__.setdefault('_redt', {})
        if (p0, p1) not in cache:
            om, vm = self.om, self.vm
            out = []
            for g in range(self.ntg):
                A0, A1 = vm.tri_off[g], vm.tri_off[g + 1]
                its = np.where((self.Rbase >= 0) & (om.tri_grp == g))[0]
                if A1 == A0 or len(its) == 0:
                    continue
                tb, tx, tP, ts = [], [], [], []
                for r, sg in ((0, 1.0), (1, -1.0), (2, 1.0)):
                    P = om.pair_c[r, its]
                    sel = (P >= p0) & (P < p1)
                    tb.append(self.Rbase[its[sel]])
                    tx.append(om.free_c[r, its[sel]])
                    tP.append(P[sel] - p0)
                    ts.append(np.full(int(sel.sum()), sg))
                tb = np.concatenate(tb)
                if len(tb) == 0:
                    continue
                c = lambda x, t=np.int64: np.ascontiguousarray(np.concatenate(x), dtype=t)  # noqa: E731
                out.append((np.ascontiguousarray(tb, dtype=np.int64), c(tx), c(tP),
                            c(ts, np.float64), A0, A1))
            cache[(p0, p1)] = out
        return cache[(p0, p1)]

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
        for gP in range(self.npg):
            for pb in _batches(om.pair_off[gP], om.pair_off[gP + 1], self.nb):
                for sd in self.cls:
                    for sm in self.cls:
                        gQ = gP + sm - sd
                        if not 0 <= gQ < self.npg:
                            continue
                        d, m = self.vrng[sd], self.orng[sm]
                        Qlo, Qhi = vm.pair_off[gQ], vm.pair_off[gQ + 1]
                        if len(d) == 0 or len(m) == 0 or Qhi == Qlo:
                            continue
                        T = self.gather(T3, pb, d, m, Qlo, Qhi)  # (P, d, m, Q)
                        Tf = T.reshape(len(pb) * len(d), len(m) * (Qhi - Qlo))
                        sy = gP - sd                  # spin of e/j (X) and n/a (Y)
                        if 0 <= sy < self.nc:
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
    def residual(self, T3, WringT, t2mn, Wv, Wo, t2P, Wpp, O6, out=None, clear=True):
        '''T3 residual on the stored blocks (flat, layout of T3): new, or
        accumulated in ``out`` (zeroed first unless ``clear`` is False).'''
        if self.use_c:
            return self._residual_c(T3, WringT, t2mn, Wv, Wo, t2P, Wpp, O6, out, clear)
        R = self._residual_np(T3, WringT, t2mn, Wv, Wo, t2P, Wpp, O6)
        if out is None:
            return R
        out = _target(out, (self.L.size,), self.dtype, clear)
        out += R
        return out

    def target(self, out=None, clear=True):
        '''The array the residual is accumulated in (see residual).'''
        return _target(out, (self.L.size,), self.dtype, clear)

    def _residual_np(self, T3, WringT, t2mn, Wv, Wo, t2P, Wpp, O6):
        '''Numpy reference: blocked GEMMs, scattered into dense buffers and
        reduced with the dense kernels.'''
        om, vm, dtype, L = self.om, self.vm, self.dtype, self.L
        nocc, nvir = om.n, vm.n
        Rd = np.zeros((om.ntri, vm.ntri), dtype=dtype)
        with timer('pass2'):
            self._pass2(T3, WringT, t2mn, Wv, Wo, t2P, Rd)
        with timer('pp_ladder'):
            if Wpp is not None:
                self._ppl(T3, Wpp, Rd)
        with timer('hh_ladder'):
            for g in (L.stored if O6 is not None else ()):
                I = slice(om.tri_off[g], om.tri_off[g + 1])
                A = slice(vm.tri_off[g], vm.tri_off[g + 1])
                Og = O6.matrix((I.start, I.stop)) if isinstance(O6, t3kern.OccOperator) else O6[I, I]
                Rd[I, A] += Og @ L.block(T3, g)
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
        for delta in self.deltas:
            a, i = self._rows(delta)
            flat = a * nocc + i
            # (no ring term: a zero operator in this reference path)
            Wd[delta] = np.zeros((len(flat), len(flat)), dtype=dtype) if WringT is None \
                else np.ascontiguousarray(WringT[flat[:, None], flat])
        top = 1 if self.L.flip else self.npg - 1   # flip: targets g <= 1 need gP <= 1
        for gP in range(top + 1):
            for pb in _batches(om.pair_off[gP], om.pair_off[gP + 1], self.nb):
                nbp = len(pb)
                Zfull = np.zeros((nbp, nvir, nocc, vm.npair), dtype=dtype)
                for gQ in range(self.npg):
                    delta = gP - gQ
                    if abs(delta) >= self.nc:
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
                        if 0 <= se < self.nc:
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
            for sc in self.cls:
                h = g - sc
                if not 0 <= h < self.npg:
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
                Tg, Rg = L.block(T3, g), L.block(R, g)
                t3kern.update_into_r(Tg, Rg, eo[I], ev[A],
                                     None if self.io is None else scratch.io_desc(Tg, Rg))
            return R
        for g in L.stored:
            I = slice(om.tri_off[g], om.tri_off[g + 1])
            A = slice(vm.tri_off[g], vm.tri_off[g + 1])
            L.block(R, g)[:] = L.block(T3, g) + L.block(R, g) / (
                eo[I][:, None] - ev[A][None, :])
        scratch.written(R)
        return R


class _CArgs:
    '''ctypes helpers for the blocked kernels.'''
    @staticmethod
    def I(x):
        return ctypes.c_int64(int(x))

    @staticmethod
    def p(a, off=0):
        return ctypes.c_void_p(int(a.ctypes.data) + int(off) * a.itemsize)


def _vtables(eng):
    '''Precombined virtual tables of the gathers, per (d, Q): the column of
    the triple {d} + Q in its stored block (int32) and its sign vsgn * vsig
    (int8, 0 if d is in Q).'''
    t = getattr(eng, '_vtab', None)
    if t is None:
        vm, L = eng.vm, eng.L
        sg = vm.sgn_c.astype(np.int64)
        ci = np.where(sg != 0, L.vcol[vm.tri_c], 0)
        assert ci.max(initial=0) < 2 ** 31
        cs = (sg * L.vsig[vm.tri_c]).astype(np.int8)
        t = eng._vtab = (np.ascontiguousarray(ci, dtype=np.int32), np.ascontiguousarray(cs))
    return t


def _cgather(eng, T3, Pidx, didx, midx, Qlo, Qhi, out, off, ldP):
    om, vm, L = eng.om, eng.vm, eng.L
    vci, vcs = _vtables(eng)
    f = getattr(t3pack._lib, 'zc_bgather2_' + t3pack._sfx(T3.dtype))
    I, p = _CArgs.I, _CArgs.p
    f(p(T3), p(L.obase), p(L.osig), p(om.tri_c), p(om.sgn_c), I(om.npair), p(vci),
      p(vcs), I(vm.npair), I(len(Pidx)), p(Pidx), I(len(didx)), p(didx), I(len(midx)),
      p(midx), I(Qlo), I(Qhi - Qlo), I(ldP), p(out, off))


def _cgather_old(eng, T3, Pidx, didx, midx, Qlo, Qhi, out, off, ldP):
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
        gpp = np.ascontiguousarray(gpp, dtype=dtype)
        vr = [(r[0], r[-1] + 1) if len(r) else (0, 0) for r in self.vrng]
        orr = [(r[0], r[-1] + 1) if len(r) else (0, 0) for r in self.orng]
        gy_cache = {}           # Y-GEMM operands depend on (sm, gQ, sy) only
        flip = self.L.flip      # closed shell: half of the spin combinations

        def gy(sm, gQ, sy):
            key = (sm, gQ, sy)
            if key not in gy_cache:
                with timer('pass1.gy'):
                    (m0, m1), (j0, j1), (e0, e1) = orr[sm], orr[sy], vr[sy]
                    Qlo, Qhi = vm.pair_off[gQ], vm.pair_off[gQ + 1]
                    nj = j1 - j0
                    G = np.empty((m1 - m0, Qhi - Qlo, nj + e1 - e0), dtype=dtype)
                    t3pack.axpby(G[:, :, :nj], G3[m0:m1, Qlo:Qhi, j0:j1], 1.0, 0.0)
                    t3pack.axpby(G[:, :, nj:], G3[m0:m1, Qlo:Qhi, nocc + e0:nocc + e1], 1.0, 0.0)
                    gy_cache[key] = G.reshape((m1 - m0) * (Qhi - Qlo), -1)
            return gy_cache[key]

        for gP in range(self.npg):
            for pb in _batches(om.pair_off[gP], om.pair_off[gP + 1], self.nb1):
                p0, p1 = pb[0], pb[-1] + 1
                for sd in self.cls:
                    for sm in self.cls:
                        gQ = gP + sm - sd
                        if not 0 <= gQ < self.npg:
                            continue
                        if flip and (gP, sd, sm) > (2 - gP, 1 - sd, 1 - sm):
                            continue      # spin-flipped image of a computed combination
                        d, m = self.vrng[sd], self.orng[sm]
                        Qlo, Qhi = vm.pair_off[gQ], vm.pair_off[gQ + 1]
                        nd, nm, nQ = len(d), len(m), Qhi - Qlo
                        if nd == 0 or nm == 0 or nQ == 0:
                            continue
                        t_ = timer('pass1.gather'); t_.__enter__()
                        T = self._workspace(len(pb) * nd * nm * nQ).reshape(len(pb), nd, nm, nQ)
                        _cgather(self, T3, pb, d, m, Qlo, Qhi, T, 0, nd * nm * nQ)
                        Tf = T.reshape(len(pb) * nd, nm * nQ)
                        t_.__exit__(None, None, None)
                        sy = gP - sd
                        # (a class without spinors - the labels need not be contiguous - has no rows)
                        if 0 <= sy < self.nc and (vr[sy][1] > vr[sy][0] or orr[sy][1] > orr[sy][0]):
                            e0, e1 = vr[sy]
                            j0, j1 = orr[sy]
                            G = np.concatenate((G_Pd[e0:e1, p0:p1, vr[sd][0]:vr[sd][1]],
                                                G_Pd[nvir + j0:nvir + j1, p0:p1,
                                                     vr[sd][0]:vr[sd][1]]), axis=0)
                            t_ = timer('pass1.gemm'); t_.__enter__()
                            # (a few rows times a wide matrix: pieces of columns,
                            # one sequential GEMM each, shared by the threads)
                            Gc = np.ascontiguousarray(G, dtype=dtype).reshape(G.shape[0], -1)
                            X = np.empty((Gc.shape[0], Tf.shape[1]), dtype=dtype)
                            gemm_tasks([(Gc.shape[0], Tf.shape[1], Gc.shape[1], 0, Gc.shape[1],
                                         0, Tf.shape[1], 0, Tf.shape[1])], Gc, Tf, X, cols=256)
                            t_.__exit__(None, None, None)
                            t_ = timer('pass1.acc'); t_.__enter__()
                            X = X.reshape(-1, nm, nQ)
                            m0, m1 = orr[sm]
                            t3pack.axpby(Xacc[e0:e1, m0:m1, Qlo:Qhi], X[:e1 - e0])
                            t3pack.axpby(Xacc[nvir + j0:nvir + j1, m0:m1, Qlo:Qhi], X[e1 - e0:])
                            t_.__exit__(None, None, None)
                            t_ = timer('pass1.gemm'); t_.__enter__()
                            # (a tall matrix times a few columns: pieces of rows)
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
                            t_.__exit__(None, None, None)
                            t_ = timer('pass1.acc'); t_.__enter__()
                            d0, d1 = vr[sd]
                            t3pack.axpby(Yall[p0:p1, d0:d1, j0:j1], Y[:, :, :j1 - j0])
                            t3pack.axpby(Yall[p0:p1, d0:d1, nocc + e0:nocc + e1], Y[:, :, j1 - j0:])
                            t_.__exit__(None, None, None)
                        if sd == sm:
                            t_ = timer('pass1.r1rA'); t_.__enter__()
                            d0, d1 = vr[sd]
                            m0, m1 = orr[sm]
                            # one matrix-vector product per pair for each of the
                            # two, all pairs in one parallel region:
                            #   r1[(d m)] += T[P, (d m), Q] gpp[P, Q]
                            #   rA[P, Q]  += f[(d m)] T[P, (d m), Q]
                            nbp, ndm = len(pb), nd * nm
                            ks = np.arange(nbp, dtype=np.int64)
                            acc = np.empty((nbp, ndm), dtype=dtype)
                            tasks = np.empty((nbp, 9), dtype=np.int64)
                            tasks[:] = (ndm, 1, nQ, 0, nQ, 0, 1, 0, 1)
                            tasks[:, 3] = ks * (ndm * nQ)
                            tasks[:, 5] = (p0 + ks) * npv + Qlo
                            tasks[:, 7] = ks * ndm
                            gemm_tasks(tasks, T, gpp, acc)
                            r1[d0:d1, m0:m1] += acc.sum(axis=0).reshape(nd, nm)
                            fdm = np.ascontiguousarray(fov[m0:m1, d0:d1].T, dtype=dtype).ravel()
                            tasks[:] = (1, nQ, ndm, 0, ndm, 0, nQ, 0, nQ)
                            tasks[:, 5] = ks * (ndm * nQ)
                            tasks[:, 7] = (p0 + ks) * npv + Qlo
                            gemm_tasks(tasks, fdm, T, rA, beta=1.0, cols=max(nQ, 1))
                            t_.__exit__(None, None, None)
        if flip:
            with timer('pass1.flip'):
                self._flip_outputs(Xacc, Yall, r1, rA)
        return Xacc, Yall, r1, rA

    def _flip_outputs(self, Xacc, Yall, r1, rA):
        '''Complete the pass-1 outputs computed from half of the spin
        combinations: X <- X + flip(X) (alpha <-> beta, pair re-sorting signs).'''
        om, vm = self.om, self.vm
        nocc, nvir = om.n, vm.n
        of, Pf, sP = t3pack.flip_pairs(om)
        vf, Qf, sQ = t3pack.flip_pairs(vm)
        noa, nva = nocc // 2, nvir // 2
        one = np.ones(nocc + nvir)
        # Xacc[(e | j), m, Q]
        t3pack.flip_sym3(Xacc, (np.concatenate((vf, nvir + of)), of, Qf),
                         (one, one[:nocc], sQ),
                         np.concatenate((np.arange(nva), nvir + np.arange(noa))))
        # Yall[P, d, (j | e)] as the view (d, P, (j | e))
        t3pack.flip_sym3(Yall.transpose(1, 0, 2), (vf, Pf, np.concatenate((of, nocc + vf))),
                         (one[:nvir], sP, one), np.arange(nva))
        r1 += r1[np.ix_(vf, of)]
        rA += sP[:, None] * sQ[None, :] * rA[np.ix_(Pf, Qf)]

    def _residual_c(self, T3, WringT, t2mn, Wv, Wo, t2P, Wpp, O6, out=None, clear=True):
        om, vm, dtype, L = self.om, self.vm, self.dtype, self.L
        R = _target(out, (L.size,), dtype, clear)
        with timer('pass2'):
            self._pass2_c(T3, WringT, t2mn, Wv, Wo, t2P, R)
        with timer('pp_ladder'):
            if Wpp is not None:         # (else in R already, see pp_ladder_rows)
                self._ppl_c(T3, Wpp, R)
        scratch.unmap(T3)               # (the blocked gathers leave the unmapping to here)
        with timer('hh_ladder'):
            for g in (L.stored if O6 is not None else ()):
                if L.nI[g] == 0 or L.nA[g] == 0:
                    continue
                I = slice(om.tri_off[g], om.tri_off[g + 1])
                if isinstance(O6, t3kern.OccOperator):
                    Tg, Rg = L.block(T3, g), L.block(R, g)
                    O6.apply(Tg, Rg, (int(I.start), int(I.stop)),
                             occ_cols(L.nI[g], np.dtype(dtype).itemsize, self.mem_mb),
                             None if self.io is None else scratch.io_desc(Tg, Rg))
                    continue
                Og = np.ascontiguousarray(O6[I, I], dtype=dtype)
                if t3kern.USE_C:
                    Tg, Rg = L.block(T3, g), L.block(R, g)
                    t3kern.occ_operator(Og, Tg, Rg,
                                        occ_cols(L.nI[g], np.dtype(dtype).itemsize, self.mem_mb),
                                        None if self.io is None else scratch.io_desc(Tg, Rg))
                else:
                    gemm(Og, L.block(T3, g), L.block(R, g), beta=1.0)
        return R

    def _pass2_c(self, T3, WringT, t2mn, Wv, Wo, t2P, R):
        om, vm, dtype, L = self.om, self.vm, self.dtype, self.L
        nocc, nvir = om.n, vm.n
        vr = [(r[0], r[-1] + 1) if len(r) else (0, 0) for r in self.vrng]
        orr = [(r[0], r[-1] + 1) if len(r) else (0, 0) for r in self.orng]
        npg = self.npg
        Wd, nD = {}, {}
        for delta in self.deltas:
            a, i = self._rows(delta)
            flat = a * nocc + i
            if WringT is not None:
                Wd[delta] = np.ascontiguousarray(WringT[flat[:, None], flat], dtype=dtype)
            nD[delta] = len(a)
        # drive right-hand sides B[(e|m), (i, Q)] per (gQ, s_i), e and m of spin gQ - s_i
        Bd = {}
        for gQ in range(npg):
            Qlo, Qhi = vm.pair_off[gQ], vm.pair_off[gQ + 1]
            for si in self.cls:
                se = gQ - si
                if not 0 <= se < self.nc or Qhi == Qlo:
                    continue
                (e0, e1), (i0, i1), (m0, m1) = vr[se], orr[si], orr[se]
                if e1 == e0 and m1 == m0:       # (a class without spinors)
                    continue
                B1 = Wv[Qlo:Qhi, e0:e1, i0:i1].transpose(1, 2, 0)          # [e, i, Q]
                B2 = t2P[m0:m1, i0:i1, Qlo:Qhi]                            # [m, i, Q]
                Bd[gQ, si] = np.ascontiguousarray(np.concatenate((B1, B2), axis=0),
                                                  dtype=dtype).reshape(
                    (e1 - e0) + (m1 - m0), -1)
        nQg = np.array([vm.pair_off[g + 1] - vm.pair_off[g] for g in range(npg)], dtype=np.int64)
        Qoff = np.ascontiguousarray(vm.pair_off[:npg], dtype=np.int64)
        lib = t3pack._lib
        fred = getattr(lib, 'zc_breduce_pass2_' + t3pack._sfx(dtype))
        fred2 = getattr(lib, 'zc_breduce2_' + t3pack._sfx(dtype))
        I_, p_ = _CArgs.I, _CArgs.p
        top = 1 if L.flip else npg - 1
        for gP in range(top + 1):
            for pb in _batches(om.pair_off[gP], om.pair_off[gP + 1], self.nb2):
                p0, p1, nbp = pb[0], pb[-1] + 1, len(pb)
                gQs = [gQ for gQ in range(npg) if abs(gP - gQ) < self.nc and nQg[gQ] > 0]
                if not gQs:
                    continue
                zoff = -np.ones((nbp, npg), dtype=np.int64)
                off = 0
                for k in range(nbp):
                    for gQ in gQs:
                        zoff[k, gQ] = off
                        off += nD[gP - gQ] * nQg[gQ]
                tmax = max(nD[gP - gQ] * nQg[gQ] for gQ in gQs) * nbp
                ring = WringT is not None   # (None: the drive terms alone - CC3; T3 is not read)
                ws = self._workspace(off + (tmax if ring else 0))
                Z = ws[:off]
                if not ring:
                    t3kern.zero(Z)
                for gQ in gQs:
                    delta = gP - gQ
                    Qlo, Qhi = vm.pair_off[gQ], vm.pair_off[gQ + 1]
                    nQ, nr = Qhi - Qlo, nD[delta]
                    if ring:
                        T = ws[off:off + nbp * nr * nQ]
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
                    # one product per pair of the batch and block of rows: all
                    # of them in one parallel region (blas.gemm_tasks)
                    if sub and ring:
                        tasks = np.empty((len(sub), nbp, 9), dtype=np.int64)
                        for j, (r0, n) in enumerate(sub):
                            tasks[j] = (n, nQ, nr, r0 * nr, nr, 0, nQ, 0, nQ)
                            tasks[j, :, 5] = np.arange(nbp) * (nr * nQ)
                            tasks[j, :, 7] = zoff[:, gQ] + r0 * nQ
                        gemm_tasks(tasks, Wd[delta], T, Z)
                    t_.__exit__(None, None, None)
                    t_ = timer('pass2.drive'); t_.__enter__()
                    # drive terms on the (s_c, s_i) sub-blocks of D_delta
                    r0 = 0
                    for sc, si in self.dsub[delta]:
                        (c0, c1), (i0, i1) = vr[sc], orr[si]
                        nc, ni = c1 - c0, i1 - i0
                        se = gP - sc
                        if nc and ni and 0 <= se < self.nc and (gQ, si) in Bd \
                                and gP + si in L.stored:
                            (e0, e1), (m0, m1) = vr[se], orr[se]
                            # A[P, c, (e | m)] of all pairs of the batch, and their
                            # products with the right-hand side in one parallel region
                            A = np.ascontiguousarray(np.concatenate(
                                (t2mn[p0:p1, e0:e1, c0:c1].transpose(0, 2, 1),
                                 Wo[m0:m1, c0:c1, p0:p1].transpose(2, 1, 0)), axis=2), dtype=dtype)
                            kk = A.shape[2]
                            tasks = np.empty((nbp, 9), dtype=np.int64)
                            tasks[:] = (nc, ni * nQ, kk, 0, kk, 0, ni * nQ, 0, ni * nQ)
                            tasks[:, 3] = np.arange(nbp) * (nc * kk)
                            tasks[:, 7] = zoff[:, gQ] + r0 * nQ
                            gemm_tasks(tasks, A, Bd[gQ, si], Z, alpha=-1.0, beta=1.0)
                        r0 += nc * ni
                    t_.__exit__(None, None, None)
                t_ = timer('pass2.reduce'); t_.__enter__()
                if _RED2:
                    for tb, tx, tP, tsg, A0, A1 in self._red_targets(p0, p1):
                        fred2(p_(R), p_(self.Rcol), I_(len(tb)), p_(tb), p_(tx), p_(tP),
                              p_(tsg), I_(A0), I_(A1), I_(vm.ntri), p_(vm.free_c),
                              p_(vm.pair_c), p_(vm.pair_grp), p_(zoff), I_(npg), p_(nQg), p_(Qoff),
                              p_(self.rowpos), I_(nocc), p_(Z))
                else:
                    fred(p_(R), p_(self.Rbase), p_(self.Rcol), I_(om.ntri), p_(om.tri_grp),
                         p_(om.free_c), p_(om.pair_c), I_(vm.ntri), p_(vm.tri_off),
                         p_(vm.free_c), p_(vm.pair_c), p_(vm.pair_grp), I_(p0), I_(p1),
                         p_(zoff), I_(npg), p_(nQg), p_(Qoff), p_(self.rowpos), I_(nocc), p_(Z))
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
        for h in range(self.npg):
            Qlo, Qhi = vm.pair_off[h], vm.pair_off[h + 1]
            Wh[h] = np.ascontiguousarray(Wpp[h], dtype=dtype)
        vspin = np.ascontiguousarray(vm.spins, dtype=np.int64)
        coff = np.array([vr[s][0] for s in self.cls], dtype=np.int64)
        ncs = np.array([vr[s][1] - vr[s][0] for s in self.cls], dtype=np.int64)
        dummy = np.zeros(1, dtype=dtype)
        nbt = self.nbt
        for g in L.stored:
            A0, A1 = vm.tri_off[g], vm.tri_off[g + 1]
            if A1 == A0:
                continue
            for rows in _batches(om.tri_off[g], om.tri_off[g + 1], nbt):
                nb = len(rows)
                X = [dummy] * self.nc
                hoff = np.zeros(self.nc, dtype=np.int64)
                # workspace [X_0 | X_1 | T]: both X stay alive for the reduction
                size = {}
                for sc in self.cls:
                    h = g - sc
                    if 0 <= h < self.npg and ncs[sc] and vm.pair_off[h + 1] > vm.pair_off[h]:
                        size[sc] = (vm.pair_off[h + 1] - vm.pair_off[h]) * nb * int(ncs[sc])
                nx = sum(size.values())
                ws = self._workspace(nx + max(size.values(), default=0))
                xo = 0
                for sc in sorted(size):
                    h = g - sc
                    Qlo, Qhi = vm.pair_off[h], vm.pair_off[h + 1]
                    T = ws[nx:nx + size[sc]].reshape(Qhi - Qlo, nb, ncs[sc])
                    fg(p_(T3), p_(L.obase), p_(L.vcol), p_(L.osig), p_(L.vsig),
                       p_(vm.tri_c), p_(vm.sgn_c), I_(vm.npair), I_(nb), p_(rows),
                       I_(Qlo), I_(Qhi - Qlo), I_(coff[sc]), I_(ncs[sc]), p_(T))
                    X[sc] = ws[xo:xo + size[sc]].reshape(Qhi - Qlo, -1)
                    gemm(Wh[h], T.reshape(Qhi - Qlo, -1), X[sc])
                    xo += size[sc]
                    hoff[sc] = Qlo
                if not size:
                    continue
                xs = (ctypes.c_void_p * self.nc)(*[x.ctypes.data for x in X])
                fr(p_(R), p_(self.Rbase), p_(self.Rcol), I_(nb), p_(rows), I_(vm.ntri),
                   I_(A0), I_(A1), p_(vm.free_c), p_(vm.pair_c), p_(vspin), p_(hoff),
                   p_(ncs), p_(coff), xs)

    def pp_ladder_rows(self, T3, slabs, R, order='slab', max_rows=None):
        '''pp ladder with the operator given in slabs of rows of its pair
        blocks: ``slabs(need)`` returns an iterator of (hs, q0, q1, W), the
        rows q0:q1 (block-local) of the blocks ``hs`` (alpha-alpha and
        beta-beta share their rows for RHF), W of shape (q1 - q0, n_h); it
        may leave out blocks that are not in the set ``need`` (None: all are
        needed).  order 'slab': for each slab all occupied triples; 'batch':
        for each batch of triples the slabs of the blocks its spin group
        needs (slabs() is called once per batch).  ``max_rows``: the largest
        q1 - q0.  R as returned by target() (C kernels only).'''
        om, vm, dtype, L = self.om, self.vm, self.dtype, self.L
        isz = np.dtype(dtype).itemsize
        vr = [(r[0], r[-1] + 1) if len(r) else (0, 0) for r in self.vrng]
        lib = t3pack._lib
        fg = getattr(lib, 'zc_bgather_vir_' + t3pack._sfx(dtype))
        fr = getattr(lib, 'zc_breduce_ppl_rows_' + t3pack._sfx(dtype))
        I_, p_ = _CArgs.I, _CArgs.p
        vspin = np.ascontiguousarray(vm.spins, dtype=np.int64)
        coff = [vr[s][0] for s in self.cls]
        ncs = [vr[s][1] - vr[s][0] for s in self.cls]
        nQ = [int(vm.pair_off[h + 1] - vm.pair_off[h]) for h in range(self.npg)]
        max_rows = max(nQ) if max_rows is None else max_rows

        def combos(g):                  # (spin of the free index, pair block) of group g
            return [(sc, g - sc) for sc in self.cls
                    if 0 <= g - sc < self.npg and ncs[sc] and nQ[g - sc]]

        # per triple of a batch: the gathered T3 of both spins of the free
        # index and one product; two thirds of the buffer budget
        per = max([sum(nQ[h] * ncs[sc] for sc, h in combos(g)) + max_rows * max(ncs)
                   for g in L.stored] + [1])
        nbt = max(1, int(self.mem_mb * 1e6 * 2 / 3 // (per * isz)))
        nbt = min(nbt, max(int(max(L.nI[g] for g in L.stored)), 1))
        ws = self._workspace(per * nbt)

        def gather(rows, sc, h, off):
            n = nQ[h] * len(rows) * ncs[sc]
            T = ws[off:off + n].reshape(nQ[h], len(rows), ncs[sc])
            with timer('ppl.gather'):
                fg(p_(T3), p_(L.obase), p_(L.vcol), p_(L.osig), p_(L.vsig), p_(vm.tri_c),
                   p_(vm.sgn_c), I_(vm.npair), I_(len(rows)), p_(rows), I_(vm.pair_off[h]),
                   I_(nQ[h]), I_(coff[sc]), I_(ncs[sc]), p_(T))
            return T, off + n

        def apply(T, off, g, rows, sc, h, q0, q1, W):
            X = ws[off:off + (q1 - q0) * len(rows) * ncs[sc]].reshape(q1 - q0, -1)
            with timer('ppl.gemm'):
                gemm(W, T.reshape(nQ[h], -1), X)
            with timer('ppl.reduce'):
                fr(p_(R), p_(self.Rbase), p_(self.Rcol), I_(len(rows)), p_(rows), I_(vm.ntri),
                   I_(vm.tri_off[g]), I_(vm.tri_off[g + 1]), p_(vm.free_c), p_(vm.pair_c),
                   p_(vspin), I_(sc), I_(vm.pair_off[h]), I_(q0), I_(q1), I_(ncs[sc]),
                   I_(coff[sc]), p_(X))

        def batches(g):
            return _batches(om.tri_off[g], om.tri_off[g + 1], nbt) \
                if vm.tri_off[g + 1] > vm.tri_off[g] else []

        if order == 'slab':
            for hs, q0, q1, W in slabs(None):
                for h in hs:
                    for g in L.stored:
                        for sc, hh in combos(g):
                            if hh != h:
                                continue
                            for rows in batches(g):
                                T, off = gather(rows, sc, h, 0)
                                apply(T, off, g, rows, sc, h, q0, q1, W)
        else:
            for g in L.stored:
                cs = combos(g)
                for rows in batches(g):
                    Ts, off = {}, 0
                    for sc, h in cs:
                        Ts[sc], off = gather(rows, sc, h, off)
                    for hs, q0, q1, W in slabs({h for _, h in cs}):
                        for sc, h in cs:
                            if h in hs:
                                apply(Ts[sc], off, g, rows, sc, h, q0, q1, W)
        scratch.unmap(T3)

    return _pass1_c, _flip_outputs, _residual_c, _pass2_c, _ppl_c, pp_ladder_rows


(SpinBlockT3._pass1_c, SpinBlockT3._flip_outputs, SpinBlockT3._residual_c,
 SpinBlockT3._pass2_c, SpinBlockT3._ppl_c, SpinBlockT3.pp_ladder_rows) = _spin_block_c_methods()


def make_engine(om, vm, dtype, mem_mb, flip=False):
    if om.spins is None:
        return DenseT3(om, vm, dtype, mem_mb)
    return SpinBlockT3(om, vm, dtype, mem_mb, flip=flip)
