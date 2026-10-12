'''CC3 without stored triples: the scheduler of csrc/cc3_fly.c (docs/ccn.md,
sections 9 and 10).

For semicanonical spinors (the front end and solver.kernel make them; F_oo, F_vv
diagonal) the triples equation of CC3 is solved in closed form,

    t3[ijk, abc] = drive[t2; <ab||ei>~, <ma||ij>~] / (e_i + e_j + e_k - e_a - e_b - e_c),

so the triples of a batch of occupied triples are made, contracted into the
singles and doubles equations (the four sums of pass 1 of the CCSDT step,
docs/factorization.md 2.4) and discarded: o^3 v^4 work per iteration, and
no array of the size of T3.  The same pass with the frequency omega in the
denominator and two drive terms gives the folded doubles-triples coupling
of the CC3 Jacobian (eom_opt.sigma for a Ground without T3).

A batch is the set of occupied triples (i, j, k) with i0 <= i < i1 <= j < k
and the i of one symmetry class; one call of the kernel zc_cc3_batch does
it completely (products, amplitudes, contractions) in the work array given.
This module makes the operand layouts once per iteration, loops over the
batches and folds the four accumulated sums into the terms of the
equations.  With symmetry labels in use (the maps of
zccsdt.cc.t3pack.maps_for with classes) only the blocks that the labels
allow are computed: the operands are dense, the triples are not.

Timers (zccsdt.timing): 'cc3fly.operands', 'cc3fly.batches' with the parts
that the kernel reports - .drive, .amps, .gather, .contract.
'''

import ctypes

import numpy as np

from zccsdt import timing
from zccsdt.cc import t3kern
from zccsdt.cc.ccsdt_opt import transposed
from zccsdt.cc.t3pack import _sfx
from zccsdt.timing import timer


def available():
    return t3kern.USE_C and hasattr(t3kern._lib, 'zc_cc3_batch_d')


class Fly:
    '''The batches of occupied triples for the maps (om, vm) and the
    diagonal Fock elements ``eo``, ``ev``; ``mem_mb``: the work array.'''

    def __init__(self, om, vm, dtype, eo, ev, mem_mb=2000.0):
        if not available():
            raise RuntimeError('CC3 without stored triples needs the kernel library with csrc/cc3_fly.c '
                               '(cmake --build build)')
        self.om, self.vm, self.dtype = om, vm, np.dtype(dtype)
        no, nv = om.n, vm.n
        i8 = lambda x: np.ascontiguousarray(x, dtype=np.int64)          # noqa: E731
        if om.spins is None:
            self.ncls = 1
            self.orng, self.vrng = i8([0, no]), i8([0, nv])
            self.qoff, self.aoff = i8([0, vm.npair]), i8([0, vm.ntri])
        else:
            self.ncls = nc = om.ncls
            self.orng = i8(np.searchsorted(om.spins, np.arange(nc + 1)))
            self.vrng = i8(np.searchsorted(vm.spins, np.arange(nc + 1)))
            self.qoff, self.aoff = i8(vm.pair_off), i8(vm.tri_off)
        self.ocls = np.zeros(no, dtype=np.int64) if om.spins is None else np.asarray(om.spins)
        self.eo = np.ascontiguousarray(eo, dtype=np.float64)
        ev = np.asarray(ev, dtype=np.float64)
        self.ev3 = np.ascontiguousarray(ev[vm.triples].sum(axis=1)) if vm.ntri else np.zeros(0)
        self.vsgn = np.ascontiguousarray(vm.sgn_c, dtype=np.int8)
        # per group g of the occupied triples: elements of work per i
        nc = self.ncls
        self.per_i = {}
        for g in range(3 * nc - 2):
            H = sum(int(self.vrng[s + 1] - self.vrng[s]) * int(self.qoff[g - s + 1] - self.qoff[g - s])
                    for s in range(nc) if 0 <= g - s <= 2 * nc - 2)
            self.per_i[g] = 3 * H + int(self.aoff[g + 1] - self.aoff[g])
        self.set_memory(mem_mb)

    def set_memory(self, mem_mb):
        isz = self.dtype.itemsize
        self.mem_mb = mem_mb
        worst = max(self.per_i.values(), default=0)
        self.nwork = int(max(worst, min(mem_mb * 1e6 / isz, worst * max(self.om.n - 2, 1))))

    def work_bytes(self):
        return self.nwork * self.dtype.itemsize

    def mem_min(self):
        '''MB of the smallest work array (one occupied triple per batch).'''
        return max(self.per_i.values(), default=0) * self.dtype.itemsize / 1e6

    def batches(self):
        '''[(i0, i1, j, k)]: i of one class, as many as the work array holds.'''
        no = self.om.n
        out = []
        for k in range(no):
            for j in range(k):
                for s in range(self.ncls):
                    lo, hi = int(self.orng[s]), min(int(self.orng[s + 1]), j)
                    if hi <= lo:
                        continue
                    g = s + int(self.ocls[j]) + int(self.ocls[k])
                    per = self.per_i[g]
                    if per == 0:
                        continue
                    step = max(1, self.nwork // per)
                    out += [(i0, min(i0 + step, hi), j, k) for i0 in range(lo, hi, step)]
        return out

    # ---- operand layouts

    def drive_operands(self, a2, a2P, Wv, W4):
        '''(At, Ao, Bv, Bo) of one drive term: a2[i,j,a,b] the doubles
        amplitudes, a2P their [i, j, (a<b)], Wv[Q, e, i] = <Q||ei>~ and
        W4[m, a, hi, lo] = <ma||lo hi>~ (the layout of the dressed ovoo
        block of the CCSDT step).'''
        dt = self.dtype
        return (transposed(a2, (1, 0, 3, 2), dt),                       # At[hi, lo, c, e] = a2[lo, hi, e, c]
                transposed(W4, (2, 3, 1, 0), dt),                       # Ao[hi, lo, c, m]
                transposed(Wv, (1, 2, 0), dt),                          # Bv[e, i, Q]
                np.ascontiguousarray(a2P, dtype=dt))                    # Bo[m, i, Q]

    def contraction_operands(self, G_mQ, oovo, goovv_P, fov):
        '''(Gv, Gsq, gsq, fov) from the operands of pass 1 of the CCSDT
        step: G_mQ[(m, Q), o + y] = <ym||Q>~, the dressed <mn||ei>, the
        packed <mn||Q> and f~[m, e].'''
        dt = self.dtype
        no, nv, nQ = self.om.n, self.vm.n, self.vm.npair
        Gv = np.ascontiguousarray(np.asarray(G_mQ, dtype=dt).reshape(no, nQ, -1)[:, :, -nv:])
        return (Gv, transposed(oovo, (1, 0, 3, 2), dt),                 # Gsq[hi, lo, x, d] = <lo hi||d x>
                transposed(goovv_P, (1, 0, 2), dt), np.ascontiguousarray(fov, dtype=dt))

    # ---- the pass

    def run(self, drives, contract, omega=0.0):
        '''All batches: the triples from the drive terms ``drives`` (one or
        two tuples of drive_operands; their sum) over the denominator
        e_ijk - e_abc + omega, contracted with ``contract``
        (contraction_operands).  Returns (r1[i, a], r2[(i<j), (a<b)]): the
        T3 terms of the singles and doubles equations.'''
        om, vm, dt = self.om, self.vm, self.dtype
        no, nv, nQ, nA = om.n, vm.n, vm.npair, vm.ntri
        Ysq = np.zeros((no, no, nv, nv), dtype=dt)
        Xo = np.zeros((no, no, nQ), dtype=dt)
        r1 = np.zeros((no, nv), dtype=dt)
        rAsq = np.zeros((no, no, nQ), dtype=dt)
        if nA and no >= 3:
            work = np.empty(self.nwork, dtype=dt)
            stats = np.zeros(4)
            f = getattr(t3kern._lib, 'zc_cc3_batch_' + _sfx(dt))
            I, D = ctypes.c_int64, ctypes.c_double
            p = lambda a: ctypes.c_void_p(a.ctypes.data)                # noqa: E731
            d1 = drives[0]
            d2 = drives[1] if len(drives) > 1 else drives[0]
            for x in tuple(d1) + tuple(d2) + tuple(contract):
                assert x.dtype == dt and x.flags.c_contiguous
            fixed = [I(no), I(nv), I(nQ), I(nA), I(self.ncls), p(self.orng), p(self.vrng), p(self.qoff), p(self.aoff),
                     p(vm.tri_c), p(self.vsgn), p(vm.free_c), p(vm.pair_c), p(self.eo), p(self.ev3), D(float(omega))]
            tail = [I(len(drives))] + [p(a) for a in tuple(d1) + tuple(d2) + tuple(contract)] \
                + [p(Ysq), p(Xo), p(r1), p(rAsq), p(work), p(stats)]
            with timer('cc3fly.batches'):
                for i0, i1, j, k in self.batches():
                    f(*fixed, I(i0), I(i1), I(j), I(k), *tail)
                for name, s in zip(('drive', 'amps', 'gather', 'contract'), stats):
                    timing.add(name, s)
        # ---- folded into the terms of the equations (as ccsdt_opt.update_amps does with pass 1)
        mm, nn = om.pairs[:, 0], om.pairs[:, 1]
        md, ne = vm.pairs[:, 0], vm.pairs[:, 1]
        Yv = Ysq[nn, mm]                                        # [P, d, y]
        rB = -(Yv[:, ne, md] - Yv[:, md, ne])                   # B0[P, a, b] = -2 Yv[P, b, a]; 1/2 (B0[md,ne] - B0[ne,md])
        rC = Xo[nn, mm] - Xo[mm, nn]                            # C0[i, j, Q] = -2 Xo[j, i, Q]; -1/2 (C0[mm,nn] - C0[nn,mm])
        return r1, rAsq[nn, mm] + rB + rC
