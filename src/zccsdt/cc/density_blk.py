'''The passes of the densities of CCSDT over T3 and Lambda_3 in the blocks of
a symmetry label: the engine of ``zccsdt.cc.density_opt`` for the layout of
``t3engine.SpinBlockT3`` / ``T3Layout`` (kernel(..., blocks=True)), next to
``lambda_blk.BlockL3`` of the Lambda equations, whose gathers it uses.

t3 and l3 are stored as the amplitude equations store T3: block g, the
triples whose classes add up to g, as a matrix (nI[g], nA[g]) in a flat
vector.  The operands and the results are the dense arrays of density_opt,
with pairs and triples in the enumeration of the grouped maps
(t3pack.TripleMap with classes); every product is split into the blocks
that the labels leave:

  pass 1   l3 alone, half-expanded as [P, D, Q] for occupied pairs of group
           gP and virtual pairs of group gQ, D the (d, m) with class(d) -
           class(m) = gP - gQ: Z and C3, one GEMM each per class of d and m
  pass 2   t3 and l3 gathered next to each other: the two stacked GEMMs of
           t3 per class of d and m; the ring density R3, one GEMM per
           occupied pair into its block of the difference of classes; the
           products with l1 and l2 where gP = gQ; H = l3 t2 for the occupied
           pairs of the group gQ and its product with t3
  gamma    Gamma[(e<f), (g<h)] = 3 sum l3 t3, block diagonal in the group of
           the pairs: per group of occupied triples and class of the free
           virtual index, one GEMM on the arrays with that index apart
  gram     per group of triples (BlockL3.gram)

In blocks the l3 t3 part of the block abcd of g2 always goes through Gamma
(density_opt: route 'gamma'): its diagonal pair blocks are as large as
those of the pp ladder operator of the amplitude equations, and their
product with the integrals of three virtual indices (gamma_pairs) runs
block by block.  The packed dressed <ab||de> is its diagonal pair blocks,
in slabs of rows of a block (Slabs).

One class is no symmetry: the blocks are then the whole arrays.  The
buffers are views of the work array of the T3 engine (t3engine.workspace),
sized by the plan of density_opt (pass_elems, gamma_elems) and given back
after the passes.

C kernels only (the gathers of csrc/t3pack.c); without them density_opt
works on the packed block.
'''

import numpy as np

from zccsdt.blas import gemm
from zccsdt.cc import ccsdt_opt, lambda_blk, t3engine, t3pack
from zccsdt.cc.t3engine import _CArgs, _batches
from zccsdt.timing import timer


def engine(eris, dtype):
    '''The BlockDensity on the blocks of the symmetry labels of these
    integrals, or None: no labels in use, integrals in the U/R form, or no C
    kernels (lambda_opt.block_engine).  The buffer budget of the T3 engine
    is left as it is.'''
    from zccsdt.cc import lambda_opt
    eng = lambda_opt.block_engine(eris, dtype)
    return None if eng is None else BlockDensity(eng)


class BlockDensity:
    '''The steps of density_opt.kernel that touch t3 and l3, in the blocks
    of ``l3eng`` (a lambda_blk.BlockL3).'''

    def __init__(self, l3eng):
        self.l = b = l3eng
        self.t3 = e = b.t3
        self.om, self.vm, self.dtype, self.L = b.om, b.vm, b.dtype, b.L
        self.size = b.size
        om, vm = self.om, self.vm
        self.nc, self.npg = b.nc, b.npg
        self.vr, self.orr, self.nQg, self.nD, self.flat = b.vr, b.orr, b.nQg, b.nD, b.flat
        self.nPg = [int(om.pair_off[g + 1] - om.pair_off[g]) for g in range(self.npg)]
        self.nvc = [r[1] - r[0] for r in self.vr]
        self.noc = [r[1] - r[0] for r in self.orr]
        self.deltas = e.deltas

    def view(self, x):
        return self.l.view(x)

    # ---- sizes, for the memory plan

    def pass_elems(self, nb):
        '''Elements of the work array of a pass with ``nb`` occupied pairs
        per batch: l3 and t3 gathered for a group of virtual pairs, and H.'''
        n, nc = 1, self.nc
        for gP in range(self.npg):
            k = min(nb, self.nPg[gP])
            for gQ in range(self.npg if k else 0):
                d = gP - gQ
                if abs(d) < nc:
                    n = max(n, k * self.nD[d] * (2 * self.nQg[gQ] + self.nPg[gQ]))
        return n

    def gamma_elems(self, ng):
        '''... of the pass that makes Gamma with ``ng`` occupied triples
        per batch: t3 and l3 with one class of the free virtual index apart.'''
        n = 1
        for g in self.L.stored:
            k = min(ng, int(self.L.nI[g]))
            if not k or not self.L.nA[g]:
                continue
            for sc in range(self.nc):
                if 0 <= g - sc < self.npg:
                    n = max(n, 2 * k * self.nQg[g - sc] * self.nvc[sc])
        return n

    def most_pairs(self):
        return max(self.nPg + [1])

    def most_triples(self):
        return max([int(self.L.nI[g]) for g in self.L.stored] + [1])

    def gamma_bytes(self):
        '''Bytes of the diagonal pair blocks of Gamma (and of the packed
        dressed <ab||de>).'''
        return sum(n * n for n in self.nQg) * self.dtype.itemsize

    def row_bytes(self):
        '''Bytes of the rows of one first virtual index of the widest pair
        block of <ab||de>: the smallest slab.'''
        return max([min(self.vm.n, n) * n for n in self.nQg] + [1]) * self.dtype.itemsize

    def block_bytes(self):
        '''Bytes of the largest pair block.'''
        return max([n * n for n in self.nQg] + [1]) * self.dtype.itemsize

    def operand_bytes(self):
        '''Bytes of what the passes hold next to the work array: the copies
        in blocks of the operand [(m, Q), y] of pass 2 (the elements whose
        labels add up), the blocks of the ring density, the operand and the
        result of one stacked GEMM, and - outside of the passes - the
        columns of one pair block of two o v^3 / 2 arrays (Slabs, the
        contraction of the block aibc).'''
        nc, noc, nvc, nQg = self.nc, self.noc, self.nvc, self.nQg
        ny = [noc[s] + nvc[s] for s in range(nc)]
        gy = sum(noc[sm] * nQg[gQ] * ny[gQ - sm] for gQ in range(self.npg) for sm in range(nc)
                 if 0 <= gQ - sm < nc)
        ring = sum(n * n for n in self.nD.values())
        one = max([ny[gQ - sm] * noc[sm] * nQg[gQ] for gQ in range(self.npg) for sm in range(nc)
                   if 0 <= gQ - sm < nc] + [0])
        cols = 2 * self.om.n * self.vm.n * max(nQg + [0])
        return (max(gy + ring + 3 * one, cols) + ring) * self.dtype.itemsize

    # ---- the stacked GEMMs of a gathered block

    def _stacked(self, X, ws, gP, gQ, p0, p1, GPd, gy, X3, Yall, has):
        '''For X[P, D, Q] (the pairs p0 <= P < p1 of group gP, Q of group
        gQ), per class of d and of m:

          X3[x, m, Q]   += sum_(P,d) GPd[x, P, d] X[P, d, m, Q]
          Yall[P, d, y] += sum_(m,Q) X[P, d, m, Q] gy[(m, Q), y]

        x and y of the class gP - class(d); x over [virtual | occupied] and
        y over [occupied | virtual], the parts that ``has`` = (xv, xo, yo,
        yv) names (GPd and Yall hold these parts only).  ``ws``: storage
        for a copy of the rows of one block of classes (may be that of
        another gathered block that is no longer needed, not of X).'''
        e, om, vm, dtype = self.t3, self.om, self.vm, self.dtype
        nv, no = vm.n, om.n
        xv, xo, yo, yv = has
        delta = gP - gQ
        Qlo, Qhi = vm.pair_off[gQ], vm.pair_off[gQ + 1]
        nbp, nr, nQ = X.shape
        xo0 = nv if xv else 0           # where the occupied x start
        yv0 = no if yo else 0           # where the virtual y start
        r0 = 0
        for sd, sm in e.dsub[delta]:
            (d0, d1), (m0, m1) = self.vr[sd], self.orr[sm]
            nd, nm = d1 - d0, m1 - m0
            sy = gP - sd                    # the class of x and of y
            rows = slice(r0, r0 + nd * nm)
            r0 += nd * nm
            if not (nd and nm and 0 <= sy < self.nc):
                continue
            (e0, e1), (j0, j1) = self.vr[sy], self.orr[sy]
            nxe, nxj = (e1 - e0 if xv else 0), (j1 - j0 if xo else 0)
            nyj, nye = (j1 - j0 if yo else 0), (e1 - e0 if yv else 0)
            if not (nxe + nxj or nyj + nye):
                continue
            if nr == nd * nm:       # one block of classes: X as it is
                Xf = X.reshape(nbp * nd, nm * nQ)
            else:                   # [(P, d), (m, Q)] in the storage of another block
                Xf = ws[:nbp * nd * nm * nQ].reshape(nbp, nd * nm, nQ)
                t3pack.axpby(Xf, X[:, rows], 1.0, 0.0)
                Xf = Xf.reshape(nbp * nd, nm * nQ)
            if nxe + nxj:
                G = np.empty((nxe + nxj, nbp, nd), dtype=dtype)
                if nxe:
                    G[:nxe] = GPd[e0:e1, p0:p1, d0:d1]
                if nxj:
                    G[nxe:] = GPd[xo0 + j0:xo0 + j1, p0:p1, d0:d1]
                R = gemm(G.reshape(nxe + nxj, nbp * nd), Xf).reshape(nxe + nxj, nm, nQ)
                if nxe:
                    t3pack.axpby(X3[e0:e1, m0:m1, Qlo:Qhi], R[:nxe])
                if nxj:
                    t3pack.axpby(X3[xo0 + j0:xo0 + j1, m0:m1, Qlo:Qhi], R[nxe:])
                del R, G
            if nyj + nye:
                R = gemm(Xf, gy(sm, gQ, sy)).reshape(nbp, nd, nyj + nye)
                if nyj:
                    t3pack.axpby(Yall[p0:p1, d0:d1, j0:j1], R[:, :, :nyj])
                if nye:
                    t3pack.axpby(Yall[p0:p1, d0:d1, yv0 + e0:yv0 + e1], R[:, :, nyj:])
                del R

    def _gy(self, G3, has):
        '''The function (class of m, gQ, class of y) -> the operand
        [(m, Q), y] of these classes, a copy made once.'''
        vm, dtype, no = self.vm, self.dtype, self.om.n
        yo, yv = has
        yv0 = no if yo else 0
        cache = {}

        def gy(sm, gQ, sy):
            key = (sm, gQ, sy)
            if key not in cache:
                (m0, m1), (j0, j1), (e0, e1) = self.orr[sm], self.orr[sy], self.vr[sy]
                Qlo, Qhi = vm.pair_off[gQ], vm.pair_off[gQ + 1]
                nj, ne = (j1 - j0 if yo else 0), (e1 - e0 if yv else 0)
                G = np.empty((m1 - m0, Qhi - Qlo, nj + ne), dtype=dtype)
                if nj:
                    t3pack.axpby(G[:, :, :nj], G3[m0:m1, Qlo:Qhi, j0:j1], 1.0, 0.0)
                if ne:
                    t3pack.axpby(G[:, :, nj:], G3[m0:m1, Qlo:Qhi, yv0 + e0:yv0 + e1], 1.0, 0.0)
                cache[key] = G.reshape((m1 - m0) * (Qhi - Qlo), nj + ne)
            return cache[key]
        return gy

    def _pairs(self, nb):
        '''(gP, pairs of a batch, [gQ]) of a pass.'''
        om, nc = self.om, self.nc
        for gP in range(self.npg):
            gQs = [gQ for gQ in range(self.npg)
                   if abs(gP - gQ) < nc and self.nQg[gQ] and self.nD[gP - gQ]]
            if not gQs:
                continue
            for pb in _batches(om.pair_off[gP], om.pair_off[gP + 1], max(1, nb)):
                yield gP, pb, gQs

    # ---- pass 1

    def pass_l(self, L3, t2mn, t2P, nb):
        '''With L[P, d, m, Q], the half-expanded l3:

          Z[P, g, i]   =  sum_(m,Q) L[P, g, m, Q] t2P[i, m, Q]
          C3x[c, i, Q] = sum_(P,e) t2mn[P, c, e] L[P, e, i, Q]

        t2mn (npo, nv, nv), t2P (no, no, nvp); returns (Z, C3x).'''
        om, vm, dtype = self.om, self.vm, self.dtype
        no, nv, npo, nvp = om.n, vm.n, om.npair, vm.npair
        L3 = self.view(L3)
        Z = np.zeros((npo, nv, no), dtype=dtype)
        C3x = np.zeros((nv, no, nvp), dtype=dtype)
        if not self.size:
            return Z, C3x
        GPd = np.ascontiguousarray(t2mn.transpose(1, 0, 2), dtype=dtype)       # [c, P, e]
        G3 = np.ascontiguousarray(t2P.transpose(1, 2, 0), dtype=dtype)         # [m, Q, i]
        gy = self._gy(G3, (True, False))
        for gP, pb, gQs in self._pairs(nb):
            p0, p1, nbp = pb[0], pb[-1] + 1, len(pb)
            for gQ in gQs:
                delta = gP - gQ
                Qlo, Qhi = vm.pair_off[gQ], vm.pair_off[gQ + 1]
                n = nbp * self.nD[delta] * self.nQg[gQ]
                ws = t3engine.workspace(2 * n, dtype)
                with timer('gather'):
                    L = self.l._gather(L3, pb, delta, Qlo, Qhi, ws[:n])
                with timer('gemm'):
                    self._stacked(L, ws[n:], gP, gQ, p0, p1, GPd, gy, C3x, Z,
                                  (True, False, True, False))
        return Z, C3x

    # ---- pass 2

    def pass_tl(self, T3, L3, GxT, GmQ, l2pp, l1v, t2pp, nb, XT, YT):
        '''With T[P, d, m, Q] and L[P, d, m, Q], the half-expanded t3 and l3:

          XT[x, (i, Q)]     += sum_(P,d) GxT[P, d, x] T[P, d, i, Q]
          YT[P, d, y]       += sum_(m,Q) T[P, d, m, Q] GmQ[(m, Q), y]
          B1[(a, i)]         = sum_(P,Q) T[P, a, i, Q] l2pp[P, Q]
          A1[P, Q]           = sum_(e,m) l1v[(e, m)] T[P, e, m, Q]
          R3[(a,j), (b,i)]   = sum_(P,Q) L[P, a, j, Q] T[P, b, i, Q]
          Kg[P', Q']         = sum_(P,d,m) H[P, d, m, P'] T[P, d, m, Q'],
                               H[P, d, m, P'] = sum_Q L[P, d, m, Q] t2pp[P', Q]

        GxT (npo, nv, nv + no), GmQ (no nvp, no + nv), XT (nv + no, no nvp),
        YT (npo, nv, no + nv); returns (B1, A1, R3, Kg).'''
        om, vm, dtype = self.om, self.vm, self.dtype
        no, nv, npo, nvp = om.n, vm.n, om.npair, vm.npair
        dm = nv * no
        T3, L3 = self.view(T3), self.view(L3)
        B1 = np.zeros(dm, dtype=dtype)
        A1 = np.zeros((npo, nvp), dtype=dtype)
        R3 = np.zeros((dm, dm), dtype=dtype)
        Kg = np.zeros((npo, nvp), dtype=dtype)
        if not self.size:
            return B1, A1, R3, Kg
        GPd = np.ascontiguousarray(GxT.transpose(2, 0, 1), dtype=dtype)        # [x, P, d]
        gy = self._gy(GmQ.reshape(no, nvp, no + nv), (True, True))
        X3 = XT.reshape(nv + no, no, nvp)
        Dd = {delta: np.zeros((self.nD[delta],) * 2, dtype=dtype) for delta in self.deltas}
        D0 = np.zeros(self.nD[0], dtype=dtype)
        l1D = np.ascontiguousarray(l1v[self.flat[0]], dtype=dtype)
        Kb, t2T = {}, {}
        for gQ in range(self.npg):      # t2pp[P', Q]: P' and Q of one group
            plo, phi = om.pair_off[gQ], om.pair_off[gQ + 1]
            Qlo, Qhi = vm.pair_off[gQ], vm.pair_off[gQ + 1]
            if phi > plo and Qhi > Qlo:
                t2T[gQ] = np.ascontiguousarray(t2pp[plo:phi, Qlo:Qhi].T, dtype=dtype)
                Kb[gQ] = np.zeros((phi - plo, Qhi - Qlo), dtype=dtype)
        for gP, pb, gQs in self._pairs(nb):
            p0, p1, nbp = pb[0], pb[-1] + 1, len(pb)
            for gQ in gQs:
                delta = gP - gQ
                Qlo, Qhi = vm.pair_off[gQ], vm.pair_off[gQ + 1]
                nr, nQ, nPq = self.nD[delta], self.nQg[gQ], self.nPg[gQ]
                n = nbp * nr * nQ
                ws = t3engine.workspace(2 * n + nbp * nr * nPq, dtype)
                with timer('gather'):
                    L = self.l._gather(L3, pb, delta, Qlo, Qhi, ws[:n])
                    T = self.l._gather(T3, pb, delta, Qlo, Qhi, ws[n:2 * n])
                with timer('ring'):
                    for k in range(nbp):
                        gemm(L[k], T[k].T, Dd[delta], beta=1.0)
                with timer('h'):
                    if gQ in t2T:
                        H = ws[2 * n:2 * n + nbp * nr * nPq].reshape(nbp * nr, nPq)
                        gemm(L.reshape(nbp * nr, nQ), t2T[gQ], H)
                        gemm(H.T, T.reshape(nbp * nr, nQ), Kb[gQ], beta=1.0)
                with timer('t'):
                    if delta == 0:
                        for k in range(nbp):
                            D0 += T[k] @ l2pp[p0 + k, Qlo:Qhi]
                        A1[p0:p1, Qlo:Qhi] = np.matmul(l1D, T)
                    # (l3 is done with: its storage takes the copies of the rows of t3)
                    self._stacked(T, ws[:n], gP, gQ, p0, p1, GPd, gy, X3, YT,
                                  (True, True, True, True))
        for delta in self.deltas:
            fl = self.flat[delta]
            R3[fl[:, None], fl] = Dd[delta]
        B1[self.flat[0]] = D0
        for gQ, K in Kb.items():
            Kg[om.pair_off[gQ]:om.pair_off[gQ + 1], vm.pair_off[gQ]:vm.pair_off[gQ + 1]] = K
        return B1, A1, R3, Kg

    # ---- Gamma

    def gamma(self, T3, L3, ng):
        '''The diagonal pair blocks of Gamma[(e<f), (g<h)] = 3 sum_(m<n<o,
        d) l3[mno, d, (ef)] t3[mno, d, (gh)]: a list, block h the virtual
        pairs of group h.'''
        om, vm, dtype, L = self.om, self.vm, self.dtype, self.L
        T3, L3 = self.view(T3), self.view(L3)
        Gam = [np.zeros((n, n), dtype=dtype) for n in self.nQg]
        if not self.size:
            return Gam
        fg = getattr(t3pack._lib, 'zc_bgather_vir_' + t3pack._sfx(dtype))
        I_, p_ = _CArgs.I, _CArgs.p
        vspin, coff, ncs = self.l._vir_args()
        nvp = vm.npair
        for g in L.stored:
            if vm.tri_off[g + 1] == vm.tri_off[g]:
                continue
            for rows in _batches(om.tri_off[g], om.tri_off[g + 1], max(1, ng)):
                nb = len(rows)
                size = self.l._vir_sizes(g, nb, ncs)
                for sc in sorted(size):
                    h = g - sc
                    Qlo, nQ, ncc, n = vm.pair_off[h], self.nQg[h], int(ncs[sc]), size[sc]
                    ws = t3engine.workspace(2 * n, dtype)
                    with timer('gather'):
                        for X, out in ((T3, ws[:n]), (L3, ws[n:])):
                            fg(p_(X), p_(L.obase), p_(L.vcol), p_(L.osig), p_(L.vsig), p_(vm.tri_c),
                               p_(vm.sgn_c), I_(nvp), I_(nb), p_(rows), I_(Qlo), I_(nQ),
                               I_(coff[sc]), I_(ncc), p_(out))
                    gemm(ws[n:].reshape(nQ, nb * ncc), ws[:n].reshape(nQ, nb * ncc).T, Gam[h],
                         alpha=3.0, beta=1.0)
        return Gam

    def gram(self, L3, T3):
        '''G[I, J] = sum_A l3[I, A] t3[J, A] (I and J of one group).'''
        return self.l.gram(L3, T3)


def gamma_pairs(out, Gam, V, vm, cols, alpha=1.0):
    '''density_opt._gamma_pairs for Gamma as the list of its diagonal pair
    blocks (BlockDensity.gamma) and the pairs of ``vm`` in the order of
    their groups: out[q, y] += alpha sum_Q G[Q, (p<q)] V[Q, p, y] and
    out[p, y] -= alpha sum_Q G[Q, (p<q)] V[Q, q, y] (``cols``; else with
    G[(p<q), Q]), Q of the group of (p<q).  V (npair, n, ncol).  In a group
    the pairs with one first index are a range of rows and of second
    indices; those with one second index are gathered.'''
    n, pidx = vm.n, vm.pidx
    if vm.npair == 0 or V.shape[2] == 0:
        return
    r = np.arange(n)
    for (lo, hi), G in zip(vm.blocks, Gam):
        if hi == lo:
            continue
        Vh = V[lo:hi]
        for p in range(n - 1):
            idx = pidx[p, p + 1:]
            sel = (idx >= lo) & (idx < hi)
            if not sel.any():
                continue
            rows, qs = idx[sel] - lo, r[p + 1:][sel]
            if rows[-1] - rows[0] == len(rows) - 1 and qs[-1] - qs[0] == len(qs) - 1:
                r0, r1 = rows[0], rows[-1] + 1
                gemm(G[:, r0:r1].T if cols else G[r0:r1], Vh[:, p, :], out[qs[0]:qs[-1] + 1],
                     alpha=alpha, beta=1.0)
            else:
                M = np.take(G, rows, axis=1 if cols else 0)
                out[qs] += alpha * gemm(M.T if cols else M, Vh[:, p, :])
        for q in range(1, n):
            idx = pidx[:q, q]
            sel = (idx >= lo) & (idx < hi)
            if not sel.any():
                continue
            rows, ps = idx[sel] - lo, r[:q][sel]
            M = np.take(G, rows, axis=1 if cols else 0)
            if ps[-1] - ps[0] == len(ps) - 1:
                gemm(M.T if cols else M, Vh[:, q, :], out[ps[0]:ps[-1] + 1], alpha=-alpha, beta=1.0)
            else:
                out[ps] -= alpha * gemm(M.T if cols else M, Vh[:, q, :])
            del M


class Slabs:
    '''The packed T1-dressed <ab||de> as its diagonal pair blocks, in slabs
    of the rows of a block: iterating gives (h, lo, hi, q0, q1, W) - block h,
    the pairs lo <= Q < hi, with W[q - q0, Q - lo] for the rows q0 <= q < q1
    of the block.  All blocks whole, dressed once and kept (``slab_bytes``
    None), or ``slab_bytes`` at a time, dressed again for every sweep.'''

    def __init__(self, eris, t1, om, vm, dtype, slab_bytes):
        self.args = (eris, t1, om, vm)
        self.keep = slab_bytes is None
        self.dtype, self.slab_bytes = np.dtype(dtype), slab_bytes
        self.kept, self.buf = {}, None

    def __iter__(self):
        eris, t1, om, vm = self.args
        isz = self.dtype.itemsize
        for h, (lo, hi) in enumerate(vm.blocks):
            n = hi - lo
            if n == 0:
                continue
            if self.keep:
                if h not in self.kept:
                    W = np.empty((n, n), dtype=self.dtype)
                    with timer('dens.vvvv'):
                        ccsdt_opt.dressed_vvvv_rows(eris, t1, om, vm, 0, n, W, block=h)
                    self.kept[h] = W
                yield h, lo, hi, 0, n, self.kept[h]
                continue
            rows = int(max(1, min(n, self.slab_bytes // (n * isz))))
            for q0 in range(0, n, rows):
                q1 = min(q0 + rows, n)
                self.buf = None
                self.buf = W = np.empty((q1 - q0, n), dtype=self.dtype)
                with timer('dens.vvvv'):
                    ccsdt_opt.dressed_vvvv_rows(eris, t1, om, vm, q0, q1, W, block=h)
                yield h, lo, hi, q0, q1, W
                del W
            self.buf = None

    def close(self):
        self.kept, self.buf = {}, None
