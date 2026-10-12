'''CC4 without stored quadruples (docs/ccn.md, section 11).

For semicanonical spinors the quadruples equation of CC4 is solved in
closed form,

    t4[ijkl, abcd] = <Q| [V~, T3] + 1/2 [[V~, T2], T2] |0> / (e_ijkl - e_abcd),

and T4 enters the doubles and triples equations of CCSDTQ only through four
sums that are linear in it:

    R2[ij, ab]  += sum_{m<n, e<f} <mn||ef> t4[ijmn, abef]
    R3[ijk, abc] += sum_{m e} f~[m, e] t4[ijkm, abce]
                  - P(c/ab) sum_{m, e<f} <cm||ef>~ t4[ijkm, abef]   (signs of the reductions of the code)
                  - P(k/ij) sum_{m<n, e} <mn||ek>~ t4[ijmn, abce]

So the quadruples of one occupied quadruple are made, contracted into R2
and R3 and discarded, the occupied quadruples one after the other.

The right-hand side is the quantity r of the (Q) correction
(zccsdt.cc.quadruples: [V, T3] + 1/2 [[V, T2], T2] for one occupied
quadruple, five products and the antisymmetrizing reductions of
csrc/tnpack.c) with the T1-dressed integrals in the place of the bare ones;
the (Q) class is used as it is, on a container of the dressed blocks
(Dressed): it reads <ab||ei> and <ma||ij> as the conjugates of <ie||ab> and
<ij||ma>, which holds for bare integrals only, so the container holds the
conjugates of the dressed blocks there.  The four sums are back(): the T4
row expanded twice by the kernels of csrc/tnpack.c ([(ab), (ef)] and
[e, (abc)]), five matrix products, the reductions onto the rows of R3.

Work per iteration: the (Q) products, o^4 v^4 (v + o) / 12 multiplications,
and o^2 v^5 / 4 for its intermediate of <vv||vv>; back(): o^4 v^5 / 24
(the product with <cm||ef>) + o^5 v^4 / 4; nothing of the size of T4 is
kept.  The excitation energies fold R4 in at a frequency omega.
'''

import ctypes
import os
from dataclasses import dataclass
from math import comb

import numpy as np

from zccsdt import timing
from zccsdt.blas import gemm
from zccsdt.cc import quadruples as Qm, t3pack, t4pack as P
from zccsdt.timing import timer


class _Shim:
    '''The integral container that zccsdt.cc.quadruples.Quadruples reads,
    made of the dressed blocks (t4engine.Dressed) and the diagonal Fock
    matrix e of the semicanonical spinors.'''

    def __init__(self, ints, e):
        no, nv = ints.nocc, ints.nvir
        a2, b2 = np.triu_indices(nv, 1)
        self.nocc = no
        self.mo_energy = np.asarray(e, dtype=float)
        self.fock = np.diag(self.mo_energy)
        dt = np.result_type(ints.vvvo_p, ints.ovoo, ints.oovv)
        # Quadruples takes <ab||ei> = -conj(ovvv[i, e, a, b]) (a < b): the dressed <ab||ei> there
        W = np.zeros((no, nv, nv, nv), dtype=dt)
        vp = np.asarray(ints.vvvo_p)                    # [(ab), e, i]
        W[:, :, a2, b2] = -np.conj(vp.transpose(2, 1, 0))
        W[:, :, b2, a2] = -W[:, :, a2, b2]
        self.ovvv = W
        # and <ma||ij> = conj(ooov[i, j, m, a])
        self.ooov = np.ascontiguousarray(np.conj(np.asarray(ints.ovoo).transpose(2, 3, 0, 1)))
        self.oooo, self.ovvo, self.oovv = ints.oooo, ints.ovvo, ints.oovv
        self._vvvv_pp = ints.vvvv_pp
        self.vvvv = None


class Back:
    '''The four sums of T4 in the doubles and triples equations, for one
    occupied quadruple at a time (back()); the operators of ``ints``.'''

    def __init__(self, ints, dtype):
        no, nv = ints.nocc, ints.nvir
        i2, j2 = np.triu_indices(no, 1)
        a2, b2 = np.triu_indices(nv, 1)
        c = lambda x: np.ascontiguousarray(x, dtype=dtype)                 # noqa: E731
        f = np.asarray(ints.f)
        self.no, self.nv, self.dtype = no, nv, np.dtype(dtype)
        self.nQ, self.nA3 = comb(nv, 2), comb(nv, 3)
        self.fov = c(f[:no, no:])                                           # [m, e]
        self.oovv_pp = c(np.asarray(ints.oovv)[i2, j2][:, a2, b2])          # [(mn), (ef)]
        self.vovv_p = c(np.asarray(ints.vovv)[:, :, a2, b2])                # [c, m, (ef)]
        self.oovo_pp = c(np.asarray(ints.oovo)[i2, j2])                     # [(mn), e, k]
        self.so22, self.so13, self.so12 = P.split(no, 2, 2), P.split(no, 1, 3), P.split(no, 1, 2)
        self.X22 = np.empty((self.nQ, self.nQ), dtype=dtype)
        self.X13 = np.empty((nv, self.nA3), dtype=dtype)
        self.Y = np.empty((nv, self.nQ), dtype=dtype)

    def __call__(self, I, row, R2, R3):
        '''R2[(ij), (ab)], R3[(ijk), (abc)] += the sums of the T4 row
        ``row`` (C(nv,4)) of the occupied quadruple I.'''
        nv, so22, so13, so12 = self.nv, self.so22, self.so13, self.so12
        with timer('cc4.back.expand'):
            X22 = P.tn_expand(self.X22, row, nv, 2, 2)                     # t4[(ab), (ef)]
            X13 = P.tn_expand(self.X13, row, nv, 1, 3)                     # t4[e, (abc)]
        with timer('cc4.back.gemm'):
            # R2: the six splits (P | P') of the quadruple
            Pp, Pq, sg = so22.idx1[I], so22.idx2[I], so22.sgn[I]
            Z = gemm(X22, np.ascontiguousarray(self.oovv_pp[Pq].T))         # [(ab), s]
            R2[Pp] += (Z * sg[None, :]).T
            # R3 with f~ and <cm||ef>~: the four splits (m | J)
            for s in range(4):
                m, J, g = so13.idx1[I, s], so13.idx2[I, s], float(so13.sgn[I, s])
                R3[J] += g * (self.fov[m] @ X13)
                gemm(np.ascontiguousarray(self.vovv_p[:, m]), X22.T, self.Y)        # [c, (ab)]
                P.tn_reduce(R3[J], self.Y, nv, 1, 2, scale=-g)
            # R3 with <mn||ek>~: the six splits (P | P'), the third occupied index k from the integral
            for s in range(6):
                Yk = gemm(np.ascontiguousarray(self.oovo_pp[Pq[s]].T), X13)        # [k, (abc)]
                pos, psg = so12.pos[:, Pp[s]], so12.psg[:, Pp[s]]
                ok = psg != 0
                R3[pos[ok]] -= (float(sg[s]) * psg[ok])[:, None] * Yk[ok]


def run(ints, e, t2, T3, omega=0.0, drives=None, back_ints=None, plan_=None, rows=None, max_memory=None):
    '''All occupied quadruples: the T4 rows over the denominator
    e_ijkl - e_abcd + omega, contracted by Back.  ``drives``: a list of
    (ints, t2, T3, weight) whose right-hand sides are summed (default
    [(ints, t2, T3, 1)]); ``back_ints``: the operators of the sums (default
    ints).  Returns (R2 [C(o,2), C(v,2)], R3 [C(o,3), C(v,3)]).  ``rows``:
    a list that receives (I, row) (tests).  ``plan_``: the Plan of the pass
    or the quadruples.Plan of its (Q) objects (default: plan() for
    ``max_memory`` MB of the whole process).  One process: the products of
    the packed layout are large, and the threads of the BLAS do them.'''
    from zccsdt import memory
    no, nv = ints.nocc, ints.nvir
    e = np.asarray(e, dtype=float)
    drives = drives or [(ints, t2, T3, 1.0)]
    dt = np.result_type(*[np.result_type(d[1], d[2], d[0].oovv) for d in drives])
    R2 = np.zeros((comb(no, 2), comb(nv, 2)), dtype=dt)
    R3 = np.zeros((comb(no, 3), comb(nv, 3)), dtype=dt)
    if no < 4 or nv < 4:
        return R2, R3
    if plan_ is None:
        plan_ = plan(no, nv, np.dtype(dt).itemsize, memory.budget(max_memory), None, 0, 1, len(drives))
    if isinstance(plan_, Plan):
        plan_ = plan_.dense
    back = Back(back_ints or ints, dt)
    qs = []
    with timer('cc4.setup'):
        for d_ints, d2, d3, w in drives:
            qs.append((Qm.Quadruples(_Shim(d_ints, e), np.asarray(d2, dtype=dt), np.asarray(d3, dtype=dt), plan_), w))
    q0 = qs[0][0]
    ev4 = q0.ev4
    row = np.empty(comb(nv, 4), dtype=dt)
    for l, xs, Is in q0.schedule():
        xvs = [q.xi_v(l, xs) for q, _ in qs]
        for I in Is:
            with timer('cc4.rhs'):
                row[:] = 0
                for (q, w), (Xv, where) in zip(qs, xvs):
                    rt3, rtt = q.right(I, Xv, where)
                    row += w * rt3
                    row += w * rtt
                row /= q0.eo4[I] + omega - ev4
            if rows is not None:
                rows.append((I, row.copy()))
            back(I, row, R2, R3)
    return R2, R3


# ------------------------------------------------------------ excitation energies

def dressed_derivative(eris, t1, r1):
    '''The derivative of the dressed blocks (t4engine.Dressed) along r1 -
    the blocks of [H~, R1] - by the five-point formula, exact for the
    dressing, a polynomial of degree four at most in t1.'''
    from dataclasses import fields
    from zccsdt.cc import t4engine
    D = [t4engine.Dressed.from_eris(eris, t1 + h * r1) for h in (1.0, -1.0, 2.0, -2.0)]
    out = {}
    for fl in fields(t4engine.Dressed):
        a = [getattr(d, fl.name) for d in D]
        if fl.name in ('nocc', 'nvir'):
            out[fl.name] = a[0]
            continue
        a = [np.asarray(x) for x in a]
        out[fl.name] = (8.0 * (a[0] - a[1]) - (a[2] - a[3])) / 12.0
    return t4engine.Dressed(**out)


def sigma(g, ints, e, r1, r2p, R3, omega, **kw):
    '''The folded Jacobian of CC4 at the frequency omega on (r1, r2p, R3):
    the Jacobian of the singles, doubles and triples equations of CCSDTQ at
    the CC4 amplitudes - that of CCSDT (eom_opt.sigma of the Ground ``g``)
    plus the T4 terms with the derivative of the dressed integrals - and the
    quadruples folded in,

        r4 = ([V', T3] + [V~, R3] + [[V', T2], T2] / 2 + [[V~, R2], T2]) / (omega + e_ijkl - e_abcd),

    contracted with the integrals of H~ (Back); the right-hand side by
    polarization: half the difference of the (Q) quantity at (t2 + r2, T3 +
    R3) and at (t2 - r2, T3 - R3), plus the one of the integrals of H' at
    (t2, T3).  ``ints``: the dressed blocks of the ground state, ``e`` the
    diagonal Fock matrix.  ``kw``: max_memory, workers of the passes
    (run_sectors).  Returns (s1, s2p, s3) as eom_opt.sigma.'''
    from zccsdt.cc import eom_opt, t3pack
    om, vm = g.om, g.vm
    no, nv = om.n, vm.n
    s1, s2, s3 = eom_opt.sigma(g, r1, r2p, R3)
    r2 = t3pack.unpack_pairs(np.asarray(r2p).reshape(om.npair, vm.npair), om, vm)
    T3 = np.asarray(g.T3).reshape(comb(no, 3), comb(nv, 3))
    R3m = np.asarray(R3).reshape(T3.shape)
    with timer('cc4.dint'):
        dints = dressed_derivative(g.eris, g.t1, np.asarray(r1).reshape(no, nv))
    t2 = g.t2
    with timer('cc4.r4'):
        a2, a3 = run_for(g.eris, ints, e, t2, T3, omega=omega, back_ints=ints,
                     drives=[(ints, t2 + r2, T3 + R3m, 0.5), (ints, t2 - r2, T3 - R3m, -0.5), (dints, t2, T3, 1.0)],
                     **kw)
    with timer('cc4.t4'):
        b2, b3 = run_for(g.eris, ints, e, t2, T3, back_ints=dints, **kw)
    s1 = s1
    s2 = s2 + (a2 + b2).reshape(s2.shape)
    s3 = s3 + (a3 + b3).reshape(s3.shape)
    return s1, s2, s3


def eom(eris, t1, t2, T3, nroots=3, conv_tol=1e-6, omega_tol=None, guess_tol=1e-4, max_cycle=60, verbose=False,
        extra=2, max_memory=None, workers=None):
    '''Excitation energies of CC4: the folded eigenproblem in the space of
    singles, doubles and triples (R4 folded in at the frequency omega), each
    root self-consistent in omega (ccn.folded_roots), started from the roots
    of the singles-doubles block.  The amplitudes of the semicanonical
    integrals ``eris`` (ccn.cc4_kernel: Result.eris).  Packed layout; where
    the integrals carry labels of a linear molecule the quadruples are made
    in the sectors of the (Q) code and the states are the totally symmetric
    ones (label shift 0).  T3 in the blocks of the labels of ``eris`` is put
    on the packed block.  ``max_memory`` (MB, the whole process), ``workers``:
    of the passes over the quadruples (plan()).
    Returns (energies, the Solved of folded_roots).'''
    from zccsdt.cc import ccn, ccsdt_opt, eom as E, eom_opt, t4engine
    if getattr(eris, 'use_labels', False):
        import copy
        T3 = ccsdt_opt.packed_t3(eris, np.asarray(T3))
        eris = copy.copy(eris)
        eris.use_labels = False
        eris.__dict__.pop('_buffers', None)
    g = eom_opt.prepare(eris, t1, t2, T3)
    ints = t4engine.Dressed.from_eris(eris, g.t1)
    e = np.asarray(eris.fock).diagonal().real
    sizes = g.sizes()
    no, nv = g.om.n, g.vm.n
    store = E.Store(sizes, g.dtype)
    # in the symmetry sectors the quadruples follow the label of the occupied quadruple: the
    # totally symmetric states (label shift 0), every vector projected on them
    from zccsdt.cc import quadruples_sym as S
    sec = None
    if getattr(eris, 'labels', None) is not None and S.available(eris) and not getattr(eris, 'no_sectors', False) \
            and not getattr(eris, 'modulus', 0):
        sec = E.Sector(g.om, g.vm, eris.labels, 0)

    def apply_at(omega):
        def apply(x, y):
            if omega is None:
                s1, s2, _ = eom_opt.sigma(g, x[0].reshape(no, nv), x[1], None, sd=True)
                y[0][:], y[1][:] = s1.ravel(), s2.ravel()
                y[2][:] = 0
                if sec is not None:
                    sec.project(y)
                return
            s1, s2, s3 = sigma(g, ints, e, x[0].reshape(no, nv), x[1], x[2], omega, max_memory=max_memory,
                               workers=workers)
            y[0][:], y[1][:], y[2][:] = s1.ravel(), s2.ravel(), s3.ravel()
            if sec is not None:
                sec.project(y)
        return apply
    precond = E.make_precond(g, True)
    gs = []
    for k in E.singles_guess(g, nroots + extra, E.symmetric_singles(eris, 0) if sec is None else sec.m1):
        v = store.new()
        v[0][k] = 1.0
        gs.append(v)
    first = E.davidson(apply_at(None), precond, gs, store, len(gs), guess_tol, 4 * len(gs) + 16, max_cycle)
    order = np.argsort(first.e.real, kind='stable')[:nroots]
    tol_w = max(1e-10, 1e-2 * conv_tol) if omega_tol is None else omega_tol
    log = (lambda k, it, lam, gap, res, n: print('  root %d step %d omega %.12f gap %.1e res %.1e %d' % (k + 1, it, lam, gap, res, n))) \
        if verbose else None
    sol = ccn.folded_roots(apply_at, precond, None, store, nroots, conv_tol, tol_w, max_cycle=max_cycle, log=log,
                           start=[(float(first.e[k].real), first.vectors[k]) for k in order])
    srt = np.argsort(sol.e.real, kind='stable')
    return sol.e.real[srt], sol


# ------------------------------------------------------------ symmetry sectors

def _sector_blocks(sec, q, I, M, XV, kpair, xl, B1, B2, C1, C2, w):
    '''B1, B2 (on two pairs), C1, C2 (on an index and a triple) += w * the
    blocks of the products of the occupied quadruple I (total label M), as
    zccsdt.cc.quadruples_sym.kernel forms them, for the (Q) object q.'''
    from zccsdt.cc import quadruples_sym as S
    ko, dtype = sec.ko, q.dtype
    b22, _, b13, _, off22, nc22, off13, nc13 = sec.layout(M)
    x13, J13, s13 = q.so13.idx1[I], q.so13.idx2[I], q.so13.sgn[I]
    p22, q22, s22 = q.so22.idx1[I], q.so22.idx2[I], q.so22.sgn[I]
    so12, T3 = q.so12, q.T3
    rg = getattr(P._lib, 'zc_rows_gemm_' + P._SFX[np.dtype(dtype)])
    p_, I64 = P._p, S._I64
    for qq, off, nr, nc in b22:
        rq, cq = sec.Q2[qq], sec.Q2[int(sec.red(M - qq))]
        LA, RA, LB, RB = [], [], [], []
        for k in range(4):
            x, J, s = int(x13[k]), int(J13[k]), float(s13[k])
            ke = int(sec.red(qq - ko[x]))
            if ke in sec.V:
                col, sg = sec.t3_rows(ke, int(sec.red(M - qq)))
                LA.append(-s * q.vvvo_t[x][np.ix_(sec.V[ke], rq)])
                RA.append(T3[J][col] * sg)
            if ke in sec.O:
                ms = sec.O[ke]
                LB.append(s * q.t2_xp[x][np.ix_(ms, rq)])
                RB.append(q.Xo[J][np.ix_(ms, cq)])
        for Ls, Rs, B in ((LA, RA, B1), (LB, RB, B2)):
            if Ls:
                X = B[off:off + nr * nc].reshape(nr, nc)
                gemm(np.ascontiguousarray(S._stack(Ls, nr, dtype).T), S._stack(Rs, nc, dtype), X, alpha=w, beta=1.0)
    for pp, off, nr, nc in b13:
        va, a3 = sec.V[pp], sec.A3[int(sec.red(M - pp))]
        LC, rows, sgs = [], [], []
        for k in range(6):
            km = int(sec.red(kpair[p22[k]] - pp))
            if km in sec.O:
                ms = sec.O[km]
                sg = so12.psg[ms, q22[k]]
                ms = ms[sg != 0]
                if len(ms):
                    LC.append(float(s22[k]) * q.ovoo_t[p22[k]][np.ix_(ms, va)])
                    rows.append(so12.pos[ms, q22[k]])
                    sgs.append(sg[sg != 0])
        if LC:
            X = np.empty((nr, nc), dtype=dtype)
            L = np.ascontiguousarray(S._stack(LC, nr, dtype))
            rr = np.ascontiguousarray(np.concatenate(rows), dtype=np.int64)
            ss = np.ascontiguousarray(np.concatenate(sgs), dtype=np.float64)
            cols = np.ascontiguousarray(a3, dtype=np.int64)
            rg(p_(T3), I64(T3.shape[1]), p_(rr), p_(ss), I64(len(rr)), p_(cols), I64(nc), p_(L), I64(nr), p_(X))
            C1[off:off + nr * nc] += w * X.ravel()
        X = C2[off:off + nr * nc].reshape(nr, nc)
        for k in (0, 1, 3):
            ke = int(sec.red(kpair[p22[k]] - pp))
            if ke in sec.V:
                L = np.ascontiguousarray(q.t2_px[p22[k]][np.ix_(sec.V[ke], va)].T)
                gemm(L, XV[int(xl[q22[k], 0])][ke], X, alpha=2.0 * float(s22[k]) * w, beta=1.0)


class BackBlocks(Back):
    '''Back for the blocks of one occupied quadruple: E22 on two pairs and
    E13 on an index and a triple (the T4 row expanded into the places of
    the blocks of the sectors).  The sums go into R2 and R3 through
    zc_rows_cols_add, a row of R3 at a time into the columns of one class
    of virtual triples - under the lock of that class where several
    processes add to the same arrays (``locks``: (the lock of R2, {label of
    the triples: lock})).'''

    def __init__(self, ints, dtype, sec):
        no, nv = ints.nocc, ints.nvir
        i2, j2 = np.triu_indices(no, 1)
        a2, b2 = np.triu_indices(nv, 1)
        c = lambda x: np.ascontiguousarray(x, dtype=dtype)                 # noqa: E731
        self.no, self.nv, self.dtype = no, nv, np.dtype(dtype)
        self.nQ, self.nA3 = comb(nv, 2), comb(nv, 3)
        self.fov = c(np.asarray(ints.f)[:no, no:])
        self.oovv_pp = c(np.asarray(ints.oovv)[i2, j2][:, a2, b2])
        self.vovv_p = c(np.asarray(ints.vovv)[:, :, a2, b2])
        self.oovo_pp = c(np.asarray(ints.oovo)[i2, j2])
        self.so22, self.so13, self.so12 = P.split(no, 2, 2), P.split(no, 1, 3), P.split(no, 1, 2)
        self.lab_pair = sec.red(sec.ko[P.tuples(no, 2)].sum(axis=1))
        self.radd = getattr(P._lib, 'zc_rows_cols_add_' + P._SFX[self.dtype])
        self.one = np.ones(1)

    def _add(self, R, rows, sg, cols, X, lock):
        rows = np.ascontiguousarray(rows, dtype=np.int64)
        sg = np.ascontiguousarray(sg, dtype=np.float64)
        X = np.ascontiguousarray(X, dtype=self.dtype)
        p_, I64 = P._p, ctypes.c_int64
        args = (p_(R), I64(R.shape[1]), p_(rows), p_(sg), I64(len(rows)), p_(cols), I64(len(cols)), p_(X))
        if lock is None:
            self.radd(*args)
        else:
            with lock:
                self.radd(*args)

    def __call__(self, sec, I, M, E22, E13, R2, R3, Y, row, locks=None):
        '''``Y`` [nv, C(nv,2)], ``row`` [C(nv,3)] (zero on entry and on
        return): work arrays of the caller.'''
        nv = self.nv
        so22, so13, so12 = self.so22, self.so13, self.so12
        b22, _, b13, _, off22, nc22, off13, nc13 = sec.layout(M)
        ko, lab_pair, add = sec.ko, self.lab_pair, self._add
        lock2, lock3 = locks if locks is not None else (None, {})
        blk22 = {qq: E22[off:off + nr * nc].reshape(nr, nc) for qq, off, nr, nc in b22}
        blk13 = {pp: E13[off:off + nr * nc].reshape(nr, nc) for pp, off, nr, nc in b13}
        Pp, Pq, sg22 = so22.idx1[I], so22.idx2[I], so22.sgn[I]
        for s in range(6):          # R2: the block of the label of P
            qq = int(lab_pair[Pp[s]])
            if qq not in blk22:
                continue
            cq = sec.Q2[int(sec.red(M - qq))]
            y = blk22[qq] @ self.oovv_pp[Pq[s]][cq]
            add(R2, [Pp[s]], [float(sg22[s])], sec.Q2c[qq], y[None, :], lock2)
        for s in range(4):          # R3: f~ and <cm||ef>~
            m, J, g = int(so13.idx1[I, s]), int(so13.idx2[I, s]), float(so13.sgn[I, s])
            pm = int(ko[m])
            lJ = int(sec.red(M - pm))
            if lJ not in sec.A3:    # (no virtual triples with the label of J)
                continue
            cols = sec.A3c[lJ]
            hit = False
            for qq, X in blk22.items():
                lc = int(sec.red(M - qq - pm))
                if lc not in sec.V:
                    continue
                cq = sec.Q2[int(sec.red(M - qq))]
                Wc = np.ascontiguousarray(self.vovv_p[sec.V[lc], m][:, cq])
                Y[np.ix_(sec.V[lc], sec.Q2[qq])] = Wc @ X.T
                hit = True
            if hit:
                P.tn_reduce(row, Y, nv, 1, 2, scale=-g)
                Y[...] = 0
            x = row[cols]
            if hit:
                row[cols] = 0
            if pm in blk13:
                x += g * (self.fov[m][sec.V[pm]] @ blk13[pm])
            elif not hit:
                continue
            add(R3, [J], self.one, cols, x[None, :], lock3.get(lJ))
        for s in range(6):          # R3: <mn||ek>~
            lP = int(lab_pair[Pq[s]])
            for pp, X in blk13.items():
                lk = int(sec.red(lP - pp))
                if lk not in sec.O:
                    continue
                ks = sec.O[lk]
                psg = so12.psg[ks, Pp[s]]
                ok = psg != 0
                if not ok.any():
                    continue
                Yk = np.ascontiguousarray(self.oovo_pp[Pq[s]][np.ix_(sec.V[pp], ks[ok])].T) @ X
                lJ = int(sec.red(M - pp))
                add(R3, so12.pos[ks[ok], Pp[s]], -float(sg22[s]) * psg[ok], sec.A3c[lJ], Yk, lock3.get(lJ))


# ------------------------------------------------------------ memory plan

# private pages of a forked process besides its work arrays: the buffers of its BLAS and what it
# writes of the pages of the parent (measured: 50 to 70 MB, HBr with 28 and 60 virtual spinors)
WORKER_HEAP = 80 << 20
SMALL = 20_000_000              # elements of T4 below which a pass is not worth other processes
PER_WORKER = 32                 # occupied quadruples per process, at least
LEAN = True                     # the kernel zc_cc4_sector_row where the library has it (False: tests)


@dataclass
class Plan:
    '''The memory of one pass over the occupied quadruples (plan()).'''
    workers: int            # processes over the occupied quadruples (1: the calling process alone)
    xo_workers: int         # of which form the intermediate Xo
    xv: int                 # occupied indices x whose Xv(x, l) are held at a time
    need: int               # bytes of the pass besides its operands (the dressed blocks, T3 packed)
    floor: int              # the least that a pass takes
    fixed: int              # of ``need``: what does not depend on the number of processes
    per_worker: int         # the work arrays of one process
    sectors: bool           # in the blocks of the symmetry labels
    dense: object           # the quadruples.Plan of the (Q) objects
    ndrives: int = 1
    slab: int = 0           # elements of the blocks of Xv of one pair (x, l), at most
    n22: int = 0            # elements of the largest set of blocks X[(ab), (cd)]
    n13: int = 0            # and X[a, (bcd)]
    auto: bool = False      # made by plans() beside the plan of the CCSDT step: add_quadruples may
                            # take more processes where the memory that is resident then leaves room


def _lean():
    return LEAN and hasattr(P._lib, 'zc_cc4_sector_row_d')


def _cores():
    from zccsdt.blas import _threads
    return max(1, _threads())


def _sector_labels(eris):
    '''(labels, modulus) if the quadruples of these integrals are made in
    the sectors of their labels (run_sectors), else (None, 0).'''
    from zccsdt.cc import quadruples_sym as S
    if getattr(eris, 'labels', None) is not None and S.available(eris) and not getattr(eris, 'no_sectors', False):
        return np.asarray(eris.labels), int(getattr(eris, 'modulus', 0) or 0)
    return None, 0


def _sectors(no, labels, modulus):
    from types import SimpleNamespace
    from zccsdt.cc import quadruples_sym as S
    sec = S.Sectors(SimpleNamespace(nocc=no, labels=np.asarray(labels), modulus=modulus))
    i64 = lambda x: np.ascontiguousarray(x, dtype=np.int64)                # noqa: E731
    sec.A3c = {k: i64(v) for k, v in sec.A3.items()}       # (the columns as the kernels take them)
    sec.Q2c = {k: i64(v) for k, v in sec.Q2.items()}
    return sec


def plan(no, nv, isz, budget=None, labels=None, modulus=0, workers=None, ndrives=1):
    '''The Plan of a pass for ``budget`` bytes (None: no limit) - what
    run() (``labels`` None) or run_sectors() allocate: R2 and R3, per drive
    the container of the conjugated dressed blocks, the (Q) object with its
    intermediate Xo and the blocks of <bc||ef>, the operators of the four
    sums, Xv for the occupied indices held at a time, and per process the
    blocks of the products of one occupied quadruple.

    From the floor (one process, Xv for three occupied indices; below it
    MemoryError with the numbers) first Xv for more occupied indices, up to
    all, then more processes, up to ``workers`` (default: the cores of the
    process; one where T4 has fewer than SMALL elements) - one after the
    other, so that no part of a plan shrinks when the budget grows.  Each
    process beyond the calling one takes its work arrays and WORKER_HEAP;
    Xo is formed by as many of them as hold its operands.'''
    nP, nQ, nI3, nA3, nA = Qm._sizes(no, nv)
    out = (nP * nQ + nI3 * nA3) * isz
    shim = (no * nv ** 3 + 2 * no * nv * nQ + 2 * no ** 3 * nv) * isz
    back = (nP * nQ + 2 * nv * no * nQ + 2 * nP * nv * no + no * nv) * isz
    base, work, xo = Qm._fixed(no, nv, isz)
    xfull = max(3, no - 1)
    if labels is None:
        mine = out + back + ndrives * shim + isz * (nQ * nQ + nv * nA3 + nv * nQ + 12 * nQ + 3 * no * nA3
                                                    + 2 * nA3 + 2 * nA)
        try:
            pl = Qm.plan(no, nv, isz, None if budget is None else (budget - mine) // ndrives)
        except MemoryError:
            floor = mine + ndrives * Qm.explicit_plan(no, nv, isz, 1, 1, 3, 1).floor
            raise MemoryError('the quadruples of CC4 need %.0f MB besides T3 and the dressed integrals for %d '
                              'occupied and %d virtual spinors; %.0f MB are left'
                              % (floor / 1e6, no, nv, budget / 1e6)) from None
        need = mine + ndrives * pl.need
        return Plan(1, 1, pl.xv, int(need), int(mine + ndrives * pl.floor), int(need), 0, False, pl, ndrives)
    sec = _sectors(no, labels, modulus)
    n22, n13 = sec.sizes(P.tuples(no, 4))
    slab, xwork = sec.xv_size()
    maxq = max(len(x) for x in sec.Q2.values())
    maxa = max(len(x) for x in sec.A3.values())
    tables = 16 * nv * nQ + 32 * nA3 + 8 * nQ
    nb = (2, 3) if _lean() else (5, 6)      # (blocks of a process: products and rows, one more of C)
    wq = isz * (nb[0] * n22 + nb[1] * n13 + nv * nQ + nA3 + 4 * (4 * nv + 4 * no + 6) * maxq
                + 4 * no * maxa) + Qm.SLACK
    drive = shim + base + work + sec.w_size() * isz + tables
    fixed0 = out + back + ndrives * drive + (xwork + slab) * isz + Qm.SLACK
    slot = ndrives * slab * isz
    heap = WORKER_HEAP

    def fixed(slots):
        return fixed0 + slots * slot

    def need(n, nxo, slots):
        if n == 1:                  # (Xo is formed before the work arrays of the (Q) object are there)
            return fixed(slots) + wq + ndrives * max(0, xo - work)
        return fixed(slots) + max(n * (wq + heap), nxo * (xo + heap))
    want = workers
    if want is None:
        want = 1 if comb(no, 4) * nA < SMALL else min(_cores(), max(1, comb(no, 4) // PER_WORKER))
    want = max(1, int(want))
    floor = need(1, 1, 3)
    if budget is None:
        n, nxo, slots = want, want, xfull
    else:
        if budget < floor:
            raise MemoryError('the quadruples of CC4 need %.0f MB besides T3 and the dressed integrals for %d '
                              'occupied and %d virtual spinors (R3 on the packed block: %.0f MB; Xo: %.0f MB per '
                              'drive); %.0f MB are left'
                              % (floor / 1e6, no, nv, nI3 * nA3 * isz / 1e6, nI3 * no * nQ * isz / 1e6, budget / 1e6))
        slots = int(min(xfull, 3 + (budget - floor) // slot))
        n = nxo = 1
        if slots == xfull and want > 1:
            rest = budget - fixed(slots)
            n = int(min(want, rest // (wq + heap)))
            nxo = int(min(n, rest // (xo + heap)))
            if n < 2 or nxo < 1:
                n = nxo = 1
    dense = Qm.explicit_plan(no, nv, isz, 1, 1, slots, 1)
    return Plan(n, nxo, slots, int(need(n, nxo, slots)), int(floor), int(fixed(slots)), int(wq), True, dense,
                ndrives, int(slab), int(n22), int(n13))


def driver_bytes(eris, dtype):
    '''Bytes that an iteration of CC4 holds around a pass (add_quadruples):
    the dressed blocks, T3 on the packed block, and what the blocks of the
    labels are put in their places with.'''
    from zccsdt.cc import ccsdt_opt, ccsdtq
    no = eris.nocc
    nv = np.asarray(eris.fock).shape[0] - no
    isz = np.dtype(dtype).itemsize
    eng = ccsdt_opt.engine_for(eris, dtype)
    return ccsdtq._dressed_bytes(no, nv, isz) + isz * (comb(no, 3) * comb(nv, 3) + eng.size + 2 * (no * nv) ** 2)


def plans(eris, dtype, opts):
    '''(the MemoryPlan of the CCSDT step, the Plan of the quadruples) of a
    CC4 run with the solver options ``opts`` (max_memory, incore_complete,
    workers; opts.plan and opts.quadruples: plans given explicitly).  With
    the memory for it the step has everything in memory and the pass all
    its processes; else the step is planned beside the floor of the pass
    (ccsdt_opt.plan_for: scratch files), and the pass gets what the step
    leaves.  MemoryError, with the numbers, if the floors do not fit.

    The two plans are summed as if the step and the pass were at their
    peaks together, which they are not: the pass runs when the step has
    given its work arrays back.  So the plan of the pass made here is the
    least that a pass gets (and the error before the first iteration);
    add_quadruples plans every pass again from what is resident then.'''
    from zccsdt import memory
    from zccsdt.cc import ccsdt_opt
    no = eris.nocc
    nv = np.asarray(eris.fock).shape[0] - no
    isz = np.dtype(dtype).itemsize
    lab, mod = _sector_labels(eris)
    workers = getattr(opts, 'workers', None)
    extra = driver_bytes(eris, dtype)
    given = getattr(opts, 'quadruples', None)
    if given is not None:
        return opts.plan or ccsdt_opt.plan_for(eris, dtype, opts, reserve=given.need + extra), given
    full = plan(no, nv, isz, None, lab, mod, workers)
    full.auto = True
    budget = None if opts.incore_complete else memory.budget(opts.max_memory)
    if budget is None:
        return opts.plan or ccsdt_opt.plan_for(eris, dtype, opts), full
    step = opts.plan
    if step is None:
        try:
            step = ccsdt_opt.plan_for(eris, dtype, opts, reserve=full.need + extra)
        except MemoryError:
            step = None
        if step is not None and not (step.t3_file or step.pp_file or step.pp_bytes is not None):
            return step, full
        try:
            step = ccsdt_opt.plan_for(eris, dtype, opts, reserve=full.floor + extra)
        except MemoryError as err:
            raise MemoryError('CC4: the pass over the quadruples takes at least %.0f MB, the dressed integrals '
                              'and T3 on the packed block %.0f MB of the budget of %.0f MB; beside them: %s'
                              % (full.floor / 1e6, extra / 1e6, budget / 1e6, err)) from None
    cached = ccsdt_opt.cached_bytes(eris)
    least = plan(no, nv, isz, budget + cached - step.need - extra, lab, mod, workers)
    least.auto = True
    return step, least


# ------------------------------------------------------------ processes over the quadruples

PHASE = [None]          # what the pass with processes is at: 'fork', 'xo', 'xv', 'quadruples' (for a sampler)


def _phase(name):
    PHASE[0] = name


def _shared(shape, dtype):
    '''An array of zeros that forked processes share (anonymous shared
    mapping).'''
    import mmap
    n = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if n == 0:
        return np.zeros(shape, dtype=dtype)
    return np.frombuffer(mmap.mmap(-1, n), dtype=dtype).reshape(shape)


def _single_thread():
    '''One thread for the OpenMP kernels and the BLAS of this process.'''
    from zccsdt.cc import t3kern
    for lib, name in ((P._lib, 'omp_set_num_threads'), (getattr(t3kern, '_lib', None), 'MKL_Set_Num_Threads')):
        try:
            getattr(lib, name)(ctypes.c_int(1))
        except AttributeError:
            pass


def _rollup(pid, keys):
    try:
        with open('/proc/%d/smaps_rollup' % pid) as f:
            return 1024 * sum(int(line.split()[1]) for line in f if line.startswith(keys))
    except (OSError, ValueError, IndexError):
        return 0


def private_bytes(pid):
    '''Private pages of a process (its own arrays and what it has written
    of the pages shared with its parent), from /proc/pid/smaps_rollup; 0 if
    unknown.'''
    return _rollup(pid, ('Private_Dirty', 'Private_Clean'))


def pss_bytes(pid):
    '''The proportional set size of a process: its private pages and its
    share of the pages it has in common with others (the arrays that the
    processes of a pass share, each page counted once over all of them);
    0 if unknown.'''
    return _rollup(pid, ('Pss:',))


class _Workers:
    '''Forked processes that serve tasks: ``serve(kind, payload, grab)`` is
    called in each of them for every map(); grab(n, chunk) hands out the
    ranges of range(n), each to one process.  The processes see the arrays
    of the caller as they were at the fork (pages shared until written) and
    its shared arrays (_shared) as they are.'''

    def __init__(self, n, serve):
        import gc
        import multiprocessing as mp
        import signal
        ctx = mp.get_context('fork')
        self.next, self.lock = ctx.RawValue('q', 0), ctx.Lock()
        self.pids, self.conns = [], []
        gc.freeze()             # (the collector of a child does not write to the objects of the parent)
        try:
            for w in range(n):
                here, there = ctx.Pipe()
                pid = os.fork()
                if pid == 0:
                    code = 1
                    try:
                        for c in self.conns + [here]:
                            c.close()
                        try:        # (not left behind if the parent is killed)
                            ctypes.CDLL(None).prctl(1, int(signal.SIGKILL))
                        except (OSError, AttributeError):
                            pass
                        _single_thread()
                        self._loop(there, serve)
                        code = 0
                    finally:
                        os._exit(code)
                there.close()
                self.pids.append(pid)
                self.conns.append(here)
        finally:
            gc.unfreeze()

    def _grab(self, n, chunk):
        while True:
            with self.lock:
                i = self.next.value
                self.next.value = i + chunk
            if i >= n:
                return
            yield i, min(n, i + chunk)

    def _loop(self, conn, serve):
        import traceback
        while True:
            try:
                msg = conn.recv()
            except EOFError:
                return
            if msg is None:
                return
            try:
                conn.send(('ok', serve(msg[0], msg[1], self._grab)))
            except BaseException:
                conn.send(('error', traceback.format_exc()))

    def map(self, kind, payload, n=None):
        '''serve(kind, payload, grab) in the first ``n`` processes (default:
        all); [its return values].'''
        from multiprocessing.connection import wait
        conns = self.conns[:n]
        self.next.value = 0
        for c in conns:
            c.send((kind, payload))
        out, errors, pending = {}, [], list(conns)
        while pending:
            ready = wait(pending, timeout=5.0)
            for c in ready:
                pending.remove(c)
                try:
                    tag, val = c.recv()
                except EOFError:
                    tag, val = 'error', 'the process ended'
                if tag == 'error':
                    errors.append(val)
                else:
                    out[id(c)] = val
            if not ready:
                for pid in list(self.pids):
                    if os.waitpid(pid, os.WNOHANG)[0]:
                        self.pids.remove(pid)
                        errors.append('process %d ended (killed for its memory?)' % pid)
            if errors:
                raise RuntimeError('CC4: a process of the pass over the quadruples failed:\n' + '\n'.join(errors))
        return [out[id(c)] for c in conns]

    def private(self):
        return sum(private_bytes(pid) for pid in self.pids)

    def close(self, kill=False):
        import signal
        for c in self.conns:
            try:
                if not kill:
                    c.send(None)
                c.close()
            except (OSError, ValueError):
                pass
        for pid in self.pids:
            try:
                if kill:
                    os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
            except (OSError, ChildProcessError):
                pass
        self.pids, self.conns = [], []


# ------------------------------------------------------------ the pass in sectors

class _Buffers:
    '''The work arrays of one process: the blocks of the products of an
    occupied quadruple, the row over the denominator in their places.'''

    def __init__(self, n22, n13, nv, dt, lean=True):
        if lean:        # (the two products of a kind summed in one array)
            self.B1, self.E22 = (np.empty(n22, dtype=dt) for _ in range(2))
            self.C1, self.E13 = (np.empty(n13, dtype=dt) for _ in range(2))
            self.B2, self.C2 = self.B1, self.C1
        else:
            self.B1, self.B2, self.B3, self.E22, self.Ew = (np.empty(n22, dtype=dt) for _ in range(5))
            self.C1, self.C2, self.E13, self.Fw, self.Fs = (np.empty(n13, dtype=dt) for _ in range(5))
        self.Y = np.zeros((nv, comb(nv, 2)), dtype=dt)
        self.row = np.zeros(comb(nv, 3), dtype=dt)
        self.ee = np.zeros(2)


class _SectorPass:
    '''What the occupied quadruples of a pass in sectors read: the (Q)
    objects of the drives, the sectors, the operators of the four sums.'''

    def __init__(self, sec, qs, back, e, omega, dt):
        from zccsdt.cc import quadruples_sym as S
        no = qs[0][0].nocc
        self.sec, self.qs, self.back, self.omega, self.dt = sec, qs, back, omega, np.dtype(dt)
        self.nv = qs[0][0].nvir
        self.quads = P.tuples(no, 4)
        self.Mq = sec.red(sec.ko[self.quads].sum(axis=1))
        self.kpair = sec.red(sec.ko[P.tuples(no, 2)].sum(axis=1))
        self.xl = P.tuples(no, 2)
        self.ev = np.ascontiguousarray(np.asarray(e, dtype=float)[no:], dtype=np.float64)
        self.kv32 = np.ascontiguousarray(sec.kv, dtype=np.int32)
        # the kernel of the row: zc_cc4_sector_row, or (a library built before it) the kernel of
        # the (Q) gradient with its third product zero, whose two-pair blocks come out conjugated
        self.row = getattr(P._lib, 'zc_cc4_sector_row_' + P._SFX[self.dt]) if _lean() else None
        self.fn = getattr(P._lib, 'zc_qg_sector_vectors_' + P._SFX[self.dt])
        self.types = (S._I64, S._INT, S._DBL)
        self.n22, self.n13 = sec.sizes(self.quads)

    def buffers(self):
        return _Buffers(self.n22, self.n13, self.nv, self.dt, self.row is not None)

    def rhs(self, b, I, XVs):
        '''The T4 row of the occupied quadruple I over its denominator, in
        the places of the blocks: b.E22, b.E13.'''
        sec, q0 = self.sec, self.qs[0][0]
        I64, INT, DBL = self.types
        p_ = P._p
        M = int(self.Mq[I])
        b22, n2, b13, n1, off22, nc22, off13, nc13 = sec.layout(M)
        head = (INT(self.nv), p_(self.kv32), INT(sec.modulus), INT(M), p_(self.ev),
                DBL(float(q0.eo4[I]) + self.omega),
                INT(sec.kmin), INT(sec.nk), p_(sec.lid_of), p_(sec.lstart), p_(sec.lmem),
                INT(sec.qmin), INT(sec.nq), p_(off22), p_(nc22), p_(off13), p_(nc13),
                p_(sec.posV), p_(sec.posQ), p_(sec.pos3))
        if self.row is not None:
            for B in (b.B1[:n2], b.E22[:n2], b.C1[:n1], b.E13[:n1]):
                B[...] = 0
            for (q, w), XV in zip(self.qs, XVs):
                _sector_blocks(sec, q, I, M, XV, self.kpair, self.xl, b.B1, b.B1, b.C1, b.C1, w)
            self.row(*head, p_(b.B1), p_(b.C1), p_(b.E22), p_(b.E13))
            return M
        for B in (b.B1[:n2], b.B2[:n2], b.B3[:n2], b.E22[:n2], b.C1[:n1], b.C2[:n1], b.E13[:n1]):
            B[...] = 0
        for (q, w), XV in zip(self.qs, XVs):
            _sector_blocks(sec, q, I, M, XV, self.kpair, self.xl, b.B1, b.B2, b.C1, b.C2, w)
        self.fn(*head, p_(b.B1), p_(b.B2), p_(b.B3), p_(b.C1), p_(b.C2),
                p_(b.E22), p_(b.Ew), p_(b.Fw), p_(b.E13), p_(b.Fs), p_(b.ee))
        if self.dt.kind == 'c':
            np.conjugate(b.E22[:n2], out=b.E22[:n2])
        return M

    def sums(self, b, I, M, R2, R3, locks=None):
        self.back(self.sec, I, M, b.E22, b.E13, R2, R3, b.Y, b.row, locks)


def run_sectors(ints, e, labels, modulus, t2, T3, omega=0.0, drives=None, back_ints=None, plan_=None,
                workers=None, max_memory=None, stats=None):
    '''run() in the symmetry sectors of ``labels`` (one per active spinor,
    occupied then virtual; ``modulus`` as eris.modulus): the products of
    every occupied quadruple in the blocks of its total label
    (zccsdt.cc.quadruples_sym), the T4 row expanded into the places of the
    blocks by zc_qg_sector_vectors (csrc/tnpack.c) and the four sums block
    by block.  The same R2, R3 as run().

    ``plan_``: the Plan (default: plan() for ``max_memory`` MB of the whole
    process - what is resident at the call is not the pass's - and
    ``workers`` processes).  With more than one process the occupied
    quadruples are shared out among forked processes, each with one thread:
    they read the operands of the caller, R2, R3, Xo and Xv are shared
    arrays; a row of R3 is added to under the lock of the class of its
    virtual triples.  ``stats``: a dict that receives what the pass
    measured (processes, their private memory, the shared arrays).'''
    from zccsdt import memory
    from zccsdt.cc import quadruples_sym as S
    no, nv = ints.nocc, ints.nvir
    e = np.asarray(e, dtype=float)
    drives = drives or [(ints, t2, T3, 1.0)]
    dt = np.dtype(np.result_type(*[np.result_type(d[1], d[2], d[0].oovv) for d in drives]))
    nP, nQ, nI3, nA3 = comb(no, 2), comb(nv, 2), comb(no, 3), comb(nv, 3)
    if no < 4 or nv < 4:
        return np.zeros((nP, nQ), dtype=dt), np.zeros((nI3, nA3), dtype=dt)
    pl = plan_ or plan(no, nv, dt.itemsize, memory.budget(max_memory), labels, modulus, workers, len(drives))
    par = pl.workers > 1
    new = _shared if par else (lambda shape, dtype: np.zeros(shape, dtype=dtype))
    R2, R3 = new((nP, nQ), dt), new((nI3, nA3), dt)
    sec = _sectors(no, labels, modulus)
    qs, secs, shims = [], [], []
    with timer('cc4.setup'):
        for d_ints, d2, d3, w in drives:
            sh = _Shim(d_ints, e)
            sh.labels, sh.modulus = np.asarray(labels), modulus
            xo = _shared((nI3, no, nQ), dt) if par else None
            qs.append((Qm.Quadruples(sh, np.asarray(d2, dtype=dt), np.asarray(d3, dtype=dt), pl.dense, xo=xo), w))
            shims.append(sh)
            # (one per drive: the blocks of <bc||ef> that a Sectors keeps are those of its integrals)
            secs.append(S.Sectors(sh))
        ps = _SectorPass(sec, qs, BackBlocks(back_ints or ints, dt, sec), e, omega, dt)
    q0 = qs[0][0]
    schedule = q0.schedule()
    if not par:
        b = ps.buffers()
        XVs, last = [{} for _ in qs], None
        with timer('cc4.pass'):
            for l, xs, Is in schedule:
                with timer('cc4.xv'):
                    if last != l:
                        XVs, last = [{} for _ in qs], l
                    for (q, _), XV, sd in zip(qs, XVs, secs):
                        for x in [x for x in XV if x not in xs]:
                            del XV[x]
                        for x in xs:
                            if int(x) not in XV:
                                XV[int(x)] = S.xv_blocks(sd, q, int(x), l)
                for I in Is:
                    with timer('cc4.rhs'):
                        M = ps.rhs(b, int(I), XVs)
                    with timer('cc4.back'):
                        ps.sums(b, int(I), M, R2, R3)
        if stats is not None:
            stats.update(workers=1, private=0, shared=0)
        return R2, R3

    # ---- several processes
    import time
    import multiprocessing as mp
    ctx = mp.get_context('fork')
    locks = (ctx.Lock(), {k: ctx.Lock() for k in sec.A3})
    nslot = max([3, min(pl.xv, no - 1)] + [len(xs) for _, xs, _ in schedule])
    slab = max(1, sec.xv_size()[0])
    arenas = [_shared((nslot * slab,), dt) for _ in qs]
    state = {}

    def serve(kind, payload, grab):
        c0 = time.process_time()
        if kind == 'xo':            # Xo[:, m] of every drive
            for m0, m1 in grab(no, max(1, no // (2 * max(1, pl.xo_workers)))):
                for (q, _), sh in zip(qs, shims):
                    q.form_xo(sh, range(m0, m1))
            return time.process_time() - c0
        Is, metas = payload
        b = state.get('b')
        if b is None:
            b = state['b'] = ps.buffers()
        XVs = [{x: {ke: a[o:o + nr * nc].reshape(nr, nc) for ke, (o, nr, nc) in blocks.items()}
                for x, blocks in meta.items()} for a, meta in zip(arenas, metas)]
        t_rhs = t_back = 0.0
        for i0, i1 in grab(len(Is), payload_chunk(len(Is))):
            for I in Is[i0:i1]:
                t0 = time.perf_counter()
                M = ps.rhs(b, int(I), XVs)
                t1 = time.perf_counter()
                ps.sums(b, int(I), M, R2, R3, locks)
                t_rhs += t1 - t0
                t_back += time.perf_counter() - t1
        return t_rhs, t_back, time.process_time() - c0

    def payload_chunk(n):
        return 1

    # (cc4.pass: its CPU time has that of the processes, which have ended when it closes)
    with timer('cc4.pass'):
        _phase('fork')
        t0 = time.perf_counter()
        pool = _Workers(pl.workers, serve)
        ok = False
        try:
            timing.add('cc4.fork', time.perf_counter() - t0)
            _phase('xo')
            t0 = time.perf_counter()
            cpu = sum(pool.map('xo', None, max(1, min(pl.xo_workers, pl.workers, no))))
            timing.add('cc4.xo', time.perf_counter() - t0, cpu)
            where, metas, last = [{} for _ in qs], [{} for _ in qs], None
            tr = tb = cpu = wall = 0.0
            for l, xs, Is in schedule:
                _phase('xv')
                with timer('cc4.xv'):
                    if last != l:
                        where, metas, last = [{} for _ in qs], [{} for _ in qs], l
                    need = [int(x) for x in xs]
                    for (q, _), sd, a, wh, meta in zip(qs, secs, arenas, where, metas):
                        for x in [x for x in wh if x not in need]:
                            del wh[x], meta[x]
                        free = [n for n in range(nslot) if n not in wh.values()]
                        for x in need:
                            if x in wh:
                                continue
                            wh[x] = free.pop()
                            o, meta[x] = wh[x] * slab, {}
                            for ke, blk in S.xv_blocks(sd, q, x, l).items():
                                a[o:o + blk.size] = blk.ravel()
                                meta[x][ke] = (o, blk.shape[0], blk.shape[1])
                                o += blk.size
                            assert o <= (wh[x] + 1) * slab
                _phase('quadruples')
                t0 = time.perf_counter()
                res = pool.map('quads', (np.asarray(Is), metas), max(1, min(pl.workers, len(Is))))
                wall += time.perf_counter() - t0
                tr, tb, cpu = tr + sum(r[0] for r in res), tb + sum(r[1] for r in res), cpu + sum(r[2] for r in res)
            # the wall time of the processes shared out by what they spent in the two parts
            tot = max(tr + tb, 1e-30)
            timing.add('cc4.rhs', wall * tr / tot, cpu * tr / tot)
            timing.add('cc4.back', wall * tb / tot, cpu * tb / tot)
            if stats is not None:
                stats.update(workers=pl.workers, private=pool.private(),
                             shared=R2.nbytes + R3.nbytes + sum(a.nbytes for a in arenas)
                             + sum(q.Xo.nbytes for q, _ in qs))
            ok = True
        finally:
            _phase(None)
            pool.close(kill=not ok)
    return R2, R3


def run_for(eris, ints, e, t2, T3, **kw):
    """run() or, where the integrals carry symmetry labels that the (Q)
    sector code takes (zccsdt.cc.quadruples_sym.available), run_sectors()."""
    labels, modulus = _sector_labels(eris)
    if labels is not None:
        return run_sectors(ints, e, labels, modulus, t2, T3, **kw)
    kw.pop('workers', None)
    kw.pop('stats', None)
    return run(ints, e, t2, T3, **kw)


# ------------------------------------------------------------ the iteration

def _blocks(eris, dtype):
    """(the layout of the blocks of T3 of these integrals, the lexicographic
    numbers of its occupied and of its virtual triples per block), or None
    where T3 is not in plain blocks of labels (the packed block itself, or
    blocks related by a spin flip)."""
    from zccsdt.cc import ccsdt_opt
    om, vm = t3pack.maps_for(eris)
    if om.spins is None:
        return None
    L = ccsdt_opt.engine_for(eris, dtype).L
    if L.flip:
        return None
    lo, lv = ccsdt_opt._lex(om), ccsdt_opt._lex(vm)
    return L, [(g, lo[om.tri_off[g]:om.tri_off[g + 1]], lv[vm.tri_off[g]:vm.tri_off[g + 1]]) for g in L.stored]


def add_quadruples(eris, t1, t2, t3, t2n, t3n, plan_=None, level_shift=0.0, stats=None, grow=None):
    '''The quadruples of CC4 in a Jacobi step of CCSDT: the new T2
    ``t2n`` and T3 ``t3n`` (the layout of ``t3``: the blocks of the labels
    of ``eris`` or the packed block; changed in place) of
    ccsdt_opt.update_amps at (t1, t2, t3) plus the sums of T4 over the
    denominators.  Returns the new T2.  ``plan_``: the Plan of the pass.
    ``grow``: (max_memory, bytes that the run still allocates outside the
    pass - DIIS vectors in memory -, workers) for a plan of plans(): the
    pass is planned again from what is resident now (the dressed blocks
    and T3 on the packed block are there, the work arrays of the CCSDT
    step are not), and that plan is taken where it has more processes;
    stats['plan']: the plan used.'''
    from zccsdt import memory
    from zccsdt.cc import ccsdt_opt, t4engine
    no = eris.nocc
    nv = np.asarray(eris.fock).shape[0] - no
    e = np.asarray(eris.fock).diagonal().real
    eo, ev = e[:no], e[no:] + level_shift
    with timer('cc4.dress'):
        ints = t4engine.Dressed.from_eris(eris, t1)
    with timer('cc4.pack'):
        dtype = np.result_type(t2, t3, ints.oovv)
        blocks = _blocks(eris, dtype)
        if blocks is None:
            T3p = np.ascontiguousarray(ccsdt_opt.packed_t3(eris, np.asarray(t3)), dtype=dtype)
        else:                       # every block to its places, one after the other
            T3p = np.zeros((comb(no, 3), comb(nv, 3)), dtype=dtype)
            for g, I, A in blocks[1]:
                if len(I) and len(A):
                    T3p[I[:, None], A[None, :]] = blocks[0].block(t3, g)
    if grow is not None and plan_ is not None and plan_.auto:
        budget = memory.budget(grow[0])
        if budget is not None:
            lab, mod = _sector_labels(eris)
            try:
                again = plan(no, nv, np.dtype(dtype).itemsize, budget - grow[1], lab, mod, grow[2])
            except MemoryError:     # (less than the plan of plans() counted on: that one stands)
                again = None
            if again is not None and (again.workers, again.xo_workers) > (plan_.workers, plan_.xo_workers):
                plan_ = again
    if stats is not None:
        stats['plan'] = plan_
    r2, r3 = run_for(eris, ints, e, t2, T3p, plan_=plan_, stats=stats)
    del ints, T3p
    with timer('cc4.update'):
        d2 = eo[P.tuples(no, 2)].sum(axis=1)[:, None] - ev[P.tuples(nv, 2)].sum(axis=1)[None, :]
        t2n = t2n + P.unpack(r2 / d2, no, nv, 2)
        eo3, ev3 = eo[P.tuples(no, 3)].sum(axis=1), P.tuple_sums(ev, 3)
        if blocks is None:
            for j0 in range(0, len(eo3), 64):       # (no array of the size of T3 for the denominators)
                r3[j0:j0 + 64] /= eo3[j0:j0 + 64, None] - ev3[None, :]
            blk = ccsdt_opt.blocked_t3(eris, r3)
            del r3
            t3n += blk.reshape(np.shape(t3n))
        else:                       # (what is outside the blocks vanishes by symmetry)
            for g, I, A in blocks[1]:
                if len(I) and len(A):
                    blk = r3[I[:, None], A[None, :]]
                    blk /= eo3[I][:, None] - ev3[A][None, :]
                    blocks[0].block(t3n, g)[...] += blk
    return t2n
