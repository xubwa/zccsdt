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
from math import comb

import numpy as np

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


def run(ints, e, t2, T3, omega=0.0, drives=None, back_ints=None, plan_=None, rows=None):
    '''All occupied quadruples: the T4 rows over the denominator
    e_ijkl - e_abcd + omega, contracted by Back.  ``drives``: a list of
    (ints, t2, T3, weight) whose right-hand sides are summed (default
    [(ints, t2, T3, 1)]); ``back_ints``: the operators of the sums (default
    ints).  Returns (R2 [C(o,2), C(v,2)], R3 [C(o,3), C(v,3)]).  ``rows``:
    a list that receives (I, row) (tests).'''
    no, nv = ints.nocc, ints.nvir
    e = np.asarray(e, dtype=float)
    drives = drives or [(ints, t2, T3, 1.0)]
    dt = np.result_type(*[np.result_type(d[1], d[2], d[0].oovv) for d in drives])
    R2 = np.zeros((comb(no, 2), comb(nv, 2)), dtype=dt)
    R3 = np.zeros((comb(no, 3), comb(nv, 3)), dtype=dt)
    if no < 4 or nv < 4:
        return R2, R3
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


def sigma(g, ints, e, r1, r2p, R3, omega):
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
    diagonal Fock matrix.  Returns (s1, s2p, s3) as eom_opt.sigma.'''
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
                     drives=[(ints, t2 + r2, T3 + R3m, 0.5), (ints, t2 - r2, T3 - R3m, -0.5), (dints, t2, T3, 1.0)])
    with timer('cc4.t4'):
        b2, b3 = run_for(g.eris, ints, e, t2, T3, back_ints=dints)
    s1 = s1
    s2 = s2 + (a2 + b2).reshape(s2.shape)
    s3 = s3 + (a3 + b3).reshape(s3.shape)
    return s1, s2, s3


def eom(eris, t1, t2, T3, nroots=3, conv_tol=1e-6, omega_tol=None, guess_tol=1e-4, max_cycle=60, verbose=False,
        extra=2):
    '''Excitation energies of CC4: the folded eigenproblem in the space of
    singles, doubles and triples (R4 folded in at the frequency omega), each
    root self-consistent in omega (ccn.folded_roots), started from the roots
    of the singles-doubles block.  The amplitudes of the semicanonical
    integrals ``eris`` (ccn.cc4_kernel: Result.eris).  Packed layout; where
    the integrals carry labels of a linear molecule the quadruples are made
    in the sectors of the (Q) code and the states are the totally symmetric
    ones (label shift 0).
    Returns (energies, the Solved of folded_roots).'''
    from zccsdt.cc import ccn, eom as E, eom_opt, t4engine
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
            s1, s2, s3 = sigma(g, ints, e, x[0].reshape(no, nv), x[1], x[2], omega)
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


class BackSectors(Back):
    '''Back for the blocks of one occupied quadruple: E22 on two pairs and
    E13 on an index and a triple (the T4 row expanded into the places of
    the blocks of the sectors).'''

    def __call__(self, sec, I, M, E22, E13, R2, R3):
        nv, no = self.nv, self.no
        so22, so13, so12 = self.so22, self.so13, self.so12
        b22, _, b13, _, off22, nc22, off13, nc13 = sec.layout(M)
        sfx = P._SFX[self.dtype]
        radd = getattr(P._lib, 'zc_rows_cols_add_' + sfx)
        p_, I64 = P._p, ctypes.c_int64
        ko = sec.ko
        lab_pair = sec.red(ko[P.tuples(no, 2)].sum(axis=1))
        blk22 = {qq: E22[off:off + nr * nc].reshape(nr, nc) for qq, off, nr, nc in b22}
        blk13 = {pp: E13[off:off + nr * nc].reshape(nr, nc) for pp, off, nr, nc in b13}

        def add(R, rows, sg, cols, X):
            rows = np.ascontiguousarray(rows, dtype=np.int64)
            sg = np.ascontiguousarray(sg, dtype=np.float64)
            cols = np.ascontiguousarray(cols, dtype=np.int64)
            X = np.ascontiguousarray(X, dtype=self.dtype)
            radd(p_(R), I64(R.shape[1]), p_(rows), p_(sg), I64(len(rows)), p_(cols), I64(len(cols)), p_(X))
        Pp, Pq, sg22 = so22.idx1[I], so22.idx2[I], so22.sgn[I]
        with timer('cc4.back.gemm'):
            for s in range(6):          # R2: the block of the label of P
                qq = int(lab_pair[Pp[s]])
                if qq not in blk22:
                    continue
                cq = sec.Q2[int(sec.red(M - qq))]
                y = blk22[qq] @ self.oovv_pp[Pq[s]][cq]
                add(R2, [Pp[s]], [float(sg22[s])], sec.Q2[qq], y[None, :])
            for s in range(4):          # R3: f~ and <cm||ef>~
                m, J, g = int(so13.idx1[I, s]), int(so13.idx2[I, s]), float(so13.sgn[I, s])
                pm = int(ko[m])
                if pm in blk13:
                    add(R3, [J], [g], sec.A3[int(sec.red(M - pm))], (self.fov[m][sec.V[pm]] @ blk13[pm])[None, :])
                Y = self.Y
                Y[...] = 0
                for qq, X in blk22.items():
                    lc = int(sec.red(M - qq - pm))
                    if lc not in sec.V:
                        continue
                    cq = sec.Q2[int(sec.red(M - qq))]
                    Wc = np.ascontiguousarray(self.vovv_p[sec.V[lc], m][:, cq])
                    Y[np.ix_(sec.V[lc], sec.Q2[qq])] = Wc @ X.T
                P.tn_reduce(R3[J], Y, nv, 1, 2, scale=-g)
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
                    add(R3, so12.pos[ks[ok], Pp[s]], -float(sg22[s]) * psg[ok], sec.A3[int(sec.red(M - pp))], Yk)


def run_sectors(ints, e, labels, modulus, t2, T3, omega=0.0, drives=None, back_ints=None):
    '''run() in the symmetry sectors of ``labels`` (one per active spinor,
    occupied then virtual; ``modulus`` as eris.modulus): the products of
    every occupied quadruple in the blocks of its total label
    (zccsdt.cc.quadruples_sym), the T4 row expanded into the places of the
    blocks by zc_qg_sector_vectors (csrc/tnpack.c) and the four sums block
    by block.  The same R2, R3 as run().'''
    from zccsdt.cc import quadruples_sym as S
    no, nv = ints.nocc, ints.nvir
    e = np.asarray(e, dtype=float)
    drives = drives or [(ints, t2, T3, 1.0)]
    dt = np.result_type(*[np.result_type(d[1], d[2], d[0].oovv) for d in drives])
    R2 = np.zeros((comb(no, 2), comb(nv, 2)), dtype=dt)
    R3 = np.zeros((comb(no, 3), comb(nv, 3)), dtype=dt)
    if no < 4 or nv < 4:
        return R2, R3
    qs, secs = [], []
    with timer('cc4.setup'):
        for d_ints, d2, d3, w in drives:
            sh = _Shim(d_ints, e)
            sh.labels, sh.modulus = np.asarray(labels), modulus
            qs.append((Qm.Quadruples(sh, np.asarray(d2, dtype=dt), np.asarray(d3, dtype=dt)), w))
            # (one per drive: the blocks of <bc||ef> that a Sectors keeps are those of its integrals)
            secs.append(S.Sectors(sh))
        sec = secs[0]
    back = BackSectors(back_ints or ints, dt)
    n22, n13 = sec.sizes(P.tuples(no, 4))
    B1, B2, B3, E22, Ew = (np.empty(n22, dtype=dt) for _ in range(5))
    C1, C2, E13, Fw, Fs = (np.empty(n13, dtype=dt) for _ in range(5))
    q0 = qs[0][0]
    quads = P.tuples(no, 4)
    kpair = sec.red(sec.ko[P.tuples(no, 2)].sum(axis=1))
    xl = P.tuples(no, 2)
    ev = np.ascontiguousarray(e[no:], dtype=np.float64)
    fn = getattr(P._lib, 'zc_qg_sector_vectors_' + P._SFX[np.dtype(dt)])
    p_, I64, INT, DBL = P._p, S._I64, S._INT, S._DBL
    kv32 = np.ascontiguousarray(sec.kv, dtype=np.int32)
    ee = np.zeros(2)
    XVs, last = [{} for _ in qs], None
    for l, xs, Is in q0.schedule():
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
            M = int(sec.red(sec.ko[quads[I]].sum()))
            b22, _, b13, _, off22, nc22, off13, nc13 = sec.layout(M)
            n2 = sum(nr * nc for _, _, nr, nc in b22)
            n1 = sum(nr * nc for _, _, nr, nc in b13)
            with timer('cc4.rhs'):
                for B in (B1[:n2], B2[:n2], B3[:n2], E22[:n2], C1[:n1], C2[:n1], E13[:n1]):
                    B[...] = 0
                for (q, w), XV in zip(qs, XVs):
                    _sector_blocks(sec, q, I, M, XV, kpair, xl, B1, B2, C1, C2, w)
                # the row over the denominator, written to the places of the blocks (E22 conjugated)
                fn(INT(nv), p_(kv32), INT(sec.modulus), INT(M), p_(ev), DBL(float(q0.eo4[I]) + omega),
                   INT(sec.kmin), INT(sec.nk), p_(sec.lid_of), p_(sec.lstart), p_(sec.lmem),
                   INT(sec.qmin), INT(sec.nq), p_(off22), p_(nc22), p_(off13), p_(nc13),
                   p_(sec.posV), p_(sec.posQ), p_(sec.pos3), p_(B1), p_(B2), p_(B3), p_(C1), p_(C2),
                   p_(E22), p_(Ew), p_(Fw), p_(E13), p_(Fs), p_(ee))
                if np.iscomplexobj(E22):
                    np.conjugate(E22[:n2], out=E22[:n2])
            back(sec, I, M, E22, E13, R2, R3)
    return R2, R3


def run_for(eris, ints, e, t2, T3, **kw):
    """run() or, where the integrals carry symmetry labels that the (Q)
    sector code takes (zccsdt.cc.quadruples_sym.available), run_sectors()."""
    from zccsdt.cc import quadruples_sym as S
    if getattr(eris, 'labels', None) is not None and S.available(eris) and not getattr(eris, 'no_sectors', False):
        return run_sectors(ints, e, eris.labels, int(getattr(eris, 'modulus', 0) or 0), t2, T3, **kw)
    return run(ints, e, t2, T3, **kw)
