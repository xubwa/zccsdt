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
        a2, a3 = run(ints, e, t2, T3, omega=omega, back_ints=ints,
                     drives=[(ints, t2 + r2, T3 + R3m, 0.5), (ints, t2 - r2, T3 - R3m, -0.5), (dints, t2, T3, 1.0)])
    with timer('cc4.t4'):
        b2, b3 = run(ints, e, t2, T3, back_ints=dints)
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
    integrals ``eris`` (ccn.cc4_kernel: Result.eris).  Packed layout.
    Returns (energies, the Solved of folded_roots).'''
    from zccsdt.cc import ccn, eom as E, eom_opt, t4engine
    g = eom_opt.prepare(eris, t1, t2, T3)
    ints = t4engine.Dressed.from_eris(eris, g.t1)
    e = np.asarray(eris.fock).diagonal().real
    sizes = g.sizes()
    no, nv = g.om.n, g.vm.n
    store = E.Store(sizes, g.dtype)

    def apply_at(omega):
        def apply(x, y):
            if omega is None:
                s1, s2, _ = eom_opt.sigma(g, x[0].reshape(no, nv), x[1], None, sd=True)
                y[0][:], y[1][:] = s1.ravel(), s2.ravel()
                y[2][:] = 0
                return
            s1, s2, s3 = sigma(g, ints, e, x[0].reshape(no, nv), x[1], x[2], omega)
            y[0][:], y[1][:], y[2][:] = s1.ravel(), s2.ravel(), s3.ravel()
        return apply
    precond = E.make_precond(g, True)
    gs = []
    for k in E.singles_guess(g, nroots + extra, E.symmetric_singles(eris, 0)):
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
