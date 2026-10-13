'''EOM-DEA: states with two electrons more than the reference determinant
(double electron attachment), through two ghost spinors (docs/eom.md,
section 11).

The reference gets two more occupied spinors g1, g2 that no integral
touches and whose Fock elements are zero (eom.ghost twice).  H does not
change their occupation, so the EE operators that empty both,

    r[g1 g2, ab]          = r[(a<b)]                    2p
    r[g1 g2 i, abc]       = r[i, (a<b<c)]               3p1h
    r[g1 g2 ij, abcd]     = r[(i<j), (a<b<c<d)]         4p2h,

are mapped to themselves by the EE Jacobian, which among them is the DEA
matrix <mu| [Hbar, R] |0>; its eigenvalues are E(N+2) - E_CC(N).

**One residual per sigma vector.**  Two operators R of this sector have
the product zero (each empties g1 and g2), T has no ghost index, and no
term of the amplitude equations without R has one.  So the residual of the
amplitude equations at T + R is, on the rows with both ghosts, exactly
linear in R:

    sigma_mu(R) = Res_mu(T + R),        mu = (g1 g2 ..; ..),

one evaluation of the residual code of the ground state - the CCSDT step
(ccsdt_opt.update_amps), the T4 equation and the T4 terms of CCSDTQ
(t4engine.residual) - on the padded spinors, with no new algebra.  The
vectors are stored short (the rows with both ghosts: the first C(o, n - 2)
rows of the packed rank-n array); the residual code works on the padded
arrays: its cost is that of an iteration of the ground state with o + 2
occupied spinors, not that of the sector.

Levels (``rank``: the highest EE rank of R; the T of the reference):

    rank 3, row3 'full'   R = 2p + 3p1h; T = T1 + T2 (CCSD; DEA-EOM-CCSD) or the T of
                          CCSDT (t3 given): P Hbar P in that space
    rank 3, row3 'cc3'    the 3p1h row to lowest order, <3p1h| [V~, R2] + [F, R3] |0>:
                          the Jacobian of CC3 in the sector (T of CC3)
    rank 4, row4 'full'   R = 2p + 3p1h + 4p2h; T to T3 (DEA-EOM-CCSDT), or to T4 (t4
                          given: the T of CCSDTQ or of CC4)
    rank 4, row4 'cc4'    the 4p2h row to lowest order, <4p2h| [V~, R3] + [[V~, R2], T2]
                          + [F, R4] |0>: the Jacobian of CC4 in the sector (T of CC4)

States are selected by masks: ``labels`` (one integer per spinor that adds,
2 m_j) with ``target`` (the sum of the labels of the two electrons, 2 Omega
of the state for a closed-shell reference) and ``parity`` (0 / 1 per spinor)
with ``target_parity``; every sigma vector is projected.

Dense layout, in memory, canonical (for 'cc3' / 'cc4': semicanonical)
spinors or not for the full rows.
'''

from dataclasses import dataclass
from math import comb
import time

import numpy as np

from zccsdt.cc import ccsdt_opt, eom, solver, t4engine, t4pack as P


@dataclass
class Result:
    e: np.ndarray               # E(N+2) - E_CC(N) per root / Hartree
    converged: np.ndarray
    residuals: np.ndarray
    vectors: list               # per root: [r2[(a<b)], r3[i, (abc)] flat, r4[(ij), (abcd)] flat]
    weights: np.ndarray         # [root, part]: norms^2
    nsigma: int = 0
    niter: int = 0
    sizes: tuple = ()           # elements of the parts that the masks leave
    timings: dict = None


class Problem:
    '''The padded integrals and amplitudes, the masks and the sigma vector
    of one level (module docstring).'''

    def __init__(self, eris, t1, t2, t3=None, t4=None, rank=3, row3='full', row4='full',
                 labels=None, target=0, parity=None, target_parity=0, mem_mb=2000.0):
        if rank not in (3, 4):
            raise ValueError('rank: 3 (R to 3p1h) or 4 (to 4p2h)')
        if rank == 4 and row3 != 'full':
            raise ValueError("row3 = 'cc3': with rank 3")
        if getattr(eris, 'spin_occ', None) is not None or getattr(eris, 'use_labels', False):
            raise ValueError('EOM-DEA: integrals on the packed layout (no spin blocks, no label blocks)')
        self.rank, self.row3, self.row4 = rank, row3, row4
        no, n = eris.nocc, np.asarray(eris.fock).shape[0]
        nv = n - no
        self.no, self.nv = no, nv
        dtype = np.result_type(t1, t2, eris.oovv, eris.fock, *([t3] if t3 is not None else []))
        self.dtype = dtype = np.dtype(dtype)
        nI3, nA3, nA4 = comb(no, 3), comb(nv, 3), comb(nv, 4)
        if t3 is None:
            t3 = np.zeros(nI3 * nA3, dtype=dtype)
        e1, t1a, t2a, t3a, _ = eom.ghost(eris, t1, t2, np.asarray(t3).reshape(nI3, nA3), 'ea')
        e2, t1p, t2p, t3p, _ = eom.ghost(e1, t1a, t2a, t3a, 'ea')
        # the orbital energies of the ghosts enter the Jacobi steps of the residual code only,
        # which are undone here: far below everything, so that no denominator is small
        ev = np.asarray(eris.mo_energy).real
        self.eg = eg = -(4.0 * float(abs(ev).max()) + 10.0)
        e2.mo_energy = np.concatenate(([eg, eg], ev))
        self.eris = e2
        self.t1 = np.ascontiguousarray(t1p, dtype=dtype)
        self.t2 = np.ascontiguousarray(t2p, dtype=dtype)
        self.t3 = np.ascontiguousarray(np.asarray(t3p, dtype=dtype).reshape(comb(no + 2, 3), nA3))
        self.t4 = None
        if rank == 4:
            self.t4 = np.zeros((comb(no + 2, 4), nA4), dtype=dtype)
            if t4 is not None:
                self.t4[comb(no + 2, 4) - comb(no, 4):] = np.asarray(t4).reshape(comb(no, 4), nA4)
            self.ints = t4engine.Dressed.from_eris(e2, self.t1)
        elif t4 is not None:
            raise ValueError('T4 with rank 3: not a level of this code')
        self.mem_mb = mem_mb
        self.step = solver.Options(mem_mb=mem_mb)
        self.plan = ccsdt_opt.plan_for(e2, dtype, solver.Options(mem_mb=mem_mb, incore_complete=True))
        # the rows of the sector and their denominators (real spinors: e_a + .. - e_i - ..)
        eo, evv = ev[:no], ev[no:]
        f = np.asarray(eris.fock).diagonal().real
        self.shapes = [(1, comb(nv, 2)), (no, nA3)] + ([(comb(no, 2), nA4)] if rank == 4 else [])
        self.sizes = [a * b for a, b in self.shapes]
        osum = [np.zeros(1), eo.copy()] + ([eo[P.tuples(no, 2)].sum(axis=1)] if rank == 4 else [])
        fsum = [np.zeros(1), f[:no].copy()] + ([f[:no][P.tuples(no, 2)].sum(axis=1)] if rank == 4 else [])
        self.den = [(P.tuple_sums(evv, k + 2)[None, :] - osum[k][:, None]) for k in range(len(self.shapes))]
        # (the Fock diagonal: [F, R_n] of the approximate rows)
        self.fden = [(P.tuple_sums(f[no:], k + 2)[None, :] - fsum[k][:, None]) for k in range(len(self.shapes))]
        # the Jacobi denominators of the residual code on the padded spinors (e_i + .. - e_a - ..)
        self.jac = [2 * eg - self.den[0], 2 * eg - self.den[1]]
        self.masks = self._masks(labels, target, parity, target_parity)
        self.nsigma = 0
        self.seconds = 0.0

    def _masks(self, labels, target, parity, target_parity):
        no, nv = self.no, self.nv
        masks = [np.ones(s, bool) for s in self.shapes]
        for lab, tgt, mod in ((labels, target, 0), (parity, target_parity, 2)):
            if lab is None:
                continue
            lab = np.asarray(lab, dtype=np.int64)
            if len(lab) != no + nv:
                raise ValueError('labels: one per spinor, occupied then virtual')
            lo, lv = lab[:no], lab[no:]
            for k in range(len(self.shapes)):
                vs = lv[P.tuples(nv, k + 2)].sum(axis=1)
                os_ = np.zeros(1, dtype=np.int64) if k == 0 else (lo if k == 1 else lo[P.tuples(no, 2)].sum(axis=1))
                d = vs[None, :] - os_[:, None] - tgt
                masks[k] &= (d % mod == 0) if mod else (d == 0)
        return masks

    def project(self, x):
        for p, m in zip(x, self.masks):
            p *= m.ravel()
        return x

    def new(self):
        return [np.zeros(s, dtype=self.dtype) for s in self.sizes]

    def _embed(self, x):
        no, nv = self.no, self.nv
        X2 = self.t2.copy()
        r2 = P.unpack(np.asarray(x[0]).reshape(1, -1), 2, nv, 2)[0, 1]      # [a, b]
        X2[0, 1] += r2
        X2[1, 0] -= r2
        X3 = self.t3.copy()
        X3[:no] += np.asarray(x[1]).reshape(self.shapes[1])
        X4 = None
        if self.rank == 4:
            X4 = self.t4.copy()
            X4[:comb(no, 2)] += np.asarray(x[2]).reshape(self.shapes[2])
        return X2, X3, X4

    # (for the 5p3h row of zccsdt.cc.dea5: the padded amplitudes on packed arrays, the places of
    #  the ghosts and of the spinors of eris, sigma parts given on [i, (abc)] and [(ij), (abcd)])
    @property
    def ghosts(self):
        return (0, 1)

    @property
    def places(self):
        return 2 + np.arange(self.no)

    def padded(self, x):
        return self._embed(x)

    def add_dense(self, y, S3, S4):
        y[1] += S3.ravel()
        y[2] += S4.ravel()
        return y

    def sigma(self, x, y):
        '''y <- the DEA matrix on x (lists of flat parts).'''
        t0 = time.perf_counter()
        no, nv = self.no, self.nv
        a2, b2 = P.tuples(nv, 2).T
        X2, X3, X4 = self._embed(x)
        _, t2n, t3n = ccsdt_opt.update_amps(self.step, self.t1, X2, X3, self.eris, mem_mb=self.mem_mb, plan=self.plan,
                                            model='cc3' if self.row3 == 'cc3' else 'ccsdt')
        s2 = ((t2n[0, 1] - X2[0, 1])[a2, b2] * self.jac[0][0]).astype(self.dtype)
        s3 = (np.asarray(t3n).reshape(X3.shape)[:no] - X3[:no]) * self.jac[1]
        del t2n, t3n
        if self.rank == 4:
            nq = comb(no, 2)
            if self.row4 == 'full':
                R4, r3, r2 = t4engine.residual(X2, X3, X4, self.ints)
                s4 = np.array(R4[:nq])
            elif self.row4 == 'cc4':
                from zccsdt.cc import ccn
                R4, r3, r2 = t4engine.residual(np.zeros_like(X2), np.zeros_like(X3), X4, self.ints)
                s4 = ccn.cc4_rhs(X2, X3, self.ints)[:nq] + self.fden[2] * np.asarray(x[2]).reshape(self.shapes[2])
            else:
                raise ValueError(self.row4)
            s2 = s2 + r2[0]
            s3 = s3 + r3[:no]
            del R4, r3, r2
            y[2][:] = s4.ravel()
        y[0][:] = s2.ravel()
        y[1][:] = s3.ravel()
        self.project(y)
        self.nsigma += 1
        self.seconds += time.perf_counter() - t0
        return y

    def precond(self, x, omega, floor=1e-3):
        for p, d in zip(x, self.den):
            den = omega - d.ravel()
            den[abs(den) < floor] = floor
            p /= den
        return x

    def guesses(self, n):
        '''Unit vectors on the ``n`` lowest pairs (a<b) of the masks.'''
        d = self.den[0].ravel().copy()
        d[~self.masks[0].ravel()] = np.inf
        out = []
        for k in np.argsort(d, kind='stable')[:n]:
            if np.isfinite(d[k]):
                v = self.new()
                v[0][k] = 1.0
                out.append(v)
        return out

    def matrix(self):
        '''The dense matrix on the elements of the masks (tests, tiny cases).'''
        idx = [(p, k) for p, m in enumerate(self.masks) for k in np.flatnonzero(m.ravel())]
        M = np.zeros((len(idx), len(idx)), dtype=self.dtype)
        for c, (p, k) in enumerate(idx):
            x, y = self.new(), self.new()
            x[p][k] = 1.0
            self.sigma(x, y)
            M[:, c] = [y[q][j] for q, j in idx]
        return M


def kernel(eris, t1, t2, t3=None, t4=None, rank=3, row3='full', row4='full', labels=None, target=0, parity=None,
           target_parity=0, nroots=1, nguess=None, conv_tol=1e-8, max_space=None, max_cycle=80, mem_mb=2000.0,
           verbose=False, stdout=None, problem=None):
    '''The ``nroots`` lowest roots of the DEA matrix of one level (module
    docstring) among the vectors of the masks: E(N+2) - E_CC(N) and the
    vectors.  ``conv_tol``: the norm of the residual of a normalized
    vector.  ``nguess``: unit vectors on the lowest pairs (a<b) to start
    from (default nroots + 2).  Returns a Result.'''
    import sys
    out = stdout or sys.stdout
    t0 = time.perf_counter()
    pb = problem or Problem(eris, t1, t2, t3, t4, rank, row3, row4, labels, target, parity, target_parity, mem_mb)
    times = {'prepare': time.perf_counter() - t0}
    store = eom.Store(pb.sizes, pb.dtype)
    gs = pb.guesses(nguess or nroots + 2)
    if not gs:
        raise ValueError('EOM-DEA: no pair of virtual spinors with these labels')
    nroots = min(nroots, len(gs))
    counts = tuple(int(m.sum()) for m in pb.masks)
    if verbose:
        print('EOM-DEA: rank %d (%s), %d roots, vectors of %s elements (%s in the masks), %s'
              % (pb.rank, ' + '.join(['2p', '3p1h', '4p2h'][:pb.rank - 1]), nroots,
                 ' + '.join(str(s) for s in pb.sizes), ' + '.join(str(c) for c in counts),
                 'real' if pb.dtype.kind == 'f' else 'complex'), file=out, flush=True)

    def log(it, n, w, res, conv, sec):
        if verbose:
            print('iter %3d  %3d vectors  %s  %.2f s' % (it, n, '  '.join(
                '%.11f %.1e%s' % (x.real, r, '*' if c else ' ') for x, r, c in zip(w, res, conv)), sec),
                file=out, flush=True)
    t1_ = time.perf_counter()
    res = eom.davidson(pb.sigma, pb.precond, gs, store, nroots, conv_tol, max_space or max(8 + 4 * nroots, len(gs) + nroots),
                       max_cycle, log=log)
    times['iterations'] = time.perf_counter() - t1_
    times['sigma'] = pb.seconds / max(pb.nsigma, 1)
    order = np.argsort(res.e.real, kind='stable')
    vec = [res.vectors[k] for k in order]
    wts = np.array([[float(np.vdot(p, p).real) for p in v] for v in vec])
    if verbose:
        for k, j in enumerate(order):
            print('  root %d  omega = %.11f  |residual| = %.1e%s  weights %s' % (
                k + 1, res.e[j].real, res.residuals[j], ' ' if res.converged[j] else ' !',
                ' '.join('%.4f' % x for x in wts[k] / wts[k].sum())), file=out, flush=True)
        print('  %d sigma vectors, %.2f s each; %.1f s' % (res.nsigma, times['sigma'], time.perf_counter() - t0),
              file=out, flush=True)
    return Result(e=res.e.real[order], converged=res.converged[order], residuals=res.residuals[order], vectors=vec,
                  weights=wts, nsigma=res.nsigma, niter=res.niter, sizes=counts, timings=times)


# ------------------------------------------------- in the blocks of the symmetry labels

def _rank_table(n, k):
    return {tuple(int(x) for x in t): r for r, t in enumerate(P.tuples(n, k))}


class ProblemSym(Problem):
    '''The same sigma vector in the blocks of the symmetry labels of a
    linear molecule (2 m_j; the spinors of ``eris`` in the order of their
    labels, the labels in use: the layout of the ground state).

    The two ghosts get labels inside the range of the others, 1 and
    target - 1, so that the operators R of the target - l_a + l_b + .. -
    l_i - .. = target - are totally symmetric on the padded spinors: T + R
    has the block structure of T, and the residual code of the ground state
    in label blocks (the CCSDT step, zccsdt.cc.t4engine_sym) is called as it
    is.  Its blocks hold the amplitudes without ghosts and those with one
    or two ghosts that are totally symmetric; a short vector is the
    elements with both, by index arrays into the flat arrays of the blocks.
    ``t3``: flat in the blocks of the engine of ``eris`` (or None), ``t4``:
    flat in the blocks of zccsdt.cc.t4engine_sym (or None).  The 3p1h and
    4p2h rows are the full ones.'''

    def __init__(self, eris, t1, t2, t3=None, t4=None, rank=3, target=0, parity=None, target_parity=0, mem_mb=2000.0,
                 batch_bytes=None):
        from zccsdt.cc import t3pack, t4engine_sym as S
        from zccsdt.eris import ERIs
        if t3pack.classes_of(eris) is None:
            raise ValueError('EOM-DEA in label blocks: labels without a modulus in use, the spinors in their order')
        self.rank, self.row3, self.row4 = rank, 'full', 'full'
        no, n = eris.nocc, np.asarray(eris.fock).shape[0]
        nv = n - no
        self.no, self.nv = no, nv
        self.dtype = dtype = np.dtype(np.result_type(t1, t2, eris.oovv, eris.fock))
        lab = np.asarray(eris.labels, dtype=np.int64)
        lo, lv = lab[:no], lab[no:]
        # the padded spinors, in the order of the labels: the ghosts before the others of their label
        glab = sorted((1, int(target) - 1))
        plo = np.concatenate((glab, lo))
        order = np.argsort(plo, kind='stable')
        where = np.empty(no + 2, dtype=np.int64)
        where[order] = np.arange(no + 2)
        g1, g2 = sorted(int(x) for x in where[:2])
        rp = where[2:]                          # the places of the occupied spinors of eris
        assert np.all(np.diff(rp) > 0)
        self.g, self.rp = (g1, g2), rp

        def pad(x, key):
            x = np.asarray(x)
            if 'o' not in key:
                return x
            out = np.zeros(tuple(no + 2 if c == 'o' else d for d, c in zip(x.shape, key)), dtype=x.dtype)
            out[np.ix_(*[rp if c == 'o' else np.arange(d) for d, c in zip(x.shape, key)])] = x
            return out
        ev = np.asarray(eris.mo_energy).real
        self.eg = eg = -(4.0 * float(abs(ev).max()) + 10.0)
        fock = np.zeros((n + 2, n + 2), dtype=np.asarray(eris.fock).dtype)
        keep = np.concatenate((rp, no + 2 + np.arange(nv)))
        fock[np.ix_(keep, keep)] = eris.fock
        e = np.full(n + 2, eg)
        e[keep] = ev
        blocks = {k: (None if getattr(eris, k) is None else pad(getattr(eris, k), k))
                  for k in ('oooo', 'ooov', 'oovv', 'ovov', 'ovvo', 'ovvv', 'vvvv')}
        e2 = ERIs(nocc=no + 2, fock=fock, mo_energy=e, e_hf=eris.e_hf, e_nuc=eris.e_nuc,
                  source=(eris.source or '') + ' + 2 ghosts', **blocks)
        if getattr(eris, '_vvvv_pp', None) is not None:
            e2._vvvv_pp = eris._vvvv_pp
        e2.labels = np.concatenate((plo[order], lv))
        e2.modulus, e2.use_labels = 0, True
        self.eris = e2
        self.lay = lay = S.layout_for(e2)
        if lay is None:
            raise ValueError('EOM-DEA in label blocks: no layout for these labels')
        self.S = S
        self.t1 = np.ascontiguousarray(pad(t1, 'ov'), dtype=dtype)
        self.t2 = np.ascontiguousarray(pad(t2, 'oovv'), dtype=dtype)
        n3 = int(lay.t3off[-1])
        assert ccsdt_opt.engine_for(e2, dtype).size == n3
        self.t3 = np.zeros(n3, dtype=dtype)
        if t3 is not None:
            # the blocks of eris -> the packed block -> the rows of the padded occupied triples -> its blocks
            rk = _rank_table(no + 2, 3)
            rows = np.array([rk[tuple(int(x) for x in rp[t])] for t in P.tuples(no, 3)])
            big = np.zeros((comb(no + 2, 3), comb(nv, 3)), dtype=dtype)
            big[rows] = ccsdt_opt.packed_t3(eris, t3)
            self.t3 = np.ascontiguousarray(lay.t3_from_packed(big), dtype=dtype)
            del big
        self.t4 = None
        if rank == 4:
            self.t4 = lay.zeros(dtype)
            if t4 is not None:
                lay0 = S.layout_for(eris)
                rk = _rank_table(no + 2, 4)
                rows = np.array([rk[tuple(int(x) for x in rp[t])] for t in P.tuples(no, 4)])
                big = np.zeros((comb(no + 2, 4), comb(nv, 4)), dtype=dtype)
                big[rows] = lay0.to_packed(np.asarray(t4).ravel())
                self.t4 = np.ascontiguousarray(lay.from_packed(big), dtype=dtype)
                del big
            self.ints = t4engine.Dressed.from_eris(e2, self.t1)
        self.mem_mb, self.batch_bytes = mem_mb, batch_bytes
        self.step = solver.Options(mem_mb=mem_mb)
        self.plan = ccsdt_opt.plan_for(e2, dtype, solver.Options(mem_mb=mem_mb, incore_complete=True))
        # ---- the elements with both ghosts: places in the flat arrays, signs (ghosts to the front),
        #      denominators e_a + .. - e_i - .., parity masks
        eo, evv = ev[:no], ev[no:]
        par = None if parity is None else np.asarray(parity, dtype=np.int64)
        back = np.full(no + 2, -1, dtype=np.int64)
        back[rp] = np.arange(no)

        def sector(tuples_):
            '''(rows with both ghosts, their signs, the spinors of eris that are left [n, k - 2]).'''
            has = (tuples_ == g1).any(axis=1) & (tuples_ == g2).any(axis=1)
            rows = np.flatnonzero(has)
            t = tuples_[rows]
            p1, p2 = np.argmax(t == g1, axis=1), np.argmax(t == g2, axis=1)
            sgn = np.where((p1 + p2 - 1) % 2, -1.0, 1.0)
            rest = np.array([[back[x] for x in row if x not in (g1, g2)] for row in t], dtype=np.int64).reshape(len(t), tuples_.shape[1] - 2)
            return rows, sgn, rest
        v2 = P.tuples(nv, 2)
        m2 = lv[v2].sum(axis=1) == target
        if par is not None:
            m2 &= (par[no:][v2].sum(axis=1) - target_parity) % 2 == 0
        self.pair_row = _rank_table(no + 2, 2)[(g1, g2)]
        self.sizes, self.den, self.masks = [len(v2)], [evv[v2].sum(axis=1)], [m2]
        self.idx, self.sgn, self.dpos = [None], [None], [None]
        v3 = P.tuples(nv, 3)
        parts = [(lay.O3cls, lay.o3, lambda k: v3[lay.A3cls[k]], lay.t3off, lay.nK, range(lay.n3),
                  lambda k: np.asarray(lay.A3cls[k]), comb(nv, 3))]
        if rank == 4:
            parts.append((lay.O4cls, lay.o4, lambda M: lay.quads[M].astype(np.int64), lay.boff, lay.nA, lay.sectors,
                          lambda M: lay.lex(M), comb(nv, 4)))
        for ocls, otup, vtup, off, ncol, which, vlex, nlex in parts:
            idx, sgn, den, mask, dpos = [], [], [], [], []
            for k in which:
                if not len(ocls[k]) or not ncol[k]:
                    continue
                rows, sg, rest = sector(otup[ocls[k]])
                if not len(rows):
                    continue
                vt = vtup(k)
                idx.append((off[k] + rows[:, None] * ncol[k] + np.arange(ncol[k])[None, :]).ravel())
                sgn.append(np.repeat(sg, ncol[k]))
                den.append((evv[vt].sum(axis=1)[None, :] - eo[rest].sum(axis=1)[:, None]).ravel())
                # the same element on the dense array [i, (abc)] / [(i<j), (abcd)] in lexicographic order
                row = rest[:, 0] if rest.shape[1] == 1 else (rest[:, 0] * (2 * no - rest[:, 0] - 1) // 2
                                                             + rest[:, 1] - rest[:, 0] - 1)
                dpos.append((row[:, None] * nlex + np.asarray(vlex(k), dtype=np.int64)[None, :]).ravel())
                if par is not None:
                    mask.append((((par[no:][vt].sum(axis=1)[None, :] - par[:no][rest].sum(axis=1)[:, None])
                                  - target_parity) % 2 == 0).ravel())
            self.idx.append(np.concatenate(idx) if idx else np.zeros(0, dtype=np.int64))
            self.sgn.append(np.concatenate(sgn) if sgn else np.zeros(0))
            self.den.append(np.concatenate(den) if den else np.zeros(0))
            self.masks.append(np.concatenate(mask) if mask else np.ones(len(self.idx[-1]), bool))
            self.dpos.append(np.concatenate(dpos) if dpos else np.zeros(0, dtype=np.int64))
            self.sizes.append(len(self.idx[-1]))
        self.shapes = [(s,) for s in self.sizes]
        self.jac = [2 * eg - self.den[0], 2 * eg - self.den[1]]
        self.nsigma, self.seconds = 0, 0.0
        self.block_sizes = (n3, lay.size if rank == 4 else 0)

    @property
    def ghosts(self):
        return self.g

    @property
    def places(self):
        return self.rp

    def _x23(self, x):
        nv, (g1, g2) = self.nv, self.g
        a2, b2 = P.tuples(nv, 2).T
        X2 = self.t2.copy()
        r2 = np.zeros((nv, nv), dtype=self.dtype)
        r2[a2, b2] = x[0]
        r2 -= r2.T
        X2[g1, g2] += r2
        X2[g2, g1] -= r2
        X3 = self.t3.copy()
        X3[self.idx[1]] += self.sgn[1] * x[1]
        return X2, X3

    def padded(self, x):
        X2, X3 = self._x23(x)
        X4 = self.t4.copy()
        X4[self.idx[2]] += self.sgn[2] * x[2]
        return X2, self.lay.t3_to_packed(X3), self.lay.to_packed(X4)

    def add_dense(self, y, S3, S4):
        y[1] += S3.ravel()[self.dpos[1]]
        y[2] += S4.ravel()[self.dpos[2]]
        return y

    def sigma(self, x, y):
        t0 = time.perf_counter()
        nv, (g1, g2) = self.nv, self.g
        a2, b2 = P.tuples(nv, 2).T
        X2, X3 = self._x23(x)
        _, t2n, t3n = ccsdt_opt.update_amps(self.step, self.t1, X2, X3, self.eris, mem_mb=self.mem_mb, plan=self.plan)
        s2 = (t2n[g1, g2] - X2[g1, g2])[a2, b2] * self.jac[0]
        s3 = (np.asarray(t3n)[self.idx[1]] - X3[self.idx[1]]) * self.sgn[1] * self.jac[1]
        del t2n, t3n
        if self.rank == 4:
            X4 = self.t4.copy()
            X4[self.idx[2]] += self.sgn[2] * x[2]
            R4, r3, r2_ = self.S.residual(self.lay, X2, X3, X4, self.ints, self.batch_bytes)
            del X4
            y[2][:] = R4[self.idx[2]] * self.sgn[2]
            s3 += np.asarray(r3).ravel()[self.idx[1]] * self.sgn[1]
            s2 = s2 + r2_[self.pair_row]
            del R4, r3, r2_
        y[0][:] = s2
        y[1][:] = s3
        self.project(y)
        self.nsigma += 1
        self.seconds += time.perf_counter() - t0
        return y
