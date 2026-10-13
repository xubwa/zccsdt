'''The 5p3h operators of EOM-DEA at lowest order, folded (zccsdt.cc.dea5),
against the matrix on determinants (zccsdt.cc.dea_strings), LiH / STO-3G
(4 + 8) with random amplitudes:

- the terms of r5 in the rows of 3p1h and 4p2h are the blocks
  <3p1h| Hbar |5p3h>, <4p2h| Hbar |5p3h> of the complete matrix (T1 enters);
- the right-hand side of the 5p3h row, term by term: [V, R4] is the block
  <5p3h| H |4p2h> without T; [[V, T2], R3], [[V, R2], T3] and
  1/2 [[[V, T2], T2], R2] are the parts of the complete block of that order
  in the amplitudes;
- the folded root is the lowest eigenvalue of the matrix of the model with
  the 5p3h rows written out.'''
from math import comb

import numpy as np
from pyscf import scf

from fockspace import random_antisym
from test_dea import TIGHT, _ghf
from zccsdt import eris as E
from zccsdt.cc import ccsdtq_ref as R, ccsdtq_terms as Q, dea, dea5, dea_strings as DS, solver, t4pack as P

NO, NV = 4, 8


def _eris():
    return E.from_scf(scf.addons.convert_to_ghf(_ghf('Li 0 0 0; H 0 0 1.5957')))


def _to_det(pb, parts):
    '''The vector of the determinant problem for the operators r[(holes),
    (particles)] of 0 .. 3 holes in the convention of the ghost route:
    c[K, B] = (-1)^(h (h - 1) / 2 + i + j + ..) r.'''
    out = pb.new()
    full = (1 << NO) - 1
    for h, r in enumerate(parts):
        if r is None:
            continue
        r = np.asarray(r).reshape(comb(NO, h), comb(NV, h + 2))
        grp = pb.vs.groups(h + 2)[(0, 0)]
        bpos = np.searchsorted(grp, (1 << P.tuples(NV, h + 2)).sum(axis=1))
        for row, holes in enumerate(P.tuples(NO, h) if h else np.zeros((1, 0), dtype=int)):
            k = int(np.searchsorted(pb.K[h], full ^ int(sum(1 << int(i) for i in holes))))
            sg = (-1.0) ** (h * (h - 1) // 2 + int(sum(holes)))
            out[h][pb.rows[h][k] + bpos] = sg * r[row]
    return out


def _from_det(pb, y):
    '''The inverse of _to_det.'''
    out = []
    full = (1 << NO) - 1
    for h in range(len(y)):
        r = np.zeros((comb(NO, h), comb(NV, h + 2)))
        grp = pb.vs.groups(h + 2)[(0, 0)]
        bpos = np.searchsorted(grp, (1 << P.tuples(NV, h + 2)).sum(axis=1))
        for row, holes in enumerate(P.tuples(NO, h) if h else np.zeros((1, 0), dtype=int)):
            k = int(np.searchsorted(pb.K[h], full ^ int(sum(1 << int(i) for i in holes))))
            sg = (-1.0) ** (h * (h - 1) // 2 + int(sum(holes)))
            r[row] = sg * y[h][pb.rows[h][k] + bpos]
        out.append(r)
    return out


def _amps(rng, scale=(0.05, 0.05, 0.05, 0.05)):
    dense = [scale[0] * rng.standard_normal((NO, NV))] + [scale[k - 1] * random_antisym(rng, NO, NV, k) for k in (2, 3, 4)]
    return dense, [dense[0]] + [Q.unique(t, k) for t, k in zip(dense[1:], (2, 3, 4))]


def _problems(eris, dense, packed):
    fock, g, _, _ = R.integrals(eris)
    det = DS.Problem(np.asarray(fock).real, np.asarray(g).real, NO, packed, 3, e_ref=eris.e_hf)
    gh = dea.Problem(eris, dense[0], dense[1], packed[2], packed[3], rank=4)
    return det, gh, dea5.Fold5(gh)


def test_the_conventions_and_the_rows_below():
    eris = _eris()
    rng = np.random.default_rng(11)
    fock, g, _, _ = R.integrals(eris)
    # the two routes on the same vector, element by element (2p, 3p1h, 4p2h; converged amplitudes:
    # the determinant matrix is Hbar, the commutator form differs by the energy on the diagonal)
    cc = solver.kernel(eris, 'ccsdt', TIGHT)
    gh = dea.Problem(eris, cc.t1, cc.t2, cc.t3, None, rank=4)
    d2 = DS.Problem(np.asarray(fock).real, np.asarray(g).real, NO, [cc.t1, Q.unique(cc.t2, 2), np.asarray(cc.t3).reshape(4, 56)],
                    2, e_ref=eris.e_hf)
    x = [rng.standard_normal(s) for s in gh.sizes]
    y = gh.sigma(x, gh.new())
    yd = _from_det(d2, d2.sigma(_to_det(d2, x), d2.new()))
    e0 = eris.e_hf + cc.e_corr - d2.const
    for a, b, c in zip(yd, y, x):
        assert abs(a.ravel() - b - e0 * c).max() < 1e-9
    # r5 in the rows of 3p1h and 4p2h, random amplitudes T1 .. T4
    dense, packed = _amps(rng)
    det, gh, f5 = _problems(eris, dense, packed)
    r5 = rng.standard_normal((comb(NO, 3), comb(NV, 5)))
    yd = _from_det(det, det.sigma(_to_det(det, [None, None, None, r5]), det.new()))
    S3, S4 = np.zeros((NO, comb(NV, 3))), np.zeros((comb(NO, 2), comb(NV, 4)))
    for row, t in enumerate(P.tuples(NO, 3)):
        f5.back(np.ascontiguousarray(r5[row]), tuple(int(v) for v in t), S3, S4)
    assert abs(yd[0]).max() < 1e-10 and abs(S3).max() > 0.1 and abs(S4).max() > 0.1
    assert abs(yd[1] - S3).max() < 1e-10 and abs(yd[2] - S4).max() < 1e-10


def test_the_right_hand_side_term_by_term():
    '''<5p3h| [Hbar, R] |0> of second-quantized operators in the Fock space
    (the commutator: no disconnected part), for R of one rank and the
    amplitudes that select one term.'''
    import scipy.sparse as sp
    from fockspace import Fock
    eris = _eris()
    rng = np.random.default_rng(12)
    fock, g, _, _ = R.integrals(eris)
    fock, g = np.asarray(fock).real, np.asarray(g).real
    n = NO + NV
    fk = Fock(NO, NV)
    H = fk.two_body(g) + sum(fock[p, q] * (fk.ad[p] @ fk.a[q]) for p in range(n) for q in range(n) if fock[p, q] != 0)
    dense, packed = _amps(rng, (0.0, 0.3, 0.3, 0.0))
    sizes = [comb(NV, 2), NO * comb(NV, 3), comb(NO, 2) * comb(NV, 4)]

    def op(holes, parts):
        m = sp.identity(2 ** n, format='csr')
        for a in parts:
            m = m @ fk.ad[NO + int(a)]
        for i in reversed(holes):
            m = m @ fk.a[int(i)]
        return m

    def commutator(amps_d, x, k):
        T = sum(fk.excitation(t) for t in amps_d if abs(t).max() > 0) if any(abs(t).max() > 0 for t in amps_d) else None
        Rop = sp.csr_matrix((2 ** n,) * 2)
        r = x[k].reshape(comb(NO, k), comb(NV, k + 2))
        for row, holes in enumerate(P.tuples(NO, k) if k else np.zeros((1, 0), dtype=int)):
            for col, parts in enumerate(P.tuples(NV, k + 2)):
                Rop = Rop + r[row, col] * op(holes, parts)

        def ex(sign, v):
            if T is None:
                return v
            out, term = v.copy(), v.copy()
            for j in range(1, n + 1):
                term = (sign / j) * (T @ term)
                out = out + term
            return out

        def hbar(v):
            return ex(-1.0, H @ ex(1.0, v))
        vec = hbar(Rop @ fk.ref) - Rop @ hbar(fk.ref)
        out = np.zeros((comb(NO, 3), comb(NV, 5)))
        for row, holes in enumerate(P.tuples(NO, 3)):
            for col, parts in enumerate(P.tuples(NV, 5)):
                out[row, col] = (op(holes, parts) @ fk.ref) @ vec
        return out

    def mine(amps_d, amps_p, x):
        gh = dea.Problem(eris, amps_d[0], amps_d[1], amps_p[2], amps_p[3], rank=4)
        return dea5.Fold5(gh).rows5(x)

    def only(k):
        x = [np.zeros(s) for s in sizes]
        x[k] = rng.standard_normal(sizes[k])
        return x

    def pick(amps, keep, sign=1.0):
        return [sign * a if k in keep else np.zeros_like(a) for k, a in enumerate(amps)]
    # [V, R4]: no amplitudes
    x = only(2)
    want, got = commutator(pick(dense, ()), x, 2), mine(pick(dense, ()), pick(packed, ()), x)
    assert abs(want).max() > 0.1 and abs(want - got).max() < 1e-10
    # [[V, T2], R3]: the part of the complete row that is odd in T2
    x = only(1)
    wp, wm = commutator(pick(dense, (1,)), x, 1), commutator(pick(dense, (1,), -1.0), x, 1)
    got = mine(pick(dense, (1,)), pick(packed, (1,)), x)
    assert abs(wp - wm).max() > 0.1 and abs(0.5 * (wp - wm) - got).max() < 1e-10
    # [[V, R2], T3]: T3 alone;  1/2 [[[V, T2], T2], R2]: T2 alone
    x = only(0)
    for keep in ((2,), (1,)):
        want, got = commutator(pick(dense, keep), x, 0), mine(pick(dense, keep), pick(packed, keep), x)
        assert abs(want).max() > 0.01 and abs(want - got).max() < 1e-10


def test_the_folded_root():
    '''CCSDT amplitudes: the root that follows the lowest root of the rows
    to 4p2h is the lowest eigenvalue of the matrix of the model, the 5p3h
    rows written out with their Fock denominators.'''
    eris = _eris()
    cc = solver.kernel(eris, 'ccsdt', TIGHT)
    gh = dea.Problem(eris, cc.t1, cc.t2, cc.t3, None, rank=4)
    f5 = dea5.Fold5(gh)
    n, n5 = sum(gh.sizes), comb(NO, 3) * comb(NV, 5)
    off = np.concatenate(([0], np.cumsum(gh.sizes)))
    M = np.zeros((n + n5, n + n5))
    for c in range(n):
        x, y = gh.new(), gh.new()
        k = int(np.searchsorted(off, c, side='right') - 1)
        x[k][c - off[k]] = 1.0
        M[:n, c] = np.concatenate(gh.sigma(x, y))
        M[n:, c] = f5.rows5(x).ravel()
    trip = P.tuples(NO, 3)
    for c in range(n5):
        r5 = np.zeros(comb(NV, 5))
        r5[c % comb(NV, 5)] = 1.0
        S3, S4 = np.zeros((NO, comb(NV, 3))), np.zeros((comb(NO, 2), comb(NV, 4)))
        t = tuple(int(v) for v in trip[c // comb(NV, 5)])
        f5.back(r5, t, S3, S4)
        M[off[1]:off[2], n + c] = S3.ravel()
        M[off[2]:off[3], n + c] = S4.ravel()
        M[n + c, n + c] = f5.ev5[c % comb(NV, 5)] - f5.eo[list(t)].sum()
    w = np.linalg.eigvals(M)
    low = np.sort(w.real[abs(w.imag) < 1e-9])[0]
    res = dea.kernel(None, None, None, problem=gh, nroots=1, nguess=28, conv_tol=1e-9)
    omega, sol, _ = dea5.kernel(gh, (float(res.e[0]), res.vectors[0]), conv_tol=1e-9)
    assert sol.converged.all() and 1e-6 < abs(omega - res.e[0]) < 1e-1
    assert abs(omega - low) < 1e-8
