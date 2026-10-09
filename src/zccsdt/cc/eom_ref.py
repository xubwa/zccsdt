'''Dense reference of equation-of-motion CCSDT (and CCSD) on spin orbitals or
spinors, real or complex: the sigma vector of the right-hand eigenproblem

    sigma_mu = <mu| [Hbar, R] |0> = sum_nu (dResidual_mu / dt_nu) r_nu = omega r_mu,

Hbar = exp(-T) H exp(T), R = R1 + R2 (+ R3) an excitation operator, mu over
the singles, doubles (and triples).  docs/eom.md.

The sigma vector is the Jacobian of the amplitude equations on r - for any
amplitudes, converged or not - and is evaluated in three ways that share as
little as possible:

'terms'    the table of the amplitude equations (zccsdt.cc.ccsdtq_terms,
           T1-dressed formalism, from Wick's theorem) differentiated term by
           term: every amplitude factor replaced by r of its rank in turn,
           and the dressed integrals by their derivative along r1,
           H' = [H~, R1] (dressed_derivative: a one-index transformation of
           the dressed integrals).  This is the form the implementation on
           packed amplitudes takes (zccsdt.cc.eom_opt).
'stencil'  the directional derivative of the residuals by the five-point
           formula, exact for a polynomial of degree <= 4
           (qlgrad_ref.jacobian): no algebra of its own.
'dressed'  three evaluations of the residuals of the dressed formalism
           (qlgrad_ref.jacobian_dressed).

(A fourth, exact one is in tests/test_eom_ref.py: the matrix of Hbar in a
Fock space.)

Ionized and electron-attached states (IP: R = 1h + 2h1p + 3h2p, EA: 1p +
2p1h + 3p2h) are the same equations with one more spin orbital that no
integral touches - a virtual one that R always excites into (IP), an
occupied one that R always excites from (EA): ghost().

Vectors: [r1[i,a], r2[i,j,a,b], r3[i,j,k,a,b,c]] dense and antisymmetric,
or flat over the unique elements (i<j<k, a<b<c in lexicographic order, rank
after rank): pack(), unpack().  Everything here is for a few dozen spin
orbitals.
'''

import numpy as np

from zccsdt.cc import ccsdt, ccsdtq_terms as Q, qlgrad_ref

RNAME = {'t2': 'r2', 't3': 'r3'}


def dressed_derivative(ge, fdr, r1, nocc):
    '''(g', f'): the derivative of the T1-dressed integrals and Fock matrix
    along r1, the integrals of [H~, R1]: every virtual bra index a picks up
    -sum_m r1[m,a] <m..||..>~ and every occupied ket index i
    +sum_e r1[i,e] <..||..e>~ (one slot at a time: first order), and the
    Fock matrix sum_ia <ri||sa>~ r1[i,a] besides.'''
    n = ge.shape[0]
    X = np.zeros((n, n), dtype=np.result_type(ge, r1))
    Y = np.zeros_like(X)
    X[nocc:, :nocc] = -r1.T
    Y[:nocc, nocc:] = r1
    dg = np.einsum('pt,tqrs->pqrs', X, ge) + np.einsum('qt,ptrs->pqrs', X, ge)
    dg += np.einsum('pqus,ru->pqrs', ge, Y) + np.einsum('pqru,su->pqrs', ge, Y)
    df = X @ fdr + fdr @ Y.T + np.einsum('risa,ia->rs', ge[:, :nocc, :, nocc:], r1)
    return dg, df


def linearized_terms(rank, with_t3=True):
    '''The terms of sigma_rank that have r2 or r3: every term of the
    residual (ccsdtq_terms.residual_terms) with each of its amplitude
    factors replaced in turn.'''
    out = []
    for coef, factors in Q.residual_terms(rank, with_t4=False, with_t3=with_t3):
        for k, (kind, ix) in enumerate(factors):
            if kind in RNAME:
                out.append((coef, factors[:k] + ((RNAME[kind], ix),) + factors[k + 1:]))
    return out


def sigma(r, amps, fock, g, nocc, how='terms'):
    '''[sigma_1, sigma_2(, sigma_3)] dense for r = [r1, r2(, r3)] and the
    amplitudes [t1, t2(, t3)] (two of each: EOM-CCSD) of the Hamiltonian
    (fock, <pq||rs>).  ``how``: see the module.'''
    n = g.shape[0]
    with_t3 = len(amps) == 3
    if how == 'stencil':
        def res(a):
            ge, fdr = ccsdt._t1_dress(g, fock, a[0], nocc, n)
            arrays = dict({'v': ge, 'f': fdr}, **{'t%d' % (k + 1): x for k, x in enumerate(a) if k})
            return [Q.residual(k + 1, arrays, nocc, with_t4=False, with_t3=with_t3) for k in range(len(a))]
        return qlgrad_ref.directional(res, list(amps), list(r))
    if how == 'dressed':
        assert with_t3
        return qlgrad_ref.jacobian_dressed(list(r), list(amps), fock, g, nocc)
    if how != 'terms':
        raise ValueError(how)
    ge, fdr = ccsdt._t1_dress(g, fock, amps[0], nocc, n)
    dg, df = dressed_derivative(ge, fdr, r[0], nocc)
    arrays = {'v': ge, 'f': fdr}
    primed = {'v': dg, 'f': df}
    for k in range(1, len(amps)):
        arrays['t%d' % (k + 1)] = primed['t%d' % (k + 1)] = amps[k]
        arrays['r%d' % (k + 1)] = r[k]
    out = []
    for rank in range(1, len(amps) + 1):
        x = Q.total(Q.residual_terms(rank, with_t4=False, with_t3=with_t3), primed, nocc, rank)
        x = x + Q.total(linearized_terms(rank, with_t3), arrays, nocc, rank)
        out.append(x if rank == 1 else Q.asym(x, rank))
    return out


# ------------------------------------------------------------------ vectors

def pack(r):
    '''The unique elements of [r1, r2, ..] as one flat vector.'''
    return np.concatenate([r[0].ravel()] + [Q.unique(x, k + 1).ravel() for k, x in enumerate(r) if k])


def sizes(nocc, nvir, nrank):
    from math import comb
    return [comb(nocc, k) * comb(nvir, k) for k in range(1, nrank + 1)]


def unpack(x, nocc, nvir, nrank):
    '''The inverse of pack().'''
    out, off = [], 0
    for k, n in enumerate(sizes(nocc, nvir, nrank)):
        u = x[off:off + n]
        off += n
        out.append(u.reshape(nocc, nvir) if k == 0 else Q.expand(u.reshape(len(Q.tuples(nocc, k + 1)), -1), nocc, nvir, k + 1))
    return out


def apply(x, amps, fock, g, nocc, how='terms'):
    '''sigma for a flat vector over the unique elements.'''
    nvir = g.shape[0] - nocc
    return pack(sigma(unpack(x, nocc, nvir, len(amps)), amps, fock, g, nocc, how))


def matrix(amps, fock, g, nocc, how='terms', select=None):
    '''The matrix of sigma among the unique elements (``select``: among
    these of them only), column by column.'''
    nvir = g.shape[0] - nocc
    n = sum(sizes(nocc, nvir, len(amps)))
    idx = np.arange(n) if select is None else np.asarray(select)
    dtype = np.result_type(fock, g, *amps)
    M = np.zeros((len(idx), len(idx)), dtype=dtype)
    for c, j in enumerate(idx):
        e = np.zeros(n, dtype=dtype)
        e[j] = 1.0
        M[:, c] = apply(e, amps, fock, g, nocc, how)[idx]
    return M


def diagonal(mo_energy, nocc, nrank):
    '''The differences of the orbital energies over the unique elements,
    e_a + .. - e_i - ..: the diagonal of the zeroth-order matrix, which
    the preconditioner of the eigensolver uses.'''
    eo, ev = np.asarray(mo_energy[:nocc]), np.asarray(mo_energy[nocc:])
    out = []
    for k in range(1, nrank + 1):
        occ, vir = Q.tuples(nocc, k), Q.tuples(len(ev), k)
        out.append((ev[vir].sum(axis=1)[None, :] - eo[occ].sum(axis=1)[:, None]).ravel())
    return np.concatenate(out)


# ------------------------------------------------- ionized and attached states

def ghost(amps, fock, g, nocc, kind):
    '''(amps, fock, g, nocc, select) of the problem with one more spin
    orbital that no integral touches: the last virtual one for kind 'ip',
    the first occupied one for 'ea'.  ``select``: the unique elements of an
    EE vector there that have the ghost index once - the elements of the IP
    vector r[i], r[i<j, a], r[i<j<k, a<b] (EA: r[a], r[i, a<b],
    r[i<j, a<b<c]), in lexicographic order of the remaining indices.  The
    matrix of sigma among them is that of the IP / EA problem; its
    eigenvalues are the ionization energies (EA: minus the electron
    affinities, E(N+1) - E(N)).'''
    n = g.shape[0]
    nvir = n - nocc
    at = n if kind == 'ip' else 0
    keep = [p for p in range(n + 1) if p != at]
    f2 = np.zeros((n + 1, n + 1), dtype=fock.dtype)
    g2 = np.zeros((n + 1,) * 4, dtype=g.dtype)
    f2[np.ix_(keep, keep)] = fock
    g2[np.ix_(keep, keep, keep, keep)] = g
    no2, nv2 = (nocc, nvir + 1) if kind == 'ip' else (nocc + 1, nvir)
    amps2, select, off = [], [], 0
    for k, t in enumerate(amps):
        rank = k + 1
        t2 = np.zeros((no2,) * rank + (nv2,) * rank, dtype=t.dtype)
        sl = (slice(0, nocc),) * rank + (slice(0, nvir),) * rank if kind == 'ip' \
            else (slice(1, None),) * rank + (slice(None),) * rank
        t2[sl] = t
        amps2.append(t2)
        occ, vir = Q.tuples(no2, rank), Q.tuples(nv2, rank)
        ho = np.ones(len(occ), bool) if kind == 'ip' else (occ == 0).any(axis=1)
        hv = (vir == nvir).any(axis=1) if kind == 'ip' else np.ones(len(vir), bool)
        select.append(off + np.flatnonzero((ho[:, None] & hv[None, :]).ravel()))
        off += len(occ) * len(vir)
    return amps2, f2, g2, no2, np.concatenate(select)


def eigenvalues(amps, fock, g, nocc, kind='ee', how='terms'):
    '''All eigenvalues of the EOM matrix (sorted by their real parts): the
    excitation energies of EOM-CCSDT (EOM-CCSD for two amplitudes), or the
    ionization energies ('ip') or E(N+1) - E(N) ('ea').'''
    select = None
    if kind != 'ee':
        amps, fock, g, nocc, select = ghost(amps, fock, g, nocc, kind)
    w = np.linalg.eigvals(matrix(amps, fock, g, nocc, how, select))
    return w[np.argsort(w.real)]
