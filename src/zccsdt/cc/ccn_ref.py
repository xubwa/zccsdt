'''Dense reference of the approximate iterative models CC2, CC3 and CC4 and
of their Jacobians (docs/ccn.md), next to CCSD, CCSDT and CCSDTQ, from the
term tables of zccsdt.cc.ccsdtq_terms.

T1-dressed formalism: T1 is in the integrals <pq||rs>~ and the Fock matrix
f~ (ccsdt._t1_dress), the cluster operator of the equations is T2 + T3 + T4.
With n the highest rank of a model (2, 3, 4 for CC2, CC3, CC4), the
equations of the ranks below n are those of the full model (CCSD, CCSDT,
CCSDTQ), and the equation of rank n keeps

    CC2:  <D| V~ + [F, T2] |0>
    CC3:  <T| [V~, T2] + [F, T3] |0>
    CC4:  <Q| [V~, T3] + 1/2 [[V~, T2], T2] + [F, T4] |0>

with V~ the dressed two-body operator and F **the occupied-occupied and
virtual-virtual blocks of the Fock matrix as it is** - not dressed with T1,
not assumed diagonal (``fock_top='bare'``).  For a Hartree-Fock reference
that is all there is to say.  For another reference (f_ia not zero) there
are choices, which the other values of ``fock_top`` implement for
comparison:

    'one'      F~1 = exp(-T1) F exp(T1): the blocks f_mi + f_me t_ie and
               f_ae - t_ma f_me (the commutator [[F, T1], T_n] kept)
    'dressed'  the Fock matrix f~ of the dressed Hamiltonian (that also has
               [[V, T1], T_n], which no definition of these models keeps)

The Jacobian (the matrix whose eigenvalues are the excitation energies of
linear response, the EOM analogue) is the derivative of these residuals:
the terms with every amplitude factor replaced in turn by r of its rank,
plus the terms with the derivative [H~, R1] of the dressed integrals
(eom_ref.dressed_derivative); the bare F has no derivative.  Its block of
rank n is the Fock operator alone, so the eigenproblem can be folded into
the ranks below n (folded(), omega-dependent).

Everything is dense, for a few dozen spin orbitals: the check of the
production code (zccsdt.cc.ccn) and of the definitions (tests/test_ccn_ref.py).
'''

from dataclasses import dataclass

import numpy as np

from zccsdt.cc import ccsdt, ccsdtq_ref, ccsdtq_terms as Q, eom_ref
from zccsdt.cc.diis import DIIS

NRANK = {'cc2': 2, 'ccsd': 2, 'cc3': 3, 'ccsdt': 3, 'cc4': 4, 'ccsdtq': 4}
# the groups of ccsdtq_terms.GROUPS in the equation of the highest rank
TOP = {
    'cc2': ('R2: V', 'R2: [F,T2]'),
    'cc3': ('R3: [V,T2]', 'R3: [F,T3]'),
    'cc4': ('R4: [V,T3]', 'R4: [F,T4]', 'R4: 1/2 [[V,T2],T2]'),
}
RNAME = {'t2': 'r2', 't3': 'r3', 't4': 'r4'}


def terms(model, rank):
    '''The terms of the equation of ``rank``; in the equation of the highest
    rank of an approximate model the Fock factor is 'f0' (the Fock operator
    of that equation) in the place of the dressed 'f'.'''
    n = NRANK[model]
    if model in TOP and rank == n:
        out = []
        for name in TOP[model]:
            for coef, factors in Q.GROUPS[name][4]:
                out.append((coef, tuple(('f0', ix) if kind == 'f' else (kind, ix) for kind, ix in factors)))
        return out
    return Q.residual_terms(rank, with_t4=n >= 4, with_t3=n >= 3)


def _total(ts, arrays, nocc, rank):
    """ccsdtq_terms.total with the factor 'f0' taken from arrays['f0']."""
    plain = [t for t in ts if all(kind != 'f0' for kind, _ in t[1])]
    top = [(c, tuple(('f', ix) if kind == 'f0' else (kind, ix) for kind, ix in fs)) for c, fs in ts
           if any(kind == 'f0' for kind, _ in fs)]
    out = Q.total(plain, arrays, nocc, rank) if plain else 0
    if top:
        out = out + Q.total(top, dict(arrays, f=arrays['f0']), nocc, rank)
    return out


def _one_body_dressed(fock, t1, nocc):
    n = fock.shape[0]
    x = np.eye(n, dtype=np.result_type(fock, t1))
    y = x.copy()
    x[nocc:, :nocc] -= t1.T
    y[:nocc, nocc:] += t1
    return x @ fock @ y.T


def _arrays(amps, fock, g, nocc, fock_top):
    ge, fdr = ccsdt._t1_dress(g, fock, amps[0], nocc, g.shape[0])
    if not np.iscomplexobj(g) and not np.iscomplexobj(fock) and not np.iscomplexobj(amps[0]):
        ge, fdr = ge.real, fdr.real
    f0 = {'bare': fock, 'one': _one_body_dressed(fock, amps[0], nocc), 'dressed': fdr}[fock_top]
    arrays = {'v': ge, 'f': fdr, 'f0': f0}
    arrays.update({'t%d' % (k + 1): a for k, a in enumerate(amps) if k})
    return arrays


def residuals(model, amps, fock, g, nocc, fock_top='bare'):
    '''[R1, .., Rn] of the model for dense amplitudes [t1, .., tn].'''
    assert len(amps) == NRANK[model]
    arrays = _arrays(amps, fock, g, nocc, fock_top)
    out = []
    for k in range(len(amps)):
        x = _total(terms(model, k + 1), arrays, nocc, k + 1)
        out.append(x if k == 0 else Q.asym(x, k + 1))
    return out


@dataclass
class Result:
    e_corr: float
    converged: bool
    ncycle: int
    amps: list                  # [t1, .., tn], dense


def kernel(fock, g, mo_energy, nocc, model='cc3', conv_tol=1e-12, conv_tol_normt=1e-10, max_cycle=200,
           diis_space=8, verbose=False, fock_top='bare', amps=None):
    '''The ground state of the model by Jacobi steps with DIIS, from the
    MP2 amplitudes (or ``amps``).'''
    n = NRANK[model]
    nvir = g.shape[0] - nocc
    o, v = slice(0, nocc), slice(nocc, None)
    dtype = np.result_type(fock, g)
    den = [ccsdtq_ref.denominator(mo_energy, nocc, k + 1) for k in range(n)]
    if amps is None:
        amps = [np.zeros((nocc,) * (k + 1) + (nvir,) * (k + 1), dtype=dtype) for k in range(n)]
        amps[0] = fock[v, o].T / den[0]
        amps[1] = g[v, v, o, o].transpose(2, 3, 0, 1) / den[1]
    e_old = ccsdtq_ref.energy(amps[0], amps[1], fock, g, nocc)
    adiis = DIIS(space=diis_space, incore=True)
    conv = False

    def parts(a):
        return [a[0].ravel()] + [Q.unique(x, k + 1).ravel() for k, x in enumerate(a) if k]
    for cycle in range(max_cycle):
        res = residuals(model, amps, fock, g, nocc, fock_top)
        new = [t + r / d for t, r, d in zip(amps, res, den)]
        normt = sum(np.linalg.norm(a - b) for a, b in zip(new, amps))
        ps = adiis.update(parts(new))
        amps = [ps[0].reshape(nocc, nvir)] + [
            Q.expand(p.reshape(len(Q.tuples(nocc, k + 1)), -1), nocc, nvir, k + 1)
            for k, p in enumerate(ps) if k]
        e = ccsdtq_ref.energy(amps[0], amps[1], fock, g, nocc)
        if verbose:
            print('cycle %3d  E_corr = %.13f  dE = %.2e  |dt| = %.2e' % (cycle + 1, e.real, (e - e_old).real, normt))
        done = abs(e - e_old) < conv_tol and normt < conv_tol_normt
        e_old = e
        if done:
            conv = True
            break
    adiis.close()
    return Result(float(np.real(e_old)), conv, cycle + 1, amps)


# ------------------------------------------------------------------ Jacobian

def sigma(model, r, amps, fock, g, nocc, fock_top='bare'):
    '''[sigma_1, .., sigma_n] dense: the Jacobian of the equations of the
    model at ``amps`` on r = [r1, .., rn].'''
    arrays = _arrays(amps, fock, g, nocc, fock_top)
    dg, df = eom_ref.dressed_derivative(arrays['v'], arrays['f'], r[0], nocc)
    n = g.shape[0]
    if fock_top == 'bare':
        df0 = np.zeros_like(df)
    elif fock_top == 'dressed':
        df0 = df
    else:
        X = np.zeros((n, n), dtype=df.dtype)
        Y = np.zeros_like(X)
        X[nocc:, :nocc] = -r[0].T
        Y[:nocc, nocc:] = r[0]
        df0 = X @ arrays['f0'] + arrays['f0'] @ Y.T
    primed = dict(arrays, v=dg, f=df, f0=df0)
    for k in range(1, len(amps)):
        arrays['r%d' % (k + 1)] = r[k]
    out = []
    for rank in range(1, len(amps) + 1):
        ts = terms(model, rank)
        lin = []
        for coef, factors in ts:
            for k, (kind, ix) in enumerate(factors):
                if kind in RNAME:
                    lin.append((coef, factors[:k] + ((RNAME[kind], ix),) + factors[k + 1:]))
        x = _total(ts, primed, nocc, rank) + _total(lin, arrays, nocc, rank)
        out.append(x if rank == 1 else Q.asym(x, rank))
    return out


def apply(model, x, amps, fock, g, nocc, fock_top='bare'):
    nvir = g.shape[0] - nocc
    return eom_ref.pack(sigma(model, eom_ref.unpack(x, nocc, nvir, len(amps)), amps, fock, g, nocc, fock_top))


def matrix(model, amps, fock, g, nocc, select=None, fock_top='bare'):
    '''The Jacobian among the unique elements (``select``: among these).'''
    nvir = g.shape[0] - nocc
    n = sum(eom_ref.sizes(nocc, nvir, len(amps)))
    idx = np.arange(n) if select is None else np.asarray(select)
    dtype = np.result_type(fock, g, *amps)
    M = np.zeros((len(idx), len(idx)), dtype=dtype)
    for c, j in enumerate(idx):
        e = np.zeros(n, dtype=dtype)
        e[j] = 1.0
        M[:, c] = apply(model, e, amps, fock, g, nocc, fock_top)[idx]
    return M


def eigenvalues(model, amps, fock, g, nocc, kind='ee', fock_top='bare'):
    '''All eigenvalues of the Jacobian, sorted by their real parts: the
    excitation energies of the model ('ip', 'ea': through the ghost spin
    orbital of eom_ref.ghost).'''
    select = None
    if kind != 'ee':
        amps, fock, g, nocc, select = eom_ref.ghost(amps, fock, g, nocc, kind)
    w = np.linalg.eigvals(matrix(model, amps, fock, g, nocc, select, fock_top))
    return w[np.argsort(w.real)]


# ------------------------------------------------------------- folded form

def folded(M, nlow, omega):
    '''The matrix of the folded eigenproblem at ``omega``: with P the ranks
    below the highest (the first ``nlow`` elements) and Q the highest rank,
    A_PP + A_PQ (omega - A_QQ)^-1 A_QP; its eigenvalue omega is an
    eigenvalue of M.'''
    App, Apq, Aqp, Aqq = M[:nlow, :nlow], M[:nlow, nlow:], M[nlow:, :nlow], M[nlow:, nlow:]
    return App + Apq @ np.linalg.solve(omega * np.eye(len(Aqq)) - Aqq, Aqp)


def folded_root(M, nlow, omega0, tol=1e-12, max_cycle=100):
    '''The eigenvalue of M reached from ``omega0`` by iterating
    omega <- (the eigenvalue of folded(M, omega) closest to omega);
    (omega, iterations).'''
    omega = omega0
    for it in range(max_cycle):
        w = np.linalg.eigvals(folded(M, nlow, omega))
        new = w[np.argmin(abs(w - omega))]
        if abs(new - omega) < tol:
            return new, it + 1
        omega = new
    return omega, max_cycle
