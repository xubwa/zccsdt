'''Dense reference of the left-hand side of CCSDTQ on spin orbitals or
spinors, real or complex: the Lambda equations, the one- and two-particle
densities, the orbital gradient and the relaxed one-particle density.
docs/lambda.md, section 13.

zccsdt.cc.lambda_ref (CCSDT) with T4 and Lambda_4: everything is evaluated
term by term from the tables of zccsdt.cc.lambda4_terms on full arrays, for
a few dozen spin orbitals, as the reference of an implementation on packed
amplitudes.  fock[p,q] and g[p,q,r,s] = <pq||rs> are the bare ones over all
spin orbitals (ccsdtq_ref.integrals), amps = [t1, t2, t3, t4] and lams =
[l1, l2, l3, l4] dense antisymmetric arrays [i.., a..].

Conventions as in lambda_ref: Lambda_k = (1/k!)^2 sum l_k[i.., a..]
i+ j+ .. b a, the left state is <0| (1 + Lambda) exp(-T), nothing is
conjugated, and

    d1[p,q]     = <0| (1 + Lambda) exp(-T) p+ q exp(T) |0>,
    d2[p,q,r,s] = <0| (1 + Lambda) exp(-T) p+ q+ s r exp(T) |0>,
    E = sum h[p,q] d1[p,q] + 1/4 sum <pq||rs> d2[p,q,r,s]

with the reference in them, not Hermitian (lambda_ref.pyscf_rdms makes the
Hermitian D[p,q] = <q+ p> of the CCSD code).  They are made in the dressed
basis and transformed back, d1 = x^T d1~ y.

With four electrons CCSDTQ is the full CI: <0| (1 + Lambda) exp(-T) is the
bra of the normalized ground state, d1 and d2 are its (Hermitian) density
matrices, and the orbital gradient vanishes in every block
(tests/test_lambda4_ref.py).
'''

from dataclasses import dataclass

import numpy as np

from zccsdt.cc import ccsdt, ccsdtq_ref as R, ccsdtq_terms as Q, lambda4_terms as L, lambda_ref as LR
from zccsdt.cc.diis import DIIS
from zccsdt.cc.lambda_ref import energy, one_index, pyscf_rdms  # noqa: F401

RANKS = (1, 2, 3, 4)


def _arrays(fock, g, nocc, amps, lams=None):
    '''The arrays of the tables: dressed f and v, the amplitudes.'''
    ge, fdr = ccsdt._t1_dress(g, fock, amps[0], nocc, g.shape[0])
    arrays = {'v': ge, 'f': fdr, 't2': amps[1], 't3': amps[2], 't4': amps[3]}
    if lams is not None:
        arrays.update({'l%d' % k: lams[k - 1] for k in RANKS})
    return arrays


def residuals(lams, amps, fock, g, nocc, arrays=None, sources=None):
    '''The residuals of the four Lambda equations, dL/dt_k: dense.
    ``sources`` = [s1, .., s4]: constant terms added to them.'''
    if arrays is None:
        arrays = _arrays(fock, g, nocc, amps)
    arrays = dict(arrays, **{'l%d' % k: lams[k - 1] for k in RANKS})
    res = [L.residual(k, arrays, nocc) for k in RANKS]
    if sources is not None:
        res = [r + s for r, s in zip(res, sources)]
    return res


@dataclass
class Result:
    converged: bool
    ncycle: int
    lams: list                  # [l1, l2, l3, l4], dense


def kernel(fock, g, mo_energy, nocc, amps, conv_tol_normt=1e-9, max_cycle=200, diis_space=8,
           verbose=False, sources=None):
    '''The Lambda equations of CCSDTQ for the converged amplitudes
    ``amps`` = [t1, t2, t3, t4], by Jacobi steps with DIIS from conj(t).'''
    nvir = g.shape[0] - nocc
    arrays = _arrays(fock, g, nocc, amps)
    den = [R.denominator(mo_energy, nocc, k) for k in RANKS]
    lams = [np.array(t, dtype=complex).conj() for t in amps]
    adiis = DIIS(space=diis_space, incore=True)
    conv = False

    def parts(a):
        return [a[0].ravel()] + [Q.unique(x, k + 1).ravel() for k, x in enumerate(a) if k]
    for cycle in range(max_cycle):
        res = residuals(lams, amps, fock, g, nocc, arrays, sources)
        new = [x + r / d for x, r, d in zip(lams, res, den)]
        norm = sum(np.linalg.norm(a - b) for a, b in zip(new, lams))
        if verbose:
            print('cycle %3d  |dl| = %.2e' % (cycle + 1, norm), flush=True)
        if norm < conv_tol_normt:
            lams, conv = new, True
            break
        ps = adiis.update(parts(new))
        lams = [ps[0].reshape(nocc, nvir)] + [
            Q.expand(p.reshape(len(Q.tuples(nocc, k + 1)), -1), nocc, nvir, k + 1)
            for k, p in enumerate(ps) if k]
    adiis.close()
    if not (np.iscomplexobj(fock) or np.iscomplexobj(g) or any(np.iscomplexobj(t) for t in amps)
            or (sources is not None and any(np.iscomplexobj(s) and abs(np.imag(s)).max() > 0 for s in sources))):
        lams = [x.real.copy() for x in lams]
    return Result(conv, cycle + 1, lams)


def lambda_ccsdtq(eris, t1, t2, t3, t4, **kwargs):
    '''(l1, l2, l3, l4) of CCSDTQ for the integrals of an ERIs container
    and converged dense amplitudes.'''
    fock, g, mo_e, nocc = R.integrals(eris)
    r = kernel(fock, g, mo_e, nocc, [t1, t2, t3, t4], **kwargs)
    if not r.converged:
        raise RuntimeError('the Lambda equations of CCSDTQ did not converge')
    return r.lams


# ----------------------------------------------------------------- densities

def dressed_densities(amps, lams, nocc):
    '''(d1~, d2~): the coefficients of the dressed one-electron integrals
    h~[p,q] and of the dressed <pq||rs> in L = sum h~ d1~ + 1/4 sum
    <pq||rs>~ d2~ - the densities <0| (1 + Lambda) exp(-T2 - T3 - T4) p+ q
    exp(T2 + T3 + T4) |0> and the same of p+ q+ s r.'''
    n = sum(amps[0].shape)
    arrays = {'v': np.empty((n, 0)), 't2': amps[1], 't3': amps[2], 't4': amps[3]}
    arrays.update({'l%d' % k: lams[k - 1] for k in RANKS})
    g1, g2 = L.density(arrays, nocc)
    occ = np.zeros(n)
    occ[:nocc] = 1.0
    d1 = g1 + np.diag(occ)
    # the reference, and the Fock matrix f~ = h~ + sum_i <pi||qi>~
    x = np.einsum('pr,qs->pqrs', g1 + 0.5 * np.diag(occ), np.diag(occ))
    x = x - x.transpose(1, 0, 2, 3)
    d2 = g2 + x - x.transpose(0, 1, 3, 2)
    return d1, d2


def densities(amps, lams, nocc):
    '''(d1[p,q], d2[p,q,r,s]) = <p+ q>, <p+ q+ s r> of the left and right
    CCSDTQ states (see the module), over all spin orbitals.'''
    n = sum(amps[0].shape)
    x, y = LR._dressing(amps[0], nocc, n)
    d1, d2 = dressed_densities(amps, lams, nocc)
    d1 = x.T @ d1 @ y
    d2 = np.einsum('pqrs,pt,qu->turs', d2, x, x)
    d2 = np.einsum('turs,rv,sw->tuvw', d2, y, y)
    return d1, d2


def orbital_gradient(fock, g, nocc, amps, lams, dressed=False):
    '''Gnh[a,i] = <0| (1 + Lambda) exp(-T) [a+_i a_a, H] exp(T) |0>, as
    lambda_ref.orbital_gradient: from the full densities, or (``dressed``)
    from the table of the dressed basis, lambda4_terms.GRADIENT - the same
    for converged t and lambda.  Half of it is the orbital gradient
    G[a,i] of zccsdt.zccsd.zvector with the Hermitian densities.'''
    if dressed:
        return L.gradient(_arrays(fock, g, nocc, amps, lams), nocc)
    d1, d2 = densities(amps, lams, nocc)
    h = fock - np.einsum('piqi->pq', g[:, :nocc, :, :nocc])
    return -one_index(h, g, d1, d2)[:nocc, nocc:].T


# ------------------------------------------------------------ relaxed density

@dataclass
class Relaxed:
    relaxed: np.ndarray         # D[p,q] = <q+ p>, Hermitian, all spin orbitals
    unrelaxed: np.ndarray
    z: np.ndarray
    gradient: np.ndarray        # G[y,x] of zccsd.zvector, all spin orbitals
    e_corr: float               # E_corr(CCSDTQ)
    amps: list
    lams: list
    d1: np.ndarray = None       # <p+ q> and <p+ q+ s r> over all spin orbitals, not Hermitian
    d2: np.ndarray = None


def relaxed_rdm1(mf, mo_coeff, nfrozen=0, ndrop=0, amps=None, lams=None, tol=1e-11,
                 conv_tol_normt=1e-11, verbose=None):
    '''The relaxed one-particle density of CCSDTQ for the SCF ``mf`` and
    its canonical spin orbitals or spinors ``mo_coeff`` (all of them,
    occupied first), the first ``nfrozen`` frozen and the last ``ndrop``
    virtual ones left out: from full arrays over all spin orbitals, for
    small molecules, through the code of the CCSD line as
    qgrad_ref.relaxed_rdm1 does it for CCSDT - the densities of the active
    spin orbitals completed by the frozen ones (qgrad_ref.full_densities),
    made Hermitian, the orbital gradient of all blocks from them
    (zvector.orbital_gradient) and the Z-vector equations with the blocks
    between frozen and active spinors (zvector._relax).  ``amps``, ``lams``:
    converged dense [t1, .., t4] and [l1, .., l4] of the active spin
    orbitals (default: ccsdtq_ref.kernel and kernel).  Returns a Relaxed;
    an expectation value is sum V[p,q] D[q,p].'''
    from zccsdt import eris as E
    from zccsdt.cc.qgrad_ref import full_densities
    from zccsdt.zccsd import zvector
    fock, g, mo_e, o = R.integrals(E.from_scf(mf, mo_coeff=mo_coeff))
    n = fock.shape[0]
    oa = o - nfrozen
    a = slice(nfrozen, n - ndrop)
    fa, ga, ea = fock[a, a], g[a, a, a, a], mo_e[a]
    if amps is None:
        r = R.kernel(fa, ga, ea, oa, 'ccsdtq', conv_tol=1e-13, conv_tol_normt=conv_tol_normt)
        if not r.converged:
            raise RuntimeError('CCSDTQ did not converge')
        amps = r.amps
    e_corr = float(np.real(R.energy(amps[0], amps[1], fa, ga, oa)))
    if lams is None:
        rl = kernel(fa, ga, ea, oa, amps, conv_tol_normt=conv_tol_normt)
        if not rl.converged:
            raise RuntimeError('the Lambda equations of CCSDTQ did not converge')
        lams = rl.lams
    D1, D2 = full_densities(*densities(amps, lams, oa), nfrozen, n)
    dm1, dm2 = pyscf_rdms(D1, D2)
    h = fock - np.einsum('piqi->pq', g[:, :o, :, :o])
    eri = 0.5 * g.transpose(0, 2, 1, 3)                 # (pq|rs) - (ps|rq) = <pr||qs>
    G = zvector.orbital_gradient(h, eri, dm1, dm2)
    spaces = (np.arange(nfrozen, o), np.arange(nfrozen), np.arange(o, n - ndrop), np.arange(n - ndrop, n))
    relaxed, z = zvector._relax(mf, mf, np.asarray(mo_coeff), mo_e, o, G, dm1, spaces, False,
                                zvector.DEGENERATE_DENSE, tol, verbose)
    return Relaxed(relaxed, dm1, z, G, e_corr, amps, lams, D1, D2)
